// FractalOpponent as a depth-aware router around UnifiedMultiHeadTransformerLSTMCell + CombinatorialPathGate.
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

public class FractalOpponent : nn.Module
{
    public int LastDepthChosen { get; set; } = 0;
    public bool DisableButterfly { get; set; } = false;
    public bool DisableDepth { get; set; } = false;
    public bool DisablePathGate { get; set; } = false;
    public int ForcedExpert { get; set; } = 0;
    private readonly Linear depthAnchorGate;

    private readonly Linear depthRamanujanHead;
    private readonly UnifiedMultiHeadTransformerLSTMCell cell;
    public readonly CombinatorialPathGate pathGate;
    private  Linear depthRouter; // logits over {0..maxDepth}
    private readonly ButterflyEffectGate butterflyGate;
    public readonly int hiddenSize;
    private readonly int maxDepth;
    public Tensor lastdepthlogits;
    public int LastWinningExpert { get; private set; } = -1;
    // Primary constructor: external unified cell
    public FractalOpponent(UnifiedMultiHeadTransformerLSTMCell cell, int hiddenSize, int maxDepth)
        : base("fractal_opponent_router")
    {
        butterflyGate = new ButterflyEffectGate(hiddenSize);
        this.cell = cell ?? throw new ArgumentNullException(nameof(cell));
        this.hiddenSize = hiddenSize;
        this.maxDepth = Math.Max(0, maxDepth);

        pathGate = new CombinatorialPathGate(hiddenSize: hiddenSize, basisCount: 4, depth: maxDepth);

        // depth router over discrete depths [0..maxDepth]
        depthRouter = nn.Linear(hiddenSize*2, maxDepth + 1);

        // NEW: Ramanujan head for depth [1,2H] -> [1,H]
        depthRamanujanHead = nn.Linear(hiddenSize * 2, hiddenSize);
        depthAnchorGate = nn.Linear(hiddenSize * 2, hiddenSize);
        RegisterComponents();

    }


    private void DebugFeedbackLoop(Tensor hCur, Tensor hNextRaw, int depth, float damping =1)
    {
        var signalMagnitude = hCur.abs().mean().ToSingle();
        var correctionMagnitude = (hNextRaw.abs().mean() * damping).ToSingle();

        // Das Verhältnis von Korrektur zu Signal
        float feedbackRatio = correctionMagnitude / (signalMagnitude + 1e-6f);

        string indent = new string(' ', depth * 2);
        Console.WriteLine($"{indent}[Depth {depth}] Feedback-Ratio: {feedbackRatio:P2} (Signal: {signalMagnitude:F4})");

        if (float.IsNaN(feedbackRatio) || feedbackRatio > 10.0f)
        {
            Console.WriteLine($"{indent} [!!!] CRITICAL: Feedback Loop Divergence at Depth {depth}");
        }
    }
    // Compatibility constructor: internal unified cell
    public FractalOpponent(int hiddenSize, int maxDepth)
        : this(new UnifiedMultiHeadTransformerLSTMCell(hiddenSize, hiddenSize, 4, 1), hiddenSize, maxDepth)
    {
        /*
        pathGate = new CombinatorialPathGate(hiddenSize: hiddenSize, basisCount: 4, depth: maxDepth);

        // depth router over discrete depths [0..maxDepth]
        depthRouter = nn.Linear(hiddenSize, maxDepth + 1);

        // Ramanujan head for depth
        depthRamanujanHead = nn.Linear(hiddenSize * 2, hiddenSize);

        // NEW: depth AnchorGate
        depthAnchorGate = nn.Linear(hiddenSize * 2, hiddenSize);
        */
        RegisterComponents();

    }

    // core recursive logic: depth-aware routing
    // x: [H] or [B,inputSize], h,c: [B,H]

    public (Tensor output, Tensor h, Tensor c) forward(Tensor x, Tensor h, Tensor c, int depth = 0)
    {
        // === 1) Shallow pass ===
        var (shallowOut, hShallow, cShallow) = cell.forward_step(x, h, c);

        // === 2) Path Gate ===
        Tensor hPath;
        if (DisablePathGate)
        {
            hPath = hShallow.alias();
            LastWinningExpert = -1;
        }
        else
        {
            hPath = pathGate.forward(hShallow);
            LastWinningExpert = pathGate.LastWinningExpert;
        }

        // === 3) Depth Decision (Korrigiert für 64-Bit Input) ===
        int steps = 0;
        if (!DisableDepth)
        {
            // Da wir hier noch keinen echten Anker haben, füllen wir mit Nullen auf (Padding)
            // um auf die vom Router erwarteten 64 Features zu kommen.
            using (var padding = torch.zeros_like(hPath))
            using (var routerInput = torch.cat(new Tensor[] { hPath, padding }, 1))
            using (var depthLogits = depthRouter.forward(routerInput))
            using (var depthProbs = depthLogits.softmax(1))
            using (var depthIdx = depthProbs.argmax(1))
            {
                // Speichern für den Entropy-Loss im Trainer
                lastdepthlogits = depthLogits.detach();

                int chosenDepth = (int)depthIdx.item<long>();
                int remaining = Math.Max(0, maxDepth - depth);
                steps = Math.Min(chosenDepth, remaining);
            }
        }

        // Wenn keine Rekursion gewählt wurde, sofort zurückkehren
        if (steps <= 0)
        {
            return (shallowOut.alias(), hPath.alias(), cShallow.alias());
        }

        // === 4) Recursive refinement ===
        var hCur = hPath.alias();
        var cCur = cShallow.alias();
        var finalOut = shallowOut.alias();

        var depthStates = new List<Tensor>();
        depthStates.Add(hCur.alias());

        for (int i = 0; i < steps; i++)
        {
            // Butterfly Gate kapselt die Rekursion
            Tensor UpdateFn(Tensor hLocal)
            {
                var (outRec, hNextRec, _) = forward(x, hLocal, cCur, depth + 1);
                // Wir müssen hier vorsichtig sein: outRec muss entsorgt werden, wenn nicht genutzt
                outRec.Dispose();
                return hNextRec;
            }

            Tensor hNext;
            if (DisableButterfly)
            {
                hNext = UpdateFn(hCur);
            }
            else
            {
                hNext = butterflyGate.forward(hCur, UpdateFn, depth);
            }

            depthStates.Add(hNext.alias());

            // Altes hCur entsorgen, um Memory Leaks zu vermeiden
            hCur.Dispose();
            hCur = hNext;
        }

        // === 5) Ramanujan depth anchor ===
        // Hier wird die Summe gebildet (Input pro State: 32, Output: 32)
        var hDepthAnchor = MultiAgentFractalCore.RamanujanSum(depthStates.ToArray(), depthRamanujanHead);

        // === 6) Depth Anchor Gate (Finaler Check: 32 + 32 = 64) ===
        // Multipliziere hDepthAnchor mit einem Dämpfungsfaktor (z.B. 0.3), 
        // bevor er ins Gate geht, oder dämpfe das Gate selbst.
        var dampedAnchor = hDepthAnchor * 0.3f;
        var depthGateInput = torch.cat(new Tensor[] { hCur, dampedAnchor }, dim: 1);
        // WICHTIG: Die Logits nochmals mit dem echten Anker aktualisieren für genaueren Loss
        using (var finalLogits = depthRouter.forward(depthGateInput))
        {
            lastdepthlogits?.Dispose(); // Altes Padding-Logit entsorgen
            lastdepthlogits = finalLogits.detach();
        }

        // Das Gate entscheidet, wie stark der Anker den aktuellen Zustand beeinflusst
        using (var depthGateRaw = depthAnchorGate.forward(depthGateInput))
        using (var depthG = depthGateRaw.sigmoid())
        {
            // hOut = g * anchor + (1-g) * hCur
            var hOut = depthG * hDepthAnchor + (1 - depthG) * hCur;

            // Cleanup
            foreach (var d in depthStates) d.Dispose();
            hDepthAnchor.Dispose();

            return (finalOut, hOut, cCur);
        }
    }
}
