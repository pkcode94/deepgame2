// FractalOpponent as a depth-aware router around UnifiedMultiHeadTransformerLSTMCell + CombinatorialPathGate.
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

public class FractalOpponent : nn.Module
{
    private readonly Linear depthAnchorGate;

    private readonly Linear depthRamanujanHead;
    private readonly UnifiedMultiHeadTransformerLSTMCell cell;
    private readonly CombinatorialPathGate pathGate;
    private readonly Linear depthRouter; // logits over {0..maxDepth}

    private readonly int hiddenSize;
    private readonly int maxDepth;
    public int LastWinningExpert { get; private set; } = -1;
    // Primary constructor: external unified cell
    public FractalOpponent(UnifiedMultiHeadTransformerLSTMCell cell, int hiddenSize, int maxDepth)
        : base("fractal_opponent_router")
    {
        this.cell = cell ?? throw new ArgumentNullException(nameof(cell));
        this.hiddenSize = hiddenSize;
        this.maxDepth = Math.Max(0, maxDepth);

        pathGate = new CombinatorialPathGate(hiddenSize: hiddenSize, basisCount: 4, depth: maxDepth);

        // depth router over discrete depths [0..maxDepth]
        depthRouter = nn.Linear(hiddenSize, maxDepth + 1);

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
        pathGate = new CombinatorialPathGate(hiddenSize: hiddenSize, basisCount: 4, depth: maxDepth);

        // depth router over discrete depths [0..maxDepth]
        depthRouter = nn.Linear(hiddenSize, maxDepth + 1);

        // Ramanujan head for depth
        depthRamanujanHead = nn.Linear(hiddenSize * 2, hiddenSize);

        // NEW: depth AnchorGate
        depthAnchorGate = nn.Linear(hiddenSize * 2, hiddenSize);

        RegisterComponents();

    }

    // core recursive logic: depth-aware routing
    // x: [H] or [B,inputSize], h,c: [B,H]
    public (Tensor output, Tensor h, Tensor c) forward(Tensor x, Tensor h, Tensor c, int depth = 0)
    {
        //Console.WriteLine("depth: " + depth);
        // 1) Shallow pass
        var (shallowOut, hShallow, cShallow) = cell.forward_step(x, h, c);

        // 2) Path Gate
        using var hPath = pathGate.forward(hShallow);
        LastWinningExpert = pathGate.LastWinningExpert;
        // 3) Depth Decision
        int steps = 0;
        using (var depthLogits = depthRouter.forward(hPath))
        using (var depthProbs = depthLogits.softmax(1))
        using (var depthIdx = depthProbs.argmax(1))
        {
            int chosenDepth = depthIdx.ToInt32();
            int remaining = Math.Max(0, maxDepth - depth);
            steps = Math.Min(chosenDepth, remaining);
        }

        if (steps <= 0)
        {
            // Return shallow results (cloned to keep graph alive if needed)
            return (shallowOut.alias(), hPath.alias(), cShallow.alias());
        }

        // 4) Recursive refinement
        // 4) Recursive refinement
        // 4) Recursive refinement
        var hCur = hPath.alias();
        var cCur = cShallow.alias();
        var finalOut = shallowOut.alias();

        // collect depth states, starting with depth 0 (shallow)
        var depthStates = new List<Tensor>();
        depthStates.Add(hCur.alias()); // [1,H]
        for (int i = 0; i < steps; i++)
        {
            // === 1. DAMPING FUNKTION (Quadratic Decay) ===
            // Je tiefer wir gehen, desto vorsichtiger wird die Korrektur.
            // Das verhindert das "Infinity Mirror" Problem und die 100% Ratio.
            float damping = (float)(1.0 / Math.Pow(depth + 1, 2));

            // Rekursions-Aufruf (holt das "Wissen" aus der nächsten Ebene)
            var (hRecursiveOut, hNextRaw, cNext) = forward(x, hCur, cCur, depth + 1);

            // === 2. NUMERICAL DEBUGGING ===
            if (DateTime.Now.Second % 50 == 0)
            {
                // Wir messen hNextRaw gegen hCur BEVOR wir sie mergen
                DebugFeedbackLoop(hCur, hNextRaw, depth, damping);
            }

            // === 3. RESIDUAL BLENDING (Der Fix für die 100% Ratio) ===
            // Statt hCur = hNextRaw nutzen wir Linear Interpolation (lerp).
            // hNext = hCur + damping * (hNextRaw - hCur)
            var hNext = torch.lerp(hCur, hNextRaw, damping);

            // Store for Ramanujan summation
            depthStates.Add(hNext.alias());

            // Cleanup old refs
            hCur.Dispose();
            cCur.Dispose();
            finalOut.Dispose();
            hNextRaw.Dispose(); // Wichtig: Raw Output der Rekursion löschen

            // Update States für den nächsten Schritt im Loop
            hCur = hNext;
            cCur = cNext;
            finalOut = hRecursiveOut;
        }

        foreach (var s in depthStates)
        {
            if (s is null)
                Console.WriteLine("[FractalOpponent] one depthStates entry is NULL");
        }
        // Ramanujan-style depth anchor over h^(0..steps)
        var hDepthAnchor = MultiAgentFractalCore.RamanujanSum(depthStates.ToArray(), depthRamanujanHead); // [1,H]

        // === AnchorGate for depth ===
        // Gate input: concat(current depth state, depth anchor)
        var depthGateInput = torch.cat(new Tensor[] { hCur, hDepthAnchor }, dim: 1); // [1,2H]
        var depthGateRaw = depthAnchorGate.forward(depthGateInput);                   // [1,H]
        var depthG = depthGateRaw.sigmoid();                                         // [1,H]

        // Blend: h_out = g * anchor + (1-g) * hCur
        var hOut = depthG * hDepthAnchor + (1 - depthG) * hCur;



        return (finalOut, hOut, cCur);
    }
}
