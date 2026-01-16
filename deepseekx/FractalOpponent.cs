// FractalOpponent as a depth-aware router around UnifiedMultiHeadTransformerLSTMCell + CombinatorialPathGate.
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

public class FractalOpponent : nn.Module
{
    // In FractalOpponent.cs oder als separater Layer:
    public readonly Linear outputHead;

// Im Konstruktor initialisieren:

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
        this.outputHead = nn.Linear(hiddenSize, 4); // 4 = A, C, G, T (no N class)
        this.RegisterComponents();

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
        // Defensive checks to surface null references early with helpful messages
        if (x is null) throw new ArgumentNullException(nameof(x), "Input tensor x is null in FractalOpponent.forward");
        if (h is null) throw new ArgumentNullException(nameof(h), "Hidden state h is null in FractalOpponent.forward");
        if (c is null) throw new ArgumentNullException(nameof(c), "Cell state c is null in FractalOpponent.forward");
        if (cell is null) throw new InvalidOperationException("UnifiedMultiHeadTransformerLSTMCell 'cell' is null. Ensure FractalOpponent was constructed properly.");
        if (butterflyGate is null) DisableButterfly = true; // fallback to disable
        if (pathGate is null) DisablePathGate = true;
        if (depthRouter is null) DisableDepth = true;
        if (depthRamanujanHead is null) throw new InvalidOperationException("depthRamanujanHead is null. The FractalOpponent must initialize all heads.");
        if (depthAnchorGate is null) throw new InvalidOperationException("depthAnchorGate is null. The FractalOpponent must initialize all gates.");

        // === 1) Path-Gate (Kombinatorische Experten-Auswahl) ===
        // Das Path-Gate transformiert h basierend auf der aktuellen Experten-Logik
        var hCur = h;
        if (!DisablePathGate)
        {
            hCur = pathGate.forward(h);
            this.LastWinningExpert = pathGate.LastWinningExpert;
        }

        // === 2) Butterfly-Gate mit Sensitivitäts-Schutz ===
        // Das Butterfly-Gate ruft die Lambda-Funktion intern ZWEIMAL auf (Base & Perturbed).
        // Wir umschließen das, um den internen Speicher der Cell zu schützen.
        Tensor hNext;
        if (!DisableButterfly)
        {
            hNext = butterflyGate.forward(hCur, (hIn) =>
            {
                // WICHTIG: Flag setzen, damit der rollierende Speicher (memK/memV) 
                // in der Zelle während der Sensitivitätsprüfung nicht modifiziert wird.
                cell.IsInSensitivityCheck = true;

                // Ein Einzelschritt durch die UMHT-Zelle
                var (stepOut, _, _) = cell.forward_step(x, hIn, c, null, null);

                cell.IsInSensitivityCheck = false; // Flag zurücksetzen
                return stepOut;
            }, depth);
        }
        else
        {
            // Falls Butterfly deaktiviert ist, normaler Schritt
            var (stepOut, _, _) = cell.forward_step(x, hCur, c, null, null);
            hNext = stepOut;
        }

        // === 3) Rekursive Tiefe (Fractal Depth) ===
        var depthStates = new List<Tensor> { hNext };
        if (!DisableDepth && depth < maxDepth)
        {
            // Der Router entscheidet, ob wir tiefer in die fraktale Struktur gehen
            // Wir nutzen hierfür den aktuellen Zustand und ein kleines MLP
            var depthGateInput = torch.cat(new Tensor[] { hNext, c }, dim: 1);
            using (var logits = depthRouter.forward(depthGateInput))
            {
                var probs = torch.nn.functional.softmax(logits, dim: 1);
                int chosenDepth = (int)probs.argmax(1).item<long>();
                this.LastDepthChosen = chosenDepth;

                if (chosenDepth > 0)
                {
                    // Rekursiver Aufruf: Das Modell "überlegt" tiefer
                    var (hSub, _, _) = this.forward(x, hNext, c, depth + 1);
                    depthStates.Add(hSub);
                    hNext = hSub;
                }
            }
        }

        // === 4) Ramanujan Depth Anchor (Stabilitäts-Kern) ===
        // Wir fassen die Zustände der verschiedenen Tiefen über eine Ramanujan-Summe zusammen
        var hDepthAnchor = MultiAgentFractalCore.RamanujanSum(depthStates.ToArray(), depthRamanujanHead);

        // Dämpfung und Kombination von aktuellem h und dem "Erinnerungs-Anker"
        var dampedAnchor = hDepthAnchor * 0.3f;
        var finalGateInput = torch.cat(new Tensor[] { hNext, dampedAnchor }, dim: 1);

        using (var gRaw = depthAnchorGate.forward(finalGateInput))
        using (var g = gRaw.sigmoid())
        {
            // Mischung aus aktuellem Pfad und tieferem Anker
            var hFinal = g * dampedAnchor + (1.0f - g) * hNext;

            // Finaler LayerNorm für numerische Stabilität
            var hOut = torch.nn.functional.layer_norm(hFinal, new long[] { hiddenSize });

            return (hOut, hOut, c);
        }
    }
}
