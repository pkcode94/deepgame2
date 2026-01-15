using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

public class ButterflyEffectGate : nn.Module
{
    private readonly float epsilon;
    private readonly float sensitivityScale;
    private readonly Linear learnedGate;

    public ButterflyEffectGate(int hiddenSize, float epsilon = 1e-3f, float sensitivityScale = 1.0f)
        : base("butterfly_effect_gate")
    {
        this.epsilon = epsilon;
        this.sensitivityScale = sensitivityScale;

        // Optional learned modulation
        learnedGate = nn.Linear(hiddenSize, hiddenSize);
        RegisterComponents();
    }

    /// <summary>
    /// hCur: current hidden state [1,H]
    /// updateFn: function that maps h -> hNextRaw
    /// depth: recursion depth for logging
    /// </summary>
    public Tensor forward(Tensor hCur, Func<Tensor, Tensor> updateFn, int depth)
    {
        // === 1. Base update ===
        var hBase = updateFn(hCur); // [1,H]

        // === 2. Perturbation ===
        var noise = torch.randn_like(hCur) * epsilon;
        var hPert = hCur + noise;
        var hPertNext = updateFn(hPert);

        // === 3. Sensitivity ===
        var diff = (hPertNext - hBase);
        float sensitivity = diff.norm().ToSingle() / (epsilon + 1e-6f);

        if (DateTime.Now.Millisecond % 5000 == 0)
        {
            string indent = new string(' ', depth * 2);
            Console.WriteLine($"{indent}[ButterflyGate] Sensitivity={sensitivity:F4}");
        }

        // === 4. Convert sensitivity to gate scalar ===
        float sNorm = (float)Math.Tanh(sensitivityScale * sensitivity);
        var sTensor = torch.full_like(hBase, sNorm); // [1,H]

        // === 5. Learned gate ===
        var gLearned = learnedGate.forward(hBase).sigmoid(); // [1,H]

        // === 6. Final gate ===
        var g = 0.5f * sTensor + 0.5f * gLearned;

        // === 7. Blend ===
        // High sensitivity → trust hBase (stabilize)
        // Low sensitivity → allow hPertNext (explore)
        var hOut = g * hBase + (1 - g) * hPertNext;

        return hOut;
    }
}
