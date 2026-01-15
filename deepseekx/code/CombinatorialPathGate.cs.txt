// CombinatorialPathGate as a small mixture-of-experts + identity gate.
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.optim.lr_scheduler.impl;

public class CombinatorialPathGate : nn.Module
{

    public bool OverrideRouting { get; set; } = false;
    public int ForcedExpert { get; set; } = 0;

    private readonly int hiddenSize;
    private readonly int numExperts;

    private readonly Linear[] experts;
    private readonly Linear expertRouter; // produces logits over experts
    private readonly Linear gate;         // identity vs transformed gate
 
    public CombinatorialPathGate(int hiddenSize, int basisCount = 4, int depth = 1)
        : base("combinatorial_path_gate_router")
    {
        this.hiddenSize = hiddenSize;
        this.numExperts = Math.Max(1, basisCount);

        experts = new Linear[numExperts];
        for (int i = 0; i < numExperts; i++)
        {
            experts[i] = nn.Linear(hiddenSize, hiddenSize);
        }

        // router over experts: [B,H] -> [B,numExperts]
        expertRouter = nn.Linear(hiddenSize, numExperts);

        // identity gate: [B,H] -> [B,H]
        gate = nn.Linear(hiddenSize, hiddenSize);

        RegisterComponents();
    }

    // forward: x shape [B,H] or [H]
    public int LastWinningExpert { get; set; }
    public float LastGateValue { get; set; }
    public int LastWinnerExpert { get; set; }
    public Tensor forward(Tensor x)
    {
        Tensor t = x;
        if (t.dim() == 1)
            t = t.unsqueeze(0); // [1,H]

        // ---- Router logits ----
        var logits = expertRouter.forward(t);              // [B,numExperts]
        var winner = logits.argmax( 1 );      // [B]

        if (winner.numel() != 1)
            throw new InvalidOperationException("Batch > 1 not supported.");

        int expertIdx = (int)winner.item<long>();

        LastWinningExpert = expertIdx;

        // ---- Hard expert output ----
        Tensor mix = experts[expertIdx]
            .forward(t)
            .tanh();                                       // [B,H]

        // ---- Identity vs transformed gate ----
        var g = torch.sigmoid(gate.forward(t));            // [B,H]
        var outTensor = g.mul(mix).add(torch.ones_like(g).sub(g).mul(t));

        return outTensor;
    }


}
