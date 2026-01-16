using System;
using System.Collections.Generic;
using System.Threading;
using TorchSharp;
using TorchSharp;
using TorchSharp.Modules;
using TorchSharp.Modules;
using static System.Net.Mime.MediaTypeNames;
using static TorchSharp.torch;
using static TorchSharp.torch;

public class UnifiedMultiHeadTransformerLSTMCell : nn.Module
{
    // rolling self-attention memory
    private Tensor? memK;   // [S,B,H]
    private Tensor? memV;   // [S,B,H]
    private readonly int memLen = 32;

    // sizes
    public readonly int InputSize;
    public readonly int HiddenSize;
    public readonly int hiddenSize; // for compatibility
    public readonly int NumHeads;

    // core modules
    private readonly LSTMCell lstm;
    private readonly MultiheadAttention selfAttn;
    private readonly MultiheadAttention crossAttn;

    private readonly Linear? inputProject;   // x -> hiddenSize if needed
    private readonly Linear residualProject; // extra projection on routed attn
    private readonly Linear routerMLP;       // router over attention banks (self vs external)

    // CT-gate parameters
    private readonly Linear W_ct_gate;       // gate over [x_t, attn]
    private readonly Linear W_ct_compress;   // compress temporal summary to bottleneck
    private readonly Linear W_ct_expand;     // expand back to hiddenSize
    private readonly int bottleneckDim;

    // final output head
    public readonly nn.Module<Tensor, Tensor> output;

    public UnifiedMultiHeadTransformerLSTMCell(
        int inputSize,
        int hiddenSize,
        int numHeads,
        int outputSize = 1)
        : base("unified_multihead_transformer_lstm_cell_router_ct")
    {
        InputSize = inputSize;
        HiddenSize = hiddenSize;
        this.hiddenSize = hiddenSize;
        NumHeads = numHeads;

        // LSTM operates in hidden space after the first projection
        // but we keep inputSize for x_t; we project x_t to hiddenSize if needed
        lstm = nn.LSTMCell(inputSize, hiddenSize);

        // self-attention over rolling memory of hidden states
        selfAttn = nn.MultiheadAttention(hiddenSize, numHeads);
        // optional cross-attention over external memory (same hidden size)
        crossAttn = nn.MultiheadAttention(hiddenSize, numHeads);

        if (inputSize != hiddenSize)
            inputProject = nn.Linear(inputSize, hiddenSize);
        else
            inputProject = null;

        residualProject = nn.Linear(hiddenSize, hiddenSize);

        // router over two banks: self-mem and external-mem
        // routerMLP: [B,H] -> [B,2]
        routerMLP = nn.Linear(hiddenSize, 2);

        // CT-gate
        bottleneckDim = Math.Max(1, hiddenSize / 4);
        // gate sees original input x_t and the attention/routed representation
        W_ct_gate = nn.Linear(inputSize + hiddenSize, hiddenSize);
        W_ct_compress = nn.Linear(hiddenSize, bottleneckDim);
        W_ct_expand = nn.Linear(bottleneckDim, hiddenSize);

        this.output = nn.Sequential(
            nn.Dropout(0.1), // 10% Dropout für die finale Entscheidung
            nn.Linear(hiddenSize, outputSize) // Hier 'outputSize' statt 'vocabSize'
        );
        // Initialisierung für stabilen Gradientenfluss
        // Sicherer Zugriff auf die internen Parameter von MultiheadAttention
        foreach (var (name, param) in crossAttn.named_parameters())
        {
            if (name.Contains("in_proj_weight"))
            {
                nn.init.xavier_uniform_(param);
            }
            if (name.Contains("in_proj_bias"))
            {
                nn.init.zeros_(param);
            }
        }
        RegisterComponents();
    }

    // Backwards-compatible single-step forward without external banks.
    public (Tensor output, Tensor h, Tensor c) forward_step(Tensor x, Tensor h, Tensor c)
    {
        return forward_step(x, h, c, externalK: null, externalV: null);
    }

    // Router- and CT-gate-aware forward step.
    // x: [B,inputSize] or [inputSize] or [1,inputSize]
    // h,c: [B,HiddenSize]
    // externalK/externalV: [S_ext,B,HiddenSize] or null
    public (Tensor output, Tensor h, Tensor c) forward_step(
        Tensor x,
        Tensor h,
        Tensor c,
        Tensor? externalK,
        Tensor? externalV)
    {

        // Ensure batch dimension for x_t
        Tensor xIn = x;
        if (xIn.dim() == 1)
            xIn = xIn.unsqueeze(0); // [1,inputSize]

        // 1) LSTM update in input space -> h_lstm
        var (hLstm, cNext) = lstm.forward(xIn, (h, c)); // hLstm: [B,H]

        // Write hLstm into rolling self-attention memory
        var cur = hLstm.unsqueeze(0); // [1,B,H]

        if (memK is null || memV is null)
        {
            memK = cur.detach();
            memV = cur.detach();
        }
        else
        {
            var newK = torch.cat(new[] { memK, cur.detach() }, 0); // [S_old+1,B,H]
            var newV = torch.cat(new[] { memV, cur.detach() }, 0);

            if (newK.shape[0] > memLen)
            {
                var start = newK.shape[0] - memLen;
                memK.Dispose();
                memV.Dispose();
                memK = newK.slice(0, start, newK.shape[0], 1);
                memV = newV.slice(0, start, newV.shape[0], 1);
            }
            else
            {
                memK.Dispose();
                memV.Dispose();
                memK = newK;
                memV = newV;
            }
        }

        // 2) SELF-ATTENTION over rolling memory
        Tensor selfOut;
        {
            // memK/memV: [S,B,H]
            var qSelf = hLstm.unsqueeze(0); // [1,B,H]
            var selfRes = selfAttn.forward(qSelf, memK!, memV!, null, false, null);
            selfOut = selfRes.Item1.squeeze(0); // [B,H]
        }

        // 3) OPTIONAL EXTERNAL ATTENTION
        Tensor externalOut;
        bool hasExternal = (externalK is not null) && (externalV is not null);

        if (hasExternal)
        {
            var qCross = hLstm.unsqueeze(0); // [1,B,H]
            var crossRes = crossAttn.forward(qCross, externalK!, externalV!, null, false, null);
            externalOut = crossRes.Item1.squeeze(0); // [B,H]
        }
        else
        {
            // Use crossAttn over the internal memory as fallback so crossAttn parameters
            // are part of the graph and receive gradients (prevents zero-grad diagnostics).
            // This mirrors self-attention but goes through the separate crossAttn module.
            var qCross = hLstm.unsqueeze(0); // [1,B,H]
            var crossRes = crossAttn.forward(qCross, memK!, memV!, null, false, null);
            externalOut = crossRes.Item1.squeeze(0); // [B,H]
        }

        // 4) ROUTER over self vs external
        var routerLogits = routerMLP.forward(hLstm);   // [B,2]
        var routerWeights = routerLogits.softmax(1);   // [B,2]

        var wSelf = routerWeights.index(new TensorIndex[] {
            TensorIndex.Colon,
            TensorIndex.Single(0)
        }).unsqueeze(1); // [B,1]

        var wExt = routerWeights.index(new TensorIndex[] {
            TensorIndex.Colon,
            TensorIndex.Single(1)
        }).unsqueeze(1); // [B,1]

        // For debugging, keep this if you want to see routing behavior:
        // Console.WriteLine($"router: self={wSelf.mean().ToSingle():F3} ext={wExt.mean().ToSingle():F3}");

        var routedAttn = wSelf * selfOut + wExt * externalOut; // [B,H]

        // residual projection on routed attention
        var routed = hLstm + routedAttn + residualProject.forward(routedAttn); // [B,H]

        // 5) CT-GATE (temporal transform gate)

        // Temporal summary: mean over time dimension of memK [S,B,H] -> [B,H]
        Tensor temporalSummary;
        if (memK is null || memK.shape[0] == 0)
        {
            temporalSummary = torch.zeros_like(hLstm);
        }
        else
        {
            temporalSummary= memK.mean(new long[] { 0L });
        }

        // Compress summary to bottleneck and expand back
        var compressedSmall = W_ct_compress.forward(temporalSummary); // [B,bottleneck]
        var expanded = W_ct_expand.forward(compressedSmall);          // [B,H]

        // Expand compressedSmall to hidden size for blending
        int csDim = (int)compressedSmall.shape[1];
        int reps = (HiddenSize + csDim - 1) / csDim;

        var compressedSmallExpanded =
            compressedSmall
                .repeat(new long[] { 1L, (long)reps })   // repeat needs long[]
                .narrow(1L, 0L, (long)HiddenSize);       // narrow needs long args

        // Gate uses original x_t and routed attention representation
        var gateInput = torch.cat(new Tensor[] { xIn, routed }, 1); // [B, input+H]
        var g = torch.sigmoid(W_ct_gate.forward(gateInput));        // [B,H]

        var hCT = torch.lerp(compressedSmallExpanded, expanded, g); // [B,H]

        // 6) Final blend (ensemble of LSTM, route
        // [B,H]

        // In UnifiedMultiHeadTransformerLSTMCell.forward
        // Ersetzen Sie den finalen Blend-Abschnitt:

        var ensemble = new List<Tensor> { hLstm, routed, hCT };
        using var stacked = torch.stack(ensemble, 0);
        var hFinal = stacked.mean(new long[] { 0 });

        // NEU: LayerNorm vor dem Dropout stabilisiert die Skalierung für den Bias
        hFinal = torch.nn.functional.layer_norm(hFinal, new long[] { HiddenSize });

        hFinal = nn.functional.dropout(hFinal, 0.2, this.training);

        // Korrektur der Rückgabewerte:
        // hLstm ist der neue Hidden-State (h), cNext ist der neue Cell-State (c)
        return (hFinal, hLstm, cNext);

    }

    public void ResetMemory()
    {
        memK?.Dispose();
        memV?.Dispose();
        memK = null;
        memV = null;
    }
}
