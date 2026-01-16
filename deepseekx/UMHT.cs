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
    public bool IsInSensitivityCheck { get; set; } = false;
    // Router- and CT-gate-aware forward step.
    // x: [B,inputSize] or [inputSize] or [1,inputSize]
    // h,c: [B,HiddenSize]
    // externalK/externalV: [S_ext,B,HiddenSize] or null
    public (Tensor output, Tensor hNext, Tensor cNext) forward_step(
    Tensor x, Tensor h, Tensor c, Tensor? externalK, Tensor? externalV)
    {
        Tensor xIn = x.dim() == 1 ? x.unsqueeze(0) : x;

        // 1) LSTM Update
        var (hLstm, cNext) = lstm.forward(xIn, (h, c));
        var cur = hLstm.unsqueeze(0); // [1, 1, HiddenSize]

        // 2) Rolling Memory Update (mit Initialisierungs-Schutz)
        if (!IsInSensitivityCheck)
        {
            if (memK is null || memV is null)
            {
                // Initialisierung: Der Speicher startet mit dem aktuellen Zustand
                memK = cur.clone();
                memV = cur.clone();
            }
            else
            {
                // Sicherstellen, dass die Dimensionen für 'cat' passen
                var nextK = torch.cat(new[] { memK, cur }, 0);
                var nextV = torch.cat(new[] { memV, cur }, 0);

                if (nextK.shape[0] > memLen)
                {
                    var start = nextK.shape[0] - memLen;
                    memK = nextK.slice(0, start, nextK.shape[0], 1).clone();
                    memV = nextV.slice(0, start, nextV.shape[0], 1).clone();
                }
                else
                {
                    memK = nextK;
                    memV = nextV;
                }
            }
        }

        // SICHERHEITS-CHECK: Falls memK immer noch null ist (sollte nicht passieren)
        // Erzeuge einen Zero-Tensor, damit Attention nicht abstürzt
        var effectiveK = memK ?? torch.zeros(new long[] { 1, 1, HiddenSize }).to(x.device);
        var effectiveV = memV ?? torch.zeros(new long[] { 1, 1, HiddenSize }).to(x.device);

        // 3) Self-Attention & External Attention
        var qSelf = hLstm.unsqueeze(0);

        // Nutze effectiveK/V statt memK!
        var (selfAttnOut, _) = selfAttn.forward(qSelf, effectiveK, effectiveV, null, false, null);
        var selfOut = selfAttnOut.squeeze(0);

        Tensor externalOut;
        if (externalK is not null && externalV is not null)
        {
            var (extRes, _) = crossAttn.forward(qSelf, externalK, externalV, null, false, null);
            externalOut = extRes.squeeze(0);
        }
        else
        {
            var (extRes, _) = crossAttn.forward(qSelf, effectiveK, effectiveV, null, false, null);
            externalOut = extRes.squeeze(0);
        }

        // 4) Router & Residual
        var routerWeights = routerMLP.forward(hLstm).softmax(1);
        var wSelf = routerWeights.index(TensorIndex.Colon, TensorIndex.Single(0)).unsqueeze(1);
        var wExt = routerWeights.index(TensorIndex.Colon, TensorIndex.Single(1)).unsqueeze(1);
        var routedAttn = (wSelf * selfOut) + (wExt * externalOut);
        var routed = hLstm + routedAttn + residualProject.forward(routedAttn);

        // 5) CT-GATE
        // Nutze auch hier effectiveK für den Mittelwert
        var temporalSummary = effectiveK.mean(new long[] { 0L });
        var compressed = W_ct_compress.forward(temporalSummary);
        var expanded = W_ct_expand.forward(compressed);
        int reps = (HiddenSize + (int)compressed.shape[1] - 1) / (int)compressed.shape[1];
        var compressedExpanded = compressed.repeat(1, reps).narrow(1, 0, HiddenSize);
        var g = torch.sigmoid(W_ct_gate.forward(torch.cat(new[] { xIn, routed }, 1)));
        var hCT = torch.lerp(compressedExpanded, expanded, g);

        // 6) Final Blend
        var hFinal = torch.stack(new[] { hLstm, routed, hCT }, 0).mean(new long[] { 0 });
        hFinal = torch.nn.functional.layer_norm(hFinal, new long[] { (long)HiddenSize });
        hFinal = hFinal.clamp(-3.0, 3.0);

        return (hFinal, hFinal, cNext);
    }
    public void ResetMemory()
    {
        memK?.Dispose();
        memV?.Dispose();
        memK = null;
        memV = null;
    }
}
