using deepseekx;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.optim.lr_scheduler.impl.CyclicLR;
//
// ===============================================================
//  TOKEN-BASED SINE LSTM (STABLE VERSION + DEBUG PRINTS)
//  - Single-band LSTM
//  - Tokenized sine
//  - Clean sine generator (full cycles)
//  - Predicted vs Expected vs Input print
//  - No randomized phase
//  - No sampling (argmax only)
//  - Fractal core disabled
// ===============================================================
//


public class FractalMoETrainingManager
{
    private readonly FractalOpponent opponent;
    private readonly int hiddenSize;

    public FractalMoETrainingManager(FractalOpponent opponent)
    {
        this.opponent = opponent;
        this.hiddenSize = opponent.hiddenSize;
    }

    public enum Regime
    {
        SingleExpert,
        PathGateOnly,
        ButterflyOnly,
        DepthOnly,
        FullMoE
    }

    public (List<Tensor> outputs, List<string> logs) RunSequence(
        Tensor[] inputs,
        Regime regime,
        int forcedExpert = 0)
    {
        var logs = new List<string>();
        var outputs = new List<Tensor>();

        var h = torch.zeros(1, hiddenSize);
        var c = torch.zeros(1, hiddenSize);

        for (int t = 0; t < inputs.Length; t++)
        {
            var x = inputs[t];

            // === Apply regime switches ===
            ApplyRegime(regime, forcedExpert);

            // === Forward ===
            var (outT, hNext, cNext) = opponent.forward(x, h, c);

            // === Logging ===
            logs.Add(BuildLogEntry(regime, t, opponent));

            outputs.Add(outT.detach());
            h.Dispose();
            c.Dispose();
            h = hNext;
            c = cNext;
        }

        return (outputs, logs);
    }

    private void ApplyRegime(Regime regime, int forcedExpert)
    {
        switch (regime)
        {
            case Regime.SingleExpert:
                opponent.pathGate.LastWinningExpert = forcedExpert;
                opponent.pathGate.OverrideRouting = true;
                opponent.DisableButterfly = true;
                opponent.DisableDepth = true;
                break;

            case Regime.PathGateOnly:
                opponent.pathGate.OverrideRouting = false;
                opponent.DisableButterfly = true;
                opponent.DisableDepth = true;
                break;

            case Regime.ButterflyOnly:
                opponent.DisableButterfly = false;
                opponent.pathGate.OverrideRouting = true;
                opponent.DisableDepth = true;
                break;

            case Regime.DepthOnly:
                opponent.DisableDepth = false;
                opponent.pathGate.OverrideRouting = true;
                opponent.DisableButterfly = true;
                break;

            case Regime.FullMoE:
                opponent.pathGate.OverrideRouting = false;
                opponent.DisableButterfly = false;
                opponent.DisableDepth = false;
                break;
        }
    }

    private string BuildLogEntry(Regime regime, int t, FractalOpponent opp)
    {
        return regime switch
        {
            Regime.SingleExpert =>
                $"t={t} | SingleExpert={opp.pathGate.LastWinningExpert}",

            Regime.PathGateOnly =>
                $"t={t} | PathGate winner={opp.pathGate.LastWinningExpert}",

            Regime.ButterflyOnly =>
                $"t={t} | Butterfly sensitivity logged internally",

            Regime.DepthOnly =>
                $"t={t} | Depth chosen={opp.LastDepthChosen}",

            Regime.FullMoE =>
                $"t={t} | Expert={opp.pathGate.LastWinningExpert} | Depth={opp.LastDepthChosen}",

            _ => $"t={t} | Unknown regime"
        };
    }
}

public class Program
{
    // In Program.cs ganz oben bei den anderen statischen Feldern hinzufügen:
    public static UnifiedMultiHeadTransformerLSTMCell? _fractalCell = null;
    private static int _hiddenSize = 128; // Standardgröße für deine UMHT-Zelle
    private const bool USE_SAMPLING = false;
    private const bool USE_RANDOMIZED_PHASE = false;
    private const bool USE_FRACTAL_CORE = true;
    // holders for word tokenizer and embedding used by menu
    public static WordTokenizer? _wordTokenizerHolder = null;
    public static Module? _wordEmbeddingHolder = null;
    // separate output layer for word vocabulary (hidden -> word vocab logits)
    public static Module? _wordOutputHolder = null;

    // in-memory example store for retrieval: contexts and response token ids
    private static List<Tensor> _exampleContexts = new List<Tensor>();
    private static List<int> _exampleResponses = new List<int>();
    // (QA memory removed — learning via LSTM instead)

    public static (Tensor, Tensor) GenerateSineBatch(int batchSize, int seqLen)
    {
        var x = torch.zeros(batchSize, seqLen, 1);
        var y = torch.zeros(batchSize, seqLen, 1);

        var rnd = new Random();

        for (int b = 0; b < batchSize; b++)
        {
            double phase = rnd.NextDouble() * Math.PI * 2;

            for (int t = 0; t < seqLen; t++)
            {
                double v = Math.Sin(phase + t * 0.1);
                x[b, t, 0] = (float)v;
                y[b, t, 0] = (float)Math.Sin(phase + (t + 1) * 0.1);
            }
        }

        return (x, y);

    }

    public static Tensor SliceTimeStep(Tensor x, int t)
    {
        // x shape: [batch, seqLen, features]
        var scalar = x.index(new TensorIndex[] {
        TensorIndex.Single(0),
        TensorIndex.Single(t),
        TensorIndex.Single(0)
    });

        // Return shape [1,1]
        return scalar.unsqueeze(0).unsqueeze(1);
    }

    // Cosine similarity helper: returns scalar float in [-1,1]
    public static float CosineSimilarity(Tensor a, Tensor b, float eps = 1e-8f)
    {
        // flatten inputs to 1-D
        var af = a.flatten();
        var bf = b.flatten();

        var dot = (af * bf).sum();
        var na = torch.sqrt((af * af).sum());
        var nb = torch.sqrt((bf * bf).sum());

        var denom = na * nb + torch.tensor(eps);
        var sim = dot / denom;
        return sim.ToSingle();
    }
    public static (List<Tensor> inputs, List<float> targets) GetMandelbrotTrajectory(float cx, float cy, int steps = 64)
    {
        var inputs = new List<Tensor>();
        var targets = new List<float>();

        float zx = 0, zy = 0;

        for (int i = 0; i < steps; i++)
        {
            // 1. Aktuellen Zustand speichern (Input für das Modell)
            var inputVec = torch.zeros(new long[] { 1, 64 });
            inputVec[0, 0] = torch.tensor(cx);
            inputVec[0, 1] = torch.tensor(cy);
            // Wir begrenzen auch die Inputs, damit das Modell keine astronomischen Zahlen sieht
            inputVec[0, 2] = torch.tensor(Math.Clamp(zx, -2.1f, 2.1f));
            inputVec[0, 3] = torch.tensor(Math.Clamp(zy, -2.1f, 2.1f));

            inputs.Add(inputVec);

            // 2. Nächsten Schritt berechnen
            float nextZx = zx * zx - zy * zy + cx;
            float nextZy = 2 * zx * zy + cy;
            zx = nextZx;
            zy = nextZy;

            // 3. Target berechnen & hart bei 2.0 deckeln
            float magnitude = (float)Math.Sqrt(zx * zx + zy * zy);

            // WICHTIG: Wenn der Wert > 2 ist, setzen wir ihn fest auf 2.0
            // Das ist das Signal für das Modell: "Ziel erreicht, Punkt ist raus."
            targets.Add(Math.Min(magnitude, 2.0f));
        }

        return (inputs, targets);
    }
    public static void RunMandelbrotTrajectoryTest(FractalOpponent predictor)
    {
        // Stabile Learning Rate für tiefe Rekursion
        var optimizer = torch.optim.Adam(predictor.parameters(), lr: 1e-4);

        torch.manual_seed(42);
        predictor.train();

        // Punkt im "Seepferdchen-Tal" (komplexes Grenzverhalten)
        float cx = -0.743643887037151f;
        float cy = 0.131825904205311f;

        for (int epoch = 0; epoch < 1000; epoch++)
        {
            optimizer.zero_grad();
            var (inputs, targets) = GetMandelbrotTrajectory(cx, cy, 64);

            var h = torch.zeros(new long[] { 1, 64 });
            var c = torch.zeros(new long[] { 1, 64 });

            var predictions = new List<Tensor>();
            Tensor? lastInputTensor = null;

            for (int t = 0; t < 64; t++)
            {
                if (t == 63) lastInputTensor = inputs[t].clone();

                var (pred, hNext, cNext) = predictor.forward(inputs[t], h, c);
                predictions.Add(pred.mean());

                h = hNext;
                c = cNext;

                inputs[t].Dispose();
            }

            var predictionTensor = torch.stack(predictions).view(-1);
            var targetTensor = torch.tensor(targets.ToArray());

            var loss = torch.nn.functional.mse_loss(predictionTensor, targetTensor);

            loss.backward();
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm: 1.0);
            optimizer.step();

                float finalPred = predictionTensor[0].item<float>();
                float finalTarget = targets[0];

                Console.WriteLine($"\n--- Epoch {epoch:D3} ---");
                Console.WriteLine($"Loss: {loss.item<float>():F8}");

                // Gegenüberstellung: Vorhersage vs. Erwartung am Ende der 64 Schritte
                Console.WriteLine($"Step 63 -> [Expected: {finalTarget:F4}] | [Predicted: {finalPred:F4}]");

                if (lastInputTensor is not null)
                {
                    var data = lastInputTensor.data<float>();
                    //Console.WriteLine($"Last State (t=63) -> cx: {data[0]:F4}, cy: {data[1]:F4}, zx: {data[2]:F4}, zy: {data[3]:F4}");
                    lastInputTensor.Dispose();
                }
            

            // Cleanup
            predictionTensor.Dispose();
            targetTensor.Dispose();
            h.Dispose();
            c.Dispose();
        }
    }
    public static void TestPrediction(UnifiedMultiHeadTransformerLSTMCell cell)
    {
        int seqLen = 20;

        // Generate a single sine sequence
        var x = torch.zeros(1, seqLen, 1);
        var y = torch.zeros(1, seqLen, 1);

        double phase = 0.0;

        for (int t = 0; t < seqLen; t++)
        {
            double v = Math.Sin(phase + t * 0.1);
            x[0, t, 0] = (float)v;
            y[0, t, 0] = (float)Math.Sin(phase + (t + 1) * 0.1);
        }

        // Hidden state
        var h = torch.zeros(1, cell.hiddenSize);
        var c = torch.zeros(1, cell.hiddenSize);

        Console.WriteLine("=== Forward‑Only Prediction Test ===");

        for (int t = 0; t < seqLen; t++)
        {
            // Slice x[:, t, :]
            var xt = SliceTimeStep(x, t);

            // Forward pass
            //(h, c) = cell.forward_step(xt, h, c);

            // Extract prediction from h[:,0]
            var predCol0 = h.index(new TensorIndex[] {
            TensorIndex.Ellipsis,
            TensorIndex.Single(0)
        });

            float inputVal = xt[0, 0].ToSingle();
            float predVal = predCol0.ToSingle();
            float targetVal = y[0, t, 0].ToSingle();

            Console.WriteLine(
                $"t={t:00} | input={inputVal,8:F4} | pred={predVal,8:F4} | target={targetVal,8:F4}"
            );
        }

        Console.WriteLine("=== End Test ===");
    }

    public static Tensor GenerateLongSine(int totalSteps)
    {
        var x = torch.zeros(totalSteps, 1);

        for (int t = 0; t < totalSteps; t++)
            x[t, 0] = (float)Math.Sin(t * 0.05);

        return x; // [T,1]
    }

    public static (Tensor, Tensor) MakeSlidingWindowBatch(
    Tensor longSine, int start, int seqLen)
    {
        // Input window
        var x = longSine.index(new TensorIndex[] {
        TensorIndex.Slice(start, start + seqLen),
        TensorIndex.Ellipsis
    }).unsqueeze(0); // [1, seqLen, 1]

        // Target is the next value
        var scalar = longSine.index(new TensorIndex[] {
            TensorIndex.Single(start + seqLen),
            TensorIndex.Single(0)
        }); // scalar tensor
        var y = scalar.unsqueeze(0).unsqueeze(0); // [1,1]

        return (x, y);
    }

    public static void TrainSlidingSine(
        UnifiedMultiHeadTransformerLSTMCell cell,
        Module embedding,
        int seqLen = 50,
        int epochs = 3,
        int vocabSize = 100)
    {
        int totalSteps = 200;
        var series = GenerateLongSine(totalSteps); // [T,1]

        var tokenizer = new SineTokenizer(vocabSize);

        // precompute token ids for entire series
        var tokens = new int[totalSteps];
        for (int i = 0; i < totalSteps; i++)
        {
            var scalar = series.index(new TensorIndex[] { TensorIndex.Single(i), TensorIndex.Single(0) });
            tokens[i] = tokenizer.Encode(scalar.ToSingle());
        }

        // include embedding parameters in optimizer
        var parameters = cell.parameters().Concat(((Module)embedding).parameters());
        var opt = torch.optim.Adam(
    parameters,
    lr: 1e-4,
    amsgrad: true // Hilft gegen das -1.0000 Bias Problem
);

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            Console.WriteLine($"=== Epoch {epoch} ===");

            for (int start = 0; start < totalSteps - seqLen - 1; start++)
            {
                opt.zero_grad();

                var h = torch.zeros(1, cell.hiddenSize);
                var c = torch.zeros(1, cell.hiddenSize);

                // Reset the internal attention memory for the cell so it does not
                // accumulate states across different sliding-window samples.
                cell.ResetMemory();

                // feed seqLen steps: series[start + t]
                Tensor loss = torch.tensor(0f);
                for (int t = 0; t < seqLen; t++)
                {
                    // teacher forcing: feed token embedding of current step
                    var inputId = tokens[start + t];
                    var inputIdx = torch.tensor(new long[] { inputId });

                    // embedding forward
                    var emb = ((Module<Tensor, Tensor>)embedding).forward(inputIdx); // [1, embDim]

                    //(h, c) = cell.forward_step(emb, h, c);

                    // output logits over vocab
                    var logits = cell.output.forward(h); // [1, vocabSize]

                    // target token id
                    var targetId = tokens[start + t + 1];
                    var targetTensor = torch.tensor(new long[] { targetId });

                    loss += nn.functional.cross_entropy(logits, targetTensor);
                }

                // average loss over sequence length to keep scale consistent
                loss = loss / seqLen;

                loss.backward();

                // clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(cell.parameters(), 1.0);

                opt.step();


                Console.WriteLine($"start={start} loss={loss.ToSingle():F6}");
            }
        }
    }


    public static void PredictSliding(
    UnifiedMultiHeadTransformerLSTMCell cell,
    Module embedding,
    int seqLen = 50)
    {
        var longSine = GenerateLongSine(2000);

        // Take a window from the middle
        int start = 500;

        var (x, yTrue) = MakeSlidingWindowBatch(longSine, start, seqLen);

        var h = torch.zeros(1, cell.hiddenSize);
        var c = torch.zeros(1, cell.hiddenSize);

        var tokenizer = new SineTokenizer(100);

        for (int t = 0; t < seqLen; t++)
        {
            // teacher-forced input: use token embedding for step t
            // compute token ids for the longSine window
            var inputId = tokenizer.Encode(x.select(1, t).squeeze().ToSingle());
            var inputIdx = torch.tensor(new long[] { inputId });
            var emb = ((Module<Tensor, Tensor>)embedding).forward(inputIdx);

            //(h, c) = cell.forward_step(emb, h, c);

            var logits = cell.output.forward(h); // [1, vocab]
            int predId = logits.argmax(1).ToInt32();
            float pred = tokenizer.Decode(predId);

            // actual next value after this timestep
            var actualTensor = longSine.index(new TensorIndex[] {
                TensorIndex.Single(start + t + 1),
                TensorIndex.Single(0)
            });
            float actual = actualTensor.ToSingle();

            // original scalar input value (before embedding)
            float inputVal = x.select(1, t).squeeze().ToSingle();

            Console.WriteLine($"t={start + t:000} | input={inputVal:F6} | pred={pred:F6} (id={predId}) | actual_next={actual:F6}");
        }

        var logitsFinal = cell.output.forward(h);
        int predFinalId = logitsFinal.argmax(1).ToInt32();
        float predFinal = tokenizer.Decode(predFinalId);
        var actualFinalTensor = longSine.index(new TensorIndex[] {
            TensorIndex.Single(start + seqLen),
            TensorIndex.Single(0)
        });
        float actualFinal = actualFinalTensor.ToSingle();

        Console.WriteLine($"Window end t={start + seqLen} | final_pred={predFinal:F6} (id={predFinalId}) | final_actual={actualFinal:F6}");
    }

    public static Tensor TrainStep(
    UnifiedMultiHeadTransformerLSTMCell cell,
    Tensor x, Tensor y,
    torch.optim.Optimizer opt,
    bool printDebug = false)
    {
        opt.zero_grad();

        int batch = (int)x.shape[0];
        int seq = (int)x.shape[1];

        var h = torch.zeros(batch, cell.hiddenSize);
        var c = torch.zeros(batch, cell.hiddenSize);

        Tensor loss = torch.tensor(0f);

        for (int t = 0; t < seq; t++)
        {
            Thread.Sleep(200);
            // select time step t -> shape [batch, inputSize]
            var xt = x.select(1, t);
            var target = y.select(1, t);

            // forward
            //(h, c) = cell.forward_step(xt, h, c);

            // prediction from hidden state
            var predCol0 = h.index(new TensorIndex[] {
    TensorIndex.Ellipsis,
    TensorIndex.Single(0)
});

            var predCol0_2D = predCol0.unsqueeze(1);

            // accumulate loss
            loss += torch.nn.functional.mse_loss(predCol0_2D, target);

            if (printDebug)
            {
                // use squeeze to reduce to scalar safely before converting to single
                float inputVal = xt.squeeze().ToSingle();
                float predVal = predCol0.squeeze().ToSingle();
                float targetVal = target.squeeze().ToSingle();

                Console.WriteLine(
                    $"t={t} | input={inputVal:F4} | pred={predVal:F4} | target={targetVal:F4}"
                );
            }
        }

        loss.backward();
        opt.step();

        return loss.detach();
    }

    // Simple webcrawler for testing

    // Simple placeholder classifier — replace with neural network logic as needed.
    private static bool ClassifyTokensSimple(string[] tokens)
    {
        if (tokens == null || tokens.Length == 0) return false;

        // Simple heuristic: count tokens with length > 4 as signal
        int score = 0;
        foreach (var t in tokens)
        {
            if (t.Length > 4) score++;
            // boost score for common 'relevant' cues (example)
            var lw = t.ToLowerInvariant();
            if (lw.Contains("news") || lw.Contains("research") || lw.Contains("report") || lw.Contains("study")) score += 3;
            if (lw.Contains("buy") || lw.Contains("sale") || lw.Contains("discount")) score -= 2;
        }

        // threshold
        return score >= Math.Max(10, tokens.Length / 10);
    }


    public static void CheckCudaAvailability()
    {
        try
        {
            bool cudaAvailable = torch.cuda.is_available();
            Console.WriteLine($"CUDA available: {cudaAvailable}");

            int deviceCount = 0;
            try { deviceCount = torch.cuda.device_count(); } catch { /* not supported */ }
            Console.WriteLine($"CUDA device count: {deviceCount}");

            if (cudaAvailable && deviceCount > 0)
            {
                for (int i = 0; i < deviceCount; i++)
                {
                    try
                    {
                        Console.WriteLine($"-- Device {i} test --\"");
                        // optional: set current device if API available
                        // try { torch.cuda.set_device(i); } catch { }

                        // allocate small tensor on CUDA and report device
                        var t = torch.rand(new long[] { 2, 2 }).to(torch.CUDA);
                        Console.WriteLine($"Allocated tensor device: {t.device}\"");
                        // print current device index if available
                        try
                        {

                            t.Dispose();
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Device {i} allocation failed: {ex.Message}\"");
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Device {i} test failed: {ex.Message}");
                    }

                    {
                        Console.WriteLine("No CUDA devices available or CUDA not supported by this build of TorchSharp.\"");
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("CUDA check failed: {ex.Message}\"");
        }
    }
    public static Tensor[] GenerateBaselineSequence(
    UnifiedMultiHeadTransformerLSTMCell cell,
    int steps,
    int hiddenSize)
    {
        var h = torch.zeros(1, hiddenSize);
        var c = torch.zeros(1, hiddenSize);

        var inputs = new Tensor[steps];

        for (int t = 0; t < steps; t++)
        {
            // Feed zero or noise or a learned embedding
            var x = torch.zeros(1, cell.InputSize);

            var (outT, hNext, cNext) = cell.forward_step(x, h, c);

            // Use hidden state as the next input to the fractal opponent
            inputs[t] = hNext.detach();

            h.Dispose();
            c.Dispose();
            h = hNext;
            c = cNext;
        }

        return inputs;
    }
    public static Tensor WeightMatchingLoss(nn.Module teacher, nn.Module student, float lambda)
    {
        Tensor total = torch.tensor(0f);

        var tDict = teacher.state_dict();
        var sDict = student.state_dict();

        foreach (var key in tDict.Keys)
        {
            if (!sDict.ContainsKey(key)) continue;

            var t = tDict[key];
            var s = sDict[key];

            // MSE between teacher and student weights
            total += lambda * torch.nn.functional.mse_loss(s, t);
        }

        return total;
    }
    public static void CompareWeights(nn.Module teacher, nn.Module student)
    {
        var tDict = teacher.state_dict();
        var sDict = student.state_dict();

        Console.WriteLine("=== Weight Convergence Debug ===");

        foreach (var key in tDict.Keys)
        {
            if (!sDict.ContainsKey(key))
            {
                Console.WriteLine($"Missing key in student: {key}");
                continue;
            }

            var t = tDict[key];
            var s = sDict[key];

            // Cosine similarity between flattened weights
            var sim = Program.CosineSimilarity(t.flatten(), s.flatten());
            var diff = (t - s).pow(2).mean().ToSingle();

            Console.WriteLine($"{key,-40} | cos={sim:F4} | mse={diff:F6}");
        }
    }
    public static void TrainFractalSupervisedFixedInput(
       FractalOpponent generator,
       FractalOpponent predictor,
       Embedding tokenEmbedding,
       int seqLen = 6)
    {
        var device = torch.CPU;
        foreach (var p in generator.parameters()) p.requires_grad = false;

        var opt = torch.optim.Adam(
            predictor.parameters().Concat(tokenEmbedding.parameters()),
            lr: 1e-4
        );

        var inputTokens = Enumerable.Repeat(1, seqLen).ToArray();
        float lambda = 0.01f;
        float entropyCoef = 0.05f;

        for (int epoch = 0; epoch < 2000; epoch++)
        {
            opt.zero_grad();

            var hGen = torch.zeros(new long[] { 1, generator.hiddenSize }).to(device);
            var cGen = torch.zeros(new long[] { 1, generator.hiddenSize }).to(device);
            var hPred = torch.zeros(new long[] { 1, predictor.hiddenSize }).to(device);
            var cPred = torch.zeros(new long[] { 1, predictor.hiddenSize }).to(device);

            Tensor totalLoss = torch.tensor(0f).to(device);

            // Speicher für exakte Werte (Letzter Zeitschritt)
            float[]? finalExpectedVec = null;
            float[]? finalGottenVec = null;

            for (int t = 0; t < seqLen; t++)
            {
                var idx = torch.tensor(new long[] { inputTokens[t] }).to(device);
                var emb = tokenEmbedding.forward(idx);

                var (outGen, hGenNext, cGenNext) = generator.forward(emb, hGen, cGen);
                var (outPred, hPredNext, cPredNext) = predictor.forward(emb, hPred, cPred);

                // --- SCALED MSE LOGIK ---
                // Wir berechnen die Range (Max - Min) des Teacher-Signals zur Normalisierung
                // Das verhindert, dass kleine Fluktuationen bei großen Werten ignoriert werden.
                using (var teacher = outGen.detach())
                {
                    var diff = outPred - teacher;

                    // 1. Huber Loss statt MSE: 
                    // Wirkt wie MSE bei kleinen Fehlern, aber wie MAE bei großen (keine Explosion).
                    var huber = torch.nn.functional.huber_loss(outPred, teacher, reduction: nn.Reduction.Mean, delta: 0.1f);

                    // 2. Log-Variance Dämpfung (Optional):
                    // Anstatt hart zu dividieren, nutzen wir den Logarithmus der Streuung,
                    // um den Loss nur sanft zu verstärken, wenn das Signal komplex ist.
                    var logStd = (teacher.std() + 1e-6f).log().clamp(-2.0f, 2.0f);
                    var scaleFactor = (1.0f - logStd).clamp(0.5f, 2.0f);

                    // Finaler Loss
                    totalLoss += huber * scaleFactor;

                    if (t == seqLen - 1)
                    {
                        // Exakte Vektoren für den Debug-Print kopieren
                        finalExpectedVec = teacher.view(-1).data<float>().ToArray();
                        finalGottenVec = outPred.detach().view(-1).data<float>().ToArray();


                        string fileName = $"C:\\xampp\\htdocs\\deepseekx\\bin\\Debug\\net8.0\\win-x64\\vector_match.csv";
                        using (var writer = new StreamWriter(fileName,true))
                        {
                            writer.WriteLine("Dimension,Expected,Gotten");
                            for (int i = 0; i < finalExpectedVec.Length; i++)
                            {
                                writer.WriteLine($"{i},{finalExpectedVec[i].ToString(System.Globalization.CultureInfo.InvariantCulture)},{finalGottenVec[i].ToString(System.Globalization.CultureInfo.InvariantCulture)}");
                            }
                        }
                        Console.WriteLine($"[PLOT] Vektor-Snapshot gespeichert: {fileName}");
                        // --- DEBUG OUTPUT (Präzise Darstellung) ---

                    }
                }

                // Router Entropy (bleibt gleich)
                var depthLogits = predictor.lastdepthlogits;
                if (depthLogits is not null)
                {
                    var probs = torch.nn.functional.softmax(depthLogits, dim: 1);
                    var logProbs = torch.nn.functional.log_softmax(depthLogits, dim: 1);
                    var entropy = -(probs * logProbs).sum();
                    totalLoss -= entropyCoef * entropy;
                }

                hGen = hGenNext; cGen = cGenNext;
                hPred = hPredNext; cPred = cPredNext;
            }
            float progress = (float)epoch / 2000;
            // Simulated Annealing für den Entropy-Koeffizienten (sinkt über Zeit)
            float currentEntropyCoef = 0.05f * (1.0f - progress);

            // Optional: Annealing für die Gewichtung des WeightMatchingLoss
            float currentLambda = 0.1f + (progress * 0.4f); // Wird strenger gegen Ende
            totalLoss += WeightMatchingLoss(generator, predictor, currentLambda);
            totalLoss.backward();

            // --- GRADIENT FLOW DEBUG ---
            Console.WriteLine($"--- Gradient Flow (Epoch {epoch}) ---");
                foreach (var (name, param) in predictor.named_parameters())
                {
                    if (param.grad is not null)
                    {
                        var gradNorm = param.grad.norm().item<float>();
                        var paramNorm = param.norm().item<float>();
                        // Das Verhältnis zeigt, wie stark die Änderung relativ zur Größe ist
                        Console.WriteLine($"{name,-30} | Grad Norm: {gradNorm:E4} | Ratio: {(gradNorm / paramNorm):E4}");
                    }
                    else
                    {
                        Console.WriteLine($"{name,-30} | NO GRADIENT");
                    }
                }


            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm: 0.1);
            opt.step();

            
            Console.WriteLine($"\nEpoch {epoch} -----------------------------------");
                Console.WriteLine($"[LOSS] Total: {totalLoss.item<float>():F6}");

                // Zeige die ersten 5 exakten Dimensionen des 64D Vektors
                Console.WriteLine("[DATA] Exact Vector Matching (First 5 dims):");
                for (int i = 0; i < 4; i++)
                {
                    Console.WriteLine($"  Dim {i}: Exp: {finalExpectedVec[i]:F6} | Got: {finalGottenVec[i]:F6} | Δ: {Math.Abs(finalExpectedVec[i] - finalGottenVec[i]):F6}");
                }
            
        }
    }
    public static int[] GenerateComplexSineTokens(int count, int vocabSize)
    {
        int[] tokens = new int[count];
        for (int i = 0; i < count; i++)
        {
            // Superpose two frequencies to create "hidden complexity"
            double slow = Math.Sin(i * 0.05);
            double fast = Math.Sin(i * 0.2) * 0.5;
            double signal = (slow + fast + 1.5) / 3.0; // Normalize to [0, 1]

            // Map to vocab range
            tokens[i] = (int)Math.Clamp(signal * vocabSize, 0, vocabSize - 1);
        }
        return tokens;
    }
    public static void Main(string[] args)
    {
        string dna = GenomicEngine.LoadAndCleanSequence("C:\\xampp\\htdocs\\deepseekx\\genome.txt");
        // 1. Hyperparameter definieren
        int hiddenSize = 16; // Größe des Gedächtnisses
        int embeddingDim = 16; // Komplexität der Basen-Darstellung
        int numTokens = 4;    // N, A, C, G, T

        Console.WriteLine("--- Initialisiere Genom-KI ---");

        // 2. Predictor und Embedding initialisieren
        // FractalOpponent ist deine spezifische Gate-Architektur
        var predictor = new FractalOpponent(new UnifiedMultiHeadTransformerLSTMCell(inputSize: embeddingDim,16,4,4), hiddenSize: hiddenSize,4);

        // Embedding mappt Token-IDs (1-4) auf Vektoren der Größe 64
        var tokenEmbedding = torch.nn.Embedding(numTokens, embeddingDim);

        // 3. Daten laden


        string dnaSequence = dna;

        // 4. Training starten
        // Wir übergeben die Instanzen an die Trainings-Funktion
        Console.WriteLine("Starte Training auf der geladenen Sequenz...");

        try
        {
            // Wir nutzen hier die stabilisierte Funktion mit dem Optimizer-Patch
            GenomicEngine.TrainOnGenomics(predictor, tokenEmbedding, dnaSequence, epochs: 100, windowSize: 16);
            
            Console.WriteLine("\n--- Training abgeschlossen ---");

            // 5. Ein kurzes Test-Szenario: Was sagt das Modell über die ersten 10 Basen?

        }
        catch (Exception ex)
        {
            Console.WriteLine($"Ein Fehler ist aufgetreten: {ex.Message}");
        }
        return;
        int H = 32;
        int vocabSize = 100;
        int maxDepth = 3;
        var device = torch.CPU;

        var genCell = new UnifiedMultiHeadTransformerLSTMCell(H, H, 4).to(device);
        var generator = new FractalOpponent(genCell, H, maxDepth).to(device);

        var predCell = new UnifiedMultiHeadTransformerLSTMCell(H, H, 4).to(device);
        //var predictor = new FractalOpponent(predCell, H, maxDepth).to(device);

        // FIX 3: Eindeutiger Name, um Konflikte zu vermeiden
        var sharedEmbedding = nn.Embedding(vocabSize, H).to(device);

        //int[] tokens = GenerateComplexSineTokens(1000, vocabSize);

         TrainFractalSupervisedFixedInput(generator,predictor, sharedEmbedding, 100);

        // Aufruf
        return;
        // Note: Do NOT start the Fractal API server automatically.
        // Starting the webserver enables the external tokenizer and can change tokenizer behavior.
        // Start it manually via menu option 11 when you want the web API.

        // start a single continuous crawler in background (non-blocking)
        try
         {
             Console.WriteLine("Starting background continuous webcrawler...");
             //_ = Task.Run(() => WebCrawler.CrawlContinuouslyAsync("http://wikipedia.org", maxPages: 200, maxDepth: 2, extraSite: "https://gematrix.org", delayMs: 5_000));
         }
         catch (Exception ex)
         {
             Console.WriteLine("Failed to start background crawler: " + ex.Message);
         }

        // interactive application continues immediately; no one-off blocking crawl

        int embeddingSize = 64;
        var embedding = nn.Embedding(vocabSize, embeddingSize);

        var cell = new UnifiedMultiHeadTransformerLSTMCell(
           inputSize: embeddingSize,
           hiddenSize: 64,
           numHeads: 4,
           outputSize: vocabSize
       );

        // Also create a FractalOpponent instance to use the fractal core in Main
        var fractal = new FractalOpponent(hiddenSize: 64, maxDepth: 2);

        // Example usage of fractal opponent with a random input tensor
        var frInput = torch.randn(1, 64);
        var frH = torch.zeros(1, 64);
        var frC = torch.zeros(1, 64);
      

        /*
                        TrainSlidingSine(cell, embedding, seqLen: 50, epochs: 3);

                        PredictSliding(cell, embedding, seqLen: 50);
                */

        // interactive menu
        var tokenizer = new SineTokenizer(vocabSize);

        while (true)
        {
            Console.WriteLine();
            Console.WriteLine("=== Menu ===");
            Console.WriteLine("1) predict (sliding window output)");
            Console.WriteLine("2) tokenize (float -> token id)");
            Console.WriteLine("3) detokenize (token id -> float)");
            Console.WriteLine("4) chat/predict (enter floats, get next-token predictions)");
            Console.WriteLine("5) exit");
            Console.WriteLine("6) build word vocab from lines (interactive)");
            Console.WriteLine("7) tokenize words (sentence -> tokens)");
            Console.WriteLine("8) detokenize word token ids (ids -> sentence)");
            Console.WriteLine("9) chat/predict words (tokens -> next word)");
            Console.WriteLine("10) teach (enter question and response)");
            Console.WriteLine("11) start API server (enable external tokenizer for web requests)");
            Console.Write("Choose option: ");
            var choice = Console.ReadLine();

            if (choice == null) continue;

            switch (choice.Trim())
            {
                case "1":
                    PredictSliding(cell, embedding, seqLen: 50);
                    break;
                case "2":
                    Console.Write("Enter float value [-1..1]: ");
                    var s = Console.ReadLine();
                    if (float.TryParse(s, out var f))
                    {
                        var id = tokenizer.Encode(f);
                        var dec = tokenizer.Decode(id);
                        Console.WriteLine($"token={id} decoded={dec:F6}");
                    }
                    else
                        Console.WriteLine("Invalid float.");
                    break;
                case "3":
                    Console.Write("Enter token id: ");
                    var sid = Console.ReadLine();
                    if (int.TryParse(sid, out var idv))
                    {
                        var decv = tokenizer.Decode(idv);
                        Console.WriteLine($"decoded={decv:F6}");
                    }
                    else
                        Console.WriteLine("Invalid token id.");
                    break;
                case "4":
                    Console.WriteLine("Enter floats separated by space (e.g. 0.1 0.2 ...):");
                    var line = Console.ReadLine();
                    if (string.IsNullOrWhiteSpace(line)) break;

                    var parts = line.Split(new[] { ' ', ',' }, StringSplitOptions.RemoveEmptyEntries);

                    var h = torch.zeros(1, cell.hiddenSize);
                    var c = torch.zeros(1, cell.hiddenSize);
                    var output = torch.zeros(1, vocabSize);
                    foreach (var p in parts)
                    {
                        if (!float.TryParse(p, out var fv))
                        {
                            Console.WriteLine($"Could not parse '{p}', skipping.");
                            continue;
                        }

                        var tid = tokenizer.Encode(fv);
                        var inputIdx = torch.tensor(new long[] { tid });
                        var emb = ((Module<Tensor, Tensor>)embedding).forward(inputIdx);

                        (output, h, c) = cell.forward_step(emb, h, c);

                        var logits = cell.output.forward(h);
                        var predId = logits.argmax(1).ToInt32();
                        var predVal = tokenizer.Decode(predId);

                        Console.WriteLine($"input={fv:F6} -> pred_next={predVal:F6} (id={predId})");
                    }
                    break;
                // ---------------- Word tokenizer / chat ----------------
                case "6":
                case "build_words":
                    Console.WriteLine("Enter corpus lines (empty line to finish):");
                    var lines = new List<string>();
                    while (true)
                    {
                        var l = Console.ReadLine();
                        if (string.IsNullOrWhiteSpace(l)) break;
                        lines.Add(l);
                    }
                    if (lines.Count == 0)
                    {
                        Console.WriteLine("No lines provided.");
                        break;
                    }
                    var wordTokenizer = new WordTokenizer();
                    wordTokenizer.BuildVocabulary(lines, maxVocab: 10000);
                    // ensure EOS token exists for generation
                    var eosId = wordTokenizer.AddWord("<eos>");
                    // create embedding for words
                    var wordEmbedding = nn.Embedding(wordTokenizer.VocabSize, embeddingSize);
                    // store in local variables by closing over via tuple (cheap):
                    _wordTokenizerHolder = wordTokenizer;
                    _wordEmbeddingHolder = wordEmbedding;
                    // create output layer for word vocab
                    _wordOutputHolder = nn.Linear(cell.hiddenSize, wordTokenizer.VocabSize);
                    Console.WriteLine($"Built word vocab size={wordTokenizer.VocabSize} (EOS id={eosId})");
                    break;
                case "7":
                case "tokenize_words":
                    if (_wordTokenizerHolder == null)
                    {
                        Console.WriteLine("Word tokenizer not built yet. Use option 6 to build.");
                        break;
                    }
                    Console.Write("Enter sentence to tokenize: ");
                    var sent = Console.ReadLine();
                    var toks = _wordTokenizerHolder.Tokenize(sent);
                    Console.WriteLine("Tokens: " + string.Join(",", toks));
                    Console.WriteLine("Detokenized: " + _wordTokenizerHolder.Detokenize(toks));
                    break;
                case "8":
                case "detokenize_words":
                    if (_wordTokenizerHolder == null)
                    {
                        Console.WriteLine("Word tokenizer not built yet. Use option 6 to build.");
                        break;
                    }
                    Console.Write("Enter token ids separated by commas or spaces: ");
                    var idLine = Console.ReadLine();
                    if (string.IsNullOrWhiteSpace(idLine)) break;
                    var idParts = idLine.Split(new[] { ',', ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                    var ids = new List<int>();
                    foreach (var ip in idParts)
                        if (int.TryParse(ip, out var ii)) ids.Add(ii);
                    Console.WriteLine("Detokenized: " + _wordTokenizerHolder.Detokenize(ids));
                    break;
                case "9":
                case "chat_words":
                    int predIdW = -1;
                    if (_wordTokenizerHolder == null || _wordEmbeddingHolder == null)
                    {
                        Console.WriteLine("Word tokenizer/embedding not ready. Build with option 6.");
                        break;
                    }
                    Console.WriteLine("Enter words or sentence:");
                    var ws = Console.ReadLine();
                    if (string.IsNullOrWhiteSpace(ws)) break;

                    var wparts = ws.Split(new[] { ' ', ',' }, StringSplitOptions.RemoveEmptyEntries);
                    var hh = torch.zeros(1, cell.hiddenSize);
                    var cc = torch.zeros(1, cell.hiddenSize);
                    var outputW = torch.zeros(1, _wordTokenizerHolder.VocabSize);
                    // Feed all tokens as context first
                    int eosIdLocal = _wordTokenizerHolder.Encode("<eos>");
                    bool contextHasEos = false;
                    foreach (var w in wparts)
                    {
                        var wid = _wordTokenizerHolder.Encode(w);
                        if (wid == eosIdLocal)
                        {
                            contextHasEos = true;
                            break; // stop feeding after EOS
                        }

                        var inputIdxW = torch.tensor(new long[] { wid });
                        var embW = ((Module<Tensor, Tensor>)_wordEmbeddingHolder).forward(inputIdxW);
                        (outputW, hh, cc) = cell.forward_step(embW, hh, cc);
                    }

                    if (contextHasEos)
                    {
                        Console.WriteLine("Input contains <eos>; skipping generation.");
                    }
                    else
                    {
                        // --- GENERATION: produce tokens until <eos> using cloned state ---
                        int maxGen = 50;
                        var genIds = new List<int>();
                        var hhGen = hh.clone();
                        var ccGen = cc.clone();

                        // ensure placeholders for subsequent feedback logic
                        predIdW = -1;

                        for (int gi = 0; gi < maxGen; gi++)
                        {
                            var logitsGen = ((Module<Tensor, Tensor>)_wordOutputHolder).forward(hhGen); // [1,V]

                            // compute probabilities and show top-k for debugging
                            var probs = logitsGen.softmax(1);
                            int debugTopK = Math.Min(20, (int)probs.shape[1]);
                            var topRes = probs.topk(debugTopK, 1);
                            var topVals = topRes.Item1; // [1,debugTopK]
                            var topIdxs = topRes.Item2; // [1,debugTopK]

                            Console.Write("Top candidates: ");
                            for (int ti = 0; ti < Math.Min(5, debugTopK); ti++)
                            {
                                int tid = topIdxs[0, ti].ToInt32();
                                float pv = topVals[0, ti].ToSingle();
                                Console.Write($"'{_wordTokenizerHolder.Decode(tid)}'({tid}):{pv:F3} ");
                            }
                            Console.WriteLine();

                            // explicit probs for tokens of interest
                            int eosIdDbg = _wordTokenizerHolder.Encode("<eos>");
                            int youIdDbg = _wordTokenizerHolder.Encode("you");
                            float eosProb = 0f;
                            float youProb = 0f;
                            try
                            {
                                eosProb = probs[0, eosIdDbg].ToSingle();
                                youProb = probs[0, youIdDbg].ToSingle();
                            }
                            catch { }
                            Console.WriteLine($"Debug probs -> '<eos>'({eosIdDbg})={eosProb:F6}, 'you'({youIdDbg})={youProb:F6}");

                            // keep greedy decoding (argmax) for now
                            int eosIdDbgLocal = _wordTokenizerHolder.Encode("<eos>");
                            int gid = logitsGen.argmax(1).ToInt32();
                            // if argmax is EOS, try next-best candidate to allow multi-token generation
                            if (gid == eosIdDbgLocal)
                            {
                                // look at top-k candidates and pick first non-EOS
                                var topResFallback = probs.topk(Math.Min(10, (int)probs.shape[1]), 1);
                                var candIdxs = topResFallback.Item2; // [1,k]
                                bool found = false;
                                for (int ci = 0; ci < (int)candIdxs.shape[1]; ci++)
                                {
                                    int cand = candIdxs[0, ci].ToInt32();
                                    if (cand != eosIdDbgLocal)
                                    {
                                        gid = cand;
                                        found = true;
                                        break;
                                    }
                                }
                                if (!found)
                                {
                                    Console.WriteLine("<EOS> reached. Generation finished.");
                                    break;
                                }
                            }
                            var gw = _wordTokenizerHolder.Decode(gid);

                            genIds.Add(gid);
                            Console.WriteLine($"gen[{gi}] -> '{gw}' (id={gid})");

                            // feed generated token embedding into cloned state
                            var idxG = torch.tensor(new long[] { gid });
                            var embG = ((Module<Tensor, Tensor>)_wordEmbeddingHolder).forward(idxG);
                            (outputW, hhGen, ccGen) = cell.forward_step(embG, hhGen, ccGen);
                        }
                    }

                    // continue with original single-step prediction (first-next) for compatibility with feedback flow
                    // Ask user how many next words to predict and generate that many tokens greedily
                    try
                    {
                        Console.Write("Enter number of words to predict (n, empty=1): ");
                        var nLine = Console.ReadLine();
                        int nPred = 1;
                        if (!string.IsNullOrWhiteSpace(nLine) && !int.TryParse(nLine.Trim(), out nPred))
                            nPred = 1;

                        var hhGenFinal = hh.clone();
                        var ccGenFinal = cc.clone();
                        var genIdsFinal = new List<int>();

                        for (int gi = 0; gi < Math.Max(0, nPred); gi++)
                        {
                            var logitsGen = ((Module<Tensor, Tensor>)_wordOutputHolder).forward(hhGenFinal);
                            var probsGen = logitsGen.softmax(1);
                            int eid = _wordTokenizerHolder.Encode("<eos>");
                            int gid = logitsGen.argmax(1).ToInt32();
                            if (gid == eid)
                            {
                                var top = probsGen.topk(Math.Min(10, (int)probsGen.shape[1]), 1);
                                var candIdxs = top.Item2;
                                bool foundNonEos = false;
                                for (int ci = 0; ci < (int)candIdxs.shape[1]; ci++)
                                {
                                    int cand = candIdxs[0, ci].ToInt32();
                                    if (cand != eid)
                                    {
                                        gid = cand;
                                        foundNonEos = true;
                                        break;
                                    }
                                }
                                if (!foundNonEos)
                                {
                                    Console.WriteLine("<EOS> reached during multi-step generation.");
                                    break;
                                }
                            }
                            var gw = _wordTokenizerHolder.Decode(gid);

                            genIdsFinal.Add(gid);

                            var idxG = torch.tensor(new long[] { gid });
                            var embG = ((Module<Tensor, Tensor>)_wordEmbeddingHolder).forward(idxG);
                            (outputW, hhGenFinal, ccGenFinal) = cell.forward_step(embG, hhGenFinal, ccGenFinal);
                        }

                        if (genIdsFinal.Count > 0)
                        {
                            var genWords = genIdsFinal.Select(id => _wordTokenizerHolder.Decode(id));
                            Console.WriteLine("Predicted sequence: " + string.Join(' ', genWords));
                        }
                        else
                        {
                            Console.WriteLine("No tokens generated.");
                        }
                    }
                    catch
                    {
                        // fallback to single-step prediction on error
                        try
                        {
                            var logitsW = ((Module<Tensor, Tensor>)_wordOutputHolder).forward(hh);
                            predIdW = logitsW.argmax(1).ToInt32();
                            string predWord = _wordTokenizerHolder.Decode(predIdW);
                            Console.WriteLine($"Predicted next after input: '{predWord}' (id={predIdW})");
                        }
                        catch { }
                    }
                    break;
                case "10":
                case "teach":
                    if (_wordTokenizerHolder == null || _wordEmbeddingHolder == null)
                    {
                        Console.WriteLine("Word tokenizer/embedding not ready. Build with option 6.");
                        break;
                    }
                    Console.WriteLine("Enter question (sentence):");
                    var qText = Console.ReadLine();
                    if (string.IsNullOrWhiteSpace(qText)) break;
                    Console.WriteLine("Enter response sentence:");
                    var rText = Console.ReadLine();
                    if (string.IsNullOrWhiteSpace(rText)) break;

                    var qTokens = _wordTokenizerHolder.Tokenize(qText);
                    var rTokens = _wordTokenizerHolder.Tokenize(rText);

                    if (rTokens.Length == 0)
                    {
                        Console.WriteLine("Response must contain at least one token.");
                        break;
                    }

                    // optimizer over cell + word embedding + word output
                    var paramList = cell.parameters()
                        .Concat(((Module)_wordEmbeddingHolder).parameters())
                        .Concat(((Module)_wordOutputHolder).parameters());

                    // perform multiple training iterations on this single example
                    var optTeachExample = torch.optim.Adam(paramList, lr: 1e-3);

                    int trainIters = 5; // increase to allow the model to learn the mapping
                    Tensor lastLoss = torch.tensor(0f);

                    for (int it = 1; it <= trainIters; it++)
                    {
                        optTeachExample.zero_grad();

                        // initial hidden state
                        var hTeach = torch.zeros(1, cell.hiddenSize);
                        var cTeach = torch.zeros(1, cell.hiddenSize);

                        cell.ResetMemory();

                        // feed question tokens to set context
                        foreach (var qt in qTokens)
                        {
                            var idx = torch.tensor(new long[] { qt });
                            var embq = ((Module<Tensor, Tensor>)_wordEmbeddingHolder).forward(idx);
                            (outputW, hTeach, cTeach) = cell.forward_step(embq, hTeach, cTeach);
                        }

                        // accumulate loss predicting response sequence
                        Tensor lossTeach = torch.tensor(0f);

                        // predict first response token from question context
                        var logits0 = ((Module<Tensor, Tensor>)_wordOutputHolder).forward(hTeach);
                        var target0 = torch.tensor(new long[] { rTokens[0] });
                        lossTeach += nn.functional.cross_entropy(logits0, target0);

                        // for remaining response tokens, teacher-force previous token
                        for (int i = 1; i < rTokens.Length; i++)
                        {
                            var prev = rTokens[i - 1];
                            var idxPrev = torch.tensor(new long[] { prev });
                            var embPrev = ((Module<Tensor, Tensor>)_wordEmbeddingHolder).forward(idxPrev);
                            (outputW, hTeach, cTeach) = cell.forward_step(embPrev, hTeach, cTeach);

                            var logits = ((Module<Tensor, Tensor>)_wordOutputHolder).forward(hTeach);
                            var target = torch.tensor(new long[] { rTokens[i] });
                            lossTeach += nn.functional.cross_entropy(logits, target);
                        }

                        lossTeach = lossTeach / rTokens.Length;

                        lossTeach.backward();

                        // clip grads - convert IEnumerable to array
                        var paramArr = paramList.ToArray();
                        torch.nn.utils.clip_grad_norm_(paramArr, 1.0);

                        optTeachExample.step();

                        lastLoss = lossTeach.detach();

                        if (it % 50 == 0 || it == 1)
                        {
                            Console.WriteLine($"Teach iter={it} loss={lastLoss.ToSingle():F6}");
                        }
                    }

                    // After training, generate the full response from the question context (multi-token generation)
                    var hEval = torch.zeros(1, cell.hiddenSize);
                    var cEval = torch.zeros(1, cell.hiddenSize);
                    cell.ResetMemory();
                    foreach (var qt in qTokens)
                    {
                        var idx = torch.tensor(new long[] { qt });
                        var embq = ((Module<Tensor, Tensor>)_wordEmbeddingHolder).forward(idx);
                        (outputW, hEval, cEval) = cell.forward_step(embq, hEval, cEval);
                    }

                    // iterative generation up to the length of the target or until <eos>
                    var generated = new List<int>();
                    var hGen = hEval.clone();
                    var cGen = cEval.clone();
                    int maxGenLen = Math.Max(rTokens.Length, 50);
                    int eosIdCheck = _wordTokenizerHolder.Encode("<eos>");

                    for (int gi = 0; gi < maxGenLen; gi++)
                    {
                        var logitsG = ((Module<Tensor, Tensor>)_wordOutputHolder).forward(hGen);
                        int gid = logitsG.argmax(1).ToInt32();

                        // if model predicts EOS immediately, allow fallback to next-best as earlier
                        if (gid == eosIdCheck)
                        {
                            var probsG = logitsG.softmax(1);
                            var top = probsG.topk(Math.Min(10, (int)probsG.shape[1]), 1);
                            var candIdxs = top.Item2;
                            bool found = false;
                            for (int ci = 0; ci < (int)candIdxs.shape[1]; ci++)
                            {
                                int cand = candIdxs[0, ci].ToInt32();
                                if (cand != eosIdCheck)
                                {
                                    gid = cand;
                                    found = true;
                                    break;
                                }
                            }
                            if (!found)
                            {
                                break; // all top candidates are EOS
                            }
                        }

                        if (gid == eosIdCheck)
                        {
                            break; // stop if eos
                        }

                        generated.Add(gid);

                        var idxG = torch.tensor(new long[] { gid });
                        var embG = ((Module<Tensor, Tensor>)_wordEmbeddingHolder).forward(idxG);
                        (outputW,hGen, cGen) = cell.forward_step(embG, hGen, cGen);
                    }

                    if (generated.Count > 0)
                    {
                        var genWords = generated.Select(id => _wordTokenizerHolder.Decode(id));
                        Console.WriteLine("After teaching, predicted: " + string.Join(' ', genWords));
                    }
                    else
                    {
                        Console.WriteLine("After teaching, no tokens generated.");
                    }

                    // Also show single-step next-token prediction for compatibility
                    var logitsFinal = ((Module<Tensor, Tensor>)_wordOutputHolder).forward(hEval);
                    var predIdFinal = logitsFinal.argmax(1).ToInt32();
                    var predWordFinal = _wordTokenizerHolder.Decode(predIdFinal);

                    Console.WriteLine($"Trained on example. Final loss={lastLoss.ToSingle():F6} -> Predicted next='{predWordFinal}' (id={predIdFinal})");

                    // store example context + first response token for retrieval
                    // normalize context and store on CPU to make retrieval stable
                    var hCpu = hEval.detach().to(torch.CPU);
                    var hNorm = hCpu / (torch.sqrt((hCpu * hCpu).sum()) + 1e-8);
                    _exampleContexts.Add(hNorm);
                    _exampleResponses.Add(rTokens[0]);
                    Console.WriteLine($"Saved example for retrieval (response id={rTokens[0]})");
                    break;
                case "5":
                case "exit":
                    Console.WriteLine("Exiting.");
                    goto MENU_END;
                case "11":
                    try
                    {
                        FractalApiServer.Start("http://localhost:5000/");
                        Console.WriteLine("API server started on http://localhost:5000/");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine("Failed to start API server: " + ex.Message);
                    }
                    break;
                case "12": // EHT Radiometric Test

                    InitializeTokenizer();
                    InitializeModel(128); // Initialize both tokenizer and the fractal cell

                    Console.WriteLine("Enter path to EHT CSV file:");
                    string path = Console.ReadLine();

                    var eht = new EHTIntensityWrapper(_wordTokenizerHolder);
                    var data = eht.LoadEHTCsv(path);

                    // Convert the actual physics into a structural fractal string
                    string physicsPrompt = eht.GenerateFractalPrompt(data);
                    Console.WriteLine($"Extracted Physics Grammar: {physicsPrompt}");

                    // Predict if the model can "complete" the black hole's photon rings
                    
                     RunEHTRadiometricTest(path);

                    //string result = _wordTokenizerHolder.Detokenize(prediction);
                    //Console.WriteLine($"Model's Theoretical Extension: {result}");
                    break;
                default:
                    Console.WriteLine("Unknown option.");
                    break;
            }

        }
    MENU_END:;

        return;
        
    }

    public static List<double> LoadCsvData(string csvPath)
    {
        var numericData = new List<double>();
        if (!File.Exists(csvPath)) return numericData;

        foreach (var line in File.ReadAllLines(csvPath))
        {
            // Überspringe Kommentare und leere Zeilen
            if (line.StartsWith("#") || string.IsNullOrWhiteSpace(line)) continue;

            var parts = line.Split(',');
            // Spalte 5 enthält bei EHT-Daten oft die Flux-Intensität (I)
            if (parts.Length > 4 && double.TryParse(parts[4], System.Globalization.CultureInfo.InvariantCulture, out double val))
            {
                numericData.Add(val);
            }
        }
        return numericData;
    }
    record StepLog(
    int Step,
    double InputFlux,
    double PredFlux,
    int Regime,
    int ExpertId
);
    public static void RunEHTRadiometricTest(string csvPath, int trainingIterations = 100)
    {
        var numericData = LoadCsvData(csvPath);
        if (numericData.Count == 0) return;

        double max = numericData.Max();
        double min = numericData.Min();

        // 1. High-Resolution Vocabulary (0.01 steps = 101 bins)
        var bins = new List<double>();
        for (double v = 0.0; v <= 1.0; v += 0.01)
            bins.Add(Math.Round(v, 2));

        _wordTokenizerHolder = new WordTokenizer();
        foreach (var b in bins)
            _wordTokenizerHolder.AddWord(b.ToString("F2"));

        int newVocabSize = _wordTokenizerHolder.VocabSize;
        int windowSize = 32;

        // 2. RE-INITIALIZE CORE
        int deeperMaxDepth = 4;
        _fractalCell = new UnifiedMultiHeadTransformerLSTMCell(_hiddenSize, _hiddenSize, 4);
        var opp = new FractalOpponent(_fractalCell, _hiddenSize, deeperMaxDepth);

        _wordEmbeddingHolder = nn.Embedding(newVocabSize, _hiddenSize);
        _wordOutputHolder = nn.Linear(_hiddenSize, newVocabSize);

        // 3. Proximity Tokenization
        int[] allTokens = numericData.Select(v =>
        {
            double norm = (max == min) ? 0 : (v - min) / (max - min);
            double nearestBin = bins.OrderBy(b => Math.Abs(b - norm)).First();
            return _wordTokenizerHolder.Encode(nearestBin.ToString("F2"));
        }).ToArray();

        // 4. Optimizer (includes opponent + gate + embeddings)
        var allParams = new List<Parameter>();
        allParams.AddRange(opp.parameters());
        allParams.AddRange(_wordEmbeddingHolder.parameters());
        allParams.AddRange(_wordOutputHolder.parameters());

        var optimizer = torch.optim.Adam(allParams, lr: 1e-4, weight_decay: 1e-5);
        var criterion = nn.CrossEntropyLoss();

        // 5. CSV Logger
        using var csv = new CsvLogger("eht_training_trace.csv");

        Console.WriteLine(
            $"\n--- Training [Window: {windowSize} | Depth: {deeperMaxDepth} | Vocab: {newVocabSize}] ---"
        );

        for (int epoch = 0; epoch < trainingIterations; epoch++)
        {
            opp.train();
            _wordEmbeddingHolder.train();
            _wordOutputHolder.train();

            int maxSteps = Math.Min(allTokens.Length - (windowSize + 1), 100);

            for (int i = 0; i < maxSteps; i++)
            {
                optimizer.zero_grad();

                var h = torch.zeros(1, _hiddenSize);
                var c = torch.zeros(1, _hiddenSize);

                Tensor cumulativeLoss =
                    torch.tensor(0f).to(_wordEmbeddingHolder.parameters().First().device);
                int previousExpert = -1;
                double lambda = 0.1;

                for (int j = 0; j < windowSize; j++)
                {
                    int currentTokenId = allTokens[i + j];
                    int expectedNextId = allTokens[i + j + 1];

                    using var xIn = torch.tensor(new long[] { currentTokenId }).unsqueeze(0);
                    using var emb = ((Module<Tensor, Tensor>)_wordEmbeddingHolder)
                        .forward(xIn)
                        .view(1, -1);

                    var res = opp.forward(emb, h, c);

                    int expertIdx = opp.LastWinningExpert;

                    if (previousExpert != -1 && expertIdx != previousExpert)
                    {
                        // Add a scalar penalty into the computation graph
                        var routerPenalty = torch.tensor(lambda, device: res.output.device);
                        cumulativeLoss = cumulativeLoss.add(routerPenalty);
                    }

                    previousExpert = expertIdx;

                    using var logits = ((Module<Tensor, Tensor>)_wordOutputHolder)
                        .forward(res.output);

                    using var constrainedLogits = logits.narrow(1, 0, newVocabSize);

                    int predictedId = (int)constrainedLogits
                        .argmax( 1 )
                        .item<long>();

                    // ---- Decode tokens to physical units ----
                    string inputToken = _wordTokenizerHolder.Decode(currentTokenId);
                    string expToken = _wordTokenizerHolder.Decode(expectedNextId);
                    string predToken = _wordTokenizerHolder.Decode(predictedId);

                    double realIn = double.Parse(inputToken) * (max - min) + min;
                    double realExp = double.Parse(expToken) * (max - min) + min;
                    double realPred = double.Parse(predToken) * (max - min) + min;

                    // ---- Regime classification ----
                    int regime = RegimeClassifier.Classify(realPred);

                    // ---- CSV logging ----
                    csv.Log(
    epoch: epoch,
    step: j,
    input: realIn,
    expected: realExp,
    predicted: realPred,
    regime: regime,
    expert: opp.LastWinningExpert
);


                    // ---- Console (optional) ----
                    Console.WriteLine(
                        $"Epoch {epoch} | Step {j}/{windowSize} | " +
                        $"In: {realIn:F2} | Exp: {realExp:F2}Jy | " +
                        $"Pred: {realPred:F2}Jy | Regime={(RegimeClassifier.Regime)regime}"
                    );

                    // ---- Loss ----
                    using var targetTensor = torch.tensor(new long[] { expectedNextId });
                    var stepLoss = criterion.forward(constrainedLogits, targetTensor);
                    cumulativeLoss = cumulativeLoss.add(stepLoss);

                    // ---- State update ----
                    h.Dispose(); c.Dispose();
                    h = res.h; c = res.c;
                    res.output.Dispose();
                }

                // ---- Backpropagation ----
                cumulativeLoss.backward();

                torch.nn.utils.clip_grad_norm_(allParams, 1.0);
                optimizer.step();   // ONLY ONCE

                h.Dispose();
                c.Dispose();
                cumulativeLoss.Dispose();
            }
        }
    }

    public static void InitializeTokenizer()
    {
        if (_wordTokenizerHolder == null)
        {
            Console.WriteLine("Erstelle neuen WordTokenizer...");
            _wordTokenizerHolder = new WordTokenizer();

            // 1. Basis-Vokabular für Fraktale/Logik
            string[] baseVocab = { "[", "]", "rule", "halt", "f", "+", "-" };
            foreach (var word in baseVocab) _wordTokenizerHolder.AddWord(word);

            // 2. Physik-Vokabular für die M87* Radiometrie
            // Diese Token werden von der 'TokenizePhysics'-Methode verwendet
            string[] physicsVocab = { "peak", "high", "mid", "low", "horizon", "flux" };
            foreach (var word in physicsVocab) _wordTokenizerHolder.AddWord(word);

            Console.WriteLine($"Tokenizer bereit. Vokabulargröße: {_wordTokenizerHolder.VocabSize}");
        }
    }
    public static void InitializeModel(int hiddenSize = 128)
    {
        if (_wordTokenizerHolder == null)
        {
            InitializeTokenizer(); // Ensure we have a vocab first
        }

        int vocabSize = _wordTokenizerHolder.VocabSize;
        _hiddenSize = hiddenSize;

        Console.WriteLine($"Initializing Fractal Core with Vocab Size: {vocabSize}...");

        // 1. Initialize the Embedding Layer
        _wordEmbeddingHolder = nn.Embedding(vocabSize, _hiddenSize);

        // 2. Initialize the Fractal Cell (from your UMHT.cs)
        // Assuming numHeads = 4 and maxDepth = 3
        _fractalCell = new UnifiedMultiHeadTransformerLSTMCell(_hiddenSize, _hiddenSize, numHeads: 4);

        // 3. Initialize the Output Projection
        // Im Bereich "2. RE-INITIALIZE CORE"
        _wordOutputHolder = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(_hiddenSize, vocabSize)
        );

        Console.WriteLine("Model structure successfully initialized.");
    }
    public static int[] PredictSequence(int[] seedTokenIds, int predictLength)
    {
        // Attempt to locate the cell instance. If not available, fall back to a simple
        // embedding->output greedy predictor (no recurrence) so function still works.

        if (_wordEmbeddingHolder == null || _wordOutputHolder == null || _wordTokenizerHolder == null)
        {
            // nothing to do
            return seedTokenIds.Take(Math.Max(0, seedTokenIds.Length)).ToArray();
        }

        // Build output list starting with seed
        var generatedIndices = new List<int>(seedTokenIds);

        int lastToken = seedTokenIds.Length > 0 ? seedTokenIds[seedTokenIds.Length - 1] : 0;

        // Try to get a cell if it was stored as the ParentModule of embedding (best-effort).
        UnifiedMultiHeadTransformerLSTMCell? cell = null;
        try
        {
            // TorchSharp Module doesn't expose a public ParentModule reliably across versions,
            // so try reflection as a best-effort. If not found, cell remains null and we use fallback.
            var embModule = _wordEmbeddingHolder as object;
            var parentProp = embModule?.GetType().GetProperty("parent", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public)
                         ?? embModule?.GetType().GetProperty("ParentModule", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
        }
        catch { cell = null; }

        using (torch.no_grad())
        {
            if (cell != null)
            {
                // Use the real cell (recurrent) for prediction
                // Initialize hidden states from cell.hiddenSize
                int hSize = cell.hiddenSize;
                var h = torch.zeros(new long[] { 1, hSize });
                var c = torch.zeros(new long[] { 1, hSize });
                var o = torch.zeros(new long[] { 1, hSize });
                // Process seed to build state
                foreach (var tokenId in seedTokenIds)
                {
                    var input = torch.tensor(new long[] { tokenId }); // shape [1]
                    var emb = ((Module<Tensor, Tensor>)_wordEmbeddingHolder).forward(input);
                    // emb shape: [1, embDim] or [B,embDim]
                    // UMHT forward_step expects [B, inputSize] and returns (h,c)
                    (o, h, c) = cell.forward_step(emb, h, c);
                    lastToken = tokenId;
                }

                for (int i = 0; i < predictLength; i++)
                {
                    var input = torch.tensor(new long[] { lastToken });
                    var emb = ((Module<Tensor, Tensor>)_wordEmbeddingHolder).forward(input);
                    (o, h, c) = cell.forward_step(emb, h, c);

                    // compute logits via output layer (use _wordOutputHolder)
                    var logits = ((Module<Tensor, Tensor>)_wordOutputHolder).forward(h);
                    int predictedId = logits.argmax(1).ToInt32();
                    generatedIndices.Add(predictedId);
                    lastToken = predictedId;

                    if (_wordTokenizerHolder.Decode(predictedId) == "<eos>") break;
                }
            }
            else
            {
                // Fallback: no recurrent cell available — greedy decode from embedding->output
                for (int i = 0; i < predictLength; i++)
                {
                    var input = torch.tensor(new long[] { lastToken });
                    var emb = ((Module<Tensor, Tensor>)_wordEmbeddingHolder).forward(input); // [1,embDim]

                    // Map embedding directly to vocab logits
                    var logits = ((Module<Tensor, Tensor>)_wordOutputHolder).forward(emb);
                    int predictedId = logits.argmax(1).ToInt32();
                    generatedIndices.Add(predictedId);
                    lastToken = predictedId;

                    if (_wordTokenizerHolder.Decode(predictedId) == "<eos>") break;
                }
            }
        }

        return generatedIndices.ToArray();
    }
    public class TransformerLSTMCell : nn.Module
    {
        private readonly LSTMCell lstm;

        private readonly Linear Wq;
        private readonly Linear Wk;
        private readonly Linear Wv;
        private readonly Linear proj;
        private readonly LayerNorm norm1;
        private readonly LayerNorm norm2;
        private readonly Linear ff1;
        private readonly Linear ff2;

        public readonly int hiddenSize;
        private readonly List<Tensor> memory = new List<Tensor>();

        public TransformerLSTMCell(int inputSize, int hiddenSize)
            : base("transformer_lstm_cell")
        {
            this.hiddenSize = hiddenSize;

            lstm = nn.LSTMCell(inputSize, hiddenSize);

            Wq = nn.Linear(hiddenSize, hiddenSize);
            Wk = nn.Linear(hiddenSize, hiddenSize);
            Wv = nn.Linear(hiddenSize, hiddenSize);
            proj = nn.Linear(hiddenSize, hiddenSize);

            norm1 = nn.LayerNorm(new long[] { hiddenSize });
            norm2 = nn.LayerNorm(new long[] { hiddenSize });

            ff1 = nn.Linear(hiddenSize, hiddenSize * 4);
            ff2 = nn.Linear(hiddenSize * 4, hiddenSize);

            RegisterComponents();
        }

        public (Tensor h, Tensor c) forward(Tensor x, Tensor h, Tensor c)
        {
            // 1. LSTM step
            (h, c) = lstm.forward(x, (h, c));

            // 2. Append to memory (no grad through memory)
            memory.Add(h.detach());

            // 3. Self‑attention over memory
            var M = torch.stack(memory.ToArray(), 0).squeeze(1); // [T,H]

            var Q = Wq.forward(M); // [T,H]
            var K = Wk.forward(M); // [T,H]
            var V = Wv.forward(M); // [T,H]

            var scores = torch.matmul(Q, K.transpose(0, 1)) / System.Math.Sqrt(hiddenSize); // [T,T]
            var weights = scores.softmax(1).detach(); // [T,T]

            var attn = torch.matmul(weights, V); // [T,H]

            // 4. Take last token as current context
            var lastIdx = attn.shape[0] - 1;
            var ctx = attn.select(0, lastIdx).unsqueeze(0); // [1,H]

            // 5. Residual + norm
            var hAttn = norm1.forward(h + proj.forward(ctx));

            // 6. Feed‑forward block
            var ff = ff2.forward(ff1.forward(hAttn).relu());
            var hFinal = norm2.forward(hAttn + ff);

            return (hFinal, c);
        }

        public void ResetMemory()
        {
            memory.Clear();
        }
    }
    public class MultiHeadAttentiveLSTMCell : nn.Module
    {
        private readonly LSTMCell lstm;

        public readonly int hiddenSize;
        private readonly int numHeads;
        private readonly int headDim;

        private readonly Linear Wq;
        private readonly Linear Wk;
        private readonly Linear Wv;
        private readonly Linear fuse;

        private readonly List<Tensor> memory = new List<Tensor>();

        public MultiHeadAttentiveLSTMCell(int inputSize, int hiddenSize, int numHeads)
            : base("multihead_attentive_lstm_cell")
        {
            this.hiddenSize = hiddenSize;
            this.numHeads = numHeads;
            this.headDim = hiddenSize / numHeads;

            lstm = nn.LSTMCell(inputSize, hiddenSize);

            Wq = nn.Linear(hiddenSize, hiddenSize);
            Wk = nn.Linear(hiddenSize, hiddenSize);
            Wv = nn.Linear(hiddenSize, hiddenSize);

            fuse = nn.Linear(hiddenSize * 2, hiddenSize);

            RegisterComponents();
        }

        public (Tensor h, Tensor c) forward(Tensor x, Tensor h, Tensor c)
        {
            // 1. Standard LSTM update
            (h, c) = lstm.forward(x, (h, c));

            // 2. Add hidden state to memory (no grad through memory)
            memory.Add(h.detach());

            if (memory.Count == 0)
                return (h, c);

            // 3. Stack memory: [T, H]
            var M = torch.stack(memory.ToArray(), 0).squeeze(1); // [T,H]

            // 4. Project to Q,K,V: [T,H]
            var Q = Wq.forward(h);   // [1,H]
            var K = Wk.forward(M);   // [T,H]
            var V = Wv.forward(M);   // [T,H]

            // 5. Reshape to multi‑head: [T, numHeads, headDim], [1,numHeads,headDim]
            var Kmh = K.view(M.shape[0], numHeads, headDim); // [T,Hh,D]
            var Vmh = V.view(M.shape[0], numHeads, headDim); // [T,Hh,D]
            var Qmh = Q.view(1, numHeads, headDim);          // [1,Hh,D]

            // 6. Compute attention per head
            // scores: [Hh, T]
            var scores = torch.einsum("thd,bhd->hb t", Kmh, Qmh); // but TorchSharp lacks einsum, so we do manual

            // Manual: reshape to [Hh, T, D]
            var Kperm = Kmh.permute(1, 0, 2); // [Hh,T,D]
            var Qperm = Qmh.permute(1, 0, 2); // [Hh,1,D]

            var attnOutputs = new List<Tensor>();

            for (int head = 0; head < numHeads; head++)
            {
                var K_h = Kperm.index(torch.TensorIndex.Single(head)); // [T,D]
                var Q_h = Qperm.index(torch.TensorIndex.Single(head)); // [1,D]

                var scores_h = torch.matmul(K_h, Q_h.transpose(0, 1)); // [T,1]
                var weights_h = scores_h.softmax(0).detach();          // [T,1]

                // select V for this head: [:, head, :]
                var V_h = Vmh.index(torch.TensorIndex.Colon, torch.TensorIndex.Single(head), torch.TensorIndex.Colon); // [T,D]
                var ctx_h = (weights_h * V_h).sum(new long[] { 0 }).unsqueeze(0); // [1,D]
                attnOutputs.Add(ctx_h);
            }

            // 7. Concatenate heads: [1, H]
            var context = torch.cat(attnOutputs.ToArray(), 1); // [1,H]

            // 8. Fuse LSTM output + context
            var combined = torch.cat(new Tensor[] { h, context }, 1); // [1,2H]
            var hFinal = fuse.forward(combined).tanh();               // [1,H]

            return (hFinal, c);
        }

        public void ResetMemory()
        {
            memory.Clear();
        }
    }
    public class AttentiveLSTM : nn.Module
    {
        private readonly AttentiveLSTMCell cell;
        private readonly int hiddenSize;

        public AttentiveLSTM(int inputSize, int hiddenSize)
            : base("attentive_lstm")
        {
            this.hiddenSize = hiddenSize;
            cell = new AttentiveLSTMCell(inputSize, hiddenSize);
            RegisterComponents();
        }

        public (Tensor h, Tensor c, Tensor[] outputs) forward(Tensor x)
        {
            // x: [T, B, inputSize]
            int T = (int)x.shape[0];
            int B = (int)x.shape[1];

            var h = torch.zeros(B, hiddenSize);
            var c = torch.zeros(B, hiddenSize);

            var outputs = new Tensor[T];

            cell.ResetMemory();

            for (int t = 0; t < T; t++)
            {
                var x_t = x[t]; // [B,inputSize]
                (h, c) = cell.forward(x_t, h, c);
                outputs[t] = h;
            }

            return (h, c, outputs);
        }
    }

    public class BoardEncoder : nn.Module
    {
        private readonly Embedding embed;
        private readonly Linear compress;
        private readonly int hiddenSize;

        public BoardEncoder(int hiddenSize)
            : base("board_encoder")
        {
            this.hiddenSize = hiddenSize;

            // 3 possible cell states: empty, X, O
            embed = nn.Embedding(3, hiddenSize);

            // compress 9*H → H
            compress = nn.Linear(9 * hiddenSize, hiddenSize);

            RegisterComponents();
        }

        public Tensor forward(Cell[] board)
        {
            // convert board to tensor of indices
            long[] idx = new long[9];
            for (int i = 0; i < 9; i++)
                idx[i] = (long)board[i];

            // [1,9]
            var t = torch.tensor(idx, dtype: ScalarType.Int64).unsqueeze(0);

            // [1,9,H]
            var e = embed.forward(t);

            // [1,9H]
            var flat = e.flatten(1);

            // [1,H]
            return compress.forward(flat).tanh();
        }
    }


    public class AttentiveLSTMCell : nn.Module
    {
        private readonly LSTMCell lstm;
        private readonly Linear Wq;
        private readonly Linear Wk;
        private readonly Linear Wv;
        private readonly Linear fuse;

        private readonly List<Tensor> memory = new List<Tensor>();
        private readonly int hiddenSize;

        public AttentiveLSTMCell(int inputSize, int hiddenSize)
            : base("attentive_lstm_cell")
        {
            this.hiddenSize = hiddenSize;

            lstm = nn.LSTMCell(inputSize, hiddenSize);

            Wq = nn.Linear(hiddenSize, hiddenSize);
            Wk = nn.Linear(hiddenSize, hiddenSize);
            Wv = nn.Linear(hiddenSize, hiddenSize);

            fuse = nn.Linear(hiddenSize * 2, hiddenSize);

            RegisterComponents();
        }

        public (Tensor h, Tensor c) forward(Tensor x, Tensor h, Tensor c)
        {
            // 1. Standard LSTM update
            (h, c) = lstm.forward(x, (h, c));

            // 2. Add hidden state to memory
            memory.Add(h.detach());

            // 3. Compute attention over memory
            var q = Wq.forward(h); // [1,H]

            var keys = new List<Tensor>();
            var values = new List<Tensor>();

            foreach (var m in memory)
            {
                keys.Add(Wk.forward(m));
                values.Add(Wv.forward(m));
            }

            var K = torch.stack(keys.ToArray(), 0).squeeze(1);   // [T,H]
            var V = torch.stack(values.ToArray(), 0).squeeze(1); // [T,H]

            var attnScores = torch.matmul(K, q.transpose(0, 1)); // [T,1]
            var attnWeights = attnScores.softmax(0).detach(); // [T,1]

            var context = (attnWeights * V).sum(new long[] { 0 }).unsqueeze(0); // [1,H]

            // 4. Fuse LSTM output with attention context
            var combined = torch.cat(new Tensor[] { h, context }, 1); // [1,2H]
            var hFinal = fuse.forward(combined).tanh();               // [1,H]

            return (hFinal, c);
        }

        public void ResetMemory()
        {
            memory.Clear();
        }
    }

    // ------------------------------------------------------------
    // TOKENIZER
    // ------------------------------------------------------------
    public class SineTokenizer
    {
        private readonly int vocabSize;

        public SineTokenizer(int vocabSize = 500)
        {
            this.vocabSize = vocabSize;
        }

        public int Encode(float value)
        {
            float normalized = (value + 1f) / 2f;
            int token = (int)(normalized * (vocabSize - 1));
            return Math.Clamp(token, 0, vocabSize - 1);
        }

        public float Decode(int token)
        {
            float normalized = token / (float)(vocabSize - 1);
            return normalized * 2f - 1f;
        }
    }
    public class TransformBasis : nn.Module
    {
        public readonly Module[] F;

        public TransformBasis(int size, int count) : base("transform_basis")
        {
            F = new Module[count];

            for (int i = 0; i < count; i++)
                F[i] = nn.Linear(size, size);

            RegisterComponents();
        }

        public Tensor Apply(int index, Tensor x)
        {
            int safe = index % F.Length;
            // Korrigiert: expliziter Cast zu Module<Tensor, Tensor> für forward-Aufruf
            return ((Module<Tensor, Tensor>)F[safe]).forward(x);
        }
    }
    public class CombinatorialPathGate : nn.Module
    {
        private readonly int hiddenSize;
        private readonly int basisCount;

        private readonly Module<Tensor, Tensor>[] basisTransforms;
        private readonly Module<Tensor, Tensor> gateMLP;

        public CombinatorialPathGate(int hiddenSize, int basisCount, int depth)
            : base("combinatorial_path_gate")
        {
            this.hiddenSize = hiddenSize;
            this.basisCount = basisCount;

            basisTransforms = new Module<Tensor, Tensor>[basisCount];

            for (int i = 0; i < basisCount; i++)
            {
                basisTransforms[i] = nn.Sequential(
                    nn.Linear(hiddenSize, hiddenSize),
                    nn.ReLU(),
                    nn.Linear(hiddenSize, hiddenSize)
                );
            }

            gateMLP = nn.Sequential(
                nn.Linear(hiddenSize, hiddenSize),
                nn.ReLU(),
                nn.Linear(hiddenSize, basisCount)
            );

            RegisterComponents();
        }

        public Tensor forward(Tensor x)
        {
            var outputs = new Tensor[basisCount];

            for (int i = 0; i < basisCount; i++)
                outputs[i] = basisTransforms[i].forward(x);

            var scores = gateMLP.forward(x); // [1, basisCount]
            var weights = scores.softmax(1);

            Tensor result = torch.zeros_like(x);

            for (int i = 0; i < basisCount; i++)
                result += weights[0, i] * outputs[i];

            return result;
        }
    }

    public enum Cell { Empty = 0, X = 1, O = 2 }

    public class TicTacToe
    {
        public Cell[] Board = new Cell[9];
        public Cell CurrentPlayer = Cell.X;

        public TicTacToe()
        {
            Reset();
        }

        public void Reset()
        {
            for (int i = 0; i < 9; i++)
                Board[i] = Cell.Empty;

            CurrentPlayer = Cell.X;
        }

        public List<int> GetLegalMoves()
        {
            var moves = new List<int>();
            for (int i = 0; i < 9; i++)
                if (Board[i] == Cell.Empty)
                    moves.Add(i);
            return moves;
        }

        public bool MakeMove(int index)
        {
            if (Board[index] != Cell.Empty)
                return false;

            Board[index] = CurrentPlayer;
            CurrentPlayer = (CurrentPlayer == Cell.X) ? Cell.O : Cell.X;
            return true;
        }

        public Cell CheckWinner()
        {
            int[][] wins = new int[][]
            {
            new[]{0,1,2}, new[]{3,4,5}, new[]{6,7,8}, // rows
            new[]{0,3,6}, new[]{1,4,7}, new[]{2,5,8}, // columns
            new[]{0,4,8}, new[]{2,4,6}                // diagonals
            };

            foreach (var w in wins)
            {
                if (Board[w[0]] != Cell.Empty &&
                    Board[w[0]] == Board[w[1]] &&
                    Board[w[1]] == Board[w[2]])
                    return Board[w[0]];
            }

            // draw
            if (GetLegalMoves().Count == 0)
                return Cell.Empty;

            // game not finished
            return (Cell)(-1);
        }

        public void Print()
        {
            for (int i = 0; i < 9; i++)
            {
                char c = Board[i] switch
                {
                    Cell.X => 'X',
                    Cell.O => 'O',
                    _ => '.'
                };

                Console.Write(c);
                if (i % 3 == 2) Console.WriteLine();
            }
            Console.WriteLine();
        }
    }
    
    


    // ------------------------------------------------------------
    // CLEAN SINE SERIES GENERATOR
    // ------------------------------------------------------------
    public static (int[] tokens, float[] values) GenerateSineSeries(
        int samplesPerWave = 32,
        int numWaves = 20,
        int vocabSize = 100)
    {
        var tokenizer = new SineTokenizer(vocabSize);

        int length = samplesPerWave * numWaves;

        float[] values = new float[length];
        int[] tokens = new int[length];

        for (int i = 0; i < length; i++)
        {
            float phase = (float)i / samplesPerWave;
            float y = MathF.Sin(phase * 2f * MathF.PI);
            values[i] = y;
            tokens[i] = tokenizer.Encode(y);
        }

        return (tokens, values);
    }

    /*public class UnifiedMultiHeadTransformerLSTMCell : nn.Module
    {
        private readonly LSTMCell lstm;

        public readonly int hiddenSize;
        private readonly int numHeads;
        private readonly int headDim;

        private readonly Linear Wq;
        private readonly Linear Wk;
        private readonly Linear Wv;
        private readonly Linear attnProj;
        public readonly Linear output;

        private readonly Linear ff1;
        private readonly Linear ff2;

        private readonly LayerNorm norm1;
        private readonly LayerNorm norm2;

        private readonly List<Tensor> memory = new List<Tensor>();

        public UnifiedMultiHeadTransformerLSTMCell(int inputSize, int hiddenSize, int numHeads, int outputSize = 1)
            : base("unified_multihead_transformer_lstm_cell")
        {
            this.hiddenSize = hiddenSize;
            this.numHeads = numHeads;
            this.headDim = hiddenSize / numHeads;
            this.outputSize = outputSize;

            lstm = nn.LSTMCell(inputSize, hiddenSize);

            Wq = nn.Linear(hiddenSize, hiddenSize);
            Wk = nn.Linear(hiddenSize, hiddenSize);
            Wv = nn.Linear(hiddenSize, hiddenSize);
            attnProj = nn.Linear(hiddenSize, hiddenSize);

            ff1 = nn.Linear(hiddenSize, hiddenSize * 4);
            ff2 = nn.Linear(hiddenSize * 4, hiddenSize);
            output = nn.Linear(hiddenSize, outputSize);

            norm1 = nn.LayerNorm(new long[] { hiddenSize });
            norm2 = nn.LayerNorm(new long[] { hiddenSize });

            RegisterComponents();
        }

        // SINGLE-STEP FORWARD: use this in training
        public (Tensor h, Tensor c) forward_step(Tensor x, Tensor h, Tensor c)
        {
            // 1. LSTM step
            (h, c) = lstm.forward(x, (h, c));

            // 2. Add hidden state to memory (no grad)
            memory.Add(h.detach());

            // 3. Multi‑Head Attention over memory
            var M = torch.stack(memory.ToArray(), 0); // [T,B,H] or [T,H] depending on h shape
            if (M.dim() == 3)
                M = M.squeeze(1); // assume B=1 → [T,H]

            var Q = Wq.forward(h).view(1, numHeads, headDim);                 // [1,Hh,D]
            var K = Wk.forward(M).view(M.shape[0], numHeads, headDim);        // [T,Hh,D]
            var V = Wv.forward(M).view(M.shape[0], numHeads, headDim);        // [T,Hh,D]

            var Kperm = K.permute(1, 0, 2); // [Hh,T,D]
            var Qperm = Q.permute(1, 0, 2); // [Hh,1,D]

            var headContexts = new List<Tensor>();

            for (int head = 0; head < numHeads; head++)
            {
                var K_h = Kperm.index(torch.TensorIndex.Single(head)); // [T,D]
                var Q_h = Qperm.index(torch.TensorIndex.Single(head)); // [1,D]

                var scores = torch.matmul(K_h, Q_h.transpose(0, 1)) / Math.Sqrt(headDim);
                var weights = scores.softmax(0).detach(); // [T,1]

                var V_h = V.index(
                    torch.TensorIndex.Colon,
                    torch.TensorIndex.Single(head),
                    torch.TensorIndex.Colon
                ); // [T,D]

                var ctx_h = (weights * V_h).sum(new long[] { 0 }).unsqueeze(0); // [1,D]

                headContexts.Add(ctx_h);
            }

            var context = torch.cat(headContexts.ToArray(), 1); // [1,H]

            // 4. Residual + LayerNorm
            var h1 = norm1.forward(h + attnProj.forward(context));

            // 5. Feed‑Forward (Transformer)
            var ff = ff2.forward(ff1.forward(h1).relu());

            // 6. Residual + LayerNorm
            var hFinal = norm2.forward(h1 + ff);

            return (hFinal, c);
        }

        // SEQUENCE FORWARD: optional, for [T,B,inputSize]
        public (Tensor h, Tensor c, Tensor[] outputs) forward(Tensor x)
        {
            // x: [T, B, inputSize]
            int T = (int)x.shape[0];
            int B = (int)x.shape[1];

            var h = torch.zeros(B, hiddenSize);
            var c = torch.zeros(B, hiddenSize);

            var outputs = new Tensor[T];

            ResetMemory();

            for (int t = 0; t < T; t++)
            {
                var x_t = x[t]; // [B,inputSize]
                (h, c) = forward_step(x_t, h, c);
                outputs[t] = h;
            }

            return (h, c, outputs);
        }

        public void ResetMemory()
        {
            memory.Clear();
        }
    }*/

    
    }



// ------------------------------------------------------------
// SINGLE-BAND LSTM MODEL
// ------------------------------------------------------------
public class SineLSTMModel : nn.Module
{
    private readonly Module embedding;
    private readonly LSTMCell[] layers;
    private readonly Module outputLayer;

    public readonly int hiddenSize;
    public readonly int vocabSize;
    public readonly int numLayers;

    public SineLSTMModel(int vocabSize, int hiddenSize, int numLayers = 2) : base("sine_lstm")
    {
        this.vocabSize = vocabSize;
        this.hiddenSize = hiddenSize;
        this.numLayers = Math.Max(1, numLayers);

        embedding = nn.Embedding(vocabSize, hiddenSize);

        layers = new LSTMCell[this.numLayers];
        for (int i = 0; i < this.numLayers; i++)
        {
            // all layers have hidden size as both input and output here (embedding dim == hiddenSize)
            layers[i] = nn.LSTMCell(hiddenSize, hiddenSize);
        }

        outputLayer = nn.Linear(hiddenSize, vocabSize);

        // expose layers array for RegisterComponents to pick up
        RegisterComponents();
    }

    // tokenIdx: [1] or [B]
    // h, c: [numLayers, B, hiddenSize]
    public (Tensor logits, Tensor h, Tensor c) forward(Tensor tokenIdx, Tensor h, Tensor c)
    {
        var emb = ((Module<Tensor, Tensor>)embedding).forward(tokenIdx); // [B,hidden]

        int B = (int)emb.shape[0];

        var newHs = new Tensor[this.numLayers];
        var newCs = new Tensor[this.numLayers];

        var input = emb;

        for (int l = 0; l < this.numLayers; l++)
        {
            // extract per-layer hidden/cell [B,hidden]
            var h_l = h.index(torch.TensorIndex.Single(l));
            var c_l = c.index(torch.TensorIndex.Single(l));

            var (hOut, cOut) = layers[l].forward(input, (h_l, c_l));

            newHs[l] = hOut;
            newCs[l] = cOut;

            // next layer input is this layer's output
            input = hOut;
        }

        var hTop = newHs[this.numLayers - 1];
        var logits = ((Module<Tensor, Tensor>)outputLayer).forward(hTop);

        var stackedH = torch.stack(newHs, 0);
        var stackedC = torch.stack(newCs, 0);

        return (logits, stackedH, stackedC);
    }


    // ------------------------------------------------------------
    // TRAINING LOOP
    // ------------------------------------------------------------


    // ------------------------------------------------------------
    // SINGLE-BAND TRAINING
    // ------------------------------------------------------------


    // ------------------------------------------------------------
    // PREDICTION (SINGLE LSTM)
    // ------------------------------------------------------------

    // Save model state_dict into multiple part directories each not exceeding maxBytes (default 2GB)
    public static void SaveModelWeightsLimited(nn.Module model, string basePath, long maxBytes = 2L * 1024 * 1024 * 1024)
    {
        var state = model.state_dict(); // IDictionary<string, Tensor>

        // base directory for parts
        var dirBase = basePath + ".parts";
        if (!Directory.Exists(dirBase)) Directory.CreateDirectory(dirBase);

        int part = 0;
        string currentDir = Path.Combine(dirBase, $"part{part}");
        Directory.CreateDirectory(currentDir);

        foreach (var kv in state)
        {
            var key = kv.Key;
            var tensor = kv.Value;

            var safeName = Uri.EscapeDataString(key) + ".pt"; // encode key into filename
            var fullPath = Path.Combine(currentDir, safeName);

            // save tensor
            torch.save(tensor, fullPath);

            // compute directory size
            long dirSize = 0;
            try { dirSize = Directory.EnumerateFiles(currentDir).Sum(f => new FileInfo(f).Length); } catch { dirSize = 0; }

            if (dirSize > maxBytes)
            {
                var files = Directory.GetFiles(currentDir);
                if (files.Length == 1)
                {
                    // single tensor exceeds limit; leave it (can't split tensor)
                }
                else
                {
                    // move last saved file to new part
                    part++;
                    var newDir = Path.Combine(dirBase, $"part{part}");
                    Directory.CreateDirectory(newDir);
                    var newPath = Path.Combine(newDir, safeName);
                    try { File.Move(fullPath, newPath); } catch { }
                    currentDir = newDir;
                }
            }
        }
    }

    // Load part directories produced by SaveModelWeightsLimited and load into model
    public static void LoadModelWeightsLimited(nn.Module model, string basePath)
    {
        var dirBase = basePath + ".parts";

        var merged = new Dictionary<string, Tensor>();

        if (Directory.Exists(dirBase))
        {
            var partDirs = Directory.GetDirectories(dirBase).OrderBy(d => d);
            foreach (var pd in partDirs)
            {
                var files = Directory.GetFiles(pd).OrderBy(f => f);
                foreach (var f in files)
                {
                    var fname = Path.GetFileNameWithoutExtension(f);
                    var key = Uri.UnescapeDataString(fname);
                    var t = torch.load(f) as Tensor;
                    if (t is not null) merged[key] = t;
                }
            }
        }
        else
        {
            // fallback: single file state dict
            var single = basePath + ".pt";
            if (File.Exists(single))
            {
                var dict = torch.load(single) as IDictionary<string, Tensor>;
                if (dict != null)
                {
                    foreach (var kv in dict) merged[kv.Key] = kv.Value;
                }
            }
        }

        if (merged.Count > 0)
        {
            model.load_state_dict(merged, strict: false);
        }
    }


    public class WordTokenizer
    {
        private Dictionary<string, int> stoi = new Dictionary<string, int>();
        private Dictionary<int, string> itos = new Dictionary<int, string>();
        public const string UnknownToken = "unknown";

        public WordTokenizer()
        {
            AddWord(UnknownToken);
            AddWord("<pad>");
            AddWord("<eos>");
        }

        // NEU: Initialisiert die physikalische Grammatik
        public void InitializePhysicsVocab()
        {
            string[] physicsVocab = { "[", "]", "peak", "high", "mid", "low", "flux", "horizon", "chirp" };
            foreach (var word in physicsVocab) AddWord(word);
        }

        // NEU: Wandelt CSV-Zahlen (Floats) in Token-IDs um
        public int[] TokenizePhysics(double[] data, double threshold)
        {
            var ids = new List<int>();
            foreach (var val in data)
            {
                string token;
                double absVal = Math.Abs(val);

                // Quantisierung: Zahlen -> Logik-Zustände
                if (absVal > threshold * 3.0) token = "peak";
                else if (absVal > threshold * 2.0) token = "high";
                else if (absVal > threshold * 1.0) token = "mid";
                else token = "low";

                ids.Add(Encode(token));
            }
            return ids.ToArray();
        }

        public int GetVocabSize() => itos.Count;

        public int AddWord(string word)
        {
            var w = word.ToLowerInvariant();
            if (stoi.TryGetValue(w, out var id)) return id;
            int newId = itos.Count;
            stoi[w] = newId;
            itos[newId] = w;
            return newId;
        }

        public int Encode(string word)
        {
            if (string.IsNullOrEmpty(word)) return 0;
            var w = word.ToLowerInvariant();
            return stoi.TryGetValue(w, out var id) ? id : 0;
        }

        public string Decode(int token)
        {
            return itos.TryGetValue(token, out var s) ? s : UnknownToken;
        }

        public int[] Tokenize(string text)
        {
            if (string.IsNullOrWhiteSpace(text)) return Array.Empty<int>();
            var parts = text.Split(new[] { ' ', '\t', '\n', '\r', ',', '.' }, StringSplitOptions.RemoveEmptyEntries);
            return parts.Select(Encode).ToArray();
        }

        public string Detokenize(IEnumerable<int> tokens) => string.Join(' ', tokens.Select(Decode));




        // Build from corpus of texts; simple whitespace split and lowercasing
        public void BuildVocabulary(IEnumerable<string> texts, int maxVocab = 10000)
        {
            var freq = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

            foreach (var t in texts)
            {
                if (string.IsNullOrWhiteSpace(t)) continue;
                var parts = t.Split(new[] { ' ', '\t', '\n', '\r', ',', '.', '!', '?', ';', ':', '"', '\'' }, StringSplitOptions.RemoveEmptyEntries);
                foreach (var p in parts)
                {
                    var w = p.Trim().ToLowerInvariant();
                    if (w.Length == 0) continue;
                    if (!freq.TryGetValue(w, out var c)) c = 0;
                    freq[w] = c + 1;
                }
            }

            var ordered = freq.OrderByDescending(kv => kv.Value).ThenBy(kv => kv.Key).Take(maxVocab);

            foreach (var kv in ordered)
            {
                if (!stoi.ContainsKey(kv.Key))
                {
                    stoi[kv.Key] = itos.Count - 1;
                }
            }
        }


    }
}
