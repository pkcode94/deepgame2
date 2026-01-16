using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

public static class GenomicEngine
{
    // Mapping: A=1, C=2, G=3, T=4, N=0
    private static readonly Dictionary<char, long> BaseToId = new Dictionary<char, long> {
        {'A', 1}, {'C', 2}, {'G', 3}, {'T', 4}, {'N', 0},
        {'a', 1}, {'c', 2}, {'g', 3}, {'t', 4}, {'n', 0}
    };

    // Lädt eine FASTA Datei (ignoriert Header-Zeilen mit '>')
    public static string LoadFasta(string path)
    {
        return string.Concat(File.ReadLines(path)
            .Where(line => !line.StartsWith(">"))
            .Select(line => line.Trim()));
    }

    public static long[] Tokenize(string sequence) =>
        sequence.Select(c => BaseToId.GetValueOrDefault(c, 0L)).ToArray();

    public static string Detokenize(long[] tokens)
    {
        var idToBase = BaseToId.GroupBy(x => x.Value).ToDictionary(x => x.Key, x => x.First().Key);
        return new string(tokens.Select(t => idToBase.GetValueOrDefault(t, 'N')).ToArray());
    }
    public static void TrainOnGenomics(
    FractalOpponent predictor,
    Embedding tokenEmbedding,
    string dnaSequence,
    int epochs = 100,
    int windowSize = 16)
    {
        var device = torch.CPU;
        var tokens = GenomicEngine.Tokenize(dnaSequence);
        char[] baseMap = { 'N', 'A', 'C', 'G', 'T' };

        // Parameter-Trennung
        var allParams = predictor.named_parameters().ToList();
        var biasParams = allParams.Where(p => p.name.Contains("bias")).Select(p => p.parameter).ToList();
        var mainParams = allParams.Where(p => !p.name.Contains("bias")).Select(p => p.parameter)
                                   .Concat(tokenEmbedding.parameters()).ToList();

        float mainLr = 0.001f;
        float biasLr = 1e-6f;
        float biasGradScale = biasLr / mainLr;

        var opt = torch.optim.Adam(mainParams, lr: mainLr, weight_decay: 1e-5f);

        Console.WriteLine($"Starte Training auf {tokens.Length} Basenpaaren...");

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int i = 0; i < tokens.Length - windowSize - 1; i += windowSize)
            {
                opt.zero_grad();

                var hPred = torch.zeros(new long[] { 1, predictor.hiddenSize }).to(device);
                var cPred = torch.zeros(new long[] { 1, predictor.hiddenSize }).to(device);

                // Input für Anzeige speichern
                string inputContext = dnaSequence.Substring(i, windowSize);
                long targetId = tokens[i + windowSize];
                char expectedBase = baseMap[targetId];

                Tensor lastOutput = null;
                for (int t = 0; t < windowSize; t++)
                {
                    var inputToken = torch.tensor(new long[] { tokens[i + t] }).to(device);
                    var emb = tokenEmbedding.forward(inputToken);
                    var (outPred, hNext, cNext) = predictor.forward(emb, hPred, cPred);
                    hPred = hNext;
                    cPred = cNext;
                    lastOutput = outPred;
                }

                var targetEmb = tokenEmbedding.forward(torch.tensor(new long[] { targetId }).to(device)).detach();
                var loss = torch.nn.functional.huber_loss(lastOutput, targetEmb);

                loss.backward();

                // --- BIAS GRADIENT SCALING PATCH ---
                if (biasParams.Count > 0)
                {
                    foreach (var p in biasParams)
                    {
                        if (p.grad is not null) p.grad.mul_(biasGradScale);
                    }
                }

                // --- PREDICTION DETERMINATION ---
                char predictedBase = 'N';
                float maxSim = -1f;
                for (int b = 1; b <= 4; b++) // A, C, G, T prüfen
                {
                    var bEmb = tokenEmbedding.forward(torch.tensor(new long[] { b }).to(device)).detach();
                    float sim = torch.nn.functional.cosine_similarity(lastOutput, bEmb).item<float>();
                    if (sim > maxSim) { maxSim = sim; predictedBase = baseMap[b]; }
                }

                // --- COLORED OUTPUT ---
                Console.Write($"Ep {epoch} | In: {inputContext} | Exp: {expectedBase} | Got: ");
                if (predictedBase == expectedBase) Console.ForegroundColor = ConsoleColor.Green;
                else Console.ForegroundColor = ConsoleColor.Red;
                Console.Write(predictedBase);
                Console.ResetColor();
                Console.WriteLine($" | Loss: {loss.item<float>():F6}");

                torch.nn.utils.clip_grad_norm_(predictor.parameters(), 0.1);
                opt.step();
            }
        }
    }
    public static void RunSimplePrediction(FractalOpponent predictor, Embedding tokenEmbedding, string testSeq)
    {
        var device = torch.CPU;
        predictor.eval();

        long[] tokens = GenomicEngine.Tokenize(testSeq);
        var h = torch.zeros(new long[] { 1, predictor.hiddenSize }).to(device);
        var c = torch.zeros(new long[] { 1, predictor.hiddenSize }).to(device);

        Console.WriteLine("\n--- Predictive Logic Analysis ---");
        Console.WriteLine($"Sequenz: {testSeq.ToUpper()}");
        Console.WriteLine("---------------------------------------------------------");
        Console.WriteLine($"{"Base",-5} | {"ID",-3} | {"Hidden Norm",-12} | {"Sensitivity",-12}");
        Console.WriteLine("---------------------------------------------------------");

        using (var noGrad = torch.no_grad())
        {
            for (int i = 0; i < tokens.Length; i++)
            {
                char baseChar = testSeq[i];
                long tokenId = tokens[i];

                var idx = torch.tensor(new long[] { tokenId }).to(device);
                var emb = tokenEmbedding.forward(idx);          // Embedding.forward verfügbar durch Embedding-Typ
                var (outPred, hNext, cNext) = predictor.forward(emb, h, c);

                float hNorm = hNext.norm().item<float>();
                float sensitivity = (hNext - h).norm().item<float>();

                Console.WriteLine($"{baseChar,-5} | {tokenId,-3} | {hNorm,-12:F4} | {sensitivity,-12:E4}");
                h = hNext;
                c = cNext;
            }
        }
        Console.WriteLine("---------------------------------------------------------");
    }
    public static void RunMutationStressTest(
    FractalOpponent predictor,
    Embedding tokenEmbedding,
    string healthySeq,
    int mutationPos,
    char newBase)
    {
        var device = torch.CPU;
        char[] mutated = healthySeq.ToCharArray();
        mutated[mutationPos] = newBase;
        string mutatedSeq = new string(mutated);

        long[] healthyTokens = GenomicEngine.Tokenize(healthySeq);
        long[] mutatedTokens = GenomicEngine.Tokenize(mutatedSeq);

        Console.WriteLine($"--- Mutation Stress Test (Pos: {mutationPos}, {healthySeq[mutationPos]} -> {newBase}) ---");

        var h = torch.zeros(new long[] { 1, predictor.hiddenSize });
        var c = torch.zeros(new long[] { 1, predictor.hiddenSize });

        for (int t = 0; t < healthyTokens.Length; t++)
        {
            var embH = tokenEmbedding.forward(torch.tensor(new long[] { healthyTokens[t] }));
            var embM = tokenEmbedding.forward(torch.tensor(new long[] { mutatedTokens[t] }));

            // Butterfly-Gate Sensitivität messen (falls dein Modell die Eigenschaft 'lastSensitivity' hat)
            // Hier simulieren wir den Impact auf den Hidden State
            var (_, hNextH, _) = predictor.forward(embH, h, c);
            var (_, hNextM, _) = predictor.forward(embM, h, c);

            float impact = (hNextH - hNextM).norm().item<float>();

            if (t == mutationPos)
                Console.WriteLine($"[MUTATION REACHED] Impact Score: {impact:E4}");

            h = hNextH; // Wir folgen dem gesunden Pfad weiter
        }
    }
    public static string LoadAndCleanSequence(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Die Datei {filePath} wurde nicht gefunden.");

        // Alle Zeilen lesen
        var lines = File.ReadAllLines(filePath);

        // Falls es eine FASTA-Datei ist, die erste Zeile (Header) entfernen
        var sequenceLines = lines.Where(l => !l.Trim().StartsWith(">"));

        // Alles zusammenfügen und alle Whitespaces (Leerzeichen, \n, \r, \t) entfernen
        string fullSequence = string.Concat(sequenceLines).Replace(" ", "");
        fullSequence = Regex.Replace(fullSequence, @"\s+", "");

        Console.WriteLine($"Sequenz geladen. Länge: {fullSequence.Length} Basenpaare.");
        return fullSequence.ToUpper(); // Sicherstellen, dass alles Großbuchstaben sind
    }


}
