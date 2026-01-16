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
    // 1. Define the IDs strictly
    private const int ID_A = 0;
    private const int ID_C = 1;
    private const int ID_G = 2;
    private const int ID_T = 3;

    // 2. Map characters to those IDs
    private static readonly Dictionary<char, long> BaseToId = new Dictionary<char, long> {
        {'A', ID_A}, {'C', ID_C}, {'G', ID_G}, {'T', ID_T},
        {'a', ID_A}, {'c', ID_C}, {'g', ID_G}, {'t', ID_T}
    };

    // 3. Map IDs back to characters in the EXACT same order
    private static readonly char[] IndexToBase = { 'A', 'C', 'G', 'T' };

    public static long[] Tokenize(string sequence) =>
        sequence.Select(c => BaseToId.GetValueOrDefault(char.ToUpper(c), 0L)).ToArray();

    public static char Detokenize(int index)
    {
        if (index < 0 || index >= IndexToBase.Length) return 'N';
        return IndexToBase[index];
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

        // Retrospektiver Speicher (Key: DNA-Position, Value: Korrekte Token-ID)
        var retrospectiveFacts = new Dictionary<long, long>();
        var rnd = new Random();

        var allParams = predictor.named_parameters().ToList();
        var mainParams = allParams.Where(p => !p.name.Contains("bias")).Select(p => p.parameter)
                                   .Concat(tokenEmbedding.parameters()).ToList();
        var biasParams = allParams.Where(p => p.name.Contains("bias")).Select(p => p.parameter).ToList();

        var opt = torch.optim.Adam(mainParams, lr: 0.0001f);
        var weights = torch.tensor(new float[] { 1.0f, 1.0f, 1.0f, 1.0f}).to(device);

        Console.WriteLine($"Starte Training mit Retrospective Memory Loop...");
        var h = torch.zeros(new long[] { 1, predictor.hiddenSize }).to(device);
        var c = torch.zeros(new long[] { 1, predictor.hiddenSize }).to(device);
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int i = 0; i < tokens.Length - windowSize - 1; i += windowSize)
            {
                Console.WriteLine("{0}/{1}",i, tokens.Length - windowSize - 1);
                predictor.train();
                opt.zero_grad();
                h = h.detach();
                c = c.detach();
                // --- 1. Normaler Vorwärts-Pass ---

                Tensor lastHidden = null;

                for (int t = 0; t < windowSize; t++)
                {
                    var inputToken = torch.tensor(new long[] { tokens[i + t] }).to(device);
                    var (outH, hNext, cNext) = predictor.forward(tokenEmbedding.forward(inputToken), h, c);
                    h = hNext; c = cNext; lastHidden = outH;
                }

                // Target und Logits
                long targetTokenId = tokens[i + windowSize];
                var targetTensor = torch.tensor(new long[] { targetTokenId }).to(device);
                // Ensure hidden has shape [batch, features]
                var hiddenFlatForHead = lastHidden.flatten(1);
                var logits = predictor.outputHead.forward(hiddenFlatForHead);

                // Loss & Backprop: use built-in cross_entropy with label smoothing
                var loss = torch.nn.functional.cross_entropy(logits, targetTensor, weight: weights, label_smoothing: 0.1f);

                loss.backward();
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0);

                // Bias Scaling
                if (biasParams.Count > 0)
                    foreach (var p in biasParams) if (p.grad is not null) p.grad.mul_(1e-3f);

                opt.step();

                // --- 2. Retrospective Memory Logic ---
                if (loss.item<float>() < 0.3f)
                    retrospectiveFacts[i + windowSize] = targetTokenId;

                // Alle 10 Schritte: Ein altes Wissen "auffrischen"
                if (i % 10 == 0 && retrospectiveFacts.Count > 0)
                {
                    var fact = retrospectiveFacts.ElementAt(rnd.Next(retrospectiveFacts.Count));
                    // Wir nutzen hier einen verkleinerten Schritt für die Erinnerung
                    //PerformMemoryReplay(predictor, tokenEmbedding, opt, tokens, (int)fact.Key, windowSize, weights, device);
                }

                    long predId = logits.argmax(1).item<long>();
                    char predChar = GenomicEngine.Detokenize((int)predId);
                    char expChar = GenomicEngine.Detokenize((int)targetTokenId);


                if (predChar == expChar)
                {
                    Console.ForegroundColor = ConsoleColor.Green;
                }
                else
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                }
                Console.WriteLine($"Ep {epoch} | Pos {i:N0} | Loss: {loss.item<float>():F5} | Memory: {retrospectiveFacts.Count} | Exp: {expChar} | Got: {predChar}");

                // Wichtig: Farbe wieder zurücksetzen für den Rest der Konsole
                Console.ResetColor();


                RunSimplePrediction(predictor, tokenEmbedding, "AAGCCCAATAAACCAC");
                RunSimplePrediction(predictor, tokenEmbedding, "ACTGGCCGAATAGGGA");
                RunSimplePrediction(predictor, tokenEmbedding, "GGCAACGACATGTGCG");
                RunSimplePrediction(predictor, tokenEmbedding, "CCCTTGCGACAGTGAC");

                RunSimplePrediction(predictor, tokenEmbedding, "TCGCCGTTGCCTAAAC");
                RunSimplePrediction(predictor, tokenEmbedding, "TTGAAGGAGTCTAGCA");
                RunSimplePrediction(predictor, tokenEmbedding, "TCCGTGTTACCAGACC");
                RunSimplePrediction(predictor, tokenEmbedding, "AAGACGTCCTCTTCAA");
                RunSimplePrediction(predictor, tokenEmbedding, "TAAATGACCCTCTCGT");
                RunSimplePrediction(predictor, tokenEmbedding, "AAACCTTTCTACTATG");
                RunSimplePrediction(predictor, tokenEmbedding, "AATGGCGCGTCGTGAA");
                RunSimplePrediction(predictor, tokenEmbedding, "GCGACGGCTGAGACGA");
                RunSimplePrediction(predictor, tokenEmbedding, "CGCGTGAATGAAGCGC");
                RunSimplePrediction(predictor, tokenEmbedding, "ACAGCTCAGGAGCCAG");
                RunSimplePrediction(predictor, tokenEmbedding, "CTACGTCGCATATCCT");
                RunSimplePrediction(predictor, tokenEmbedding, "ACTGGAGGTGAAGCGA");
                RunSimplePrediction(predictor, tokenEmbedding, "TATCGATACGTAGGAG");
                RunSimplePrediction(predictor, tokenEmbedding, "GCCTTCGTAGGCTGTT");


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

    Console.WriteLine($"\n--- Fractal Detailed Analysis: {testSeq.ToUpper()} ---");
    Console.WriteLine("------------------------------------------------------------------------------------------------------");
    // Columns: Base, ID, Pred, A(%), C(%), G(%), T(%), Sensitivity
    Console.WriteLine($"{"Base",-4} | {"ID",-2} | {"Pred",-4} | {"A (%)",-6} | {"C (%)",-6} | {"G (%)",-6} | {"T (%)",-6} | {"Sens",-8}");
    Console.WriteLine("------------------------------------------------------------------------------------------------------");

    using (var noGrad = torch.no_grad())
    {
        for (int i = 0; i < tokens.Length; i++)
        {
            var inputToken = torch.tensor(new long[] { tokens[i] }).to(device);
            var (hState, hNext, cNext) = predictor.forward(tokenEmbedding.forward(inputToken), h, c);

            using (var logits = predictor.outputHead.forward(hState))
            {
                var probs = torch.nn.functional.softmax(logits, dim: 1);
                int predIndex = (int)probs.argmax(1).item<long>(); // ArgMax on probabilities
                char predChar = GenomicEngine.Detokenize(predIndex);
                char actualChar = testSeq[i];

                // PRINTING LOGIC
                SetConsoleColorByBase(actualChar);
                Console.Write($"{actualChar,-4}");
                Console.ResetColor();
                Console.Write($" | {tokens[i],-2} | ");

                // Color green if ArgMax actually matches the input
                if (predChar == actualChar) Console.ForegroundColor = ConsoleColor.Green;
                else SetConsoleColorByBase(predChar);

                Console.Write($"{predChar,-4}");
                Console.ResetColor();
                Console.Write(" | ");

                // Ensure headers match these indices: 0=A, 1=C, 2=G, 3=T
                float pA = probs[0, 0].item<float>() * 100f;
                float pC = probs[0, 1].item<float>() * 100f;
                float pG = probs[0, 2].item<float>() * 100f;
                float pT = probs[0, 3].item<float>() * 100f;

                // Sensitivity proxy: change in hidden state magnitude
                float sensitivity = 0f;
                try { sensitivity = (hNext - h).norm().item<float>(); } catch { }

                Console.WriteLine($"{pA,6:F1} | {pC,6:F1} | {pG,6:F1} | {pT,6:F1} | {sensitivity,8:E4}");
            }
            h = hNext; c = cNext;
        }
    }
    Console.WriteLine("------------------------------------------------------------------------------------------------------");
}

// Hilfsmethode für Genetik-Farbstandards
private static void SetConsoleColorByBase(char c)
{
    switch (char.ToUpper(c))
    {
        case 'A': Console.ForegroundColor = ConsoleColor.Red; break;    // Adenin: Oft Rot
        case 'C': Console.ForegroundColor = ConsoleColor.Blue; break;   // Cytosin: Blau
        case 'G': Console.ForegroundColor = ConsoleColor.Yellow; break; // Guanin: Gelb
        case 'T': Console.ForegroundColor = ConsoleColor.Cyan; break;   // Thymin: Cyan/Grün
        default: Console.ForegroundColor = ConsoleColor.Gray; break;    // N: Grau
    }
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
