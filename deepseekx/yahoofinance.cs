using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Net.Http;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace deepseekx
{
    // Lightweight numeric tokenizer that maps continuous values to nearest bin indices
    // Compute ideal action for each timestep using future return over horizon H
    // -1 = sell, 0 = hold, +1 = buy  (we'll remap to {0,1,2} for CE)
    
    public class NumericTokenizer
    {
        private readonly double[] bins;
        public int VocabSize => bins.Length;
        public NumericTokenizer(double[] bins)
        {
            this.bins = bins ?? throw new ArgumentNullException(nameof(bins));
        }

        public int Encode(double value)
        {
            // find nearest bin (bins assumed sorted ascending)
            int best = 0;
            double bestDiff = Math.Abs(value - bins[0]);
            for (int i = 1; i < bins.Length; i++)
            {
                var d = Math.Abs(value - bins[i]);
                if (d < bestDiff) { bestDiff = d; best = i; }
            }
            return best;
        }

        public double Decode(int token)
        {
            if (token < 0) token = 0;
            if (token >= bins.Length) token = bins.Length - 1;
            return bins[token];
        }
    }

    public static class YahooFinancePredictor
    {
        public static int[] ComputeActionLabels(double[] normalizedCloses, int horizon = 3, double epsilon = 0.002)
        {
            int n = normalizedCloses.Length;
            var labels = new int[n];

            for (int t = 0; t < n; t++)
            {
                int futureIdx = t + horizon;
                if (futureIdx >= n)
                {
                    labels[t] = 1; // default to hold near the end
                    continue;
                }

                double pNow = normalizedCloses[t];
                double pFuture = normalizedCloses[futureIdx];
                double r = (pFuture - pNow) / (pNow == 0 ? 1.0 : pNow);

                if (r > epsilon)
                    labels[t] = 2; // buy
                else if (r < -epsilon)
                    labels[t] = 0; // sell
                else
                    labels[t] = 1; // hold
            }

            return labels;
        }
        // Attempt to fetch historical close prices using YahooFinanceApi if available
        public static List<double> FetchClosePrices(string symbol, DateTime from, DateTime to)
        {
            // 1) Try to find a Yahoo finance API via reflection and invoke any suitable GetHistorical method
            try
            {
                Console.WriteLine("Trying to fetch historical data via reflection from loaded assemblies...");
                var assemblies = AppDomain.CurrentDomain.GetAssemblies();
                foreach (var asm in assemblies)
                {
                    Type[] types = null;
                    try { types = asm.GetTypes(); } catch { continue; }
                    foreach (var t in types)
                    {
                        var tname = t.FullName ?? t.Name;
                        if (!tname.ToLowerInvariant().Contains("yahoo")) continue;

                        // enumerate candidate methods
                        var methods = t.GetMethods(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Instance);
                        foreach (var m in methods)
                        {
                            var mname = m.Name.ToLowerInvariant();
                            if (!(mname.Contains("histor") || mname.Contains("history"))) continue;

                            Console.WriteLine($"Found candidate method: {t.FullName}.{m.Name}");

                            var pars = m.GetParameters();
                            var args = new List<object?>();
                            bool needFrom = true, needTo = true;
                            bool skip = false;

                            foreach (var p in pars)
                            {
                                var pt = p.ParameterType;
                                if (pt == typeof(string))
                                {
                                    // assume symbol or interval string
                                    // if parameter name contains "symbol" use symbol, otherwise try common interval keywords
                                    var pname = p.Name.ToLowerInvariant();
                                    if (pname.Contains("symbol") || args.Count == 0)
                                        args.Add(symbol);
                                    else
                                        args.Add("1d");
                                }
                                else if (pt == typeof(DateTime))
                                {
                                    if (needFrom) { args.Add(from); needFrom = false; }
                                    else if (needTo) { args.Add(to); needTo = false; }
                                    else args.Add(to);
                                }
                                else if (pt == typeof(DateTimeOffset))
                                {
                                    if (needFrom) { args.Add(new DateTimeOffset(from)); needFrom = false; }
                                    else if (needTo) { args.Add(new DateTimeOffset(to)); needTo = false; }
                                    else args.Add(new DateTimeOffset(to));
                                }
                                else if (pt.IsEnum)
                                {
                                    // pick first enum value
                                    var ev = Enum.GetValues(pt).GetValue(0);
                                    args.Add(ev);
                                }
                                else if (pt == typeof(int))
                                {
                                    args.Add(1);
                                }
                                else
                                {
                                    // unknown parameter type — attempt default
                                    try { args.Add(Type.Missing); } catch { skip = true; break; }
                                }
                            }

                            if (skip) continue;

                            try
                            {
                                object? instance = null;
                                if (!m.IsStatic)
                                {
                                    try { instance = Activator.CreateInstance(t); } catch { instance = null; }
                                    if (instance == null) continue;
                                }

                                var res = m.Invoke(instance, args.ToArray());

                                // If method returned a Task, wait and get Result
                                if (res is System.Threading.Tasks.Task task)
                                {
                                    task.GetAwaiter().GetResult();
                                    var resultProp = task.GetType().GetProperty("Result");
                                    if (resultProp != null)
                                    {
                                        var hist = resultProp.GetValue(task) as System.Collections.IEnumerable;
                                        var closes = ExtractClosesFromEnumerable(hist);
                                        if (closes.Count > 0) return closes;
                                    }
                                    else
                                    {
                                        // Task without result
                                        continue;
                                    }
                                }
                                else if (res is System.Collections.IEnumerable en)
                                {
                                    var closes = ExtractClosesFromEnumerable(en);
                                    if (closes.Count > 0) return closes;
                                }
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"Invocation failed for {t.FullName}.{m.Name}: {ex.Message}");
                                continue;
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Reflection fetch attempt failed: " + ex.Message);
            }

            // 2) Fallback: try to read common local CSV files
            try
            {
                var candidates = new[] { $"{symbol}.csv", $"{symbol}_history.csv", $"yahoo_{symbol}.csv", $"{symbol.ToUpperInvariant()}.csv" };
                foreach (var f in candidates)
                {
                    if (File.Exists(f))
                    {
                        Console.WriteLine($"Found local CSV fallback: {f}");
                        var closes = ParseCloseFromCsv(f);
                        if (closes.Count > 0) return closes;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("CSV fallback failed: " + ex.Message);
            }

            // 3) HTTP fallback: try Stooq free CSV service (no API key required)
            try
            {
                Console.WriteLine("Attempting to fetch data from Stooq.com...");
                using (var client = new HttpClient())
                {
                    var candidates = new[] { symbol.ToLowerInvariant(), symbol.ToLowerInvariant() + ".us" };
                    foreach (var sym in candidates)
                    {
                        var url = $"https://stooq.com/q/d/l/?s={sym}&d1={from:yyyyMMdd}&d2={to:yyyyMMdd}&i=d";
                        try
                        {
                            var resp = client.GetStringAsync(url).GetAwaiter().GetResult();
                            if (string.IsNullOrWhiteSpace(resp)) continue;
                            var lines = resp.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
                            if (lines.Length <= 1) continue;

                            var header = lines[0].Split(new[] { ',', ';' });
                            int closeIdx = -1;
                            for (int i = 0; i < header.Length; i++)
                            {
                                var h = header[i].Trim().ToLowerInvariant();
                                if (h.Contains("close") || h.Contains("adj close") || h.Contains("adj_close")) { closeIdx = i; break; }
                            }

                            int start = 1;
                            if (closeIdx == -1)
                            {
                                closeIdx = 4;
                                start = 0;
                            }

                            var closes = new List<double>();
                            for (int i = start; i < lines.Length; i++)
                            {
                                var ln = lines[i];
                                if (string.IsNullOrWhiteSpace(ln)) continue;
                                var parts = ln.Split(new[] { ',', ';' });
                                if (parts.Length <= closeIdx) continue;
                                var cs = parts[closeIdx].Trim().Trim('"');
                                if (double.TryParse(cs, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out var v)) closes.Add(v);
                            }

                            if (closes.Count > 0) return closes;
                        }
                        catch (Exception) { continue; }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("HTTP fetch failed: " + ex.Message);
            }

            // 4) Final fallback: generate synthetic random-walk data for the requested date range
            Console.WriteLine("Fetch failed — generating synthetic random-walk data as fallback.");
            var synthetic = GenerateSyntheticPrices(from, to);
            return synthetic;
        }

        private static List<double> ExtractClosesFromEnumerable(System.Collections.IEnumerable? hist)
        {
            var closes = new List<double>();
            if (hist == null) return closes;
            foreach (var item in hist)
            {
                if (item == null) continue;
                // check for Close, AdjClose, ClosePrice properties
                var props = item.GetType().GetProperties();
                object? val = null;
                foreach (var name in new[] { "Close", "AdjClose", "ClosePrice", "close" })
                {
                    var p = props.FirstOrDefault(pp => string.Equals(pp.Name, name, StringComparison.OrdinalIgnoreCase));
                    if (p != null) { val = p.GetValue(item); break; }
                }
                if (val == null)
                {
                    // maybe item is a primitive numeric
                    if (item is double d) { closes.Add(d); continue; }
                    if (item is decimal dec) { closes.Add((double)dec); continue; }
                    continue;
                }

                if (val is double dd) closes.Add(dd);
                else if (val is decimal decc) closes.Add((double)decc);
                else if (val is float f) closes.Add((double)f);
                else if (val is string s && double.TryParse(s, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out var parsed)) closes.Add(parsed);
            }
            return closes;
        }

        private static List<double> ParseCloseFromCsv(string path)
        {
            var closes = new List<double>();
            var lines = File.ReadAllLines(path);
            if (lines.Length == 0) return closes;

            // Try to detect header and index of Close column
            var header = lines[0].Split(new[] { ',', ';' });
            int closeIdx = -1;
            for (int i = 0; i < header.Length; i++)
            {
                var h = header[i].Trim().ToLowerInvariant();
                if (h.Contains("close") || h.Contains("adj close") || h.Contains("adj_close")) { closeIdx = i; break; }
            }

            int start = 0;
            if (closeIdx == -1)
            {
                // maybe no header; try to assume Close is 4th column (Date,Open,High,Low,Close,...)
                closeIdx = 4;
                start = 0;
            }
            else
            {
                start = 1; // skip header
            }

            for (int i = start; i < lines.Length; i++)
            {
                var ln = lines[i];
                if (string.IsNullOrWhiteSpace(ln)) continue;
                var parts = ln.Split(new[] { ',', ';' });
                if (parts.Length <= closeIdx) continue;
                var cs = parts[closeIdx].Trim().Trim('"');
                if (double.TryParse(cs, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out var v)) closes.Add(v);
            }

            return closes;
        }

        private static List<double> GenerateSyntheticPrices(DateTime from, DateTime to)
        {
            var closes = new List<double>();
            if (to <= from) { closes.Add(1.0); return closes; }
            var days = (int)(to - from).TotalDays;
            var rnd = new Random(1234);
            double price = 100.0;
            for (int i = 0; i <= days; i++)
            {
                // small random walk
                var change = (rnd.NextDouble() - 0.5) * 0.02; // +/-1%
                price = Math.Max(0.01, price * (1.0 + change));
                closes.Add(price);
            }
            return closes;
        }

        // Main routine: train on first half, predict on second half
        public static void Run(string symbol, DateTime from, DateTime to,
          int vocabBins = 101, int hiddenSize = 64, int agentCount = 4, int maxDepth = 2,
          int epochs = 6, int orderK = 1)
        {
            Console.WriteLine($"Fetching data for {symbol} {from:yyyy-MM-dd}..{to:yyyy-MM-dd}...");
            var closes = FetchClosePrices(symbol, from, to);
            if (closes == null || closes.Count < 20)
            {
                Console.WriteLine("Not enough data fetched. Aborting.");
                return;
            }

            string[] actionNames = { "SELL", "HOLD", "BUY" };

            // Normalize to 0..1
            double min = closes.Min();
            double max = closes.Max();
            var norm = closes.Select(v => (max == min) ? 0.0 : (v - min) / (max - min)).ToArray();

            // Compute ideal actions from future returns
            var actionLabels = ComputeActionLabels(norm, horizon: 3, epsilon: 0.002);

            // Build uniform bins 0..1 and tokenizer
            var bins = new double[vocabBins];
            for (int i = 0; i < vocabBins; i++)
                bins[i] = Math.Round(i / (double)(vocabBins - 1), 6);
            var tokenizer = new NumericTokenizer(bins);

            // Quantize prices to tokens
            var tokens = norm.Select(v => tokenizer.Encode(v)).ToArray();

            int split = tokens.Length / 2;
            var trainToks = tokens.Take(split).ToArray();
            var testToks = tokens.Skip(split).ToArray();
            var trainActs = actionLabels.Take(split).ToArray();
            var testActs = actionLabels.Skip(split).ToArray();

            Console.WriteLine($"Data points: {tokens.Length}, train={trainToks.Length}, test={testToks.Length}, vocab={tokenizer.VocabSize}");

            // Model: embedding + multiagent fractal core + 3-class action head
            var embedding = nn.Embedding(tokenizer.VocabSize, hiddenSize);
            var core = new MultiAgentFractalCore(hiddenSize, maxDepth, agentCount);
            var output = nn.Linear(hiddenSize, 3); // SELL, HOLD, BUY

            // Parameters
            var paramList = core.parameters()
                .Concat(((Module)embedding).parameters())
                .Concat(((Module)output).parameters())
                .ToArray();
            var opt = torch.optim.Adam(paramList, lr: 1e-3);

            var criterion = nn.functional.cross_entropy;

            int windowSize = 32;

            // --------------------
            // Training loop
            // --------------------
            Console.WriteLine("Starting training...");
            for (int ep = 1; ep <= epochs; ep++)
            {
                float totalLoss = 0f;
                int steps = 0;
                core.train();
                ((Module)embedding).train();
                ((Module)output).train();

                for (int i = 0; i < trainToks.Length - 1; i++)
                {
                    // skip until we have a full window
                    if (i < windowSize - 1)
                        continue;

                    int start = i - windowSize + 1;
                    int end = i; // inclusive
                    int[] window = trainToks[start..(end + 1)]; // length = windowSize

                    int targetAction = trainActs[i]; // 0=sell,1=hold,2=buy

                    opt.zero_grad();

                    // [W] -> [W, hidden]
                    var inputIdx = torch.tensor(window, dtype: torch.int64);
                    var emb = embedding.forward(inputIdx); // [W, hidden]

                    // Feed sequence through fractal core step-by-step
                    Tensor global = null;
                    for (int t = 0; t < windowSize; t++)
                    {
                        var e_t = emb[t].unsqueeze(0); // [1, hidden]
                        global = core.forward(e_t, orderK); // [1, hidden]
                    }

                    var logits = output.forward(global); // [1,3]

                    var targetTensor = torch.tensor(new long[] { (long)targetAction });
                    var loss = criterion(logits, targetTensor);

                    // logging: decode current price (last in window) + action
                    int inId = trainToks[i];
                    double inVal = tokenizer.Decode(inId);
                    int predAction = logits.argmax(1).ToInt32();
                    string expAct = actionNames[targetAction];
                    string predAct = actionNames[predAction];

                    float lossVal = 0f;
                    try { lossVal = loss.to(torch.CPU).ToSingle(); } catch { }

                    Console.WriteLine($"Train ep={ep} step={i} price={inVal:F6} expected={expAct} pred={predAct} loss={lossVal:F6}");

                    loss.backward();
                    opt.step();

                    totalLoss += loss.to(torch.CPU).ToSingle();
                    steps++;
                }
                Console.WriteLine($"Epoch {ep}/{epochs} — avg loss={(steps > 0 ? totalLoss / steps : 0):F6}");
            }

            // --------------------
            // Prediction on test set (no training)
            // --------------------
            Console.WriteLine("Predicting on test set...");
            core.eval();
            ((Module)embedding).eval();
            ((Module)output).eval();

            int correct = 0;
            int total = 0;
            var preds = new List<int>();

            for (int i = 0; i < testToks.Length - 1; i++)
            {
                if (i < windowSize - 1)
                    continue;

                int start = i - windowSize + 1;
                int end = i;
                int[] window = testToks[start..(end + 1)];

                int expectedAct = testActs[i];

                var inputIdx = torch.tensor(window, dtype: torch.int64);
                var emb = embedding.forward(inputIdx);

                Tensor global = null;
                for (int t = 0; t < windowSize; t++)
                {
                    var e_t = emb[t].unsqueeze(0);
                    global = core.forward(e_t, orderK);
                }

                var logits = output.forward(global);
                int predAct = logits.argmax(1).ToInt32();
                preds.Add(predAct);

                if (predAct == expectedAct) correct++;
                total++;
            }

            Console.WriteLine($"Test accuracy: {correct}/{total} = {(total > 0 ? 100.0 * correct / total : 0):F2}%");

            // Print some sample predictions
            Console.WriteLine("Sample predictions (price + actions):");
            for (int i = windowSize - 1, j = 0; i < testToks.Length - 1 && j < Math.Min(20, preds.Count); i++, j++)
            {
                double priceVal = tokenizer.Decode(testToks[i]);
                string expAct = actionNames[testActs[i]];
                string predAct = actionNames[preds[j]];
                Console.WriteLine($"i={i} price={priceVal:F4} pred={predAct} expected={expAct}");
            }

            Console.WriteLine("Done.");
        }
    }
}
