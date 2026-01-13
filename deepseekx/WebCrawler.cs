using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Linq;
using System.Threading;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public static class WebCrawler
{
    // existing splitter
    public static Func<Uri, string, string[]> HtmlSplitter = DefaultHtmlSplitter;

    // --- Collection & classifier state ---
    private static readonly HashSet<string> _vocab = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
    private static readonly List<string> _collectedEntries = new List<string>(); // full anchor entries
    private static readonly Queue<string> _pendingCandidates = new Queue<string>();
    private static readonly object _lock = new object();

    private static WordTokenizer? _tokenizer = null;

    // simple LSTM-based sequence classifier
    private static Module? _clsEmbedding = null;
    private static LSTMCell? _clsLstm = null;
    private static Module? _clsOutput = null;
    private static int _clsVocabSize = 0;
    private static int _clsNumClasses = 0;
    private const int EmbDim = 64;
    private const int HiddenDim = 128;

    private static readonly List<string> _classNames = new List<string>();

    // --- Extra site routine state ---
    private static string? _extraSiteUrl = null;
    private static CancellationTokenSource? _extraSiteCts = null;
    private static Task? _extraSiteTask = null;

    // --- Default HTML splitter (unchanged) ---
    private static string[] DefaultHtmlSplitter(Uri uri, string html)
    {
        if (string.IsNullOrEmpty(html)) return Array.Empty<string>();

        var marker = "Latest words calculated for Gematria";
        var markerIdx = html.IndexOf(marker, StringComparison.OrdinalIgnoreCase);
        if (markerIdx < 0) return Array.Empty<string>();

        try
        {
            int divStart = html.LastIndexOf("<div", markerIdx, StringComparison.OrdinalIgnoreCase);
            if (divStart < 0) divStart = markerIdx;
            int divEnd = html.IndexOf("</div>", markerIdx, StringComparison.OrdinalIgnoreCase);
            if (divEnd < 0) divEnd = html.Length - 1;
            else divEnd += "</div>".Length;
            var segment = html.Substring(divStart, Math.Min(html.Length - divStart, divEnd - divStart));

            var results = new List<string>();
            var aRegex = new Regex(@"<a[^>]*>(.*?)</a>", RegexOptions.IgnoreCase | RegexOptions.Singleline);
            foreach (Match m in aRegex.Matches(segment))
            {
                var txt = m.Groups[1].Value;
                txt = System.Net.WebUtility.HtmlDecode(txt).Trim();
                if (!string.IsNullOrWhiteSpace(txt)) results.Add(txt);
            }

            if (results.Count > 0) return results.ToArray();

            var qRegex = new Regex(@"[?&]word=([^\""'&<>\s]+(?:%20|\+|[^\""'&<>\s])*)", RegexOptions.IgnoreCase);
            var qResults = new List<string>();
            foreach (Match m in qRegex.Matches(segment))
            {
                var raw = m.Groups[1].Value;
                try { raw = Uri.UnescapeDataString(raw); } catch { }
                raw = System.Net.WebUtility.HtmlDecode(raw).Trim();
                if (!string.IsNullOrWhiteSpace(raw)) qResults.Add(raw);
            }
            if (qResults.Count > 0) return qResults.ToArray();
            return Array.Empty<string>();
        }
        catch { return Array.Empty<string>(); }
    }

    // --- Common add parts helper (thread-safe) ---
    private static void AddParts(IEnumerable<string> parts)
    {
        lock (_lock)
        {
            foreach (var p in parts)
            {
                if (!_collectedEntries.Contains(p)) _collectedEntries.Add(p);

                var toks = p.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries).Select(t => t.Trim()).Where(t => t.Length > 0);
                foreach (var tk in toks)
                {
                    if (!_vocab.Contains(tk)) _vocab.Add(tk);
                }
            }
        }
    }

    // --- Simple crawl once and collect entries ---
    public static async Task CollectFromSiteOnceAsync(string url)
    {
        try
        {
            using var client = new HttpClient();
            client.Timeout = TimeSpan.FromSeconds(10);
            try { client.DefaultRequestHeaders.UserAgent.ParseAdd("Mozilla/5.0 (compatible; DeepSeekX/1.0)"); } catch { }

            var uri = new Uri(url);
            var content = await client.GetStringAsync(uri);
            var parts = HtmlSplitter(uri, content);

            AddParts(parts);

            Console.WriteLine($"Collected {parts.Length} entries, vocab size now {_vocab.Count}");
        }
        catch (Exception ex)
        {
            Console.WriteLine("Collect failed: " + ex.Message);
        }
    }

    // --- Extra-site worker ---
    private static async Task ExtraSiteWorkerAsync(string extraSiteUrl, int intervalMs, CancellationToken token)
    {
        using var client = new HttpClient();
        client.Timeout = TimeSpan.FromSeconds(10);
        try { client.DefaultRequestHeaders.UserAgent.ParseAdd("Mozilla/5.0 (compatible; DeepSeekX/1.0)"); } catch { }

        Uri extraUri;
        try { extraUri = new Uri(extraSiteUrl); } catch { Console.WriteLine("Invalid extra site URL"); return; }

        while (!token.IsCancellationRequested)
        {
            try
            {
                string content = string.Empty;
                try { content = await client.GetStringAsync(extraUri); } catch (Exception ex) { Console.WriteLine($"Extra site fetch failed: {ex.Message}"); }

                if (!string.IsNullOrEmpty(content))
                {
                    string[] parts = HtmlSplitter(extraUri, content);
                    if (parts != null && parts.Length > 0)
                    {
                        AddParts(parts);
                        Console.WriteLine($"[ExtraSite] Collected {parts.Length} entries, vocab size now {_vocab.Count}");
                    }
                    else
                    {
                        Console.WriteLine("[ExtraSite] No parts extracted from extra site.");
                    }
                }
            }
            catch (OperationCanceledException) { break; }
            catch (Exception ex) { Console.WriteLine("Extra site worker error: " + ex.Message); }

            try { await Task.Delay(intervalMs, token); } catch (TaskCanceledException) { break; }
        }

        Console.WriteLine("Extra site routine stopped.");
    }

    public static void StartExtraSiteRoutine(string extraSiteUrl, int intervalMs = 10000)
    {
        StopExtraSiteRoutine();

        _extraSiteUrl = extraSiteUrl;
        _extraSiteCts = new CancellationTokenSource();
        _extraSiteTask = Task.Run(() => ExtraSiteWorkerAsync(extraSiteUrl, intervalMs, _extraSiteCts.Token));

        Console.WriteLine($"Started extra site routine for {_extraSiteUrl} (interval {intervalMs} ms)");
    }

    public static void StopExtraSiteRoutine()
    {
        try
        {
            if (_extraSiteCts != null)
            {
                _extraSiteCts.Cancel();
                _extraSiteCts.Dispose();
                _extraSiteCts = null;
            }
        }
        catch { }
        _extraSiteTask = null;
        _extraSiteUrl = null;
    }

    public static bool IsExtraSiteRunning() => _extraSiteTask != null && !_extraSiteTask.IsCompleted && !(_extraSiteCts?.IsCancellationRequested ?? true);

    // --- Build tokenizer and classifier from collected vocab and class names ---
    public static void InitializeClassifierFromCollected()
    {
        if (_vocab.Count == 0)
        {
            Console.WriteLine("No vocabulary collected.");
            return;
        }
        if (_classNames.Count == 0)
        {
            Console.WriteLine("No classes defined. Use menu to add class names first.");
            return;
        }

        // build tokenizer
        _tokenizer = new WordTokenizer();
        _tokenizer.BuildVocabulary(_vocab, maxVocab: 10000);
        _clsVocabSize = _tokenizer.VocabSize;
        _clsNumClasses = _classNames.Count;

        // build model: embedding + LSTMCell + linear
        _clsEmbedding = nn.Embedding(_clsVocabSize, EmbDim);
        _clsLstm = nn.LSTMCell(EmbDim, HiddenDim);
        _clsOutput = nn.Linear(HiddenDim, _clsNumClasses);

        Console.WriteLine($"Initialized classifier: vocab={_clsVocabSize} classes={_clsNumClasses}");
    }

    // Predict class for a given text sequence
    public static int PredictClass(string text)
    {
        if (_tokenizer == null || _clsEmbedding == null || _clsLstm == null || _clsOutput == null) return -1;
        var toks = _tokenizer.Tokenize(text);
        if (toks.Length == 0) return -1;

        var h = torch.zeros(1, HiddenDim);
        var c = torch.zeros(1, HiddenDim);

        foreach (var t in toks)
        {
            var idx = torch.tensor(new long[] { t });
            var emb = ((Module<Tensor, Tensor>)_clsEmbedding).forward(idx); // [1,emb]
            (h, c) = _clsLstm.forward(emb, (h, c));
        }

        var logits = ((Module<Tensor, Tensor>)_clsOutput).forward(h);
        var pred = logits.argmax(1).ToInt32();
        return pred;
    }

    // Train classifier on single labeled example
    public static void TrainClassifierOnExample(string text, int labelId, int iters = 200, double lr = 1e-3)
    {
        if (_tokenizer == null || _clsEmbedding == null || _clsLstm == null || _clsOutput == null)
        {
            Console.WriteLine("Classifier not initialized.");
            return;
        }

        var toks = _tokenizer.Tokenize(text);
        if (toks.Length == 0) { Console.WriteLine("Empty tokenization."); return; }

        var parameters = ((Module)_clsEmbedding).parameters().Concat(((Module)_clsOutput).parameters()).Concat(((Module)_clsLstm).parameters());
        var opt = torch.optim.Adam(parameters, lr: lr);

        for (int e = 0; e < iters; e++)
        {
            opt.zero_grad();

            var h = torch.zeros(1, HiddenDim);
            var c = torch.zeros(1, HiddenDim);

            foreach (var t in toks)
            {
                var idx = torch.tensor(new long[] { t });
                var emb = ((Module<Tensor, Tensor>)_clsEmbedding).forward(idx);
                (h, c) = _clsLstm.forward(emb, (h, c));
            }

            var logits = ((Module<Tensor, Tensor>)_clsOutput).forward(h); // [1,C]
            var target = torch.tensor(new long[] { labelId });
            var loss = nn.functional.cross_entropy(logits, target);
            loss.backward();
            opt.step();
        }
    }

    // Interactive review of collected entries for labeling and training
    public static void LabelAndTrainInteractive()
    {
        if (_tokenizer == null) { Console.WriteLine("Classifier not initialized."); return; }

        foreach (var entry in _collectedEntries.ToArray())
        {
            Console.WriteLine($"Entry: {entry}");
            int pred = PredictClass(entry);
            if (pred >= 0 && pred < _classNames.Count)
                Console.WriteLine($"Predicted class: {pred} => {_classNames[pred]}");
            else
                Console.WriteLine("No prediction available.");

            Console.Write("Is this correct? (y)es / (n)umber to provide class id / skip: ");
            var r = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(r)) continue;
            if (r.Equals("y", StringComparison.OrdinalIgnoreCase) || r.Equals("yes", StringComparison.OrdinalIgnoreCase))
            {
                if (pred >= 0) { TrainClassifierOnExample(entry, pred, iters: 100); Console.WriteLine("Trained on predicted label."); }
            }
            else if (int.TryParse(r, out var cid))
            {
                if (cid >= 0 && cid < _classNames.Count)
                {
                    TrainClassifierOnExample(entry, cid, iters: 200);
                    Console.WriteLine($"Trained on provided class id {cid} ({_classNames[cid]}).");
                }
                else Console.WriteLine("Invalid class id.");
            }
            else
            {
                Console.WriteLine("Skipped.");
            }
        }
    }

    // Simple menu exposed to caller
    public static async Task InteractiveMenuAsync()
    {
        while (true)
        {
            Console.WriteLine();
            Console.WriteLine("=== WebCrawler Interactive ===");
            Console.WriteLine("1) Collect from site once (provide URL)");
            Console.WriteLine("2) List collected entries & vocab size");
            Console.WriteLine("3) Manage classes (add/list/clear)");
            Console.WriteLine("4) Initialize classifier from collected vocab");
            Console.WriteLine("5) Label & train on collected entries");
            Console.WriteLine("6) Predict class for a sequence");
            Console.WriteLine("7) Train on a custom labeled example");
            Console.WriteLine("8) Manage extra-site routine (start/stop/manual)");
            Console.WriteLine("9) Exit menu");
            Console.Write("Choose: ");
            var choice = Console.ReadLine();
            if (choice == null) continue;
            switch (choice.Trim())
            {
                case "1":
                    Console.Write("URL to collect from: ");
                    var url = Console.ReadLine();
                    if (string.IsNullOrWhiteSpace(url)) { Console.WriteLine("No URL."); break; }
                    await CollectFromSiteOnceAsync(url.Trim());
                    break;
                case "2":
                    Console.WriteLine($"Collected entries: {_collectedEntries.Count}");
                    for (int i = 0; i < _collectedEntries.Count; i++) Console.WriteLine($"{i}: {_collectedEntries[i]}");
                    Console.WriteLine($"Vocabulary size: {_vocab.Count}");
                    break;
                case "3":
                    Console.WriteLine("Class management: 1)add 2)list 3)clear");
                    var opt = Console.ReadLine();
                    if (opt == "1") { Console.Write("New class name: "); var cn = Console.ReadLine(); if (!string.IsNullOrWhiteSpace(cn)) { _classNames.Add(cn.Trim()); Console.WriteLine("Added."); } }
                    else if (opt == "2") { for (int i = 0; i < _classNames.Count; i++) Console.WriteLine($"{i}: {_classNames[i]}"); }
                    else if (opt == "3") { _classNames.Clear(); Console.WriteLine("Cleared."); }
                    break;
                case "4":
                    InitializeClassifierFromCollected();
                    break;
                case "5":
                    LabelAndTrainInteractive();
                    break;
                case "6":
                    Console.Write("Enter sequence to classify: ");
                    var seq = Console.ReadLine();
                    if (string.IsNullOrWhiteSpace(seq)) break;
                    var p = PredictClass(seq);
                    if (p >= 0 && p < _classNames.Count) Console.WriteLine($"Predicted: {p} => {_classNames[p]}"); else Console.WriteLine("No prediction available or classifier not initialized.");
                    break;
                case "7":
                    Console.Write("Enter sequence: "); var seqIn = Console.ReadLine(); if (string.IsNullOrWhiteSpace(seqIn)) break;
                    Console.Write("Enter class id: "); var cidLine = Console.ReadLine(); if (!int.TryParse(cidLine, out var cidv)) { Console.WriteLine("Invalid id."); break; }
                    TrainClassifierOnExample(seqIn, cidv, iters: 300);
                    Console.WriteLine("Trained on provided example.");
                    break;
                case "8":
                    await ManageExtraSiteMenuAsync();
                    break;
                case "9":
                    return;
                default:
                    Console.WriteLine("Unknown option.");
                    break;
            }
        }
    }

    private static async Task ManageExtraSiteMenuAsync()
    {
        while (true)
        {
            Console.WriteLine();
            Console.WriteLine("--- Extra-site routine ---");
            Console.WriteLine($"Current extra site: {_extraSiteUrl ?? "(none)"}");
            Console.WriteLine($"Running: {IsExtraSiteRunning()}");
            Console.WriteLine("1) Set extra site URL");
            Console.WriteLine("2) Start routine");
            Console.WriteLine("3) Stop routine");
            Console.WriteLine("4) Fetch once now");
            Console.WriteLine("5) Back");
            Console.Write("Choose: ");
            var c = Console.ReadLine();
            if (c == null) continue;
            switch (c.Trim())
            {
                case "1":
                    Console.Write("Extra site URL: ");
                    var u = Console.ReadLine();
                    if (string.IsNullOrWhiteSpace(u)) Console.WriteLine("No URL."); else { _extraSiteUrl = u.Trim(); Console.WriteLine("Saved."); }
                    break;
                case "2":
                    if (string.IsNullOrWhiteSpace(_extraSiteUrl)) { Console.WriteLine("Set URL first."); break; }
                    StartExtraSiteRoutine(_extraSiteUrl!, intervalMs: 10_000);
                    break;
                case "3":
                    StopExtraSiteRoutine();
                    Console.WriteLine("Stopped.");
                    break;
                case "4":
                    if (string.IsNullOrWhiteSpace(_extraSiteUrl)) { Console.WriteLine("Set URL first."); break; }
                    await CollectFromSiteOnceAsync(_extraSiteUrl!);
                    break;
                case "5":
                    return;
                default:
                    Console.WriteLine("Unknown.");
                    break;
            }
        }
    }
}