using System;
using System.Collections.Generic;
using System.Text;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using System.Net;
using System.Text.Json;
using System.Threading.Tasks;
using System.Threading;
using System.IO;
using System.Linq;
using static TorchSharp.torch.nn;
using System.Reflection;
using System.Text.RegularExpressions;


public class MultiAgentFractalCore : nn.Module
{

    private readonly Linear globalAnchorGate;

    private readonly List<FractalOpponent> agents;

    private readonly Linear crossAttentionQ;
    private readonly Linear crossAttentionK;
    private readonly Linear crossAttentionV;

    private readonly Linear globalFuse;

    private readonly int hiddenSize;
    private readonly int agentCount;
    private static Linear globalRamanujanHead;

    // Expose hidden size publicly for external users (e.g. ChessTrainer)
    public int HiddenSize => hiddenSize;

    // Allow external callers to evaluate a specific internal agent by index
    public (Tensor output, Tensor h, Tensor c) EvaluateAgent(int agentIndex, Tensor x, Tensor h, Tensor c)
    {
        if (agentIndex < 0 || agentIndex >= agentCount)
            throw new ArgumentOutOfRangeException(nameof(agentIndex));
        return agents[agentIndex].forward(x, h, c);
    }

    public static Tensor RamanujanSum(Tensor[] states, Linear projection)
    {
        if (states == null || states.Length == 0)
        {
            long expectedInputSize = projection.weight.shape[1];
            return torch.zeros(new long[] { 1, expectedInputSize }, device: projection.weight.device);
        }

        // Build two EMAs over the states with different decay rates (alpha1, alpha2)
        // Stack states to [S,H]
        var M = torch.cat(states, dim: 0); // [S,H]

        float alpha1 = 0.4f;
        float alpha2 = 0.8f;

        var anchor1 = torch.zeros(1, M.shape[1], device: M.device);
        var anchor2 = torch.zeros(1, M.shape[1], device: M.device);

        for (int t = 0; t < M.shape[0]; t++)
        {
            var h_t = M[t].unsqueeze(0); // [1,H]
            anchor1 = (1 - alpha1) * anchor1 + alpha1 * h_t;
            anchor2 = (1 - alpha2) * anchor2 + alpha2 * h_t;
        }

        // Concatenate anchors -> [1, 2H]
        var concat = torch.cat(new Tensor[] { anchor1, anchor2 }, dim: 1);
        var hStar = projection.forward(concat);
        if (DateTime.Now.Millisecond % 5000 == 0)
            DebugRamanujanEnergy(states, hStar);
        return hStar;
    }
    public static void DebugRamanujanEnergy(Tensor[] states, Tensor ramanujanOutput)
    {
        using (var scope = torch.NewDisposeScope())
        {
            // Durchschnittliche Energie der Input-Zustände
            var inputEnergy = torch.stack(states).pow(2).mean().ToSingle();
            // Energie des Ankers
            var anchorEnergy = ramanujanOutput.pow(2).mean().ToSingle();

            // Kohärenz-Faktor: Wie stark weicht der Anker vom Durchschnitt ab?
            float coherence = anchorEnergy / (inputEnergy + 1e-6f);

            Console.WriteLine($"[DEBUG Ramanujan] Energy Ratio: {coherence:F4} | Anchor L2: {anchorEnergy:F4}");

            if (coherence > 5.0f)
                Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine(" --> WARNING: High Resonance in Ramanujan Head!");
            Console.ResetColor();
        }
    }
    public MultiAgentFractalCore(int hiddenSize, int maxDepth, int agentCount)
        : base("multi_agent_fractal_core")
    {
        this.hiddenSize = hiddenSize;
        this.agentCount = agentCount;

        agents = new List<FractalOpponent>();
        for (int i = 0; i < agentCount; i++)
            agents.Add(new FractalOpponent(hiddenSize, maxDepth));

        crossAttentionQ = nn.Linear(hiddenSize, hiddenSize);
        crossAttentionK = nn.Linear(hiddenSize, hiddenSize);
        crossAttentionV = nn.Linear(hiddenSize, hiddenSize);

        // flatten(A*H) -> H
        globalFuse = nn.Linear(hiddenSize * agentCount, hiddenSize);

        // Ramanujan head: collapses u1,u2 -> g*
        globalRamanujanHead = nn.Linear(hiddenSize * 2, hiddenSize);

        // NEW: AnchorGate over reasoning anchor
        globalAnchorGate = nn.Linear(hiddenSize * 2, hiddenSize);

        RegisterComponents();

        try
        {
            long bytes = GetParameterByteSize();
            Console.WriteLine($"MultiAgentFractalCore parameter size: {FormatBytes(bytes)} ({bytes} bytes)");
        }
        catch (Exception ex)
        {
            Console.WriteLine("Could not compute parameter size: " + ex.Message);
        }
    }

    /// <summary>
    /// Ein Reasoning-Schritt über alle Agenten:
    /// - jeder Agent macht seinen Fraktal-Schritt
    /// - danach Cross-Attention zwischen allen Agenten
    /// - daraus entsteht ein globaler Zustand
    /// </summary>
    private (Tensor[] hNext, Tensor[] cNext, Tensor global) ReasonOnce(
        Tensor x,
        Tensor[] hCur,
        Tensor[] cCur)
    {
        var hLocal = new Tensor[agentCount];
        var cNext = new Tensor[agentCount];
        var output = new Tensor[agentCount];
        // 1) Jeder Agent updatet seinen internen Zustand
        for (int i = 0; i < agentCount; i++)
        {
            (output[i], hLocal[i], cNext[i]) = agents[i].forward(x, hCur[i], cCur[i]);
        }

        // 2) Cross-Agent-Attention
        var H = torch.cat(hLocal, 0); // [A,H]

        var Q = crossAttentionQ.forward(H); // [A,H]
        var K = crossAttentionK.forward(H); // [A,H]
        var V = crossAttentionV.forward(H); // [A,H]

        var scores = torch.matmul(Q, K.transpose(0, 1)); // [A,A]
        var weights = scores.softmax(1);                 // [A,A]

        var H_attn = torch.matmul(weights, V); // [A,H]

        // 3) Aktualisierte Hidden-States nach wechselseitigem Denken
        var hNext = new Tensor[agentCount];
        for (int i = 0; i < agentCount; i++)
        {
            var h_i = H_attn.index(torch.TensorIndex.Single(i)).unsqueeze(0); // [1,H]
            hNext[i] = h_i;
        }

        // 4) Globaler Zustand: alle Agenten zusammen
        var H_flat = H_attn.flatten(0, 1).unsqueeze(0); // [1, A*H]
        var global = globalFuse.forward(H_flat).tanh(); // [1,H]

        return (hNext, cNext, global);
    }

    /// <summary>
    /// Generalisierte Ordnung: orderK = wie oft ReasonOnce angewendet wird.
    /// orderK = 1 -> 1st order
    /// orderK = 2 -> 2nd order (jeder denkt über die anderen nach, nachdem alle einmal gedacht haben)
    /// usw.
    /// </summary>
    public Tensor forward(Tensor x, int orderK = 1)
    {
        var h = new Tensor[agentCount];
        var c = new Tensor[agentCount];

        for (int i = 0; i < agentCount; i++)
        {
            h[i] = torch.zeros(1, hiddenSize);
            c[i] = torch.zeros(1, hiddenSize);
        }

        var globals = new List<Tensor>();

        int steps = Math.Max(1, orderK);
        Tensor lastGlobal = null;

        for (int k = 0; k < steps; k++)
        {
           
            Tensor global;
            (h, c, global) = ReasonOnce(x, h, c);
            globals.Add(global); // [1,H]
            lastGlobal = global;
        }

        // Ramanujan-style anchor over reasoning iterations
        var globalAnchor = RamanujanSum(globals.ToArray(), globalRamanujanHead); // [1,H]

        // === AnchorGate ===
        // Gate input: concat(current global, anchor) -> [1,2H]
        var gateInput = torch.cat(new Tensor[] { lastGlobal, globalAnchor }, dim: 1); // [1,2H]
        var gateRaw = globalAnchorGate.forward(gateInput);                            // [1,H]
        var g = gateRaw.sigmoid();                                                    // [1,H] in (0,1)

        // Blend: h_out = g * anchor + (1-g) * lastGlobal
        var globalOut = g * globalAnchor + (1 - g) * lastGlobal; // [1,H]
        lastGlobal = globalOut;
        return globalOut;
    }

    // Compute total parameter size (in bytes) for this module using state_dict
    public long GetParameterByteSize()
    {
        long total = 0;
        var dict = this.state_dict(); // IDictionary<string, Tensor>
        foreach (var kv in dict)
        {
            var tensor = kv.Value;
            if (tensor is null) continue;
            // compute number of elements
            long elems = 1;
            foreach (var d in tensor.shape)
            {
                elems *= d;
            }
            int bytesPerElem = 4; // default float32
            try
            {
                switch (tensor.dtype)
                {
                    case ScalarType.Float32:
                        bytesPerElem = 4; break;
                    case ScalarType.Float64:
                        bytesPerElem = 8; break;
                    case ScalarType.Int64:
                        bytesPerElem = 8; break;
                    case ScalarType.Int32:
                        bytesPerElem = 4; break;
                    case ScalarType.Int16:
                        bytesPerElem = 2; break;
                    case ScalarType.Byte:
                    case ScalarType.Bool:
                        bytesPerElem = 1; break;
                    default:
                        bytesPerElem = 4; break;
                }
            }
            catch
            {
                bytesPerElem = 4;
            }

            total += elems * (long)bytesPerElem;
        }
        return total;
    }

    private static string FormatBytes(long bytes)
    {
        const long KB = 1024;
        const long MB = KB * 1024;
        const long GB = MB * 1024;
        if (bytes >= GB) return (bytes / (double)GB).ToString("F2") + " GB";
        if (bytes >= MB) return (bytes / (double)MB).ToString("F2") + " MB";
        if (bytes >= KB) return (bytes / (double)KB).ToString("F2") + " KB";
        return bytes + " bytes";
    }
}

    // Sparse activation module: only apply linear transform to neurons with positive activation.
    // Input shape: [1, hiddenSize]
    public class SparseActivationModule : nn.Module
    {
        private readonly Linear fc;
        private readonly int hiddenSize;

        public SparseActivationModule(int hiddenSize) : base("sparse_activation")
        {
            this.hiddenSize = hiddenSize;
            // Linear layer of shape (hiddenSize, hiddenSize)
            fc = nn.Linear(hiddenSize, hiddenSize);
            RegisterComponents();
        }

        // Forward takes x: [1, hiddenSize] and returns reconstructed tensor of same shape
        public Tensor forward(Tensor x)
        {
            var input = x;
            var mask = input.gt(0);
            var active = input.masked_select(mask);
            var inputForLinear = torch.zeros_like(input);
            inputForLinear = inputForLinear.masked_scatter(mask, active);
            var transformed = fc.forward(inputForLinear);
            var output = input.clone();
            var transformedActive = transformed.masked_select(mask);
            output = output.masked_scatter(mask, transformedActive);
            return output;
        }
    }

    // Lightweight HTTP API server built on HttpListener to expose basic endpoints that the PHP UI expects.
    public static class FractalApiServer
    {
        // When true, HTTP handler will attempt to use ExternalTokenizer (Microsoft.ML.Tokenizers) as a secondary tokenizer.
        // Default false so in-process menu flows use the original Program._wordTokenizerHolder behavior.
        public static bool UseExternalTokenizer { get; private set; } = false;

        // expand embedding and output layers if tokenizer added new tokens
        private static void EnsureVocabExpanded(int requiredVocab)
        {
            if (Program._wordTokenizerHolder == null) return;
            if (Program._wordEmbeddingHolder == null || Program._wordOutputHolder == null) return;

            // Embedding
            try
            {
                // get embedding state dict and locate weight
                var embState = ((dynamic)Program._wordEmbeddingHolder).state_dict();
                string embKey = null;
                foreach (var kObj in (System.Collections.IEnumerable)embState.Keys)
                {
                    var k = kObj.ToString();
                    if (k.ToLowerInvariant().Contains("weight")) { embKey = k; break; }
                }
                 if (embKey != null)
                 {
                    var oldWeight = (Tensor)embState[embKey];
                    long oldVocab = oldWeight.shape[0];
                    long embDim = oldWeight.shape[1];
                     if (requiredVocab > oldVocab)
                     {
                         Console.WriteLine($"Expanding embedding from {oldVocab} to {requiredVocab} tokens");
                         var rand = torch.randn(new long[] { requiredVocab - oldVocab, embDim }) * 0.01f;
                         var newWeight = torch.cat(new Tensor[] { oldWeight, rand }, 0);

                         var newEmb = nn.Embedding(requiredVocab, (int)embDim);
                         var newState = new Dictionary<string, Tensor>();
                         // find key name in newEmb state dict
                        var newKeysEnumerable = (System.Collections.IEnumerable)((dynamic)newEmb).state_dict().Keys;
                        string newWeightKey = null;
                        foreach (var kObj in newKeysEnumerable)
                        {
                            var k = kObj.ToString(); if (k.ToLowerInvariant().Contains("weight")) { newWeightKey = k; break; }
                        }
                        if (newWeightKey == null) newWeightKey = "weight";
                         newState[newWeightKey] = newWeight;
                         newEmb.load_state_dict(newState, strict: false);

                         Program._wordEmbeddingHolder = newEmb;
                         Console.WriteLine("Embedding expanded successfully.");
                     }
                 }
             }
             catch (Exception ex)
             {
                 Console.WriteLine("Embedding expansion error: " + ex.Message);
             }

             // Output linear (may be Sequential or Linear)
             try
             {
                var outState = ((dynamic)Program._wordOutputHolder).state_dict();
                // find weight (2D) and bias (1D) by iterating keys
                string outWeightKey = null;
                string outBiasKey = null;
                foreach (var kObj in (System.Collections.IEnumerable)outState.Keys)
                {
                    var k = kObj.ToString();
                    var t = (Tensor)outState[k];
                    if (outWeightKey == null && t.dim() == 2) outWeightKey = k;
                    if (outBiasKey == null && t.dim() == 1) outBiasKey = k;
                    if (outWeightKey != null && outBiasKey != null) break;
                }
                 if (outWeightKey != null)
                 {
                    var oldOutW = (Tensor)outState[outWeightKey]; // [oldVocab, hidden]
                    long oldVocabOut = oldOutW.shape[0];
                    long hidden = oldOutW.shape[1];
                     if (requiredVocab > oldVocabOut)
                     {
                         Console.WriteLine($"Expanding output layer from {oldVocabOut} to {requiredVocab} tokens");
                         var randW = torch.randn(new long[] { requiredVocab - oldVocabOut, hidden }) * 0.01f;
                         var newOutW = torch.cat(new Tensor[] { oldOutW, randW }, 0);

                        Tensor oldBias = null;
                        if (!string.IsNullOrEmpty(outBiasKey)) oldBias = (Tensor)outState[outBiasKey];
                         Tensor newBias = null;
                         if (!object.ReferenceEquals(oldBias, null))
                         {
                             var zeros = torch.zeros(new long[] { requiredVocab - oldVocabOut });
                             newBias = torch.cat(new Tensor[] { oldBias, zeros }, 0);
                         }

                         // Build new output module: create a new linear with output size = requiredVocab
                         var newLinear = nn.Linear((int)hidden, requiredVocab);
                         var state = new Dictionary<string, Tensor>();
                         state["weight"] = newOutW;
                         if (!object.ReferenceEquals(newBias, null)) state["bias"] = newBias;
                         newLinear.load_state_dict(state, strict: false);

                         Program._wordOutputHolder = newLinear;
                         Console.WriteLine("Output layer expanded successfully.");
                     }
                 }
             }
             catch (Exception ex)
             {
                 Console.WriteLine("Output expansion error: " + ex.Message);
             }
         }

        private static HttpListener? _listener;
        private static CancellationTokenSource? _cts;
        private static List<Interaction> _interactions = new List<Interaction>();
        private static SparseActivationModule? _sparseModule;

        // New: in-memory suggestion store
        private static List<Suggestion> _suggestions = new List<Suggestion>();

        public class Interaction
        {
            public string Id { get; set; } = Guid.NewGuid().ToString();
            public string Question { get; set; } = "";
            public string Answer { get; set; } = "";
            public List<string> Context { get; set; } = new List<string>();
            public long Timestamp { get; set; } = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        }

        // New suggestion record
        public class Suggestion
        {
            public string Id { get; set; } = Guid.NewGuid().ToString();
            public string InteractionId { get; set; } = "";
            public string SuggestionText { get; set; } = "";
            public string SessionId { get; set; } = "";
            public string Status { get; set; } = "pending"; // pending | accepted | rejected
            public long Timestamp { get; set; } = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        }

        public static void Start(string prefix = "http://localhost:5000/")
        {
            if (_listener != null) return;

            // Mark that we are running as webserver -> allow external tokenizer
            UseExternalTokenizer = true;

            // Try to load words_alpha.txt into Program tokenizer first
            try
            {
                var baseDir = AppContext.BaseDirectory;
                var path1 = Path.Combine(baseDir, "words_alpha.txt");
                var path2 = Path.Combine(Directory.GetCurrentDirectory(), "words_alpha.txt");
                string? wordsPath = null;
                if (File.Exists(path1)) wordsPath = path1;
                else if (File.Exists(path2)) wordsPath = path2;

                if (wordsPath != null)
                {
                    var lines = File.ReadAllLines(wordsPath).Where(l => !string.IsNullOrWhiteSpace(l)).Select(l => l.Trim()).ToArray();
                    // Initialize tokenizer and add words
                    Program._wordTokenizerHolder = new WordTokenizer();
                    foreach (var w in lines) Program._wordTokenizerHolder.AddWord(w);
                    Console.WriteLine($"Loaded vocabulary from {wordsPath} with {Program._wordTokenizerHolder.VocabSize} entries");
                }

                // Ensure model initialized
                Program.InitializeModel(128);
                _sparseModule = new SparseActivationModule(128);
            }
            catch (Exception ex)
            {
                Console.WriteLine("Vocabulary/model init failed: " + ex.Message);
            }

            _listener = new HttpListener();
            _listener.Prefixes.Add(prefix);
            _cts = new CancellationTokenSource();
            _listener.Start();
            Task.Run(() => ListenLoop(_cts.Token));
            Console.WriteLine($"Fractal API Server listening on {prefix}");
        }

        public static void Stop()
        {
            try
            {
                _cts?.Cancel();
                _listener?.Stop();
            }
            catch { }
            finally
            {
                _listener = null;
                _cts = null;
            }
        }

        private static async Task ListenLoop(CancellationToken ct)
        {
            if (_listener == null) return;
            while (!ct.IsCancellationRequested)
            {
                HttpListenerContext ctx;
                try
                {
                    ctx = await _listener.GetContextAsync();
                }
                catch (OperationCanceledException) { break; }
                catch (Exception ex)
                {
                    Console.WriteLine("Listener error: " + ex.Message);
                    break;
                }

                _ = Task.Run(() => HandleRequest(ctx));
            }
        }

        private static async Task HandleRequest(HttpListenerContext ctx)
        {
            var req = ctx.Request;
            var resp = ctx.Response;
            resp.ContentType = "application/json; charset=utf-8";

            // Always add CORS headers to allow browser-based UI to call this server
            try
            {
                resp.Headers.Add("Access-Control-Allow-Origin", "*");
                resp.Headers.Add("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
                resp.Headers.Add("Access-Control-Allow-Headers", "Content-Type, Accept, Authorization");
            }
            catch { /* some headers may be restricted on certain platforms; ignore failures */ }

            // Handle preflight OPTIONS requests quickly
            if (req.HttpMethod == "OPTIONS")
            {
                try
                {
                    resp.StatusCode = 200;
                    try { resp.OutputStream.Close(); } catch { }
                }
                catch { }
                return;
            }

            try
            {
                // Log incoming request for debugging
                try { Console.WriteLine($"Incoming request: {req.HttpMethod} {req.Url}"); } catch { }

                var path = req.Url?.AbsolutePath ?? "";

                // POST /api/query handled earlier
                if (path.StartsWith("/api/query") && req.HttpMethod == "POST")
                {
                    using var sr = new System.IO.StreamReader(req.InputStream, req.ContentEncoding);
                    var body = await sr.ReadToEndAsync();
                    var doc = JsonDocument.Parse(string.IsNullOrWhiteSpace(body) ? "{}" : body);
                    string question = doc.RootElement.GetPropertyOrDefault("question", "");
                    List<string> context = new List<string>();
                    if (doc.RootElement.TryGetProperty("context", out var ctxEl) && ctxEl.ValueKind == JsonValueKind.Array)
                    {
                        foreach (var it in ctxEl.EnumerateArray()) if (it.ValueKind == JsonValueKind.String) context.Add(it.GetString()!);
                    }

                    // Simple pipeline: if Program has tokenizer and embedding use PredictSequence
                    string answer = "";
                    string id = Guid.NewGuid().ToString();
                    try
                    {
                        if (Program._wordTokenizerHolder != null && Program._wordEmbeddingHolder != null && Program._wordOutputHolder != null && Program._fractalCell != null)
                        {
                            // Tokenize input words by using external tokenizer only when running as webserver
                            int[] toks;
                            string[] extTokens = null;
                            if (UseExternalTokenizer)
                                extTokens = ExternalTokenizer.TryTokenizeStrings(question);

                            if (extTokens != null && extTokens.Length > 0)
                            {
                                var tmp = new List<int>();
                                foreach (var tokStr in extTokens)
                                {
                                    var s = tokStr ?? string.Empty;
                                    // Try to encode; if unknown (id == 0) and token isn't the UnknownToken, add it to vocab
                                    int tokenId = Program._wordTokenizerHolder.Encode(s);
                                    /*if (tokenId == 0 && !string.Equals(s, WordTokenizer.UnknownToken, StringComparison.OrdinalIgnoreCase))
                                    {
                                        try
                                        {
                                            tokenId = Program._wordTokenizerHolder.AddWord(s);
                                        }
                                        catch { /* ignore add failures  }
                                    }*/
                                    tmp.Add(tokenId);
                                }
                                toks = tmp.ToArray();

                                // Ensure embedding/output layers can handle the new vocab size
                                try
                                {
                                    EnsureVocabExpanded(Program._wordTokenizerHolder.VocabSize);
                                }
                                catch (Exception ex)
                                {
                                    Console.WriteLine("EnsureVocabExpanded failed: " + ex.Message);
                                }
                            }
                             else
                             {
                                 toks = Program._wordTokenizerHolder.Tokenize(question);
                             }

                            // Use recurrent cell with sparse activation applied to hidden state
                            var h = torch.zeros(1, Program._fractalCell.hiddenSize);
                            var c = torch.zeros(1, Program._fractalCell.hiddenSize);

                            var generated = new List<int>();
                            int last = toks.Length > 0 ? toks.Last() : 0;

                            for (int i = 0; i < Math.Max(1, toks.Length); i++)
                            {
                                int tid = i < toks.Length ? toks[i] : last;
                                // clamp tid to vocab range to avoid embedding index errors
                                try
                                {
                                    var vocab = GetVocabSizeSafe();
                                    if (vocab > 0 && (tid < 0 || tid >= vocab))
                                    {
                                        //var unk = GetUnknownTokenId();
                                        //Console.WriteLine($"Token id {tid} out of range (vocab={vocab}), using unknown id {unk}");
                                        //tid = unk;
                                    }
                                }
                                catch { }

                                var input = torch.tensor(new long[] { tid });
                                var emb = ((Module<Tensor, Tensor>)Program._wordEmbeddingHolder).forward(input);
                                var res = Program._fractalCell.forward_step(emb, h, c);
                                h.Dispose(); c.Dispose();
                                h = res.h; c = res.c;
                                last = tid;
                            }

                            // Generation loop
                            int maxGen = 8;
                            var hh = h.clone();
                            var cc = c.clone();
                            for (int gi = 0; gi < maxGen; gi++)
                            {
                                // Apply sparse activation to hh
                                Tensor hhAct;
                                if (_sparseModule != null)
                                {
                                    hhAct = _sparseModule.forward(hh);
                                }
                                else
                                {
                                    hhAct = hh;
                                }

                                var logits = ((Module<Tensor, Tensor>)Program._wordOutputHolder).forward(hhAct);
                                int pid = logits.argmax(1).ToInt32();
                                generated.Add(pid);
                                if (Program._wordTokenizerHolder.Decode(pid) == "<eos>") break;

                                // feed back
                                var idxVal = pid;
                                try
                                {
                                    var vocab = GetVocabSizeSafe();
                                    if (vocab > 0 && (idxVal < 0 || idxVal >= vocab))
                                    {
                                        //var unk = GetUnknownTokenId();
                                        //Console.WriteLine($"Generated token id {idxVal} out of range (vocab={vocab}), using unknown id {unk}");
                                        //idxVal = unk;
                                    }
                                }
                                catch { }
                                var idx = torch.tensor(new long[] { idxVal });
                                var embNext = ((Module<Tensor, Tensor>)Program._wordEmbeddingHolder).forward(idx);
                                var res = Program._fractalCell.forward_step(embNext, hh, cc);
                                hh.Dispose(); cc.Dispose();
                                hh = res.h; cc = res.c;
                            }

                            var words = generated.Select(i => Program._wordTokenizerHolder.Decode(i));
                            answer = string.Join(' ', words);

                            h.Dispose(); c.Dispose();
                        }
                        else if (Program._wordTokenizerHolder != null && Program._wordEmbeddingHolder != null && Program._wordOutputHolder != null)
                        {
                            // Fallback non-recurrent: greedy decode from embedding->output using sparse activation
                            var toks = Program._wordTokenizerHolder.Tokenize(question);
                            int last = toks.Length > 0 ? toks.Last() : 0;
                            var generated = new List<int>();
                            for (int i = 0; i < Math.Max(1, toks.Length); i++)
                            {
                                int tid = i < toks.Length ? toks[i] : last;
                                try
                                {
                                    var vocab = GetVocabSizeSafe();
                                    if (vocab > 0 && (tid < 0 || tid >= vocab))
                                    {
                                        //var unk = GetUnknownTokenId();
                                        //Console.WriteLine($"Token id {tid} out of range (vocab={vocab}), using unknown id {unk}");
                                        //tid = unk;
                                    }
                                }
                                catch { }
                                var input = torch.tensor(new long[] { tid });
                                var emb = ((Module<Tensor, Tensor>)Program._wordEmbeddingHolder).forward(input);

                                Tensor embAct;
                                if (_sparseModule != null)
                                {
                                    embAct = _sparseModule.forward(emb);
                                }
                                else embAct = emb;

                                var logits = ((Module<Tensor, Tensor>)Program._wordOutputHolder).forward(embAct);
                                int pid = logits.argmax(1).ToInt32();
                                generated.Add(pid);
                                last = pid;
                            }
                            var words = generated.Select(i => Program._wordTokenizerHolder.Decode(i));
                            answer = string.Join(' ', words);
                        }
                        else
                        {
                            // fallback: echo
                            answer = "(no model) " + question;
                        }
                    }
                    catch (Exception ex)
                    {
                        answer = "(error) " + ex.Message;
                    }

                    var interaction = new Interaction { Id = id, Question = question, Answer = answer, Context = context, Timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() };
                    lock (_interactions) { _interactions.Insert(0, interaction); if (_interactions.Count > 1000) _interactions.RemoveRange(1000, _interactions.Count - 1000); }

                    var outObj = new { id = id, answer = answer, context = context };
                    var bytes = JsonSerializer.SerializeToUtf8Bytes(outObj);
                    await resp.OutputStream.WriteAsync(bytes, 0, bytes.Length);
                    resp.StatusCode = 200;
                }
                // New: accept suggestions from clients
                else if (path.StartsWith("/api/suggest") && req.HttpMethod == "POST")
                {
                    using var sr = new System.IO.StreamReader(req.InputStream, req.ContentEncoding);
                    var body = await sr.ReadToEndAsync();
                    var doc = JsonDocument.Parse(string.IsNullOrWhiteSpace(body) ? "{}" : body);
                    string interactionId = doc.RootElement.GetPropertyOrDefault("interactionId", "");
                    string suggestionText = doc.RootElement.GetPropertyOrDefault("suggestion", "");
                    string sessionId = doc.RootElement.GetPropertyOrDefault("sessionId", "");

                    var sug = new Suggestion { InteractionId = interactionId, SuggestionText = suggestionText, SessionId = sessionId, Status = "pending", Timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() };
                    lock (_suggestions) { _suggestions.Insert(0, sug); if (_suggestions.Count > 5000) _suggestions.RemoveRange(5000, _suggestions.Count - 5000); }

                    var outObj = new { id = sug.Id, status = sug.Status };
                    var bytes = JsonSerializer.SerializeToUtf8Bytes(outObj);
                    await resp.OutputStream.WriteAsync(bytes, 0, bytes.Length);
                    resp.StatusCode = 200;
                }
                // Admin: list suggestions
                else if (path.StartsWith("/api/suggestions") && req.HttpMethod == "GET")
                {
                    Suggestion[] snapshot;
                    lock (_suggestions) { snapshot = _suggestions.ToArray(); }
                    var bytes = JsonSerializer.SerializeToUtf8Bytes(snapshot);
                    await resp.OutputStream.WriteAsync(bytes, 0, bytes.Length);
                    resp.StatusCode = 200;
                }
                // Admin: verify/accept/reject a suggestion
                else if (path.StartsWith("/api/suggestions/verify") && req.HttpMethod == "POST")
                {
                    using var sr = new System.IO.StreamReader(req.InputStream, req.ContentEncoding);
                    var body = await sr.ReadToEndAsync();
                    var doc = JsonDocument.Parse(string.IsNullOrWhiteSpace(body) ? "{}" : body);
                    string id = doc.RootElement.GetPropertyOrDefault("id", "");
                    bool accept = doc.RootElement.GetPropertyOrDefaultBool("accept", false);

                    Suggestion? s = null;
                    lock (_suggestions) { s = _suggestions.FirstOrDefault(x => x.Id == id); }
                    if (s == null)
                    {
                        resp.StatusCode = 404;
                        var bytes = JsonSerializer.SerializeToUtf8Bytes(new { error = "suggestion_not_found" });
                        await resp.OutputStream.WriteAsync(bytes, 0, bytes.Length);
                    }
                    else
                    {
                        s.Status = accept ? "accepted" : "rejected";

                        // If accepted, optionally apply to the interaction (overwrite answer) or append a note
                        if (accept && !string.IsNullOrEmpty(s.InteractionId))
                        {
                            lock (_interactions)
                            {
                                var it = _interactions.Find(x => x.Id == s.InteractionId);
                                if (it != null)
                                {
                                    it.Answer = s.SuggestionText + " [accepted_suggestion]";
                                }
                            }
                        }

                        var outObj = new { id = s.Id, status = s.Status };
                        var bytes = JsonSerializer.SerializeToUtf8Bytes(outObj);
                        await resp.OutputStream.WriteAsync(bytes, 0, bytes.Length);
                        resp.StatusCode = 200;
                    }
                }
                else if (path.StartsWith("/api/interactions") && req.HttpMethod == "GET")
                {
                    Interaction[] snapshot;
                    lock (_interactions) { snapshot = _interactions.ToArray(); }
                    var bytes = JsonSerializer.SerializeToUtf8Bytes(snapshot);
                    await resp.OutputStream.WriteAsync(bytes, 0, bytes.Length);
                    resp.StatusCode = 200;
                }
                else if (path.StartsWith("/api/reinforce") && req.HttpMethod == "POST")
                {
                    using var sr = new System.IO.StreamReader(req.InputStream, req.ContentEncoding);
                    var body = await sr.ReadToEndAsync();
                    var doc = JsonDocument.Parse(string.IsNullOrWhiteSpace(body) ? "{}" : body);
                    string id = doc.RootElement.GetPropertyOrDefault("id", "");
                    bool positive = doc.RootElement.GetPropertyOrDefaultBool("positive", true);

                    // For now just mark interaction and return OK
                    lock (_interactions)
                    {
                        var it = _interactions.Find(x => x.Id == id);
                        if (it != null)
                        {
                            // Append a note in the answer field for traceability
                            it.Answer += positive ? " [reinforced:+]" : " [reinforced:-]";
                        }
                    }

                    var outObj = new { status = "ok" };
                    var bytes = JsonSerializer.SerializeToUtf8Bytes(outObj);
                    await resp.OutputStream.WriteAsync(bytes, 0, bytes.Length);
                    resp.StatusCode = 200;
                }
                else if (path.StartsWith("/api/retrain_window") && req.HttpMethod == "POST")
                {
                    // receive examples and perform lightweight server-side update (stub)
                    using var sr = new System.IO.StreamReader(req.InputStream, req.ContentEncoding);
                    var body = await sr.ReadToEndAsync();
                    // In a real implementation you'd convert examples to token ids and run training on Program._fractalCell etc.

                    var outObj = new { status = "retrain_started" };
                    var bytes = JsonSerializer.SerializeToUtf8Bytes(outObj);
                    await resp.OutputStream.WriteAsync(bytes, 0, bytes.Length);
                    resp.StatusCode = 200;
                }
                else if (path.StartsWith("/api/train_recent_window") && req.HttpMethod == "POST")
                {
                    // Trigger retrain on recent interactions (stub)
                    Task.Run(() => DoTrainRecentWindow());
                    var outObj = new { status = "training_triggered" };
                    var bytes = JsonSerializer.SerializeToUtf8Bytes(outObj);
                    await resp.OutputStream.WriteAsync(bytes, 0, bytes.Length);
                    resp.StatusCode = 200;
                }
                else
                {
                    Console.WriteLine($"Unhandled request: {req.HttpMethod} {req.Url}");
                    resp.StatusCode = 404;
                    var bytes = JsonSerializer.SerializeToUtf8Bytes(new { error = "not_found", method = req.HttpMethod, path = path });
                    await resp.OutputStream.WriteAsync(bytes, 0, bytes.Length);
                }
            }
            catch (Exception ex)
            {
                try
                {
                    resp.StatusCode = 500;
                    var bytes = JsonSerializer.SerializeToUtf8Bytes(new { error = ex.Message });
                    await resp.OutputStream.WriteAsync(bytes, 0, bytes.Length);
                }
                catch { }
            }
            finally
            {
                try { resp.OutputStream.Close(); } catch { }
            }
        }

    // NEW: Ramanujan extrapolation head for globals
    
    private static void DoTrainRecentWindow()
        {
            // Example stub: take the most recent N interactions and print them.
            Interaction[] snapshot;
            lock (_interactions) { snapshot = _interactions.Take(50).ToArray(); }
            Console.WriteLine($"Training on {snapshot.Length} recent interactions (stub)...");

            // TODO: map to tokens, prepare training batches, and call Program teaching APIs.
            Thread.Sleep(1000);
            Console.WriteLine("Training stub complete.");
        }

        // JsonDocument helpers
        private static string GetPropertyOrDefault(this JsonElement el, string name, string def)
        {
            if (el.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String) return v.GetString()!;
            return def;
        }

        private static bool GetPropertyOrDefaultBool(this JsonElement el, string name, bool def)
        {
            if (el.TryGetProperty(name, out var v))
            {
                if (v.ValueKind == JsonValueKind.True) return true;
                if (v.ValueKind == JsonValueKind.False) return false;
                if (v.ValueKind == JsonValueKind.Number && v.TryGetInt32(out var iv)) return iv != 0;
                if (v.ValueKind == JsonValueKind.String && bool.TryParse(v.GetString(), out var bv)) return bv;
            }
            return def;
        }

        // Helpers to safely query tokenizer
        private static long GetVocabSizeSafe()
        {
            try
            {
                if (Program._wordTokenizerHolder == null) return 0;
                return Program._wordTokenizerHolder.VocabSize;
            }
            catch { return 0; }
        }


        // External tokenizer helper using reflection to utilize Microsoft.ML.Tokenizers if available
        public static class ExternalTokenizer
        {
            public static string[] TryTokenizeStrings(string text)
            {
                try
                {
                    if (string.IsNullOrWhiteSpace(text)) return null;

                    // Heuristic: detect if input looks like code (C# or C++)
                    if (LooksLikeCode(text, out var lang))
                    {
                        return CodeTokenize(text, lang);
                    }

                    // Try to find a loaded assembly that matches Tokenizers
                    var asm = AppDomain.CurrentDomain.GetAssemblies()
                        .FirstOrDefault(a => a.GetName().Name.IndexOf("Tokenizers", StringComparison.OrdinalIgnoreCase) >= 0);

                    if (asm == null)
                    {
                        try { asm = Assembly.Load("Microsoft.ML.Tokenizers"); } catch { }
                        if (asm == null) asm = AppDomain.CurrentDomain.GetAssemblies()
                            .FirstOrDefault(a => a.GetName().Name.IndexOf("Microsoft.ML", StringComparison.OrdinalIgnoreCase) >= 0);
                        // if still null, we fall back to simple whitespace split below
                    }

                    if (asm != null)
                    {
                        foreach (var t in asm.GetTypes())
                        {
                            // look for convenient candidate methods
                            var methods = t.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static);
                            foreach (var m in methods)
                            {
                                var name = m.Name.ToLowerInvariant();
                                if (!(name.Contains("token") || name.Contains("encode") || name.Contains("split"))) continue;

                                var pars = m.GetParameters();
                                if (pars.Length != 1 || pars[0].ParameterType != typeof(string)) continue;

                                object inst = null;
                                if (!m.IsStatic)
                                {
                                    var ctor = t.GetConstructor(Type.EmptyTypes);
                                    if (ctor == null) continue;
                                    inst = Activator.CreateInstance(t);
                                }

                                object? res = null;
                                try { res = m.Invoke(inst, new object[] { text }); } catch { continue; }
                                if (res == null) continue;

                                if (res is IEnumerable<string> es) return es.ToArray();
                                if (res is string[] sa) return sa;
                                if (res is IEnumerable<int> ei) return ei.Select(i => i.ToString()).ToArray();

                                // fallback: use ToString and split
                                var s = res.ToString();
                                if (!string.IsNullOrWhiteSpace(s)) return s.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                            }
                        }
                    }
                }
                catch
                {
                    // ignore and fall through to fallback
                }

                // Fallback: simple tokenization by word-ish regex (keeps identifiers intact)
                var fallback = Regex.Matches(text, @"[A-Za-z_@][A-Za-z0-9_@]*|\d+\.\d+|\d+|\S", RegexOptions.Multiline)
                    .Cast<Match>().Select(m => m.Value).Where(s => !string.IsNullOrWhiteSpace(s)).ToArray();
                return fallback.Length > 0 ? fallback : null;
            }

            private static bool LooksLikeCode(string text, out string lang)
            {
                lang = "csharp"; // default if code-like
                if (text.Contains("#include") || text.Contains("std::") || Regex.IsMatch(text, @"\bprintf\b"))
                {
                    lang = "cpp"; return true;
                }
                if (text.Contains("using ") || text.Contains("namespace ") || text.Contains("Console.") || text.Contains("public ") || text.Contains("class ") || text.Contains("::"))
                {
                    lang = "csharp"; return true;
                }
                // if many symbols typical for code
                int symbolCount = text.Count(c => "{}();<>;=+-/*%&|[]".IndexOf(c) >= 0);
                if (symbolCount > 5 || text.Contains(";\n") || text.Contains("\n{") ) { lang = "csharp"; return true; }
                return false;
            }

            private static string[] CodeTokenize(string text, string lang)
            {
                // Generic code tokenizer: strings, comments, identifiers, numbers, operators, punctuation
                // Regular expression pattern for tokenizing code (escaped for C# string literal)
                var tokenPattern = "@?\\\"(?:\\\\.|[^\\\"\\\\])*\\\"|/\\*.*?\\*/|//.*?$|\\b[0-9]+(?:\\.[0-9]+)?\\b|[A-Za-z_@][A-Za-z0-9_@]*|::|->|<=|>=|==|!=|\\+\\+|--|\\|\\||&&|\\+=|-=|\\*=|/=?|[%&\\|\\^~<>!=+\\-*/?:]+|[{}()\\[\\];,\\.]";

                var matches = Regex.Matches(text, tokenPattern, RegexOptions.Singleline | RegexOptions.Multiline);
                var list = new List<string>(matches.Count);
                foreach (Match m in matches)
                {
                    var v = m.Value;
                    if (string.IsNullOrWhiteSpace(v)) continue;
                    // strip comments
                    if (v.StartsWith("//") || v.StartsWith("/*")) continue;

                    // If the token consists only of punctuation/symbol characters and length > 1,
                    // split it into single-character tokens so symbols like '{' and '}' are emitted separately.
                    if (Regex.IsMatch(v, "^[^\\w\\s]+$") && v.Length > 1)
                    {
                        foreach (var ch in v)
                        {
                            // skip whitespace characters just in case
                            if (char.IsWhiteSpace(ch)) continue;
                            list.Add(ch.ToString());
                        }
                    }
                    else
                    {
                        list.Add(v);
                    }
                }
                return list.ToArray();
            }
        }

    }

