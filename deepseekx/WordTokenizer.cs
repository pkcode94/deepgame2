using System;
using System.Collections.Generic;
using System.Linq;

public class WordTokenizer
{
    private readonly Dictionary<string, int> stoi = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
    private readonly List<string> itos = new List<string>();

    // reserve 0 for <unk>
    public int VocabSize => itos.Count;
    public void InitializePhysicsVocab()
    {
        string[] physicsVocab = { "[", "]", "peak", "high", "mid", "low", "flux", "horizon", "chirp" };
        foreach (var word in physicsVocab) AddWord(word);
    }
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
    public WordTokenizer()
    {
    }

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
                itos.Add(kv.Key);
                stoi[kv.Key] = itos.Count - 1;
            }
        }
    }

    public int Encode(string word)
    {
        if (string.IsNullOrEmpty(word)) return 0;
        var w = word.ToLowerInvariant();
        if (stoi.TryGetValue(w, out var id)) return id;
        return 0; // unknown
    }

    public string Decode(int token)
    {
        if (token < 0 || token >= itos.Count) return "0";
        return itos[token];
    }

    public int[] Tokenize(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return Array.Empty<int>();
        var parts = text.Split(new[] { ' ', '\t', '\n', '\r', ',', '.', '!', '?', ';', ':', '"', '\'' }, StringSplitOptions.RemoveEmptyEntries);
        var ids = new List<int>(parts.Length);
        foreach (var p in parts)
        {
            ids.Add(Encode(p));
        }
        return ids.ToArray();
    }

    public string Detokenize(IEnumerable<int> tokens)
    {
        return string.Join(' ', tokens.Select(t => Decode(t)));
    }

    // allow adding words manually
    public int AddWord(string word)
    {
        var w = word.ToLowerInvariant();
        if (stoi.TryGetValue(w, out var id)) return id;
        itos.Add(w);
        id = itos.Count - 1;
        stoi[w] = id;
        return id;
    }
}
