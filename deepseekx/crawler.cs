using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace AdversarialFractalGame
{
    public class RecurrenceAwareCrawler
    {
        private readonly HttpClient _http;
        private readonly TimeSpan _politenessDelay;

        public class UrlState
        {
            public Uri Uri { get; set; } = null!;
            public int VisitCount { get; set; }
            public string? LastContentHash { get; set; }
            public DateTime LastVisited { get; set; }
            public int StreakUnchanged { get; set; }
        }

        private class QueueItem
        {
            public Uri Uri { get; set; } = null!;
            public int Depth { get; set; }
            public DateTime Due { get; set; }
        }

        public RecurrenceAwareCrawler(TimeSpan? politeness = null)
        {
            _http = new HttpClient();
            _politenessDelay = politeness ?? TimeSpan.FromSeconds(1);
        }

        // Crawl starting from seed URLs until cancellation or maxPages reached. Respects "recurrence" by scheduling
        // re-visits later when content is observed unchanged, and scheduling sooner when content changes.
        public async Task<IDictionary<string, UrlState>> CrawlAsync(IEnumerable<string> seeds, int maxPages = 100, int maxDepth = 2, CancellationToken cancellation = default)
        {
            var visited = new ConcurrentDictionary<string, UrlState>(StringComparer.OrdinalIgnoreCase);
            var pq = new PriorityQueue<QueueItem, DateTime>();

            foreach (var s in seeds)
            {
                if (TryNormalize(s, out var u)) pq.Enqueue(new QueueItem { Uri = u, Depth = 0, Due = DateTime.UtcNow }, DateTime.UtcNow);
            }

            int processed = 0;

            while (pq.Count > 0 && processed < maxPages && !cancellation.IsCancellationRequested)
            {
                var next = pq.Dequeue();
                if (next.Due > DateTime.UtcNow)
                {
                    // Sleep until due or cancellation
                    var wait = next.Due - DateTime.UtcNow;
                    try { await Task.Delay(wait, cancellation); } catch (TaskCanceledException) { break; }
                }

                var url = next.Uri;
                var key = url.AbsoluteUri;

                // Politeness
                await Task.Delay(_politenessDelay, cancellation);

                string? content = null;
                try
                {
                    using var resp = await _http.GetAsync(url, cancellation);
                    if (!resp.IsSuccessStatusCode) continue;
                    content = await resp.Content.ReadAsStringAsync(cancellation);
                }
                catch
                {
                    // transient network error: schedule a retry later
                    var retryDue = DateTime.UtcNow + TimeSpan.FromMinutes(5);
                    pq.Enqueue(new QueueItem { Uri = url, Depth = next.Depth, Due = retryDue }, retryDue);
                    continue;
                }

                processed++;

                var hash = ComputeHash(content ?? "");

                var state = visited.GetOrAdd(key, _ => new UrlState { Uri = url, VisitCount = 0, LastContentHash = null, LastVisited = DateTime.MinValue, StreakUnchanged = 0 });

                bool changed = state.LastContentHash is null || state.LastContentHash != hash;

                state.VisitCount++;
                state.LastVisited = DateTime.UtcNow;

                if (changed)
                {
                    state.StreakUnchanged = 0;
                    state.LastContentHash = hash;
                    Console.WriteLine($"[CHANGED] {key} (visit #{state.VisitCount})");
                }
                else
                {
                    state.StreakUnchanged++;
                    Console.WriteLine($"[UNCHANGED] {key} (streak {state.StreakUnchanged})");
                }

                // Decide next revisit delay based on recurrence (unchanged streak)
                TimeSpan nextDelay;
                if (state.StreakUnchanged == 0)
                {
                    // content changed recently -> revisit sooner
                    nextDelay = TimeSpan.FromMinutes(1);
                }
                else
                {
                    // exponential backoff up to a cap
                    var minutes = Math.Min(60, Math.Pow(2, state.StreakUnchanged));
                    nextDelay = TimeSpan.FromMinutes(minutes);
                }

                var nextDue = DateTime.UtcNow + nextDelay;
                // enqueue next revisit
                pq.Enqueue(new QueueItem { Uri = url, Depth = next.Depth, Due = nextDue }, nextDue);

                // extract links and schedule new discoveries
                if (next.Depth < maxDepth && content is not null)
                {
                    foreach (var link in ExtractLinks(content))
                    {
                        if (!TryNormalize(link, out var child)) continue;
                        // stay within same host as seed or allow all? keep same host for safety
                        if (!string.Equals(child.Host, url.Host, StringComparison.OrdinalIgnoreCase)) continue;
                        var ckey = child.AbsoluteUri;
                        if (visited.ContainsKey(ckey)) continue;
                        // schedule immediate discovery
                        pq.Enqueue(new QueueItem { Uri = child, Depth = next.Depth + 1, Due = DateTime.UtcNow }, DateTime.UtcNow);
                    }
                }
            }

            return visited.ToDictionary(kv => kv.Key, kv => kv.Value);
        }

        private static string ComputeHash(string content)
        {
            using var sha = SHA256.Create();
            var bytes = Encoding.UTF8.GetBytes(content ?? "");
            var h = sha.ComputeHash(bytes);
            return Convert.ToBase64String(h);
        }

        private static IEnumerable<string> ExtractLinks(string html)
        {
            // Simple href extractor. For robust crawling replace with an HTML parser.
            var hrefRegex = new Regex("href\\s*=\\s*\"([^\"]+)\"", RegexOptions.IgnoreCase);
            foreach (Match m in hrefRegex.Matches(html))
            {
                var v = m.Groups[1].Value;
                if (string.IsNullOrWhiteSpace(v)) continue;
                yield return v;
            }
        }

        private static bool TryNormalize(string raw, out Uri uri)
        {
            uri = null!;
            try
            {
                if (raw.StartsWith("//")) raw = "https:" + raw;
                if (!raw.StartsWith("http://", StringComparison.OrdinalIgnoreCase) && !raw.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
                {
                    // relative or protocol-relative; reject here
                    return Uri.TryCreate(raw, UriKind.Absolute, out uri);
                }
                return Uri.TryCreate(raw, UriKind.Absolute, out uri);
            }
            catch
            {
                return false;
            }
        }
    }
}
