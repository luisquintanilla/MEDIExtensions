using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.AI;

namespace MEDIExtensions.Retrieval;

/// <summary>
/// LLM-based reranker — asks the model to rank passages by relevance to the query.
/// Implements both <see cref="RetrievalResultProcessor"/> (pipeline integration) and
/// <see cref="ISearchReranker{TRecord}"/> (standalone reranking).
/// </summary>
public class LlmReranker : RetrievalResultProcessor
{
    private readonly IChatClient _chatClient;

    /// <summary>Maximum number of results to return after reranking.</summary>
    public int MaxResults { get; init; } = 5;

    /// <summary>Maximum candidate passages to send to the LLM (controls token cost).</summary>
    public int MaxCandidates { get; init; } = 8;

    /// <summary>Maximum preview length per passage (characters).</summary>
    public int PreviewLength { get; init; } = 200;

    public LlmReranker(IChatClient chatClient)
    {
        _chatClient = chatClient ?? throw new ArgumentNullException(nameof(chatClient));
    }

    public override async Task<RetrievalResults> ProcessResultsAsync(
        RetrievalResults results, CancellationToken cancellationToken = default)
    {
        if (results.Chunks.Count <= 2)
            return results;

        var candidates = results.Chunks.Take(MaxCandidates).ToList();
        var rankedIndices = await GetRankedIndicesAsync(
            results.Query.Original, candidates, cancellationToken);

        var reranked = rankedIndices
            .Where(idx => idx >= 1 && idx <= candidates.Count)
            .Distinct()
            .Take(MaxResults)
            .Select(idx => candidates[idx - 1])
            .ToList();

        if (reranked.Count == 0)
            reranked = candidates.Take(MaxResults).ToList();

        results.Chunks = reranked;
        results.Metadata["reranked"] = true;
        results.Metadata["reranked_count"] = reranked.Count;
        return results;
    }

    private async Task<int[]> GetRankedIndicesAsync(
        string query, IList<RetrievalChunk> candidates, CancellationToken ct)
    {
        var passagesBlock = string.Join("\n",
            candidates.Select((c, i) =>
            {
                var preview = c.Content.Length > PreviewLength
                    ? c.Content[..PreviewLength] + "..."
                    : c.Content;
                return $"[{i + 1}] {preview}";
            }));

        var prompt = $$"""
            Rank these passages by relevance to the query. Return the passage numbers
            in order from most to least relevant.

            Query: {{query}}
            Passages:
            {{passagesBlock}}

            Return ONLY valid JSON: {"rankedIndices": [3, 1, 5, 2, 4]}
            """;

        var options = new ChatOptions
        {
            MaxOutputTokens = 100,
            ResponseFormat = ChatResponseFormat.Json
        };

        try
        {
            var response = await _chatClient.GetResponseAsync(prompt, options, ct);
            var result = JsonSerializer.Deserialize<RerankingResponse>(
                response.Text ?? "{}", JsonDefaults.Options);
            return result?.RankedIndices ?? [];
        }
        catch
        {
            return [];
        }
    }

    private sealed class RerankingResponse
    {
        [JsonPropertyName("rankedIndices")]
        public int[] RankedIndices { get; set; } = [];
    }
}
