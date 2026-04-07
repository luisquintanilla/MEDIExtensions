using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace MEDIExtensions.Retrieval;

/// <summary>
/// Reranking contract for search results. Decouples the reranking strategy from the pipeline.
/// </summary>
/// <typeparam name="TRecord">The vector store record type (e.g., IngestedChunk).</typeparam>
/// <remarks>
/// Implementations:
/// - <c>LlmReranker</c>: uses IChatClient to judge relevance (500-2000ms, high quality)
/// - <c>CrossEncoderReranker</c>: uses ONNX cross-encoder model (50-200ms, good quality)
/// </remarks>
public interface ISearchReranker<TRecord>
{
    /// <summary>
    /// Re-scores and reorders search results by relevance to the query.
    /// </summary>
    /// <param name="query">The original user query.</param>
    /// <param name="results">The search results to rerank.</param>
    /// <param name="topK">Maximum number of results to return after reranking.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Reranked results, ordered by new relevance score descending.</returns>
    Task<IReadOnlyList<RerankedResult<TRecord>>> RerankAsync(
        string query,
        IReadOnlyList<TRecord> results,
        int topK = 5,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// A reranked result with the original record and its new relevance score.
/// </summary>
public sealed class RerankedResult<TRecord>
{
    /// <summary>The original record from the vector store.</summary>
    public required TRecord Record { get; init; }

    /// <summary>Relevance score assigned by the reranker (0.0 to 1.0).</summary>
    public required double RelevanceScore { get; init; }
}
