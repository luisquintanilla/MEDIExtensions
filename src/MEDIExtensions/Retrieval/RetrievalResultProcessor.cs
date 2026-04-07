using System.Threading;
using System.Threading.Tasks;

namespace MEDIExtensions.Retrieval;

/// <summary>
/// Post-search processor that transforms, filters, or annotates retrieval results.
/// </summary>
/// <remarks>
/// Implementations:
/// - <c>LlmReranker</c>: re-scores chunks via LLM relevance judgement
/// - <c>CragValidator</c>: routes results through correct/ambiguous/incorrect paths
/// - <c>CrossEncoderReranker</c>: ONNX-based cross-encoder scoring (MEDIExtensions.Onnx)
/// </remarks>
public abstract class RetrievalResultProcessor
{
    /// <summary>
    /// Display name for logging and tracing.
    /// </summary>
    public virtual string Name => GetType().Name;

    /// <summary>
    /// Processes retrieval results after search. May reorder, filter, annotate, or augment.
    /// </summary>
    public abstract Task<RetrievalResults> ProcessResultsAsync(
        RetrievalResults results,
        CancellationToken cancellationToken = default);
}
