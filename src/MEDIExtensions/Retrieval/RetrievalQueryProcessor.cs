using System.Threading;
using System.Threading.Tasks;

namespace MEDIExtensions.Retrieval;

/// <summary>
/// Pre-search processor that transforms or expands a retrieval query.
/// Mirrors the MEDI IngestionChunkProcessor pattern for the retrieval side.
/// </summary>
/// <remarks>
/// Implementations:
/// - <c>MultiQueryExpander</c>: generates N query variants + RRF merge
/// - <c>HydeQueryTransformer</c>: rewrites query as hypothetical document embedding
/// </remarks>
public abstract class RetrievalQueryProcessor
{
    /// <summary>
    /// Display name for logging and tracing.
    /// </summary>
    public virtual string Name => GetType().Name;

    /// <summary>
    /// Transforms or expands the query before search is executed.
    /// </summary>
    public abstract Task<RetrievalQuery> ProcessQueryAsync(
        RetrievalQuery query,
        CancellationToken cancellationToken = default);
}
