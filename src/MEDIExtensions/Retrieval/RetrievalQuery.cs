using System.Collections.Generic;

namespace MEDIExtensions.Retrieval;

/// <summary>
/// A query flowing through the retrieval pipeline, potentially expanded into variants.
/// </summary>
public sealed class RetrievalQuery
{
    /// <summary>
    /// The original user query.
    /// </summary>
    public string Original { get; }

    /// <summary>
    /// Query variants produced by pre-search processors (e.g., multi-query expansion).
    /// Defaults to a single variant matching the original query.
    /// </summary>
    public IList<string> Variants { get; set; }

    /// <summary>
    /// Metadata attached by processors for downstream use.
    /// </summary>
    public IDictionary<string, object?> Metadata { get; } = new Dictionary<string, object?>();

    public RetrievalQuery(string original)
    {
        Original = original;
        Variants = new List<string> { original };
    }
}
