using System.Collections.Generic;

namespace MEDIExtensions.Retrieval;

/// <summary>
/// A single retrieved chunk with its relevance score and raw record data.
/// </summary>
public sealed class RetrievalChunk
{
    /// <summary>The text content of the chunk.</summary>
    public required string Content { get; init; }

    /// <summary>Relevance score (higher is better). Source depends on the pipeline stage.</summary>
    public double Score { get; set; }

    /// <summary>Raw record data from the vector store.</summary>
    public IDictionary<string, object?> Record { get; init; } = new Dictionary<string, object?>();
}

/// <summary>
/// A set of retrieved chunks for a query, potentially annotated by result processors.
/// </summary>
public sealed class RetrievalResults
{
    /// <summary>The query that produced these results.</summary>
    public required RetrievalQuery Query { get; init; }

    /// <summary>The retrieved chunks, ordered by score descending.</summary>
    public IList<RetrievalChunk> Chunks { get; set; } = [];

    /// <summary>Metadata attached by result processors (e.g., CRAG confidence path).</summary>
    public IDictionary<string, object?> Metadata { get; } = new Dictionary<string, object?>();
}
