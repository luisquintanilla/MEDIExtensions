using System.Diagnostics;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.VectorData;

namespace MEDIExtensions.Retrieval;

/// <summary>
/// Retrieval pipeline for MEDI — mirrors <c>IngestionPipeline</c> design.
/// Flow: query → QueryProcessors → vector search → ResultProcessors → results
/// </summary>
/// <remarks>
/// With an empty processor list, this behaves identically to a raw
/// <c>VectorStoreCollection.SearchAsync</c> call. Processors (reranking,
/// CRAG, query expansion, HyDE) are added via DI based on appsettings.json.
/// </remarks>
public sealed class RetrievalPipeline : IDisposable
{
    private readonly ActivitySource _activitySource;
    private readonly ILogger? _logger;

    /// <summary>Pre-search query processors (expansion, HyDE, etc.).</summary>
    public IList<RetrievalQueryProcessor> QueryProcessors { get; } = [];

    /// <summary>Post-search result processors (reranking, CRAG, etc.).</summary>
    public IList<RetrievalResultProcessor> ResultProcessors { get; } = [];

    public RetrievalPipeline(
        RetrievalPipelineOptions? options = null,
        ILoggerFactory? loggerFactory = null)
    {
        _activitySource = new((options ?? new()).ActivitySourceName);
        _logger = loggerFactory?.CreateLogger<RetrievalPipeline>();
    }

    /// <summary>
    /// Executes the retrieval pipeline: query processing → vector search → result processing.
    /// </summary>
    /// <typeparam name="TKey">The vector store key type.</typeparam>
    /// <typeparam name="TRecord">The vector store record type.</typeparam>
    /// <param name="collection">The vector store collection to search.</param>
    /// <param name="query">The user query.</param>
    /// <param name="topK">Maximum results to retrieve per search variant.</param>
    /// <param name="contentSelector">Extracts text content from a record.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task<RetrievalResults> RetrieveAsync<TKey, TRecord>(
        VectorStoreCollection<TKey, TRecord> collection,
        string query,
        int topK = 5,
        Func<TRecord, string>? contentSelector = null,
        CancellationToken cancellationToken = default)
        where TKey : notnull
        where TRecord : class
    {
        using var activity = _activitySource.StartActivity("RetrievalPipeline.Retrieve");
        activity?.SetTag("retrieval.query", query);
        activity?.SetTag("retrieval.topK", topK);

        // 1. Apply query processors
        var processedQuery = new RetrievalQuery(query);
        foreach (var qp in QueryProcessors)
        {
            using var qpActivity = _activitySource.StartActivity($"QueryProcessor.{qp.Name}");
            _logger?.LogDebug("Applying query processor: {Processor}", qp.Name);
            processedQuery = await qp.ProcessQueryAsync(processedQuery, cancellationToken);
        }

        activity?.SetTag("retrieval.variants", processedQuery.Variants.Count);

        // 2. Execute vector search for each variant
        var rawChunks = new List<RetrievalChunk>();
        foreach (var variant in processedQuery.Variants)
        {
            using var searchActivity = _activitySource.StartActivity("VectorSearch");
            searchActivity?.SetTag("retrieval.variant", variant);

            await foreach (var hit in collection.SearchAsync(variant, top: topK, cancellationToken: cancellationToken))
            {
                var content = contentSelector != null && hit.Record is not null
                    ? contentSelector(hit.Record)
                    : hit.Record?.ToString() ?? string.Empty;

                rawChunks.Add(new RetrievalChunk
                {
                    Content = content,
                    Score = hit.Score ?? 0,
                    Record = ExtractRecordProperties(hit.Record)
                });
            }
        }

        // 3. Deduplicate + RRF merge if multiple variants
        var merged = DeduplicateAndMerge(rawChunks, processedQuery.Variants.Count > 1);
        var results = new RetrievalResults
        {
            Query = processedQuery,
            Chunks = merged
        };

        activity?.SetTag("retrieval.rawCount", rawChunks.Count);
        activity?.SetTag("retrieval.mergedCount", merged.Count);

        // 4. Apply result processors (reranking, CRAG, etc.)
        foreach (var rp in ResultProcessors)
        {
            using var rpActivity = _activitySource.StartActivity($"ResultProcessor.{rp.Name}");
            _logger?.LogDebug("Applying result processor: {Processor}", rp.Name);
            results = await rp.ProcessResultsAsync(results, cancellationToken);
        }

        activity?.SetTag("retrieval.finalCount", results.Chunks.Count);
        _logger?.LogInformation(
            "Retrieval complete: {Variants} variants, {Raw} raw → {Final} final chunks",
            processedQuery.Variants.Count, rawChunks.Count, results.Chunks.Count);

        return results;
    }

    /// <summary>
    /// Reciprocal Rank Fusion deduplication for multi-variant results.
    /// </summary>
    private static IList<RetrievalChunk> DeduplicateAndMerge(
        List<RetrievalChunk> raw, bool useRrf)
    {
        if (!useRrf)
        {
            return raw
                .GroupBy(c => c.Content)
                .Select(g => g.OrderByDescending(c => c.Score).First())
                .OrderByDescending(c => c.Score)
                .ToList();
        }

        const int k = 60;
        var rrfScores = new Dictionary<string, double>();
        var chunkLookup = new Dictionary<string, RetrievalChunk>();

        int rank = 0;
        foreach (var chunk in raw)
        {
            rank++;
            var key = chunk.Content;
            if (!rrfScores.ContainsKey(key))
            {
                rrfScores[key] = 0;
                chunkLookup[key] = chunk;
            }
            rrfScores[key] += 1.0 / (k + rank);
        }

        return rrfScores
            .OrderByDescending(kv => kv.Value)
            .Select(kv =>
            {
                var chunk = chunkLookup[kv.Key];
                chunk.Score = kv.Value;
                return chunk;
            })
            .ToList();
    }

    private static IDictionary<string, object?> ExtractRecordProperties<TRecord>(TRecord record)
    {
        if (record is IDictionary<string, object?> dict)
            return dict;

        var props = new Dictionary<string, object?>();
        foreach (var prop in typeof(TRecord).GetProperties())
        {
            props[prop.Name] = prop.GetValue(record);
        }
        return props;
    }

    public void Dispose()
    {
        _activitySource.Dispose();
    }
}

/// <summary>Configuration options for the retrieval pipeline.</summary>
public sealed class RetrievalPipelineOptions
{
    /// <summary>
    /// The name of the <see cref="ActivitySource"/> used for OpenTelemetry tracing.
    /// Defaults to "MEDIExtensions.Retrieval".
    /// </summary>
    public string ActivitySourceName { get; set; } = "MEDIExtensions.Retrieval";
}
