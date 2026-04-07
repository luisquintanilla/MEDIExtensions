using MEDIExtensions.Retrieval;
using MEDIExtensions.Tests.Utils;

namespace MEDIExtensions.Tests.Retrieval;

public class RetrievalPipelineTests : IDisposable
{
    private readonly RetrievalPipeline _pipeline = new();
    private readonly TestVectorStoreCollection _collection = new();

    public void Dispose() => _pipeline.Dispose();

    [Fact]
    public void Constructor_AcceptsNullOptions()
    {
        using var pipeline = new RetrievalPipeline(options: null, loggerFactory: null);
        Assert.NotNull(pipeline);
    }

    [Fact]
    public async Task RetrieveAsync_NoProcessors_ReturnsRawSearchResults()
    {
        _collection.AddSearchResult("chunk1", 0.9);
        _collection.AddSearchResult("chunk2", 0.8);

        var results = await _pipeline.RetrieveAsync(
            _collection, "test query",
            contentSelector: r => (string)r["content"]!);

        Assert.Equal(2, results.Chunks.Count);
        Assert.Equal("test query", results.Query.Original);
    }

    [Fact]
    public async Task RetrieveAsync_QueryProcessorsAppliedInOrder()
    {
        var order = new List<string>();

        _pipeline.QueryProcessors.Add(new TrackingQueryProcessor("first", order));
        _pipeline.QueryProcessors.Add(new TrackingQueryProcessor("second", order));
        _collection.AddSearchResult("result", 0.9);

        await _pipeline.RetrieveAsync(
            _collection, "test",
            contentSelector: r => (string)r["content"]!);

        Assert.Equal(["first", "second"], order);
    }

    [Fact]
    public async Task RetrieveAsync_ResultProcessorsAppliedInOrder()
    {
        var order = new List<string>();

        _pipeline.ResultProcessors.Add(new TrackingResultProcessor("first", order));
        _pipeline.ResultProcessors.Add(new TrackingResultProcessor("second", order));
        _collection.AddSearchResult("result", 0.9);

        await _pipeline.RetrieveAsync(
            _collection, "test",
            contentSelector: r => (string)r["content"]!);

        Assert.Equal(["first", "second"], order);
    }

    [Fact]
    public async Task RetrieveAsync_SingleVariant_SimpleDedup()
    {
        // Add duplicate content with different scores
        _collection.AddSearchResult("same content", 0.9);
        _collection.AddSearchResult("same content", 0.7);
        _collection.AddSearchResult("unique", 0.8);

        var results = await _pipeline.RetrieveAsync(
            _collection, "test",
            contentSelector: r => (string)r["content"]!);

        // "same content" should be deduplicated, keeping the higher score
        var contents = results.Chunks.Select(c => c.Content).ToList();
        Assert.Equal(2, contents.Count);
        Assert.Contains("same content", contents);
        Assert.Contains("unique", contents);
    }

    [Fact]
    public async Task RetrieveAsync_MultipleVariants_UsesRrfMerge()
    {
        // Query processor that adds a variant
        _pipeline.QueryProcessors.Add(new VariantAddingProcessor());
        _collection.AddSearchResult("result1", 0.9);
        _collection.AddSearchResult("result2", 0.8);

        var results = await _pipeline.RetrieveAsync(
            _collection, "test",
            contentSelector: r => (string)r["content"]!);

        // With 2 variants searching same collection, results get RRF-boosted
        Assert.True(results.Chunks.Count > 0);
    }

    [Fact]
    public async Task RetrieveAsync_NullContentSelector_UsesToString()
    {
        _collection.AddSearchResult("content", 0.9);

        var results = await _pipeline.RetrieveAsync<string, Dictionary<string, object?>>(
            _collection, "test");

        // Uses ToString() on the dictionary record
        Assert.Single(results.Chunks);
    }

    [Fact]
    public void Dispose_DisposesWithoutError()
    {
        var pipeline = new RetrievalPipeline();
        pipeline.Dispose(); // Should not throw
    }

    private sealed class TrackingQueryProcessor(string name, List<string> order) : RetrievalQueryProcessor
    {
        public override string Name => name;

        public override Task<RetrievalQuery> ProcessQueryAsync(
            RetrievalQuery query, CancellationToken cancellationToken = default)
        {
            order.Add(name);
            return Task.FromResult(query);
        }
    }

    private sealed class TrackingResultProcessor(string name, List<string> order) : RetrievalResultProcessor
    {
        public override string Name => name;

        public override Task<RetrievalResults> ProcessResultsAsync(
            RetrievalResults results, CancellationToken cancellationToken = default)
        {
            order.Add(name);
            return Task.FromResult(results);
        }
    }

    private sealed class VariantAddingProcessor : RetrievalQueryProcessor
    {
        public override Task<RetrievalQuery> ProcessQueryAsync(
            RetrievalQuery query, CancellationToken cancellationToken = default)
        {
            query.Variants = [query.Original, "variant of " + query.Original];
            return Task.FromResult(query);
        }
    }
}
