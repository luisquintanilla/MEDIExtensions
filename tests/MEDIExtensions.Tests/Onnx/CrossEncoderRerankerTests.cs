using MEDIExtensions.Onnx;
using MEDIExtensions.Retrieval;

namespace MEDIExtensions.Tests.Onnx;

public class CrossEncoderRerankerTests
{
    // --- Constructor Validation ---

    [Fact]
    public void Constructor_ThrowsOnNullOptions()
    {
        Assert.Throws<ArgumentNullException>(() => new CrossEncoderReranker(null!));
    }

    [Fact]
    public void Constructor_ThrowsOnEmptyModelPath()
    {
        var options = new CrossEncoderRerankerOptions
        {
            ModelPath = "",
            TokenizerPath = "some/path"
        };

        Assert.Throws<ArgumentException>(() => new CrossEncoderReranker(options));
    }

    [Fact]
    public void Constructor_ThrowsOnEmptyTokenizerPath()
    {
        var options = new CrossEncoderRerankerOptions
        {
            ModelPath = "some/model.onnx",
            TokenizerPath = ""
        };

        Assert.Throws<ArgumentException>(() => new CrossEncoderReranker(options));
    }

    [Fact]
    public void Constructor_AcceptsValidOptions()
    {
        var options = new CrossEncoderRerankerOptions
        {
            ModelPath = "model.onnx",
            TokenizerPath = "tokenizer/"
        };

        var reranker = new CrossEncoderReranker(options);
        Assert.NotNull(reranker);
    }

    // --- Options Defaults ---

    [Fact]
    public void Options_HasCorrectDefaults()
    {
        var options = new CrossEncoderRerankerOptions
        {
            ModelPath = "model.onnx",
            TokenizerPath = "tokenizer/"
        };

        Assert.Equal(512, options.MaxTokenLength);
        Assert.Equal(32, options.BatchSize);
        Assert.Null(options.GpuDeviceId);
        Assert.False(options.FallbackToCpu);
        Assert.Equal(5, options.MaxResults);
        Assert.Equal(["logits", "output"], options.PreferredOutputNames);
    }

    // --- ProcessResultsAsync Edge Cases ---

    [Fact]
    public async Task ProcessResultsAsync_EmptyChunks_ReturnsUnchanged()
    {
        var options = new CrossEncoderRerankerOptions
        {
            ModelPath = "model.onnx",
            TokenizerPath = "tokenizer/"
        };
        var reranker = new CrossEncoderReranker(options);

        var results = new RetrievalResults
        {
            Query = new RetrievalQuery("test query"),
            Chunks = []
        };

        var output = await reranker.ProcessResultsAsync(results);
        Assert.Empty(output.Chunks);
    }

    [Fact]
    public async Task ProcessResultsAsync_SingleChunk_ReturnsUnchanged()
    {
        var options = new CrossEncoderRerankerOptions
        {
            ModelPath = "model.onnx",
            TokenizerPath = "tokenizer/"
        };
        var reranker = new CrossEncoderReranker(options);

        var chunk = new RetrievalChunk { Content = "only chunk", Score = 0.8 };
        var results = new RetrievalResults
        {
            Query = new RetrievalQuery("test query"),
            Chunks = [chunk]
        };

        var output = await reranker.ProcessResultsAsync(results);
        Assert.Single(output.Chunks);
        Assert.Same(chunk, output.Chunks[0]);
    }

    // --- RerankAsync Edge Cases ---

    [Fact]
    public async Task RerankAsync_EmptyResults_ReturnsEmptyList()
    {
        var options = new CrossEncoderRerankerOptions
        {
            ModelPath = "model.onnx",
            TokenizerPath = "tokenizer/"
        };
        var reranker = new CrossEncoderReranker(options);

        var result = await reranker.RerankAsync("query", [], topK: 5);
        Assert.Empty(result);
    }

    // --- Interface Conformance ---

    [Fact]
    public void ImplementsRetrievalResultProcessor()
    {
        var options = new CrossEncoderRerankerOptions
        {
            ModelPath = "model.onnx",
            TokenizerPath = "tokenizer/"
        };
        var reranker = new CrossEncoderReranker(options);

        Assert.IsAssignableFrom<RetrievalResultProcessor>(reranker);
    }

    [Fact]
    public void ImplementsISearchReranker()
    {
        var options = new CrossEncoderRerankerOptions
        {
            ModelPath = "model.onnx",
            TokenizerPath = "tokenizer/"
        };
        var reranker = new CrossEncoderReranker(options);

        Assert.IsAssignableFrom<ISearchReranker<RetrievalChunk>>(reranker);
    }

    [Fact]
    public void ImplementsIDisposable()
    {
        var options = new CrossEncoderRerankerOptions
        {
            ModelPath = "model.onnx",
            TokenizerPath = "tokenizer/"
        };
        var reranker = new CrossEncoderReranker(options);

        Assert.IsAssignableFrom<IDisposable>(reranker);
        reranker.Dispose(); // Should not throw
    }

    // --- Name ---

    [Fact]
    public void Name_DefaultsToTypeName()
    {
        var options = new CrossEncoderRerankerOptions
        {
            ModelPath = "model.onnx",
            TokenizerPath = "tokenizer/"
        };
        var reranker = new CrossEncoderReranker(options);

        Assert.Equal("CrossEncoderReranker", reranker.Name);
    }

    // --- Dispose ---

    [Fact]
    public void Dispose_CanBeCalledMultipleTimes()
    {
        var options = new CrossEncoderRerankerOptions
        {
            ModelPath = "model.onnx",
            TokenizerPath = "tokenizer/"
        };
        var reranker = new CrossEncoderReranker(options);

        reranker.Dispose();
        reranker.Dispose(); // Should not throw
    }

    [Fact]
    public async Task RerankAsync_AfterDispose_ThrowsObjectDisposed()
    {
        var options = new CrossEncoderRerankerOptions
        {
            ModelPath = "model.onnx",
            TokenizerPath = "tokenizer/"
        };
        var reranker = new CrossEncoderReranker(options);
        reranker.Dispose();

        var chunks = new List<RetrievalChunk>
        {
            new() { Content = "a", Score = 0.5 },
            new() { Content = "b", Score = 0.3 }
        };

        await Assert.ThrowsAsync<ObjectDisposedException>(
            () => reranker.RerankAsync("query", chunks, topK: 2));
    }
}
