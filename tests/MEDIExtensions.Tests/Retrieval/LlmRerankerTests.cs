using MEDIExtensions.Retrieval;
using MEDIExtensions.Tests.Utils;

namespace MEDIExtensions.Tests.Retrieval;

public class LlmRerankerTests
{
    [Fact]
    public void Constructor_ThrowsOnNullChatClient()
    {
        Assert.Throws<ArgumentNullException>("chatClient", () => new LlmReranker(null!));
    }

    [Fact]
    public async Task ProcessResultsAsync_TwoOrFewerChunks_ReturnsUnchanged()
    {
        using var client = TestChatClient.WithJsonResponse("""{"rankedIndices": [2, 1]}""");
        var reranker = new LlmReranker(client);
        var results = CreateResults("chunk1", "chunk2");

        var output = await reranker.ProcessResultsAsync(results);

        Assert.Equal(2, output.Chunks.Count);
        Assert.Equal("chunk1", output.Chunks[0].Content);
    }

    [Fact]
    public async Task ProcessResultsAsync_ReranksBasedOnLlmResponse()
    {
        using var client = TestChatClient.WithJsonResponse("""{"rankedIndices": [3, 1, 2]}""");
        var reranker = new LlmReranker(client);
        var results = CreateResults("A", "B", "C");

        var output = await reranker.ProcessResultsAsync(results);

        Assert.Equal("C", output.Chunks[0].Content);
        Assert.Equal("A", output.Chunks[1].Content);
        Assert.Equal("B", output.Chunks[2].Content);
    }

    [Fact]
    public async Task ProcessResultsAsync_FiltersOutOfRangeIndices()
    {
        using var client = TestChatClient.WithJsonResponse("""{"rankedIndices": [99, 1, -1, 2]}""");
        var reranker = new LlmReranker(client);
        var results = CreateResults("A", "B", "C");

        var output = await reranker.ProcessResultsAsync(results);

        Assert.Equal(2, output.Chunks.Count);
        Assert.Equal("A", output.Chunks[0].Content);
        Assert.Equal("B", output.Chunks[1].Content);
    }

    [Fact]
    public async Task ProcessResultsAsync_DeduplicatesIndices()
    {
        using var client = TestChatClient.WithJsonResponse("""{"rankedIndices": [2, 2, 2, 1]}""");
        var reranker = new LlmReranker(client);
        var results = CreateResults("A", "B", "C");

        var output = await reranker.ProcessResultsAsync(results);

        Assert.Equal(2, output.Chunks.Count);
        Assert.Equal("B", output.Chunks[0].Content);
        Assert.Equal("A", output.Chunks[1].Content);
    }

    [Fact]
    public async Task ProcessResultsAsync_FallsBackToTopCandidatesOnInvalidJson()
    {
        using var client = TestChatClient.WithJsonResponse("not valid json");
        var reranker = new LlmReranker(client);
        var results = CreateResults("A", "B", "C");

        var output = await reranker.ProcessResultsAsync(results);

        // Fallback: top MaxResults candidates returned as-is
        Assert.True(output.Chunks.Count > 0);
        Assert.Equal("A", output.Chunks[0].Content);
    }

    [Fact]
    public async Task ProcessResultsAsync_FallsBackToTopCandidatesOnException()
    {
        using var client = TestChatClient.WithException(new InvalidOperationException("LLM down"));
        var reranker = new LlmReranker(client);
        var results = CreateResults("A", "B", "C");

        var output = await reranker.ProcessResultsAsync(results);

        Assert.True(output.Chunks.Count > 0);
        Assert.Equal("A", output.Chunks[0].Content);
    }

    [Fact]
    public async Task ProcessResultsAsync_SetsRerankedMetadata()
    {
        using var client = TestChatClient.WithJsonResponse("""{"rankedIndices": [1, 2, 3]}""");
        var reranker = new LlmReranker(client);
        var results = CreateResults("A", "B", "C");

        var output = await reranker.ProcessResultsAsync(results);

        Assert.True((bool)output.Metadata["reranked"]!);
        Assert.Equal(3, (int)output.Metadata["reranked_count"]!);
    }

    private static RetrievalResults CreateResults(params string[] contents)
    {
        var query = new RetrievalQuery("test query");
        return new RetrievalResults
        {
            Query = query,
            Chunks = contents.Select((c, i) => new RetrievalChunk
            {
                Content = c,
                Score = 1.0 - i * 0.1
            }).ToList()
        };
    }
}
