using MEDIExtensions.Retrieval;
using MEDIExtensions.Tests.Utils;

namespace MEDIExtensions.Tests.Retrieval;

public class CragValidatorTests
{
    [Fact]
    public void Constructor_ThrowsOnNullChatClient()
    {
        Assert.Throws<ArgumentNullException>("chatClient", () => new CragValidator(null!));
    }

    [Fact]
    public async Task ProcessResultsAsync_EmptyChunks_ReturnsIncorrectPath()
    {
        using var client = TestChatClient.WithJsonResponse("""{"score": 5}""");
        var validator = new CragValidator(client);
        var results = new RetrievalResults
        {
            Query = new RetrievalQuery("test"),
            Chunks = []
        };

        var output = await validator.ProcessResultsAsync(results);

        Assert.Equal(0, output.Metadata["crag_score"]);
        Assert.Equal("Incorrect", output.Metadata["crag_path"]);
        Assert.True((bool)output.Metadata["low_confidence"]!);
    }

    [Fact]
    public async Task ProcessResultsAsync_HighScore_ReturnsCorrectPath()
    {
        using var client = TestChatClient.WithJsonResponse("""{"score": 5, "reasoning": "great match"}""");
        var validator = new CragValidator(client);
        var results = CreateResultsWithChunks(3);

        var output = await validator.ProcessResultsAsync(results);

        Assert.Equal(5, output.Metadata["crag_score"]);
        Assert.Equal("Correct", output.Metadata["crag_path"]);
        Assert.False(output.Metadata.ContainsKey("low_confidence"));
    }

    [Fact]
    public async Task ProcessResultsAsync_MidScore_ReturnsAmbiguousPath()
    {
        using var client = TestChatClient.WithJsonResponse("""{"score": 3, "reasoning": "partial match"}""");
        var validator = new CragValidator(client);
        var results = CreateResultsWithChunks(3);

        var output = await validator.ProcessResultsAsync(results);

        Assert.Equal(3, output.Metadata["crag_score"]);
        Assert.Equal("Ambiguous", output.Metadata["crag_path"]);
        Assert.True((bool)output.Metadata["needs_followup"]!);
    }

    [Fact]
    public async Task ProcessResultsAsync_LowScore_ReturnsIncorrectAndClearsChunks()
    {
        using var client = TestChatClient.WithJsonResponse("""{"score": 1, "reasoning": "off topic"}""");
        var validator = new CragValidator(client);
        var results = CreateResultsWithChunks(3);

        var output = await validator.ProcessResultsAsync(results);

        Assert.Equal(1, output.Metadata["crag_score"]);
        Assert.Equal("Incorrect", output.Metadata["crag_path"]);
        Assert.True((bool)output.Metadata["low_confidence"]!);
        Assert.Empty(output.Chunks);
    }

    [Fact]
    public async Task ProcessResultsAsync_ScoreOutOfRange_ClampedToDefault()
    {
        using var client = TestChatClient.WithJsonResponse("""{"score": 99}""");
        var validator = new CragValidator(client);
        var results = CreateResultsWithChunks(3);

        var output = await validator.ProcessResultsAsync(results);

        // Score 99 is not in [1,5] → defaults to 3 (Ambiguous)
        Assert.Equal(3, output.Metadata["crag_score"]);
        Assert.Equal("Ambiguous", output.Metadata["crag_path"]);
    }

    [Fact]
    public async Task ProcessResultsAsync_LlmThrows_DefaultsToAmbiguous()
    {
        using var client = TestChatClient.WithException(new InvalidOperationException("fail"));
        var validator = new CragValidator(client);
        var results = CreateResultsWithChunks(3);

        var output = await validator.ProcessResultsAsync(results);

        Assert.Equal(3, output.Metadata["crag_score"]);
        Assert.Equal("Ambiguous", output.Metadata["crag_path"]);
    }

    [Fact]
    public async Task ProcessResultsAsync_StoresReasoningInMetadata()
    {
        using var client = TestChatClient.WithJsonResponse("""{"score": 4, "reasoning": "relevant results"}""");
        var validator = new CragValidator(client);
        var results = CreateResultsWithChunks(3);

        var output = await validator.ProcessResultsAsync(results);

        Assert.Equal("relevant results", output.Metadata["crag_reasoning"]);
    }

    private static RetrievalResults CreateResultsWithChunks(int count)
    {
        return new RetrievalResults
        {
            Query = new RetrievalQuery("test query"),
            Chunks = Enumerable.Range(0, count)
                .Select(i => new RetrievalChunk { Content = $"Chunk {i} content", Score = 0.9 - i * 0.1 })
                .ToList()
        };
    }
}
