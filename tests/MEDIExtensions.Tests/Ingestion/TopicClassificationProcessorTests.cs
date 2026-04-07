using MEDIExtensions.Ingestion;
using MEDIExtensions.Tests.Utils;
using Microsoft.Extensions.DataIngestion;

namespace MEDIExtensions.Tests.Ingestion;

public class TopicClassificationProcessorTests
{
    private static readonly string[] Taxonomy = ["web", "data", "security", "performance", "architecture"];

    [Fact]
    public void Constructor_ThrowsOnNullChatClient()
    {
        Assert.Throws<ArgumentNullException>("chatClient",
            () => new TopicClassificationProcessor(null!, Taxonomy));
    }

    [Fact]
    public void Constructor_ThrowsOnNullTaxonomy()
    {
        using var client = TestChatClient.WithJsonResponse("{}");
        Assert.Throws<ArgumentNullException>("taxonomy",
            () => new TopicClassificationProcessor(client, null!));
    }

    [Fact]
    public async Task ProcessAsync_ValidPrimary_StoresTopics()
    {
        using var client = TestChatClient.WithJsonResponse(
            """{"primary": "security", "secondary": ["web", "architecture"]}""");
        var processor = new TopicClassificationProcessor(client, Taxonomy);
        var chunks = CreateChunks("content about security");

        var results = await processor.ProcessAsync(chunks).ToListAsync();

        Assert.Single(results);
        Assert.Equal("security", results[0].Metadata["topic_primary"]);
        Assert.Contains("web", (string)results[0].Metadata["topic_secondary"]!);
        Assert.Contains("architecture", (string)results[0].Metadata["topic_secondary"]!);
    }

    [Fact]
    public async Task ProcessAsync_InvalidPrimary_DefaultsToUnknown()
    {
        using var client = TestChatClient.WithJsonResponse(
            """{"primary": "not_in_taxonomy", "secondary": []}""");
        var processor = new TopicClassificationProcessor(client, Taxonomy);
        var chunks = CreateChunks("some content");

        var results = await processor.ProcessAsync(chunks).ToListAsync();

        Assert.Single(results);
        Assert.Equal("unknown", results[0].Metadata["topic_primary"]);
    }

    [Fact]
    public async Task ProcessAsync_LlmThrows_DefaultsToUnknown()
    {
        using var client = TestChatClient.WithException(new InvalidOperationException("fail"));
        var processor = new TopicClassificationProcessor(client, Taxonomy);
        var chunks = CreateChunks("content");

        var results = await processor.ProcessAsync(chunks).ToListAsync();

        Assert.Single(results);
        Assert.Equal("unknown", results[0].Metadata["topic_primary"]);
        Assert.Equal("", results[0].Metadata["topic_secondary"]);
    }

    private static async IAsyncEnumerable<IngestionChunk<string>> CreateChunks(params string[] contents)
    {
        await Task.CompletedTask;
        var doc = new IngestionDocument("test-doc");
        foreach (var content in contents)
        {
            yield return new IngestionChunk<string>(content, doc);
        }
    }
}
