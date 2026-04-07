using MEDIExtensions.Ingestion;
using MEDIExtensions.Tests.Utils;
using Microsoft.Extensions.DataIngestion;

namespace MEDIExtensions.Tests.Ingestion;

public class TreeIndexProcessorTests
{
    [Fact]
    public void Constructor_ThrowsOnNullChatClient()
    {
        Assert.Throws<ArgumentNullException>("chatClient", () => new TreeIndexProcessor(null!));
    }

    [Fact]
    public async Task ProcessAsync_YieldsLeavesThenBranchesThenRoot()
    {
        using var client = TestChatClient.WithJsonResponse("""{"summary": "A summary of the content."}""");
        var processor = new TreeIndexProcessor(client);
        var chunks = CreateChunks("doc1.md", "First chunk content.", "Second chunk content.");

        var results = await processor.ProcessAsync(chunks).ToListAsync();

        // Should have: 2 leaves + 1 branch + 1 root = 4 chunks
        var leaves = results.Where(r => r.Metadata.ContainsKey("level") && (int)r.Metadata["level"]! == 0).ToList();
        var branches = results.Where(r => r.Metadata.ContainsKey("level") && (int)r.Metadata["level"]! == 1).ToList();
        var roots = results.Where(r => r.Metadata.ContainsKey("level") && (int)r.Metadata["level"]! == 2).ToList();

        Assert.Equal(2, leaves.Count);
        Assert.Single(branches);
        Assert.Single(roots);
    }

    [Fact]
    public async Task ProcessAsync_SetsLevelMetadata()
    {
        using var client = TestChatClient.WithJsonResponse("""{"summary": "Summary text."}""");
        var processor = new TreeIndexProcessor(client);
        var chunks = CreateChunks("test.md", "Content of the document.");

        var results = await processor.ProcessAsync(chunks).ToListAsync();

        var leaf = results.First(r => (int)r.Metadata["level"]! == 0);
        Assert.Equal("original", leaf.Metadata["chunk_type"]);

        var branch = results.First(r => (int)r.Metadata["level"]! == 1);
        Assert.Equal("branch_summary", branch.Metadata["chunk_type"]);
        Assert.Equal("root", branch.Metadata["parent_id"]);

        var root = results.First(r => (int)r.Metadata["level"]! == 2);
        Assert.Equal("root_summary", root.Metadata["chunk_type"]);
    }

    [Fact]
    public async Task ProcessAsync_SummarizeFails_UsesFallbackText()
    {
        using var client = TestChatClient.WithException(new InvalidOperationException("LLM unavailable"));
        var processor = new TreeIndexProcessor(client);
        var chunks = CreateChunks("test.md", "Some content.");

        var results = await processor.ProcessAsync(chunks).ToListAsync();

        var branches = results.Where(r => r.Metadata.ContainsKey("level") && (int)r.Metadata["level"]! == 1).ToList();
        Assert.Single(branches);
        Assert.Equal("Summary unavailable", branches[0].Content);
    }

    private static async IAsyncEnumerable<IngestionChunk<string>> CreateChunks(
        string docId, params string[] contents)
    {
        await Task.CompletedTask;
        var doc = new IngestionDocument(docId);
        foreach (var content in contents)
        {
            yield return new IngestionChunk<string>(content, doc);
        }
    }
}
