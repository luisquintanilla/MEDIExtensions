using MEDIExtensions.Ingestion;
using MEDIExtensions.Tests.Utils;
using Microsoft.Extensions.DataIngestion;

namespace MEDIExtensions.Tests.Ingestion;

public class HypotheticalQueryProcessorTests
{
    [Fact]
    public void Constructor_ThrowsOnNullChatClient()
    {
        Assert.Throws<ArgumentNullException>("chatClient", () => new HypotheticalQueryProcessor(null!));
    }

    [Fact]
    public async Task ProcessAsync_YieldsOriginalThenQuestions()
    {
        using var client = TestChatClient.WithJsonResponse(
            """{"questions": ["What is dependency injection?", "How do you configure DI in .NET?", "What are the benefits of DI?"]}""");
        var processor = new HypotheticalQueryProcessor(client, questionsPerChunk: 3);
        var chunks = CreateChunks("Dependency injection is a design pattern used in .NET.");

        var results = await processor.ProcessAsync(chunks).ToListAsync();

        // First chunk should be original
        Assert.Equal("original", results[0].Metadata["chunk_type"]);
        Assert.Equal("Dependency injection is a design pattern used in .NET.", results[0].Content);

        // Remaining should be hypothetical queries
        var questionChunks = results.Where(c => (string)c.Metadata["chunk_type"]! == "hypothetical_query").ToList();
        Assert.Equal(3, questionChunks.Count);
        Assert.All(questionChunks, q => Assert.NotEmpty(q.Metadata["parent_chunk_id"]!.ToString()!));
    }

    [Fact]
    public async Task ProcessAsync_FiltersShortQuestions()
    {
        using var client = TestChatClient.WithJsonResponse(
            """{"questions": ["short?", "", "This is a valid question that is long enough?"]}""");
        var processor = new HypotheticalQueryProcessor(client, questionsPerChunk: 3);
        var chunks = CreateChunks("Content about something.");

        var results = await processor.ProcessAsync(chunks).ToListAsync();

        var questionChunks = results.Where(c => (string)c.Metadata["chunk_type"]! == "hypothetical_query").ToList();
        Assert.Single(questionChunks); // Only the long-enough question passes
    }

    [Fact]
    public async Task ProcessAsync_LlmThrows_YieldsOriginalOnly()
    {
        using var client = TestChatClient.WithException(new InvalidOperationException("fail"));
        var processor = new HypotheticalQueryProcessor(client);
        var chunks = CreateChunks("Some content here.");

        var results = await processor.ProcessAsync(chunks).ToListAsync();

        Assert.Single(results);
        Assert.Equal("original", results[0].Metadata["chunk_type"]);
        Assert.Equal("Some content here.", results[0].Content);
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
