using MEDIExtensions.Ingestion;
using MEDIExtensions.Tests.Utils;
using Microsoft.Extensions.DataIngestion;

namespace MEDIExtensions.Tests.Ingestion;

public class EntityExtractionProcessorTests
{
    [Fact]
    public void Constructor_ThrowsOnNullChatClient()
    {
        Assert.Throws<ArgumentNullException>("chatClient", () => new EntityExtractionProcessor(null!));
    }

    [Fact]
    public async Task ProcessAsync_ExtractsEntitiesFromChunks()
    {
        using var client = TestChatClient.WithJsonResponse(
            """{"people": ["John Doe"], "organizations": ["Microsoft"], "technologies": ["ASP.NET"], "versions": ["8.0"]}""");
        var processor = new EntityExtractionProcessor(client);
        var chunks = CreateChunks("John Doe from Microsoft works on ASP.NET 8.0");

        var results = await processor.ProcessAsync(chunks).ToListAsync();

        Assert.Single(results);
        Assert.Equal("John Doe", results[0].Metadata["entities_people"]);
        Assert.Equal("Microsoft", results[0].Metadata["entities_organizations"]);
        Assert.Equal("ASP.NET", results[0].Metadata["entities_technologies"]);
        Assert.Equal("8.0", results[0].Metadata["entities_versions"]);
    }

    [Fact]
    public async Task ProcessAsync_LlmThrows_SetsEmptyEntities()
    {
        using var client = TestChatClient.WithException(new InvalidOperationException("fail"));
        var processor = new EntityExtractionProcessor(client);
        var chunks = CreateChunks("some content");

        var results = await processor.ProcessAsync(chunks).ToListAsync();

        Assert.Single(results);
        Assert.Equal("", results[0].Metadata["entities_people"]);
        Assert.Equal("", results[0].Metadata["entities_organizations"]);
        Assert.Equal("", results[0].Metadata["entities_technologies"]);
        Assert.Equal("", results[0].Metadata["entities_versions"]);
    }

    [Fact]
    public async Task ProcessAsync_NullArrays_SetsEmptyStrings()
    {
        using var client = TestChatClient.WithJsonResponse(
            """{"people": null, "organizations": null, "technologies": null, "versions": null}""");
        var processor = new EntityExtractionProcessor(client);
        var chunks = CreateChunks("minimal content");

        var results = await processor.ProcessAsync(chunks).ToListAsync();

        Assert.Single(results);
        Assert.Equal("", results[0].Metadata["entities_people"]);
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
