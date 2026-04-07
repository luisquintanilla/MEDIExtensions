using MEDIExtensions.Retrieval;
using MEDIExtensions.Tests.Utils;

namespace MEDIExtensions.Tests.Retrieval;

public class AdaptiveRouterTests
{
    [Fact]
    public void Constructor_ThrowsOnNullChatClient()
    {
        Assert.Throws<ArgumentNullException>("chatClient", () => new AdaptiveRouter(null!));
    }

    [Theory]
    [InlineData("vector")]
    [InlineData("tree")]
    [InlineData("entity")]
    public async Task ProcessQueryAsync_ValidParadigm_SetsMetadata(string paradigm)
    {
        using var client = TestChatClient.WithJsonResponse(
            $$"""{"paradigm": "{{paradigm}}", "reasoning": "test reasoning"}""");
        var router = new AdaptiveRouter(client);
        var query = new RetrievalQuery("test query");

        var output = await router.ProcessQueryAsync(query);

        Assert.Equal(paradigm, output.Metadata["search_paradigm"]);
    }

    [Fact]
    public async Task ProcessQueryAsync_InvalidParadigm_DefaultsToVector()
    {
        using var client = TestChatClient.WithJsonResponse(
            """{"paradigm": "unknown_paradigm", "reasoning": "bad"}""");
        var router = new AdaptiveRouter(client);
        var query = new RetrievalQuery("test query");

        var output = await router.ProcessQueryAsync(query);

        Assert.Equal("vector", output.Metadata["search_paradigm"]);
    }

    [Fact]
    public async Task ProcessQueryAsync_LlmThrows_DefaultsToVector()
    {
        using var client = TestChatClient.WithException(new InvalidOperationException("fail"));
        var router = new AdaptiveRouter(client);
        var query = new RetrievalQuery("test query");

        var output = await router.ProcessQueryAsync(query);

        Assert.Equal("vector", output.Metadata["search_paradigm"]);
    }

    [Fact]
    public async Task ProcessQueryAsync_StoresReasoningWhenPresent()
    {
        using var client = TestChatClient.WithJsonResponse(
            """{"paradigm": "tree", "reasoning": "broad question about architecture"}""");
        var router = new AdaptiveRouter(client);
        var query = new RetrievalQuery("what are the design patterns used?");

        var output = await router.ProcessQueryAsync(query);

        Assert.Equal("broad question about architecture", output.Metadata["router_reasoning"]);
    }

    [Fact]
    public async Task ProcessQueryAsync_NullReasoning_NoMetadataKey()
    {
        using var client = TestChatClient.WithJsonResponse("""{"paradigm": "vector"}""");
        var router = new AdaptiveRouter(client);
        var query = new RetrievalQuery("test");

        var output = await router.ProcessQueryAsync(query);

        Assert.Equal("vector", output.Metadata["search_paradigm"]);
        // Reasoning is null → not stored (depends on JSON deserialization default)
    }
}
