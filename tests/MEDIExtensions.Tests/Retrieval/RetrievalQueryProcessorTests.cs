using MEDIExtensions.Retrieval;

namespace MEDIExtensions.Tests.Retrieval;

public class RetrievalQueryProcessorTests
{
    private sealed class TestQueryProcessor : RetrievalQueryProcessor
    {
        public override Task<RetrievalQuery> ProcessQueryAsync(
            RetrievalQuery query, CancellationToken cancellationToken = default)
            => Task.FromResult(query);
    }

    private sealed class NamedQueryProcessor : RetrievalQueryProcessor
    {
        public override string Name => "CustomName";

        public override Task<RetrievalQuery> ProcessQueryAsync(
            RetrievalQuery query, CancellationToken cancellationToken = default)
            => Task.FromResult(query);
    }

    [Fact]
    public void Name_DefaultsToTypeName()
    {
        var processor = new TestQueryProcessor();
        Assert.Equal("TestQueryProcessor", processor.Name);
    }

    [Fact]
    public void Name_CanBeOverridden()
    {
        var processor = new NamedQueryProcessor();
        Assert.Equal("CustomName", processor.Name);
    }
}
