using MEDIExtensions.Retrieval;

namespace MEDIExtensions.Tests.Retrieval;

public class RetrievalResultProcessorTests
{
    private sealed class TestResultProcessor : RetrievalResultProcessor
    {
        public override Task<RetrievalResults> ProcessResultsAsync(
            RetrievalResults results, CancellationToken cancellationToken = default)
            => Task.FromResult(results);
    }

    private sealed class NamedResultProcessor : RetrievalResultProcessor
    {
        public override string Name => "CustomResultName";

        public override Task<RetrievalResults> ProcessResultsAsync(
            RetrievalResults results, CancellationToken cancellationToken = default)
            => Task.FromResult(results);
    }

    [Fact]
    public void Name_DefaultsToTypeName()
    {
        var processor = new TestResultProcessor();
        Assert.Equal("TestResultProcessor", processor.Name);
    }

    [Fact]
    public void Name_CanBeOverridden()
    {
        var processor = new NamedResultProcessor();
        Assert.Equal("CustomResultName", processor.Name);
    }
}
