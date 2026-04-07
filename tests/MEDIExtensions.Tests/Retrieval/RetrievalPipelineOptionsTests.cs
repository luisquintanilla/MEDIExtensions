using MEDIExtensions.Retrieval;

namespace MEDIExtensions.Tests.Retrieval;

public class RetrievalPipelineOptionsTests
{
    [Fact]
    public void ActivitySourceName_DefaultsToMEDIExtensionsRetrieval()
    {
        var options = new RetrievalPipelineOptions();
        Assert.Equal("MEDIExtensions.Retrieval", options.ActivitySourceName);
    }

    [Fact]
    public void ActivitySourceName_CanBeChanged()
    {
        var options = new RetrievalPipelineOptions
        {
            ActivitySourceName = "Custom.Source"
        };
        Assert.Equal("Custom.Source", options.ActivitySourceName);
    }
}
