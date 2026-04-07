using MEDIExtensions.Retrieval;

namespace MEDIExtensions.Tests.Retrieval;

public class RetrievalQueryTests
{
    [Fact]
    public void Constructor_SetsOriginal()
    {
        var query = new RetrievalQuery("test query");
        Assert.Equal("test query", query.Original);
    }

    [Fact]
    public void Constructor_InitializesVariantsWithOriginal()
    {
        var query = new RetrievalQuery("test query");
        Assert.Single(query.Variants);
        Assert.Equal("test query", query.Variants[0]);
    }

    [Fact]
    public void Constructor_InitializesEmptyMetadata()
    {
        var query = new RetrievalQuery("test query");
        Assert.Empty(query.Metadata);
    }

    [Fact]
    public void Variants_CanBeReplaced()
    {
        var query = new RetrievalQuery("original");
        query.Variants = ["variant1", "variant2"];

        Assert.Equal(2, query.Variants.Count);
        Assert.Equal("variant1", query.Variants[0]);
        Assert.Equal("original", query.Original); // Original unchanged
    }

    [Fact]
    public void Metadata_CanBeModified()
    {
        var query = new RetrievalQuery("test");
        query.Metadata["key1"] = "value1";
        query.Metadata["key2"] = 42;

        Assert.Equal("value1", query.Metadata["key1"]);
        Assert.Equal(42, query.Metadata["key2"]);
    }
}
