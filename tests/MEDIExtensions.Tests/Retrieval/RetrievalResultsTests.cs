using MEDIExtensions.Retrieval;

namespace MEDIExtensions.Tests.Retrieval;

public class RetrievalResultsTests
{
    [Fact]
    public void Chunks_DefaultsToEmpty()
    {
        var results = new RetrievalResults { Query = new RetrievalQuery("test") };
        Assert.Empty(results.Chunks);
    }

    [Fact]
    public void Chunks_CanBeReplaced()
    {
        var results = new RetrievalResults { Query = new RetrievalQuery("test") };
        var chunk = new RetrievalChunk { Content = "content" };
        results.Chunks = [chunk];

        Assert.Single(results.Chunks);
        Assert.Equal("content", results.Chunks[0].Content);
    }

    [Fact]
    public void Metadata_DefaultsToEmpty()
    {
        var results = new RetrievalResults { Query = new RetrievalQuery("test") };
        Assert.Empty(results.Metadata);
    }

    [Fact]
    public void Chunk_ScoreIsMutable()
    {
        var chunk = new RetrievalChunk { Content = "test" };
        Assert.Equal(0, chunk.Score);

        chunk.Score = 0.95;
        Assert.Equal(0.95, chunk.Score);
    }

    [Fact]
    public void Chunk_RecordDefaultsToEmpty()
    {
        var chunk = new RetrievalChunk { Content = "test" };
        Assert.Empty(chunk.Record);
    }
}
