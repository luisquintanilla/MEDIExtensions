using MEDIExtensions.Onnx;

namespace MEDIExtensions.Tests.Onnx;

public class CrossEncoderRerankerOptionsTests
{
    [Fact]
    public void DefaultMaxTokenLength_Is512()
    {
        var options = new CrossEncoderRerankerOptions
        {
            ModelPath = "model.onnx",
            TokenizerPath = "tok/"
        };
        Assert.Equal(512, options.MaxTokenLength);
    }

    [Fact]
    public void DefaultBatchSize_Is32()
    {
        var options = new CrossEncoderRerankerOptions
        {
            ModelPath = "model.onnx",
            TokenizerPath = "tok/"
        };
        Assert.Equal(32, options.BatchSize);
    }

    [Fact]
    public void DefaultGpuDeviceId_IsNull()
    {
        var options = new CrossEncoderRerankerOptions
        {
            ModelPath = "model.onnx",
            TokenizerPath = "tok/"
        };
        Assert.Null(options.GpuDeviceId);
    }

    [Fact]
    public void DefaultFallbackToCpu_IsFalse()
    {
        var options = new CrossEncoderRerankerOptions
        {
            ModelPath = "model.onnx",
            TokenizerPath = "tok/"
        };
        Assert.False(options.FallbackToCpu);
    }

    [Fact]
    public void DefaultMaxResults_Is5()
    {
        var options = new CrossEncoderRerankerOptions
        {
            ModelPath = "model.onnx",
            TokenizerPath = "tok/"
        };
        Assert.Equal(5, options.MaxResults);
    }

    [Fact]
    public void DefaultPreferredOutputNames_HasLogitsAndOutput()
    {
        var options = new CrossEncoderRerankerOptions
        {
            ModelPath = "model.onnx",
            TokenizerPath = "tok/"
        };
        Assert.Equal(["logits", "output"], options.PreferredOutputNames);
    }

    [Fact]
    public void Properties_CanBeSet()
    {
        var options = new CrossEncoderRerankerOptions
        {
            ModelPath = "/models/cross-encoder.onnx",
            TokenizerPath = "/models/tokenizer/",
            MaxTokenLength = 256,
            BatchSize = 16,
            GpuDeviceId = 0,
            FallbackToCpu = true,
            MaxResults = 10,
            PreferredOutputNames = ["custom_output"]
        };

        Assert.Equal("/models/cross-encoder.onnx", options.ModelPath);
        Assert.Equal("/models/tokenizer/", options.TokenizerPath);
        Assert.Equal(256, options.MaxTokenLength);
        Assert.Equal(16, options.BatchSize);
        Assert.Equal(0, options.GpuDeviceId);
        Assert.True(options.FallbackToCpu);
        Assert.Equal(10, options.MaxResults);
        Assert.Equal(["custom_output"], options.PreferredOutputNames);
    }
}
