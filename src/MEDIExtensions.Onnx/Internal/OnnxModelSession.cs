using Microsoft.ML.OnnxRuntime;

namespace MEDIExtensions.Onnx.Internal;

/// <summary>
/// Discovered ONNX model tensor metadata.
/// </summary>
internal sealed record OnnxModelMetadata(
    string InputIdsName,
    string AttentionMaskName,
    string? TokenTypeIdsName,
    string OutputTensorName,
    int HiddenDim,
    int OutputRank);

/// <summary>
/// Manages an ONNX InferenceSession for cross-encoder scoring.
/// Handles session creation, metadata discovery, batch inference, and disposal.
/// Ported from MLNet.TextInference.Onnx.OnnxTextModelScorerTransformer.
/// </summary>
internal sealed class OnnxModelSession : IDisposable
{
    private readonly InferenceSession _session;
    private readonly OnnxModelMetadata _metadata;

    public OnnxModelMetadata Metadata => _metadata;

    public OnnxModelSession(string modelPath, int? gpuDeviceId, bool fallbackToCpu, string[]? preferredOutputNames)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}");

        _session = CreateSession(modelPath, gpuDeviceId, fallbackToCpu);
        _metadata = DiscoverMetadata(_session, preferredOutputNames);
    }

    /// <summary>
    /// Runs ONNX inference on a tokenized batch. Returns raw logits per item.
    /// For cross-encoders, each output is typically [1] (single logit) or [hidden_dim].
    /// </summary>
    public float[][] Score(TokenizedBatch batch)
    {
        int totalRows = batch.Count;
        int seqLen = batch.SequenceLength;
        var allOutputs = new List<float[]>(totalRows);

        int batchSize = totalRows; // Process all at once — caller handles chunking
        for (int start = 0; start < totalRows; start += batchSize)
        {
            int count = Math.Min(batchSize, totalRows - start);
            var batchOutputs = RunOnnxBatch(
                batch.TokenIds, batch.AttentionMasks, batch.TokenTypeIds,
                start, count, seqLen);
            allOutputs.AddRange(batchOutputs);
        }

        return [.. allOutputs];
    }

    private float[][] RunOnnxBatch(
        long[][] tokenIds, long[][] attentionMasks, long[][] tokenTypeIds,
        int startIdx, int batchSize, int seqLen)
    {
        var idsArray = new long[batchSize * seqLen];
        var maskArray = new long[batchSize * seqLen];
        var typeIdsArray = _metadata.TokenTypeIdsName != null ? new long[batchSize * seqLen] : null;

        for (int b = 0; b < batchSize; b++)
        {
            Array.Copy(tokenIds[startIdx + b], 0, idsArray, b * seqLen, seqLen);
            Array.Copy(attentionMasks[startIdx + b], 0, maskArray, b * seqLen, seqLen);
            if (typeIdsArray != null)
                Array.Copy(tokenTypeIds[startIdx + b], 0, typeIdsArray, b * seqLen, seqLen);
        }

        var inputs = new Dictionary<string, OrtValue>
        {
            [_metadata.InputIdsName] = OrtValue.CreateTensorValueFromMemory(idsArray, [batchSize, seqLen]),
            [_metadata.AttentionMaskName] = OrtValue.CreateTensorValueFromMemory(maskArray, [batchSize, seqLen])
        };

        if (_metadata.TokenTypeIdsName != null && typeIdsArray != null)
            inputs[_metadata.TokenTypeIdsName] = OrtValue.CreateTensorValueFromMemory(typeIdsArray, [batchSize, seqLen]);

        try
        {
            using var results = _session.Run(new RunOptions(), inputs, [_metadata.OutputTensorName]);
            var output = results[0];
            var outputSpan = output.GetTensorDataAsSpan<float>();

            var batchOutputs = new float[batchSize][];

            if (_metadata.OutputRank == 2)
            {
                // Pooled output: [batch, hidden_dim]
                for (int b = 0; b < batchSize; b++)
                    batchOutputs[b] = outputSpan.Slice(b * _metadata.HiddenDim, _metadata.HiddenDim).ToArray();
            }
            else
            {
                // Unpooled output: [batch, seq_len, hidden_dim]
                int rowSize = seqLen * _metadata.HiddenDim;
                for (int b = 0; b < batchSize; b++)
                    batchOutputs[b] = outputSpan.Slice(b * rowSize, rowSize).ToArray();
            }

            return batchOutputs;
        }
        finally
        {
            foreach (var ortValue in inputs.Values)
                ortValue.Dispose();
        }
    }

    private static InferenceSession CreateSession(string modelPath, int? gpuDeviceId, bool fallbackToCpu)
    {
        var options = new SessionOptions();

        if (gpuDeviceId.HasValue)
        {
            try
            {
                options.AppendExecutionProvider_CUDA(gpuDeviceId.Value);
            }
            catch (Exception) when (fallbackToCpu)
            {
                // CUDA libraries not available — fall back to CPU
            }
        }

        try
        {
            return new InferenceSession(modelPath, options);
        }
        catch (OnnxRuntimeException) when (fallbackToCpu)
        {
            return new InferenceSession(modelPath, new SessionOptions());
        }
    }

    private static OnnxModelMetadata DiscoverMetadata(InferenceSession session, string[]? preferredOutputNames)
    {
        var inputMeta = session.InputMetadata;
        var outputMeta = session.OutputMetadata;

        string inputIdsName = FindTensorName(inputMeta, ["input_ids"], inputMeta.Keys.First());
        string attentionMaskName = FindTensorName(inputMeta, ["attention_mask"], "attention_mask");
        string? tokenTypeIdsName = TryFindTensorName(inputMeta, ["token_type_ids"]);

        string outputName;
        int hiddenDim;
        int outputRank;

        if (preferredOutputNames != null)
        {
            var preferred = TryFindTensorName(outputMeta, preferredOutputNames);
            if (preferred != null)
            {
                outputName = preferred;
            }
            else
            {
                outputName = outputMeta.Keys.First();
            }
        }
        else
        {
            // Default auto-discovery for cross-encoders
            var pooledName = TryFindTensorName(outputMeta, ["logits", "output", "sentence_embedding", "pooler_output"]);
            outputName = pooledName ?? outputMeta.Keys.First();
        }

        var dims = outputMeta[outputName].Dimensions;
        outputRank = dims.Length;
        int lastDim = (int)dims.Last();
        hiddenDim = lastDim > 0 ? lastDim : 1; // Cross-encoders typically output [batch, 1]

        return new OnnxModelMetadata(inputIdsName, attentionMaskName, tokenTypeIdsName, outputName, hiddenDim, outputRank);
    }

    private static string FindTensorName(
        IReadOnlyDictionary<string, NodeMetadata> metadata, string[] candidates, string fallback)
    {
        return TryFindTensorName(metadata, candidates) ?? fallback;
    }

    private static string? TryFindTensorName(
        IReadOnlyDictionary<string, NodeMetadata> metadata, string[] candidates)
    {
        foreach (var candidate in candidates)
        {
            if (metadata.ContainsKey(candidate))
                return candidate;
        }
        return null;
    }

    public void Dispose() => _session.Dispose();
}
