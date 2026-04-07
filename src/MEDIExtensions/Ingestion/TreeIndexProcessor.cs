using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DataIngestion;

namespace MEDIExtensions.Ingestion;

/// <summary>
/// RAPTOR-style tree index processor — generates summary nodes at multiple levels.
/// </summary>
/// <remarks>
/// For each document's chunks, generates:
/// - Level 1 (Branch): LLM summary of all chunks within the same document
/// - Level 2 (Root): LLM summary of all branch summaries (corpus overview)
///
/// Summary nodes are injected INTO the same chunk stream with Level metadata.
/// The vector store writer stores them in the same collection as leaf chunks.
/// Standard <c>VectorStoreCollection.SearchAsync</c> naturally returns a mix of
/// leaf chunks (specific detail) and summary nodes (broader context) — "free RAPTOR."
///
/// Requires <c>IngestedChunk</c> to have Level, ParentId, and ChunkType fields.
/// </remarks>
public sealed class TreeIndexProcessor : IngestionChunkProcessor<string>
{
    private readonly IChatClient _chatClient;

    public TreeIndexProcessor(IChatClient chatClient)
    {
        _chatClient = chatClient ?? throw new ArgumentNullException(nameof(chatClient));
    }

    public override async IAsyncEnumerable<IngestionChunk<string>> ProcessAsync(
        IAsyncEnumerable<IngestionChunk<string>> chunks,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        // Collect all chunks, grouping by source document for summarization
        var chunksByDoc = new Dictionary<string, List<IngestionChunk<string>>>();
        var allChunks = new List<IngestionChunk<string>>();

        await foreach (var chunk in chunks.WithCancellation(ct))
        {
            // Mark as leaf node
            chunk.Metadata["level"] = 0;
            chunk.Metadata["chunk_type"] = "original";
            chunk.Metadata["parent_id"] = "";

            allChunks.Add(chunk);

            var docId = SanitizeDocId(chunk.Document.Identifier);
            if (!chunksByDoc.TryGetValue(docId, out var list))
                chunksByDoc[docId] = list = [];
            list.Add(chunk);
        }

        // Yield all original leaf chunks first
        foreach (var chunk in allChunks)
            yield return chunk;

        // Level 1 (Branch): Generate document-level summaries
        var branchSummaries = new List<(string DocId, string Summary)>();
        foreach (var (docId, docChunks) in chunksByDoc)
        {
            var combinedText = string.Join("\n\n", docChunks.Select(c => c.Content));
            var summary = await SummarizeAsync(
                $"Summarize the following text in 2-3 concise sentences. Capture key concepts and technologies.\n\n{combinedText}", ct);

            branchSummaries.Add((docId, summary));

            // Update leaf chunks to reference their branch parent
            foreach (var leaf in docChunks)
                leaf.Metadata["parent_id"] = $"branch-{docId}";

            // Create and yield branch summary chunk
            var branchChunk = new IngestionChunk<string>(summary, docChunks[0].Document, docChunks[0].Context);
            branchChunk.Metadata["level"] = 1;
            branchChunk.Metadata["chunk_type"] = "branch_summary";
            branchChunk.Metadata["parent_id"] = "root";
            yield return branchChunk;
        }

        // Level 2 (Root): Generate corpus-level summary
        if (branchSummaries.Count > 0)
        {
            var allBranchText = string.Join("\n\n",
                branchSummaries.Select(b => $"[{b.DocId}]: {b.Summary}"));

            var rootSummary = await SummarizeAsync(
                "Write a single 2-3 sentence overview of the entire corpus:\n\n" + allBranchText, ct);

            var rootChunk = new IngestionChunk<string>(rootSummary, allChunks[0].Document, allChunks[0].Context);
            rootChunk.Metadata["level"] = 2;
            rootChunk.Metadata["chunk_type"] = "root_summary";
            rootChunk.Metadata["parent_id"] = "";
            yield return rootChunk;
        }
    }

    private async Task<string> SummarizeAsync(string prompt, CancellationToken ct)
    {
        var fullPrompt = prompt + "\nReturn ONLY valid JSON: {\"summary\": \"your summary here\"}";

        try
        {
            var response = await _chatClient.GetResponseAsync(fullPrompt,
                new ChatOptions
                {
                    MaxOutputTokens = 200,
                    ResponseFormat = ChatResponseFormat.Json
                }, ct);

            var result = JsonSerializer.Deserialize<SummaryResponse>(
                response.Text ?? "{}", JsonDefaults.Options);
            return result?.Summary ?? "Summary unavailable";
        }
        catch
        {
            return "Summary unavailable";
        }
    }

    private static string SanitizeDocId(string identifier) =>
        Path.GetFileNameWithoutExtension(identifier).Replace(" ", "-").ToLowerInvariant();

    private sealed class SummaryResponse
    {
        [JsonPropertyName("summary")]
        public string Summary { get; set; } = "";
    }
}
