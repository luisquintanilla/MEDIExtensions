using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DataIngestion;

namespace MEDIExtensions.Ingestion;

/// <summary>
/// MEDI pipeline processor that generates hypothetical questions each chunk could answer,
/// storing them as additional retrieval vectors in the same collection.
/// This is "reverse HyDE" — bridging the query-document gap at ingestion time.
/// </summary>
/// <remarks>
/// For each original chunk, yields:
/// 1. The original chunk (with <c>chunk_type=original</c> metadata)
/// 2. N question chunks (with <c>chunk_type=hypothetical_query</c> and <c>parent_chunk_id</c>)
///
/// Question chunks contain the question text as their Content — the embedding
/// model creates vectors in "question space," directly matching user queries.
/// </remarks>
public sealed class HypotheticalQueryProcessor : IngestionChunkProcessor<string>
{
    private readonly IChatClient _chatClient;
    private readonly int _questionsPerChunk;

    public HypotheticalQueryProcessor(IChatClient chatClient, int questionsPerChunk = 3)
    {
        _chatClient = chatClient ?? throw new ArgumentNullException(nameof(chatClient));
        _questionsPerChunk = questionsPerChunk;
    }

    public override async IAsyncEnumerable<IngestionChunk<string>> ProcessAsync(
        IAsyncEnumerable<IngestionChunk<string>> chunks,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        await foreach (var chunk in chunks.WithCancellation(ct))
        {
            var chunkId = chunk.Content.GetHashCode().ToString();

            // Yield original chunk first (with consistent metadata keys)
            chunk.Metadata["chunk_type"] = "original";
            chunk.Metadata["parent_chunk_id"] = "";
            chunk.Metadata["hypothetical_questions"] = "";
            yield return chunk;

            // Generate hypothetical questions (best-effort)
            var questionChunks = new List<IngestionChunk<string>>();
            try
            {
                var questions = await GenerateQuestionsAsync(chunk.Content, ct);
                chunk.Metadata["hypothetical_questions"] = string.Join(" | ", questions);

                foreach (var question in questions)
                {
                    var qChunk = new IngestionChunk<string>(question, chunk.Document, chunk.Context);
                    qChunk.Metadata["chunk_type"] = "hypothetical_query";
                    qChunk.Metadata["parent_chunk_id"] = chunkId;
                    qChunk.Metadata["hypothetical_questions"] = "";
                    questionChunks.Add(qChunk);
                }
            }
            catch
            {
                // Best-effort: if generation fails, just continue with original
            }

            foreach (var qChunk in questionChunks)
                yield return qChunk;
        }
    }

    private async Task<List<string>> GenerateQuestionsAsync(string chunkContent, CancellationToken ct)
    {
        var prompt = $"Generate exactly {_questionsPerChunk} questions that the following "
            + "text passage answers.\n"
            + "Return ONLY valid JSON matching: {\"questions\": [\"question1?\", \"question2?\", \"question3?\"]}"
            + $"\n\nPassage:\n{chunkContent}";

        var response = await _chatClient.GetResponseAsync(prompt,
            new ChatOptions
            {
                MaxOutputTokens = 200,
                ResponseFormat = ChatResponseFormat.Json
            }, ct);

        var result = JsonSerializer.Deserialize<QuestionsResponse>(
            response.Text ?? "{}", JsonDefaults.Options);

        return (result?.Questions ?? [])
            .Where(q => !string.IsNullOrWhiteSpace(q) && q.Length > 10)
            .Take(_questionsPerChunk)
            .ToList();
    }

    private sealed class QuestionsResponse
    {
        [JsonPropertyName("questions")]
        public List<string> Questions { get; set; } = [];
    }
}
