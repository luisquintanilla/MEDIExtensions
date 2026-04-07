using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DataIngestion;

namespace MEDIExtensions.Ingestion;

/// <summary>
/// MEDI pipeline processor that classifies each chunk into primary and optional secondary topics.
/// Stores <c>topic_primary</c> (single label) and <c>topic_secondary</c> (comma-separated) as chunk metadata.
/// </summary>
public sealed class TopicClassificationProcessor : IngestionChunkProcessor<string>
{
    private readonly IChatClient _chatClient;
    private readonly string[] _taxonomy;

    /// <summary>
    /// Creates a new topic classification processor.
    /// </summary>
    /// <param name="chatClient">The chat client to use for classification.</param>
    /// <param name="taxonomy">Valid topic labels (e.g., "web", "data", "security", "performance", "architecture").</param>
    public TopicClassificationProcessor(IChatClient chatClient, string[] taxonomy)
    {
        _chatClient = chatClient ?? throw new ArgumentNullException(nameof(chatClient));
        _taxonomy = taxonomy ?? throw new ArgumentNullException(nameof(taxonomy));
    }

    public override async IAsyncEnumerable<IngestionChunk<string>> ProcessAsync(
        IAsyncEnumerable<IngestionChunk<string>> chunks,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        await foreach (var chunk in chunks.WithCancellation(ct))
        {
            string primary = "unknown";
            string secondary = "";

            try
            {
                var prompt = BuildPrompt(chunk.Content);
                var options = new ChatOptions
                {
                    MaxOutputTokens = 100,
                    ResponseFormat = ChatResponseFormat.Json
                };

                var response = await _chatClient.GetResponseAsync(prompt, options, ct);
                var parsed = JsonSerializer.Deserialize<TopicResponse>(
                    response.Text ?? "{}", JsonDefaults.Options)
                    ?? new TopicResponse();

                if (_taxonomy.Contains(parsed.Primary))
                {
                    primary = parsed.Primary;
                    var validSecondary = (parsed.Secondary ?? [])
                        .Where(t => _taxonomy.Contains(t) && t != primary)
                        .Distinct();
                    secondary = string.Join(",", validSecondary);
                }
            }
            catch
            {
                // Default to unknown on failure
            }

            chunk.Metadata["topic_primary"] = primary;
            chunk.Metadata["topic_secondary"] = secondary;

            yield return chunk;
        }
    }

    private string BuildPrompt(string text) => $$"""
        Classify this text into topics from: [{{string.Join(", ", _taxonomy.Select(t => $"\"{t}\""))}}].

        Return ONLY valid JSON (no extra text):
        {"primary": "topic", "secondary": ["topic2", "topic3"]}

        If the text fits only one topic, return an empty secondary array.

        Text: {{text}}
        """;

    private sealed class TopicResponse
    {
        [JsonPropertyName("primary")]
        public string Primary { get; set; } = "";

        [JsonPropertyName("secondary")]
        public List<string> Secondary { get; set; } = [];
    }
}
