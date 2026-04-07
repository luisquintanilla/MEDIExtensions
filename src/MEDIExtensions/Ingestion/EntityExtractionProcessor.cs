using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DataIngestion;

namespace MEDIExtensions.Ingestion;

/// <summary>
/// MEDI <see cref="IngestionChunkProcessor{T}"/> that uses an LLM to extract
/// named entities (people, organizations, technologies, versions) from each
/// chunk and stores them as metadata for downstream filtered vector search.
/// </summary>
public sealed class EntityExtractionProcessor : IngestionChunkProcessor<string>
{
    private readonly IChatClient _chatClient;

    private static readonly string[] EntityKeys =
        ["entities_people", "entities_organizations", "entities_technologies", "entities_versions"];

    public EntityExtractionProcessor(IChatClient chatClient)
    {
        _chatClient = chatClient ?? throw new ArgumentNullException(nameof(chatClient));
    }

    public override async IAsyncEnumerable<IngestionChunk<string>> ProcessAsync(
        IAsyncEnumerable<IngestionChunk<string>> chunks,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        await foreach (var chunk in chunks.WithCancellation(ct))
        {
            try
            {
                var prompt = $$"""
                    Extract named entities from the following text.
                    Return ONLY valid JSON matching: {"people": ["name"], "organizations": ["org"], "technologies": ["tech"], "versions": ["v1"]}
                    Use empty arrays [] when none found for a category.

                    Text:
                    {{chunk.Content}}
                    """;

                var options = new ChatOptions
                {
                    MaxOutputTokens = 300,
                    ResponseFormat = ChatResponseFormat.Json
                };

                var response = await _chatClient.GetResponseAsync(prompt, options, ct);
                var entities = JsonSerializer.Deserialize<EntityResponse>(
                    response.Text ?? "{}", JsonDefaults.Options);

                if (entities is not null)
                {
                    chunk.Metadata["entities_people"] = string.Join(", ", entities.People ?? []);
                    chunk.Metadata["entities_organizations"] = string.Join(", ", entities.Organizations ?? []);
                    chunk.Metadata["entities_technologies"] = string.Join(", ", entities.Technologies ?? []);
                    chunk.Metadata["entities_versions"] = string.Join(", ", entities.Versions ?? []);
                }
                else
                {
                    SetEmptyEntities(chunk);
                }
            }
            catch
            {
                SetEmptyEntities(chunk);
            }

            yield return chunk;
        }
    }

    private static void SetEmptyEntities(IngestionChunk<string> chunk)
    {
        foreach (var key in EntityKeys)
            chunk.Metadata[key] = "";
    }

    private sealed class EntityResponse
    {
        [JsonPropertyName("people")]
        public string[]? People { get; set; }

        [JsonPropertyName("organizations")]
        public string[]? Organizations { get; set; }

        [JsonPropertyName("technologies")]
        public string[]? Technologies { get; set; }

        [JsonPropertyName("versions")]
        public string[]? Versions { get; set; }
    }
}
