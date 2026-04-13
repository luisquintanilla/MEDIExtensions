using Microsoft.Extensions.AI;
using Microsoft.Extensions.DataRetrieval;

namespace MEDIExtensions.Retrieval;

/// <summary>
/// Hypothetical Document Embedding (HyDE) — generates a hypothetical answer to the query,
/// then searches using that answer's embedding instead of the raw query.
/// This bridges the query-document semantic gap.
/// </summary>
public class HydeQueryTransformer : RetrievalQueryProcessor
{
    private readonly IChatClient _chatClient;

    public HydeQueryTransformer(IChatClient chatClient)
    {
        _chatClient = chatClient ?? throw new ArgumentNullException(nameof(chatClient));
    }

    public override async Task<RetrievalQuery> ProcessAsync(
        RetrievalQuery query, CancellationToken cancellationToken = default)
    {
        var prompt = $"""
            Write a short, factual paragraph (3-4 sentences) that would directly answer this question.
            Write as if you are a documentation page. Do NOT include any preamble or explanation.

            Question: {query.Text}
            """;

        var options = new ChatOptions
        {
            MaxOutputTokens = 250
        };

        string hypothetical;
        try
        {
            var response = await _chatClient.GetResponseAsync(prompt, options, cancellationToken);
            hypothetical = (response.Text ?? "").Trim();
        }
        catch
        {
            return query; // Fall back to original query on failure
        }

        if (string.IsNullOrWhiteSpace(hypothetical))
            return query;

        // Replace variants with the hypothetical answer (keeps original in metadata)
        query.Variants = [hypothetical];
        query.Metadata["hyde_hypothetical"] = hypothetical;
        return query;
    }
}
