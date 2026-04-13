using Microsoft.Extensions.AI;
using Microsoft.Extensions.DataRetrieval;
using Microsoft.Extensions.DependencyInjection;

namespace MEDIExtensions.DependencyInjection;

/// <summary>
/// Fluent builder for composing a <see cref="RetrievalPipeline"/> with query and result processors.
/// Returned by <see cref="ServiceCollectionExtensions.AddRetrievalPipeline"/>.
/// </summary>
public class RetrievalPipelineBuilder
{
    internal List<Func<IServiceProvider, RetrievalQueryProcessor>> QueryProcessorFactories { get; } = [];
    internal List<Func<IServiceProvider, RetrievalResultProcessor>> ResultProcessorFactories { get; } = [];

    // -- Query Processors --

    /// <summary>Adds a query processor resolved from DI via <see cref="ActivatorUtilities"/>.</summary>
    public RetrievalPipelineBuilder UseQueryProcessor<T>() where T : RetrievalQueryProcessor
    {
        QueryProcessorFactories.Add(sp => ActivatorUtilities.CreateInstance<T>(sp));
        return this;
    }

    /// <summary>Adds adaptive query routing that classifies queries and selects the best search paradigm.</summary>
    public RetrievalPipelineBuilder UseAdaptiveRouting()
    {
        QueryProcessorFactories.Add(sp =>
            new Retrieval.AdaptiveRouter(sp.GetRequiredService<IChatClient>()));
        return this;
    }

    /// <summary>Adds multi-query expansion with Reciprocal Rank Fusion.</summary>
    public RetrievalPipelineBuilder UseQueryExpansion(Action<QueryExpansionOptions>? configure = null)
    {
        QueryProcessorFactories.Add(sp =>
        {
            var options = new QueryExpansionOptions();
            configure?.Invoke(options);
            return new Retrieval.MultiQueryExpander(sp.GetRequiredService<IChatClient>())
            {
                VariantCount = options.VariantCount
            };
        });
        return this;
    }

    /// <summary>Adds HyDE (Hypothetical Document Embeddings) query transformation.</summary>
    public RetrievalPipelineBuilder UseHyDE()
    {
        QueryProcessorFactories.Add(sp =>
            new Retrieval.HydeQueryTransformer(sp.GetRequiredService<IChatClient>()));
        return this;
    }

    /// <summary>Adds RAPTOR-style tree traversal for hierarchical search.</summary>
    public RetrievalPipelineBuilder UseTreeSearch(Action<TreeSearchOptions>? configure = null)
    {
        QueryProcessorFactories.Add(_ =>
        {
            var options = new TreeSearchOptions();
            configure?.Invoke(options);
            return new Retrieval.TreeSearchRetriever
            {
                ResultsPerLevel = options.ResultsPerLevel
            };
        });
        return this;
    }

    // -- Result Processors --

    /// <summary>Adds a result processor resolved from DI via <see cref="ActivatorUtilities"/>.</summary>
    public RetrievalPipelineBuilder UseResultProcessor<T>() where T : RetrievalResultProcessor
    {
        ResultProcessorFactories.Add(sp => ActivatorUtilities.CreateInstance<T>(sp));
        return this;
    }

    /// <summary>Adds LLM-based reranking of search results.</summary>
    public RetrievalPipelineBuilder UseLlmReranking(Action<LlmRerankingOptions>? configure = null)
    {
        ResultProcessorFactories.Add(sp =>
        {
            var options = new LlmRerankingOptions();
            configure?.Invoke(options);
            return new Retrieval.LlmReranker(sp.GetRequiredService<IChatClient>())
            {
                MaxResults = options.MaxResults,
                MaxCandidates = options.MaxCandidates,
                PreviewLength = options.PreviewLength
            };
        });
        return this;
    }

    /// <summary>Adds CRAG quality gate that routes results based on relevance confidence.</summary>
    public RetrievalPipelineBuilder UseCrag(Action<CragOptions>? configure = null)
    {
        ResultProcessorFactories.Add(sp =>
        {
            var options = new CragOptions();
            configure?.Invoke(options);
            return new Retrieval.CragValidator(sp.GetRequiredService<IChatClient>())
            {
                EvaluateTopN = options.EvaluateTopN,
                PreviewLength = options.PreviewLength
            };
        });
        return this;
    }
}

// -- Per-processor option classes --

public class QueryExpansionOptions
{
    public int VariantCount { get; set; } = 3;
}

public class TreeSearchOptions
{
    public int ResultsPerLevel { get; set; } = 3;
}

public class LlmRerankingOptions
{
    public int MaxResults { get; set; } = 5;
    public int MaxCandidates { get; set; } = 8;
    public int PreviewLength { get; set; } = 200;
}

public class CragOptions
{
    public int EvaluateTopN { get; set; } = 3;
    public int PreviewLength { get; set; } = 300;
}
