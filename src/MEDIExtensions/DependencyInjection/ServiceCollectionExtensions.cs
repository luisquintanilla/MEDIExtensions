using Microsoft.Extensions.DataIngestion;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace MEDIExtensions.DependencyInjection;

/// <summary>
/// Extension methods for registering retrieval and ingestion pipeline builders.
/// </summary>
public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Registers a <see cref="RetrievalPipeline"/> singleton and returns a
    /// <see cref="RetrievalPipelineBuilder"/> for composing processors.
    /// </summary>
    /// <example>
    /// <code>
    /// builder.Services.AddRetrievalPipeline()
    ///     .UseQueryExpansion(o => o.VariantCount = 5)
    ///     .UseLlmReranking()
    ///     .UseCrag();
    /// </code>
    /// </example>
    public static RetrievalPipelineBuilder AddRetrievalPipeline(
        this IServiceCollection services)
    {
        var pipelineBuilder = new RetrievalPipelineBuilder();

        services.AddSingleton(sp =>
        {
            var loggerFactory = sp.GetService<ILoggerFactory>();
            var pipeline = new RetrievalPipeline(loggerFactory: loggerFactory);

            foreach (var factory in pipelineBuilder.QueryProcessorFactories)
                pipeline.QueryProcessors.Add(factory(sp));

            foreach (var factory in pipelineBuilder.ResultProcessorFactories)
                pipeline.ResultProcessors.Add(factory(sp));

            return pipeline;
        });

        return pipelineBuilder;
    }

    /// <summary>
    /// Registers a <see cref="RetrievalPipeline"/> singleton using a custom factory and returns a
    /// <see cref="RetrievalPipelineBuilder"/> for composing processors on top.
    /// </summary>
    public static RetrievalPipelineBuilder AddRetrievalPipeline(
        this IServiceCollection services,
        Func<IServiceProvider, RetrievalPipeline> pipelineFactory)
    {
        var pipelineBuilder = new RetrievalPipelineBuilder();

        services.AddSingleton(sp =>
        {
            var pipeline = pipelineFactory(sp);

            foreach (var factory in pipelineBuilder.QueryProcessorFactories)
                pipeline.QueryProcessors.Add(factory(sp));

            foreach (var factory in pipelineBuilder.ResultProcessorFactories)
                pipeline.ResultProcessors.Add(factory(sp));

            return pipeline;
        });

        return pipelineBuilder;
    }

    /// <summary>
    /// Registers an <see cref="IngestionPipelineBuilder{T}"/> singleton for composing
    /// ingestion pipelines. Use <c>string</c> for standard text content.
    /// </summary>
    /// <example>
    /// <code>
    /// builder.Services.AddIngestionPipeline&lt;string&gt;()
    ///     .UseEntityExtraction()
    ///     .UseTopicClassification(o => o.Taxonomy = ["web", "data", "security"]);
    /// </code>
    /// </example>
    public static IngestionPipelineBuilder<T> AddIngestionPipeline<T>(
        this IServiceCollection services)
    {
        var pipelineBuilder = new IngestionPipelineBuilder<T>();
        services.AddSingleton(pipelineBuilder);
        return pipelineBuilder;
    }

    /// <summary>
    /// Registers an <see cref="IngestionPipelineBuilder{T}"/> with <c>string</c> content type (the common case).
    /// </summary>
    public static IngestionPipelineBuilder<string> AddIngestionPipeline(
        this IServiceCollection services)
        => services.AddIngestionPipeline<string>();
}
