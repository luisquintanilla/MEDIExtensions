using Microsoft.Extensions.AI;
using Microsoft.Extensions.DataIngestion;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace MEDIExtensions.DependencyInjection;

/// <summary>
/// Fluent builder/factory for composing <see cref="IngestionPipeline{T}"/> instances with
/// document and chunk processors. Registered as a singleton; call <see cref="Build"/> to
/// create configured pipeline instances (pipelines are <see cref="IDisposable"/>).
/// </summary>
/// <typeparam name="T">The chunk content type (typically <c>string</c>).</typeparam>
public class IngestionPipelineBuilder<T>
{
    internal List<Func<IServiceProvider, IngestionDocumentProcessor>> DocumentProcessorFactories { get; } = [];
    internal List<Func<IServiceProvider, IngestionChunkProcessor<T>>> ChunkProcessorFactories { get; } = [];

    // -- Document Processors --

    /// <summary>Adds a document processor created by the given factory.</summary>
    public IngestionPipelineBuilder<T> UseDocumentProcessor(
        Func<IServiceProvider, IngestionDocumentProcessor> factory)
    {
        DocumentProcessorFactories.Add(factory);
        return this;
    }

    /// <summary>Adds a document processor resolved from DI.</summary>
    public IngestionPipelineBuilder<T> UseDocumentProcessor<TProcessor>()
        where TProcessor : IngestionDocumentProcessor
    {
        DocumentProcessorFactories.Add(sp =>
            ActivatorUtilities.CreateInstance<TProcessor>(sp));
        return this;
    }

    // -- Chunk Processors --

    /// <summary>Adds a chunk processor created by the given factory.</summary>
    public IngestionPipelineBuilder<T> UseChunkProcessor(
        Func<IServiceProvider, IngestionChunkProcessor<T>> factory)
    {
        ChunkProcessorFactories.Add(factory);
        return this;
    }

    /// <summary>Adds a chunk processor resolved from DI.</summary>
    public IngestionPipelineBuilder<T> UseChunkProcessor<TProcessor>()
        where TProcessor : IngestionChunkProcessor<T>
    {
        ChunkProcessorFactories.Add(sp =>
            ActivatorUtilities.CreateInstance<TProcessor>(sp));
        return this;
    }

    // -- Convenience methods for MEDIExtensions processors --

    /// <summary>Adds entity extraction (people, orgs, technologies, versions) to chunks.</summary>
    public IngestionPipelineBuilder<T> UseEntityExtraction()
    {
        ChunkProcessorFactories.Add(sp =>
            (IngestionChunkProcessor<T>)(object)new Ingestion.EntityExtractionProcessor(
                sp.GetRequiredService<IChatClient>()));
        return this;
    }

    /// <summary>Adds topic classification with a configurable taxonomy.</summary>
    public IngestionPipelineBuilder<T> UseTopicClassification(
        Action<TopicClassificationOptions>? configure = null)
    {
        ChunkProcessorFactories.Add(sp =>
        {
            var options = new TopicClassificationOptions();
            configure?.Invoke(options);
            return (IngestionChunkProcessor<T>)(object)new Ingestion.TopicClassificationProcessor(
                sp.GetRequiredService<IChatClient>(), options.Taxonomy);
        });
        return this;
    }

    /// <summary>Adds hypothetical query generation for reverse-HyDE chunk enrichment.</summary>
    public IngestionPipelineBuilder<T> UseHypotheticalQueries(
        Action<HypotheticalQueryOptions>? configure = null)
    {
        ChunkProcessorFactories.Add(sp =>
        {
            var options = new HypotheticalQueryOptions();
            configure?.Invoke(options);
            return (IngestionChunkProcessor<T>)(object)new Ingestion.HypotheticalQueryProcessor(
                sp.GetRequiredService<IChatClient>(), options.QuestionsPerChunk);
        });
        return this;
    }

    /// <summary>Adds RAPTOR-style tree index generation (leaf → branch → root summaries).</summary>
    public IngestionPipelineBuilder<T> UseTreeIndex()
    {
        ChunkProcessorFactories.Add(sp =>
            (IngestionChunkProcessor<T>)(object)new Ingestion.TreeIndexProcessor(
                sp.GetRequiredService<IChatClient>()));
        return this;
    }

    /// <summary>
    /// Creates a configured <see cref="IngestionPipeline{T}"/> with all registered processors applied.
    /// The caller is responsible for disposing the returned pipeline.
    /// </summary>
    public IngestionPipeline<T> Build(
        IServiceProvider serviceProvider,
        IngestionDocumentReader reader,
        IngestionChunker<T> chunker,
        IngestionChunkWriter<T> writer,
        ILoggerFactory? loggerFactory = null)
    {
        var pipeline = new IngestionPipeline<T>(reader, chunker, writer, loggerFactory: loggerFactory);

        foreach (var factory in DocumentProcessorFactories)
            pipeline.DocumentProcessors.Add(factory(serviceProvider));

        foreach (var factory in ChunkProcessorFactories)
            pipeline.ChunkProcessors.Add(factory(serviceProvider));

        return pipeline;
    }
}

// -- Per-processor option classes --

public class TopicClassificationOptions
{
    public string[] Taxonomy { get; set; } = ["web", "data", "performance", "security", "architecture"];
}

public class HypotheticalQueryOptions
{
    public int QuestionsPerChunk { get; set; } = 3;
}
