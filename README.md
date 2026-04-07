# MEDIExtensions

Community extensions for [Microsoft.Extensions.DataIngestion](https://www.nuget.org/packages/Microsoft.Extensions.DataIngestion) (MEDI) — adding retrieval pipeline, reranking, quality gates, ingestion enrichers, and advanced RAG patterns to .NET's AI ecosystem.

## Why

MEDI provides a clean 4-stage ingestion pipeline (Read → Chunk → Enrich → Write), but stops at ingestion. Every competing framework — LangChain, LlamaIndex, Haystack, Spring AI — covers the full RAG lifecycle. MEDIExtensions bridges this gap with retrieval-side abstractions and concrete implementations built on MEAI's `IChatClient`.

## Architecture

```
Microsoft.Extensions.DataIngestion          ← MEDI (ingestion)
  └─ IngestionPipeline<T>
       ├─ DocumentProcessors
       ├─ Chunker
       ├─ ChunkProcessors      ← MEDIExtensions.Ingestion adds enrichers here
       └─ Writer

MEDIExtensions.Retrieval                    ← NEW (retrieval)
  └─ RetrievalPipeline
       ├─ QueryProcessors       ← Pre-search: expansion, HyDE, routing
       ├─ VectorSearch          ← VectorStoreCollection.SearchAsync
       └─ ResultProcessors      ← Post-search: reranking, CRAG
```

## Packages

| Package | Description |
|---------|-------------|
| **MEDIExtensions** | All retrieval + ingestion processors. Single dependency: MEDI (MEAI + VectorData transitive). |
| **MEDIExtensions.Onnx** | ONNX cross-encoder reranker (planned). |

## Retrieval Components

### Pipeline & Abstractions

| Type | Description |
|------|-------------|
| `RetrievalPipeline` | Core orchestrator with OTel `ActivitySource`. Query processors → vector search → result processors. |
| `RetrievalQueryProcessor` | Abstract base for pre-search processing. |
| `RetrievalResultProcessor` | Abstract base for post-search processing. |
| `ISearchReranker` | Interface for reranking strategies. |
| `RetrievalQuery` | Query data type with variants + metadata. |
| `RetrievalResults` | Results + chunks data types. |

### Pre-Search Processors

| Processor | What It Does |
|-----------|-------------|
| `MultiQueryExpander` | Generates N query variants via `IChatClient`, merges results with Reciprocal Rank Fusion. |
| `HydeQueryTransformer` | Generates a hypothetical answer document, searches with its embedding instead of the raw query. |
| `AdaptiveRouter` | LLM-based query classification → routes to appropriate search paradigm (vector, tree, structured). |
| `TreeSearchRetriever` | Top-down tree traversal for RAPTOR-style hierarchical retrieval. |

### Post-Search Processors

| Processor | What It Does |
|-----------|-------------|
| `LlmReranker` | Scores each chunk's relevance via `IChatClient`, reorders by score. |
| `CragValidator` | 3-path quality routing: Correct (pass through) / Ambiguous (re-rank) / Incorrect (expand search). |

### Generation Orchestrators

| Orchestrator | What It Does |
|-------------|-------------|
| `SelfRagOrchestrator` | Adaptive retrieval decision → generate → self-critique → retry loop. |
| `SpeculativeRagOrchestrator` | `Task.WhenAll` parallel drafting with small model + verification by large model. |

## Ingestion Components

MEDI `IngestionChunkProcessor<string>` implementations:

| Processor | What It Does |
|-----------|-------------|
| `EntityExtractionProcessor` | Extracts people, organizations, technologies, versions → metadata for filtered search. |
| `TopicClassificationProcessor` | Assigns primary + secondary topic labels from a configurable taxonomy. |
| `HypotheticalQueryProcessor` | Reverse HyDE — generates questions each chunk answers as additional retrieval vectors. |
| `TreeIndexProcessor` | RAPTOR-style 3-level tree summaries (leaf → branch → root) in the same vector collection. |

## Usage

```csharp
// Register retrieval pipeline with processors
builder.Services.AddSingleton(sp =>
{
    var chatClient = sp.GetRequiredService<IChatClient>();
    var pipeline = new RetrievalPipeline(loggerFactory: sp.GetService<ILoggerFactory>());

    pipeline.QueryProcessors.Add(new MultiQueryExpander(chatClient));
    pipeline.ResultProcessors.Add(new LlmReranker(chatClient));
    pipeline.ResultProcessors.Add(new CragValidator(chatClient));

    return pipeline;
});

// Use in search
var results = await pipeline.RetrieveAsync(
    vectorCollection, query, topK: 5,
    contentSelector: chunk => chunk.Text);
```

```csharp
// Add ingestion enrichers to MEDI pipeline
var pipeline = new IngestionPipeline<string>(reader, chunker, writer)
{
    ChunkProcessors =
    {
        new ContextualChunkEnricher(chatClient),
        new EntityExtractionProcessor(chatClient),
        new TopicClassificationProcessor(chatClient, ["web", "data", "security"]),
    }
};
```

## Dependencies

Single direct dependency:

```xml
<PackageReference Include="Microsoft.Extensions.DataIngestion" Version="10.5.0-preview.1.26181.4" />
```

MEAI (`IChatClient`, `IEmbeddingGenerator`) and VectorData (`VectorStoreCollection`) come transitively.

## Target Frameworks

`net9.0` and `net10.0`

## Related

- [advanced-rag](../advanced-rag/) — Reference application showing full integration
- [medi-advanced-rag-investigation](../medi-advanced-rag-investigation/) — Research (28 docs, 22 samples) that produced these implementations
- [dotnet/extensions](https://github.com/dotnet/extensions) — Upstream MEDI / MEAI
