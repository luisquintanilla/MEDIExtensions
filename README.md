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
| **MEDIExtensions.Onnx** | ONNX cross-encoder reranker — local inference with ms-marco-MiniLM, BGE, and custom models. |

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

### Fluent Builder APIs (Recommended)

The builder APIs compose pipelines via familiar `IServiceCollection` extension methods:

```csharp
// Retrieval pipeline — query expansion + tree search + reranking + quality gate
builder.Services.AddRetrievalPipeline()
    .UseQueryExpansion(o => o.VariantCount = 5)
    .UseTreeSearch()
    .UseLlmReranking(o => o.MaxResults = 10)
    .UseCrag();

// Ingestion pipeline — enrichment + RAPTOR tree indexing
builder.Services.AddIngestionPipeline<string>()
    .UseEntityExtraction()
    .UseTopicClassification(o => o.Taxonomy = ["web", "data", "security"])
    .UseHypotheticalQueries(o => o.QuestionsPerChunk = 3)
    .UseTreeIndex();
```

Every `.UseX()` is optional, has per-processor `Action<TOptions>` configuration, and chains in execution order — the same pattern as `AddChatClient().UseFunctionInvocation()`.

Generic escape hatches for custom processors:

```csharp
builder.Services.AddRetrievalPipeline()
    .UseQueryProcessor<MyCustomExpander>()
    .UseResultProcessor<MyCustomReranker>();
```

### Manual Registration

```csharp
// Register retrieval pipeline with processors directly
builder.Services.AddSingleton(sp =>
{
    var chatClient = sp.GetRequiredService<IChatClient>();
    var pipeline = new RetrievalPipeline(loggerFactory: sp.GetService<ILoggerFactory>());

    pipeline.QueryProcessors.Add(new MultiQueryExpander(chatClient));
    pipeline.ResultProcessors.Add(new LlmReranker(chatClient));
    pipeline.ResultProcessors.Add(new CragValidator(chatClient));

    return pipeline;
});
```

## Dependencies

Single direct dependency:

```xml
<PackageReference Include="Microsoft.Extensions.DataIngestion" Version="10.5.0-preview.1.26181.4" />
```

MEAI (`IChatClient`, `IEmbeddingGenerator`) and VectorData (`VectorStoreCollection`) come transitively.

## Test Coverage

89 unit tests across 16 test files covering all processors, orchestrators, and ONNX inference:

| Category | Tests | Coverage |
|----------|-------|----------|
| Retrieval processors | 8 test classes | AdaptiveRouter, MultiQueryExpander, HyDE, TreeSearch, LlmReranker, CRAG |
| Ingestion processors | 4 test classes | EntityExtraction, TopicClassification, HypotheticalQueries, TreeIndex |
| ONNX reranker | 2 test classes | CrossEncoder scoring, options validation |
| Generation orchestrators | 2 test classes | SelfRAG flow, SpeculativeRAG parallel drafting |

All tests use mocked `IChatClient` and `VectorStoreCollection` — no external services required.

## Target Frameworks

`net9.0` and `net10.0`

## Related

- [advanced-rag](https://github.com/luisquintanilla/advanced-rag) — Reference application showing full integration
- [medi-advanced-rag-investigation](../medi-advanced-rag-investigation/) — Research (28 docs, 22 samples) that produced these implementations
- [dotnet/extensions](https://github.com/dotnet/extensions) — Upstream MEDI / MEAI
- [dotnet/extensions fork](https://github.com/luisquintanilla/extensions/tree/feature/retrieval-abstractions) — Retrieval pipeline abstractions
