# erio-context-store

`erio-context-store` provides vector-backed context storage and semantic
retrieval APIs for retrieval-augmented workflows.

It handles indexing, persistence configuration, and top-k search over embedded
content.

## Quickstart

```rust,no_run
use std::path::PathBuf;
use std::sync::Arc;

use erio_context_store::{ContextConfig, ContextStore, HnswConfig};
use erio_embedding::{EmbeddingConfig, EmbeddingEngine, GemmaEmbedding};

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let embedding: Arc<dyn EmbeddingEngine> =
        Arc::new(GemmaEmbedding::new(EmbeddingConfig::default())?);

    let config = ContextConfig {
        path: PathBuf::from("/tmp/erio-context-store"),
        index: HnswConfig::default(),
    };

    let store = ContextStore::new(config, embedding).await?;
    let _results = store.search("what is erio?", 5, None).await?;
    Ok(())
}
```

## API tour

- Store/config: `ContextStore`, `ContextConfig`, `HnswConfig`
- Query/ops outputs: `SearchResult`, `StorageStats`
- Error type: `ContextStoreError`
- Modules: `store`, `config`, `error`

## Related crates

- Consumes `erio-embedding::EmbeddingEngine` for vector generation.
- Complements `erio-llm-client` in RAG pipelines.
- Docs: <https://docs.rs/erio-context-store>
- Source: <https://github.com/NomanworkGroup/erio/tree/main/crates/context-store>

## Compatibility

- MSRV: Rust 1.93
- License: Apache-2.0
