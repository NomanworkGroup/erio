# erio-embedding

`erio-embedding` defines embedding engine abstractions and concrete
implementations used by Erio retrieval pipelines.

It supports local model execution via `GemmaEmbedding` and includes additional
configuration/types for embedding tasks.

## Quickstart

Set `ERIO_MODEL_DIR` to a directory containing local model assets before
constructing `GemmaEmbedding`.

```bash
export ERIO_MODEL_DIR=/absolute/path/to/local/embedding-model
```

```rust,no_run
use erio_embedding::{EmbeddingConfig, EmbeddingEngine, GemmaEmbedding};

# fn main() -> Result<(), erio_embedding::EmbeddingError> {
let _engine: Box<dyn EmbeddingEngine> = Box::new(GemmaEmbedding::new(EmbeddingConfig::default())?);
# Ok(())
# }
```

## docs.rs behavior

Real usage requires `ERIO_MODEL_DIR` to point to valid local model assets.
During docs.rs builds (`DOCS_RS=1`), the build script skips model download and
validation by setting `ERIO_MODEL_DIR` to a dummy directory, so docs/examples
are for API reference and are not runnable there.

## API tour

- Engine API: `EmbeddingEngine`
- Implementations: `GemmaEmbedding`, `RemoteEmbedding`
- Config/types: `EmbeddingConfig`, `TaskType`
- Error type: `EmbeddingError`
- Modules: `engine`, `gemma`, `remote`, `config`, `task`, `model`, `error`

## Related crates

- Used by `erio-context-store` for vector ingestion and semantic search.
- Often paired with `erio-llm-client` in RAG workflows.
- Docs: <https://docs.rs/erio-embedding>
- Source: <https://github.com/NomanworkGroup/erio/tree/main/crates/embedding>

## Compatibility

- MSRV: Rust 1.93
- License: Apache-2.0
