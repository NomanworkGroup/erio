# erio

Erio is a Rust workspace for building an agent runtime with modular crates for
LLM calls, tool execution, workflows, event handling, embeddings, and vector
context storage.

## Workspace layout

- `crates/core` (`erio-core`): core types, messages, retry config, and shared
  errors.
- `crates/tools` (`erio-tools`): tool trait, schema helpers, registry, and tool
  execution primitives.
- `crates/llm-client` (`erio-llm-client`): provider abstraction and OpenAI
  adapter.
- `crates/workflow` (`erio-workflow`): DAG-based workflow engine and step
  orchestration.
- `crates/event-bus` (`erio-event-bus`): event model, event bus, and pluggable
  event sources.
- `crates/embedding` (`erio-embedding`): embedding engine abstractions and
  model integrations.
- `crates/context-store` (`erio-context-store`): vector-backed context storage
  and semantic retrieval.

## Examples

- `examples/react-agent`: ReAct (Reason + Act) loop agent using LLM + Tools.
- `examples/plan-execute-agent`: Plan-then-execute agent using the workflow DAG
  engine.
- `examples/multi-agent-chat`: Multi-agent coordination pipeline via EventBus.
- `examples/file-watcher-agent`: Event-driven file summariser using EventBus +
  LLM.
- `examples/rag-agent`: Retrieval-augmented generation using ContextStore,
  Embedding, and LLM.

## Requirements

- Rust `1.93.0` or newer (workspace minimum)
- Cargo (ships with Rust)
- `OPENAI_API_KEY` and `OPENAI_BASE_URL` environment variables for LLM examples

## Local embedding model (EmbeddingGemma)

`erio-embedding` uses a local EmbeddingGemma model. Model files are fetched at build time from public GitHub Release assets.

- Offline/custom path: set `ERIO_MODEL_DIR` to a directory containing:
  - `embeddinggemma-300M-Q8_0.gguf`
  - `tokenizer.json`
  - `2_Dense/model.safetensors`
  - `3_Dense/model.safetensors`
- For maintainers: the CI workflow publishes these assets and requires `HF_TOKEN` (after accepting the
  [EmbeddingGemma license](https://huggingface.co/google/embeddinggemma-300m)).

## Running examples

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1

cargo run -p react-agent -- --prompt "Please uppercase the word hello"
cargo run -p plan-execute-agent -- --prompt "Add 10 and 20, then multiply by 3"
cargo run -p multi-agent-chat -- --prompt "Explain why Rust is great"
cargo run -p file-watcher-agent -- --files README.md
cargo run -p rag-agent -- --query "What is erio?" --documents README.md
```

## Common commands

Run from repository root:

```bash
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets
```

## License
- Apache-2.0
