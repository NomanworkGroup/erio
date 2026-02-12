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

## Common commands

Run from repository root:

```bash
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets
```

## License
- Apache-2.0
