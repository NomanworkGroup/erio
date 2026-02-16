# erio-tools

`erio-tools` defines the tool execution layer for Erio: a `Tool` trait,
JSON-schema-like input modeling, result types, and an in-memory registry.

Use it when you want an agent to discover tools, validate parameters, and invoke
tool implementations consistently.

## Quickstart

```rust,no_run
use erio_tools::{PropertyType, ToolRegistry, ToolResult, ToolSchema};

let mut _registry = ToolRegistry::new();

let _schema = ToolSchema::builder()
    .property("query", PropertyType::String, "Search query", true)
    .build();

let _ok = ToolResult::success("done");
```

## API tour

- Key types: `Tool`, `ToolRegistry`, `ToolResult`
- Schema helpers: `ToolSchema`, `ToolSchemaBuilder`, `PropertyType`
- Error re-export: `ToolError` (from `erio-core`)

## Related crates

- Built on `erio-core` for shared error/message contracts.
- Commonly paired with `erio-llm-client` to expose tools to LLM providers.
- Docs: <https://docs.rs/erio-tools>
- Source: <https://github.com/NomanworkGroup/erio/tree/main/crates/tools>

## Compatibility

- MSRV: Rust 1.93
- License: Apache-2.0
