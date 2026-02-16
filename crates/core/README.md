# erio-core

`erio-core` provides shared runtime primitives for Erio crates: message types,
retry configuration, and common error variants.

It is the lowest-level dependency in the workspace and is designed to stay small
and stable so higher-level crates (`erio-tools`, `erio-llm-client`,
`erio-workflow`, and others) can compose on top of it.

## Quickstart

```rust,no_run
use erio_core::{Message, RetryConfig};

let _system = Message::system("You are a helpful assistant.");
let _user = Message::user("Summarize this text.");

let _retry = RetryConfig::builder().max_attempts(3).build();
```

## API tour

- Key types: `Message`, `Content`, `Role`, `ToolCall`
- Error/config: `CoreError`, `ToolError`, `RetryConfig`
- Modules: `config`, `error`, `message`

## Related crates

- Consumed by `erio-tools`, `erio-llm-client`, and workflow/agent crates for
  shared message and error contracts.
- Docs: <https://docs.rs/erio-core>
- Source: <https://github.com/NomanworkGroup/erio/tree/main/crates/core>

## Compatibility

- MSRV: Rust 1.93
- License: Apache-2.0
