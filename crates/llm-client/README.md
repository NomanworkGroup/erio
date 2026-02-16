# erio-llm-client

`erio-llm-client` is Erio's provider abstraction for chat/completion style LLM
calls. It includes request/response models, error handling, and an OpenAI-
compatible provider implementation.

Use this crate to keep provider integration behind a trait while sharing a
single request/response shape across agents and workflows.

## Quickstart

```rust,no_run
use erio_core::Message;
use erio_llm_client::{CompletionRequest, LlmProvider, OpenAiProvider};

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let provider = OpenAiProvider::new("https://api.openai.com/v1", "YOUR_API_KEY");

    let request = CompletionRequest::new("gpt-4o-mini")
        .message(Message::system("You are concise."))
        .message(Message::user("Say hello in one sentence."));

    let _response = provider.complete(request).await?;
    Ok(())
}
```

## API tour

- Provider types: `LlmProvider`, `OpenAiProvider`
- Request types: `CompletionRequest`, `ToolDefinition`
- Response types: `CompletionResponse`, `StreamChunk`, `Usage`
- Error type: `LlmError`
- Modules: `openai`, `provider`, `request`, `response`, `error`

## Related crates

- Uses `erio-core::Message` as the canonical message format.
- Integrates with `erio-tools` by passing `ToolDefinition` in requests.
- Docs: <https://docs.rs/erio-llm-client>
- Source: <https://github.com/NomanworkGroup/erio/tree/main/crates/llm-client>

## Compatibility

- MSRV: Rust 1.93
- License: Apache-2.0
