//! LLM provider trait definition.

use crate::error::LlmError;
use crate::request::CompletionRequest;
use crate::response::CompletionResponse;

/// An LLM backend that can generate completions.
#[async_trait::async_trait]
pub trait LlmProvider: Send + Sync {
    /// Returns the provider name (e.g., "openai", "anthropic").
    fn name(&self) -> &str;

    /// Generates a completion for the given request.
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError>;
}
