//! Completion response types for LLM providers.

use erio_core::ToolCall;
use serde::Deserialize;

/// Token usage statistics.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// A completion response from an LLM provider.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    /// Text content of the response.
    pub content: Option<String>,
    /// Tool calls requested by the model.
    pub tool_calls: Vec<ToolCall>,
    /// Model that generated the response.
    pub model: String,
    /// Token usage statistics.
    pub usage: Option<Usage>,
}

/// A chunk from a streaming completion response.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamChunk {
    /// A text content delta.
    Delta { content: String },
    /// A tool call delta.
    ToolCallDelta {
        index: u32,
        id: Option<String>,
        name: Option<String>,
        arguments: String,
    },
    /// Stream is complete.
    Done,
}

/// Raw `OpenAI` API response structure for deserialization.
#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiResponse {
    pub choices: Vec<OpenAiChoice>,
    pub model: String,
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiChoice {
    pub message: OpenAiMessage,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiMessage {
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<OpenAiToolCall>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiToolCall {
    pub id: String,
    pub function: OpenAiFunction,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiFunction {
    pub name: String,
    pub arguments: String,
}

impl OpenAiResponse {
    /// Converts the raw `OpenAI` response to a `CompletionResponse`.
    pub fn into_completion_response(self) -> Result<CompletionResponse, crate::LlmError> {
        let choice = self
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| crate::LlmError::InvalidResponse("empty choices array".into()))?;

        let tool_calls = choice
            .message
            .tool_calls
            .into_iter()
            .map(|tc| {
                let arguments: serde_json::Value =
                    serde_json::from_str(&tc.function.arguments).unwrap_or_default();
                ToolCall {
                    id: tc.id,
                    name: tc.function.name,
                    arguments,
                }
            })
            .collect();

        Ok(CompletionResponse {
            content: choice.message.content,
            tool_calls,
            model: self.model,
            usage: self.usage,
        })
    }
}
