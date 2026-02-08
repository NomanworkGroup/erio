//! Completion request types for LLM providers.

use erio_core::Message;
use serde::Serialize;

/// Definition of a tool that the LLM can call.
#[derive(Debug, Clone, Serialize)]
pub struct ToolDefinition {
    /// Unique name of the tool.
    pub name: String,
    /// Human-readable description of what the tool does.
    pub description: String,
    /// JSON Schema describing the tool's parameters.
    pub parameters: serde_json::Value,
}

/// A request to complete a conversation.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    /// Model identifier (e.g., "gpt-4", "claude-3-opus").
    pub model: String,
    /// Conversation messages.
    pub messages: Vec<Message>,
    /// Available tools for function calling.
    pub tools: Option<Vec<ToolDefinition>>,
    /// Maximum tokens in the response.
    pub max_tokens: Option<u32>,
    /// Sampling temperature (0.0 - 2.0).
    pub temperature: Option<f32>,
    /// Whether to stream the response.
    pub stream: bool,
}

impl CompletionRequest {
    /// Creates a new request for the given model.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            messages: Vec::new(),
            tools: None,
            max_tokens: None,
            temperature: None,
            stream: false,
        }
    }

    /// Adds a message to the conversation.
    #[must_use]
    pub fn message(mut self, msg: Message) -> Self {
        self.messages.push(msg);
        self
    }

    /// Sets the temperature.
    #[must_use]
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Sets the maximum number of tokens.
    #[must_use]
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Sets the available tools.
    #[must_use]
    pub fn tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Sets whether to stream the response.
    #[must_use]
    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }
}
