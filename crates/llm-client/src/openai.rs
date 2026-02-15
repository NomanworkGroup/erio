//! OpenAI-compatible LLM provider.

use crate::error::LlmError;
use crate::provider::LlmProvider;
use crate::request::CompletionRequest;
use crate::response::{CompletionResponse, OpenAiResponse};
use erio_core::RetryConfig;

/// Provider for `OpenAI`-compatible APIs (`OpenAI`, Azure, Groq, Ollama, vLLM).
pub struct OpenAiProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    retry_config: RetryConfig,
}

impl OpenAiProvider {
    /// Creates a new provider with the given base URL and API key.
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            api_key: api_key.into(),
            retry_config: RetryConfig::default(),
        }
    }

    /// Sets a custom reqwest client.
    #[must_use]
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    /// Sets the retry configuration.
    #[must_use]
    pub fn with_retry(mut self, config: RetryConfig) -> Self {
        self.retry_config = config;
        self
    }

    /// Converts an internal [`erio_core::Message`] to the JSON shape the
    /// OpenAI chat-completions API expects.
    ///
    /// Key differences from the naive serde output:
    /// - Assistant tool-call content is split into a top-level `tool_calls`
    ///   array (with `type: "function"` and stringified arguments).
    /// - Tool-result messages use `tool_call_id` + plain-string `content`.
    fn message_to_openai_json(msg: &erio_core::Message) -> serde_json::Value {
        use erio_core::Role;

        match msg.role {
            Role::Assistant => {
                let text: Option<String> = msg
                    .content
                    .iter()
                    .filter_map(|c| c.as_text().map(String::from))
                    .reduce(|a, b| format!("{a}\n{b}"));

                let mut obj = serde_json::json!({ "role": "assistant" });
                obj["content"] = text.map_or(serde_json::Value::Null, |t| serde_json::json!(t));

                let tool_calls: Vec<serde_json::Value> = msg
                    .tool_calls()
                    .map(|tc| {
                        serde_json::json!({
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments.to_string(),
                            }
                        })
                    })
                    .collect();
                if !tool_calls.is_empty() {
                    obj["tool_calls"] = serde_json::json!(tool_calls);
                }

                obj
            }
            Role::Tool => {
                serde_json::json!({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.text().unwrap_or(""),
                })
            }
            role => {
                serde_json::json!({
                    "role": role,
                    "content": msg.text().unwrap_or(""),
                })
            }
        }
    }

    fn build_body(request: &CompletionRequest) -> serde_json::Value {
        let messages: Vec<serde_json::Value> = request
            .messages
            .iter()
            .map(Self::message_to_openai_json)
            .collect();

        let mut body = serde_json::json!({
            "model": request.model,
            "messages": messages,
        });

        if let Some(temp) = request.temperature {
            body["temperature"] = serde_json::json!(temp);
        }
        if let Some(max) = request.max_tokens {
            body["max_tokens"] = serde_json::json!(max);
        }
        if let Some(tools) = &request.tools {
            let tool_defs: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(tool_defs);
        }

        body
    }

    async fn send_request(&self, body: &serde_json::Value) -> Result<CompletionResponse, LlmError> {
        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(body)
            .send()
            .await
            .map_err(|e| LlmError::Network(e.to_string()))?;

        let status = response.status().as_u16();

        match status {
            200..=299 => {
                let openai_response: OpenAiResponse = response
                    .json()
                    .await
                    .map_err(|e| LlmError::InvalidResponse(e.to_string()))?;
                openai_response.into_completion_response()
            }
            401 => Err(LlmError::Auth),
            429 => {
                let message = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "rate limited".into());
                // OpenAI returns 429 for both rate limits and quota exceeded.
                // Quota errors should not be retried.
                if message.contains("insufficient_quota") {
                    Err(LlmError::Api { status, message })
                } else {
                    Err(LlmError::RateLimited { retry_after: None })
                }
            }
            _ => {
                let message = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "unknown error".into());
                Err(LlmError::Api { status, message })
            }
        }
    }
}

#[async_trait::async_trait]
impl LlmProvider for OpenAiProvider {
    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "openai"
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let body = Self::build_body(&request);

        let mut last_error = None;

        for attempt in 0..self.retry_config.max_attempts {
            match self.send_request(&body).await {
                Ok(response) => return Ok(response),
                Err(err) => {
                    if !err.is_retryable() || attempt + 1 >= self.retry_config.max_attempts {
                        return Err(err);
                    }
                    let delay = self.retry_config.delay_for_attempt(attempt);
                    tokio::time::sleep(delay).await;
                    last_error = Some(err);
                }
            }
        }

        Err(last_error.unwrap_or(LlmError::Network("no attempts made".into())))
    }
}
