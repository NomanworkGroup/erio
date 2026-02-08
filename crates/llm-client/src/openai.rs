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

    fn build_body(request: &CompletionRequest) -> serde_json::Value {
        let mut body = serde_json::json!({
            "model": request.model,
            "messages": request.messages,
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

    async fn send_request(
        &self,
        body: &serde_json::Value,
    ) -> Result<CompletionResponse, LlmError> {
        let url = format!("{}/v1/chat/completions", self.base_url);

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
            429 => Err(LlmError::RateLimited { retry_after: None }),
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
