//! Erio LLM Client - LLM provider abstraction and adapters for the agent runtime.

pub mod error;
pub mod openai;
pub mod provider;
pub mod request;
pub mod response;

pub use error::LlmError;
pub use openai::OpenAiProvider;
pub use provider::LlmProvider;
pub use request::{CompletionRequest, ToolDefinition};
pub use response::{CompletionResponse, StreamChunk, Usage};

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // === LlmError Display Tests ===

    #[test]
    fn llm_error_rate_limited_displays_message() {
        let err = LlmError::RateLimited { retry_after: None };
        assert_eq!(err.to_string(), "Rate limited");
    }

    #[test]
    fn llm_error_api_displays_status_and_message() {
        let err = LlmError::Api {
            status: 500,
            message: "internal server error".into(),
        };
        assert_eq!(err.to_string(), "API error (500): internal server error");
    }

    #[test]
    fn llm_error_invalid_response_displays_detail() {
        let err = LlmError::InvalidResponse("missing choices field".into());
        assert_eq!(err.to_string(), "Invalid response: missing choices field");
    }

    #[test]
    fn llm_error_timeout_displays_message() {
        let err = LlmError::Timeout;
        assert_eq!(err.to_string(), "Request timeout");
    }

    #[test]
    fn llm_error_auth_displays_message() {
        let err = LlmError::Auth;
        assert_eq!(err.to_string(), "Authentication failed");
    }

    #[test]
    fn llm_error_network_displays_detail() {
        let err = LlmError::Network("connection refused".into());
        assert_eq!(err.to_string(), "Network error: connection refused");
    }

    // === Retryable Tests ===

    #[test]
    fn llm_error_rate_limited_is_retryable() {
        let err = LlmError::RateLimited {
            retry_after: Some(Duration::from_secs(5)),
        };
        assert!(err.is_retryable());
    }

    #[test]
    fn llm_error_timeout_is_retryable() {
        let err = LlmError::Timeout;
        assert!(err.is_retryable());
    }

    #[test]
    fn llm_error_network_is_retryable() {
        let err = LlmError::Network("connection reset".into());
        assert!(err.is_retryable());
    }

    #[test]
    fn llm_error_api_500_is_retryable() {
        let err = LlmError::Api {
            status: 500,
            message: "server error".into(),
        };
        assert!(err.is_retryable());
    }

    #[test]
    fn llm_error_auth_is_not_retryable() {
        let err = LlmError::Auth;
        assert!(!err.is_retryable());
    }

    #[test]
    fn llm_error_invalid_response_is_not_retryable() {
        let err = LlmError::InvalidResponse("bad json".into());
        assert!(!err.is_retryable());
    }

    #[test]
    fn llm_error_api_400_is_not_retryable() {
        let err = LlmError::Api {
            status: 400,
            message: "bad request".into(),
        };
        assert!(!err.is_retryable());
    }

    // === OpenAiProvider Tests (wiremock) ===

    fn valid_openai_response() -> serde_json::Value {
        serde_json::json!({
            "choices": [{
                "message": {
                    "content": "Hello from GPT!",
                    "tool_calls": []
                }
            }],
            "model": "gpt-4",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        })
    }

    fn tool_call_openai_response() -> serde_json::Value {
        serde_json::json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"London\"}"
                        }
                    }]
                }
            }],
            "model": "gpt-4",
            "usage": null
        })
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn openai_provider_returns_name() {
        let provider = OpenAiProvider::new("http://localhost", "key");
        assert_eq!(provider.name(), "openai");
    }

    /// Creates a reqwest client that bypasses proxy for test use.
    fn no_proxy_client() -> reqwest::Client {
        reqwest::Client::builder().no_proxy().build().unwrap()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn openai_provider_sends_correct_request() {
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(valid_openai_response()))
            .expect(1)
            .mount(&mock_server)
            .await;

        let provider = OpenAiProvider::new(mock_server.uri(), "test-key")
            .with_client(no_proxy_client())
            .with_retry(erio_core::RetryConfig::no_retry());
        let request = CompletionRequest::new("gpt-4").message(erio_core::Message::user("Hello"));

        let response = provider.complete(request).await.unwrap();
        assert_eq!(response.content, Some("Hello from GPT!".into()));
        assert_eq!(response.model, "gpt-4");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn openai_provider_parses_tool_calls() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(tool_call_openai_response()))
            .mount(&mock_server)
            .await;

        let provider = OpenAiProvider::new(mock_server.uri(), "key")
            .with_client(no_proxy_client())
            .with_retry(erio_core::RetryConfig::no_retry());
        let request = CompletionRequest::new("gpt-4")
            .message(erio_core::Message::user("What's the weather?"));

        let response = provider.complete(request).await.unwrap();
        assert!(response.content.is_none());
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].name, "get_weather");
        assert_eq!(response.tool_calls[0].arguments["city"], "London");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn openai_provider_returns_auth_error_on_401() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&mock_server)
            .await;

        let provider = OpenAiProvider::new(mock_server.uri(), "bad-key")
            .with_client(no_proxy_client())
            .with_retry(erio_core::RetryConfig::no_retry());
        let request = CompletionRequest::new("gpt-4").message(erio_core::Message::user("Hello"));

        let result = provider.complete(request).await;
        assert!(matches!(result, Err(LlmError::Auth)));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn openai_provider_returns_rate_limited_on_429() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(429))
            .mount(&mock_server)
            .await;

        let provider = OpenAiProvider::new(mock_server.uri(), "key")
            .with_client(no_proxy_client())
            .with_retry(erio_core::RetryConfig::no_retry());
        let request = CompletionRequest::new("gpt-4").message(erio_core::Message::user("Hello"));

        let result = provider.complete(request).await;
        assert!(matches!(result, Err(LlmError::RateLimited { .. })));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn openai_provider_returns_api_error_on_500() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&mock_server)
            .await;

        let provider = OpenAiProvider::new(mock_server.uri(), "key")
            .with_client(no_proxy_client())
            .with_retry(erio_core::RetryConfig::no_retry());
        let request = CompletionRequest::new("gpt-4").message(erio_core::Message::user("Hello"));

        let result = provider.complete(request).await;
        assert!(matches!(result, Err(LlmError::Api { status: 500, .. })));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn openai_provider_retries_on_429_then_succeeds() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        // First two calls return 429, third succeeds
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(429))
            .up_to_n_times(2)
            .expect(2)
            .mount(&mock_server)
            .await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(valid_openai_response()))
            .expect(1)
            .mount(&mock_server)
            .await;

        let provider = OpenAiProvider::new(mock_server.uri(), "key")
            .with_client(no_proxy_client())
            .with_retry(
                erio_core::RetryConfig::builder()
                    .max_attempts(3)
                    .initial_delay(Duration::from_millis(1))
                    .build(),
            );
        let request = CompletionRequest::new("gpt-4").message(erio_core::Message::user("Hello"));

        let response = provider.complete(request).await.unwrap();
        assert_eq!(response.content, Some("Hello from GPT!".into()));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn openai_provider_does_not_retry_on_401() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(401))
            .expect(1) // Should only be called once - no retry
            .mount(&mock_server)
            .await;

        let provider = OpenAiProvider::new(mock_server.uri(), "bad-key")
            .with_client(no_proxy_client())
            .with_retry(
                erio_core::RetryConfig::builder()
                    .max_attempts(3)
                    .initial_delay(Duration::from_millis(1))
                    .build(),
            );
        let request = CompletionRequest::new("gpt-4").message(erio_core::Message::user("Hello"));

        let result = provider.complete(request).await;
        assert!(matches!(result, Err(LlmError::Auth)));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn openai_provider_exhausts_retries_on_persistent_429() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(429))
            .expect(3) // 1 initial + 2 retries
            .mount(&mock_server)
            .await;

        let provider = OpenAiProvider::new(mock_server.uri(), "key")
            .with_client(no_proxy_client())
            .with_retry(
                erio_core::RetryConfig::builder()
                    .max_attempts(3)
                    .initial_delay(Duration::from_millis(1))
                    .build(),
            );
        let request = CompletionRequest::new("gpt-4").message(erio_core::Message::user("Hello"));

        let result = provider.complete(request).await;
        assert!(matches!(result, Err(LlmError::RateLimited { .. })));
    }

    // === CompletionResponse Tests ===

    #[test]
    fn response_parses_openai_text_content() {
        let json = serde_json::json!({
            "choices": [{
                "message": {
                    "content": "Hello, world!",
                    "tool_calls": []
                }
            }],
            "model": "gpt-4",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        });

        let raw: response::OpenAiResponse = serde_json::from_value(json).unwrap();
        let resp = raw.into_completion_response().unwrap();

        assert_eq!(resp.content, Some("Hello, world!".into()));
        assert!(resp.tool_calls.is_empty());
        assert_eq!(resp.model, "gpt-4");
        assert_eq!(resp.usage.as_ref().unwrap().prompt_tokens, 10);
        assert_eq!(resp.usage.as_ref().unwrap().completion_tokens, 5);
        assert_eq!(resp.usage.as_ref().unwrap().total_tokens, 15);
    }

    #[test]
    fn response_parses_openai_tool_calls() {
        let json = serde_json::json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"London\"}"
                        }
                    }]
                }
            }],
            "model": "gpt-4",
            "usage": null
        });

        let raw: response::OpenAiResponse = serde_json::from_value(json).unwrap();
        let resp = raw.into_completion_response().unwrap();

        assert!(resp.content.is_none());
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].id, "call_123");
        assert_eq!(resp.tool_calls[0].name, "get_weather");
        assert_eq!(resp.tool_calls[0].arguments["city"], "London");
    }

    #[test]
    fn response_parses_openai_multiple_tool_calls() {
        let json = serde_json::json!({
            "choices": [{
                "message": {
                    "content": "Let me check both.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"city\":\"London\"}"
                            }
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "get_time",
                                "arguments": "{\"timezone\":\"UTC\"}"
                            }
                        }
                    ]
                }
            }],
            "model": "gpt-4",
            "usage": null
        });

        let raw: response::OpenAiResponse = serde_json::from_value(json).unwrap();
        let resp = raw.into_completion_response().unwrap();

        assert_eq!(resp.content, Some("Let me check both.".into()));
        assert_eq!(resp.tool_calls.len(), 2);
        assert_eq!(resp.tool_calls[0].name, "get_weather");
        assert_eq!(resp.tool_calls[1].name, "get_time");
    }

    #[test]
    fn response_returns_error_for_empty_choices() {
        let json = serde_json::json!({
            "choices": [],
            "model": "gpt-4",
            "usage": null
        });

        let raw: response::OpenAiResponse = serde_json::from_value(json).unwrap();
        let result = raw.into_completion_response();

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LlmError::InvalidResponse(_)));
    }

    #[test]
    fn response_handles_no_usage() {
        let json = serde_json::json!({
            "choices": [{
                "message": {
                    "content": "OK",
                    "tool_calls": []
                }
            }],
            "model": "gpt-4"
        });

        let raw: response::OpenAiResponse = serde_json::from_value(json).unwrap();
        let resp = raw.into_completion_response().unwrap();

        assert!(resp.usage.is_none());
    }

    // === StreamChunk Tests ===

    #[test]
    fn stream_chunk_delta_holds_content() {
        let chunk = StreamChunk::Delta {
            content: "Hello".into(),
        };
        assert_eq!(
            chunk,
            StreamChunk::Delta {
                content: "Hello".into()
            }
        );
    }

    #[test]
    fn stream_chunk_done_variant() {
        let chunk = StreamChunk::Done;
        assert_eq!(chunk, StreamChunk::Done);
    }

    // === CompletionRequest Tests ===

    #[test]
    fn request_new_sets_model() {
        let req = CompletionRequest::new("gpt-4");
        assert_eq!(req.model, "gpt-4");
        assert!(req.messages.is_empty());
        assert!(req.tools.is_none());
        assert!(req.max_tokens.is_none());
        assert!(req.temperature.is_none());
        assert!(!req.stream);
    }

    #[test]
    fn request_builder_adds_message() {
        let req = CompletionRequest::new("gpt-4").message(erio_core::Message::user("Hello"));
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].text(), Some("Hello"));
    }

    #[test]
    fn request_builder_chains_messages() {
        let req = CompletionRequest::new("gpt-4")
            .message(erio_core::Message::system("You are helpful"))
            .message(erio_core::Message::user("Hi"));
        assert_eq!(req.messages.len(), 2);
    }

    #[test]
    fn request_builder_sets_temperature() {
        let req = CompletionRequest::new("gpt-4").temperature(0.7);
        assert_eq!(req.temperature, Some(0.7));
    }

    #[test]
    fn request_builder_sets_max_tokens() {
        let req = CompletionRequest::new("gpt-4").max_tokens(1024);
        assert_eq!(req.max_tokens, Some(1024));
    }

    #[test]
    fn request_builder_sets_tools() {
        let tools = vec![ToolDefinition {
            name: "shell".into(),
            description: "Run a shell command".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"]
            }),
        }];
        let req = CompletionRequest::new("gpt-4").tools(tools);
        assert_eq!(req.tools.as_ref().unwrap().len(), 1);
        assert_eq!(req.tools.as_ref().unwrap()[0].name, "shell");
    }

    #[test]
    fn request_builder_sets_stream() {
        let req = CompletionRequest::new("gpt-4").stream(true);
        assert!(req.stream);
    }

    // === CoreError Conversion ===

    #[test]
    fn llm_error_converts_to_core_error() {
        let llm_err = LlmError::Api {
            status: 500,
            message: "server error".into(),
        };
        let core_err: erio_core::CoreError = llm_err.into();
        assert!(matches!(core_err, erio_core::CoreError::Llm { .. }));
    }

    #[test]
    fn llm_error_rate_limited_converts_with_429_status() {
        let llm_err = LlmError::RateLimited { retry_after: None };
        let core_err: erio_core::CoreError = llm_err.into();
        match core_err {
            erio_core::CoreError::Llm { status, .. } => {
                assert_eq!(status, Some(429));
            }
            _ => panic!("Expected CoreError::Llm"),
        }
    }

    #[test]
    fn llm_error_auth_converts_with_401_status() {
        let llm_err = LlmError::Auth;
        let core_err: erio_core::CoreError = llm_err.into();
        match core_err {
            erio_core::CoreError::Llm { status, .. } => {
                assert_eq!(status, Some(401));
            }
            _ => panic!("Expected CoreError::Llm"),
        }
    }
}
