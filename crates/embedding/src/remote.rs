//! Remote embedding engine calling an OpenAI-compatible embedding API.

use crate::config::EmbeddingConfig;
use crate::engine::EmbeddingEngine;
use crate::error::EmbeddingError;

/// Response format from an OpenAI-compatible embedding API.
#[derive(serde::Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(serde::Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

/// Remote embedding engine that calls an OpenAI-compatible embedding API.
pub struct RemoteEmbedding {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    config: EmbeddingConfig,
}

impl RemoteEmbedding {
    /// Creates a new `RemoteEmbedding` with the given base URL and API key.
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        config: EmbeddingConfig,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            api_key: api_key.into(),
            config,
        }
    }

    /// Sets a custom reqwest client (e.g. for testing with `no_proxy()`).
    #[must_use]
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    async fn post_embeddings(
        &self,
        input: serde_json::Value,
    ) -> Result<EmbeddingResponse, EmbeddingError> {
        let url = format!("{}/v1/embeddings", self.base_url);
        let body = serde_json::json!({
            "model": self.config.model_id,
            "input": input,
        });

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| EmbeddingError::Inference(format!("request failed: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            let body_text = response.text().await.unwrap_or_else(|_| "unknown".into());
            return Err(EmbeddingError::Inference(format!(
                "API error {status}: {body_text}"
            )));
        }

        response
            .json::<EmbeddingResponse>()
            .await
            .map_err(|e| EmbeddingError::Inference(format!("failed to parse response: {e}")))
    }
}

#[async_trait::async_trait]
impl EmbeddingEngine for RemoteEmbedding {
    fn name(&self) -> &'static str {
        "remote"
    }

    fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.is_empty() {
            return Err(EmbeddingError::InvalidInput(
                "text must not be empty".into(),
            ));
        }
        let resp = self
            .post_embeddings(serde_json::Value::String(text.to_owned()))
            .await?;
        resp.data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .ok_or_else(|| EmbeddingError::Inference("empty response data".into()))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.iter().any(|t| t.is_empty()) {
            return Err(EmbeddingError::InvalidInput(
                "text must not be empty".into(),
            ));
        }
        let input: Vec<serde_json::Value> = texts
            .iter()
            .map(|t| serde_json::Value::String((*t).to_owned()))
            .collect();
        let resp = self
            .post_embeddings(serde_json::Value::Array(input))
            .await?;
        Ok(resp.data.into_iter().map(|d| d.embedding).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn no_proxy_client() -> reqwest::Client {
        reqwest::Client::builder().no_proxy().build().unwrap()
    }

    fn test_config() -> EmbeddingConfig {
        EmbeddingConfig::builder()
            .model_id("text-embedding-test")
            .dimensions(3)
            .build()
    }

    fn mock_response(embeddings: Vec<Vec<f32>>) -> serde_json::Value {
        let data: Vec<serde_json::Value> = embeddings
            .into_iter()
            .enumerate()
            .map(|(i, emb)| {
                serde_json::json!({
                    "embedding": emb,
                    "index": i,
                    "object": "embedding"
                })
            })
            .collect();
        serde_json::json!({
            "data": data,
            "model": "text-embedding-test",
            "object": "list",
            "usage": {"prompt_tokens": 5, "total_tokens": 5}
        })
    }

    #[test]
    fn remote_returns_name() {
        let engine = RemoteEmbedding::new("http://localhost", "key", test_config());
        assert_eq!(engine.name(), "remote");
    }

    #[test]
    fn remote_returns_correct_dimensions() {
        let engine = RemoteEmbedding::new("http://localhost", "key", test_config());
        assert_eq!(engine.dimensions(), 3);
    }

    #[tokio::test]
    async fn remote_rejects_empty_input() {
        let engine = RemoteEmbedding::new("http://localhost", "key", test_config());
        let result = engine.embed("").await;
        assert!(matches!(
            result.unwrap_err(),
            EmbeddingError::InvalidInput(_)
        ));
    }

    #[tokio::test]
    async fn remote_sends_correct_request() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .and(header("Authorization", "Bearer test-key"))
            .and(header("Content-Type", "application/json"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(mock_response(vec![vec![0.1, 0.2, 0.3]])),
            )
            .expect(1)
            .mount(&server)
            .await;

        let engine = RemoteEmbedding::new(server.uri(), "test-key", test_config())
            .with_client(no_proxy_client());
        let result = engine.embed("hello").await.unwrap();
        assert_eq!(result, vec![0.1, 0.2, 0.3]);
    }

    #[tokio::test]
    async fn remote_parses_openai_embedding_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(mock_response(vec![vec![1.0, 2.0, 3.0]])),
            )
            .mount(&server)
            .await;

        let engine =
            RemoteEmbedding::new(server.uri(), "key", test_config()).with_client(no_proxy_client());
        let result = engine.embed("test").await.unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn remote_embed_batch_sends_array_input() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_response(vec![
                vec![0.1, 0.2, 0.3],
                vec![0.4, 0.5, 0.6],
            ])))
            .mount(&server)
            .await;

        let engine =
            RemoteEmbedding::new(server.uri(), "key", test_config()).with_client(no_proxy_client());
        let results = engine.embed_batch(&["hello", "world"]).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], vec![0.1, 0.2, 0.3]);
        assert_eq!(results[1], vec![0.4, 0.5, 0.6]);
    }

    #[tokio::test]
    async fn remote_returns_error_on_401() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(401).set_body_string("Unauthorized"))
            .mount(&server)
            .await;

        let engine = RemoteEmbedding::new(server.uri(), "bad-key", test_config())
            .with_client(no_proxy_client());
        let result = engine.embed("test").await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("401"),
            "Expected 401 in error: {err}"
        );
    }

    #[tokio::test]
    async fn remote_returns_error_on_500() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
            .mount(&server)
            .await;

        let engine =
            RemoteEmbedding::new(server.uri(), "key", test_config()).with_client(no_proxy_client());
        let result = engine.embed("test").await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("500"),
            "Expected 500 in error: {err}"
        );
    }
}
