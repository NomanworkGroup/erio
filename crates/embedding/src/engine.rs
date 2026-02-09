//! Embedding engine trait definition.

use crate::error::EmbeddingError;

/// An engine that computes vector embeddings from text.
#[async_trait::async_trait]
pub trait EmbeddingEngine: Send + Sync {
    /// Returns the engine name (e.g., "gemma", "remote").
    fn name(&self) -> &str;

    /// Returns the output vector dimensions.
    fn dimensions(&self) -> usize;

    /// Computes an embedding vector for a single text input.
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;

    /// Computes embedding vectors for a batch of text inputs.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// A configurable mock embedding engine for tests.
    struct MockEngine {
        embed_fn: Arc<dyn Fn(&str) -> Result<Vec<f32>, EmbeddingError> + Send + Sync>,
        dims: usize,
    }

    impl MockEngine {
        fn returning_ok(dims: usize, values: Vec<f32>) -> Self {
            Self {
                embed_fn: Arc::new(move |_| Ok(values.clone())),
                dims,
            }
        }

        fn returning_err(err_fn: impl Fn() -> EmbeddingError + Send + Sync + 'static) -> Self {
            Self {
                embed_fn: Arc::new(move |_| Err(err_fn())),
                dims: 0,
            }
        }
    }

    #[async_trait::async_trait]
    impl EmbeddingEngine for MockEngine {
        fn name(&self) -> &str {
            "mock"
        }

        fn dimensions(&self) -> usize {
            self.dims
        }

        async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
            (self.embed_fn)(text)
        }

        async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            texts.iter().map(|t| (self.embed_fn)(t)).collect()
        }
    }

    #[tokio::test]
    async fn mock_engine_returns_configured_embedding() {
        let mock = MockEngine::returning_ok(3, vec![0.1, 0.2, 0.3]);
        assert_eq!(mock.dimensions(), 3);
        let result = mock.embed("hello").await.unwrap();
        assert_eq!(result, vec![0.1, 0.2, 0.3]);
    }

    #[tokio::test]
    async fn mock_engine_can_simulate_errors() {
        let mock = MockEngine::returning_err(|| {
            EmbeddingError::Inference("simulated failure".into())
        });
        let result = mock.embed("test").await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), EmbeddingError::Inference(_)));
    }
}
