//! Embedding engine trait definition.

use crate::error::EmbeddingError;

/// An engine that computes vector embeddings from text.
#[async_trait::async_trait]
pub trait EmbeddingEngine: Send + Sync {
    /// Returns the engine name (e.g., "in-memory", "candle").
    fn name(&self) -> &str;

    /// Returns the output vector dimensions.
    fn dimensions(&self) -> usize;

    /// Computes an embedding vector for a single text input.
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;

    /// Computes embedding vectors for a batch of text inputs.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError>;
}
