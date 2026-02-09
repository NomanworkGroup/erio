//! In-memory deterministic embedding engine for testing and development.

use crate::config::EmbeddingConfig;
use crate::engine::EmbeddingEngine;
use crate::error::EmbeddingError;

/// A deterministic, hash-based embedding engine useful for testing.
///
/// Produces consistent embeddings by hashing input text, so the same input
/// always yields the same output vector. No ML model is required.
pub struct InMemoryEmbedding {
    config: EmbeddingConfig,
}

impl InMemoryEmbedding {
    /// Creates a new `InMemoryEmbedding` with the given configuration.
    pub fn new(config: EmbeddingConfig) -> Self {
        Self { config }
    }

    /// Simple hash function for deterministic float generation.
    fn hash_text(text: &str) -> u64 {
        // FNV-1a hash
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in text.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        hash
    }

    /// Generates a deterministic float vector from text.
    fn generate_vector(&self, text: &str) -> Vec<f32> {
        let dims = self.config.dimensions;
        let mut vec = Vec::with_capacity(dims);
        let mut hash = Self::hash_text(text);

        for _ in 0..dims {
            // Map hash bits to a float in [-1.0, 1.0]
            #[allow(clippy::cast_precision_loss)] // intentional: 16-bit value fits in f32
            let val = ((hash & 0xFFFF) as f32 / 32768.0) - 1.0;
            vec.push(val);
            // Advance the hash for the next dimension
            hash = hash.wrapping_mul(0x0100_0000_01b3).wrapping_add(0x9e37_79b9_7f4a_7c15);
        }

        if self.config.normalize {
            let magnitude: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if magnitude > 0.0 {
                for x in &mut vec {
                    *x /= magnitude;
                }
            }
        }

        vec
    }
}

#[async_trait::async_trait]
impl EmbeddingEngine for InMemoryEmbedding {
    #[allow(clippy::unnecessary_literal_bound)] // trait signature uses &str
    fn name(&self) -> &str {
        "in-memory"
    }

    fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.is_empty() {
            return Err(EmbeddingError::InvalidInput("text must not be empty".into()));
        }
        Ok(self.generate_vector(text))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        texts.iter().map(|text| {
            if text.is_empty() {
                return Err(EmbeddingError::InvalidInput("text must not be empty".into()));
            }
            Ok(self.generate_vector(text))
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> EmbeddingConfig {
        EmbeddingConfig::builder()
            .dimensions(8)
            .normalize(true)
            .build()
    }

    #[test]
    fn returns_correct_dimensions() {
        let engine = InMemoryEmbedding::new(test_config());
        assert_eq!(engine.dimensions(), 8);
    }

    #[tokio::test]
    async fn embed_returns_vector_of_configured_dimensions() {
        let engine = InMemoryEmbedding::new(test_config());
        let vec = engine.embed("hello world").await.unwrap();
        assert_eq!(vec.len(), 8);
    }

    #[tokio::test]
    async fn same_input_produces_same_output() {
        let engine = InMemoryEmbedding::new(test_config());
        let v1 = engine.embed("hello").await.unwrap();
        let v2 = engine.embed("hello").await.unwrap();
        assert_eq!(v1, v2);
    }

    #[tokio::test]
    async fn different_inputs_produce_different_outputs() {
        let engine = InMemoryEmbedding::new(test_config());
        let v1 = engine.embed("hello").await.unwrap();
        let v2 = engine.embed("world").await.unwrap();
        assert_ne!(v1, v2);
    }

    #[tokio::test]
    async fn embed_batch_returns_correct_count() {
        let engine = InMemoryEmbedding::new(test_config());
        let results = engine.embed_batch(&["a", "b", "c"]).await.unwrap();
        assert_eq!(results.len(), 3);
        for vec in &results {
            assert_eq!(vec.len(), 8);
        }
    }

    #[tokio::test]
    async fn embed_batch_preserves_order() {
        let engine = InMemoryEmbedding::new(test_config());
        let individual_a = engine.embed("alpha").await.unwrap();
        let individual_b = engine.embed("beta").await.unwrap();
        let batch = engine.embed_batch(&["alpha", "beta"]).await.unwrap();
        assert_eq!(batch[0], individual_a);
        assert_eq!(batch[1], individual_b);
    }

    #[tokio::test]
    async fn rejects_empty_input() {
        let engine = InMemoryEmbedding::new(test_config());
        let result = engine.embed("").await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            EmbeddingError::InvalidInput(_)
        ));
    }

    #[tokio::test]
    async fn normalized_output_has_unit_length() {
        let config = EmbeddingConfig::builder()
            .dimensions(8)
            .normalize(true)
            .build();
        let engine = InMemoryEmbedding::new(config);
        let vec = engine.embed("test normalization").await.unwrap();
        let magnitude: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (magnitude - 1.0).abs() < 1e-5,
            "Expected unit length, got {magnitude}"
        );
    }
}
