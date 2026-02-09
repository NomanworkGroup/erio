//! Configuration for embedding engines.

/// Configuration for an embedding engine.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Output vector dimensions.
    pub dimensions: usize,
    /// Maximum number of texts to embed in a single batch.
    pub batch_size: usize,
    /// Maximum input text length in tokens.
    pub max_input_length: usize,
    /// Whether to L2-normalize output vectors to unit length.
    pub normalize: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            dimensions: 384,
            batch_size: 32,
            max_input_length: 512,
            normalize: true,
        }
    }
}

/// Builder for `EmbeddingConfig`.
#[derive(Debug)]
#[must_use]
pub struct EmbeddingConfigBuilder {
    config: EmbeddingConfig,
}

impl EmbeddingConfig {
    /// Creates a builder with default values.
    pub fn builder() -> EmbeddingConfigBuilder {
        EmbeddingConfigBuilder {
            config: Self::default(),
        }
    }
}

impl EmbeddingConfigBuilder {
    pub fn dimensions(mut self, dimensions: usize) -> Self {
        self.config.dimensions = dimensions;
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    pub fn max_input_length(mut self, max_input_length: usize) -> Self {
        self.config.max_input_length = max_input_length;
        self
    }

    pub fn normalize(mut self, normalize: bool) -> Self {
        self.config.normalize = normalize;
        self
    }

    pub fn build(self) -> EmbeddingConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_sensible_values() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.dimensions, 384);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.max_input_length, 512);
        assert!(config.normalize);
    }

    #[test]
    fn builder_sets_dimensions() {
        let config = EmbeddingConfig::builder().dimensions(768).build();
        assert_eq!(config.dimensions, 768);
    }

    #[test]
    fn builder_sets_batch_size() {
        let config = EmbeddingConfig::builder().batch_size(64).build();
        assert_eq!(config.batch_size, 64);
    }

    #[test]
    fn builder_sets_max_input_length() {
        let config = EmbeddingConfig::builder()
            .max_input_length(1024)
            .build();
        assert_eq!(config.max_input_length, 1024);
    }

    #[test]
    fn builder_sets_normalize() {
        let config = EmbeddingConfig::builder().normalize(false).build();
        assert!(!config.normalize);
    }

    #[test]
    fn builder_chains_all_fields() {
        let config = EmbeddingConfig::builder()
            .dimensions(768)
            .batch_size(16)
            .max_input_length(256)
            .normalize(false)
            .build();
        assert_eq!(config.dimensions, 768);
        assert_eq!(config.batch_size, 16);
        assert_eq!(config.max_input_length, 256);
        assert!(!config.normalize);
    }
}
