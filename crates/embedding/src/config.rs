//! Configuration for embedding engines.

use crate::task::TaskType;

/// Configuration for an embedding engine.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// `HuggingFace` model identifier (e.g. "ggml-org/embeddinggemma-300M-GGUF").
    pub model_id: String,
    /// Output vector dimensions.
    pub dimensions: usize,
    /// Maximum number of texts to embed in a single batch.
    pub batch_size: usize,
    /// Maximum input text length in tokens.
    pub max_input_length: usize,
    /// Whether to L2-normalize output vectors to unit length.
    pub normalize: bool,
    /// Task type for prompt formatting.
    pub task_type: TaskType,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_id: "ggml-org/embeddinggemma-300M-GGUF".to_owned(),
            dimensions: 768,
            batch_size: 32,
            max_input_length: 2048,
            normalize: true,
            task_type: TaskType::default(),
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
    pub fn model_id(mut self, model_id: impl Into<String>) -> Self {
        self.config.model_id = model_id.into();
        self
    }

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

    pub fn task_type(mut self, task_type: TaskType) -> Self {
        self.config.task_type = task_type;
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
        assert_eq!(config.model_id, "ggml-org/embeddinggemma-300M-GGUF");
        assert_eq!(config.dimensions, 768);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.max_input_length, 2048);
        assert!(config.normalize);
        assert_eq!(config.task_type, TaskType::SearchResult);
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
        let config = EmbeddingConfig::builder().max_input_length(1024).build();
        assert_eq!(config.max_input_length, 1024);
    }

    #[test]
    fn builder_sets_normalize() {
        let config = EmbeddingConfig::builder().normalize(false).build();
        assert!(!config.normalize);
    }

    #[test]
    fn builder_sets_model_id() {
        let config = EmbeddingConfig::builder().model_id("custom/model").build();
        assert_eq!(config.model_id, "custom/model");
    }

    #[test]
    fn builder_sets_task_type() {
        let config = EmbeddingConfig::builder()
            .task_type(TaskType::CodeRetrieval)
            .build();
        assert_eq!(config.task_type, TaskType::CodeRetrieval);
    }

    #[test]
    fn builder_chains_all_fields() {
        let config = EmbeddingConfig::builder()
            .model_id("custom/model")
            .dimensions(512)
            .batch_size(16)
            .max_input_length(256)
            .normalize(false)
            .task_type(TaskType::Clustering)
            .build();
        assert_eq!(config.model_id, "custom/model");
        assert_eq!(config.dimensions, 512);
        assert_eq!(config.batch_size, 16);
        assert_eq!(config.max_input_length, 256);
        assert!(!config.normalize);
        assert_eq!(config.task_type, TaskType::Clustering);
    }
}
