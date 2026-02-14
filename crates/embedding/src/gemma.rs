//! Local embedding engine using GGUF-quantized `EmbeddingGemma` model.

use std::path::PathBuf;
use std::sync::Arc;

use crate::config::EmbeddingConfig;
use crate::engine::EmbeddingEngine;
use crate::error::EmbeddingError;
use crate::model::EmbeddingGemmaModel;
use crate::task::format_query;

/// Paths to all required model files.
#[derive(Debug, Clone)]
pub struct ModelFiles {
    /// Path to the GGUF backbone model file.
    pub gguf_path: PathBuf,
    /// Path to the first dense layer safetensors (`2_Dense/model.safetensors`).
    pub dense1_path: PathBuf,
    /// Path to the second dense layer safetensors (`3_Dense/model.safetensors`).
    pub dense2_path: PathBuf,
    /// Path to the tokenizer file (tokenizer.json).
    pub tokenizer_path: PathBuf,
}

/// Downloads required model files from `HuggingFace` Hub.
///
/// Downloads the GGUF backbone from `ggml-org/embeddinggemma-300M-GGUF` and
/// the dense layers + tokenizer from `google/embeddinggemma-300m`.
/// Requires `HF_TOKEN` environment variable for the gated Google model.
pub async fn download_model_files(gguf_filename: &str) -> Result<ModelFiles, EmbeddingError> {
    let token = std::env::var("HF_TOKEN").ok();
    let api = hf_hub::api::tokio::ApiBuilder::from_env()
        .with_token(token)
        .build()
        .map_err(|e| EmbeddingError::ModelLoad(format!("failed to create HF API: {e}")))?;

    // Download GGUF backbone (not gated)
    let gguf_repo = api.model("ggml-org/embeddinggemma-300M-GGUF".to_owned());
    let gguf_path = gguf_repo
        .get(gguf_filename)
        .await
        .map_err(|e| EmbeddingError::ModelLoad(format!("failed to download GGUF model: {e}")))?;

    // Download tokenizer + dense layers from gated Google repo (requires HF_TOKEN)
    let google_repo = api.model("google/embeddinggemma-300m".to_owned());
    let tokenizer_path = google_repo
        .get("tokenizer.json")
        .await
        .map_err(|e| EmbeddingError::ModelLoad(format!("failed to download tokenizer: {e}")))?;
    let dense1_path = google_repo
        .get("2_Dense/model.safetensors")
        .await
        .map_err(|e| EmbeddingError::ModelLoad(format!("failed to download 2_Dense: {e}")))?;
    let dense2_path = google_repo
        .get("3_Dense/model.safetensors")
        .await
        .map_err(|e| EmbeddingError::ModelLoad(format!("failed to download 3_Dense: {e}")))?;

    Ok(ModelFiles {
        gguf_path,
        dense1_path,
        dense2_path,
        tokenizer_path,
    })
}

/// Local embedding engine using quantized `EmbeddingGemma` model via candle.
pub struct GemmaEmbedding {
    config: EmbeddingConfig,
    model: Arc<EmbeddingGemmaModel>,
}

impl GemmaEmbedding {
    /// Creates a new `GemmaEmbedding` by downloading and loading the model.
    pub async fn new(config: EmbeddingConfig) -> Result<Self, EmbeddingError> {
        let model_files = download_model_files("embeddinggemma-300M-Q8_0.gguf").await?;
        Self::from_files(config, &model_files)
    }

    /// Creates a `GemmaEmbedding` from pre-downloaded model files.
    pub fn from_files(
        config: EmbeddingConfig,
        model_files: &ModelFiles,
    ) -> Result<Self, EmbeddingError> {
        let model = EmbeddingGemmaModel::load(
            &model_files.gguf_path,
            &model_files.dense1_path,
            &model_files.dense2_path,
            &model_files.tokenizer_path,
        )?;
        Ok(Self {
            config,
            model: Arc::new(model),
        })
    }
}

#[async_trait::async_trait]
impl EmbeddingEngine for GemmaEmbedding {
    fn name(&self) -> &'static str {
        "gemma"
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
        let prompt = format_query(text, self.config.task_type);
        self.model.embed(&prompt)
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.iter().any(|t| t.is_empty()) {
            return Err(EmbeddingError::InvalidInput(
                "text must not be empty".into(),
            ));
        }
        texts
            .iter()
            .map(|text| {
                let prompt = format_query(text, self.config.task_type);
                self.model.embed(&prompt)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // All tests that require model download are #[ignore].
    // Run with: cargo test -p erio-embedding -- --ignored

    #[test]
    fn model_files_struct_holds_paths() {
        let files = ModelFiles {
            gguf_path: PathBuf::from("/tmp/model.gguf"),
            dense1_path: PathBuf::from("/tmp/2_Dense/model.safetensors"),
            dense2_path: PathBuf::from("/tmp/3_Dense/model.safetensors"),
            tokenizer_path: PathBuf::from("/tmp/tokenizer.json"),
        };
        assert!(files.gguf_path.ends_with("model.gguf"));
        assert!(files.dense1_path.ends_with("model.safetensors"));
        assert!(files.dense2_path.ends_with("model.safetensors"));
        assert!(files.tokenizer_path.ends_with("tokenizer.json"));
    }

    #[tokio::test]
    #[ignore = "requires model download"]
    async fn downloads_all_required_files() {
        let files = download_model_files("embeddinggemma-300M-Q8_0.gguf")
            .await
            .unwrap();
        assert!(files.gguf_path.exists());
        assert!(files.dense1_path.exists());
        assert!(files.dense2_path.exists());
        assert!(files.tokenizer_path.exists());
    }

    #[tokio::test]
    #[ignore = "requires model download"]
    async fn gemma_returns_name() {
        let engine = GemmaEmbedding::new(EmbeddingConfig::default())
            .await
            .unwrap();
        assert_eq!(engine.name(), "gemma");
    }

    #[tokio::test]
    #[ignore = "requires model download"]
    async fn gemma_returns_correct_dimensions() {
        let engine = GemmaEmbedding::new(EmbeddingConfig::default())
            .await
            .unwrap();
        assert_eq!(engine.dimensions(), 768);
    }

    #[tokio::test]
    #[ignore = "requires model download"]
    async fn gemma_embed_returns_correct_dimensions() {
        let engine = GemmaEmbedding::new(EmbeddingConfig::default())
            .await
            .unwrap();
        let vec = engine.embed("hello world").await.unwrap();
        assert_eq!(vec.len(), 768);
    }

    #[tokio::test]
    #[ignore = "requires model download"]
    async fn gemma_embed_rejects_empty_input() {
        let engine = GemmaEmbedding::new(EmbeddingConfig::default())
            .await
            .unwrap();
        let result = engine.embed("").await;
        assert!(matches!(
            result.unwrap_err(),
            EmbeddingError::InvalidInput(_)
        ));
    }

    #[tokio::test]
    #[ignore = "requires model download"]
    async fn gemma_same_input_same_output() {
        let engine = GemmaEmbedding::new(EmbeddingConfig::default())
            .await
            .unwrap();
        let v1 = engine.embed("test determinism").await.unwrap();
        let v2 = engine.embed("test determinism").await.unwrap();
        assert_eq!(v1, v2);
    }

    #[tokio::test]
    #[ignore = "requires model download"]
    async fn gemma_different_inputs_different_outputs() {
        let engine = GemmaEmbedding::new(EmbeddingConfig::default())
            .await
            .unwrap();
        let v1 = engine.embed("hello").await.unwrap();
        let v2 = engine.embed("world").await.unwrap();
        assert_ne!(v1, v2);
    }

    #[tokio::test]
    #[ignore = "requires model download"]
    async fn gemma_normalized_unit_length() {
        let engine = GemmaEmbedding::new(EmbeddingConfig::default())
            .await
            .unwrap();
        let vec = engine.embed("test normalization").await.unwrap();
        let magnitude: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (magnitude - 1.0).abs() < 1e-4,
            "Expected unit length, got {magnitude}"
        );
    }

    #[tokio::test]
    #[ignore = "requires model download"]
    async fn gemma_embed_batch_correct_count() {
        let engine = GemmaEmbedding::new(EmbeddingConfig::default())
            .await
            .unwrap();
        let results = engine.embed_batch(&["a", "b", "c"]).await.unwrap();
        assert_eq!(results.len(), 3);
        for vec in &results {
            assert_eq!(vec.len(), 768);
        }
    }

    #[tokio::test]
    #[ignore = "requires model download"]
    async fn gemma_embed_batch_preserves_order() {
        let engine = GemmaEmbedding::new(EmbeddingConfig::default())
            .await
            .unwrap();
        let v_a = engine.embed("alpha").await.unwrap();
        let v_b = engine.embed("beta").await.unwrap();
        let batch = engine.embed_batch(&["alpha", "beta"]).await.unwrap();
        assert_eq!(batch[0], v_a);
        assert_eq!(batch[1], v_b);
    }
}
