//! Local embedding engine using the GGUF-quantized `EmbeddingGemma` model.
//!
//! This engine never downloads model files at runtime.
//!
//! `GemmaEmbedding::new` loads from a local model directory specified by `ERIO_MODEL_DIR`.
//! The embedding crate's `build.rs` populates this automatically at build time by downloading
//! public GitHub Release assets, or you can set `ERIO_MODEL_DIR` manually for offline builds.

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

fn model_files_from_env() -> Result<ModelFiles, EmbeddingError> {
    let base_dir = std::env::var("ERIO_MODEL_DIR").unwrap_or_else(|_| env!("ERIO_MODEL_DIR").to_owned());

    let base = PathBuf::from(base_dir);
    let files = ModelFiles {
        gguf_path: base.join("embeddinggemma-300M-Q8_0.gguf"),
        dense1_path: base.join("2_Dense/model.safetensors"),
        dense2_path: base.join("3_Dense/model.safetensors"),
        tokenizer_path: base.join("tokenizer.json"),
    };

    for path in [
        &files.gguf_path,
        &files.dense1_path,
        &files.dense2_path,
        &files.tokenizer_path,
    ] {
        if !path.exists() {
            return Err(EmbeddingError::ModelLoad(format!(
                "required model file missing: {}",
                path.display()
            )));
        }
    }

    Ok(files)
}

/// Local embedding engine using quantized `EmbeddingGemma` model via candle.
pub struct GemmaEmbedding {
    config: EmbeddingConfig,
    model: Arc<EmbeddingGemmaModel>,
}

impl GemmaEmbedding {
    /// Creates a new `GemmaEmbedding` from build-time prepared model files.
    pub fn new(config: EmbeddingConfig) -> Result<Self, EmbeddingError> {
        let model_files = model_files_from_env()?;
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
    use std::fs;
    use std::sync::{Mutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};

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

    #[test]
    fn gemma_new_is_sync_constructor_signature() {
        let constructor: fn(EmbeddingConfig) -> Result<GemmaEmbedding, EmbeddingError> =
            GemmaEmbedding::new;
        let _ = constructor;
    }

    #[test]
    #[allow(unsafe_code)]
    fn model_files_from_env_errors_when_required_files_are_missing() {
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock poisoned");

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let temp_dir = std::env::temp_dir().join(format!(
            "erio-embedding-missing-model-files-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&temp_dir).expect("failed to create temp model dir");
        fs::write(temp_dir.join("tokenizer.json"), b"{}")
            .expect("failed to create partial model file");

        let previous_model_dir = std::env::var("ERIO_MODEL_DIR").ok();
        unsafe { std::env::set_var("ERIO_MODEL_DIR", &temp_dir) };

        let result = model_files_from_env();

        match previous_model_dir {
            Some(value) => unsafe { std::env::set_var("ERIO_MODEL_DIR", value) },
            None => unsafe { std::env::remove_var("ERIO_MODEL_DIR") },
        }

        fs::remove_dir_all(&temp_dir).expect("failed to cleanup temp model dir");

        match result {
            Err(EmbeddingError::ModelLoad(message)) => {
                assert!(
                    message.contains("required model file missing"),
                    "unexpected error message: {message}"
                );
            }
            other => panic!("expected EmbeddingError::ModelLoad, got {other:?}"),
        }
    }

    #[tokio::test]
    #[ignore = "requires model download"]
    async fn gemma_returns_name() {
        let engine = GemmaEmbedding::new(EmbeddingConfig::default()).unwrap();
        assert_eq!(engine.name(), "gemma");
    }

    #[tokio::test]
    #[ignore = "requires model download"]
    async fn gemma_returns_correct_dimensions() {
        let engine = GemmaEmbedding::new(EmbeddingConfig::default()).unwrap();
        assert_eq!(engine.dimensions(), 768);
    }

    #[tokio::test]
    #[ignore = "requires model download"]
    async fn gemma_embed_returns_correct_dimensions() {
        let engine = GemmaEmbedding::new(EmbeddingConfig::default()).unwrap();
        let vec = engine.embed("hello world").await.unwrap();
        assert_eq!(vec.len(), 768);
    }

    #[tokio::test]
    #[ignore = "requires model download"]
    async fn gemma_embed_rejects_empty_input() {
        let engine = GemmaEmbedding::new(EmbeddingConfig::default()).unwrap();
        let result = engine.embed("").await;
        assert!(matches!(
            result.unwrap_err(),
            EmbeddingError::InvalidInput(_)
        ));
    }

    #[tokio::test]
    #[ignore = "requires model download"]
    async fn gemma_same_input_same_output() {
        let engine = GemmaEmbedding::new(EmbeddingConfig::default()).unwrap();
        let v1 = engine.embed("test determinism").await.unwrap();
        let v2 = engine.embed("test determinism").await.unwrap();
        assert_eq!(v1, v2);
    }

    #[tokio::test]
    #[ignore = "requires model download"]
    async fn gemma_different_inputs_different_outputs() {
        let engine = GemmaEmbedding::new(EmbeddingConfig::default()).unwrap();
        let v1 = engine.embed("hello").await.unwrap();
        let v2 = engine.embed("world").await.unwrap();
        assert_ne!(v1, v2);
    }

    #[tokio::test]
    #[ignore = "requires model download"]
    async fn gemma_normalized_unit_length() {
        let engine = GemmaEmbedding::new(EmbeddingConfig::default()).unwrap();
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
        let engine = GemmaEmbedding::new(EmbeddingConfig::default()).unwrap();
        let results = engine.embed_batch(&["a", "b", "c"]).await.unwrap();
        assert_eq!(results.len(), 3);
        for vec in &results {
            assert_eq!(vec.len(), 768);
        }
    }

    #[tokio::test]
    #[ignore = "requires model download"]
    async fn gemma_embed_batch_preserves_order() {
        let engine = GemmaEmbedding::new(EmbeddingConfig::default()).unwrap();
        let v_a = engine.embed("alpha").await.unwrap();
        let v_b = engine.embed("beta").await.unwrap();
        let batch = engine.embed_batch(&["alpha", "beta"]).await.unwrap();
        assert_eq!(batch[0], v_a);
        assert_eq!(batch[1], v_b);
    }
}
