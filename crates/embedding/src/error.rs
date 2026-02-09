//! Embedding-specific error types.

use thiserror::Error;

/// Errors specific to embedding engine operations.
#[derive(Debug, Error)]
pub enum EmbeddingError {
    #[error("Model load failed: {0}")]
    ModelLoad(String),

    #[error("Tokenization error: {0}")]
    Tokenization(String),

    #[error("Inference failed: {0}")]
    Inference(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
}

impl EmbeddingError {
    /// Returns `true` if the error is potentially transient and the operation could be retried.
    ///
    /// Embedding errors are generally permanent (model/input problems), so this always returns `false`.
    pub fn is_retryable(&self) -> bool {
        false
    }
}

impl From<EmbeddingError> for erio_core::CoreError {
    fn from(err: EmbeddingError) -> Self {
        Self::Embedding {
            message: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Display Tests ===

    #[test]
    fn model_load_displays_message() {
        let err = EmbeddingError::ModelLoad("weights not found".into());
        assert_eq!(err.to_string(), "Model load failed: weights not found");
    }

    #[test]
    fn tokenization_displays_message() {
        let err = EmbeddingError::Tokenization("unknown token".into());
        assert_eq!(err.to_string(), "Tokenization error: unknown token");
    }

    #[test]
    fn inference_displays_message() {
        let err = EmbeddingError::Inference("out of memory".into());
        assert_eq!(err.to_string(), "Inference failed: out of memory");
    }

    #[test]
    fn invalid_input_displays_message() {
        let err = EmbeddingError::InvalidInput("empty text".into());
        assert_eq!(err.to_string(), "Invalid input: empty text");
    }

    #[test]
    fn dimension_mismatch_displays_details() {
        let err = EmbeddingError::DimensionMismatch {
            expected: 384,
            got: 768,
        };
        assert_eq!(
            err.to_string(),
            "Dimension mismatch: expected 384, got 768"
        );
    }

    // === Retryable Tests ===

    #[test]
    fn model_load_is_not_retryable() {
        let err = EmbeddingError::ModelLoad("missing file".into());
        assert!(!err.is_retryable());
    }

    #[test]
    fn tokenization_is_not_retryable() {
        let err = EmbeddingError::Tokenization("bad token".into());
        assert!(!err.is_retryable());
    }

    #[test]
    fn inference_is_not_retryable() {
        let err = EmbeddingError::Inference("oom".into());
        assert!(!err.is_retryable());
    }

    #[test]
    fn invalid_input_is_not_retryable() {
        let err = EmbeddingError::InvalidInput("empty".into());
        assert!(!err.is_retryable());
    }

    #[test]
    fn dimension_mismatch_is_not_retryable() {
        let err = EmbeddingError::DimensionMismatch {
            expected: 384,
            got: 768,
        };
        assert!(!err.is_retryable());
    }

    // === CoreError Conversion ===

    #[test]
    fn converts_to_core_error() {
        let emb_err = EmbeddingError::ModelLoad("test".into());
        let core_err: erio_core::CoreError = emb_err.into();
        assert!(matches!(core_err, erio_core::CoreError::Embedding { .. }));
    }

    #[test]
    fn conversion_preserves_message() {
        let emb_err = EmbeddingError::InvalidInput("empty text".into());
        let core_err: erio_core::CoreError = emb_err.into();
        match core_err {
            erio_core::CoreError::Embedding { message } => {
                assert_eq!(message, "Invalid input: empty text");
            }
            _ => panic!("Expected CoreError::Embedding"),
        }
    }
}
