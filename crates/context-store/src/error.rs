//! Error types for the context store.

use thiserror::Error;

/// Errors specific to context store operations.
#[derive(Debug, Error)]
pub enum ContextStoreError {
    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Document not found: {0}")]
    NotFound(String),
}

impl ContextStoreError {
    /// Returns `true` if the error is potentially transient and the operation could be retried.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::Storage(_))
    }
}

impl From<ContextStoreError> for erio_core::CoreError {
    fn from(err: ContextStoreError) -> Self {
        Self::ContextStore {
            message: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Display Tests ===

    #[test]
    fn storage_error_displays_message() {
        let err = ContextStoreError::Storage("disk full".into());
        assert_eq!(err.to_string(), "Storage error: disk full");
    }

    #[test]
    fn embedding_error_displays_message() {
        let err = ContextStoreError::Embedding("inference failed".into());
        assert_eq!(err.to_string(), "Embedding error: inference failed");
    }

    #[test]
    fn invalid_input_displays_message() {
        let err = ContextStoreError::InvalidInput("empty text".into());
        assert_eq!(err.to_string(), "Invalid input: empty text");
    }

    #[test]
    fn not_found_displays_message() {
        let err = ContextStoreError::NotFound("doc_123".into());
        assert_eq!(err.to_string(), "Document not found: doc_123");
    }

    // === Retryable Tests ===

    #[test]
    fn storage_error_is_retryable() {
        let err = ContextStoreError::Storage("timeout".into());
        assert!(err.is_retryable());
    }

    #[test]
    fn embedding_error_is_not_retryable() {
        let err = ContextStoreError::Embedding("model failed".into());
        assert!(!err.is_retryable());
    }

    #[test]
    fn invalid_input_is_not_retryable() {
        let err = ContextStoreError::InvalidInput("empty".into());
        assert!(!err.is_retryable());
    }

    #[test]
    fn not_found_is_not_retryable() {
        let err = ContextStoreError::NotFound("id".into());
        assert!(!err.is_retryable());
    }

    // === CoreError Conversion ===

    #[test]
    fn converts_to_core_error() {
        let err = ContextStoreError::Storage("test".into());
        let core_err: erio_core::CoreError = err.into();
        assert!(matches!(
            core_err,
            erio_core::CoreError::ContextStore { .. }
        ));
    }

    #[test]
    fn conversion_preserves_message() {
        let err = ContextStoreError::InvalidInput("bad input".into());
        let core_err: erio_core::CoreError = err.into();
        match core_err {
            erio_core::CoreError::ContextStore { message } => {
                assert_eq!(message, "Invalid input: bad input");
            }
            _ => panic!("Expected CoreError::ContextStore"),
        }
    }
}
