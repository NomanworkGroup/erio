//! LLM-specific error types.

use std::time::Duration;
use thiserror::Error;

/// Errors specific to LLM provider operations.
#[derive(Debug, Error)]
pub enum LlmError {
    #[error("Rate limited")]
    RateLimited {
        /// Suggested retry delay from the server.
        retry_after: Option<Duration>,
    },

    #[error("API error ({status}): {message}")]
    Api {
        status: u16,
        message: String,
    },

    #[error("Invalid response: {0}")]
    InvalidResponse(String),

    #[error("Request timeout")]
    Timeout,

    #[error("Authentication failed")]
    Auth,

    #[error("Network error: {0}")]
    Network(String),
}

impl LlmError {
    /// Returns `true` if the error is potentially transient and the operation could be retried.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::RateLimited { .. } | Self::Timeout | Self::Network(_) => true,
            Self::Api { status, .. } => *status >= 500,
            Self::Auth | Self::InvalidResponse(_) => false,
        }
    }
}

impl From<LlmError> for erio_core::CoreError {
    fn from(err: LlmError) -> Self {
        let (message, status) = match &err {
            LlmError::RateLimited { .. } => (err.to_string(), Some(429)),
            LlmError::Api { status, message } => (message.clone(), Some(*status)),
            LlmError::Auth => (err.to_string(), Some(401)),
            LlmError::Timeout | LlmError::Network(_) | LlmError::InvalidResponse(_) => {
                (err.to_string(), None)
            }
        };
        Self::Llm { message, status }
    }
}
