//! Error types for the Erio agent runtime.

use std::time::Duration;
use thiserror::Error;

fn llm_error_display(message: &str, status: &Option<u16>) -> String {
    match status {
        Some(code) => format!("LLM error ({code}): {message}"),
        None => format!("LLM error: {message}"),
    }
}

fn execution_failed_display(message: &str, exit_code: &Option<i32>) -> String {
    match exit_code {
        Some(code) => format!("Tool execution failed: {message} (exit code: {code})"),
        None => format!("Tool execution failed: {message}"),
    }
}

/// Tool-specific errors.
#[derive(Debug, Error)]
pub enum ToolError {
    #[error("Tool timeout after {}s", .0.as_secs())]
    Timeout(Duration),

    #[error("Invalid parameters: {0}")]
    InvalidParams(String),

    #[error("Tool not found: {0}")]
    NotFound(String),

    #[error("{}", execution_failed_display(.message, .exit_code))]
    ExecutionFailed {
        message: String,
        exit_code: Option<i32>,
    },

    #[error("Tool execution cancelled")]
    Cancelled,
}

impl ToolError {
    /// Returns `true` if the error is potentially transient and the operation could be retried.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::Timeout(_))
    }
}

/// Core errors for the Erio runtime.
#[derive(Debug, Error)]
pub enum CoreError {
    #[error("{}", llm_error_display(.message, .status))]
    Llm { message: String, status: Option<u16> },

    #[error("Tool error: {0}")]
    Tool(#[from] ToolError),

    #[error("Embedding error: {message}")]
    Embedding { message: String },

    #[error("Context store error: {message}")]
    ContextStore { message: String },
}

impl CoreError {
    /// Returns `true` if the error is potentially transient and the operation could be retried.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Llm { status, .. } => {
                // 429 (rate limit), 5xx (server errors), or network errors (no status)
                matches!(status, None | Some(429 | 500..))
            }
            Self::Tool(err) => err.is_retryable(),
            Self::Embedding { .. } | Self::ContextStore { .. } => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // === CoreError Display Tests ===

    #[test]
    fn llm_error_displays_message_with_status() {
        let err = CoreError::Llm {
            message: "rate limited".into(),
            status: Some(429),
        };
        assert_eq!(err.to_string(), "LLM error (429): rate limited");
    }

    #[test]
    fn llm_error_displays_message_without_status() {
        let err = CoreError::Llm {
            message: "connection failed".into(),
            status: None,
        };
        assert_eq!(err.to_string(), "LLM error: connection failed");
    }

    // === ToolError Display Tests ===

    #[test]
    fn tool_error_timeout_displays_duration() {
        let err = ToolError::Timeout(Duration::from_secs(30));
        assert_eq!(err.to_string(), "Tool timeout after 30s");
    }

    #[test]
    fn tool_error_invalid_params_displays_message() {
        let err = ToolError::InvalidParams("missing 'command' field".into());
        assert_eq!(err.to_string(), "Invalid parameters: missing 'command' field");
    }

    #[test]
    fn tool_error_not_found_displays_name() {
        let err = ToolError::NotFound("calculator".into());
        assert_eq!(err.to_string(), "Tool not found: calculator");
    }

    #[test]
    fn tool_error_execution_failed_displays_details() {
        let err = ToolError::ExecutionFailed {
            message: "command failed".into(),
            exit_code: Some(1),
        };
        assert!(err.to_string().contains("command failed"));
        assert!(err.to_string().contains("exit code: 1"));
    }

    // === Retryable Tests ===

    #[test]
    fn tool_error_timeout_is_retryable() {
        let err = ToolError::Timeout(Duration::from_secs(30));
        assert!(err.is_retryable());
    }

    #[test]
    fn tool_error_invalid_params_is_not_retryable() {
        let err = ToolError::InvalidParams("bad json".into());
        assert!(!err.is_retryable());
    }

    #[test]
    fn tool_error_not_found_is_not_retryable() {
        let err = ToolError::NotFound("unknown".into());
        assert!(!err.is_retryable());
    }

    #[test]
    fn tool_error_cancelled_is_not_retryable() {
        let err = ToolError::Cancelled;
        assert!(!err.is_retryable());
    }

    #[test]
    fn core_error_llm_429_is_retryable() {
        let err = CoreError::Llm {
            message: "rate limited".into(),
            status: Some(429),
        };
        assert!(err.is_retryable());
    }

    #[test]
    fn core_error_llm_500_is_retryable() {
        let err = CoreError::Llm {
            message: "server error".into(),
            status: Some(500),
        };
        assert!(err.is_retryable());
    }

    #[test]
    fn core_error_llm_400_is_not_retryable() {
        let err = CoreError::Llm {
            message: "bad request".into(),
            status: Some(400),
        };
        assert!(!err.is_retryable());
    }

    #[test]
    fn core_error_llm_no_status_is_retryable() {
        // Network errors without HTTP status should be retryable
        let err = CoreError::Llm {
            message: "connection reset".into(),
            status: None,
        };
        assert!(err.is_retryable());
    }

    // === From Conversions ===

    #[test]
    fn core_error_from_tool_error() {
        let tool_err = ToolError::NotFound("test".into());
        let core_err: CoreError = tool_err.into();
        assert!(matches!(core_err, CoreError::Tool(_)));
    }

    // === Embedding Error Tests ===

    #[test]
    fn core_error_embedding_displays_message() {
        let err = CoreError::Embedding {
            message: "model load failed".into(),
        };
        assert_eq!(err.to_string(), "Embedding error: model load failed");
    }

    #[test]
    fn core_error_embedding_is_not_retryable() {
        let err = CoreError::Embedding {
            message: "dimension mismatch".into(),
        };
        assert!(!err.is_retryable());
    }

    // === Context Store Error Tests ===

    #[test]
    fn core_error_context_store_displays_message() {
        let err = CoreError::ContextStore {
            message: "table not found".into(),
        };
        assert_eq!(err.to_string(), "Context store error: table not found");
    }

    #[test]
    fn core_error_context_store_is_not_retryable() {
        let err = CoreError::ContextStore {
            message: "storage failure".into(),
        };
        assert!(!err.is_retryable());
    }
}
