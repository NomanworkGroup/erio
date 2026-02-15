//! Tool execution result types.

use serde::{Deserialize, Serialize};

/// Result of a tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResult {
    /// Successful execution with content.
    Success { success: bool, content: String },
    /// Failed execution with error message.
    Error { success: bool, error: String },
}

impl ToolResult {
    /// Creates a successful result with the given content.
    pub fn success(content: impl Into<String>) -> Self {
        Self::Success {
            success: true,
            content: content.into(),
        }
    }

    /// Creates an error result with the given message.
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error {
            success: false,
            error: message.into(),
        }
    }

    /// Returns `true` if this is a successful result.
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success { .. })
    }

    /// Returns the content if this is a successful result.
    pub fn content(&self) -> Option<&str> {
        match self {
            Self::Success { content, .. } => Some(content),
            Self::Error { .. } => None,
        }
    }

    /// Returns the error message if this is an error result.
    pub fn error_message(&self) -> Option<&str> {
        match self {
            Self::Success { .. } => None,
            Self::Error { error, .. } => Some(error),
        }
    }
}
