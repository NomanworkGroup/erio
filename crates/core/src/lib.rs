//! Erio Core - Core types, traits, and error handling for the agent runtime.

pub mod config;
pub mod error;
pub mod message;

pub use config::RetryConfig;
pub use error::{CoreError, ToolError};
pub use message::{Content, Message, Role, ToolCall};
