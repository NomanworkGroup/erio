//! Tool trait definition.

use crate::result::ToolResult;
use crate::schema::ToolSchema;
use erio_core::ToolError;
use serde_json::Value;

/// A tool that can be executed by an agent.
#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    /// Returns the unique name of this tool.
    fn name(&self) -> &str;

    /// Returns a human-readable description of what this tool does.
    fn description(&self) -> &str;

    /// Returns the JSON Schema describing the tool's parameters.
    fn schema(&self) -> ToolSchema;

    /// Executes the tool with the given parameters.
    async fn execute(&self, params: Value) -> Result<ToolResult, ToolError>;
}
