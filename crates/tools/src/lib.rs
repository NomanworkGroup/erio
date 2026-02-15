//! Erio Tools - Tool trait, registry, and execution for the agent runtime.

mod registry;
mod result;
mod schema;
mod tool;

pub use erio_core::ToolError;
pub use registry::ToolRegistry;
pub use result::ToolResult;
pub use schema::{PropertyType, ToolSchema, ToolSchemaBuilder};
pub use tool::Tool;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // === ToolResult Tests ===

    #[test]
    fn tool_result_success_creates_success_variant() {
        let result = ToolResult::success("Hello, world!");
        assert!(result.is_success());
        assert_eq!(result.content(), Some("Hello, world!"));
    }

    #[test]
    fn tool_result_error_creates_error_variant() {
        let result = ToolResult::error("Something went wrong");
        assert!(!result.is_success());
        assert_eq!(result.error_message(), Some("Something went wrong"));
    }

    #[test]
    fn tool_result_serializes_success() {
        let result = ToolResult::success("output");
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["success"], true);
        assert_eq!(json["content"], "output");
    }

    #[test]
    fn tool_result_serializes_error() {
        let result = ToolResult::error("failed");
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["success"], false);
        assert_eq!(json["error"], "failed");
    }

    // === ToolSchema Tests ===

    #[test]
    fn tool_schema_builder_creates_schema() {
        let schema = ToolSchema::builder()
            .property("command", PropertyType::String, "The command to run", true)
            .property(
                "timeout",
                PropertyType::Integer,
                "Timeout in seconds",
                false,
            )
            .build();

        assert!(schema.has_property("command"));
        assert!(schema.has_property("timeout"));
        assert!(schema.is_required("command"));
        assert!(!schema.is_required("timeout"));
    }

    #[test]
    fn tool_schema_serializes_to_json_schema() {
        let schema = ToolSchema::builder()
            .property("name", PropertyType::String, "The name", true)
            .build();

        let json = schema.to_json_schema();
        assert_eq!(json["type"], "object");
        assert!(json["properties"]["name"].is_object());
        assert_eq!(json["properties"]["name"]["type"], "string");
        assert_eq!(json["required"], json!(["name"]));
    }

    #[test]
    fn tool_schema_empty_has_no_properties() {
        let schema = ToolSchema::builder().build();
        assert_eq!(schema.property_count(), 0);
    }

    // === Tool Trait Tests ===

    struct EchoTool;

    #[async_trait::async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }

        fn description(&self) -> &str {
            "Echoes the input message"
        }

        fn schema(&self) -> ToolSchema {
            ToolSchema::builder()
                .property("message", PropertyType::String, "Message to echo", true)
                .build()
        }

        async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, ToolError> {
            let message = params["message"]
                .as_str()
                .ok_or_else(|| ToolError::InvalidParams("missing 'message' field".into()))?;
            Ok(ToolResult::success(message))
        }
    }

    #[test]
    fn tool_returns_name() {
        let tool = EchoTool;
        assert_eq!(tool.name(), "echo");
    }

    #[test]
    fn tool_returns_description() {
        let tool = EchoTool;
        assert_eq!(tool.description(), "Echoes the input message");
    }

    #[test]
    fn tool_returns_schema() {
        let tool = EchoTool;
        let schema = tool.schema();
        assert!(schema.has_property("message"));
    }

    #[tokio::test]
    async fn tool_executes_successfully() {
        let tool = EchoTool;
        let params = json!({"message": "Hello"});
        let result = tool.execute(params).await.unwrap();
        assert!(result.is_success());
        assert_eq!(result.content(), Some("Hello"));
    }

    #[tokio::test]
    async fn tool_returns_error_for_invalid_params() {
        let tool = EchoTool;
        let params = json!({});
        let result = tool.execute(params).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidParams(_)));
    }

    // === ToolRegistry Tests ===

    #[test]
    fn registry_starts_empty() {
        let registry = ToolRegistry::new();
        assert_eq!(registry.len(), 0);
        assert!(registry.is_empty());
    }

    #[test]
    fn registry_registers_tool() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
    }

    #[test]
    fn registry_gets_tool_by_name() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        let tool = registry.get("echo");
        assert!(tool.is_some());
        assert_eq!(tool.unwrap().name(), "echo");
    }

    #[test]
    fn registry_returns_none_for_unknown_tool() {
        let registry = ToolRegistry::new();
        assert!(registry.get("unknown").is_none());
    }

    #[test]
    fn registry_lists_all_tools() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        let tools = registry.list();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0], "echo");
    }

    #[test]
    fn registry_contains_checks_existence() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        assert!(registry.contains("echo"));
        assert!(!registry.contains("unknown"));
    }

    #[tokio::test]
    async fn registry_executes_tool_by_name() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        let result = registry.execute("echo", json!({"message": "test"})).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().content(), Some("test"));
    }

    #[tokio::test]
    async fn registry_returns_not_found_for_unknown_tool() {
        let registry = ToolRegistry::new();
        let result = registry.execute("unknown", json!({})).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::NotFound(_)));
    }
}
