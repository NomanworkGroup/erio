//! Integration tests: Workflow → LLM → Tools
//!
//! Tests the full flow of a workflow that calls an LLM provider
//! and executes tools based on LLM responses.

use std::sync::Arc;

use erio_llm_client::{CompletionRequest, CompletionResponse, LlmError, LlmProvider, Usage};
use erio_tools::{Tool, ToolRegistry, ToolResult, ToolSchema};
use erio_workflow::{Step, StepOutput, Workflow, WorkflowContext, WorkflowEngine, WorkflowError};
use serde_json::json;
use tokio::sync::Mutex;

// ============================================================
// Mock LLM Provider
// ============================================================

/// A mock LLM provider that returns pre-configured responses in order.
struct MockLlmProvider {
    responses: Mutex<Vec<CompletionResponse>>,
}

impl MockLlmProvider {
    fn new(responses: Vec<CompletionResponse>) -> Self {
        Self {
            responses: Mutex::new(responses),
        }
    }

    fn single(content: &str) -> Self {
        Self::new(vec![CompletionResponse {
            content: Some(content.into()),
            tool_calls: vec![],
            model: "mock".into(),
            usage: Some(Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            }),
        }])
    }
}

#[async_trait::async_trait]
impl LlmProvider for MockLlmProvider {
    fn name(&self) -> &str {
        "mock"
    }

    async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let mut responses = self.responses.lock().await;
        if responses.is_empty() {
            return Err(LlmError::InvalidResponse("no more mock responses".into()));
        }
        Ok(responses.remove(0))
    }
}

// ============================================================
// Step Adapters
// ============================================================

/// A workflow step that calls an LLM provider.
struct LlmStep {
    step_id: String,
    provider: Arc<dyn LlmProvider>,
    prompt: String,
}

impl LlmStep {
    fn new(id: &str, provider: Arc<dyn LlmProvider>, prompt: &str) -> Self {
        Self {
            step_id: id.into(),
            provider,
            prompt: prompt.into(),
        }
    }
}

#[async_trait::async_trait]
impl Step for LlmStep {
    fn id(&self) -> &str {
        &self.step_id
    }

    async fn execute(&self, _ctx: &mut WorkflowContext) -> Result<StepOutput, WorkflowError> {
        let request =
            CompletionRequest::new("mock-model").message(erio_core::Message::user(&self.prompt));

        let response =
            self.provider
                .complete(request)
                .await
                .map_err(|e| WorkflowError::StepFailed {
                    step_id: self.step_id.clone(),
                    message: e.to_string(),
                })?;

        let content = response.content.unwrap_or_default();

        // Store tool calls in metadata if present
        let output = if response.tool_calls.is_empty() {
            StepOutput::new(&content)
        } else {
            let calls_json = serde_json::to_value(&response.tool_calls).unwrap_or_default();
            StepOutput::new(&content).with_metadata(calls_json)
        };

        Ok(output)
    }
}

/// A workflow step that executes tools from a registry based on previous step output.
struct ToolStep {
    step_id: String,
    registry: Arc<ToolRegistry>,
    tool_name: String,
}

impl ToolStep {
    fn new(id: &str, registry: Arc<ToolRegistry>, tool_name: &str) -> Self {
        Self {
            step_id: id.into(),
            registry,
            tool_name: tool_name.into(),
        }
    }
}

#[async_trait::async_trait]
impl Step for ToolStep {
    fn id(&self) -> &str {
        &self.step_id
    }

    async fn execute(&self, ctx: &mut WorkflowContext) -> Result<StepOutput, WorkflowError> {
        // Read params from the previous step's output value as JSON
        let params: serde_json::Value = ctx
            .step_outputs()
            .values()
            .last()
            .map(|o| serde_json::from_str(o.value()).unwrap_or(json!({})))
            .unwrap_or(json!({}));

        let result = self
            .registry
            .execute(&self.tool_name, params)
            .await
            .map_err(|e| WorkflowError::StepFailed {
                step_id: self.step_id.clone(),
                message: e.to_string(),
            })?;

        let content = result.content().unwrap_or("").to_string();
        Ok(StepOutput::new(&content))
    }
}

// ============================================================
// Mock Tool
// ============================================================

struct UppercaseTool;

#[async_trait::async_trait]
impl Tool for UppercaseTool {
    fn name(&self) -> &str {
        "uppercase"
    }

    fn description(&self) -> &str {
        "Converts input text to uppercase"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::builder()
            .property(
                "text",
                erio_tools::PropertyType::String,
                "Text to uppercase",
                true,
            )
            .build()
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, erio_core::ToolError> {
        let text = params["text"]
            .as_str()
            .ok_or_else(|| erio_core::ToolError::InvalidParams("missing 'text'".into()))?;
        Ok(ToolResult::success(text.to_uppercase()))
    }
}

// ============================================================
// Integration Tests
// ============================================================

#[tokio::test]
async fn llm_step_executes_in_workflow() {
    let llm = Arc::new(MockLlmProvider::single("Plan: do the thing"));

    let workflow = Workflow::builder()
        .step(LlmStep::new("plan", llm, "Plan the task"), &[])
        .build()
        .unwrap();

    let engine = WorkflowEngine::new();
    let result = engine.run(workflow).await.unwrap();

    assert!(result.is_completed("plan"));
    assert_eq!(result.output("plan").unwrap().value(), "Plan: do the thing");
}

#[tokio::test]
async fn tool_step_executes_in_workflow() {
    let mut registry = ToolRegistry::new();
    registry.register(UppercaseTool);
    let registry = Arc::new(registry);

    // A step that produces JSON params, then a tool step that consumes them
    struct ParamStep;

    #[async_trait::async_trait]
    impl Step for ParamStep {
        fn id(&self) -> &str {
            "params"
        }
        async fn execute(&self, _ctx: &mut WorkflowContext) -> Result<StepOutput, WorkflowError> {
            Ok(StepOutput::new(r#"{"text":"hello world"}"#))
        }
    }

    let workflow = Workflow::builder()
        .step(ParamStep, &[])
        .step(
            ToolStep::new("run_tool", registry, "uppercase"),
            &["params"],
        )
        .build()
        .unwrap();

    let engine = WorkflowEngine::new();
    let result = engine.run(workflow).await.unwrap();

    assert_eq!(result.output("run_tool").unwrap().value(), "HELLO WORLD");
}

#[tokio::test]
async fn full_pipeline_llm_tool_llm() {
    // DAG: plan(LLM) → execute_tool → synthesize(LLM)
    let llm = Arc::new(MockLlmProvider::new(vec![
        // First call: plan step returns JSON tool params
        CompletionResponse {
            content: Some(r#"{"text":"make this loud"}"#.into()),
            tool_calls: vec![],
            model: "mock".into(),
            usage: None,
        },
        // Second call: synthesize step
        CompletionResponse {
            content: Some("Final summary: MAKE THIS LOUD".into()),
            tool_calls: vec![],
            model: "mock".into(),
            usage: None,
        },
    ]));

    let mut registry = ToolRegistry::new();
    registry.register(UppercaseTool);
    let registry = Arc::new(registry);

    let workflow = Workflow::builder()
        .step(LlmStep::new("plan", llm.clone(), "Plan"), &[])
        .step(
            ToolStep::new("execute_tool", registry, "uppercase"),
            &["plan"],
        )
        .step(
            LlmStep::new("synthesize", llm, "Summarize results"),
            &["execute_tool"],
        )
        .build()
        .unwrap();

    let engine = WorkflowEngine::new();
    let result = engine.run(workflow).await.unwrap();

    assert!(result.is_completed("plan"));
    assert!(result.is_completed("execute_tool"));
    assert!(result.is_completed("synthesize"));
    assert_eq!(
        result.output("execute_tool").unwrap().value(),
        "MAKE THIS LOUD"
    );
    assert_eq!(
        result.output("synthesize").unwrap().value(),
        "Final summary: MAKE THIS LOUD"
    );
}

#[tokio::test]
async fn llm_error_propagates_to_dependent_steps() {
    // LLM provider that fails
    struct FailingLlm;

    #[async_trait::async_trait]
    impl LlmProvider for FailingLlm {
        fn name(&self) -> &str {
            "failing"
        }
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, LlmError> {
            Err(LlmError::Api {
                status: 500,
                message: "server down".into(),
            })
        }
    }

    let llm: Arc<dyn LlmProvider> = Arc::new(FailingLlm);

    let mut registry = ToolRegistry::new();
    registry.register(UppercaseTool);
    let registry = Arc::new(registry);

    let workflow = Workflow::builder()
        .step(LlmStep::new("plan", llm, "Plan"), &[])
        .step(
            ToolStep::new("execute_tool", registry, "uppercase"),
            &["plan"],
        )
        .build()
        .unwrap();

    let engine = WorkflowEngine::new();
    let result = engine.run(workflow).await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, WorkflowError::StepFailed { ref step_id, .. } if step_id == "plan"),
        "expected plan step to fail, got: {err}"
    );
}

#[tokio::test]
async fn step_context_passes_output_to_next() {
    // Step that reads a specific dependency's output
    struct ReadDepStep {
        id: String,
        dep_id: String,
    }

    #[async_trait::async_trait]
    impl Step for ReadDepStep {
        fn id(&self) -> &str {
            &self.id
        }
        async fn execute(&self, ctx: &mut WorkflowContext) -> Result<StepOutput, WorkflowError> {
            let prev = ctx
                .output(&self.dep_id)
                .map(|o| format!("received: {}", o.value()))
                .unwrap_or_else(|| "no input".into());
            Ok(StepOutput::new(&prev))
        }
    }

    let llm = Arc::new(MockLlmProvider::single("LLM output here"));

    let workflow = Workflow::builder()
        .step(LlmStep::new("first", llm, "Generate"), &[])
        .step(
            ReadDepStep {
                id: "second".into(),
                dep_id: "first".into(),
            },
            &["first"],
        )
        .build()
        .unwrap();

    let engine = WorkflowEngine::new();
    let result = engine.run(workflow).await.unwrap();

    assert_eq!(
        result.output("second").unwrap().value(),
        "received: LLM output here"
    );
}
