//! Plan-Execute agent — LLM generates a plan, workflow engine executes it.

use std::fmt;
use std::sync::Arc;

use erio_core::Message;
use erio_llm_client::{CompletionRequest, LlmProvider};
use erio_tools::ToolRegistry;
use erio_workflow::{Step, StepOutput, Workflow, WorkflowContext, WorkflowEngine, WorkflowError};

/// Errors from the plan-execute agent.
#[derive(Debug)]
pub enum PlanError {
    /// LLM provider returned an error.
    Llm(erio_llm_client::LlmError),
    /// The plan JSON was invalid or unparseable.
    InvalidPlan(String),
    /// Workflow execution failed.
    Execution(WorkflowError),
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Llm(e) => write!(f, "LLM error: {e}"),
            Self::InvalidPlan(msg) => write!(f, "invalid plan: {msg}"),
            Self::Execution(e) => write!(f, "execution error: {e}"),
        }
    }
}

impl std::error::Error for PlanError {}

/// A single step in a plan.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PlanStep {
    /// Unique step identifier.
    pub id: String,
    /// Tool to execute.
    pub tool: String,
    /// Parameters to pass to the tool.
    pub params: serde_json::Value,
    /// IDs of steps this depends on.
    pub deps: Vec<String>,
}

/// A parsed execution plan from the LLM.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Plan {
    steps: Vec<PlanStep>,
}

impl Plan {
    /// Parses a plan from a JSON value.
    pub fn from_json(value: &serde_json::Value) -> Result<Self, PlanError> {
        serde_json::from_value(value.clone())
            .map_err(|e| PlanError::InvalidPlan(e.to_string()))
    }

    /// Returns the steps in this plan.
    pub fn steps(&self) -> &[PlanStep] {
        &self.steps
    }
}

/// Output from a completed plan execution.
#[derive(Debug)]
pub struct PlanOutput {
    results: std::collections::HashMap<String, String>,
}

impl PlanOutput {
    /// Returns the result of a specific step.
    pub fn step_result(&self, step_id: &str) -> Option<&str> {
        self.results.get(step_id).map(String::as_str)
    }

    /// Returns the number of steps executed.
    pub fn step_count(&self) -> usize {
        self.results.len()
    }
}

/// A workflow step that executes a tool from the registry.
struct ToolExecStep {
    id: String,
    tool_name: String,
    params: serde_json::Value,
    registry: Arc<ToolRegistry>,
}

#[async_trait::async_trait]
impl Step for ToolExecStep {
    fn id(&self) -> &str {
        &self.id
    }

    async fn execute(&self, _ctx: &mut WorkflowContext) -> Result<StepOutput, WorkflowError> {
        let result = self
            .registry
            .execute(&self.tool_name, self.params.clone())
            .await
            .map_err(|e| WorkflowError::StepFailed {
                step_id: self.id.clone(),
                message: e.to_string(),
            })?;
        let content = result.content().unwrap_or("").to_string();
        Ok(StepOutput::new(&content))
    }
}

const DEFAULT_MODEL: &str = "gpt-4";

/// Plan-Execute agent: asks an LLM to produce a JSON plan, then runs it as a workflow DAG.
pub struct PlanExecuteAgent {
    llm: Arc<dyn LlmProvider>,
    tools: Arc<ToolRegistry>,
    model: String,
}

impl PlanExecuteAgent {
    /// Creates a new agent with default settings.
    pub fn new(llm: Arc<dyn LlmProvider>, tools: Arc<ToolRegistry>) -> Self {
        Self {
            llm,
            tools,
            model: DEFAULT_MODEL.into(),
        }
    }

    /// Runs the plan-execute loop: ask LLM for a plan, then execute it.
    pub async fn run(&self, prompt: &str) -> Result<PlanOutput, PlanError> {
        // 1. Ask LLM for a plan
        let request = CompletionRequest::new(&self.model)
            .message(Message::system(
                "You are a planner. Return a JSON object with a \"steps\" array. \
                 Each step has: id, tool, params, deps.",
            ))
            .message(Message::user(prompt));

        let response = self.llm.complete(request).await.map_err(PlanError::Llm)?;
        let raw = response.content.unwrap_or_default();

        // 2. Parse the plan
        let json: serde_json::Value =
            serde_json::from_str(&raw).map_err(|e| PlanError::InvalidPlan(e.to_string()))?;
        let plan = Plan::from_json(&json)?;

        // 3. Build a workflow DAG from the plan steps
        let mut builder = Workflow::builder();
        for step in plan.steps() {
            let deps: Vec<&str> = step.deps.iter().map(String::as_str).collect();
            builder = builder.step(
                ToolExecStep {
                    id: step.id.clone(),
                    tool_name: step.tool.clone(),
                    params: step.params.clone(),
                    registry: Arc::clone(&self.tools),
                },
                &deps,
            );
        }
        let workflow = builder.build().map_err(PlanError::Execution)?;

        // 4. Execute
        let engine = WorkflowEngine::new();
        let ctx = engine.run(workflow).await.map_err(PlanError::Execution)?;

        // 5. Collect results
        let mut results = std::collections::HashMap::new();
        for step in plan.steps() {
            if let Some(output) = ctx.output(&step.id) {
                results.insert(step.id.clone(), output.value().to_string());
            }
        }

        Ok(PlanOutput { results })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use erio_llm_client::{CompletionResponse, LlmError};
    use erio_tools::{PropertyType, Tool, ToolResult, ToolSchema};
    use tokio::sync::Mutex;

    // === Mock LLM ===

    struct MockLlm {
        responses: Mutex<Vec<CompletionResponse>>,
    }

    impl MockLlm {
        fn new(responses: Vec<CompletionResponse>) -> Self {
            Self {
                responses: Mutex::new(responses),
            }
        }

        fn text(content: &str) -> CompletionResponse {
            CompletionResponse {
                content: Some(content.into()),
                tool_calls: vec![],
                model: "mock".into(),
                usage: None,
            }
        }
    }

    #[async_trait::async_trait]
    impl LlmProvider for MockLlm {
        fn name(&self) -> &str {
            "mock"
        }
        async fn complete(
            &self,
            _req: CompletionRequest,
        ) -> Result<CompletionResponse, LlmError> {
            let mut responses = self.responses.lock().await;
            if responses.is_empty() {
                return Err(LlmError::InvalidResponse("no more responses".into()));
            }
            Ok(responses.remove(0))
        }
    }

    // === Mock Tools ===

    struct AddTool;

    #[async_trait::async_trait]
    impl Tool for AddTool {
        fn name(&self) -> &str {
            "add"
        }
        fn description(&self) -> &str {
            "Adds two numbers"
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema::builder()
                .property("a", PropertyType::Integer, "First number", true)
                .property("b", PropertyType::Integer, "Second number", true)
                .build()
        }
        async fn execute(
            &self,
            params: serde_json::Value,
        ) -> Result<ToolResult, erio_core::ToolError> {
            let a = params["a"].as_i64().unwrap_or(0);
            let b = params["b"].as_i64().unwrap_or(0);
            Ok(ToolResult::success(format!("{}", a + b)))
        }
    }

    struct MultiplyTool;

    #[async_trait::async_trait]
    impl Tool for MultiplyTool {
        fn name(&self) -> &str {
            "multiply"
        }
        fn description(&self) -> &str {
            "Multiplies two numbers"
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema::builder()
                .property("a", PropertyType::Integer, "First number", true)
                .property("b", PropertyType::Integer, "Second number", true)
                .build()
        }
        async fn execute(
            &self,
            params: serde_json::Value,
        ) -> Result<ToolResult, erio_core::ToolError> {
            let a = params["a"].as_i64().unwrap_or(0);
            let b = params["b"].as_i64().unwrap_or(0);
            Ok(ToolResult::success(format!("{}", a * b)))
        }
    }

    fn make_registry() -> ToolRegistry {
        let mut reg = ToolRegistry::new();
        reg.register(AddTool);
        reg.register(MultiplyTool);
        reg
    }

    // === Tests ===

    #[tokio::test]
    async fn parses_single_step_plan() {
        let plan_json = serde_json::json!({
            "steps": [
                {"id": "s1", "tool": "add", "params": {"a": 1, "b": 2}, "deps": []}
            ]
        });

        let plan = Plan::from_json(&plan_json).unwrap();
        assert_eq!(plan.steps().len(), 1);
        assert_eq!(plan.steps()[0].id, "s1");
    }

    #[tokio::test]
    async fn parses_multi_step_plan_with_deps() {
        let plan_json = serde_json::json!({
            "steps": [
                {"id": "s1", "tool": "add", "params": {"a": 1, "b": 2}, "deps": []},
                {"id": "s2", "tool": "multiply", "params": {"a": 3, "b": 4}, "deps": []},
                {"id": "s3", "tool": "add", "params": {"a": 0, "b": 0}, "deps": ["s1", "s2"]}
            ]
        });

        let plan = Plan::from_json(&plan_json).unwrap();
        assert_eq!(plan.steps().len(), 3);
        assert_eq!(plan.steps()[2].deps, vec!["s1", "s2"]);
    }

    #[tokio::test]
    async fn executes_single_step_plan() {
        let llm = Arc::new(MockLlm::new(vec![MockLlm::text(
            &serde_json::json!({
                "steps": [
                    {"id": "s1", "tool": "add", "params": {"a": 10, "b": 20}, "deps": []}
                ]
            })
            .to_string(),
        )]));
        let registry = Arc::new(make_registry());

        let agent = PlanExecuteAgent::new(llm, registry);
        let output = agent.run("Add 10 and 20").await.unwrap();

        assert_eq!(output.step_result("s1").unwrap(), "30");
    }

    #[tokio::test]
    async fn executes_parallel_independent_steps() {
        let llm = Arc::new(MockLlm::new(vec![MockLlm::text(
            &serde_json::json!({
                "steps": [
                    {"id": "s1", "tool": "add", "params": {"a": 1, "b": 2}, "deps": []},
                    {"id": "s2", "tool": "multiply", "params": {"a": 3, "b": 4}, "deps": []}
                ]
            })
            .to_string(),
        )]));
        let registry = Arc::new(make_registry());

        let agent = PlanExecuteAgent::new(llm, registry);
        let output = agent.run("Compute both").await.unwrap();

        assert_eq!(output.step_result("s1").unwrap(), "3");
        assert_eq!(output.step_result("s2").unwrap(), "12");
    }

    #[tokio::test]
    async fn executes_dependent_steps_in_order() {
        let llm = Arc::new(MockLlm::new(vec![MockLlm::text(
            &serde_json::json!({
                "steps": [
                    {"id": "s1", "tool": "add", "params": {"a": 5, "b": 5}, "deps": []},
                    {"id": "s2", "tool": "multiply", "params": {"a": 2, "b": 3}, "deps": ["s1"]}
                ]
            })
            .to_string(),
        )]));
        let registry = Arc::new(make_registry());

        let agent = PlanExecuteAgent::new(llm, registry);
        let output = agent.run("Chain").await.unwrap();

        assert_eq!(output.step_result("s1").unwrap(), "10");
        assert_eq!(output.step_result("s2").unwrap(), "6");
        assert_eq!(output.step_count(), 2);
    }

    #[tokio::test]
    async fn invalid_plan_json_returns_error() {
        let llm = Arc::new(MockLlm::new(vec![MockLlm::text("not valid json plan")]));
        let registry = Arc::new(make_registry());

        let agent = PlanExecuteAgent::new(llm, registry);
        let result = agent.run("Bad plan").await;

        assert!(matches!(result, Err(PlanError::InvalidPlan(_))));
    }

    #[tokio::test]
    async fn llm_failure_returns_error() {
        struct FailLlm;

        #[async_trait::async_trait]
        impl LlmProvider for FailLlm {
            fn name(&self) -> &str {
                "fail"
            }
            async fn complete(
                &self,
                _req: CompletionRequest,
            ) -> Result<CompletionResponse, LlmError> {
                Err(LlmError::Api {
                    status: 500,
                    message: "down".into(),
                })
            }
        }

        let llm: Arc<dyn LlmProvider> = Arc::new(FailLlm);
        let registry = Arc::new(make_registry());

        let agent = PlanExecuteAgent::new(llm, registry);
        let result = agent.run("Will fail").await;

        assert!(matches!(result, Err(PlanError::Llm(_))));
    }
}
