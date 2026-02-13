//! Plan-Execute agent — LLM generates a plan, workflow engine executes it.
//!
//! Usage:
//!   plan-execute-agent --prompt "Add 10 and 20, then multiply by 3"
//!   `OPENAI_API_KEY=sk`-... plan-execute-agent -p "Compute (5+5) * (3+4)"

use std::fmt;
use std::sync::Arc;

use clap::Parser;
use erio_core::Message;
use erio_llm_client::{CompletionRequest, LlmProvider, OpenAiProvider};
use erio_tools::{PropertyType, Tool, ToolRegistry, ToolResult, ToolSchema};
use erio_workflow::{Step, StepOutput, Workflow, WorkflowContext, WorkflowEngine, WorkflowError};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// Plan-Execute agent: LLM produces a JSON plan, then a DAG workflow executes it.
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// The prompt describing what to compute.
    #[arg(short, long)]
    prompt: String,

    /// OpenAI-compatible API base URL.
    #[arg(long, env = "OPENAI_BASE_URL", default_value = "https://api.openai.com/v1")]
    base_url: String,

    /// API key (reads from `OPENAI_API_KEY` env var by default).
    #[arg(long, env = "OPENAI_API_KEY", default_value = "")]
    api_key: String,

    /// Model name to use.
    #[arg(short, long, default_value = "gpt-4")]
    model: String,
}

// ---------------------------------------------------------------------------
// Agent types
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum PlanError {
    Llm(erio_llm_client::LlmError),
    InvalidPlan(String),
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

#[derive(Debug, Clone, serde::Deserialize)]
struct PlanStep {
    id: String,
    tool: String,
    params: serde_json::Value,
    deps: Vec<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct Plan {
    steps: Vec<PlanStep>,
}

// ---------------------------------------------------------------------------
// Workflow step adapter
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Plan-execute pipeline
// ---------------------------------------------------------------------------

async fn run_plan_execute(
    llm: Arc<dyn LlmProvider>,
    tools: Arc<ToolRegistry>,
    model: &str,
    prompt: &str,
) -> Result<(), PlanError> {
    // 1. Ask LLM for a plan
    println!("Generating plan...");
    let request = CompletionRequest::new(model)
        .message(Message::system(
            "You are a planner. Return a JSON object with a \"steps\" array. \
             Each step has: id, tool, params, deps.",
        ))
        .message(Message::user(prompt));

    let response = llm.complete(request).await.map_err(PlanError::Llm)?;
    let raw = response.content.unwrap_or_default();

    // 2. Parse the plan
    let json: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| PlanError::InvalidPlan(e.to_string()))?;
    let plan: Plan =
        serde_json::from_value(json).map_err(|e| PlanError::InvalidPlan(e.to_string()))?;

    println!("Plan has {} steps:", plan.steps.len());
    for step in &plan.steps {
        let deps = if step.deps.is_empty() {
            String::new()
        } else {
            format!(" (depends on: {})", step.deps.join(", "))
        };
        println!("  {} -> {}{deps}", step.id, step.tool);
    }

    // 3. Build a workflow DAG
    let mut builder = Workflow::builder();
    for step in &plan.steps {
        let deps: Vec<&str> = step.deps.iter().map(String::as_str).collect();
        builder = builder.step(
            ToolExecStep {
                id: step.id.clone(),
                tool_name: step.tool.clone(),
                params: step.params.clone(),
                registry: Arc::clone(&tools),
            },
            &deps,
        );
    }
    let workflow = builder.build().map_err(PlanError::Execution)?;

    // 4. Execute
    println!("\nExecuting...");
    let engine = WorkflowEngine::new();
    let ctx = engine.run(workflow).await.map_err(PlanError::Execution)?;

    // 5. Print results
    println!("\nResults:");
    for step in &plan.steps {
        if let Some(output) = ctx.output(&step.id) {
            println!("  {} = {}", step.id, output.value());
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Built-in demo tools
// ---------------------------------------------------------------------------

struct AddTool;

#[async_trait::async_trait]
impl Tool for AddTool {
    fn name(&self) -> &'static str {
        "add"
    }
    fn description(&self) -> &'static str {
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
    fn name(&self) -> &'static str {
        "multiply"
    }
    fn description(&self) -> &'static str {
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let llm: Arc<dyn LlmProvider> = Arc::new(OpenAiProvider::new(&cli.base_url, &cli.api_key));

    let mut registry = ToolRegistry::new();
    registry.register(AddTool);
    registry.register(MultiplyTool);
    let tools = Arc::new(registry);

    if let Err(e) = run_plan_execute(llm, tools, &cli.model, &cli.prompt).await {
        eprintln!("Error: {e}");
    }
}
