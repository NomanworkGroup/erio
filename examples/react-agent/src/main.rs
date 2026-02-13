//! `ReAct` agent — Think -> Act -> Observe loop using LLM + Tools.
//!
//! Usage:
//!   react-agent --prompt "What is 2+2?" --base-url <https://api.openai.com/v1>
//!   `OPENAI_API_KEY=sk`-... react-agent -p "Uppercase hello"

use std::fmt;
use std::sync::Arc;

use clap::Parser;
use erio_core::Message;
use erio_llm_client::{CompletionRequest, LlmProvider, OpenAiProvider};
use erio_tools::{PropertyType, Tool, ToolRegistry, ToolResult, ToolSchema};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// `ReAct` (Reason + Act) loop agent example.
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// The prompt to send to the agent.
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

    /// Maximum number of `ReAct` iterations.
    #[arg(long, default_value_t = 10)]
    max_iterations: usize,
}

// ---------------------------------------------------------------------------
// Agent types
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum AgentError {
    Llm(erio_llm_client::LlmError),
    Tool(erio_core::ToolError),
    MaxIterations(usize),
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Llm(e) => write!(f, "LLM error: {e}"),
            Self::Tool(e) => write!(f, "Tool error: {e}"),
            Self::MaxIterations(n) => write!(f, "exceeded max iterations ({n})"),
        }
    }
}

impl std::error::Error for AgentError {}

struct ReactAgent {
    llm: Arc<dyn LlmProvider>,
    tools: Arc<ToolRegistry>,
    max_iterations: usize,
    model: String,
}

impl ReactAgent {
    async fn run(&self, prompt: &str) -> Result<String, AgentError> {
        let mut messages = vec![
            Message::system("You are a helpful assistant. Use tools when needed."),
            Message::user(prompt),
        ];

        for step in 1..=self.max_iterations {
            let request = self.build_request(&messages);
            let response = self.llm.complete(request).await.map_err(AgentError::Llm)?;

            if response.tool_calls.is_empty() {
                let answer = response.content.unwrap_or_default();
                messages.push(Message::assistant(&answer));
                println!("[step {step}] Final answer");
                return Ok(answer);
            }

            for call in &response.tool_calls {
                println!("[step {step}] Calling tool: {} ", call.name);
                messages.push(Message {
                    role: erio_core::Role::Assistant,
                    content: vec![erio_core::Content::ToolCall(call.clone())],
                    tool_call_id: None,
                });
                let result = self
                    .tools
                    .execute(&call.name, call.arguments.clone())
                    .await
                    .map_err(AgentError::Tool)?;
                let content = result.content().unwrap_or("").to_string();
                println!("[step {step}] Tool result: {content}");
                messages.push(Message::tool_result(&call.id, &content));
            }
        }

        Err(AgentError::MaxIterations(self.max_iterations))
    }

    fn build_request(&self, messages: &[Message]) -> CompletionRequest {
        let mut req = CompletionRequest::new(&self.model);
        for msg in messages {
            req = req.message(msg.clone());
        }
        req
    }
}

// ---------------------------------------------------------------------------
// Built-in demo tool
// ---------------------------------------------------------------------------

struct UppercaseTool;

#[async_trait::async_trait]
impl Tool for UppercaseTool {
    fn name(&self) -> &'static str {
        "uppercase"
    }
    fn description(&self) -> &'static str {
        "Converts text to uppercase"
    }
    fn schema(&self) -> ToolSchema {
        ToolSchema::builder()
            .property("text", PropertyType::String, "Text to uppercase", true)
            .build()
    }
    async fn execute(
        &self,
        params: serde_json::Value,
    ) -> Result<ToolResult, erio_core::ToolError> {
        let text = params["text"].as_str().unwrap_or("");
        Ok(ToolResult::success(text.to_uppercase()))
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
    registry.register(UppercaseTool);
    let tools = Arc::new(registry);

    let agent = ReactAgent {
        llm,
        tools,
        max_iterations: cli.max_iterations,
        model: cli.model,
    };

    match agent.run(&cli.prompt).await {
        Ok(answer) => println!("\n{answer}"),
        Err(e) => eprintln!("Error: {e}"),
    }
}
