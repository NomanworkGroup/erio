//! Multi-agent chat — specialist agents coordinated via `EventBus`.
//!
//! Usage:
//!   multi-agent-chat --prompt "Write an article about Rust"
//!   `OPENAI_API_KEY=sk`-... multi-agent-chat -p "Explain async/await"

use std::fmt;
use std::sync::Arc;

use clap::Parser;
use erio_core::Message;
use erio_event_bus::{Event, EventBus, EventBusError, EventSource};
use erio_llm_client::{CompletionRequest, LlmError, LlmProvider, OpenAiProvider};
use tokio::sync::Mutex;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// Multi-agent chat: specialist agents coordinate through an event bus.
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// The prompt to send to the agent pipeline.
    #[arg(short, long)]
    prompt: String,

    /// OpenAI-compatible API base URL.
    #[arg(
        long,
        env = "OPENAI_BASE_URL",
        default_value = "https://api.openai.com/v1"
    )]
    base_url: String,

    /// API key (reads from `OPENAI_API_KEY` env var by default).
    #[arg(long, env = "OPENAI_API_KEY", default_value = "")]
    api_key: String,

    /// Model name to use.
    #[arg(short, long, default_value = "gpt-4o-mini")]
    model: String,

    /// Comma-separated list of agent roles (`name:system_prompt`).
    /// Defaults to researcher + writer if not provided.
    #[arg(long, value_delimiter = ',')]
    agents: Vec<String>,
}

// ---------------------------------------------------------------------------
// Agent types
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum CoordinatorError {
    AgentLlm { agent: String, source: LlmError },
    Bus(EventBusError),
    NoAgents,
}

impl fmt::Display for CoordinatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AgentLlm { agent, source } => {
                write!(f, "agent '{agent}' LLM error: {source}")
            }
            Self::Bus(e) => write!(f, "event bus error: {e}"),
            Self::NoAgents => write!(f, "no agents registered"),
        }
    }
}

impl std::error::Error for CoordinatorError {}

struct SpecialistAgent {
    name: String,
    system_prompt: String,
    llm: Arc<dyn LlmProvider>,
    model: String,
    prompt: Arc<Mutex<Option<String>>>,
}

impl SpecialistAgent {
    fn new(
        name: impl Into<String>,
        system_prompt: impl Into<String>,
        llm: Arc<dyn LlmProvider>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            system_prompt: system_prompt.into(),
            llm,
            model: model.into(),
            prompt: Arc::new(Mutex::new(None)),
        }
    }

    async fn set_prompt(&self, user_prompt: String) {
        let mut guard = self.prompt.lock().await;
        *guard = Some(user_prompt);
    }
}

#[async_trait::async_trait]
impl EventSource for SpecialistAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &'static str {
        "Specialist agent event source"
    }

    async fn next_event(&self) -> Result<Option<Event>, EventBusError> {
        let pending = {
            let mut guard = self.prompt.lock().await;
            guard.take()
        };

        let Some(pending) = pending else {
            return Ok(None);
        };

        let request = CompletionRequest::new(&self.model)
            .message(Message::system(&self.system_prompt))
            .message(Message::user(&pending));

        let response = self
            .llm
            .complete(request)
            .await
            .map_err(|e| EventBusError::Source(e.to_string()))?;

        let reply_text = response.content.unwrap_or_default();
        let event = Event::new(
            &self.name,
            "reply",
            serde_json::json!({ "content": reply_text }),
        );

        Ok(Some(event))
    }
}

async fn run_coordinator(
    agents: Vec<Arc<SpecialistAgent>>,
    prompt: &str,
) -> Result<(), CoordinatorError> {
    if agents.is_empty() {
        return Err(CoordinatorError::NoAgents);
    }

    let bus = EventBus::new();
    let mut rx = bus.subscribe();
    let mut accumulated = prompt.to_string();

    for agent in &agents {
        agent.set_prompt(accumulated.clone()).await;

        let event = agent
            .next_event()
            .await
            .map_err(|e| CoordinatorError::AgentLlm {
                agent: agent.name.clone(),
                source: LlmError::InvalidResponse(e.to_string()),
            })?;

        let Some(event) = event else {
            continue;
        };

        let _ = bus.publish(event);

        let received = rx
            .recv()
            .await
            .map_err(|_| CoordinatorError::Bus(EventBusError::ChannelClosed))?;

        let reply_text = received.data["content"]
            .as_str()
            .unwrap_or_default()
            .to_string();

        println!("--- {} ---\n{reply_text}\n", received.source);

        accumulated = format!(
            "{prompt}\n\nPrevious agent ({}) responded:\n{reply_text}",
            received.source
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn parse_agent_spec(spec: &str) -> (String, String) {
    if let Some((name, prompt)) = spec.split_once(':') {
        (name.to_string(), prompt.to_string())
    } else {
        (spec.to_string(), format!("You are a {spec}."))
    }
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let llm: Arc<dyn LlmProvider> = Arc::new(OpenAiProvider::new(&cli.base_url, &cli.api_key));

    let agents: Vec<Arc<SpecialistAgent>> = if cli.agents.is_empty() {
        vec![
            Arc::new(SpecialistAgent::new(
                "researcher",
                "You are a researcher. Provide thorough research findings.",
                Arc::clone(&llm),
                &cli.model,
            )),
            Arc::new(SpecialistAgent::new(
                "writer",
                "You are a writer. Produce a polished article from the research.",
                Arc::clone(&llm),
                &cli.model,
            )),
        ]
    } else {
        cli.agents
            .iter()
            .map(|spec| {
                let (name, prompt) = parse_agent_spec(spec);
                Arc::new(SpecialistAgent::new(
                    name,
                    prompt,
                    Arc::clone(&llm),
                    &cli.model,
                ))
            })
            .collect()
    };

    if let Err(e) = run_coordinator(agents, &cli.prompt).await {
        eprintln!("Error: {e}");
    }
}
