//! Multi-agent chat — specialist agents coordinated via `EventBus`.

use std::fmt;
use std::sync::Arc;

use erio_core::Message;
use erio_event_bus::{Event, EventBus, EventBusError, EventSource};
use erio_llm_client::{CompletionRequest, LlmError, LlmProvider};
use tokio::sync::Mutex;

const DEFAULT_MODEL: &str = "gpt-4";

/// Errors from the multi-agent coordinator.
#[derive(Debug)]
pub enum CoordinatorError {
    /// An agent's LLM call failed.
    AgentLlm { agent: String, source: LlmError },
    /// `EventBus` operation failed.
    Bus(EventBusError),
    /// No agents were registered.
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

/// A single agent reply collected during coordination.
#[derive(Debug, Clone)]
pub struct AgentReply {
    pub agent_name: String,
    pub content: String,
}

/// The output of a completed coordinator run.
#[derive(Debug)]
pub struct CoordinatorOutput {
    replies: Vec<AgentReply>,
}

impl CoordinatorOutput {
    /// Returns the reply from the named agent, if any.
    pub fn reply_from(&self, name: &str) -> Option<&str> {
        self.replies
            .iter()
            .find(|r| r.agent_name == name)
            .map(|r| r.content.as_str())
    }

    /// Returns all collected replies in order.
    pub fn all_replies(&self) -> &[AgentReply] {
        &self.replies
    }

    /// Number of agents that produced replies.
    pub fn agent_count(&self) -> usize {
        self.replies.len()
    }
}

/// A specialist agent that implements `EventSource`.
///
/// When polled, it takes a pending prompt, calls the LLM, and returns a reply event.
pub struct SpecialistAgent {
    name: String,
    system_prompt: String,
    llm: Arc<dyn LlmProvider>,
    model: String,
    prompt: Arc<Mutex<Option<String>>>,
}

impl SpecialistAgent {
    /// Creates a new specialist agent with default settings.
    pub fn new(
        name: impl Into<String>,
        system_prompt: impl Into<String>,
        llm: Arc<dyn LlmProvider>,
    ) -> Self {
        Self {
            name: name.into(),
            system_prompt: system_prompt.into(),
            llm,
            model: DEFAULT_MODEL.into(),
            prompt: Arc::new(Mutex::new(None)),
        }
    }

    /// Returns a builder for custom configuration.
    pub fn builder(
        name: impl Into<String>,
        system_prompt: impl Into<String>,
        llm: Arc<dyn LlmProvider>,
    ) -> SpecialistAgentBuilder {
        SpecialistAgentBuilder {
            name: name.into(),
            system_prompt: system_prompt.into(),
            llm,
            model: DEFAULT_MODEL.into(),
        }
    }

    /// Sets the pending prompt for this agent.
    pub async fn set_prompt(&self, user_prompt: String) {
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

/// Builder for `SpecialistAgent`.
pub struct SpecialistAgentBuilder {
    name: String,
    system_prompt: String,
    llm: Arc<dyn LlmProvider>,
    model: String,
}

impl SpecialistAgentBuilder {
    /// Sets the model name.
    #[must_use]
    pub fn model(mut self, model: &str) -> Self {
        self.model = model.into();
        self
    }

    /// Builds the specialist agent.
    pub fn build(self) -> SpecialistAgent {
        SpecialistAgent {
            name: self.name,
            system_prompt: self.system_prompt,
            llm: self.llm,
            model: self.model,
            prompt: Arc::new(Mutex::new(None)),
        }
    }
}

/// Orchestrates specialist agents through the event bus.
pub struct Coordinator {
    agents: Vec<Arc<SpecialistAgent>>,
}

impl Coordinator {
    /// Creates a new coordinator with no agents.
    pub fn new() -> Self {
        Self { agents: vec![] }
    }

    /// Registers a specialist agent.
    #[must_use]
    pub fn register(mut self, agent: SpecialistAgent) -> Self {
        self.agents.push(Arc::new(agent));
        self
    }

    /// Runs the coordination pipeline with the given user prompt.
    pub async fn run(&self, prompt: &str) -> Result<CoordinatorOutput, CoordinatorError> {
        if self.agents.is_empty() {
            return Err(CoordinatorError::NoAgents);
        }

        let bus = EventBus::new();
        let mut rx = bus.subscribe();

        let mut replies = Vec::new();
        let mut accumulated = prompt.to_string();

        for agent in &self.agents {
            agent.set_prompt(accumulated.clone()).await;

            // Poll the agent directly so LLM errors propagate
            let event = agent.next_event().await.map_err(|e| {
                CoordinatorError::AgentLlm {
                    agent: agent.name.clone(),
                    source: LlmError::InvalidResponse(e.to_string()),
                }
            })?;

            let Some(event) = event else {
                continue;
            };

            // Publish to bus for subscribers
            let _ = bus.publish(event);

            let received = rx
                .recv()
                .await
                .map_err(|_| CoordinatorError::Bus(EventBusError::ChannelClosed))?;

            let reply_text = received.data["content"]
                .as_str()
                .unwrap_or_default()
                .to_string();

            replies.push(AgentReply {
                agent_name: received.source.clone(),
                content: reply_text.clone(),
            });

            // Build context for the next agent
            accumulated = format!(
                "{prompt}\n\nPrevious agent ({}) responded:\n{reply_text}",
                received.source
            );
        }

        Ok(CoordinatorOutput { replies })
    }
}

impl Default for Coordinator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use erio_llm_client::CompletionResponse;
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
        fn name(&self) -> &'static str {
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

    // === Tests ===

    #[tokio::test]
    async fn coordinator_with_no_agents_returns_error() {
        let coordinator = Coordinator::new();
        let result = coordinator.run("hello").await;
        assert!(matches!(result, Err(CoordinatorError::NoAgents)));
    }

    #[tokio::test]
    async fn single_agent_produces_reply() {
        let llm = Arc::new(MockLlm::new(vec![MockLlm::text("I'm a researcher")]));
        let agent = SpecialistAgent::new("researcher", "You research topics.", llm);

        let coordinator = Coordinator::new().register(agent);
        let output = coordinator.run("Tell me about Rust").await.unwrap();

        assert_eq!(output.agent_count(), 1);
        assert_eq!(output.reply_from("researcher"), Some("I'm a researcher"));
    }

    #[tokio::test]
    async fn reply_event_has_correct_source_and_type() {
        let llm = Arc::new(MockLlm::new(vec![MockLlm::text("hello")]));
        let agent = SpecialistAgent::new("analyst", "You analyze.", llm);

        agent.set_prompt("test prompt".into()).await;
        let event = agent.next_event().await.unwrap().unwrap();

        assert_eq!(event.source, "analyst");
        assert_eq!(event.event_type, "reply");
    }

    #[tokio::test]
    async fn researcher_reply_used_as_writer_context() {
        let researcher_llm = Arc::new(MockLlm::new(vec![MockLlm::text("Research findings")]));
        let writer_llm = Arc::new(MockLlm::new(vec![MockLlm::text("Written article")]));

        let researcher = SpecialistAgent::new("researcher", "You research.", researcher_llm);
        let writer = SpecialistAgent::new("writer", "You write.", writer_llm);

        let coordinator = Coordinator::new().register(researcher).register(writer);
        let output = coordinator.run("Write about Rust").await.unwrap();

        assert_eq!(output.reply_from("researcher"), Some("Research findings"));
        assert_eq!(output.reply_from("writer"), Some("Written article"));
    }

    #[tokio::test]
    async fn two_agents_both_produce_replies() {
        let llm1 = Arc::new(MockLlm::new(vec![MockLlm::text("Reply A")]));
        let llm2 = Arc::new(MockLlm::new(vec![MockLlm::text("Reply B")]));

        let agent1 = SpecialistAgent::new("agent-a", "You are A.", llm1);
        let agent2 = SpecialistAgent::new("agent-b", "You are B.", llm2);

        let coordinator = Coordinator::new().register(agent1).register(agent2);
        let output = coordinator.run("go").await.unwrap();

        assert_eq!(output.agent_count(), 2);
    }

    #[tokio::test]
    async fn reply_from_returns_none_for_unknown_agent() {
        let llm = Arc::new(MockLlm::new(vec![MockLlm::text("hello")]));
        let agent = SpecialistAgent::new("known", "You reply.", llm);

        let coordinator = Coordinator::new().register(agent);
        let output = coordinator.run("test").await.unwrap();

        assert!(output.reply_from("unknown").is_none());
    }

    #[tokio::test]
    async fn reply_from_returns_content_for_known_agent() {
        let llm = Arc::new(MockLlm::new(vec![MockLlm::text("specific content")]));
        let agent = SpecialistAgent::new("expert", "You are an expert.", llm);

        let coordinator = Coordinator::new().register(agent);
        let output = coordinator.run("question").await.unwrap();

        assert_eq!(output.reply_from("expert"), Some("specific content"));
    }

    #[tokio::test]
    async fn llm_failure_on_researcher_propagates_error() {
        struct FailLlm;

        #[async_trait::async_trait]
        impl LlmProvider for FailLlm {
            fn name(&self) -> &'static str {
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
        let agent = SpecialistAgent::new("researcher", "You research.", llm);

        let coordinator = Coordinator::new().register(agent);
        let result = coordinator.run("fail").await;

        assert!(matches!(
            result,
            Err(CoordinatorError::AgentLlm { agent, .. }) if agent == "researcher"
        ));
    }

    #[tokio::test]
    async fn llm_failure_on_writer_propagates_error() {
        struct FailLlm;

        #[async_trait::async_trait]
        impl LlmProvider for FailLlm {
            fn name(&self) -> &'static str {
                "fail"
            }
            async fn complete(
                &self,
                _req: CompletionRequest,
            ) -> Result<CompletionResponse, LlmError> {
                Err(LlmError::Api {
                    status: 500,
                    message: "writer down".into(),
                })
            }
        }

        let good_llm = Arc::new(MockLlm::new(vec![MockLlm::text("research done")]));
        let bad_llm: Arc<dyn LlmProvider> = Arc::new(FailLlm);

        let researcher = SpecialistAgent::new("researcher", "You research.", good_llm);
        let writer = SpecialistAgent::new("writer", "You write.", bad_llm);

        let coordinator = Coordinator::new().register(researcher).register(writer);
        let result = coordinator.run("fail on writer").await;

        assert!(matches!(
            result,
            Err(CoordinatorError::AgentLlm { agent, .. }) if agent == "writer"
        ));
    }

    #[tokio::test]
    async fn builder_sets_custom_model() {
        let llm = Arc::new(MockLlm::new(vec![MockLlm::text("ok")]));
        let agent = SpecialistAgent::builder("test", "You test.", llm)
            .model("gpt-3.5-turbo")
            .build();

        assert_eq!(agent.model, "gpt-3.5-turbo");
    }

    #[tokio::test]
    async fn agent_count_matches_registered_agents() {
        let llm1 = Arc::new(MockLlm::new(vec![MockLlm::text("a")]));
        let llm2 = Arc::new(MockLlm::new(vec![MockLlm::text("b")]));
        let llm3 = Arc::new(MockLlm::new(vec![MockLlm::text("c")]));

        let coordinator = Coordinator::new()
            .register(SpecialistAgent::new("a", "A", llm1))
            .register(SpecialistAgent::new("b", "B", llm2))
            .register(SpecialistAgent::new("c", "C", llm3));

        let output = coordinator.run("test").await.unwrap();
        assert_eq!(output.agent_count(), 3);
    }

    #[tokio::test]
    async fn all_replies_preserves_order() {
        let llm1 = Arc::new(MockLlm::new(vec![MockLlm::text("first")]));
        let llm2 = Arc::new(MockLlm::new(vec![MockLlm::text("second")]));

        let coordinator = Coordinator::new()
            .register(SpecialistAgent::new("alpha", "A", llm1))
            .register(SpecialistAgent::new("beta", "B", llm2));

        let output = coordinator.run("order test").await.unwrap();
        let replies = output.all_replies();

        assert_eq!(replies[0].agent_name, "alpha");
        assert_eq!(replies[0].content, "first");
        assert_eq!(replies[1].agent_name, "beta");
        assert_eq!(replies[1].content, "second");
    }
}
