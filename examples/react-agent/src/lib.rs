//! ReAct agent — Think → Act → Observe loop using LLM + Tools.

use std::fmt;
use std::sync::Arc;

use erio_core::Message;
use erio_llm_client::{CompletionRequest, LlmProvider};
use erio_tools::ToolRegistry;

/// Errors from the ReAct agent.
#[derive(Debug)]
pub enum AgentError {
    /// LLM provider returned an error.
    Llm(erio_llm_client::LlmError),
    /// Tool execution failed.
    Tool(erio_core::ToolError),
    /// Exceeded maximum iteration count.
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

/// The result of a completed ReAct run.
#[derive(Debug)]
pub struct AgentOutput {
    answer: String,
    steps: usize,
    messages: Vec<Message>,
}

impl AgentOutput {
    /// The final text answer from the agent.
    pub fn final_answer(&self) -> &str {
        &self.answer
    }

    /// Number of LLM calls made.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Full message history from the run.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }
}

const DEFAULT_MAX_ITERATIONS: usize = 10;
const DEFAULT_MODEL: &str = "gpt-4";

/// A ReAct (Reason + Act) loop agent.
///
/// Iteratively calls an LLM, executes any requested tools,
/// and feeds results back until the LLM produces a final answer.
pub struct ReactAgent {
    llm: Arc<dyn LlmProvider>,
    tools: Arc<ToolRegistry>,
    max_iterations: usize,
    model: String,
}

impl ReactAgent {
    /// Creates a new agent with default settings.
    pub fn new(llm: Arc<dyn LlmProvider>, tools: Arc<ToolRegistry>) -> Self {
        Self {
            llm,
            tools,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            model: DEFAULT_MODEL.into(),
        }
    }

    /// Returns a builder for custom configuration.
    pub fn builder(llm: Arc<dyn LlmProvider>, tools: Arc<ToolRegistry>) -> ReactAgentBuilder {
        ReactAgentBuilder {
            llm,
            tools,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            model: DEFAULT_MODEL.into(),
        }
    }

    /// Runs the ReAct loop with the given user prompt.
    pub async fn run(&self, prompt: &str) -> Result<AgentOutput, AgentError> {
        let mut messages = vec![
            Message::system("You are a helpful assistant. Use tools when needed."),
            Message::user(prompt),
        ];

        for step in 1..=self.max_iterations {
            let request = self.build_request(&messages);
            let response = self.llm.complete(request).await.map_err(AgentError::Llm)?;

            if response.tool_calls.is_empty() {
                // Final answer — no more tool calls
                let answer = response.content.unwrap_or_default();
                messages.push(Message::assistant(&answer));
                return Ok(AgentOutput {
                    answer,
                    steps: step,
                    messages,
                });
            }

            // Execute each tool call and collect results
            for call in &response.tool_calls {
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

/// Builder for `ReactAgent`.
pub struct ReactAgentBuilder {
    llm: Arc<dyn LlmProvider>,
    tools: Arc<ToolRegistry>,
    max_iterations: usize,
    model: String,
}

impl ReactAgentBuilder {
    /// Sets the maximum number of LLM calls before aborting.
    #[must_use]
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Sets the model name.
    #[must_use]
    pub fn model(mut self, model: &str) -> Self {
        self.model = model.into();
        self
    }

    /// Builds the agent.
    pub fn build(self) -> ReactAgent {
        ReactAgent {
            llm: self.llm,
            tools: self.tools,
            max_iterations: self.max_iterations,
            model: self.model,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use erio_core::ToolCall;
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

        fn tool_call(name: &str, args: serde_json::Value) -> CompletionResponse {
            CompletionResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: format!("call_{name}"),
                    name: name.into(),
                    arguments: args,
                }],
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

    // === Mock Tool ===

    struct UpperTool;

    #[async_trait::async_trait]
    impl Tool for UpperTool {
        fn name(&self) -> &str {
            "uppercase"
        }
        fn description(&self) -> &str {
            "Uppercases text"
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

    // === Helper ===

    fn make_registry() -> ToolRegistry {
        let mut reg = ToolRegistry::new();
        reg.register(UpperTool);
        reg
    }

    // === Tests ===

    #[tokio::test]
    async fn returns_final_answer_without_tool_calls() {
        let llm = Arc::new(MockLlm::new(vec![MockLlm::text("Hello!")]));
        let registry = Arc::new(make_registry());

        let agent = ReactAgent::new(llm, registry);
        let output = agent.run("Say hello").await.unwrap();

        assert_eq!(output.final_answer(), "Hello!");
        assert_eq!(output.steps(), 1);
    }

    #[tokio::test]
    async fn executes_tool_then_returns_answer() {
        let llm = Arc::new(MockLlm::new(vec![
            // Step 1: LLM requests tool call
            MockLlm::tool_call("uppercase", serde_json::json!({"text": "hello"})),
            // Step 2: LLM sees tool result, returns final answer
            MockLlm::text("The result is: HELLO"),
        ]));
        let registry = Arc::new(make_registry());

        let agent = ReactAgent::new(llm, registry);
        let output = agent.run("Uppercase hello").await.unwrap();

        assert_eq!(output.final_answer(), "The result is: HELLO");
        assert_eq!(output.steps(), 2);
    }

    #[tokio::test]
    async fn multiple_tool_calls_in_sequence() {
        let llm = Arc::new(MockLlm::new(vec![
            MockLlm::tool_call("uppercase", serde_json::json!({"text": "first"})),
            MockLlm::tool_call("uppercase", serde_json::json!({"text": "second"})),
            MockLlm::text("Done: FIRST and SECOND"),
        ]));
        let registry = Arc::new(make_registry());

        let agent = ReactAgent::new(llm, registry);
        let output = agent.run("Process both").await.unwrap();

        assert_eq!(output.final_answer(), "Done: FIRST and SECOND");
        assert_eq!(output.steps(), 3);
    }

    #[tokio::test]
    async fn respects_max_iterations() {
        // LLM always returns tool calls — should hit the limit
        let llm = Arc::new(MockLlm::new(vec![
            MockLlm::tool_call("uppercase", serde_json::json!({"text": "a"})),
            MockLlm::tool_call("uppercase", serde_json::json!({"text": "b"})),
            MockLlm::tool_call("uppercase", serde_json::json!({"text": "c"})),
            MockLlm::tool_call("uppercase", serde_json::json!({"text": "d"})),
            MockLlm::tool_call("uppercase", serde_json::json!({"text": "e"})),
        ]));
        let registry = Arc::new(make_registry());

        let agent = ReactAgent::builder(llm, registry)
            .max_iterations(3)
            .build();
        let result = agent.run("Loop forever").await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, AgentError::MaxIterations(3)),
            "expected MaxIterations, got: {err}"
        );
    }

    #[tokio::test]
    async fn llm_error_propagates() {
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

        let agent = ReactAgent::new(llm, registry);
        let result = agent.run("Will fail").await;

        assert!(matches!(result, Err(AgentError::Llm(_))));
    }

    #[tokio::test]
    async fn tool_not_found_returns_error() {
        let llm = Arc::new(MockLlm::new(vec![MockLlm::tool_call(
            "nonexistent",
            serde_json::json!({}),
        )]));
        let registry = Arc::new(ToolRegistry::new()); // empty

        let agent = ReactAgent::new(llm, registry);
        let result = agent.run("Call missing tool").await;

        assert!(matches!(result, Err(AgentError::Tool(_))));
    }

    #[tokio::test]
    async fn collects_message_history() {
        let llm = Arc::new(MockLlm::new(vec![
            MockLlm::tool_call("uppercase", serde_json::json!({"text": "hi"})),
            MockLlm::text("Done"),
        ]));
        let registry = Arc::new(make_registry());

        let agent = ReactAgent::new(llm, registry);
        let output = agent.run("Test history").await.unwrap();

        // Should have: system + user + assistant(tool_call) + tool_result + assistant(final)
        assert!(output.messages().len() >= 4);
    }
}
