//! Event-driven file summariser using `EventBus` + LLM.

use std::fmt;
use std::sync::Arc;

use erio_core::Message;
use erio_event_bus::{EventBus, EventBusError};
use erio_llm_client::{CompletionRequest, LlmError, LlmProvider};

/// Event type for newly created files.
pub const FILE_CREATED_EVENT: &str = "file.created";
/// Event type for modified files.
pub const FILE_MODIFIED_EVENT: &str = "file.modified";

const DEFAULT_MODEL: &str = "gpt-4";
const DEFAULT_MAX_EVENTS: usize = 100;
const DEFAULT_SYSTEM_PROMPT: &str = "You are a file summariser. Given a file's content, produce a concise summary.";

/// Errors from the file watcher agent.
#[derive(Debug)]
pub enum WatcherError {
    /// Event bus operation failed.
    Bus(EventBusError),
    /// LLM call failed.
    Llm(LlmError),
    /// Invalid event data.
    InvalidEvent(String),
}

impl fmt::Display for WatcherError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bus(e) => write!(f, "event bus error: {e}"),
            Self::Llm(e) => write!(f, "LLM error: {e}"),
            Self::InvalidEvent(msg) => write!(f, "invalid event: {msg}"),
        }
    }
}

impl std::error::Error for WatcherError {}

/// Summary of a single file event.
#[derive(Debug, Clone)]
pub struct FileSummary {
    pub path: String,
    pub summary: String,
    pub event_type: String,
}

/// Output from the file watcher agent.
#[derive(Debug)]
pub struct WatcherOutput {
    summaries: Vec<FileSummary>,
}

impl WatcherOutput {
    /// Returns all collected summaries.
    pub fn summaries(&self) -> &[FileSummary] {
        &self.summaries
    }

    /// Returns the number of summaries.
    pub fn summary_count(&self) -> usize {
        self.summaries.len()
    }

    /// Finds a summary by file path.
    pub fn find_by_path(&self, path: &str) -> Option<&FileSummary> {
        self.summaries.iter().find(|s| s.path == path)
    }
}

/// An agent that watches for file events and summarises their content.
pub struct FileWatcherAgent {
    llm: Arc<dyn LlmProvider>,
    model: String,
    max_events: usize,
    system_prompt: String,
}

impl FileWatcherAgent {
    /// Creates a new file watcher agent with default settings.
    pub fn new(llm: Arc<dyn LlmProvider>) -> Self {
        Self {
            llm,
            model: DEFAULT_MODEL.into(),
            max_events: DEFAULT_MAX_EVENTS,
            system_prompt: DEFAULT_SYSTEM_PROMPT.into(),
        }
    }

    /// Returns a builder for custom configuration.
    pub fn builder(llm: Arc<dyn LlmProvider>) -> FileWatcherAgentBuilder {
        FileWatcherAgentBuilder {
            llm,
            model: DEFAULT_MODEL.into(),
            max_events: DEFAULT_MAX_EVENTS,
            system_prompt: DEFAULT_SYSTEM_PROMPT.into(),
        }
    }

    /// Runs the agent, processing file events from the bus.
    pub async fn run(&self, bus: &EventBus) -> Result<WatcherOutput, WatcherError> {
        let mut rx = bus.subscribe();
        let mut summaries = Vec::new();

        loop {
            let polled = bus.poll_sources().await.map_err(WatcherError::Bus)?;
            if polled == 0 {
                break;
            }

            // Drain all received events
            while let Ok(event) = rx.try_recv() {
                if event.event_type != FILE_CREATED_EVENT
                    && event.event_type != FILE_MODIFIED_EVENT
                {
                    continue;
                }

                let path = event.data["path"]
                    .as_str()
                    .ok_or_else(|| WatcherError::InvalidEvent("missing 'path' field".into()))?
                    .to_string();

                let file_content = event.data["content"]
                    .as_str()
                    .ok_or_else(|| {
                        WatcherError::InvalidEvent("missing 'content' field".into())
                    })?
                    .to_string();

                let request = CompletionRequest::new(&self.model)
                    .message(Message::system(&self.system_prompt))
                    .message(Message::user(format!(
                        "Summarise the following file ({path}):\n\n{file_content}"
                    )));

                let response = self.llm.complete(request).await.map_err(WatcherError::Llm)?;
                let summary = response.content.unwrap_or_default();

                summaries.push(FileSummary {
                    path,
                    summary,
                    event_type: event.event_type.clone(),
                });

                if summaries.len() >= self.max_events {
                    return Ok(WatcherOutput { summaries });
                }
            }
        }

        Ok(WatcherOutput { summaries })
    }
}

/// Builder for `FileWatcherAgent`.
pub struct FileWatcherAgentBuilder {
    llm: Arc<dyn LlmProvider>,
    model: String,
    max_events: usize,
    system_prompt: String,
}

impl FileWatcherAgentBuilder {
    /// Sets the model name.
    #[must_use]
    pub fn model(mut self, model: &str) -> Self {
        self.model = model.into();
        self
    }

    /// Sets the maximum number of file events to process.
    #[must_use]
    pub fn max_events(mut self, n: usize) -> Self {
        self.max_events = n;
        self
    }

    /// Sets the system prompt for summarisation.
    #[must_use]
    pub fn system_prompt(mut self, prompt: &str) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Builds the agent.
    pub fn build(self) -> FileWatcherAgent {
        FileWatcherAgent {
            llm: self.llm,
            model: self.model,
            max_events: self.max_events,
            system_prompt: self.system_prompt,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use erio_event_bus::{Event, EventSource};
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

    // === Mock EventSource ===

    struct MockEventSource {
        source_name: String,
        events: Mutex<Vec<Event>>,
    }

    impl MockEventSource {
        fn new(name: &str, events: Vec<Event>) -> Self {
            Self {
                source_name: name.into(),
                events: Mutex::new(events),
            }
        }
    }

    #[async_trait::async_trait]
    impl EventSource for MockEventSource {
        fn name(&self) -> &str {
            &self.source_name
        }

        fn description(&self) -> &'static str {
            "Mock file event source"
        }

        async fn next_event(&self) -> Result<Option<Event>, EventBusError> {
            let mut events = self.events.lock().await;
            if events.is_empty() {
                Ok(None)
            } else {
                Ok(Some(events.remove(0)))
            }
        }
    }

    fn file_event(event_type: &str, path: &str, content: &str) -> Event {
        Event::new(
            "watcher",
            event_type,
            serde_json::json!({ "path": path, "content": content }),
        )
    }

    // === Tests ===

    #[tokio::test]
    async fn processes_file_created_event() {
        let llm = Arc::new(MockLlm::new(vec![MockLlm::text("Summary of main.rs")]));
        let agent = FileWatcherAgent::new(llm);

        let mut bus = EventBus::new();
        bus.register(MockEventSource::new(
            "fs",
            vec![file_event(FILE_CREATED_EVENT, "main.rs", "fn main() {}")],
        ));

        let output = agent.run(&bus).await.unwrap();
        assert_eq!(output.summary_count(), 1);
        assert_eq!(output.summaries()[0].path, "main.rs");
        assert_eq!(output.summaries()[0].event_type, FILE_CREATED_EVENT);
    }

    #[tokio::test]
    async fn processes_file_modified_event() {
        let llm = Arc::new(MockLlm::new(vec![MockLlm::text("Updated summary")]));
        let agent = FileWatcherAgent::new(llm);

        let mut bus = EventBus::new();
        bus.register(MockEventSource::new(
            "fs",
            vec![file_event(FILE_MODIFIED_EVENT, "lib.rs", "pub mod foo;")],
        ));

        let output = agent.run(&bus).await.unwrap();
        assert_eq!(output.summary_count(), 1);
        assert_eq!(output.summaries()[0].event_type, FILE_MODIFIED_EVENT);
    }

    #[tokio::test]
    async fn skips_unrecognised_event_types() {
        let llm = Arc::new(MockLlm::new(vec![]));
        let agent = FileWatcherAgent::new(llm);

        let mut bus = EventBus::new();
        bus.register(MockEventSource::new(
            "fs",
            vec![Event::new("fs", "file.deleted", serde_json::json!({}))],
        ));

        let output = agent.run(&bus).await.unwrap();
        assert_eq!(output.summary_count(), 0);
    }

    #[tokio::test]
    async fn processes_multiple_file_events() {
        let llm = Arc::new(MockLlm::new(vec![
            MockLlm::text("Summary 1"),
            MockLlm::text("Summary 2"),
        ]));
        let agent = FileWatcherAgent::new(llm);

        let mut bus = EventBus::new();
        bus.register(MockEventSource::new(
            "fs-a",
            vec![file_event(FILE_CREATED_EVENT, "a.rs", "// a")],
        ));
        bus.register(MockEventSource::new(
            "fs-b",
            vec![file_event(FILE_CREATED_EVENT, "b.rs", "// b")],
        ));

        let output = agent.run(&bus).await.unwrap();
        assert_eq!(output.summary_count(), 2);
    }

    #[tokio::test]
    async fn summary_count_matches_event_count() {
        let llm = Arc::new(MockLlm::new(vec![
            MockLlm::text("s1"),
            MockLlm::text("s2"),
            MockLlm::text("s3"),
        ]));
        let agent = FileWatcherAgent::new(llm);

        let mut bus = EventBus::new();
        bus.register(MockEventSource::new(
            "fs-1",
            vec![file_event(FILE_CREATED_EVENT, "1.rs", "1")],
        ));
        bus.register(MockEventSource::new(
            "fs-2",
            vec![file_event(FILE_MODIFIED_EVENT, "2.rs", "2")],
        ));
        bus.register(MockEventSource::new(
            "fs-3",
            vec![file_event(FILE_CREATED_EVENT, "3.rs", "3")],
        ));

        let output = agent.run(&bus).await.unwrap();
        assert_eq!(output.summary_count(), 3);
    }

    #[tokio::test]
    async fn find_by_path_returns_correct_summary() {
        let llm = Arc::new(MockLlm::new(vec![MockLlm::text("Found it")]));
        let agent = FileWatcherAgent::new(llm);

        let mut bus = EventBus::new();
        bus.register(MockEventSource::new(
            "fs",
            vec![file_event(FILE_CREATED_EVENT, "target.rs", "code")],
        ));

        let output = agent.run(&bus).await.unwrap();
        let found = output.find_by_path("target.rs").unwrap();
        assert_eq!(found.summary, "Found it");
    }

    #[tokio::test]
    async fn find_by_path_returns_none_for_unknown() {
        let llm = Arc::new(MockLlm::new(vec![MockLlm::text("ok")]));
        let agent = FileWatcherAgent::new(llm);

        let mut bus = EventBus::new();
        bus.register(MockEventSource::new(
            "fs",
            vec![file_event(FILE_CREATED_EVENT, "known.rs", "code")],
        ));

        let output = agent.run(&bus).await.unwrap();
        assert!(output.find_by_path("unknown.rs").is_none());
    }

    #[tokio::test]
    async fn invalid_event_missing_path_returns_error() {
        let llm = Arc::new(MockLlm::new(vec![]));
        let agent = FileWatcherAgent::new(llm);

        let mut bus = EventBus::new();
        bus.register(MockEventSource::new(
            "fs",
            vec![Event::new(
                "fs",
                FILE_CREATED_EVENT,
                serde_json::json!({ "content": "hello" }),
            )],
        ));

        let result = agent.run(&bus).await;
        assert!(matches!(result, Err(WatcherError::InvalidEvent(msg)) if msg.contains("path")));
    }

    #[tokio::test]
    async fn invalid_event_missing_content_returns_error() {
        let llm = Arc::new(MockLlm::new(vec![]));
        let agent = FileWatcherAgent::new(llm);

        let mut bus = EventBus::new();
        bus.register(MockEventSource::new(
            "fs",
            vec![Event::new(
                "fs",
                FILE_CREATED_EVENT,
                serde_json::json!({ "path": "test.rs" }),
            )],
        ));

        let result = agent.run(&bus).await;
        assert!(
            matches!(result, Err(WatcherError::InvalidEvent(msg)) if msg.contains("content"))
        );
    }

    #[tokio::test]
    async fn llm_error_propagates() {
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
        let agent = FileWatcherAgent::new(llm);

        let mut bus = EventBus::new();
        bus.register(MockEventSource::new(
            "fs",
            vec![file_event(FILE_CREATED_EVENT, "test.rs", "code")],
        ));

        let result = agent.run(&bus).await;
        assert!(matches!(result, Err(WatcherError::Llm(_))));
    }

    #[tokio::test]
    async fn max_events_limits_processing() {
        let llm = Arc::new(MockLlm::new(vec![
            MockLlm::text("s1"),
            MockLlm::text("s2"),
            MockLlm::text("s3"),
        ]));
        let agent = FileWatcherAgent::builder(llm).max_events(2).build();

        let mut bus = EventBus::new();
        bus.register(MockEventSource::new(
            "fs-1",
            vec![file_event(FILE_CREATED_EVENT, "1.rs", "1")],
        ));
        bus.register(MockEventSource::new(
            "fs-2",
            vec![file_event(FILE_CREATED_EVENT, "2.rs", "2")],
        ));
        bus.register(MockEventSource::new(
            "fs-3",
            vec![file_event(FILE_CREATED_EVENT, "3.rs", "3")],
        ));

        let output = agent.run(&bus).await.unwrap();
        assert_eq!(output.summary_count(), 2);
    }

    #[tokio::test]
    async fn builder_sets_custom_system_prompt() {
        let llm = Arc::new(MockLlm::new(vec![]));
        let agent = FileWatcherAgent::builder(llm)
            .system_prompt("Custom prompt")
            .build();

        assert_eq!(agent.system_prompt, "Custom prompt");
    }

    #[tokio::test]
    async fn builder_sets_custom_model() {
        let llm = Arc::new(MockLlm::new(vec![]));
        let agent = FileWatcherAgent::builder(llm)
            .model("gpt-3.5-turbo")
            .build();

        assert_eq!(agent.model, "gpt-3.5-turbo");
    }

    #[tokio::test]
    async fn empty_source_produces_empty_output() {
        let llm = Arc::new(MockLlm::new(vec![]));
        let agent = FileWatcherAgent::new(llm);

        let mut bus = EventBus::new();
        bus.register(MockEventSource::new("fs", vec![]));

        let output = agent.run(&bus).await.unwrap();
        assert_eq!(output.summary_count(), 0);
    }
}
