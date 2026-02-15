//! Event-driven file summariser using `EventBus` + LLM.
//!
//! Reads files from the provided paths and produces LLM-powered summaries.
//!
//! Usage:
//!   file-watcher-agent --files src/main.rs,src/lib.rs
//!   `OPENAI_API_KEY=sk`-... file-watcher-agent -f README.md

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

/// Event-driven file summariser powered by LLM.
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Comma-separated file paths to summarise.
    #[arg(short, long, value_delimiter = ',')]
    files: Vec<String>,

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

    /// Maximum number of files to process.
    #[arg(long, default_value_t = 100)]
    max_files: usize,
}

// ---------------------------------------------------------------------------
// Agent types
// ---------------------------------------------------------------------------

const FILE_CREATED_EVENT: &str = "file.created";

#[derive(Debug)]
enum WatcherError {
    Bus(EventBusError),
    Llm(LlmError),
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

// ---------------------------------------------------------------------------
// File event source
// ---------------------------------------------------------------------------

struct FileEventSource {
    source_name: String,
    events: Mutex<Vec<Event>>,
}

impl FileEventSource {
    fn new(name: &str, events: Vec<Event>) -> Self {
        Self {
            source_name: name.into(),
            events: Mutex::new(events),
        }
    }
}

#[async_trait::async_trait]
impl EventSource for FileEventSource {
    fn name(&self) -> &str {
        &self.source_name
    }

    fn description(&self) -> &'static str {
        "File event source"
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

// ---------------------------------------------------------------------------
// Summariser
// ---------------------------------------------------------------------------

async fn summarise_files(
    llm: Arc<dyn LlmProvider>,
    model: &str,
    bus: &EventBus,
    max_files: usize,
) -> Result<(), WatcherError> {
    let mut rx = bus.subscribe();
    let system_prompt =
        "You are a file summariser. Given a file's content, produce a concise summary.";
    let mut count = 0;

    loop {
        let polled = bus.poll_sources().await.map_err(WatcherError::Bus)?;
        if polled == 0 {
            break;
        }

        while let Ok(event) = rx.try_recv() {
            if event.event_type != FILE_CREATED_EVENT {
                continue;
            }

            let path = event.data["path"]
                .as_str()
                .ok_or_else(|| WatcherError::InvalidEvent("missing 'path' field".into()))?;

            let file_content = event.data["content"]
                .as_str()
                .ok_or_else(|| WatcherError::InvalidEvent("missing 'content' field".into()))?;

            let request = CompletionRequest::new(model)
                .message(Message::system(system_prompt))
                .message(Message::user(format!(
                    "Summarise the following file ({path}):\n\n{file_content}"
                )));

            let response = llm.complete(request).await.map_err(WatcherError::Llm)?;
            let summary = response.content.unwrap_or_default();

            println!("--- {path} ---\n{summary}\n");

            count += 1;
            if count >= max_files {
                return Ok(());
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    if cli.files.is_empty() {
        eprintln!("Error: provide at least one file with --files");
        std::process::exit(1);
    }

    let llm: Arc<dyn LlmProvider> = Arc::new(OpenAiProvider::new(&cli.base_url, &cli.api_key));

    // Read files and create events
    let mut events = Vec::new();
    for path in &cli.files {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Warning: cannot read {path}: {e}");
                continue;
            }
        };
        events.push(Event::new(
            "fs",
            FILE_CREATED_EVENT,
            serde_json::json!({ "path": path, "content": content }),
        ));
    }

    let mut bus = EventBus::new();
    bus.register(FileEventSource::new("fs", events));

    if let Err(e) = summarise_files(llm, &cli.model, &bus, cli.max_files).await {
        eprintln!("Error: {e}");
    }
}
