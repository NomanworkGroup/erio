//! Retrieval-augmented generation agent using `ContextStore` + Embedding + LLM.
//!
//! Ingests text files into a vector store, then answers a query with retrieved context.
//!
//! Usage:
//!   rag-agent --query "What is Rust?" --documents docs/intro.txt,docs/faq.txt
//!   `OPENAI_API_KEY=sk`-... rag-agent -q "How does async work?" -d chapter1.txt

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use erio_context_store::{ContextConfig, ContextStore, ContextStoreError, HnswConfig};
use erio_core::Message;
use erio_embedding::{EmbeddingConfig, EmbeddingEngine, RemoteEmbedding};
use erio_llm_client::{CompletionRequest, LlmError, LlmProvider, OpenAiProvider};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// Retrieval-augmented generation agent.
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// The query to answer.
    #[arg(short, long)]
    query: String,

    /// Comma-separated document file paths to ingest.
    #[arg(short, long, value_delimiter = ',')]
    documents: Vec<String>,

    /// OpenAI-compatible API base URL.
    #[arg(long, env = "OPENAI_BASE_URL", default_value = "https://api.openai.com/v1")]
    base_url: String,

    /// API key (reads from `OPENAI_API_KEY` env var by default).
    #[arg(long, env = "OPENAI_API_KEY", default_value = "")]
    api_key: String,

    /// Model name to use for generation.
    #[arg(short, long, default_value = "gpt-4")]
    model: String,

    /// Embedding API base URL (defaults to --base-url value).
    #[arg(long, env = "EMBEDDING_BASE_URL")]
    embedding_url: Option<String>,

    /// Number of documents to retrieve.
    #[arg(long, default_value_t = 5)]
    top_k: usize,

    /// Directory to store the vector index.
    #[arg(long, default_value = "/tmp/rag-agent-store")]
    store_path: PathBuf,
}

// ---------------------------------------------------------------------------
// Agent types
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum RagError {
    Store(ContextStoreError),
    Llm(LlmError),
    EmptyQuery,
}

impl fmt::Display for RagError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(e) => write!(f, "store error: {e}"),
            Self::Llm(e) => write!(f, "LLM error: {e}"),
            Self::EmptyQuery => write!(f, "query must not be empty"),
        }
    }
}

impl std::error::Error for RagError {}

// ---------------------------------------------------------------------------
// RAG pipeline
// ---------------------------------------------------------------------------

async fn run_rag(
    llm: Arc<dyn LlmProvider>,
    store: &ContextStore,
    model: &str,
    query: &str,
    top_k: usize,
) -> Result<String, RagError> {
    if query.is_empty() {
        return Err(RagError::EmptyQuery);
    }

    let results = store
        .search(query, top_k, None)
        .await
        .map_err(RagError::Store)?;

    let sources: Vec<String> = results.iter().map(|r| r.content.clone()).collect();

    println!("Retrieved {} documents", sources.len());

    let context = if sources.is_empty() {
        "No relevant documents found.".to_string()
    } else {
        sources
            .iter()
            .enumerate()
            .map(|(i, s)| format!("[{}] {s}", i + 1))
            .collect::<Vec<_>>()
            .join("\n\n")
    };

    let request = CompletionRequest::new(model)
        .message(Message::system(
            "You are a helpful assistant. Answer questions using the provided context.",
        ))
        .message(Message::user(format!(
            "Context:\n{context}\n\nQuestion: {query}"
        )));

    let response = llm.complete(request).await.map_err(RagError::Llm)?;
    Ok(response.content.unwrap_or_default())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let llm: Arc<dyn LlmProvider> = Arc::new(OpenAiProvider::new(&cli.base_url, &cli.api_key));

    let embedding_url = cli.embedding_url.as_deref().unwrap_or(&cli.base_url);
    let embedding: Arc<dyn EmbeddingEngine> = Arc::new(RemoteEmbedding::new(
        embedding_url,
        &cli.api_key,
        EmbeddingConfig::default(),
    ));

    let config = ContextConfig {
        path: cli.store_path,
        index: HnswConfig::default(),
    };

    let store = match ContextStore::new(config, embedding).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: failed to create context store: {e}");
            std::process::exit(1);
        }
    };

    // Ingest documents
    for path in &cli.documents {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Warning: cannot read {path}: {e}");
                continue;
            }
        };
        println!("Ingesting {path}...");
        if let Err(e) = store.add(&content, serde_json::json!({"path": path})).await {
            eprintln!("Warning: failed to ingest {path}: {e}");
        }
    }

    match run_rag(llm, &store, &cli.model, &cli.query, cli.top_k).await {
        Ok(answer) => println!("\n{answer}"),
        Err(e) => eprintln!("Error: {e}"),
    }
}
