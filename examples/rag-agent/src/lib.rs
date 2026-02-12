//! Retrieval-augmented generation agent using `ContextStore` + Embedding + LLM.

use std::fmt;
use std::sync::Arc;

use erio_context_store::{ContextStore, ContextStoreError};
use erio_core::Message;
use erio_llm_client::{CompletionRequest, LlmError, LlmProvider};

const DEFAULT_MODEL: &str = "gpt-4";
const DEFAULT_TOP_K: usize = 5;
const DEFAULT_SYSTEM_PROMPT: &str =
    "You are a helpful assistant. Answer questions using the provided context.";

/// Errors from the RAG agent.
#[derive(Debug)]
pub enum RagError {
    /// Context store operation failed.
    Store(ContextStoreError),
    /// LLM call failed.
    Llm(LlmError),
    /// Query was empty.
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

/// A document to ingest into the RAG agent.
pub struct Document {
    pub content: String,
    pub metadata: serde_json::Value,
}

impl Document {
    /// Creates a document with no metadata.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            metadata: serde_json::json!({}),
        }
    }

    /// Creates a document with metadata.
    pub fn with_metadata(content: impl Into<String>, metadata: serde_json::Value) -> Self {
        Self {
            content: content.into(),
            metadata,
        }
    }
}

/// The answer from a RAG query.
#[derive(Debug)]
pub struct RagAnswer {
    answer: String,
    sources: Vec<String>,
    retrieved_count: usize,
}

impl RagAnswer {
    /// The generated answer.
    pub fn answer(&self) -> &str {
        &self.answer
    }

    /// The source documents used for context.
    pub fn sources(&self) -> &[String] {
        &self.sources
    }

    /// Number of documents retrieved.
    pub fn retrieved_count(&self) -> usize {
        self.retrieved_count
    }
}

/// A retrieval-augmented generation agent.
pub struct RagAgent {
    llm: Arc<dyn LlmProvider>,
    store: ContextStore,
    model: String,
    top_k: usize,
    system_prompt: String,
}

impl RagAgent {
    /// Creates a new RAG agent with default settings.
    pub fn new(llm: Arc<dyn LlmProvider>, store: ContextStore) -> Self {
        Self {
            llm,
            store,
            model: DEFAULT_MODEL.into(),
            top_k: DEFAULT_TOP_K,
            system_prompt: DEFAULT_SYSTEM_PROMPT.into(),
        }
    }

    /// Returns a builder for custom configuration.
    pub fn builder(llm: Arc<dyn LlmProvider>, store: ContextStore) -> RagAgentBuilder {
        RagAgentBuilder {
            llm,
            store,
            model: DEFAULT_MODEL.into(),
            top_k: DEFAULT_TOP_K,
            system_prompt: DEFAULT_SYSTEM_PROMPT.into(),
        }
    }

    /// Ingests documents into the context store.
    pub async fn ingest(&self, docs: Vec<Document>) -> Result<Vec<String>, RagError> {
        let mut ids = Vec::with_capacity(docs.len());
        for doc in docs {
            let id = self
                .store
                .add(&doc.content, doc.metadata)
                .await
                .map_err(RagError::Store)?;
            ids.push(id);
        }
        Ok(ids)
    }

    /// Queries the RAG agent with a question.
    pub async fn query(&self, question: &str) -> Result<RagAnswer, RagError> {
        if question.is_empty() {
            return Err(RagError::EmptyQuery);
        }

        let results = self
            .store
            .search(question, self.top_k, None)
            .await
            .map_err(RagError::Store)?;

        let sources: Vec<String> = results.iter().map(|r| r.content.clone()).collect();
        let retrieved_count = sources.len();

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

        let request = CompletionRequest::new(&self.model)
            .message(Message::system(&self.system_prompt))
            .message(Message::user(format!(
                "Context:\n{context}\n\nQuestion: {question}"
            )));

        let response = self.llm.complete(request).await.map_err(RagError::Llm)?;
        let answer = response.content.unwrap_or_default();

        Ok(RagAnswer {
            answer,
            sources,
            retrieved_count,
        })
    }
}

/// Builder for `RagAgent`.
pub struct RagAgentBuilder {
    llm: Arc<dyn LlmProvider>,
    store: ContextStore,
    model: String,
    top_k: usize,
    system_prompt: String,
}

impl RagAgentBuilder {
    /// Sets the model name.
    #[must_use]
    pub fn model(mut self, model: &str) -> Self {
        self.model = model.into();
        self
    }

    /// Sets the number of documents to retrieve.
    #[must_use]
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Sets the system prompt.
    #[must_use]
    pub fn system_prompt(mut self, prompt: &str) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Builds the RAG agent.
    pub fn build(self) -> RagAgent {
        RagAgent {
            llm: self.llm,
            store: self.store,
            model: self.model,
            top_k: self.top_k,
            system_prompt: self.system_prompt,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use erio_context_store::{ContextConfig, HnswConfig};
    use erio_embedding::EmbeddingError;
    use erio_llm_client::CompletionResponse;
    use tokio::sync::Mutex;

    // === Fake Embedding ===

    struct FakeEmbedding;

    #[async_trait::async_trait]
    impl erio_embedding::EmbeddingEngine for FakeEmbedding {
        fn name(&self) -> &'static str {
            "fake"
        }

        fn dimensions(&self) -> usize {
            3
        }

        async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
            let bytes = text.as_bytes();
            Ok(vec![
                f32::from(*bytes.first().unwrap_or(&0)),
                f32::from(*bytes.get(1).unwrap_or(&0)),
                f32::from(*bytes.get(2).unwrap_or(&0)),
            ])
        }

        async fn embed_batch(
            &self,
            texts: &[&str],
        ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            let mut results = Vec::with_capacity(texts.len());
            for text in texts {
                results.push(self.embed(text).await?);
            }
            Ok(results)
        }
    }

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

    // === Helpers ===

    fn test_config(dir: &tempfile::TempDir) -> ContextConfig {
        ContextConfig {
            path: dir.path().to_path_buf(),
            index: HnswConfig::default(),
        }
    }

    async fn make_agent(
        dir: &tempfile::TempDir,
        responses: Vec<CompletionResponse>,
    ) -> RagAgent {
        let store = ContextStore::new(test_config(dir), Arc::new(FakeEmbedding))
            .await
            .unwrap();
        let llm = Arc::new(MockLlm::new(responses));
        RagAgent::new(llm, store)
    }

    async fn make_builder_agent(
        dir: &tempfile::TempDir,
        responses: Vec<CompletionResponse>,
    ) -> RagAgentBuilder {
        let store = ContextStore::new(test_config(dir), Arc::new(FakeEmbedding))
            .await
            .unwrap();
        let llm = Arc::new(MockLlm::new(responses));
        RagAgent::builder(llm, store)
    }

    // === Tests ===

    #[tokio::test]
    async fn ingest_single_document_returns_id() {
        let dir = tempfile::tempdir().unwrap();
        let agent = make_agent(&dir, vec![]).await;

        let ids = agent
            .ingest(vec![Document::new("hello world")])
            .await
            .unwrap();

        assert_eq!(ids.len(), 1);
        assert!(!ids[0].is_empty());
    }

    #[tokio::test]
    async fn ingest_multiple_documents_returns_all_ids() {
        let dir = tempfile::tempdir().unwrap();
        let agent = make_agent(&dir, vec![]).await;

        let ids = agent
            .ingest(vec![
                Document::new("first doc"),
                Document::new("second doc"),
                Document::new("third doc"),
            ])
            .await
            .unwrap();

        assert_eq!(ids.len(), 3);
    }

    #[tokio::test]
    async fn ingest_empty_document_returns_store_error() {
        let dir = tempfile::tempdir().unwrap();
        let agent = make_agent(&dir, vec![]).await;

        let result = agent.ingest(vec![Document::new("")]).await;
        assert!(matches!(result, Err(RagError::Store(_))));
    }

    #[tokio::test]
    async fn query_empty_string_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let agent = make_agent(&dir, vec![]).await;

        let result = agent.query("").await;
        assert!(matches!(result, Err(RagError::EmptyQuery)));
    }

    #[tokio::test]
    async fn query_on_empty_store_returns_llm_answer() {
        let dir = tempfile::tempdir().unwrap();
        let agent = make_agent(&dir, vec![MockLlm::text("No docs found")]).await;

        let answer = agent.query("anything").await.unwrap();
        assert_eq!(answer.answer(), "No docs found");
        assert_eq!(answer.retrieved_count(), 0);
    }

    #[tokio::test]
    async fn query_retrieves_relevant_document() {
        let dir = tempfile::tempdir().unwrap();
        let agent = make_agent(&dir, vec![MockLlm::text("Answer based on context")]).await;

        agent
            .ingest(vec![Document::new("hello world")])
            .await
            .unwrap();

        let answer = agent.query("hello").await.unwrap();
        assert!(!answer.sources().is_empty());
    }

    #[tokio::test]
    async fn query_respects_top_k() {
        let dir = tempfile::tempdir().unwrap();
        let store = ContextStore::new(test_config(&dir), Arc::new(FakeEmbedding))
            .await
            .unwrap();
        let llm = Arc::new(MockLlm::new(vec![MockLlm::text("ok")]));
        let agent = RagAgent::builder(llm, store).top_k(1).build();

        agent
            .ingest(vec![
                Document::new("aaa first"),
                Document::new("aab second"),
                Document::new("aac third"),
            ])
            .await
            .unwrap();

        let answer = agent.query("aaa").await.unwrap();
        assert_eq!(answer.retrieved_count(), 1);
    }

    #[tokio::test]
    async fn answer_contains_llm_response() {
        let dir = tempfile::tempdir().unwrap();
        let agent = make_agent(&dir, vec![MockLlm::text("The answer is 42")]).await;

        agent
            .ingest(vec![Document::new("some content")])
            .await
            .unwrap();

        let answer = agent.query("what is the answer").await.unwrap();
        assert_eq!(answer.answer(), "The answer is 42");
    }

    #[tokio::test]
    async fn sources_contains_retrieved_content() {
        let dir = tempfile::tempdir().unwrap();
        let agent = make_agent(&dir, vec![MockLlm::text("ok")]).await;

        agent
            .ingest(vec![Document::new("hello world")])
            .await
            .unwrap();

        let answer = agent.query("hello").await.unwrap();
        assert!(answer.sources().iter().any(|s| s.contains("hello world")));
    }

    #[tokio::test]
    async fn llm_error_propagates_from_query() {
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

        let dir = tempfile::tempdir().unwrap();
        let store = ContextStore::new(test_config(&dir), Arc::new(FakeEmbedding))
            .await
            .unwrap();
        let llm: Arc<dyn LlmProvider> = Arc::new(FailLlm);
        let agent = RagAgent::new(llm, store);

        let result = agent.query("question").await;
        assert!(matches!(result, Err(RagError::Llm(_))));
    }

    #[tokio::test]
    async fn ingest_then_query_full_pipeline() {
        let dir = tempfile::tempdir().unwrap();
        let agent = make_agent(
            &dir,
            vec![MockLlm::text("Rust is a systems programming language")],
        )
        .await;

        let ids = agent
            .ingest(vec![
                Document::new("Rust is fast and safe"),
                Document::new("Python is great for scripting"),
            ])
            .await
            .unwrap();

        assert_eq!(ids.len(), 2);

        let answer = agent.query("Tell me about Rust").await.unwrap();
        assert_eq!(answer.answer(), "Rust is a systems programming language");
        assert!(!answer.sources().is_empty());
    }

    #[tokio::test]
    async fn metadata_is_preserved_in_ingested_documents() {
        let dir = tempfile::tempdir().unwrap();
        let agent = make_agent(&dir, vec![MockLlm::text("ok")]).await;

        let meta = serde_json::json!({"source": "docs", "chapter": 3});
        agent
            .ingest(vec![Document::with_metadata("content with meta", meta)])
            .await
            .unwrap();

        // Verify the document was stored (search returns it)
        let answer = agent.query("content").await.unwrap();
        assert_eq!(answer.retrieved_count(), 1);
    }

    #[tokio::test]
    async fn builder_sets_top_k() {
        let dir = tempfile::tempdir().unwrap();
        let agent = make_builder_agent(&dir, vec![]).await.top_k(10).build();
        assert_eq!(agent.top_k, 10);
    }

    #[tokio::test]
    async fn builder_sets_system_prompt() {
        let dir = tempfile::tempdir().unwrap();
        let agent = make_builder_agent(&dir, vec![])
            .await
            .system_prompt("Custom RAG prompt")
            .build();
        assert_eq!(agent.system_prompt, "Custom RAG prompt");
    }
}
