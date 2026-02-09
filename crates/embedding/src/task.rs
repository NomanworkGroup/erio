//! Task type definitions and prompt formatting for embedding models.

/// The type of embedding task, used to format input prompts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TaskType {
    /// Search result retrieval.
    #[default]
    SearchResult,
    /// Search query.
    SearchQuery,
    /// Text classification.
    Classification,
    /// Text clustering.
    Clustering,
    /// Semantic similarity.
    SemanticSimilarity,
    /// Fact verification.
    FactVerification,
    /// Code retrieval.
    CodeRetrieval,
}

impl TaskType {
    /// Returns the human-readable description used in prompt formatting.
    pub fn description(self) -> &'static str {
        match self {
            Self::SearchResult => "search result",
            Self::SearchQuery => "search query",
            Self::Classification => "classification",
            Self::Clustering => "clustering",
            Self::SemanticSimilarity => "semantic similarity",
            Self::FactVerification => "fact verification",
            Self::CodeRetrieval => "code retrieval",
        }
    }
}

/// Formats a query text with a task-type prefix for embedding.
pub fn format_query(text: &str, task_type: TaskType) -> String {
    format!("task: {} | query: {text}", task_type.description())
}

/// Formats a document text with an optional title for embedding.
pub fn format_document(text: &str, title: Option<&str>) -> String {
    let title = title.unwrap_or("none");
    format!("title: {title} | text: {text}")
}

#[cfg(test)]
mod tests {
    use super::*;

    // === TaskType::description() tests ===

    #[test]
    fn search_result_description() {
        assert_eq!(TaskType::SearchResult.description(), "search result");
    }

    #[test]
    fn search_query_description() {
        assert_eq!(TaskType::SearchQuery.description(), "search query");
    }

    #[test]
    fn classification_description() {
        assert_eq!(TaskType::Classification.description(), "classification");
    }

    #[test]
    fn clustering_description() {
        assert_eq!(TaskType::Clustering.description(), "clustering");
    }

    #[test]
    fn semantic_similarity_description() {
        assert_eq!(
            TaskType::SemanticSimilarity.description(),
            "semantic similarity"
        );
    }

    #[test]
    fn fact_verification_description() {
        assert_eq!(TaskType::FactVerification.description(), "fact verification");
    }

    #[test]
    fn code_retrieval_description() {
        assert_eq!(TaskType::CodeRetrieval.description(), "code retrieval");
    }

    // === Default ===

    #[test]
    fn default_is_search_result() {
        assert_eq!(TaskType::default(), TaskType::SearchResult);
    }

    // === format_query() tests ===

    #[test]
    fn format_query_with_search_result() {
        let result = format_query("what is rust", TaskType::SearchResult);
        assert_eq!(result, "task: search result | query: what is rust");
    }

    #[test]
    fn format_query_with_classification() {
        let result = format_query("hello world", TaskType::Classification);
        assert_eq!(result, "task: classification | query: hello world");
    }

    // === format_document() tests ===

    #[test]
    fn format_document_without_title() {
        let result = format_document("some document text", None);
        assert_eq!(result, "title: none | text: some document text");
    }

    #[test]
    fn format_document_with_title() {
        let result = format_document("some document text", Some("My Title"));
        assert_eq!(result, "title: My Title | text: some document text");
    }
}
