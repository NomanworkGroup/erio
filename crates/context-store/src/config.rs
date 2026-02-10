//! Configuration for the context store.

use std::path::PathBuf;

/// HNSW index parameters.
pub struct HnswConfig {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 100,
            ef_search: 50,
        }
    }
}

/// Configuration for creating a [`crate::store::ContextStore`].
pub struct ContextConfig {
    pub path: PathBuf,
    pub index: HnswConfig,
}

/// A single result from a semantic search.
pub struct SearchResult {
    pub content: String,
    pub score: f32,
    pub metadata: serde_json::Value,
}

/// Statistics about the context store.
pub struct StorageStats {
    pub document_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hnsw_config_has_sensible_defaults() {
        let config = HnswConfig::default();
        assert_eq!(config.m, 16);
        assert_eq!(config.ef_construction, 100);
        assert_eq!(config.ef_search, 50);
    }

    #[test]
    fn context_config_stores_path() {
        let config = ContextConfig {
            path: PathBuf::from("/tmp/store"),
            index: HnswConfig::default(),
        };
        assert_eq!(config.path, PathBuf::from("/tmp/store"));
    }

    #[test]
    fn context_config_uses_default_hnsw() {
        let config = ContextConfig {
            path: PathBuf::from("/tmp/test"),
            index: HnswConfig::default(),
        };
        assert_eq!(config.index.m, 16);
    }

    #[test]
    fn search_result_holds_content_and_score() {
        let result = SearchResult {
            content: "hello world".into(),
            score: 0.95,
            metadata: serde_json::json!({"source": "test"}),
        };
        assert_eq!(result.content, "hello world");
        assert!((result.score - 0.95).abs() < f32::EPSILON);
        assert_eq!(result.metadata["source"], "test");
    }

    #[test]
    fn storage_stats_reports_document_count() {
        let stats = StorageStats { document_count: 42 };
        assert_eq!(stats.document_count, 42);
    }
}
