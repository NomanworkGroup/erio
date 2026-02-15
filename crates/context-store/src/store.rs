//! Context store backed by `LanceDB`.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::types::Float32Type;
use arrow_array::{FixedSizeListArray, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};

use crate::config::{ContextConfig, SearchResult, StorageStats};
use crate::error::ContextStoreError;

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

fn generate_id() -> String {
    let count = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{nanos:x}-{count:x}")
}

pub struct ContextStore {
    table: lancedb::Table,
    embedding: Arc<dyn erio_embedding::EmbeddingEngine>,
}

impl ContextStore {
    pub async fn new(
        config: ContextConfig,
        embedding: Arc<dyn erio_embedding::EmbeddingEngine>,
    ) -> Result<Self, ContextStoreError> {
        let dims = embedding.dimensions();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("metadata", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    i32::try_from(dims).map_err(|e| {
                        ContextStoreError::InvalidInput(format!("bad dimensions: {e}"))
                    })?,
                ),
                false,
            ),
        ]));

        let db = lancedb::connect(config.path.to_string_lossy().as_ref())
            .execute()
            .await
            .map_err(|e| ContextStoreError::Storage(e.to_string()))?;

        let table = db
            .create_empty_table("context", schema)
            .execute()
            .await
            .map_err(|e| ContextStoreError::Storage(e.to_string()))?;

        Ok(Self { table, embedding })
    }

    pub async fn add(
        &self,
        content: &str,
        metadata: serde_json::Value,
    ) -> Result<String, ContextStoreError> {
        if content.is_empty() {
            return Err(ContextStoreError::InvalidInput(
                "content must not be empty".into(),
            ));
        }

        let id = generate_id();
        let vector = self
            .embedding
            .embed(content)
            .await
            .map_err(|e| ContextStoreError::Embedding(e.to_string()))?;

        let dims = i32::try_from(self.embedding.dimensions())
            .map_err(|e| ContextStoreError::InvalidInput(format!("bad dimensions: {e}")))?;

        let schema = self
            .table
            .schema()
            .await
            .map_err(|e| ContextStoreError::Storage(e.to_string()))?;

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![id.as_str()])),
                Arc::new(StringArray::from(vec![content])),
                Arc::new(StringArray::from(vec![metadata.to_string().as_str()])),
                Arc::new(
                    FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                        vec![Some(vector.into_iter().map(Some).collect::<Vec<_>>())],
                        dims,
                    ),
                ),
            ],
        )
        .map_err(|e| ContextStoreError::Storage(e.to_string()))?;

        let batches = RecordBatchIterator::new(
            vec![Ok(batch)],
            self.table
                .schema()
                .await
                .map_err(|e| ContextStoreError::Storage(e.to_string()))?,
        );

        self.table
            .add(batches)
            .execute()
            .await
            .map_err(|e| ContextStoreError::Storage(e.to_string()))?;

        Ok(id)
    }

    pub async fn search(
        &self,
        query: &str,
        k: usize,
        filter: Option<String>,
    ) -> Result<Vec<SearchResult>, ContextStoreError> {
        let count = self
            .table
            .count_rows(None)
            .await
            .map_err(|e| ContextStoreError::Storage(e.to_string()))?;
        if count == 0 {
            return Ok(vec![]);
        }

        let vector = self
            .embedding
            .embed(query)
            .await
            .map_err(|e| ContextStoreError::Embedding(e.to_string()))?;

        let mut builder = self
            .table
            .query()
            .nearest_to(vector)
            .map_err(|e| ContextStoreError::Storage(e.to_string()))?
            .limit(k);

        if let Some(f) = filter {
            builder = builder.only_if(f);
        }

        let batches: Vec<RecordBatch> = builder
            .execute()
            .await
            .map_err(|e| ContextStoreError::Storage(e.to_string()))?
            .try_collect()
            .await
            .map_err(|e| ContextStoreError::Storage(e.to_string()))?;

        let mut results = Vec::new();
        for batch in &batches {
            let content_col = batch
                .column_by_name("content")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| ContextStoreError::Storage("missing content column".into()))?;

            let metadata_col = batch
                .column_by_name("metadata")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| ContextStoreError::Storage("missing metadata column".into()))?;

            let distance_col = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::Float32Array>())
                .ok_or_else(|| ContextStoreError::Storage("missing _distance column".into()))?;

            for i in 0..batch.num_rows() {
                let meta_str = metadata_col.value(i);
                let metadata: serde_json::Value =
                    serde_json::from_str(meta_str).unwrap_or_default();

                results.push(SearchResult {
                    content: content_col.value(i).to_string(),
                    score: 1.0 - distance_col.value(i),
                    metadata,
                });
            }
        }

        Ok(results)
    }

    pub async fn stats(&self) -> Result<StorageStats, ContextStoreError> {
        let count = self
            .table
            .count_rows(None)
            .await
            .map_err(|e| ContextStoreError::Storage(e.to_string()))?;
        Ok(StorageStats {
            document_count: count,
        })
    }

    pub async fn delete(&self, id: &str) -> Result<(), ContextStoreError> {
        let escaped = id.replace('\'', "''");
        let filter = format!("id = '{escaped}'");

        let count_before = self
            .table
            .count_rows(Some(filter.clone()))
            .await
            .map_err(|e| ContextStoreError::Storage(e.to_string()))?;

        if count_before == 0 {
            return Err(ContextStoreError::NotFound(id.to_string()));
        }

        self.table
            .delete(&filter)
            .await
            .map_err(|e| ContextStoreError::Storage(e.to_string()))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ContextConfig;
    use crate::config::HnswConfig;
    use erio_embedding::EmbeddingError;
    use std::sync::Arc;

    struct FakeEmbedding;

    #[async_trait::async_trait]
    impl erio_embedding::EmbeddingEngine for FakeEmbedding {
        fn name(&self) -> &str {
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

        async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            let mut results = Vec::with_capacity(texts.len());
            for text in texts {
                results.push(self.embed(text).await?);
            }
            Ok(results)
        }
    }

    fn test_config(dir: &tempfile::TempDir) -> ContextConfig {
        ContextConfig {
            path: dir.path().to_path_buf(),
            index: HnswConfig::default(),
        }
    }

    fn fake_embedding() -> Arc<dyn erio_embedding::EmbeddingEngine> {
        Arc::new(FakeEmbedding)
    }

    // === Construction ===

    #[tokio::test]
    async fn new_creates_empty_store() {
        let dir = tempfile::tempdir().unwrap();
        let store = ContextStore::new(test_config(&dir), fake_embedding())
            .await
            .unwrap();
        let stats = store.stats().await.unwrap();
        assert_eq!(stats.document_count, 0);
    }

    // === Add ===

    #[tokio::test]
    async fn add_returns_document_id() {
        let dir = tempfile::tempdir().unwrap();
        let store = ContextStore::new(test_config(&dir), fake_embedding())
            .await
            .unwrap();
        let id = store
            .add("hello world", serde_json::json!({}))
            .await
            .unwrap();
        assert!(!id.is_empty());
    }

    #[tokio::test]
    async fn add_increments_document_count() {
        let dir = tempfile::tempdir().unwrap();
        let store = ContextStore::new(test_config(&dir), fake_embedding())
            .await
            .unwrap();
        store.add("first", serde_json::json!({})).await.unwrap();
        store.add("second", serde_json::json!({})).await.unwrap();
        let stats = store.stats().await.unwrap();
        assert_eq!(stats.document_count, 2);
    }

    #[tokio::test]
    async fn add_rejects_empty_content() {
        let dir = tempfile::tempdir().unwrap();
        let store = ContextStore::new(test_config(&dir), fake_embedding())
            .await
            .unwrap();
        let result = store.add("", serde_json::json!({})).await;
        assert!(result.is_err());
    }

    // === Search ===

    #[tokio::test]
    async fn search_returns_matching_results() {
        let dir = tempfile::tempdir().unwrap();
        let store = ContextStore::new(test_config(&dir), fake_embedding())
            .await
            .unwrap();
        store
            .add("hello world", serde_json::json!({}))
            .await
            .unwrap();
        store
            .add("hello there", serde_json::json!({}))
            .await
            .unwrap();
        store.add("goodbye", serde_json::json!({})).await.unwrap();

        let results = store.search("hello", 2, None).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn search_returns_results_with_scores() {
        let dir = tempfile::tempdir().unwrap();
        let store = ContextStore::new(test_config(&dir), fake_embedding())
            .await
            .unwrap();
        store.add("hello", serde_json::json!({})).await.unwrap();

        let results = store.search("hello", 1, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].score >= 0.0);
    }

    #[tokio::test]
    async fn search_respects_k_limit() {
        let dir = tempfile::tempdir().unwrap();
        let store = ContextStore::new(test_config(&dir), fake_embedding())
            .await
            .unwrap();
        for i in 0..10 {
            store
                .add(&format!("document {i}"), serde_json::json!({}))
                .await
                .unwrap();
        }

        let results = store.search("document", 3, None).await.unwrap();
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn search_on_empty_store_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = ContextStore::new(test_config(&dir), fake_embedding())
            .await
            .unwrap();
        let results = store.search("anything", 5, None).await.unwrap();
        assert!(results.is_empty());
    }

    // === Delete ===

    #[tokio::test]
    async fn delete_removes_document() {
        let dir = tempfile::tempdir().unwrap();
        let store = ContextStore::new(test_config(&dir), fake_embedding())
            .await
            .unwrap();
        let id = store
            .add("to be deleted", serde_json::json!({}))
            .await
            .unwrap();

        store.delete(&id).await.unwrap();

        let stats = store.stats().await.unwrap();
        assert_eq!(stats.document_count, 0);
    }

    #[tokio::test]
    async fn delete_nonexistent_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let store = ContextStore::new(test_config(&dir), fake_embedding())
            .await
            .unwrap();
        let result = store.delete("nonexistent_id").await;
        assert!(result.is_err());
    }

    // === Metadata Filtering ===

    #[tokio::test]
    async fn search_with_metadata_filter_narrows_results() {
        let dir = tempfile::tempdir().unwrap();
        let store = ContextStore::new(test_config(&dir), fake_embedding())
            .await
            .unwrap();
        store
            .add("hello alpha", serde_json::json!({"source": "docs"}))
            .await
            .unwrap();
        store
            .add("hello beta", serde_json::json!({"source": "code"}))
            .await
            .unwrap();
        store
            .add("hello gamma", serde_json::json!({"source": "docs"}))
            .await
            .unwrap();

        let filter = "metadata LIKE '%\"source\":\"docs\"%'".to_string();
        let results = store.search("hello", 10, Some(filter)).await.unwrap();
        assert_eq!(results.len(), 2);
        for r in &results {
            assert_eq!(r.metadata["source"], "docs");
        }
    }

    #[tokio::test]
    async fn search_filter_returns_empty_when_nothing_matches() {
        let dir = tempfile::tempdir().unwrap();
        let store = ContextStore::new(test_config(&dir), fake_embedding())
            .await
            .unwrap();
        store
            .add("hello", serde_json::json!({"source": "docs"}))
            .await
            .unwrap();

        let filter = "metadata LIKE '%\"source\":\"nonexistent\"%'".to_string();
        let results = store.search("hello", 10, Some(filter)).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn search_preserves_metadata_in_results() {
        let dir = tempfile::tempdir().unwrap();
        let store = ContextStore::new(test_config(&dir), fake_embedding())
            .await
            .unwrap();
        let meta = serde_json::json!({"source": "test", "priority": 5});
        store.add("hello world", meta.clone()).await.unwrap();

        let results = store.search("hello", 1, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].metadata["source"], "test");
        assert_eq!(results[0].metadata["priority"], 5);
    }
}
