//! Erio Context Store - vector storage and semantic search.

pub mod config;
pub mod error;
pub mod store;

pub use config::{ContextConfig, HnswConfig, SearchResult, StorageStats};
pub use error::ContextStoreError;
pub use store::ContextStore;
