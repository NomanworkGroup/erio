//! Erio Embedding - embedding engine abstraction for computing vector embeddings.

pub mod config;
pub mod engine;
pub mod error;
pub mod in_memory;

pub use config::EmbeddingConfig;
pub use engine::EmbeddingEngine;
pub use error::EmbeddingError;
pub use in_memory::InMemoryEmbedding;
