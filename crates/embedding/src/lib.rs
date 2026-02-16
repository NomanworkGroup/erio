#![doc = include_str!("../README.md")]

//! Erio Embedding - embedding engine abstraction for computing vector embeddings.
//!
//! Local embedding engines (e.g. `GemmaEmbedding`) load model files from `ERIO_MODEL_DIR`.
//! The embedding crate's build script fetches the default model assets at build time.

pub mod config;
pub mod engine;
pub mod error;
pub mod gemma;
pub mod model;
pub mod remote;
pub mod task;

pub use config::EmbeddingConfig;
pub use engine::EmbeddingEngine;
pub use error::EmbeddingError;
pub use gemma::GemmaEmbedding;
pub use remote::RemoteEmbedding;
pub use task::TaskType;
