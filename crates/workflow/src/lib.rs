#![doc = include_str!("../README.md")]

//! Erio Workflow - DAG workflow engine for orchestrating steps.

pub mod builder;
pub mod checkpoint;
pub mod conditional;
pub mod context;
pub mod dag;
pub mod engine;
pub mod error;
pub mod step;

pub use builder::Workflow;
pub use checkpoint::Checkpoint;
pub use context::WorkflowContext;
pub use dag::Dag;
pub use engine::WorkflowEngine;
pub use error::WorkflowError;
pub use step::{Step, StepOutput};
