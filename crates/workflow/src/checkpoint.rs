//! Workflow checkpointing and recovery.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::WorkflowError;
use crate::context::WorkflowContext;
use crate::step::StepOutput;

/// A serializable snapshot of workflow progress.
///
/// Stores which steps have completed and their outputs, allowing
/// a workflow to be resumed from the last checkpoint after a crash.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Checkpoint {
    completed: HashMap<String, StepOutput>,
}

impl Checkpoint {
    /// Creates an empty checkpoint.
    pub fn new() -> Self {
        Self::default()
    }

    /// Marks a step as completed with its output.
    pub fn mark_completed(&mut self, step_id: &str, output: StepOutput) {
        self.completed.insert(step_id.into(), output);
    }

    /// Returns `true` if the step has been completed.
    pub fn is_completed(&self, step_id: &str) -> bool {
        self.completed.contains_key(step_id)
    }

    /// Returns the IDs of all completed steps.
    pub fn completed_ids(&self) -> Vec<&str> {
        self.completed.keys().map(String::as_str).collect()
    }

    /// Returns the output of a completed step.
    pub fn output(&self, step_id: &str) -> Option<&StepOutput> {
        self.completed.get(step_id)
    }

    /// Converts this checkpoint into a `WorkflowContext` for resuming execution.
    pub fn into_context(self) -> WorkflowContext {
        let mut ctx = WorkflowContext::new();
        for (id, output) in self.completed {
            ctx.set_output(&id, output);
        }
        ctx
    }

    /// Saves the checkpoint to a JSON file.
    pub async fn save(&self, path: &Path) -> Result<(), WorkflowError> {
        let json = serde_json::to_string_pretty(self).map_err(|e| WorkflowError::Checkpoint {
            message: format!("serialize failed: {e}"),
        })?;
        tokio::fs::write(path, json)
            .await
            .map_err(|e| WorkflowError::Checkpoint {
                message: format!("write failed: {e}"),
            })
    }

    /// Loads a checkpoint from a JSON file.
    pub async fn load(path: &Path) -> Result<Self, WorkflowError> {
        let json =
            tokio::fs::read_to_string(path)
                .await
                .map_err(|e| WorkflowError::Checkpoint {
                    message: format!("read failed: {e}"),
                })?;
        serde_json::from_str(&json).map_err(|e| WorkflowError::Checkpoint {
            message: format!("deserialize failed: {e}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Checkpoint Data Tests ===

    #[test]
    fn checkpoint_stores_completed_steps() {
        let mut checkpoint = Checkpoint::new();
        checkpoint.mark_completed("a", StepOutput::new("A result"));
        checkpoint.mark_completed("b", StepOutput::new("B result"));

        assert!(checkpoint.is_completed("a"));
        assert!(checkpoint.is_completed("b"));
        assert!(!checkpoint.is_completed("c"));
    }

    #[test]
    fn checkpoint_returns_completed_ids() {
        let mut checkpoint = Checkpoint::new();
        checkpoint.mark_completed("x", StepOutput::new("X"));
        checkpoint.mark_completed("y", StepOutput::new("Y"));

        let mut ids = checkpoint.completed_ids();
        ids.sort();
        assert_eq!(ids, vec!["x", "y"]);
    }

    #[test]
    fn checkpoint_returns_output_for_completed_step() {
        let mut checkpoint = Checkpoint::new();
        checkpoint.mark_completed("a", StepOutput::new("hello"));

        let output = checkpoint.output("a").unwrap();
        assert_eq!(output.value(), "hello");
    }

    // === Serialization Tests ===

    #[test]
    fn checkpoint_serializes_to_json() {
        let mut checkpoint = Checkpoint::new();
        checkpoint.mark_completed("a", StepOutput::new("result_a"));

        let json = serde_json::to_string(&checkpoint).unwrap();
        assert!(json.contains("result_a"));
    }

    #[test]
    fn checkpoint_roundtrips_through_json() {
        let mut original = Checkpoint::new();
        original.mark_completed("a", StepOutput::new("A"));
        original.mark_completed("b", StepOutput::new("B"));

        let json = serde_json::to_string(&original).unwrap();
        let restored: Checkpoint = serde_json::from_str(&json).unwrap();

        assert!(restored.is_completed("a"));
        assert!(restored.is_completed("b"));
        assert_eq!(restored.output("a").unwrap().value(), "A");
        assert_eq!(restored.output("b").unwrap().value(), "B");
    }

    // === File Persistence Tests ===

    #[tokio::test]
    async fn saves_and_loads_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("checkpoint.json");

        let mut checkpoint = Checkpoint::new();
        checkpoint.mark_completed("step_1", StepOutput::new("output_1"));
        checkpoint.mark_completed("step_2", StepOutput::new("output_2"));

        checkpoint.save(&path).await.unwrap();
        assert!(path.exists());

        let loaded = Checkpoint::load(&path).await.unwrap();
        assert!(loaded.is_completed("step_1"));
        assert!(loaded.is_completed("step_2"));
        assert_eq!(loaded.output("step_1").unwrap().value(), "output_1");
    }

    #[tokio::test]
    async fn load_returns_error_for_missing_file() {
        let result = Checkpoint::load(Path::new("/tmp/nonexistent_ckpt.json")).await;
        assert!(result.is_err());
    }

    // === Integration with WorkflowContext ===

    #[test]
    fn converts_to_workflow_context() {
        let mut checkpoint = Checkpoint::new();
        checkpoint.mark_completed("a", StepOutput::new("A"));
        checkpoint.mark_completed("b", StepOutput::new("B"));

        let ctx = checkpoint.into_context();

        assert!(ctx.is_completed("a"));
        assert!(ctx.is_completed("b"));
        assert_eq!(ctx.output("a").unwrap().value(), "A");
    }
}
