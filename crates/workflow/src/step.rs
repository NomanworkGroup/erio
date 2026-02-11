//! Step trait for workflow execution units.

use crate::context::WorkflowContext;
use crate::WorkflowError;

/// A single execution unit in a workflow.
#[async_trait::async_trait]
pub trait Step: Send + Sync {
    /// Returns the unique identifier for this step.
    fn id(&self) -> &str;

    /// Executes the step with the given workflow context.
    async fn execute(&self, ctx: &mut WorkflowContext) -> Result<StepOutput, WorkflowError>;
}

/// The output produced by a step execution.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StepOutput {
    value: String,
    metadata: Option<serde_json::Value>,
    skipped: bool,
}

impl StepOutput {
    /// Creates a new step output with the given value.
    pub fn new(value: &str) -> Self {
        Self {
            value: value.into(),
            metadata: None,
            skipped: false,
        }
    }

    /// Creates a skipped step output (condition was false).
    pub fn skipped() -> Self {
        Self {
            value: String::new(),
            metadata: None,
            skipped: true,
        }
    }

    /// Returns `true` if this output represents a skipped step.
    pub fn is_skipped(&self) -> bool {
        self.skipped
    }

    /// Returns the output value.
    pub fn value(&self) -> &str {
        &self.value
    }

    /// Returns the metadata, if any.
    pub fn metadata(&self) -> Option<&serde_json::Value> {
        self.metadata.as_ref()
    }

    /// Attaches metadata to the output.
    #[must_use]
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Mock Step ===

    struct AddStep {
        id: String,
        value: String,
    }

    impl AddStep {
        fn new(id: &str, value: &str) -> Self {
            Self {
                id: id.into(),
                value: value.into(),
            }
        }
    }

    #[async_trait::async_trait]
    impl Step for AddStep {
        fn id(&self) -> &str {
            &self.id
        }

        async fn execute(&self, _ctx: &mut WorkflowContext) -> Result<StepOutput, WorkflowError> {
            Ok(StepOutput::new(&self.value))
        }
    }

    // === Trait Tests ===

    #[test]
    fn step_returns_id() {
        let step = AddStep::new("build", "result");
        assert_eq!(step.id(), "build");
    }

    #[tokio::test]
    async fn step_executes_and_returns_output() {
        let step = AddStep::new("build", "compiled");
        let mut ctx = WorkflowContext::new();
        let output = step.execute(&mut ctx).await.unwrap();
        assert_eq!(output.value(), "compiled");
    }

    // === StepOutput Tests ===

    #[test]
    fn step_output_stores_value() {
        let output = StepOutput::new("hello");
        assert_eq!(output.value(), "hello");
    }

    #[test]
    fn step_output_with_metadata() {
        let output =
            StepOutput::new("result").with_metadata(serde_json::json!({"exit_code": 0}));
        assert_eq!(output.value(), "result");
        assert!(output.metadata().is_some());
    }
}
