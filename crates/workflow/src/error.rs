//! Error types for the workflow engine.

use thiserror::Error;

fn cycle_display(steps: &[String]) -> String {
    steps.join(" -> ")
}

/// Errors from the workflow engine.
#[derive(Debug, Error)]
pub enum WorkflowError {
    #[error("Cycle detected: {}", cycle_display(.steps))]
    CycleDetected { steps: Vec<String> },

    #[error("Step not found: {step_id}")]
    StepNotFound { step_id: String },

    #[error("Duplicate step: {step_id}")]
    DuplicateStep { step_id: String },

    #[error("Step '{step_id}' failed: {message}")]
    StepFailed { step_id: String, message: String },

    #[error("Step '{step_id}' skipped: dependency '{dependency_id}' failed")]
    DependencyFailed {
        step_id: String,
        dependency_id: String,
    },

    #[error("Step '{step_id}' depends on unknown step '{dependency_id}'")]
    MissingDependency {
        step_id: String,
        dependency_id: String,
    },

    #[error("Workflow has no steps")]
    EmptyWorkflow,

    #[error("Checkpoint error: {message}")]
    Checkpoint { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Display Tests ===

    #[test]
    fn cycle_detected_displays_steps() {
        let err = WorkflowError::CycleDetected {
            steps: vec!["a".into(), "b".into(), "a".into()],
        };
        assert_eq!(err.to_string(), "Cycle detected: a -> b -> a");
    }

    #[test]
    fn step_not_found_displays_id() {
        let err = WorkflowError::StepNotFound {
            step_id: "missing".into(),
        };
        assert_eq!(err.to_string(), "Step not found: missing");
    }

    #[test]
    fn duplicate_step_displays_id() {
        let err = WorkflowError::DuplicateStep {
            step_id: "dup".into(),
        };
        assert_eq!(err.to_string(), "Duplicate step: dup");
    }

    #[test]
    fn step_failed_displays_id_and_message() {
        let err = WorkflowError::StepFailed {
            step_id: "build".into(),
            message: "compile error".into(),
        };
        assert_eq!(err.to_string(), "Step 'build' failed: compile error");
    }

    #[test]
    fn dependency_failed_displays_ids() {
        let err = WorkflowError::DependencyFailed {
            step_id: "deploy".into(),
            dependency_id: "build".into(),
        };
        assert_eq!(
            err.to_string(),
            "Step 'deploy' skipped: dependency 'build' failed"
        );
    }

    #[test]
    fn missing_dependency_displays_ids() {
        let err = WorkflowError::MissingDependency {
            step_id: "b".into(),
            dependency_id: "unknown".into(),
        };
        assert_eq!(
            err.to_string(),
            "Step 'b' depends on unknown step 'unknown'"
        );
    }

    #[test]
    fn empty_workflow_displays_message() {
        let err = WorkflowError::EmptyWorkflow;
        assert_eq!(err.to_string(), "Workflow has no steps");
    }

    #[test]
    fn checkpoint_error_displays_message() {
        let err = WorkflowError::Checkpoint {
            message: "write failed".into(),
        };
        assert_eq!(err.to_string(), "Checkpoint error: write failed");
    }
}
