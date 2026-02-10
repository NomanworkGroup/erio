//! Workflow context for passing data between steps.

use std::collections::HashMap;

use crate::step::StepOutput;

/// Shared context that flows through a workflow execution.
///
/// Stores outputs from completed steps so downstream steps can access them.
#[derive(Debug, Clone, Default)]
pub struct WorkflowContext {
    outputs: HashMap<String, StepOutput>,
}

impl WorkflowContext {
    /// Creates an empty workflow context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the output of a completed step, if available.
    pub fn output(&self, step_id: &str) -> Option<&StepOutput> {
        self.outputs.get(step_id)
    }

    /// Returns all step outputs.
    pub fn step_outputs(&self) -> &HashMap<String, StepOutput> {
        &self.outputs
    }

    /// Stores the output of a completed step.
    pub fn set_output(&mut self, step_id: &str, output: StepOutput) {
        self.outputs.insert(step_id.into(), output);
    }

    /// Returns `true` if the step has completed.
    pub fn is_completed(&self, step_id: &str) -> bool {
        self.outputs.contains_key(step_id)
    }

    /// Returns the IDs of all completed steps.
    pub fn completed_step_ids(&self) -> Vec<&str> {
        self.outputs.keys().map(String::as_str).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Construction ===

    #[test]
    fn new_context_has_no_outputs() {
        let ctx = WorkflowContext::new();
        assert!(ctx.step_outputs().is_empty());
    }

    // === Step Output Storage ===

    #[test]
    fn stores_step_output() {
        let mut ctx = WorkflowContext::new();
        let output = StepOutput::new("hello");
        ctx.set_output("step_a", output);
        assert!(ctx.output("step_a").is_some());
        assert_eq!(ctx.output("step_a").unwrap().value(), "hello");
    }

    #[test]
    fn returns_none_for_unknown_step() {
        let ctx = WorkflowContext::new();
        assert!(ctx.output("unknown").is_none());
    }

    #[test]
    fn stores_multiple_outputs() {
        let mut ctx = WorkflowContext::new();
        ctx.set_output("a", StepOutput::new("A result"));
        ctx.set_output("b", StepOutput::new("B result"));
        assert_eq!(ctx.step_outputs().len(), 2);
    }

    // === Completed Steps ===

    #[test]
    fn tracks_completed_steps() {
        let mut ctx = WorkflowContext::new();
        ctx.set_output("a", StepOutput::new("done"));
        assert!(ctx.is_completed("a"));
        assert!(!ctx.is_completed("b"));
    }

    #[test]
    fn completed_step_ids_returns_all() {
        let mut ctx = WorkflowContext::new();
        ctx.set_output("x", StepOutput::new("1"));
        ctx.set_output("y", StepOutput::new("2"));
        let mut ids = ctx.completed_step_ids();
        ids.sort();
        assert_eq!(ids, vec!["x", "y"]);
    }
}
