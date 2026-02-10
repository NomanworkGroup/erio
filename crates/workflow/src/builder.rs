//! Fluent builder for constructing workflows.

use std::collections::HashMap;
use std::sync::Arc;

use crate::dag::Dag;
use crate::step::Step;
use crate::WorkflowError;

/// A validated workflow ready for execution.
///
/// Contains steps arranged in a DAG with dependency relationships.
pub struct Workflow {
    steps: HashMap<String, Arc<dyn Step>>,
    dag: Dag,
}

impl Workflow {
    /// Creates a new builder for constructing a workflow.
    pub fn builder() -> WorkflowBuilder {
        WorkflowBuilder::new()
    }

    /// Returns the number of steps.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Returns the IDs of all steps in topological order.
    pub fn step_ids(&self) -> Vec<&str> {
        // Unwrap is safe: DAG was validated at build time
        self.dag.topological_order().expect("DAG was validated")
    }

    /// Returns a step by its ID.
    pub fn step(&self, id: &str) -> Option<Arc<dyn Step>> {
        self.steps.get(id).cloned()
    }

    /// Returns groups of steps that can run in parallel.
    pub fn parallel_groups(&self) -> Result<Vec<Vec<&str>>, WorkflowError> {
        self.dag.parallel_groups()
    }

    /// Returns a reference to the internal DAG.
    pub fn dag(&self) -> &Dag {
        &self.dag
    }
}

/// Builder for constructing a `Workflow` with validated dependencies.
pub struct WorkflowBuilder {
    steps: Vec<(Box<dyn Step>, Vec<String>)>,
}

impl WorkflowBuilder {
    fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Adds a step with its dependency IDs.
    #[must_use]
    pub fn step(mut self, step: impl Step + 'static, deps: &[&str]) -> Self {
        let dep_list = deps.iter().map(|d| (*d).into()).collect();
        self.steps.push((Box::new(step), dep_list));
        self
    }

    /// Validates the DAG and builds the workflow.
    pub fn build(self) -> Result<Workflow, WorkflowError> {
        if self.steps.is_empty() {
            return Err(WorkflowError::EmptyWorkflow);
        }

        let mut dag = Dag::new();
        let mut step_map: HashMap<String, Arc<dyn Step>> = HashMap::new();

        for (step, deps) in self.steps {
            let id = step.id().to_string();
            let dep_refs: Vec<&str> = deps.iter().map(String::as_str).collect();
            dag.add_node(&id, &dep_refs)?;
            step_map.insert(id, Arc::from(step));
        }

        // Validate DAG is acyclic
        dag.topological_order()?;

        Ok(Workflow {
            steps: step_map,
            dag,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::WorkflowContext;
    use crate::step::{Step, StepOutput};
    use crate::WorkflowError;

    // === Mock Step ===

    struct MockStep {
        step_id: String,
        output: String,
    }

    impl MockStep {
        fn new(id: &str, output: &str) -> Self {
            Self {
                step_id: id.into(),
                output: output.into(),
            }
        }
    }

    #[async_trait::async_trait]
    impl Step for MockStep {
        fn id(&self) -> &str {
            &self.step_id
        }

        async fn execute(&self, _ctx: &mut WorkflowContext) -> Result<StepOutput, WorkflowError> {
            Ok(StepOutput::new(&self.output))
        }
    }

    // === Builder Tests ===

    #[test]
    fn builds_workflow_with_single_step() {
        let workflow = Workflow::builder()
            .step(MockStep::new("a", "result"), &[])
            .build()
            .unwrap();

        assert_eq!(workflow.step_count(), 1);
    }

    #[test]
    fn builds_workflow_with_dependencies() {
        let workflow = Workflow::builder()
            .step(MockStep::new("a", "A"), &[])
            .step(MockStep::new("b", "B"), &["a"])
            .step(MockStep::new("c", "C"), &["a", "b"])
            .build()
            .unwrap();

        assert_eq!(workflow.step_count(), 3);
    }

    #[test]
    fn build_rejects_empty_workflow() {
        let result = Workflow::builder().build();
        assert!(matches!(result, Err(WorkflowError::EmptyWorkflow)));
    }

    #[test]
    fn build_rejects_duplicate_step_ids() {
        let result = Workflow::builder()
            .step(MockStep::new("a", "1"), &[])
            .step(MockStep::new("a", "2"), &[])
            .build();

        assert!(matches!(
            result,
            Err(WorkflowError::DuplicateStep { step_id }) if step_id == "a"
        ));
    }

    #[test]
    fn build_rejects_missing_dependency() {
        let result = Workflow::builder()
            .step(MockStep::new("b", "B"), &["unknown"])
            .build();

        assert!(matches!(
            result,
            Err(WorkflowError::MissingDependency { .. })
        ));
    }

    #[test]
    fn workflow_returns_step_ids() {
        let workflow = Workflow::builder()
            .step(MockStep::new("x", "X"), &[])
            .step(MockStep::new("y", "Y"), &["x"])
            .build()
            .unwrap();

        let ids = workflow.step_ids();
        assert_eq!(ids, vec!["x", "y"]);
    }

    #[test]
    fn workflow_returns_parallel_groups() {
        let workflow = Workflow::builder()
            .step(MockStep::new("a", "A"), &[])
            .step(MockStep::new("b", "B"), &[])
            .step(MockStep::new("c", "C"), &["a", "b"])
            .build()
            .unwrap();

        let groups = workflow.parallel_groups().unwrap();
        assert_eq!(groups.len(), 2);
        let mut first = groups[0].clone();
        first.sort();
        assert_eq!(first, vec!["a", "b"]);
        assert_eq!(groups[1], vec!["c"]);
    }

    #[test]
    fn workflow_gets_step_by_id() {
        let workflow = Workflow::builder()
            .step(MockStep::new("a", "A"), &[])
            .build()
            .unwrap();

        assert!(workflow.step("a").is_some());
        assert!(workflow.step("unknown").is_none());
    }
}
