//! Workflow execution engine with parallel step execution.

use std::sync::Arc;

use tokio::sync::Mutex;

use std::path::Path;

use crate::WorkflowError;
use crate::builder::Workflow;
use crate::checkpoint::Checkpoint;
use crate::context::WorkflowContext;
use crate::step::StepOutput;

/// Executes workflows by resolving the DAG and running steps.
///
/// Independent steps are executed in parallel using tokio tasks.
#[derive(Debug, Clone, Default)]
pub struct WorkflowEngine;

impl WorkflowEngine {
    /// Creates a new workflow engine.
    pub fn new() -> Self {
        Self
    }

    /// Runs a workflow to completion.
    ///
    /// Steps are executed in parallel groups determined by the DAG.
    /// If any step fails, dependent steps are skipped and the error is returned.
    pub async fn run(&self, workflow: Workflow) -> Result<WorkflowContext, WorkflowError> {
        let groups = workflow.parallel_groups()?;
        let ctx = Arc::new(Mutex::new(WorkflowContext::new()));
        let failed: Arc<Mutex<Option<WorkflowError>>> = Arc::new(Mutex::new(None));

        for group in groups {
            // Check if a previous step already failed
            if failed.lock().await.is_some() {
                break;
            }

            if group.len() == 1 {
                // Single step — run directly (no spawn overhead)
                let step_id = group[0];
                let step = workflow.step(step_id).expect("DAG validated step exists");

                let mut ctx_guard = ctx.lock().await;
                match step.execute(&mut ctx_guard).await {
                    Ok(output) => {
                        ctx_guard.set_output(step_id, output);
                    }
                    Err(e) => {
                        return Err(e);
                    }
                }
            } else {
                // Multiple independent steps — run in parallel
                let mut handles = Vec::with_capacity(group.len());

                for step_id in &group {
                    let step = workflow.step(step_id).expect("DAG validated step exists");
                    let ctx_clone = ctx.clone();
                    let failed_clone = failed.clone();
                    let step_id_owned = (*step_id).to_string();

                    let handle = tokio::spawn(async move {
                        // Take a snapshot of context for this step
                        let mut ctx_snapshot = ctx_clone.lock().await.clone();
                        drop(ctx_clone); // Release lock during execution

                        match step.execute(&mut ctx_snapshot).await {
                            Ok(output) => Ok((step_id_owned, output)),
                            Err(e) => {
                                *failed_clone.lock().await = Some(WorkflowError::StepFailed {
                                    step_id: step_id_owned.clone(),
                                    message: e.to_string(),
                                });
                                Err(e)
                            }
                        }
                    });

                    handles.push(handle);
                }

                // Collect results
                let mut first_error: Option<WorkflowError> = None;
                let mut outputs: Vec<(String, StepOutput)> = Vec::new();

                for handle in handles {
                    match handle.await {
                        Ok(Ok((id, output))) => outputs.push((id, output)),
                        Ok(Err(e)) => {
                            if first_error.is_none() {
                                first_error = Some(e);
                            }
                        }
                        Err(join_err) => {
                            if first_error.is_none() {
                                first_error = Some(WorkflowError::StepFailed {
                                    step_id: "unknown".into(),
                                    message: format!("Task panicked: {join_err}"),
                                });
                            }
                        }
                    }
                }

                // If any step in this group failed, return the error
                if let Some(err) = first_error {
                    return Err(err);
                }

                // Store all outputs
                let mut ctx_guard = ctx.lock().await;
                for (id, output) in outputs {
                    ctx_guard.set_output(&id, output);
                }
            }
        }

        let result = ctx.lock().await.clone();
        Ok(result)
    }

    /// Runs a workflow with checkpointing after each group completes.
    ///
    /// If a checkpoint file already exists at the path, completed steps are skipped.
    pub async fn run_with_checkpoint(
        &self,
        workflow: Workflow,
        checkpoint_path: &Path,
    ) -> Result<WorkflowContext, WorkflowError> {
        let groups = workflow.parallel_groups()?;

        // Load existing checkpoint or create new
        let mut checkpoint = if checkpoint_path.exists() {
            Checkpoint::load(checkpoint_path).await?
        } else {
            Checkpoint::new()
        };

        let ctx = Arc::new(Mutex::new(checkpoint.clone().into_context()));

        for group in groups {
            // Filter out already-completed steps
            let pending: Vec<&str> = group
                .iter()
                .filter(|id| !checkpoint.is_completed(id))
                .copied()
                .collect();

            if pending.is_empty() {
                continue;
            }

            if pending.len() == 1 {
                let step_id = pending[0];
                let step = workflow.step(step_id).expect("DAG validated");
                let mut ctx_guard = ctx.lock().await;
                let output = step.execute(&mut ctx_guard).await?;
                ctx_guard.set_output(step_id, output.clone());
                checkpoint.mark_completed(step_id, output);
            } else {
                let mut handles = Vec::with_capacity(pending.len());

                for step_id in &pending {
                    let step = workflow.step(step_id).expect("DAG validated");
                    let ctx_clone = ctx.clone();
                    let step_id_owned = (*step_id).to_string();

                    let handle = tokio::spawn(async move {
                        let mut ctx_snapshot = ctx_clone.lock().await.clone();
                        drop(ctx_clone);
                        let output = step.execute(&mut ctx_snapshot).await?;
                        Ok::<_, WorkflowError>((step_id_owned, output))
                    });
                    handles.push(handle);
                }

                let mut first_error: Option<WorkflowError> = None;
                let mut outputs: Vec<(String, StepOutput)> = Vec::new();

                for handle in handles {
                    match handle.await {
                        Ok(Ok((id, output))) => outputs.push((id, output)),
                        Ok(Err(e)) => {
                            if first_error.is_none() {
                                first_error = Some(e);
                            }
                        }
                        Err(join_err) => {
                            if first_error.is_none() {
                                first_error = Some(WorkflowError::StepFailed {
                                    step_id: "unknown".into(),
                                    message: format!("Task panicked: {join_err}"),
                                });
                            }
                        }
                    }
                }

                if let Some(err) = first_error {
                    // Save checkpoint before returning error
                    checkpoint.save(checkpoint_path).await?;
                    return Err(err);
                }

                let mut ctx_guard = ctx.lock().await;
                for (id, output) in outputs {
                    ctx_guard.set_output(&id, output.clone());
                    checkpoint.mark_completed(&id, output);
                }
            }

            // Save checkpoint after each group
            checkpoint.save(checkpoint_path).await?;
        }

        let result = ctx.lock().await.clone();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WorkflowError;
    use crate::builder::Workflow;
    use crate::context::WorkflowContext;
    use crate::step::{Step, StepOutput};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    // === Mock Steps ===

    struct ValueStep {
        step_id: String,
        output: String,
    }

    impl ValueStep {
        fn new(id: &str, output: &str) -> Self {
            Self {
                step_id: id.into(),
                output: output.into(),
            }
        }
    }

    #[async_trait::async_trait]
    impl Step for ValueStep {
        fn id(&self) -> &str {
            &self.step_id
        }

        async fn execute(&self, _ctx: &mut WorkflowContext) -> Result<StepOutput, WorkflowError> {
            Ok(StepOutput::new(&self.output))
        }
    }

    /// Step that reads a dependency's output and appends to it.
    struct AppendStep {
        step_id: String,
        dep_id: String,
        suffix: String,
    }

    impl AppendStep {
        fn new(id: &str, dep_id: &str, suffix: &str) -> Self {
            Self {
                step_id: id.into(),
                dep_id: dep_id.into(),
                suffix: suffix.into(),
            }
        }
    }

    #[async_trait::async_trait]
    impl Step for AppendStep {
        fn id(&self) -> &str {
            &self.step_id
        }

        async fn execute(&self, ctx: &mut WorkflowContext) -> Result<StepOutput, WorkflowError> {
            let prev = ctx
                .output(&self.dep_id)
                .map(|o| o.value().to_string())
                .unwrap_or_default();
            Ok(StepOutput::new(&format!("{prev}{}", self.suffix)))
        }
    }

    /// Step that fails.
    struct FailStep {
        step_id: String,
        message: String,
    }

    impl FailStep {
        fn new(id: &str, message: &str) -> Self {
            Self {
                step_id: id.into(),
                message: message.into(),
            }
        }
    }

    #[async_trait::async_trait]
    impl Step for FailStep {
        fn id(&self) -> &str {
            &self.step_id
        }

        async fn execute(&self, _ctx: &mut WorkflowContext) -> Result<StepOutput, WorkflowError> {
            Err(WorkflowError::StepFailed {
                step_id: self.step_id.clone(),
                message: self.message.clone(),
            })
        }
    }

    /// Step that tracks execution via an atomic counter.
    struct CountStep {
        step_id: String,
        counter: Arc<AtomicUsize>,
        delay: Option<Duration>,
    }

    impl CountStep {
        fn new(id: &str, counter: Arc<AtomicUsize>) -> Self {
            Self {
                step_id: id.into(),
                counter,
                delay: None,
            }
        }

        fn with_delay(mut self, delay: Duration) -> Self {
            self.delay = Some(delay);
            self
        }
    }

    #[async_trait::async_trait]
    impl Step for CountStep {
        fn id(&self) -> &str {
            &self.step_id
        }

        async fn execute(&self, _ctx: &mut WorkflowContext) -> Result<StepOutput, WorkflowError> {
            self.counter.fetch_add(1, Ordering::SeqCst);
            if let Some(d) = self.delay {
                tokio::time::sleep(d).await;
            }
            Ok(StepOutput::new("done"))
        }
    }

    // === Engine Tests ===

    #[tokio::test]
    async fn runs_single_step() {
        let workflow = Workflow::builder()
            .step(ValueStep::new("a", "hello"), &[])
            .build()
            .unwrap();

        let engine = WorkflowEngine::new();
        let result = engine.run(workflow).await.unwrap();

        assert!(result.is_completed("a"));
        assert_eq!(result.output("a").unwrap().value(), "hello");
    }

    #[tokio::test]
    async fn runs_linear_chain_passing_context() {
        let workflow = Workflow::builder()
            .step(ValueStep::new("a", "start"), &[])
            .step(AppendStep::new("b", "a", "_middle"), &["a"])
            .step(AppendStep::new("c", "b", "_end"), &["b"])
            .build()
            .unwrap();

        let engine = WorkflowEngine::new();
        let result = engine.run(workflow).await.unwrap();

        assert_eq!(result.output("c").unwrap().value(), "start_middle_end");
    }

    #[tokio::test]
    async fn runs_parallel_independent_steps() {
        let counter = Arc::new(AtomicUsize::new(0));

        let workflow = Workflow::builder()
            .step(
                CountStep::new("a", counter.clone()).with_delay(Duration::from_millis(50)),
                &[],
            )
            .step(
                CountStep::new("b", counter.clone()).with_delay(Duration::from_millis(50)),
                &[],
            )
            .step(
                CountStep::new("c", counter.clone()).with_delay(Duration::from_millis(50)),
                &[],
            )
            .build()
            .unwrap();

        let engine = WorkflowEngine::new();
        let start = std::time::Instant::now();
        let result = engine.run(workflow).await.unwrap();
        let elapsed = start.elapsed();

        // All 3 should have run
        assert_eq!(counter.load(Ordering::SeqCst), 3);
        assert!(result.is_completed("a"));
        assert!(result.is_completed("b"));
        assert!(result.is_completed("c"));

        // Should run in parallel (< 120ms), not sequentially (>= 150ms)
        assert!(elapsed < Duration::from_millis(120));
    }

    #[tokio::test]
    async fn step_failure_propagates_error() {
        let workflow = Workflow::builder()
            .step(FailStep::new("a", "boom"), &[])
            .build()
            .unwrap();

        let engine = WorkflowEngine::new();
        let result = engine.run(workflow).await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            WorkflowError::StepFailed { step_id, .. } if step_id == "a"
        ));
    }

    #[tokio::test]
    async fn dependent_step_skipped_when_dependency_fails() {
        let counter = Arc::new(AtomicUsize::new(0));

        let workflow = Workflow::builder()
            .step(FailStep::new("a", "boom"), &[])
            .step(CountStep::new("b", counter.clone()), &["a"])
            .build()
            .unwrap();

        let engine = WorkflowEngine::new();
        let result = engine.run(workflow).await;

        // Workflow fails
        assert!(result.is_err());
        // Step b never ran
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn diamond_workflow_executes_correctly() {
        //     a
        //    / \
        //   b   c
        //    \ /
        //     d
        let workflow = Workflow::builder()
            .step(ValueStep::new("a", "A"), &[])
            .step(AppendStep::new("b", "a", "_B"), &["a"])
            .step(AppendStep::new("c", "a", "_C"), &["a"])
            .step(AppendStep::new("d", "b", "_D"), &["b", "c"])
            .build()
            .unwrap();

        let engine = WorkflowEngine::new();
        let result = engine.run(workflow).await.unwrap();

        assert_eq!(result.output("a").unwrap().value(), "A");
        assert_eq!(result.output("b").unwrap().value(), "A_B");
        assert_eq!(result.output("c").unwrap().value(), "A_C");
        // d depends on b, reads b's output
        assert_eq!(result.output("d").unwrap().value(), "A_B_D");
    }

    // === Checkpointed Run Tests ===

    #[tokio::test]
    async fn checkpointed_run_saves_checkpoint_file() {
        let dir = tempfile::tempdir().unwrap();
        let ckpt_path = dir.path().join("checkpoint.json");

        let workflow = Workflow::builder()
            .step(ValueStep::new("a", "A"), &[])
            .step(ValueStep::new("b", "B"), &["a"])
            .build()
            .unwrap();

        let engine = WorkflowEngine::new();
        let result = engine
            .run_with_checkpoint(workflow, &ckpt_path)
            .await
            .unwrap();

        assert!(ckpt_path.exists());
        assert!(result.is_completed("a"));
        assert!(result.is_completed("b"));
    }

    #[tokio::test]
    async fn checkpointed_run_skips_completed_steps() {
        let dir = tempfile::tempdir().unwrap();
        let ckpt_path = dir.path().join("checkpoint.json");

        // Pre-populate checkpoint with step "a" completed
        let mut pre_checkpoint = crate::checkpoint::Checkpoint::new();
        pre_checkpoint.mark_completed("a", StepOutput::new("A"));
        pre_checkpoint.save(&ckpt_path).await.unwrap();

        let counter = Arc::new(AtomicUsize::new(0));

        let workflow = Workflow::builder()
            .step(CountStep::new("a", counter.clone()), &[])
            .step(CountStep::new("b", counter.clone()), &["a"])
            .build()
            .unwrap();

        let engine = WorkflowEngine::new();
        let result = engine
            .run_with_checkpoint(workflow, &ckpt_path)
            .await
            .unwrap();

        // Step "a" was already in checkpoint, should not run again
        // Only step "b" should have run
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert!(result.is_completed("a"));
        assert!(result.is_completed("b"));
    }

    #[tokio::test]
    async fn returns_all_completed_step_ids() {
        let workflow = Workflow::builder()
            .step(ValueStep::new("x", "1"), &[])
            .step(ValueStep::new("y", "2"), &[])
            .build()
            .unwrap();

        let engine = WorkflowEngine::new();
        let result = engine.run(workflow).await.unwrap();

        let mut ids = result.completed_step_ids();
        ids.sort();
        assert_eq!(ids, vec!["x", "y"]);
    }
}
