//! Conditional step that executes based on a runtime predicate.

use crate::WorkflowError;
use crate::context::WorkflowContext;
use crate::step::{Step, StepOutput};

/// A step that only executes its inner step when a condition is met.
///
/// If the condition returns `false`, the step produces a skipped output
/// and downstream steps still see it as completed.
pub struct ConditionalStep<S: Step> {
    id: String,
    inner: S,
    condition: Box<dyn Fn(&WorkflowContext) -> bool + Send + Sync>,
}

impl<S: Step> ConditionalStep<S> {
    /// Creates a conditional step wrapping `inner` with the given predicate.
    pub fn new(
        id: &str,
        inner: S,
        condition: impl Fn(&WorkflowContext) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            id: id.into(),
            inner,
            condition: Box::new(condition),
        }
    }
}

#[async_trait::async_trait]
impl<S: Step + 'static> Step for ConditionalStep<S> {
    fn id(&self) -> &str {
        &self.id
    }

    async fn execute(&self, ctx: &mut WorkflowContext) -> Result<StepOutput, WorkflowError> {
        if (self.condition)(ctx) {
            self.inner.execute(ctx).await
        } else {
            Ok(StepOutput::skipped())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Mock inner step ===

    struct MockInner {
        id: String,
        output: String,
    }

    impl MockInner {
        fn new(id: &str, output: &str) -> Self {
            Self {
                id: id.into(),
                output: output.into(),
            }
        }
    }

    #[async_trait::async_trait]
    impl Step for MockInner {
        fn id(&self) -> &str {
            &self.id
        }

        async fn execute(&self, _ctx: &mut WorkflowContext) -> Result<StepOutput, WorkflowError> {
            Ok(StepOutput::new(&self.output))
        }
    }

    // === ConditionalStep Tests ===

    #[tokio::test]
    async fn executes_inner_when_condition_is_true() {
        let inner = MockInner::new("check", "executed");
        let step = ConditionalStep::new("check", inner, |_ctx| true);

        let mut ctx = WorkflowContext::new();
        let output = step.execute(&mut ctx).await.unwrap();

        assert_eq!(output.value(), "executed");
        assert!(!output.is_skipped());
    }

    #[tokio::test]
    async fn skips_when_condition_is_false() {
        let inner = MockInner::new("check", "executed");
        let step = ConditionalStep::new("check", inner, |_ctx| false);

        let mut ctx = WorkflowContext::new();
        let output = step.execute(&mut ctx).await.unwrap();

        assert!(output.is_skipped());
        assert_eq!(output.value(), "");
    }

    #[tokio::test]
    async fn condition_receives_workflow_context() {
        let inner = MockInner::new("check", "ran");
        let step = ConditionalStep::new("check", inner, |ctx| {
            // Only run if step "gate" produced "open"
            ctx.output("gate")
                .map(|o| o.value() == "open")
                .unwrap_or(false)
        });

        // Without gate output → skip
        let mut ctx = WorkflowContext::new();
        let output = step.execute(&mut ctx).await.unwrap();
        assert!(output.is_skipped());

        // With gate output "closed" → skip
        let mut ctx2 = WorkflowContext::new();
        ctx2.set_output("gate", StepOutput::new("closed"));
        let output2 = step.execute(&mut ctx2).await.unwrap();
        assert!(output2.is_skipped());

        // With gate output "open" → execute
        let mut ctx3 = WorkflowContext::new();
        ctx3.set_output("gate", StepOutput::new("open"));
        let output3 = step.execute(&mut ctx3).await.unwrap();
        assert_eq!(output3.value(), "ran");
    }

    #[test]
    fn returns_correct_id() {
        let inner = MockInner::new("inner_id", "val");
        let step = ConditionalStep::new("my_cond", inner, |_| true);
        assert_eq!(step.id(), "my_cond");
    }

    #[tokio::test]
    async fn inner_error_propagates() {
        struct FailInner;

        #[async_trait::async_trait]
        impl Step for FailInner {
            fn id(&self) -> &str {
                "fail"
            }
            async fn execute(
                &self,
                _ctx: &mut WorkflowContext,
            ) -> Result<StepOutput, WorkflowError> {
                Err(WorkflowError::StepFailed {
                    step_id: "fail".into(),
                    message: "boom".into(),
                })
            }
        }

        let step = ConditionalStep::new("cond_fail", FailInner, |_| true);
        let mut ctx = WorkflowContext::new();
        let result = step.execute(&mut ctx).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn skipped_output_has_metadata_marker() {
        let output = StepOutput::skipped();
        assert!(output.is_skipped());
        assert_eq!(output.value(), "");
    }

    #[tokio::test]
    async fn integrates_with_workflow_builder() {
        use crate::builder::Workflow;
        use crate::engine::WorkflowEngine;

        struct GateStep;

        #[async_trait::async_trait]
        impl Step for GateStep {
            fn id(&self) -> &str {
                "gate"
            }
            async fn execute(
                &self,
                _ctx: &mut WorkflowContext,
            ) -> Result<StepOutput, WorkflowError> {
                Ok(StepOutput::new("open"))
            }
        }

        let workflow = Workflow::builder()
            .step(GateStep, &[])
            .step(
                ConditionalStep::new("guarded", MockInner::new("guarded", "success"), |ctx| {
                    ctx.output("gate")
                        .map(|o| o.value() == "open")
                        .unwrap_or(false)
                }),
                &["gate"],
            )
            .build()
            .unwrap();

        let engine = WorkflowEngine::new();
        let result = engine.run(workflow).await.unwrap();

        assert_eq!(result.output("guarded").unwrap().value(), "success");
    }

    #[tokio::test]
    async fn skipped_step_in_workflow_still_completes() {
        use crate::builder::Workflow;
        use crate::engine::WorkflowEngine;

        struct GateStep;

        #[async_trait::async_trait]
        impl Step for GateStep {
            fn id(&self) -> &str {
                "gate"
            }
            async fn execute(
                &self,
                _ctx: &mut WorkflowContext,
            ) -> Result<StepOutput, WorkflowError> {
                Ok(StepOutput::new("closed"))
            }
        }

        let workflow = Workflow::builder()
            .step(GateStep, &[])
            .step(
                ConditionalStep::new("guarded", MockInner::new("guarded", "ran"), |ctx| {
                    ctx.output("gate")
                        .map(|o| o.value() == "open")
                        .unwrap_or(false)
                }),
                &["gate"],
            )
            .build()
            .unwrap();

        let engine = WorkflowEngine::new();
        let result = engine.run(workflow).await.unwrap();

        // Step is "completed" but output is skipped
        assert!(result.is_completed("guarded"));
        assert!(result.output("guarded").unwrap().is_skipped());
    }
}
