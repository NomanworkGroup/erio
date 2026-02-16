# erio-workflow

`erio-workflow` provides a DAG-based workflow engine for orchestrating async
steps with dependencies, checkpointing support, and typed execution context.

It is useful for plan-then-execute style agents where each step may depend on
prior outputs.

## Quickstart

```rust,no_run
use erio_workflow::{Workflow, WorkflowEngine};

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let workflow = Workflow::builder().build()?;
    let engine = WorkflowEngine::new();
    let _ctx = engine.run(workflow).await?;
    Ok(())
}
```

## API tour

- Workflow construction/execution: `Workflow`, `Dag`, `WorkflowEngine`
- Step contracts: `Step`, `StepOutput`, `WorkflowContext`
- Control/state: `Checkpoint`
- Error type: `WorkflowError`
- Modules: `builder`, `dag`, `engine`, `step`, `context`, `checkpoint`,
  `conditional`, `error`

## Related crates

- Commonly executes steps that call `erio-tools` and `erio-llm-client`.
- Shares runtime conventions with the rest of the Erio workspace.
- Docs: <https://docs.rs/erio-workflow>
- Source: <https://github.com/NomanworkGroup/erio/tree/main/crates/workflow>

## Compatibility

- MSRV: Rust 1.93
- License: Apache-2.0
