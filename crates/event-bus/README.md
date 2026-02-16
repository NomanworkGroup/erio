# erio-event-bus

`erio-event-bus` provides an async event bus with a simple event model and
pluggable event sources.

Use it to coordinate components (or specialist agents) via publish/subscribe
events instead of direct coupling.

## Quickstart

```rust,no_run
use erio_event_bus::{Event, EventBus};

let bus = EventBus::new();
let mut _rx = bus.subscribe();

let event = Event::new("demo-source", "tick", serde_json::json!({"ok": true}));
let _ = bus.publish(event);
```

## API tour

- Core types: `Event`, `EventBus`
- Source contract: `EventSource`
- Error type: `EventBusError`
- Modules: `event`, `bus`, `source`, `error`

## Related crates

- Often paired with `erio-llm-client` in event-driven agent pipelines.
- Uses `serde_json` payloads for crate-agnostic event data.
- Docs: <https://docs.rs/erio-event-bus>
- Source: <https://github.com/NomanworkGroup/erio/tree/main/crates/event-bus>

## Compatibility

- MSRV: Rust 1.93
- License: Apache-2.0
