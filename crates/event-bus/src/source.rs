//! Pluggable event source trait.

use crate::error::EventBusError;
use crate::event::Event;

/// A pluggable source of events for the event bus.
///
/// Implementors produce events that the bus distributes to subscribers.
#[async_trait::async_trait]
pub trait EventSource: Send + Sync {
    /// Returns the name of this source.
    fn name(&self) -> &str;

    /// Returns a human-readable description of this source.
    fn description(&self) -> &str;

    /// Polls for the next event from this source.
    ///
    /// Returns `Ok(Some(event))` when an event is available,
    /// `Ok(None)` when the source is exhausted, or an error.
    async fn next_event(&self) -> Result<Option<Event>, EventBusError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // === Fake Implementation ===

    struct FakeSource {
        events: tokio::sync::Mutex<Vec<Event>>,
    }

    impl FakeSource {
        fn new(events: Vec<Event>) -> Self {
            Self {
                events: tokio::sync::Mutex::new(events),
            }
        }

        fn empty() -> Self {
            Self::new(vec![])
        }
    }

    #[async_trait::async_trait]
    impl EventSource for FakeSource {
        #[allow(clippy::unnecessary_literal_bound)]
        fn name(&self) -> &str {
            "fake"
        }

        #[allow(clippy::unnecessary_literal_bound)]
        fn description(&self) -> &str {
            "A fake event source for testing"
        }

        async fn next_event(&self) -> Result<Option<Event>, EventBusError> {
            let mut events = self.events.lock().await;
            if events.is_empty() {
                Ok(None)
            } else {
                Ok(Some(events.remove(0)))
            }
        }
    }

    // === Failing Source ===

    struct FailingSource;

    #[async_trait::async_trait]
    impl EventSource for FailingSource {
        #[allow(clippy::unnecessary_literal_bound)]
        fn name(&self) -> &str {
            "failing"
        }

        #[allow(clippy::unnecessary_literal_bound)]
        fn description(&self) -> &str {
            "Always fails"
        }

        async fn next_event(&self) -> Result<Option<Event>, EventBusError> {
            Err(EventBusError::Source("connection lost".into()))
        }
    }

    // === Name / Description Tests ===

    #[test]
    fn source_returns_name() {
        let source = FakeSource::empty();
        assert_eq!(source.name(), "fake");
    }

    #[test]
    fn source_returns_description() {
        let source = FakeSource::empty();
        assert_eq!(source.description(), "A fake event source for testing");
    }

    // === next_event Tests ===

    #[tokio::test]
    async fn next_event_returns_event_when_available() {
        let event = Event::new("fake", "test.event", json!({"key": "value"}));
        let source = FakeSource::new(vec![event.clone()]);
        let result = source.next_event().await.unwrap();
        assert_eq!(result, Some(event));
    }

    #[tokio::test]
    async fn next_event_returns_none_when_exhausted() {
        let source = FakeSource::empty();
        let result = source.next_event().await.unwrap();
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn next_event_drains_events_in_order() {
        let e1 = Event::new("fake", "first", json!(1));
        let e2 = Event::new("fake", "second", json!(2));
        let source = FakeSource::new(vec![e1.clone(), e2.clone()]);

        assert_eq!(source.next_event().await.unwrap(), Some(e1));
        assert_eq!(source.next_event().await.unwrap(), Some(e2));
        assert_eq!(source.next_event().await.unwrap(), None);
    }

    #[tokio::test]
    async fn failing_source_returns_error() {
        let source = FailingSource;
        let result = source.next_event().await;
        assert!(result.is_err());
    }

    // === Trait Object Tests ===

    #[tokio::test]
    async fn source_works_as_trait_object() {
        let event = Event::new("fake", "dyn.test", json!(null));
        let source: Box<dyn EventSource> = Box::new(FakeSource::new(vec![event.clone()]));
        assert_eq!(source.name(), "fake");
        let result = source.next_event().await.unwrap();
        assert_eq!(result, Some(event));
    }
}
