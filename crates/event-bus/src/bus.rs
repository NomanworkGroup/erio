//! Event bus that aggregates sources and distributes events to subscribers.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{Mutex, broadcast};

use crate::error::EventBusError;
use crate::event::Event;
use crate::source::EventSource;

/// Default channel capacity for the event broadcast channel.
const DEFAULT_CHANNEL_CAPACITY: usize = 256;

/// An event bus that manages sources and broadcasts events to subscribers.
pub struct EventBus {
    sources: HashMap<String, Box<dyn EventSource>>,
    sender: broadcast::Sender<Event>,
    running: Arc<Mutex<bool>>,
    capacity: usize,
}

impl EventBus {
    /// Creates a new event bus with default channel capacity.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CHANNEL_CAPACITY)
    }

    /// Creates a new event bus with the specified channel capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self {
            sources: HashMap::new(),
            sender,
            running: Arc::new(Mutex::new(false)),
            capacity,
        }
    }

    /// Registers an event source.
    pub fn register<S: EventSource + 'static>(&mut self, source: S) {
        let name = source.name().to_string();
        self.sources.insert(name, Box::new(source));
    }

    /// Returns `true` if a source with the given name is registered.
    pub fn has_source(&self, name: &str) -> bool {
        self.sources.contains_key(name)
    }

    /// Returns the names of all registered sources.
    pub fn source_names(&self) -> Vec<&str> {
        self.sources.keys().map(String::as_str).collect()
    }

    /// Returns the number of registered sources.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Returns the channel capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Subscribes to events from the bus.
    ///
    /// Returns a receiver that will get all events broadcast by the bus.
    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.sender.subscribe()
    }

    /// Returns `true` if the bus is currently running.
    pub async fn is_running(&self) -> bool {
        *self.running.lock().await
    }

    /// Publishes an event directly to all subscribers.
    pub fn publish(&self, event: Event) -> Result<(), EventBusError> {
        self.sender
            .send(event)
            .map(|_| ())
            .map_err(|_| EventBusError::ChannelClosed)
    }

    /// Polls all registered sources once and broadcasts any events found.
    ///
    /// Returns the number of events collected.
    pub async fn poll_sources(&self) -> Result<usize, EventBusError> {
        let mut count = 0;
        for source in self.sources.values() {
            match source.next_event().await {
                Ok(Some(event)) => {
                    // Ignore send errors if no subscribers are listening
                    let _ = self.sender.send(event);
                    count += 1;
                }
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!(source = source.name(), error = %e, "source poll failed");
                }
            }
        }
        Ok(count)
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // === Fake Source ===

    struct FakeSource {
        source_name: String,
        events: tokio::sync::Mutex<Vec<Event>>,
    }

    impl FakeSource {
        fn new(name: &str, events: Vec<Event>) -> Self {
            Self {
                source_name: name.to_string(),
                events: tokio::sync::Mutex::new(events),
            }
        }

        fn empty(name: &str) -> Self {
            Self::new(name, vec![])
        }
    }

    #[async_trait::async_trait]
    impl EventSource for FakeSource {
        fn name(&self) -> &str {
            &self.source_name
        }

        #[allow(clippy::unnecessary_literal_bound)]
        fn description(&self) -> &str {
            "Fake source"
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

    // === Construction Tests ===

    #[test]
    fn new_creates_empty_bus() {
        let bus = EventBus::new();
        assert_eq!(bus.source_count(), 0);
        assert!(bus.source_names().is_empty());
    }

    #[test]
    fn default_creates_empty_bus() {
        let bus = EventBus::default();
        assert_eq!(bus.source_count(), 0);
    }

    #[test]
    fn with_capacity_sets_capacity() {
        let bus = EventBus::with_capacity(64);
        assert_eq!(bus.capacity(), 64);
    }

    // === Registration Tests ===

    #[test]
    fn register_adds_source() {
        let mut bus = EventBus::new();
        bus.register(FakeSource::empty("test"));
        assert!(bus.has_source("test"));
        assert_eq!(bus.source_count(), 1);
    }

    #[test]
    fn register_multiple_sources() {
        let mut bus = EventBus::new();
        bus.register(FakeSource::empty("a"));
        bus.register(FakeSource::empty("b"));
        assert_eq!(bus.source_count(), 2);
        assert!(bus.has_source("a"));
        assert!(bus.has_source("b"));
    }

    #[test]
    fn register_replaces_existing_source() {
        let mut bus = EventBus::new();
        bus.register(FakeSource::empty("src"));
        bus.register(FakeSource::empty("src"));
        assert_eq!(bus.source_count(), 1);
    }

    #[test]
    fn has_source_returns_false_for_unknown() {
        let bus = EventBus::new();
        assert!(!bus.has_source("nonexistent"));
    }

    // === Publish / Subscribe Tests ===

    #[tokio::test]
    async fn publish_sends_event_to_subscriber() {
        let bus = EventBus::new();
        let mut rx = bus.subscribe();
        let event = Event::new("src", "test", json!(42));
        bus.publish(event.clone()).unwrap();
        let received = rx.recv().await.unwrap();
        assert_eq!(received, event);
    }

    #[tokio::test]
    async fn publish_sends_to_multiple_subscribers() {
        let bus = EventBus::new();
        let mut rx1 = bus.subscribe();
        let mut rx2 = bus.subscribe();
        let event = Event::new("src", "test", json!(null));
        bus.publish(event.clone()).unwrap();
        assert_eq!(rx1.recv().await.unwrap(), event);
        assert_eq!(rx2.recv().await.unwrap(), event);
    }

    // === Poll Sources Tests ===

    #[tokio::test]
    async fn poll_sources_returns_zero_with_no_sources() {
        let bus = EventBus::new();
        let count = bus.poll_sources().await.unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn poll_sources_collects_event_from_source() {
        let mut bus = EventBus::new();
        let event = Event::new("fake", "polled", json!("data"));
        bus.register(FakeSource::new("fake", vec![event.clone()]));
        let mut rx = bus.subscribe();
        let count = bus.poll_sources().await.unwrap();
        assert_eq!(count, 1);
        let received = rx.recv().await.unwrap();
        assert_eq!(received, event);
    }

    #[tokio::test]
    async fn poll_sources_returns_zero_when_sources_empty() {
        let mut bus = EventBus::new();
        bus.register(FakeSource::empty("empty"));
        let count = bus.poll_sources().await.unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn poll_sources_collects_from_multiple_sources() {
        let mut bus = EventBus::new();
        let e1 = Event::new("src-a", "type", json!(1));
        let e2 = Event::new("src-b", "type", json!(2));
        bus.register(FakeSource::new("src-a", vec![e1]));
        bus.register(FakeSource::new("src-b", vec![e2]));
        let _rx = bus.subscribe();
        let count = bus.poll_sources().await.unwrap();
        assert_eq!(count, 2);
    }

    // === Running State Tests ===

    #[tokio::test]
    async fn initially_not_running() {
        let bus = EventBus::new();
        assert!(!bus.is_running().await);
    }

    // === Source Names Tests ===

    #[test]
    fn source_names_returns_registered_names() {
        let mut bus = EventBus::new();
        bus.register(FakeSource::empty("alpha"));
        bus.register(FakeSource::empty("beta"));
        let mut names = bus.source_names();
        names.sort_unstable();
        assert_eq!(names, vec!["alpha", "beta"]);
    }
}
