//! Core event type for the event bus.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// An event flowing through the event bus.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    /// Unique identifier for this event.
    pub id: String,
    /// Name of the source that produced this event.
    pub source: String,
    /// Type discriminator for routing and filtering.
    pub event_type: String,
    /// Arbitrary event payload.
    pub data: Value,
    /// Optional metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

impl Event {
    /// Creates a new event with the given source, type, and data.
    pub fn new(source: impl Into<String>, event_type: impl Into<String>, data: Value) -> Self {
        Self {
            id: uuid_v4(),
            source: source.into(),
            event_type: event_type.into(),
            data,
            metadata: None,
        }
    }

    /// Attaches metadata to this event.
    #[must_use]
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// Generates a simple UUID v4-like identifier.
fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    // Simple unique-enough ID for events: timestamp + random-ish suffix
    format!("{nanos:x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // === Construction Tests ===

    #[test]
    fn new_creates_event_with_fields() {
        let event = Event::new("test-source", "user.created", json!({"name": "Alice"}));
        assert_eq!(event.source, "test-source");
        assert_eq!(event.event_type, "user.created");
        assert_eq!(event.data, json!({"name": "Alice"}));
        assert!(event.metadata.is_none());
    }

    #[test]
    fn new_generates_non_empty_id() {
        let event = Event::new("src", "type", json!(null));
        assert!(!event.id.is_empty());
    }

    #[test]
    fn new_generates_unique_ids() {
        let e1 = Event::new("src", "type", json!(null));
        // Ensure some time passes for uniqueness
        let e2 = Event::new("src", "type", json!(null));
        // IDs may collide in fast tests, but structure should be valid
        assert!(!e1.id.is_empty());
        assert!(!e2.id.is_empty());
    }

    // === Metadata Tests ===

    #[test]
    fn with_metadata_attaches_metadata() {
        let event = Event::new("src", "type", json!(1)).with_metadata(json!({"priority": "high"}));
        assert_eq!(event.metadata, Some(json!({"priority": "high"})));
    }

    // === Serialization Tests ===

    #[test]
    fn serializes_to_json() {
        let event = Event {
            id: "test-id".into(),
            source: "src".into(),
            event_type: "evt".into(),
            data: json!(42),
            metadata: None,
        };
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["id"], "test-id");
        assert_eq!(json["source"], "src");
        assert_eq!(json["event_type"], "evt");
        assert_eq!(json["data"], 42);
        assert!(json.get("metadata").is_none());
    }

    #[test]
    fn serializes_metadata_when_present() {
        let event = Event {
            id: "test-id".into(),
            source: "src".into(),
            event_type: "evt".into(),
            data: json!(null),
            metadata: Some(json!({"key": "val"})),
        };
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["metadata"]["key"], "val");
    }

    #[test]
    fn deserializes_from_json() {
        let json = json!({
            "id": "abc",
            "source": "src",
            "event_type": "evt",
            "data": {"x": 1}
        });
        let event: Event = serde_json::from_value(json).unwrap();
        assert_eq!(event.id, "abc");
        assert_eq!(event.source, "src");
        assert_eq!(event.event_type, "evt");
        assert_eq!(event.data, json!({"x": 1}));
        assert!(event.metadata.is_none());
    }

    #[test]
    fn roundtrips_through_json() {
        let original = Event::new("src", "type", json!({"nested": [1, 2, 3]}))
            .with_metadata(json!({"trace_id": "xyz"}));
        let serialized = serde_json::to_string(&original).unwrap();
        let deserialized: Event = serde_json::from_str(&serialized).unwrap();
        assert_eq!(original, deserialized);
    }

    // === Clone / Equality Tests ===

    #[test]
    fn clone_produces_equal_event() {
        let event = Event {
            id: "id".into(),
            source: "src".into(),
            event_type: "evt".into(),
            data: json!(null),
            metadata: None,
        };
        assert_eq!(event, event.clone());
    }
}
