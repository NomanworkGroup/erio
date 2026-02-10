//! Error types for the event bus.

use thiserror::Error;

/// Errors specific to event bus operations.
#[derive(Debug, Error)]
pub enum EventBusError {
    #[error("Source error: {0}")]
    Source(String),

    #[error("Channel closed")]
    ChannelClosed,
}

impl EventBusError {
    /// Returns `true` if the error is potentially transient.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::Source(_))
    }
}

impl From<EventBusError> for erio_core::CoreError {
    fn from(err: EventBusError) -> Self {
        Self::EventBus {
            message: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_error_displays_message() {
        let err = EventBusError::Source("notify failed".into());
        assert_eq!(err.to_string(), "Source error: notify failed");
    }

    #[test]
    fn channel_closed_displays_message() {
        let err = EventBusError::ChannelClosed;
        assert_eq!(err.to_string(), "Channel closed");
    }

    #[test]
    fn source_error_is_retryable() {
        let err = EventBusError::Source("transient".into());
        assert!(err.is_retryable());
    }

    #[test]
    fn channel_closed_is_not_retryable() {
        let err = EventBusError::ChannelClosed;
        assert!(!err.is_retryable());
    }

    #[test]
    fn converts_to_core_error() {
        let err = EventBusError::Source("test".into());
        let core_err: erio_core::CoreError = err.into();
        assert!(matches!(core_err, erio_core::CoreError::EventBus { .. }));
    }

    #[test]
    fn conversion_preserves_message() {
        let err = EventBusError::ChannelClosed;
        let core_err: erio_core::CoreError = err.into();
        match core_err {
            erio_core::CoreError::EventBus { message } => {
                assert_eq!(message, "Channel closed");
            }
            _ => panic!("Expected CoreError::EventBus"),
        }
    }
}
