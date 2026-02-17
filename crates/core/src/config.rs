//! Configuration types for the Erio agent runtime.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Configuration for retry behavior with exponential backoff.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of attempts (including the initial attempt).
    pub max_attempts: u32,

    /// Initial delay before the first retry.
    #[serde(
        rename = "initial_delay_ms",
        serialize_with = "serialize_duration_ms",
        deserialize_with = "deserialize_duration_ms"
    )]
    pub initial_delay: Duration,

    /// Maximum delay between retries.
    #[serde(
        rename = "max_delay_ms",
        serialize_with = "serialize_duration_ms",
        deserialize_with = "deserialize_duration_ms"
    )]
    pub max_delay: Duration,

    /// Multiplier for exponential backoff.
    pub multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            multiplier: 2.0,
        }
    }
}

impl RetryConfig {
    /// Creates a new builder with default values.
    pub fn builder() -> RetryConfigBuilder {
        RetryConfigBuilder::default()
    }

    /// Creates a config that performs no retries (single attempt).
    pub fn no_retry() -> Self {
        Self {
            max_attempts: 1,
            ..Default::default()
        }
    }

    /// Calculates the delay for a given attempt number (0-indexed).
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let delay_ms = self.initial_delay.as_millis() as f64 * self.multiplier.powi(attempt as i32);
        let delay = Duration::from_millis(delay_ms as u64);
        delay.min(self.max_delay)
    }
}

/// Builder for `RetryConfig`.
#[derive(Debug, Default)]
#[must_use]
pub struct RetryConfigBuilder {
    config: RetryConfig,
}

impl RetryConfigBuilder {
    /// Sets the maximum number of attempts.
    pub fn max_attempts(mut self, n: u32) -> Self {
        self.config.max_attempts = n;
        self
    }

    /// Sets the initial delay before the first retry.
    pub fn initial_delay(mut self, delay: Duration) -> Self {
        self.config.initial_delay = delay;
        self
    }

    /// Sets the maximum delay between retries.
    pub fn max_delay(mut self, delay: Duration) -> Self {
        self.config.max_delay = delay;
        self
    }

    /// Sets the multiplier for exponential backoff.
    pub fn multiplier(mut self, m: f64) -> Self {
        self.config.multiplier = m;
        self
    }

    /// Builds the `RetryConfig`.
    pub fn build(self) -> RetryConfig {
        self.config
    }
}

#[allow(clippy::cast_possible_truncation)]
fn serialize_duration_ms<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_u64(duration.as_millis() as u64)
}

fn deserialize_duration_ms<'de, D>(deserializer: D) -> Result<Duration, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let ms = u64::deserialize(deserializer)?;
    Ok(Duration::from_millis(ms))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // === RetryConfig Tests ===

    #[test]
    fn retry_config_default_values() {
        let config = RetryConfig::default();

        assert_eq!(config.max_attempts, 3);
        assert_eq!(config.initial_delay, Duration::from_millis(100));
        assert_eq!(config.max_delay, Duration::from_secs(10));
        assert!((config.multiplier - 2.0_f64).abs() < f64::EPSILON);
    }

    #[test]
    fn retry_config_builder_sets_max_attempts() {
        let config = RetryConfig::builder().max_attempts(5).build();

        assert_eq!(config.max_attempts, 5);
    }

    #[test]
    fn retry_config_builder_sets_initial_delay() {
        let config = RetryConfig::builder()
            .initial_delay(Duration::from_millis(500))
            .build();

        assert_eq!(config.initial_delay, Duration::from_millis(500));
    }

    #[test]
    fn retry_config_builder_sets_max_delay() {
        let config = RetryConfig::builder()
            .max_delay(Duration::from_secs(30))
            .build();

        assert_eq!(config.max_delay, Duration::from_secs(30));
    }

    #[test]
    fn retry_config_builder_sets_multiplier() {
        let config = RetryConfig::builder().multiplier(1.5).build();

        assert!((config.multiplier - 1.5_f64).abs() < f64::EPSILON);
    }

    #[test]
    fn retry_config_delay_for_attempt_increases_exponentially() {
        let config = RetryConfig::builder()
            .initial_delay(Duration::from_millis(100))
            .multiplier(2.0)
            .max_delay(Duration::from_secs(10))
            .build();

        // Attempt 0: 100ms
        assert_eq!(config.delay_for_attempt(0), Duration::from_millis(100));
        // Attempt 1: 200ms
        assert_eq!(config.delay_for_attempt(1), Duration::from_millis(200));
        // Attempt 2: 400ms
        assert_eq!(config.delay_for_attempt(2), Duration::from_millis(400));
        // Attempt 3: 800ms
        assert_eq!(config.delay_for_attempt(3), Duration::from_millis(800));
    }

    #[test]
    fn retry_config_delay_capped_at_max() {
        let config = RetryConfig::builder()
            .initial_delay(Duration::from_secs(1))
            .multiplier(10.0)
            .max_delay(Duration::from_secs(5))
            .build();

        // Attempt 0: 1s
        assert_eq!(config.delay_for_attempt(0), Duration::from_secs(1));
        // Attempt 1: 10s but capped at 5s
        assert_eq!(config.delay_for_attempt(1), Duration::from_secs(5));
        // Attempt 2: still capped at 5s
        assert_eq!(config.delay_for_attempt(2), Duration::from_secs(5));
    }

    #[test]
    fn retry_config_no_retry_returns_single_attempt() {
        let config = RetryConfig::no_retry();

        assert_eq!(config.max_attempts, 1);
    }

    // === Serde Tests ===

    #[test]
    fn retry_config_serializes_to_json() {
        let config = RetryConfig::builder()
            .max_attempts(5)
            .initial_delay(Duration::from_millis(200))
            .build();

        let json = serde_json::to_value(&config).unwrap();

        assert_eq!(json["max_attempts"], 5);
        assert_eq!(json["initial_delay_ms"], 200);
    }

    #[test]
    fn retry_config_deserializes_from_json() {
        let json = serde_json::json!({
            "max_attempts": 4,
            "initial_delay_ms": 500,
            "max_delay_ms": 30000,
            "multiplier": 1.5
        });

        let config: RetryConfig = serde_json::from_value(json).unwrap();

        assert_eq!(config.max_attempts, 4);
        assert_eq!(config.initial_delay, Duration::from_millis(500));
        assert_eq!(config.max_delay, Duration::from_secs(30));
        assert!((config.multiplier - 1.5_f64).abs() < f64::EPSILON);
    }

    #[test]
    fn retry_config_serde_roundtrip() {
        let original = RetryConfig::builder()
            .max_attempts(7)
            .initial_delay(Duration::from_millis(250))
            .max_delay(Duration::from_secs(60))
            .multiplier(3.0)
            .build();

        let json = serde_json::to_string(&original).unwrap();
        let deserialized: RetryConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(original, deserialized);
    }
}
