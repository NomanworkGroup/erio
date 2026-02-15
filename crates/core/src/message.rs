//! Message types for LLM conversations.

use serde::{Deserialize, Serialize};

/// Role of a message participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// User/human message.
    User,
    /// Assistant/AI response.
    Assistant,
    /// System instructions.
    System,
    /// Tool execution result.
    Tool,
}

/// A tool call request from the assistant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call.
    pub id: String,
    /// Name of the tool to invoke.
    pub name: String,
    /// Arguments to pass to the tool.
    pub arguments: serde_json::Value,
}

/// Content within a message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    /// Plain text content.
    Text {
        /// The text value.
        text: String,
    },
    /// A tool call request.
    ToolCall(ToolCall),
}

impl Content {
    /// Returns the text content if this is a text variant.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text } => Some(text),
            Self::ToolCall(_) => None,
        }
    }
}

/// A message in a conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Role of the message sender.
    pub role: Role,
    /// Content blocks in this message.
    pub content: Vec<Content>,
    /// ID of the tool call this message responds to (for Tool role).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    /// Creates a user message with text content.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![Content::Text { text: text.into() }],
            tool_call_id: None,
        }
    }

    /// Creates an assistant message with text content.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![Content::Text { text: text.into() }],
            tool_call_id: None,
        }
    }

    /// Creates a system message with text content.
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: vec![Content::Text { text: text.into() }],
            tool_call_id: None,
        }
    }

    /// Creates a tool result message.
    pub fn tool_result(call_id: impl Into<String>, result: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: vec![Content::Text {
                text: result.into(),
            }],
            tool_call_id: Some(call_id.into()),
        }
    }

    /// Returns the first text content in this message.
    pub fn text(&self) -> Option<&str> {
        self.content.iter().find_map(Content::as_text)
    }

    /// Returns the tool call ID if this is a tool result message.
    pub fn tool_call_id(&self) -> Option<&str> {
        self.tool_call_id.as_deref()
    }

    /// Returns an iterator over tool calls in this message.
    pub fn tool_calls(&self) -> impl Iterator<Item = &ToolCall> {
        self.content.iter().filter_map(|c| match c {
            Content::ToolCall(tc) => Some(tc),
            Content::Text { .. } => None,
        })
    }

    /// Returns true if the message has no content.
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Role Tests ===

    #[test]
    fn role_serializes_to_lowercase() {
        assert_eq!(serde_json::to_string(&Role::User).unwrap(), "\"user\"");
        assert_eq!(
            serde_json::to_string(&Role::Assistant).unwrap(),
            "\"assistant\""
        );
        assert_eq!(serde_json::to_string(&Role::System).unwrap(), "\"system\"");
        assert_eq!(serde_json::to_string(&Role::Tool).unwrap(), "\"tool\"");
    }

    #[test]
    fn role_deserializes_from_lowercase() {
        assert_eq!(
            serde_json::from_str::<Role>("\"user\"").unwrap(),
            Role::User
        );
        assert_eq!(
            serde_json::from_str::<Role>("\"assistant\"").unwrap(),
            Role::Assistant
        );
        assert_eq!(
            serde_json::from_str::<Role>("\"system\"").unwrap(),
            Role::System
        );
        assert_eq!(
            serde_json::from_str::<Role>("\"tool\"").unwrap(),
            Role::Tool
        );
    }

    // === Message Construction Tests ===

    #[test]
    fn message_user_creates_user_message() {
        let msg = Message::user("Hello");

        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.text(), Some("Hello"));
    }

    #[test]
    fn message_assistant_creates_assistant_message() {
        let msg = Message::assistant("Hi there");

        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.text(), Some("Hi there"));
    }

    #[test]
    fn message_system_creates_system_message() {
        let msg = Message::system("You are helpful");

        assert_eq!(msg.role, Role::System);
        assert_eq!(msg.text(), Some("You are helpful"));
    }

    #[test]
    fn message_tool_result_creates_tool_message() {
        let msg = Message::tool_result("call_123", "result data");

        assert_eq!(msg.role, Role::Tool);
        assert_eq!(msg.tool_call_id(), Some("call_123"));
        assert_eq!(msg.text(), Some("result data"));
    }

    // === Content Tests ===

    #[test]
    fn content_text_returns_text() {
        let content = Content::Text {
            text: "hello".into(),
        };
        assert_eq!(content.as_text(), Some("hello"));
    }

    #[test]
    fn content_tool_call_returns_none_for_text() {
        let content = Content::ToolCall(ToolCall {
            id: "id".into(),
            name: "shell".into(),
            arguments: serde_json::json!({}),
        });
        assert_eq!(content.as_text(), None);
    }

    // === ToolCall Tests ===

    #[test]
    fn tool_call_serializes_correctly() {
        let call = ToolCall {
            id: "call_abc123".into(),
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "/tmp/test.txt"}),
        };

        let json = serde_json::to_value(&call).unwrap();

        assert_eq!(json["id"], "call_abc123");
        assert_eq!(json["name"], "read_file");
        assert_eq!(json["arguments"]["path"], "/tmp/test.txt");
    }

    #[test]
    fn tool_call_deserializes_correctly() {
        let json = serde_json::json!({
            "id": "call_xyz",
            "name": "shell",
            "arguments": {"command": "ls -la"}
        });

        let call: ToolCall = serde_json::from_value(json).unwrap();

        assert_eq!(call.id, "call_xyz");
        assert_eq!(call.name, "shell");
        assert_eq!(call.arguments["command"], "ls -la");
    }

    // === Message Serde Tests ===

    #[test]
    fn message_text_serde_roundtrip() {
        let original = Message::user("Test message");

        let json = serde_json::to_string(&original).unwrap();
        let deserialized: Message = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.role, original.role);
        assert_eq!(deserialized.text(), original.text());
    }

    #[test]
    fn message_with_tool_calls_serde_roundtrip() {
        let original = Message {
            role: Role::Assistant,
            content: vec![
                Content::Text {
                    text: "I'll help you with that.".into(),
                },
                Content::ToolCall(ToolCall {
                    id: "call_1".into(),
                    name: "shell".into(),
                    arguments: serde_json::json!({"command": "pwd"}),
                }),
            ],
            tool_call_id: None,
        };

        let json = serde_json::to_string(&original).unwrap();
        let deserialized: Message = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.role, Role::Assistant);
        assert_eq!(deserialized.content.len(), 2);
    }

    // === Message Accessor Tests ===

    #[test]
    fn message_text_returns_first_text_content() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![
                Content::Text {
                    text: "First".into(),
                },
                Content::Text {
                    text: "Second".into(),
                },
            ],
            tool_call_id: None,
        };

        assert_eq!(msg.text(), Some("First"));
    }

    #[test]
    fn message_text_returns_none_when_no_text() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![Content::ToolCall(ToolCall {
                id: "id".into(),
                name: "test".into(),
                arguments: serde_json::json!({}),
            })],
            tool_call_id: None,
        };

        assert_eq!(msg.text(), None);
    }

    #[test]
    fn message_tool_calls_returns_all_tool_calls() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![
                Content::Text {
                    text: "Let me help".into(),
                },
                Content::ToolCall(ToolCall {
                    id: "call_1".into(),
                    name: "shell".into(),
                    arguments: serde_json::json!({}),
                }),
                Content::ToolCall(ToolCall {
                    id: "call_2".into(),
                    name: "read_file".into(),
                    arguments: serde_json::json!({}),
                }),
            ],
            tool_call_id: None,
        };

        let calls: Vec<_> = msg.tool_calls().collect();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[1].name, "read_file");
    }

    #[test]
    fn message_is_empty_when_no_content() {
        let msg = Message {
            role: Role::User,
            content: vec![],
            tool_call_id: None,
        };

        assert!(msg.is_empty());
    }

    #[test]
    fn message_is_not_empty_with_content() {
        let msg = Message::user("hello");
        assert!(!msg.is_empty());
    }
}
