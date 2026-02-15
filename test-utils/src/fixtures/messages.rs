//! Pre-built message fixtures for testing.

use erio_core::{Content, Message, Role, ToolCall};

/// Creates a simple user message.
pub fn user_message(text: &str) -> Message {
    Message::user(text)
}

/// Creates a simple assistant message.
pub fn assistant_message(text: &str) -> Message {
    Message::assistant(text)
}

/// Creates a system message.
pub fn system_message(text: &str) -> Message {
    Message::system(text)
}

/// Creates a tool result message.
pub fn tool_result_message(call_id: &str, result: &str) -> Message {
    Message::tool_result(call_id, result)
}

/// Creates an assistant message with a tool call.
pub fn assistant_with_tool_call(text: &str, tool_name: &str, call_id: &str) -> Message {
    Message {
        role: Role::Assistant,
        content: vec![
            Content::Text { text: text.into() },
            Content::ToolCall(ToolCall {
                id: call_id.into(),
                name: tool_name.into(),
                arguments: serde_json::json!({}),
            }),
        ],
        tool_call_id: None,
    }
}

/// Creates a conversation with a simple exchange.
pub fn simple_conversation() -> Vec<Message> {
    vec![
        system_message("You are a helpful assistant."),
        user_message("Hello!"),
        assistant_message("Hi there! How can I help you today?"),
    ]
}

/// Creates a conversation with tool usage.
pub fn conversation_with_tool_use() -> Vec<Message> {
    vec![
        system_message("You are a helpful assistant with tool access."),
        user_message("What files are in the current directory?"),
        assistant_with_tool_call("I'll check that for you.", "shell", "call_001"),
        tool_result_message("call_001", "file1.txt\nfile2.txt\nREADME.md"),
        assistant_message("The current directory contains: file1.txt, file2.txt, and README.md"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_message_creates_user_role() {
        let msg = user_message("test");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.text(), Some("test"));
    }

    #[test]
    fn assistant_message_creates_assistant_role() {
        let msg = assistant_message("test");
        assert_eq!(msg.role, Role::Assistant);
    }

    #[test]
    fn system_message_creates_system_role() {
        let msg = system_message("test");
        assert_eq!(msg.role, Role::System);
    }

    #[test]
    fn tool_result_has_call_id() {
        let msg = tool_result_message("call_123", "result");
        assert_eq!(msg.tool_call_id(), Some("call_123"));
    }

    #[test]
    fn assistant_with_tool_call_has_tool_call() {
        let msg = assistant_with_tool_call("text", "shell", "call_1");
        let calls: Vec<_> = msg.tool_calls().collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
    }

    #[test]
    fn simple_conversation_has_three_messages() {
        let conv = simple_conversation();
        assert_eq!(conv.len(), 3);
    }

    #[test]
    fn conversation_with_tool_use_has_five_messages() {
        let conv = conversation_with_tool_use();
        assert_eq!(conv.len(), 5);
    }
}
