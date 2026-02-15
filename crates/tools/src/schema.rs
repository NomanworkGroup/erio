//! Tool schema types for JSON Schema generation.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;

/// JSON Schema property types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PropertyType {
    String,
    Integer,
    Number,
    Boolean,
    Array,
    Object,
}

impl PropertyType {
    fn as_str(self) -> &'static str {
        match self {
            Self::String => "string",
            Self::Integer => "integer",
            Self::Number => "number",
            Self::Boolean => "boolean",
            Self::Array => "array",
            Self::Object => "object",
        }
    }
}

/// A property definition in a tool schema.
#[derive(Debug, Clone)]
struct Property {
    prop_type: PropertyType,
    description: String,
    required: bool,
}

/// Schema describing tool parameters as JSON Schema.
#[derive(Debug, Clone, Default)]
pub struct ToolSchema {
    properties: HashMap<String, Property>,
    property_order: Vec<String>,
}

impl ToolSchema {
    /// Creates a new schema builder.
    pub fn builder() -> ToolSchemaBuilder {
        ToolSchemaBuilder::default()
    }

    /// Returns `true` if the schema has the given property.
    pub fn has_property(&self, name: &str) -> bool {
        self.properties.contains_key(name)
    }

    /// Returns `true` if the given property is required.
    pub fn is_required(&self, name: &str) -> bool {
        self.properties.get(name).is_some_and(|p| p.required)
    }

    /// Returns the number of properties in the schema.
    pub fn property_count(&self) -> usize {
        self.properties.len()
    }

    /// Converts the schema to a JSON Schema object.
    pub fn to_json_schema(&self) -> Value {
        let mut properties = json!({});
        let mut required = Vec::new();

        for name in &self.property_order {
            if let Some(prop) = self.properties.get(name) {
                properties[name] = json!({
                    "type": prop.prop_type.as_str(),
                    "description": prop.description,
                });
                if prop.required {
                    required.push(name.clone());
                }
            }
        }

        json!({
            "type": "object",
            "properties": properties,
            "required": required,
        })
    }
}

/// Builder for constructing tool schemas.
#[derive(Debug, Default)]
pub struct ToolSchemaBuilder {
    properties: HashMap<String, Property>,
    property_order: Vec<String>,
}

impl ToolSchemaBuilder {
    /// Adds a property to the schema.
    #[must_use]
    pub fn property(
        mut self,
        name: impl Into<String>,
        prop_type: PropertyType,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.property_order.push(name.clone());
        self.properties.insert(
            name,
            Property {
                prop_type,
                description: description.into(),
                required,
            },
        );
        self
    }

    /// Builds the schema.
    #[must_use]
    pub fn build(self) -> ToolSchema {
        ToolSchema {
            properties: self.properties,
            property_order: self.property_order,
        }
    }
}
