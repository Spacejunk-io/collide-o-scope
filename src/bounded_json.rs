//! Dependency-light, duplicate-key-rejecting JSON parser with caller limits.

use std::fmt;

use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JsonLimits {
    pub max_bytes: usize,
    pub max_depth: usize,
    pub max_nodes: usize,
    pub max_key_bytes: usize,
    pub max_string_bytes: usize,
}

#[derive(Debug)]
pub enum BoundedJsonError {
    OverBytes { observed: usize, limit: usize },
    Json(serde_json::Error),
}

impl fmt::Display for BoundedJsonError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OverBytes { observed, limit } => {
                write!(formatter, "JSON is {observed} bytes; limit is {limit}")
            }
            Self::Json(error) => write!(formatter, "bounded JSON: {error}"),
        }
    }
}

impl std::error::Error for BoundedJsonError {}

struct ParseBudget {
    remaining_nodes: usize,
}

impl ParseBudget {
    fn consume<E: de::Error>(&mut self, limit: usize) -> Result<(), E> {
        if self.remaining_nodes == 0 {
            return Err(E::custom(format_args!("JSON exceeds {limit} nodes")));
        }
        self.remaining_nodes -= 1;
        Ok(())
    }
}

struct BoundedValueSeed<'a> {
    depth: usize,
    budget: &'a mut ParseBudget,
    limits: JsonLimits,
}

impl<'de> DeserializeSeed<'de> for BoundedValueSeed<'_> {
    type Value = serde_json::Value;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if self.depth > self.limits.max_depth {
            return Err(de::Error::custom(format_args!(
                "JSON exceeds depth {}",
                self.limits.max_depth
            )));
        }
        self.budget.consume(self.limits.max_nodes)?;
        deserializer.deserialize_any(BoundedValueVisitor {
            depth: self.depth,
            budget: self.budget,
            limits: self.limits,
        })
    }
}

struct BoundedValueVisitor<'a> {
    depth: usize,
    budget: &'a mut ParseBudget,
    limits: JsonLimits,
}

impl BoundedValueVisitor<'_> {
    fn bounded_string<E: de::Error>(&self, value: &str) -> Result<String, E> {
        if value.len() > self.limits.max_string_bytes {
            return Err(E::custom(format_args!(
                "JSON string is {} bytes; limit is {}",
                value.len(),
                self.limits.max_string_bytes
            )));
        }
        Ok(value.to_owned())
    }
}

impl<'de> Visitor<'de> for BoundedValueVisitor<'_> {
    type Value = serde_json::Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("bounded JSON")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(serde_json::Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(serde_json::Value::Number(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(serde_json::Value::Number(value.into()))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        serde_json::Number::from_f64(value)
            .map(serde_json::Value::Number)
            .ok_or_else(|| E::custom("JSON number must be finite"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.bounded_string(value).map(serde_json::Value::String)
    }

    fn visit_borrowed_str<E>(self, value: &'de str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_str(value)
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if value.len() > self.limits.max_string_bytes {
            return Err(E::custom(format_args!(
                "JSON string is {} bytes; limit is {}",
                value.len(),
                self.limits.max_string_bytes
            )));
        }
        Ok(serde_json::Value::String(value))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(serde_json::Value::Null)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(serde_json::Value::Null)
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        BoundedValueSeed {
            depth: self.depth + 1,
            budget: self.budget,
            limits: self.limits,
        }
        .deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element_seed(BoundedValueSeed {
            depth: self.depth + 1,
            budget: self.budget,
            limits: self.limits,
        })? {
            values.push(value);
        }
        Ok(serde_json::Value::Array(values))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = serde_json::Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if key.is_empty()
                || key.len() > self.limits.max_key_bytes
                || key.chars().any(char::is_control)
            {
                return Err(de::Error::custom("JSON object key is invalid or over cap"));
            }
            if values.contains_key(&key) {
                return Err(de::Error::custom(format_args!(
                    "duplicate JSON object key '{key}'"
                )));
            }
            let value = map.next_value_seed(BoundedValueSeed {
                depth: self.depth + 1,
                budget: self.budget,
                limits: self.limits,
            })?;
            values.insert(key, value);
        }
        Ok(serde_json::Value::Object(values))
    }
}

pub fn parse_bounded_json(
    bytes: &[u8],
    limits: JsonLimits,
) -> Result<serde_json::Value, BoundedJsonError> {
    if bytes.len() > limits.max_bytes {
        return Err(BoundedJsonError::OverBytes {
            observed: bytes.len(),
            limit: limits.max_bytes,
        });
    }
    let mut budget = ParseBudget {
        remaining_nodes: limits.max_nodes,
    };
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let value = BoundedValueSeed {
        depth: 0,
        budget: &mut budget,
        limits,
    }
    .deserialize(&mut deserializer)
    .map_err(BoundedJsonError::Json)?;
    deserializer.end().map_err(BoundedJsonError::Json)?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_LIMITS: JsonLimits = JsonLimits {
        max_bytes: 1_024,
        max_depth: 4,
        max_nodes: 16,
        max_key_bytes: 16,
        max_string_bytes: 32,
    };

    #[test]
    fn duplicate_keys_and_every_bound_fail_closed() {
        assert!(parse_bounded_json(br#"{"a":1,"a":2}"#, TEST_LIMITS).is_err());
        assert!(parse_bounded_json(br#"[[[[[0]]]]]"#, TEST_LIMITS).is_err());
        assert!(parse_bounded_json(&vec![b' '; 1_025], TEST_LIMITS).is_err());
        assert!(parse_bounded_json(br#"{"a":"valid"}"#, TEST_LIMITS).is_ok());
    }
}
