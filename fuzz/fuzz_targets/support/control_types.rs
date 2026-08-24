//! Compile-only host identity types for the dependency-light controller/OSC
//! fuzz binaries. Their serde laws mirror the production newtypes; the parser
//! and decoder implementations themselves are included verbatim from `src/`.

#![allow(dead_code)]

use serde::de;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::ffi::OsStr;
use std::path::PathBuf;

macro_rules! nonzero_id {
    ($name:ident, $wire:ty, $message:literal) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name($wire);

        impl $name {
            pub const fn new(value: $wire) -> Option<Self> {
                if value == 0 {
                    None
                } else {
                    Some(Self(value))
                }
            }

            pub const fn get(self) -> $wire {
                self.0
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                self.0.serialize(serializer)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = <$wire>::deserialize(deserializer)?;
                Self::new(value).ok_or_else(|| de::Error::custom($message))
            }
        }
    };
}

nonzero_id!(StableLayerId, u64, "stable layer id must be non-zero");
nonzero_id!(SceneId, u16, "scene id must be non-zero");
nonzero_id!(GroupId, u64, "group id must be non-zero");
nonzero_id!(NodeId, u64, "visual node id must be non-zero");

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SavedLayerPosition(u32);

impl SavedLayerPosition {
    pub const fn new(value: u32) -> Option<Self> {
        if value <= 4095 {
            Some(Self(value))
        } else {
            None
        }
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

impl Serialize for SavedLayerPosition {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SavedLayerPosition {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u32::deserialize(deserializer)?;
        Self::new(value)
            .ok_or_else(|| de::Error::custom("saved layer position must be no greater than 4095"))
    }
}

pub fn state_root_from(
    local_app_data: Option<&OsStr>,
    xdg_state_home: Option<&OsStr>,
    home: Option<&OsStr>,
) -> PathBuf {
    if let Some(base) = local_app_data {
        return PathBuf::from(base).join("collide-o-scope");
    }
    if let Some(base) = xdg_state_home {
        return PathBuf::from(base).join("collide-o-scope");
    }
    if let Some(base) = home {
        return PathBuf::from(base)
            .join(".local")
            .join("state")
            .join("collide-o-scope");
    }
    PathBuf::from(".collide-o-scope")
}
