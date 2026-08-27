//! One-level group and A/B composition foundation.
//!
//! The complete bounded domain and CPU reference law are implemented here;
//! renderer ownership remains in the evaluated frame plan.

use std::collections::BTreeSet;
use std::fmt;

use serde::de::{self, SeqAccess, Visitor};
use serde::ser::{SerializeSeq, SerializeStruct};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::image_routing::StableLayerId;
use crate::mixing_boundary::BusMixerState;
use crate::performance::{AuthoringValueLaw, SavedLayerPosition};
use crate::spatial::{spatial_transform_value_law, SpatialTransform};
use crate::visual_rack::{
    GroupId, ImageMatte, LegacyRackScope, RackError, RouteCaptureError, RuntimeImageMatte,
    RuntimeRackError, RuntimeVisualRack, VisualRack,
};

pub const MAX_GROUPS: usize = 16;
/// Advanced group membership and planner admission ceiling. Direct-root
/// legacy stacks remain dynamically sized and are not a policy-limited feature.
pub const MAX_COMPOSITION_LAYERS: usize = 256;
/// Bounded deserialization preallocation hint, not a root admission cap.
pub const MAX_ROOT_ITEMS: usize = MAX_COMPOSITION_LAYERS + MAX_GROUPS;
pub const MAX_GROUP_NAME_BYTES: usize = 64;

/// A/B is a performance lane, not Parameter Morph's A/B state. Program items
/// are composited over the A/B crossfade result.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BusAssignment {
    A,
    B,
    #[default]
    Program,
}

impl BusAssignment {
    pub const ALL: [Self; 3] = [Self::Program, Self::A, Self::B];

    pub const fn key(self) -> &'static str {
        match self {
            Self::Program => "program",
            Self::A => "a",
            Self::B => "b",
        }
    }

    pub fn try_from_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|value| value.key() == key)
    }
}

/// Engine-owned law for the scalar/discrete group-value authoring seam.
pub(crate) fn group_value_law(param: &str) -> Option<AuthoringValueLaw> {
    match param {
        "opacity" => Some(AuthoringValueLaw::Unit([0.0, 1.0])),
        "solo" | "bypass" => Some(AuthoringValueLaw::Toggle),
        "bus" => Some(AuthoringValueLaw::Discrete(
            BusAssignment::ALL
                .into_iter()
                .map(BusAssignment::key)
                .collect(),
        )),
        "name" => None,
        _ => spatial_transform_value_law(param),
    }
}

/// Engine-owned law for the optional image-matte value seam on a group.
pub(crate) fn group_matte_value_law(param: &str) -> Option<AuthoringValueLaw> {
    match param {
        "amount" | "threshold" => Some(AuthoringValueLaw::Unit([0.0, 1.0])),
        "softness" => Some(AuthoringValueLaw::Unit([0.0, 0.5])),
        _ => None,
    }
}

/// Bounded UTF-8 display name. Identity always remains [`GroupId`].
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GroupName(String);

impl GroupName {
    pub fn new(value: impl Into<String>) -> Result<Self, CompositionError> {
        let value = value.into();
        if value.len() > MAX_GROUP_NAME_BYTES {
            return Err(CompositionError::GroupNameTooLong {
                bytes: value.len(),
                limit: MAX_GROUP_NAME_BYTES,
            });
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Serialize for GroupName {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for GroupName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct NameVisitor;

        impl Visitor<'_> for NameVisitor {
            type Value = GroupName;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    formatter,
                    "a UTF-8 group name no longer than {MAX_GROUP_NAME_BYTES} bytes"
                )
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                if value.len() > MAX_GROUP_NAME_BYTES {
                    return Err(E::custom(format_args!(
                        "group name is {} bytes; limit is {MAX_GROUP_NAME_BYTES}",
                        value.len()
                    )));
                }
                Ok(GroupName(value.to_owned()))
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                if value.len() > MAX_GROUP_NAME_BYTES {
                    return Err(E::custom(format_args!(
                        "group name is {} bytes; limit is {MAX_GROUP_NAME_BYTES}",
                        value.len()
                    )));
                }
                Ok(GroupName(value))
            }
        }

        deserializer.deserialize_string(NameVisitor)
    }
}

/// Bounded, ordered group membership. Members are layer positions—not root
/// items—so nesting is unrepresentable in the type system. Empty groups are
/// valid and remain available as transparent image taps.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GroupMembers(Vec<SavedLayerPosition>);

impl GroupMembers {
    pub fn try_from_vec(entries: Vec<SavedLayerPosition>) -> Result<Self, CompositionError> {
        if entries.len() > MAX_COMPOSITION_LAYERS {
            return Err(CompositionError::TooManyLayers {
                count: entries.len(),
                limit: MAX_COMPOSITION_LAYERS,
            });
        }
        let mut seen = BTreeSet::new();
        for layer in &entries {
            if !seen.insert(*layer) {
                return Err(CompositionError::DuplicateLayer(*layer));
            }
        }
        Ok(Self(entries))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "saved-tree inspection is exercised by patch/editor tests"
        )
    )]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = SavedLayerPosition> + '_ {
        self.0.iter().copied()
    }

    #[allow(
        dead_code,
        reason = "retained as the checked primitive for the saved-tree editor API"
    )]
    pub fn move_layer(
        &mut self,
        layer: SavedLayerPosition,
        new_index: usize,
    ) -> Result<(), CompositionError> {
        if new_index >= self.0.len() {
            return Err(CompositionError::InvalidMoveIndex {
                index: new_index,
                len: self.0.len(),
            });
        }
        let old_index = self
            .0
            .iter()
            .position(|candidate| *candidate == layer)
            .ok_or(CompositionError::UnknownLayer(layer))?;
        let entry = self.0.remove(old_index);
        self.0.insert(new_index, entry);
        Ok(())
    }
}

impl Serialize for GroupMembers {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for member in &self.0 {
            sequence.serialize_element(member)?;
        }
        sequence.end()
    }
}

impl<'de> Deserialize<'de> for GroupMembers {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct MembersVisitor;

        impl<'de> Visitor<'de> for MembersVisitor {
            type Value = GroupMembers;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    formatter,
                    "at most {MAX_COMPOSITION_LAYERS} unique saved layer positions"
                )
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut members = Vec::with_capacity(
                    sequence
                        .size_hint()
                        .unwrap_or(0)
                        .min(MAX_COMPOSITION_LAYERS),
                );
                while let Some(member) = sequence.next_element::<SavedLayerPosition>()? {
                    if members.len() == MAX_COMPOSITION_LAYERS {
                        return Err(de::Error::custom(format_args!(
                            "a group may contain at most {MAX_COMPOSITION_LAYERS} layers"
                        )));
                    }
                    members.push(member);
                }
                GroupMembers::try_from_vec(members).map_err(de::Error::custom)
            }
        }

        deserializer.deserialize_seq(MembersVisitor)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Group {
    pub id: GroupId,
    pub name: GroupName,
    pub members: GroupMembers,
    pub opacity: f32,
    pub transform: SpatialTransform,
    pub rack: VisualRack,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matte: Option<ImageMatte>,
    pub solo: bool,
    pub bypass: bool,
    pub bus: BusAssignment,
}

impl Group {
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "saved groups are allocated by patch/editor paths")
    )]
    pub fn empty(id: GroupId, name: GroupName) -> Self {
        Self {
            id,
            name,
            members: GroupMembers::default(),
            opacity: 1.0,
            transform: SpatialTransform::default(),
            rack: VisualRack::empty(),
            matte: None,
            solo: false,
            bypass: false,
            bus: BusAssignment::Program,
        }
    }

    fn sanitized(mut self) -> Result<Self, CompositionError> {
        self.opacity = finite_unit(self.opacity, 1.0);
        self.transform = self.transform.sanitized();
        self.matte = self.matte.map(ImageMatte::sanitized);
        self.rack
            .validate_for_scope(LegacyRackScope::Group)
            .map_err(|error| CompositionError::GroupRack {
                group_id: self.id,
                error,
            })?;
        Ok(self)
    }

    pub fn referenced_group_ids(&self) -> impl Iterator<Item = GroupId> + '_ {
        self.rack
            .referenced_group_ids()
            .chain(self.matte.and_then(|matte| matte.tap.referenced_group()))
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "saved-reference invalidation is an editor compatibility path"
        )
    )]
    fn mark_group_output_missing(&mut self, removed: GroupId) {
        self.rack.mark_group_output_missing(removed);
        if let Some(matte) = &mut self.matte {
            matte.mark_group_output_missing(removed);
        }
    }
}

impl<'de> Deserialize<'de> for Group {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            id: GroupId,
            #[serde(default)]
            name: GroupName,
            #[serde(default)]
            members: GroupMembers,
            #[serde(default = "one")]
            opacity: f32,
            #[serde(default)]
            transform: SpatialTransform,
            #[serde(default)]
            rack: VisualRack,
            #[serde(default)]
            matte: Option<ImageMatte>,
            #[serde(default)]
            solo: bool,
            #[serde(default)]
            bypass: bool,
            #[serde(default)]
            bus: BusAssignment,
        }

        let raw = Raw::deserialize(deserializer)?;
        Group {
            id: raw.id,
            name: raw.name,
            members: raw.members,
            opacity: raw.opacity,
            transform: raw.transform,
            rack: raw.rack,
            matte: raw.matte,
            solo: raw.solo,
            bypass: raw.bypass,
            bus: raw.bus,
        }
        .sanitized()
        .map_err(de::Error::custom)
    }
}

const fn one() -> f32 {
    1.0
}

/// Root order is back-to-front. A group occupies one root position and its
/// members form one contiguous block in flattened layer order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RootItem {
    Layer {
        layer: SavedLayerPosition,
        #[serde(default)]
        bus: BusAssignment,
    },
    Group {
        group_id: GroupId,
    },
}

impl RootItem {
    pub const fn key(self) -> RootItemKey {
        match self {
            Self::Layer { layer, .. } => RootItemKey::Layer(layer),
            Self::Group { group_id } => RootItemKey::Group(group_id),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RootItemKey {
    Layer(SavedLayerPosition),
    Group(GroupId),
}

#[derive(Debug, Clone, Default, PartialEq)]
struct Groups(Vec<Group>);

impl Serialize for Groups {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for group in &self.0 {
            sequence.serialize_element(group)?;
        }
        sequence.end()
    }
}

impl<'de> Deserialize<'de> for Groups {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct GroupsVisitor;

        impl<'de> Visitor<'de> for GroupsVisitor {
            type Value = Groups;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "at most {MAX_GROUPS} one-level groups")
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut groups =
                    Vec::with_capacity(sequence.size_hint().unwrap_or(0).min(MAX_GROUPS));
                while let Some(group) = sequence.next_element::<Group>()? {
                    if groups.len() == MAX_GROUPS {
                        return Err(de::Error::custom(format_args!(
                            "a composition may contain at most {MAX_GROUPS} groups"
                        )));
                    }
                    groups.push(group);
                }
                Ok(Groups(groups))
            }
        }

        deserializer.deserialize_seq(GroupsVisitor)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct RootItems(Vec<RootItem>);

impl Serialize for RootItems {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for item in &self.0 {
            sequence.serialize_element(item)?;
        }
        sequence.end()
    }
}

impl<'de> Deserialize<'de> for RootItems {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct RootVisitor;

        impl<'de> Visitor<'de> for RootVisitor {
            type Value = RootItems;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a finite sequence of root composition items")
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut root =
                    Vec::with_capacity(sequence.size_hint().unwrap_or(0).min(MAX_ROOT_ITEMS));
                while let Some(item) = sequence.next_element::<RootItem>()? {
                    root.push(item);
                }
                Ok(RootItems(root))
            }
        }

        deserializer.deserialize_seq(RootVisitor)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompositionError {
    TooManyGroups {
        count: usize,
        limit: usize,
    },
    TooManyLayers {
        count: usize,
        limit: usize,
    },
    GroupNameTooLong {
        bytes: usize,
        limit: usize,
    },
    DuplicateGroupId(GroupId),
    DuplicateRootItem(RootItemKey),
    DuplicateLayer(SavedLayerPosition),
    UnknownGroup(GroupId),
    UnrootedGroup(GroupId),
    UnknownLayer(SavedLayerPosition),
    MissingLayer(SavedLayerPosition),
    InvalidNextGroupId {
        next: u64,
        greatest_observed: u64,
    },
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "reported by the retained saved-tree allocation API"
        )
    )]
    GroupIdExhausted,
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "reported by the retained saved-tree reorder API")
    )]
    InvalidMoveIndex {
        index: usize,
        len: usize,
    },
    GroupRack {
        group_id: GroupId,
        error: RackError,
    },
}

impl fmt::Display for CompositionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyGroups { count, limit } => write!(
                formatter,
                "composition has {count} groups; limit is {limit}"
            ),
            Self::TooManyLayers { count, limit } => write!(
                formatter,
                "composition has {count} layers; limit is {limit}"
            ),
            Self::GroupNameTooLong { bytes, limit } => {
                write!(formatter, "group name is {bytes} bytes; limit is {limit}")
            }
            Self::DuplicateGroupId(id) => write!(formatter, "duplicate group id {}", id.get()),
            Self::DuplicateRootItem(item) => write!(formatter, "duplicate root item {item:?}"),
            Self::DuplicateLayer(layer) => write!(
                formatter,
                "layer position {} occurs more than once in composition",
                layer.get()
            ),
            Self::UnknownGroup(id) => {
                write!(formatter, "root references missing group {}", id.get())
            }
            Self::UnrootedGroup(id) => {
                write!(formatter, "group {} is not present at the root", id.get())
            }
            Self::UnknownLayer(layer) => write!(
                formatter,
                "composition references unknown layer position {}",
                layer.get()
            ),
            Self::MissingLayer(layer) => write!(
                formatter,
                "composition omits layer position {}",
                layer.get()
            ),
            Self::InvalidNextGroupId {
                next,
                greatest_observed,
            } => write!(
                formatter,
                "next group id {next} must advance past live or missing referenced id {greatest_observed}"
            ),
            Self::GroupIdExhausted => formatter.write_str("group identity space is exhausted"),
            Self::InvalidMoveIndex { index, len } => write!(
                formatter,
                "composition move index {index} exceeds length {len}"
            ),
            Self::GroupRack { group_id, error } => write!(
                formatter,
                "group {} rack is invalid: {error}",
                group_id.get()
            ),
        }
    }
}

impl std::error::Error for CompositionError {}

#[derive(Debug, Clone, PartialEq)]
pub struct CompositionTree {
    groups: Vec<Group>,
    root: Vec<RootItem>,
    /// Zero means exhausted. Otherwise this is beyond every live, deleted, or
    /// missing-referenced group ID observed by this composition.
    next_group_id: u64,
    bus_crossfade: f32,
    /// The B8 mixing boundary: wipe/blend mix laws, the dirty-mixer fault
    /// stage, and the bus melt. Values, never topology; skip-serialized at
    /// the exact-legacy default so pre-B8 patches keep their bytes.
    mixer: BusMixerState,
}

impl CompositionTree {
    pub fn legacy_for_layers(
        layers_back_to_front: &[SavedLayerPosition],
    ) -> Result<Self, CompositionError> {
        let root = layers_back_to_front
            .iter()
            .copied()
            .map(|layer| RootItem::Layer {
                layer,
                bus: BusAssignment::Program,
            })
            .collect();
        let tree = Self {
            groups: Vec::new(),
            root,
            next_group_id: 1,
            bus_crossfade: 0.5,
            mixer: BusMixerState::default(),
        };
        tree.validate_for_layers(layers_back_to_front)?;
        Ok(tree)
    }

    pub fn try_from_parts(
        groups: Vec<Group>,
        root: Vec<RootItem>,
        next_group_id: Option<u64>,
        bus_crossfade: f32,
    ) -> Result<Self, CompositionError> {
        if groups.len() > MAX_GROUPS {
            return Err(CompositionError::TooManyGroups {
                count: groups.len(),
                limit: MAX_GROUPS,
            });
        }
        let groups = groups
            .into_iter()
            .map(Group::sanitized)
            .collect::<Result<Vec<_>, _>>()?;
        let mut greatest_observed = groups.iter().map(|group| group.id.get()).max().unwrap_or(0);
        for group in &groups {
            for referenced in group.referenced_group_ids() {
                greatest_observed = greatest_observed.max(referenced.get());
            }
        }
        for item in &root {
            if let RootItem::Group { group_id } = item {
                greatest_observed = greatest_observed.max(group_id.get());
            }
        }
        let inferred = greatest_observed.checked_add(1).unwrap_or(0);
        let next_group_id = next_group_id.unwrap_or(inferred.max(1));
        if next_group_id != 0 && next_group_id <= greatest_observed
            || next_group_id == 0 && greatest_observed != u64::MAX
        {
            return Err(CompositionError::InvalidNextGroupId {
                next: next_group_id,
                greatest_observed,
            });
        }
        let tree = Self {
            groups,
            root,
            next_group_id,
            bus_crossfade: finite_unit(bus_crossfade, 0.5),
            mixer: BusMixerState::default(),
        };
        tree.validate_structure()?;
        Ok(tree)
    }

    /// Install the mixing-boundary state on a constructed tree. Sanitizes on
    /// entry, so no construction path can carry hostile values.
    pub fn with_mixer(mut self, mixer: BusMixerState) -> Self {
        self.mixer = mixer.sanitized();
        self
    }

    pub fn groups(&self) -> impl ExactSizeIterator<Item = &Group> {
        self.groups.iter()
    }

    pub fn root(&self) -> &[RootItem] {
        &self.root
    }

    pub fn group(&self, id: GroupId) -> Option<&Group> {
        self.groups.iter().find(|group| group.id == id)
    }

    pub fn group_mut(&mut self, id: GroupId) -> Option<&mut Group> {
        self.groups.iter_mut().find(|group| group.id == id)
    }

    pub fn contains_group(&self, id: GroupId) -> bool {
        self.group(id).is_some()
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "cursor introspection supports persistence/editor verification"
        )
    )]
    pub const fn next_group_id_raw(&self) -> u64 {
        self.next_group_id
    }

    pub fn bus_crossfade(&self) -> f32 {
        self.bus_crossfade
    }

    pub fn set_bus_crossfade(&mut self, value: f32) {
        self.bus_crossfade = finite_unit(value, 0.5);
    }

    pub fn mixer(&self) -> BusMixerState {
        self.mixer
    }

    pub fn set_mixer(&mut self, mixer: BusMixerState) {
        self.mixer = mixer.sanitized();
    }

    /// Advance the cursor for a reference found outside this tree (for example
    /// a layer rack). Missing references count: future groups may never inherit
    /// their identities.
    pub fn observe_group_reference(&mut self, id: GroupId) {
        if self.next_group_id != 0 && id.get() >= self.next_group_id {
            self.next_group_id = id.get().checked_add(1).unwrap_or(0);
        }
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "saved-tree mutation is retained for patch/editor tooling"
        )
    )]
    pub fn insert_empty_group(
        &mut self,
        name: GroupName,
        root_index: usize,
    ) -> Result<GroupId, CompositionError> {
        if self.groups.len() == MAX_GROUPS {
            return Err(CompositionError::TooManyGroups {
                count: self.groups.len() + 1,
                limit: MAX_GROUPS,
            });
        }
        if root_index > self.root.len() {
            return Err(CompositionError::InvalidMoveIndex {
                index: root_index,
                len: self.root.len(),
            });
        }
        let id = GroupId::new(self.next_group_id).ok_or(CompositionError::GroupIdExhausted)?;
        self.next_group_id = self.next_group_id.checked_add(1).unwrap_or(0);
        self.groups.push(Group::empty(id, name));
        self.root
            .insert(root_index, RootItem::Group { group_id: id });
        Ok(id)
    }

    /// Explicit deletion invalidates this output and ungroups members in place.
    /// Merely emptying a group does not call this path and leaves a transparent,
    /// addressable output.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "saved-tree mutation is retained for patch/editor tooling"
        )
    )]
    pub fn remove_group_ungroup(&mut self, id: GroupId) -> Result<Group, CompositionError> {
        let group_index = self
            .groups
            .iter()
            .position(|group| group.id == id)
            .ok_or(CompositionError::UnknownGroup(id))?;
        let root_index = self
            .root
            .iter()
            .position(|item| item.key() == RootItemKey::Group(id))
            .ok_or(CompositionError::UnrootedGroup(id))?;
        let group = self.groups.remove(group_index);
        self.root.remove(root_index);
        for (offset, layer) in group.members.iter().enumerate() {
            self.root.insert(
                root_index + offset,
                RootItem::Layer {
                    layer,
                    bus: group.bus,
                },
            );
        }
        for remaining in &mut self.groups {
            remaining.mark_group_output_missing(id);
        }
        Ok(group)
    }

    #[allow(
        dead_code,
        reason = "retained checked reorder operation for future saved-tree editor controls"
    )]
    pub fn move_root_item(
        &mut self,
        key: RootItemKey,
        new_index: usize,
    ) -> Result<(), CompositionError> {
        if new_index >= self.root.len() {
            return Err(CompositionError::InvalidMoveIndex {
                index: new_index,
                len: self.root.len(),
            });
        }
        let old_index = self
            .root
            .iter()
            .position(|item| item.key() == key)
            .ok_or(match key {
                RootItemKey::Layer(layer) => CompositionError::UnknownLayer(layer),
                RootItemKey::Group(group) => CompositionError::UnknownGroup(group),
            })?;
        let item = self.root.remove(old_index);
        self.root.insert(new_index, item);
        Ok(())
    }

    #[allow(
        dead_code,
        reason = "retained checked reorder operation for future saved-tree editor controls"
    )]
    pub fn move_group_member(
        &mut self,
        group_id: GroupId,
        layer: SavedLayerPosition,
        new_index: usize,
    ) -> Result<(), CompositionError> {
        self.group_mut(group_id)
            .ok_or(CompositionError::UnknownGroup(group_id))?
            .members
            .move_layer(layer, new_index)
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "saved group-output diagnostics are exposed to editor tests"
        )
    )]
    pub fn group_output_status(&self, id: GroupId) -> GroupOutputStatus {
        self.group(id).map_or(GroupOutputStatus::Missing, |group| {
            GroupOutputStatus::Available {
                transparent_empty: group.members.is_empty(),
                stage: GroupOutputStage::PostProcessingPreAdmission,
            }
        })
    }

    pub fn validate_for_layers(
        &self,
        expected_layers: &[SavedLayerPosition],
    ) -> Result<(), CompositionError> {
        self.validate_structure()?;
        let mut expected = BTreeSet::new();
        for layer in expected_layers {
            if !expected.insert(*layer) {
                return Err(CompositionError::DuplicateLayer(*layer));
            }
        }
        let actual: BTreeSet<_> = self
            .flatten_unchecked()
            .layers
            .iter()
            .map(|layer| layer.layer)
            .collect();
        if let Some(unknown) = actual.difference(&expected).next() {
            return Err(CompositionError::UnknownLayer(*unknown));
        }
        if let Some(missing) = expected.difference(&actual).next() {
            return Err(CompositionError::MissingLayer(*missing));
        }
        Ok(())
    }

    fn validate_structure(&self) -> Result<(), CompositionError> {
        let mut group_ids = BTreeSet::new();
        for group in &self.groups {
            if !group_ids.insert(group.id) {
                return Err(CompositionError::DuplicateGroupId(group.id));
            }
            group
                .rack
                .validate_for_scope(LegacyRackScope::Group)
                .map_err(|error| CompositionError::GroupRack {
                    group_id: group.id,
                    error,
                })?;
        }
        let mut root_keys = BTreeSet::new();
        let mut layers = BTreeSet::new();
        let mut rooted_groups = BTreeSet::new();
        for item in &self.root {
            if !root_keys.insert(item.key()) {
                return Err(CompositionError::DuplicateRootItem(item.key()));
            }
            match *item {
                RootItem::Layer { layer, .. } => {
                    if !layers.insert(layer) {
                        return Err(CompositionError::DuplicateLayer(layer));
                    }
                }
                RootItem::Group { group_id } => {
                    if !group_ids.contains(&group_id) {
                        return Err(CompositionError::UnknownGroup(group_id));
                    }
                    rooted_groups.insert(group_id);
                    for layer in self
                        .group(group_id)
                        .expect("validated group exists")
                        .members
                        .iter()
                    {
                        if !layers.insert(layer) {
                            return Err(CompositionError::DuplicateLayer(layer));
                        }
                    }
                }
            }
        }
        if let Some(unrooted) = group_ids.difference(&rooted_groups).next() {
            return Err(CompositionError::UnrootedGroup(*unrooted));
        }
        Ok(())
    }

    pub fn flatten(&self) -> Result<FlattenedComposition, CompositionError> {
        self.validate_structure()?;
        Ok(self.flatten_unchecked())
    }

    fn flatten_unchecked(&self) -> FlattenedComposition {
        let any_solo = self.groups.iter().any(|group| group.solo);
        let mut layers = Vec::new();
        let mut groups = Vec::with_capacity(self.groups.len());
        for item in &self.root {
            match *item {
                RootItem::Layer { layer, bus } => layers.push(FlattenedLayer {
                    layer,
                    group_id: None,
                    bus,
                    admitted_to_program: !any_solo,
                }),
                RootItem::Group { group_id } => {
                    let group = self.group(group_id).expect("structure was validated");
                    let start = layers.len();
                    let admitted = !any_solo || group.solo;
                    layers.extend(group.members.iter().map(|layer| FlattenedLayer {
                        layer,
                        group_id: Some(group.id),
                        bus: group.bus,
                        admitted_to_program: admitted,
                    }));
                    groups.push(FlattenedGroupSpan {
                        group_id,
                        start,
                        len: group.members.len(),
                        bus: group.bus,
                        admitted_to_program: admitted,
                        bypass: group.bypass,
                        output_stage: GroupOutputStage::PostProcessingPreAdmission,
                    });
                }
            }
        }
        FlattenedComposition { layers, groups }
    }
}

impl Serialize for CompositionTree {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // The mixer block is skip-serialized at the exact-legacy default so
        // every pre-B8 patch keeps its bytes and canonical hash.
        let mixer_authored = self.mixer != BusMixerState::default();
        let mut state =
            serializer.serialize_struct("CompositionTree", if mixer_authored { 5 } else { 4 })?;
        state.serialize_field("groups", &Groups(self.groups.clone()))?;
        state.serialize_field("root", &RootItems(self.root.clone()))?;
        state.serialize_field("next_group_id", &self.next_group_id)?;
        state.serialize_field("bus_crossfade", &self.bus_crossfade)?;
        if mixer_authored {
            state.serialize_field("mixer", &self.mixer)?;
        }
        state.end()
    }
}

impl<'de> Deserialize<'de> for CompositionTree {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            #[serde(default)]
            groups: Groups,
            #[serde(default)]
            root: RootItems,
            #[serde(default)]
            next_group_id: Option<u64>,
            #[serde(default = "half")]
            bus_crossfade: f32,
            #[serde(default)]
            mixer: BusMixerState,
        }

        let raw = Raw::deserialize(deserializer)?;
        CompositionTree::try_from_parts(
            raw.groups.0,
            raw.root.0,
            raw.next_group_id,
            raw.bus_crossfade,
        )
        .map(|tree| tree.with_mixer(raw.mixer))
        .map_err(de::Error::custom)
    }
}

const fn half() -> f32 {
    0.5
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupOutputStage {
    /// Post group transform/rack/matte (unless bypassed), before group opacity,
    /// solo admission, and A/B/Program lane admission.
    PostProcessingPreAdmission,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "group-output diagnostics are retained for editor surfaces"
    )
)]
pub enum GroupOutputStatus {
    Available {
        transparent_empty: bool,
        stage: GroupOutputStage,
    },
    Missing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlattenedLayer {
    pub layer: SavedLayerPosition,
    pub group_id: Option<GroupId>,
    pub bus: BusAssignment,
    pub admitted_to_program: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlattenedGroupSpan {
    pub group_id: GroupId,
    pub start: usize,
    pub len: usize,
    pub bus: BusAssignment,
    pub admitted_to_program: bool,
    pub bypass: bool,
    pub output_stage: GroupOutputStage,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlattenedComposition {
    pub layers: Vec<FlattenedLayer>,
    pub groups: Vec<FlattenedGroupSpan>,
}

/// Runtime group membership uses process-stable layer identities exclusively.
/// It is never serialized and never treats a numeric ID as a vector index.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeGroupMembers(Vec<StableLayerId>);

impl RuntimeGroupMembers {
    pub fn try_from_vec(entries: Vec<StableLayerId>) -> Result<Self, RuntimeCompositionError> {
        if entries.len() > MAX_COMPOSITION_LAYERS {
            return Err(RuntimeCompositionError::TooManyLayers {
                count: entries.len(),
                limit: MAX_COMPOSITION_LAYERS,
            });
        }
        let mut seen = BTreeSet::new();
        for layer_id in &entries {
            if !seen.insert(*layer_id) {
                return Err(RuntimeCompositionError::DuplicateLayer(*layer_id));
            }
        }
        Ok(Self(entries))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "runtime group inspection is exercised by composition tests"
        )
    )]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = StableLayerId> + '_ {
        self.0.iter().copied()
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "runtime checked reorder is exercised by composition tests"
        )
    )]
    fn move_layer(
        &mut self,
        layer_id: StableLayerId,
        new_index: usize,
    ) -> Result<(), RuntimeCompositionError> {
        if new_index >= self.0.len() {
            return Err(RuntimeCompositionError::InvalidMoveIndex {
                index: new_index,
                len: self.0.len(),
            });
        }
        let old_index = self
            .0
            .iter()
            .position(|candidate| *candidate == layer_id)
            .ok_or(RuntimeCompositionError::UnknownLayer(layer_id))?;
        let entry = self.0.remove(old_index);
        self.0.insert(new_index, entry);
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeGroup {
    pub id: GroupId,
    pub name: GroupName,
    pub members: RuntimeGroupMembers,
    pub opacity: f32,
    pub transform: SpatialTransform,
    pub rack: RuntimeVisualRack,
    pub matte: Option<RuntimeImageMatte>,
    pub solo: bool,
    pub bypass: bool,
    pub bus: BusAssignment,
}

impl RuntimeGroup {
    fn sanitized(mut self) -> Result<Self, RuntimeCompositionError> {
        self.opacity = finite_unit(self.opacity, 1.0);
        self.transform = self.transform.sanitized();
        self.matte = self.matte.map(RuntimeImageMatte::sanitized);
        self.rack
            .validate_for_scope(LegacyRackScope::Group)
            .map_err(|error| RuntimeCompositionError::GroupRack {
                group_id: self.id,
                error,
            })?;
        Ok(self)
    }

    fn referenced_group_ids(&self) -> impl Iterator<Item = GroupId> + '_ {
        self.rack
            .referenced_group_ids()
            .chain(self.matte.and_then(RuntimeImageMatte::referenced_group))
    }

    fn mark_group_output_missing(&mut self, removed: GroupId) {
        self.rack.mark_group_output_missing(removed);
        if let Some(matte) = &mut self.matte {
            matte.mark_group_output_missing(removed);
        }
    }

    fn mark_layer_output_missing(&mut self, removed: StableLayerId) {
        self.rack.mark_layer_output_missing(removed);
        if let Some(matte) = &mut self.matte {
            matte.mark_layer_output_missing(removed);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeRootItem {
    Layer {
        layer_id: StableLayerId,
        bus: BusAssignment,
    },
    Group {
        group_id: GroupId,
    },
}

impl RuntimeRootItem {
    pub const fn key(self) -> RuntimeRootItemKey {
        match self {
            Self::Layer { layer_id, .. } => RuntimeRootItemKey::Layer(layer_id),
            Self::Group { group_id } => RuntimeRootItemKey::Group(group_id),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RuntimeRootItemKey {
    Layer(StableLayerId),
    Group(GroupId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeCompositionError {
    TooManyGroups {
        count: usize,
        limit: usize,
    },
    TooManyLayers {
        count: usize,
        limit: usize,
    },
    DuplicateGroupId(GroupId),
    DuplicateRootItem(RuntimeRootItemKey),
    DuplicateLayer(StableLayerId),
    UnknownGroup(GroupId),
    UnrootedGroup(GroupId),
    UnknownLayer(StableLayerId),
    MissingLayer(StableLayerId),
    InvalidNextGroupId {
        next: u64,
        greatest_observed: u64,
    },
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "reported by the retained runtime group allocation API"
        )
    )]
    GroupIdExhausted,
    InvalidMoveIndex {
        index: usize,
        len: usize,
    },
    GroupRack {
        group_id: GroupId,
        error: RuntimeRackError,
    },
}

impl fmt::Display for RuntimeCompositionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyGroups { count, limit } => {
                write!(
                    formatter,
                    "runtime composition has {count} groups; limit is {limit}"
                )
            }
            Self::TooManyLayers { count, limit } => {
                write!(
                    formatter,
                    "runtime composition has {count} layers; limit is {limit}"
                )
            }
            Self::DuplicateGroupId(id) => {
                write!(formatter, "duplicate runtime group id {}", id.get())
            }
            Self::DuplicateRootItem(item) => {
                write!(formatter, "duplicate runtime root item {item:?}")
            }
            Self::DuplicateLayer(layer_id) => write!(
                formatter,
                "live layer {} occurs more than once in runtime composition",
                layer_id.get()
            ),
            Self::UnknownGroup(id) => write!(
                formatter,
                "runtime root references missing group {}",
                id.get()
            ),
            Self::UnrootedGroup(id) => write!(
                formatter,
                "runtime group {} is not present at the root",
                id.get()
            ),
            Self::UnknownLayer(layer_id) => write!(
                formatter,
                "runtime composition references unknown live layer {}",
                layer_id.get()
            ),
            Self::MissingLayer(layer_id) => write!(
                formatter,
                "runtime composition omits live layer {}",
                layer_id.get()
            ),
            Self::InvalidNextGroupId {
                next,
                greatest_observed,
            } => write!(
                formatter,
                "next runtime group id {next} must advance past live or missing referenced id {greatest_observed}"
            ),
            Self::GroupIdExhausted => {
                formatter.write_str("runtime group identity space is exhausted")
            }
            Self::InvalidMoveIndex { index, len } => write!(
                formatter,
                "runtime composition move index {index} exceeds length {len}"
            ),
            Self::GroupRack { group_id, error } => write!(
                formatter,
                "runtime group {} rack is invalid: {error}",
                group_id.get()
            ),
        }
    }
}

impl std::error::Error for RuntimeCompositionError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompositionResolveError {
    MissingSavedLayer(SavedLayerPosition),
    DuplicateLiveLayer {
        layer_id: StableLayerId,
        first_position: SavedLayerPosition,
        second_position: SavedLayerPosition,
    },
    MissingLiveLayer(StableLayerId),
    DuplicateSavedPosition {
        position: SavedLayerPosition,
        first_layer_id: StableLayerId,
        second_layer_id: StableLayerId,
    },
    InvalidSaved(CompositionError),
    InvalidRuntime(RuntimeCompositionError),
    RouteCapture {
        group_id: GroupId,
        error: RouteCaptureError,
    },
}

impl fmt::Display for CompositionResolveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSavedLayer(position) => write!(
                formatter,
                "saved composition layer position {} cannot be resolved atomically",
                position.get()
            ),
            Self::DuplicateLiveLayer {
                layer_id,
                first_position,
                second_position,
            } => write!(
                formatter,
                "saved positions {} and {} both resolve to live layer {}; refusing to retarget",
                first_position.get(),
                second_position.get(),
                layer_id.get()
            ),
            Self::MissingLiveLayer(layer_id) => write!(
                formatter,
                "live composition layer {} cannot be captured atomically",
                layer_id.get()
            ),
            Self::DuplicateSavedPosition {
                position,
                first_layer_id,
                second_layer_id,
            } => write!(
                formatter,
                "live layers {} and {} both capture as saved position {}; refusing to retarget",
                first_layer_id.get(),
                second_layer_id.get(),
                position.get()
            ),
            Self::InvalidSaved(error) => write!(formatter, "saved composition is invalid: {error}"),
            Self::InvalidRuntime(error) => {
                write!(formatter, "runtime composition is invalid: {error}")
            }
            Self::RouteCapture { group_id, error } => write!(
                formatter,
                "runtime group {} routes cannot be captured: {error}",
                group_id.get()
            ),
        }
    }
}

impl std::error::Error for CompositionResolveError {}

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeComposition {
    groups: Vec<RuntimeGroup>,
    root: Vec<RuntimeRootItem>,
    next_group_id: u64,
    bus_crossfade: f32,
    mixer: BusMixerState,
}

impl CompositionTree {
    /// Resolve every unique saved membership exactly once, then commit the
    /// complete stable-ID tree. A missing or duplicate mapping returns an error
    /// without publishing a partially resolved composition.
    pub fn resolve(
        &self,
        mut layer_at_position: impl FnMut(SavedLayerPosition) -> Option<StableLayerId>,
    ) -> Result<RuntimeComposition, CompositionResolveError> {
        let flattened = self
            .flatten()
            .map_err(CompositionResolveError::InvalidSaved)?;
        let member_positions: BTreeSet<_> =
            flattened.layers.iter().map(|layer| layer.layer).collect();
        let mut all_positions = member_positions.clone();
        for group in &self.groups {
            all_positions.extend(group.rack.selected_layer_positions());
            if let Some(position) = group.matte.and_then(ImageMatte::selected_layer_position) {
                all_positions.insert(position);
            }
        }
        let mut resolved = std::collections::BTreeMap::new();
        let mut reverse = std::collections::BTreeMap::new();
        for position in all_positions {
            let layer_id = layer_at_position(position);
            if let Some(layer_id) = layer_id {
                if let Some(first_position) = reverse.insert(layer_id, position) {
                    return Err(CompositionResolveError::DuplicateLiveLayer {
                        layer_id,
                        first_position,
                        second_position: position,
                    });
                }
            }
            resolved.insert(position, layer_id);
        }
        for position in member_positions {
            if resolved.get(&position).copied().flatten().is_none() {
                return Err(CompositionResolveError::MissingSavedLayer(position));
            }
        }

        let groups = self
            .groups
            .iter()
            .map(|group| {
                let members = group
                    .members
                    .iter()
                    .map(|position| {
                        resolved
                            .get(&position)
                            .copied()
                            .flatten()
                            .expect("flattening resolved every group member")
                    })
                    .collect();
                Ok(RuntimeGroup {
                    id: group.id,
                    name: group.name.clone(),
                    members: RuntimeGroupMembers::try_from_vec(members)
                        .map_err(CompositionResolveError::InvalidRuntime)?,
                    opacity: group.opacity,
                    transform: group.transform,
                    rack: group.rack.resolve_routes(
                        |position| resolved.get(&position).copied().flatten(),
                        |group_id| self.contains_group(group_id),
                    ),
                    matte: group.matte.map(|matte| {
                        RuntimeImageMatte::resolve_routes(
                            matte,
                            &mut |position| resolved.get(&position).copied().flatten(),
                            &|group_id| self.contains_group(group_id),
                        )
                    }),
                    solo: group.solo,
                    bypass: group.bypass,
                    bus: group.bus,
                })
            })
            .collect::<Result<Vec<_>, CompositionResolveError>>()?;
        let root = self
            .root
            .iter()
            .map(|item| match *item {
                RootItem::Layer { layer, bus } => RuntimeRootItem::Layer {
                    layer_id: resolved
                        .get(&layer)
                        .copied()
                        .flatten()
                        .expect("flattening resolved every direct root layer"),
                    bus,
                },
                RootItem::Group { group_id } => RuntimeRootItem::Group { group_id },
            })
            .collect();
        RuntimeComposition::try_from_parts(
            groups,
            root,
            Some(self.next_group_id),
            self.bus_crossfade,
        )
        .map(|runtime| runtime.with_mixer(self.mixer))
        .map_err(CompositionResolveError::InvalidRuntime)
    }
}

impl RuntimeComposition {
    pub fn try_from_parts(
        groups: Vec<RuntimeGroup>,
        root: Vec<RuntimeRootItem>,
        next_group_id: Option<u64>,
        bus_crossfade: f32,
    ) -> Result<Self, RuntimeCompositionError> {
        if groups.len() > MAX_GROUPS {
            return Err(RuntimeCompositionError::TooManyGroups {
                count: groups.len(),
                limit: MAX_GROUPS,
            });
        }
        let groups = groups
            .into_iter()
            .map(RuntimeGroup::sanitized)
            .collect::<Result<Vec<_>, _>>()?;
        let mut greatest_observed = groups.iter().map(|group| group.id.get()).max().unwrap_or(0);
        for group in &groups {
            for referenced in group.referenced_group_ids() {
                greatest_observed = greatest_observed.max(referenced.get());
            }
        }
        for item in &root {
            if let RuntimeRootItem::Group { group_id } = item {
                greatest_observed = greatest_observed.max(group_id.get());
            }
        }
        let inferred = greatest_observed.checked_add(1).unwrap_or(0);
        let next_group_id = next_group_id.unwrap_or(inferred.max(1));
        if next_group_id != 0 && next_group_id <= greatest_observed
            || next_group_id == 0 && greatest_observed != u64::MAX
        {
            return Err(RuntimeCompositionError::InvalidNextGroupId {
                next: next_group_id,
                greatest_observed,
            });
        }
        let runtime = Self {
            groups,
            root,
            next_group_id,
            bus_crossfade: finite_unit(bus_crossfade, 0.5),
            mixer: BusMixerState::default(),
        };
        runtime.validate_structure()?;
        Ok(runtime)
    }

    /// Install the mixing-boundary state on a constructed runtime tree.
    /// Sanitizes on entry, so no construction path can carry hostile values.
    pub fn with_mixer(mut self, mixer: BusMixerState) -> Self {
        self.mixer = mixer.sanitized();
        self
    }

    pub fn groups(&self) -> impl ExactSizeIterator<Item = &RuntimeGroup> {
        self.groups.iter()
    }

    pub fn root(&self) -> &[RuntimeRootItem] {
        &self.root
    }

    pub fn group(&self, id: GroupId) -> Option<&RuntimeGroup> {
        self.groups.iter().find(|group| group.id == id)
    }

    pub fn group_mut(&mut self, id: GroupId) -> Option<&mut RuntimeGroup> {
        self.groups.iter_mut().find(|group| group.id == id)
    }

    pub fn contains_group(&self, id: GroupId) -> bool {
        self.group(id).is_some()
    }

    pub const fn next_group_id_raw(&self) -> u64 {
        self.next_group_id
    }

    pub fn bus_crossfade(&self) -> f32 {
        self.bus_crossfade
    }

    pub fn set_bus_crossfade(&mut self, value: f32) {
        self.bus_crossfade = finite_unit(value, 0.5);
    }

    pub fn mixer(&self) -> BusMixerState {
        self.mixer
    }

    pub fn set_mixer(&mut self, mixer: BusMixerState) {
        self.mixer = mixer.sanitized();
    }

    #[allow(
        dead_code,
        reason = "retained for external reference scans that advance monotonic group identity"
    )]
    pub fn observe_group_reference(&mut self, id: GroupId) {
        if self.next_group_id != 0 && id.get() >= self.next_group_id {
            self.next_group_id = id.get().checked_add(1).unwrap_or(0);
        }
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "runtime group creation is currently driven by editor tests"
        )
    )]
    pub fn insert_empty_group(
        &mut self,
        name: GroupName,
        root_index: usize,
    ) -> Result<GroupId, RuntimeCompositionError> {
        if self.groups.len() == MAX_GROUPS {
            return Err(RuntimeCompositionError::TooManyGroups {
                count: self.groups.len() + 1,
                limit: MAX_GROUPS,
            });
        }
        if root_index > self.root.len() {
            return Err(RuntimeCompositionError::InvalidMoveIndex {
                index: root_index,
                len: self.root.len(),
            });
        }
        let id =
            GroupId::new(self.next_group_id).ok_or(RuntimeCompositionError::GroupIdExhausted)?;
        self.next_group_id = self.next_group_id.checked_add(1).unwrap_or(0);
        self.groups.push(RuntimeGroup {
            id,
            name,
            members: RuntimeGroupMembers::default(),
            opacity: 1.0,
            transform: SpatialTransform::default(),
            rack: RuntimeVisualRack::empty(),
            matte: None,
            solo: false,
            bypass: false,
            bus: BusAssignment::Program,
        });
        self.root
            .insert(root_index, RuntimeRootItem::Group { group_id: id });
        Ok(id)
    }

    pub fn remove_group_ungroup(
        &mut self,
        id: GroupId,
    ) -> Result<RuntimeGroup, RuntimeCompositionError> {
        let group_index = self
            .groups
            .iter()
            .position(|group| group.id == id)
            .ok_or(RuntimeCompositionError::UnknownGroup(id))?;
        let root_index = self
            .root
            .iter()
            .position(|item| item.key() == RuntimeRootItemKey::Group(id))
            .ok_or(RuntimeCompositionError::UnrootedGroup(id))?;
        let group = self.groups.remove(group_index);
        self.root.remove(root_index);
        for (offset, layer_id) in group.members.iter().enumerate() {
            self.root.insert(
                root_index + offset,
                RuntimeRootItem::Layer {
                    layer_id,
                    bus: group.bus,
                },
            );
        }
        for remaining in &mut self.groups {
            remaining.mark_group_output_missing(id);
        }
        Ok(group)
    }

    /// Invalidate every live route to an explicitly deleted layer. Membership
    /// removal remains the caller's operation; this method only guarantees
    /// that image donors cannot retarget if the saved position is later reused.
    pub fn mark_layer_output_missing(&mut self, removed: StableLayerId) {
        for group in &mut self.groups {
            group.mark_layer_output_missing(removed);
        }
    }

    pub fn move_root_item(
        &mut self,
        key: RuntimeRootItemKey,
        new_index: usize,
    ) -> Result<(), RuntimeCompositionError> {
        if new_index >= self.root.len() {
            return Err(RuntimeCompositionError::InvalidMoveIndex {
                index: new_index,
                len: self.root.len(),
            });
        }
        let old_index = self
            .root
            .iter()
            .position(|item| item.key() == key)
            .ok_or(match key {
                RuntimeRootItemKey::Layer(layer_id) => {
                    RuntimeCompositionError::UnknownLayer(layer_id)
                }
                RuntimeRootItemKey::Group(group_id) => {
                    RuntimeCompositionError::UnknownGroup(group_id)
                }
            })?;
        let item = self.root.remove(old_index);
        self.root.insert(new_index, item);
        Ok(())
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "runtime group reorder is currently driven by editor tests"
        )
    )]
    pub fn move_group_member(
        &mut self,
        group_id: GroupId,
        layer_id: StableLayerId,
        new_index: usize,
    ) -> Result<(), RuntimeCompositionError> {
        self.group_mut(group_id)
            .ok_or(RuntimeCompositionError::UnknownGroup(group_id))?
            .members
            .move_layer(layer_id, new_index)
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "runtime group-output diagnostics are exposed to editor tests"
        )
    )]
    pub fn group_output_status(&self, id: GroupId) -> GroupOutputStatus {
        self.group(id).map_or(GroupOutputStatus::Missing, |group| {
            GroupOutputStatus::Available {
                transparent_empty: group.members.is_empty(),
                stage: GroupOutputStage::PostProcessingPreAdmission,
            }
        })
    }

    pub fn validate_for_layers(
        &self,
        expected_layers: &[StableLayerId],
    ) -> Result<(), RuntimeCompositionError> {
        self.validate_structure()?;
        let mut expected = BTreeSet::new();
        for layer_id in expected_layers {
            if !expected.insert(*layer_id) {
                return Err(RuntimeCompositionError::DuplicateLayer(*layer_id));
            }
        }
        let actual: BTreeSet<_> = self
            .flatten_unchecked()
            .layers
            .iter()
            .map(|layer| layer.layer_id)
            .collect();
        if let Some(unknown) = actual.difference(&expected).next() {
            return Err(RuntimeCompositionError::UnknownLayer(*unknown));
        }
        if let Some(missing) = expected.difference(&actual).next() {
            return Err(RuntimeCompositionError::MissingLayer(*missing));
        }
        Ok(())
    }

    fn validate_structure(&self) -> Result<(), RuntimeCompositionError> {
        let mut group_ids = BTreeSet::new();
        for group in &self.groups {
            if !group_ids.insert(group.id) {
                return Err(RuntimeCompositionError::DuplicateGroupId(group.id));
            }
            group
                .rack
                .validate_for_scope(LegacyRackScope::Group)
                .map_err(|error| RuntimeCompositionError::GroupRack {
                    group_id: group.id,
                    error,
                })?;
        }
        let mut root_keys = BTreeSet::new();
        let mut layers = BTreeSet::new();
        let mut rooted_groups = BTreeSet::new();
        for item in &self.root {
            if !root_keys.insert(item.key()) {
                return Err(RuntimeCompositionError::DuplicateRootItem(item.key()));
            }
            match *item {
                RuntimeRootItem::Layer { layer_id, .. } => {
                    if !layers.insert(layer_id) {
                        return Err(RuntimeCompositionError::DuplicateLayer(layer_id));
                    }
                }
                RuntimeRootItem::Group { group_id } => {
                    if !group_ids.contains(&group_id) {
                        return Err(RuntimeCompositionError::UnknownGroup(group_id));
                    }
                    rooted_groups.insert(group_id);
                    for layer_id in self
                        .group(group_id)
                        .expect("validated runtime group exists")
                        .members
                        .iter()
                    {
                        if !layers.insert(layer_id) {
                            return Err(RuntimeCompositionError::DuplicateLayer(layer_id));
                        }
                    }
                }
            }
        }
        if let Some(unrooted) = group_ids.difference(&rooted_groups).next() {
            return Err(RuntimeCompositionError::UnrootedGroup(*unrooted));
        }
        Ok(())
    }

    pub fn flatten(&self) -> Result<RuntimeFlattenedComposition, RuntimeCompositionError> {
        self.validate_structure()?;
        Ok(self.flatten_unchecked())
    }

    fn flatten_unchecked(&self) -> RuntimeFlattenedComposition {
        let any_solo = self.groups.iter().any(|group| group.solo);
        let mut layers = Vec::new();
        let mut groups = Vec::with_capacity(self.groups.len());
        for item in &self.root {
            match *item {
                RuntimeRootItem::Layer { layer_id, bus } => layers.push(RuntimeFlattenedLayer {
                    layer_id,
                    group_id: None,
                    bus,
                    admitted_to_program: !any_solo,
                }),
                RuntimeRootItem::Group { group_id } => {
                    let group = self.group(group_id).expect("structure was validated");
                    let start = layers.len();
                    let admitted = !any_solo || group.solo;
                    layers.extend(group.members.iter().map(|layer_id| RuntimeFlattenedLayer {
                        layer_id,
                        group_id: Some(group.id),
                        bus: group.bus,
                        admitted_to_program: admitted,
                    }));
                    groups.push(FlattenedGroupSpan {
                        group_id,
                        start,
                        len: group.members.len(),
                        bus: group.bus,
                        admitted_to_program: admitted,
                        bypass: group.bypass,
                        output_stage: GroupOutputStage::PostProcessingPreAdmission,
                    });
                }
            }
        }
        RuntimeFlattenedComposition { layers, groups }
    }

    /// Capture every unique live identity exactly once. The runtime tree is
    /// immutable during mapping; failure cannot partially rewrite memberships.
    pub fn capture(
        &self,
        mut position_of_layer: impl FnMut(StableLayerId) -> Option<SavedLayerPosition>,
    ) -> Result<CompositionTree, CompositionResolveError> {
        let flattened = self
            .flatten()
            .map_err(CompositionResolveError::InvalidRuntime)?;
        let member_ids: BTreeSet<_> = flattened
            .layers
            .iter()
            .map(|layer| layer.layer_id)
            .collect();
        let mut all_ids = member_ids.clone();
        for group in &self.groups {
            all_ids.extend(group.rack.selected_layer_ids());
            if let Some(layer_id) = group.matte.and_then(RuntimeImageMatte::selected_layer_id) {
                all_ids.insert(layer_id);
            }
        }
        let mut captured = std::collections::BTreeMap::new();
        let mut reverse = std::collections::BTreeMap::new();
        for layer_id in all_ids {
            let position = position_of_layer(layer_id);
            if let Some(position) = position {
                if let Some(first_layer_id) = reverse.insert(position, layer_id) {
                    return Err(CompositionResolveError::DuplicateSavedPosition {
                        position,
                        first_layer_id,
                        second_layer_id: layer_id,
                    });
                }
            }
            captured.insert(layer_id, position);
        }
        for layer_id in member_ids {
            if captured.get(&layer_id).copied().flatten().is_none() {
                return Err(CompositionResolveError::MissingLiveLayer(layer_id));
            }
        }
        let groups = self
            .groups
            .iter()
            .map(|group| {
                let members = group
                    .members
                    .iter()
                    .map(|layer_id| {
                        captured
                            .get(&layer_id)
                            .copied()
                            .flatten()
                            .expect("flattening captured every runtime group member")
                    })
                    .collect();
                Ok(Group {
                    id: group.id,
                    name: group.name.clone(),
                    members: GroupMembers::try_from_vec(members)
                        .map_err(CompositionResolveError::InvalidSaved)?,
                    opacity: group.opacity,
                    transform: group.transform,
                    rack: group
                        .rack
                        .capture_routes(|layer_id| captured.get(&layer_id).copied().flatten())
                        .map_err(|error| CompositionResolveError::RouteCapture {
                            group_id: group.id,
                            error,
                        })?,
                    matte: group.matte.map(|matte| {
                        matte.capture_routes(&mut |layer_id| {
                            captured.get(&layer_id).copied().flatten()
                        })
                    }),
                    solo: group.solo,
                    bypass: group.bypass,
                    bus: group.bus,
                })
            })
            .collect::<Result<Vec<_>, CompositionResolveError>>()?;
        let root = self
            .root
            .iter()
            .map(|item| match *item {
                RuntimeRootItem::Layer { layer_id, bus } => RootItem::Layer {
                    layer: captured
                        .get(&layer_id)
                        .copied()
                        .flatten()
                        .expect("flattening captured every direct runtime layer"),
                    bus,
                },
                RuntimeRootItem::Group { group_id } => RootItem::Group { group_id },
            })
            .collect();
        CompositionTree::try_from_parts(groups, root, Some(self.next_group_id), self.bus_crossfade)
            .map(|tree| tree.with_mixer(self.mixer))
            .map_err(CompositionResolveError::InvalidSaved)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeFlattenedLayer {
    pub layer_id: StableLayerId,
    pub group_id: Option<GroupId>,
    pub bus: BusAssignment,
    pub admitted_to_program: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeFlattenedComposition {
    pub layers: Vec<RuntimeFlattenedLayer>,
    pub groups: Vec<FlattenedGroupSpan>,
}

/// Linear-light premultiplied RGBA used only as the CPU reference law. Values
/// may be HDR; alpha is sanitized to [0, 1].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg(test)]
pub struct PremultipliedRgba(pub [f32; 4]);

#[cfg(test)]
impl PremultipliedRgba {
    pub fn new(value: [f32; 4]) -> Self {
        Self([
            finite(value[0], 0.0),
            finite(value[1], 0.0),
            finite(value[2], 0.0),
            finite_unit(value[3], 0.0),
        ])
    }

    pub fn from_straight_linear(rgb: [f32; 3], alpha: f32) -> Self {
        let alpha = finite_unit(alpha, 0.0);
        Self::new([
            finite(rgb[0], 0.0) * alpha,
            finite(rgb[1], 0.0) * alpha,
            finite(rgb[2], 0.0) * alpha,
            alpha,
        ])
    }

    /// `self` over `under`, both premultiplied.
    pub fn over(self, under: Self) -> Self {
        let keep = 1.0 - self.0[3];
        Self::new([
            self.0[0] + under.0[0] * keep,
            self.0[1] + under.0[1] * keep,
            self.0[2] + under.0[2] * keep,
            self.0[3] + under.0[3] * keep,
        ])
    }

    pub fn lerp(self, other: Self, amount: f32) -> Self {
        let amount = finite_unit(amount, 0.5);
        let inverse = 1.0 - amount;
        Self::new([
            self.0[0] * inverse + other.0[0] * amount,
            self.0[1] * inverse + other.0[1] * amount,
            self.0[2] * inverse + other.0[2] * amount,
            self.0[3] * inverse + other.0[3] * amount,
        ])
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg(test)]
pub struct BusSample {
    pub bus: BusAssignment,
    /// Already includes layer/group opacity and solo admission. Group taps are
    /// captured before producing this value.
    pub pixel: PremultipliedRgba,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg(test)]
pub struct BusReferenceResult {
    pub a: PremultipliedRgba,
    pub b: PremultipliedRgba,
    pub program: PremultipliedRgba,
    pub ab_crossfade: PremultipliedRgba,
    pub output: PremultipliedRgba,
}

/// Composite back-to-front within three independent lanes, interpolate the A
/// and B premultiplied-linear accumulators, then place Program over that result.
#[cfg(test)]
pub fn composite_bus_reference(
    samples_back_to_front: impl IntoIterator<Item = BusSample>,
    crossfade: f32,
) -> BusReferenceResult {
    let mut result = BusReferenceResult::default();
    for sample in samples_back_to_front {
        match sample.bus {
            BusAssignment::A => result.a = sample.pixel.over(result.a),
            BusAssignment::B => result.b = sample.pixel.over(result.b),
            BusAssignment::Program => result.program = sample.pixel.over(result.program),
        }
    }
    result.ab_crossfade = result.a.lerp(result.b, crossfade);
    result.output = result.program.over(result.ab_crossfade);
    result
}

#[cfg(test)]
fn finite(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

fn finite_unit(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_routing::LayerImageStage;
    use crate::visual_rack::{
        EdgeTiming, MaskParams, NodeId, ResolvedImageSource, SavedImageSource, SavedImageTap,
        VisualNodeKind,
    };

    fn position(value: u32) -> SavedLayerPosition {
        SavedLayerPosition::new(value).unwrap()
    }

    fn id(value: u64) -> GroupId {
        GroupId::new(value).unwrap()
    }

    fn live(value: u64) -> StableLayerId {
        StableLayerId::new(value).unwrap()
    }

    fn group(value: u64, members: &[u32]) -> Group {
        Group {
            id: id(value),
            name: GroupName::new(format!("Group {value}")).unwrap(),
            members: GroupMembers::try_from_vec(members.iter().copied().map(position).collect())
                .unwrap(),
            opacity: 1.0,
            transform: SpatialTransform::default(),
            rack: VisualRack::empty(),
            matte: None,
            solo: false,
            bypass: false,
            bus: BusAssignment::Program,
        }
    }

    fn selected_image_matte(layer_position: u32) -> ImageMatte {
        ImageMatte {
            tap: SavedImageTap {
                source: SavedImageSource::SelectedLayer {
                    layer_position: position(layer_position),
                    stage: LayerImageStage::PostLocalEffects,
                },
                timing: EdgeTiming::CurrentFrame,
            },
            ..Default::default()
        }
    }

    fn group_with_selected_routes() -> (Group, NodeId) {
        let mut group = group(7, &[0]);
        let node_id = group
            .rack
            .push(VisualNodeKind::Mask(MaskParams::Image(
                selected_image_matte(1),
            )))
            .unwrap();
        group.matte = Some(selected_image_matte(1));
        (group, node_id)
    }

    #[test]
    fn recorder_group_laws_are_engine_owned_and_refuse_topology() {
        assert_eq!(
            group_value_law("opacity"),
            Some(AuthoringValueLaw::Unit([0.0, 1.0]))
        );
        assert_eq!(group_value_law("solo"), Some(AuthoringValueLaw::Toggle));
        assert_eq!(
            group_value_law("bus"),
            Some(AuthoringValueLaw::Discrete(vec!["program", "a", "b"]))
        );
        assert_eq!(
            group_value_law("position_x"),
            spatial_transform_value_law("position_x")
        );
        for refused in ["name", "members", "rack", "matte", "not_a_group_field"] {
            assert_eq!(group_value_law(refused), None, "{refused}");
        }

        assert_eq!(
            group_matte_value_law("softness"),
            Some(AuthoringValueLaw::Unit([0.0, 0.5]))
        );
        for refused in ["tap", "channel", "invert", "route", "not_a_matte_field"] {
            assert_eq!(group_matte_value_law(refused), None, "{refused}");
        }
        for bus in BusAssignment::ALL {
            assert_eq!(BusAssignment::try_from_key(bus.key()), Some(bus));
        }
        assert_eq!(BusAssignment::try_from_key("sidechain"), None);
    }

    #[test]
    fn group_ids_are_nonzero_u64_and_cursor_cannot_rewind() {
        assert!(serde_json::from_str::<GroupId>("0").is_err());
        assert_eq!(
            serde_json::from_str::<GroupId>("4294967297").unwrap().get(),
            4_294_967_297
        );
        let error = CompositionTree::try_from_parts(
            vec![group(9, &[])],
            vec![RootItem::Group { group_id: id(9) }],
            Some(9),
            0.5,
        )
        .unwrap_err();
        assert!(matches!(error, CompositionError::InvalidNextGroupId { .. }));
    }

    #[test]
    fn group_constructor_and_serde_reject_legacy_rack_markers() {
        let mut invalid = group(1, &[]);
        invalid.rack = VisualRack::synthetic_legacy(LegacyRackScope::Layer);
        assert!(matches!(
            CompositionTree::try_from_parts(
                vec![invalid],
                vec![RootItem::Group { group_id: id(1) }],
                Some(2),
                0.5,
            ),
            Err(CompositionError::GroupRack {
                error: RackError::LegacyMarkerOnGroup(_),
                ..
            })
        ));

        let json = r#"{
            "groups":[{
                "id":1,
                "members":[],
                "rack":{
                    "nodes":[{
                        "stable_id":1,
                        "kind":{"kind":"legacy_canonical"}
                    }],
                    "next_node_id":3
                }
            }],
            "root":[{"kind":"group","group_id":1}],
            "next_group_id":2
        }"#;
        assert!(serde_json::from_str::<CompositionTree>(json)
            .unwrap_err()
            .to_string()
            .contains("not valid in a group rack"));
    }

    #[test]
    fn seventeenth_group_is_rejected_during_deserialization() {
        let groups = (1..=17)
            .map(|value| format!(r#"{{"id":{value},"name":"g","members":[]}}"#))
            .collect::<Vec<_>>()
            .join(",");
        let root = (1..=17)
            .map(|value| format!(r#"{{"kind":"group","group_id":{value}}}"#))
            .collect::<Vec<_>>()
            .join(",");
        let json = format!(r#"{{"groups":[{groups}],"root":[{root}],"next_group_id":18}}"#);
        assert!(serde_json::from_str::<CompositionTree>(&json)
            .unwrap_err()
            .to_string()
            .contains("at most 16"));
    }

    #[test]
    fn hostile_name_and_nested_member_shape_reject() {
        let name = "x".repeat(MAX_GROUP_NAME_BYTES + 1);
        let json = format!(
            r#"{{"groups":[{{"id":1,"name":"{name}","members":[]}}],"root":[{{"kind":"group","group_id":1}}],"next_group_id":2}}"#
        );
        assert!(serde_json::from_str::<CompositionTree>(&json).is_err());

        let nested = r#"{"groups":[{"id":1,"members":[{"kind":"group","group_id":2}]}],"root":[{"kind":"group","group_id":1}],"next_group_id":3}"#;
        assert!(serde_json::from_str::<CompositionTree>(nested).is_err());
    }

    #[test]
    fn validation_requires_every_layer_and_group_exactly_once() {
        let duplicate = CompositionTree::try_from_parts(
            vec![group(1, &[0])],
            vec![
                RootItem::Layer {
                    layer: position(0),
                    bus: BusAssignment::Program,
                },
                RootItem::Group { group_id: id(1) },
            ],
            Some(2),
            0.5,
        );
        assert!(matches!(
            duplicate,
            Err(CompositionError::DuplicateLayer(_))
        ));

        let tree = CompositionTree::legacy_for_layers(&[position(0)]).unwrap();
        assert!(matches!(
            tree.validate_for_layers(&[position(0), position(1)]),
            Err(CompositionError::MissingLayer(_))
        ));
    }

    #[test]
    fn flatten_preserves_contiguous_group_block_and_solo_law() {
        let mut first = group(7, &[1, 2]);
        first.bus = BusAssignment::A;
        first.solo = true;
        let tree = CompositionTree::try_from_parts(
            vec![first],
            vec![
                RootItem::Layer {
                    layer: position(0),
                    bus: BusAssignment::Program,
                },
                RootItem::Group { group_id: id(7) },
                RootItem::Layer {
                    layer: position(3),
                    bus: BusAssignment::B,
                },
            ],
            Some(8),
            0.5,
        )
        .unwrap();
        tree.validate_for_layers(&[position(0), position(1), position(2), position(3)])
            .unwrap();
        let flat = tree.flatten().unwrap();
        assert_eq!(
            flat.layers
                .iter()
                .map(|entry| entry.layer)
                .collect::<Vec<_>>(),
            vec![position(0), position(1), position(2), position(3)]
        );
        assert_eq!(flat.groups[0].start, 1);
        assert_eq!(flat.groups[0].len, 2);
        assert!(!flat.layers[0].admitted_to_program);
        assert!(flat.layers[1].admitted_to_program);
        assert!(flat.layers[2].admitted_to_program);
        assert!(!flat.layers[3].admitted_to_program);
        assert_eq!(
            flat.groups[0].output_stage,
            GroupOutputStage::PostProcessingPreAdmission
        );
    }

    #[test]
    fn empty_group_stays_transparent_available_until_explicit_delete() {
        let mut tree = CompositionTree::try_from_parts(
            vec![group(12, &[])],
            vec![RootItem::Group { group_id: id(12) }],
            Some(13),
            0.5,
        )
        .unwrap();
        assert_eq!(
            tree.group_output_status(id(12)),
            GroupOutputStatus::Available {
                transparent_empty: true,
                stage: GroupOutputStage::PostProcessingPreAdmission,
            }
        );
        tree.remove_group_ungroup(id(12)).unwrap();
        assert_eq!(tree.group_output_status(id(12)), GroupOutputStatus::Missing);
        let replacement = tree
            .insert_empty_group(GroupName::new("new").unwrap(), 0)
            .unwrap();
        assert!(replacement.get() > 12);
    }

    #[test]
    fn missing_group_reference_advances_cursor_and_is_not_reused() {
        let mut donor = group(2, &[]);
        donor.matte = Some(ImageMatte {
            tap: crate::visual_rack::SavedImageTap {
                source: SavedImageSource::MissingGroupOutput { group_id: id(99) },
                ..Default::default()
            },
            ..Default::default()
        });
        let mut tree = CompositionTree::try_from_parts(
            vec![donor],
            vec![RootItem::Group { group_id: id(2) }],
            None,
            0.5,
        )
        .unwrap();
        assert_eq!(tree.next_group_id_raw(), 100);
        assert_eq!(
            tree.insert_empty_group(GroupName::new("next").unwrap(), 1)
                .unwrap(),
            id(100)
        );
    }

    fn close(actual: PremultipliedRgba, expected: [f32; 4]) {
        for (actual, expected) in actual.0.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn cpu_bus_reference_keeps_lanes_independent_and_program_on_top() {
        let red = PremultipliedRgba::from_straight_linear([1.0, 0.0, 0.0], 1.0);
        let blue = PremultipliedRgba::from_straight_linear([0.0, 0.0, 1.0], 1.0);
        let green_half = PremultipliedRgba::from_straight_linear([0.0, 1.0, 0.0], 0.5);
        let result = composite_bus_reference(
            [
                BusSample {
                    bus: BusAssignment::A,
                    pixel: red,
                },
                BusSample {
                    bus: BusAssignment::B,
                    pixel: blue,
                },
                BusSample {
                    bus: BusAssignment::Program,
                    pixel: green_half,
                },
            ],
            0.25,
        );
        close(result.a, [1.0, 0.0, 0.0, 1.0]);
        close(result.b, [0.0, 0.0, 1.0, 1.0]);
        close(result.ab_crossfade, [0.75, 0.0, 0.25, 1.0]);
        close(result.output, [0.375, 0.5, 0.125, 1.0]);
    }

    #[test]
    fn serde_sanitizes_crossfade_opacity_and_preserves_empty_groups() {
        let yaml = "groups:\n  - id: 4\n    name: empty\n    members: []\n    opacity: .nan\nroot:\n  - kind: group\n    group_id: 4\nnext_group_id: 5\nbus_crossfade: .inf\n";
        let tree = serde_yaml::from_str::<CompositionTree>(yaml).unwrap();
        assert_eq!(tree.group(id(4)).unwrap().opacity, 1.0);
        assert_eq!(tree.bus_crossfade(), 0.5);
        assert!(tree.group(id(4)).unwrap().members.is_empty());
    }

    #[test]
    fn mixer_is_skip_serialized_at_default_and_travels_through_every_carry() {
        use crate::mixing_boundary::{BusMixerState, MeltParams, WipePattern};

        // A default tree serializes without a mixer block, so every pre-B8
        // patch keeps its bytes and canonical hash; an absent block decodes
        // to the exact-legacy bus.
        let default_tree = CompositionTree::legacy_for_layers(&[]).unwrap();
        let yaml = serde_yaml::to_string(&default_tree).unwrap();
        assert!(
            !yaml.contains("mixer"),
            "default must omit the block: {yaml}"
        );
        let restored = serde_yaml::from_str::<CompositionTree>(&yaml).unwrap();
        assert!(restored.mixer().is_exact_legacy_bus());

        // An authored mixer round-trips through serde, the runtime resolve,
        // and the runtime capture without losing a value.
        let mut authored = default_tree.clone();
        authored.set_mixer(BusMixerState {
            mix: crate::mixing_boundary::BusMixParams {
                pattern: WipePattern::Circle,
                border: 0.4,
                ..Default::default()
            },
            dirt: crate::mixing_boundary::DirtParams {
                dirt: 0.6,
                ..Default::default()
            },
            melt: MeltParams {
                melt: 1.2,
                hold: 1.5,
                ..Default::default()
            },
        });
        let yaml = serde_yaml::to_string(&authored).unwrap();
        let restored = serde_yaml::from_str::<CompositionTree>(&yaml).unwrap();
        assert_eq!(restored.mixer(), authored.mixer());
        let runtime = restored.resolve(|_| None).unwrap();
        assert_eq!(runtime.mixer(), authored.mixer());
        let captured = runtime.capture(|_| None).unwrap();
        assert_eq!(captured.mixer(), authored.mixer());

        // Hostile scalars in a saved block sanitize to neutral values.
        let hostile = "next_group_id: 1\nbus_crossfade: 0.5\nmixer:\n  melt:\n    melt: .nan\n    hold: .inf\n";
        let tree = serde_yaml::from_str::<CompositionTree>(hostile).unwrap();
        assert_eq!(tree.mixer().melt.melt, 0.0);
        assert_eq!(tree.mixer().melt.hold, MeltParams::default().hold);
    }

    #[test]
    fn atomic_runtime_resolution_reorder_and_capture_follow_stable_ids() {
        let saved = CompositionTree::try_from_parts(
            vec![group(7, &[1, 2])],
            vec![
                RootItem::Layer {
                    layer: position(0),
                    bus: BusAssignment::B,
                },
                RootItem::Group { group_id: id(7) },
            ],
            Some(8),
            0.25,
        )
        .unwrap();
        let mut runtime = saved
            .resolve(|position| match position.get() {
                0 => Some(live(101)),
                1 => Some(live(102)),
                2 => Some(live(103)),
                _ => None,
            })
            .unwrap();
        runtime
            .validate_for_layers(&[live(101), live(102), live(103)])
            .unwrap();
        runtime
            .move_root_item(RuntimeRootItemKey::Group(id(7)), 0)
            .unwrap();
        runtime.move_group_member(id(7), live(103), 0).unwrap();
        let flat = runtime.flatten().unwrap();
        assert_eq!(
            flat.layers
                .iter()
                .map(|entry| entry.layer_id)
                .collect::<Vec<_>>(),
            vec![live(103), live(102), live(101)]
        );
        assert_eq!(flat.groups[0].start, 0);
        assert_eq!(flat.groups[0].len, 2);

        // Capture after an external live-stack reorder. The authored positions
        // change, but membership continues to name the same stable identities.
        let captured = runtime
            .capture(|layer_id| match layer_id.get() {
                101 => Some(position(2)),
                102 => Some(position(0)),
                103 => Some(position(1)),
                _ => None,
            })
            .unwrap();
        assert_eq!(captured.next_group_id_raw(), 8);
        assert_eq!(captured.bus_crossfade(), 0.25);
        assert_eq!(
            captured.root(),
            &[
                RootItem::Group { group_id: id(7) },
                RootItem::Layer {
                    layer: position(2),
                    bus: BusAssignment::B,
                },
            ]
        );
        assert_eq!(
            captured
                .group(id(7))
                .unwrap()
                .members
                .iter()
                .collect::<Vec<_>>(),
            vec![position(1), position(0)]
        );
        let re_resolved = captured
            .resolve(|position| match position.get() {
                0 => Some(live(102)),
                1 => Some(live(103)),
                2 => Some(live(101)),
                _ => None,
            })
            .unwrap();
        assert_eq!(re_resolved, runtime);
    }

    #[test]
    fn group_rack_and_matte_donors_follow_stable_id_across_reorder_and_capture() {
        let (group, node_id) = group_with_selected_routes();
        let saved = CompositionTree::try_from_parts(
            vec![group],
            vec![RootItem::Group { group_id: id(7) }],
            Some(8),
            0.5,
        )
        .unwrap();
        let runtime = saved
            .resolve(|position| match position.get() {
                0 => Some(live(10)),
                1 => Some(live(42)),
                _ => None,
            })
            .unwrap();
        let runtime_group = runtime.group(id(7)).unwrap();
        let selected = ResolvedImageSource::SelectedLayer {
            layer_id: live(42),
            saved_position: position(1),
            stage: LayerImageStage::PostLocalEffects,
        };
        assert_eq!(
            runtime_group.rack.image_mask_route(node_id).unwrap().source,
            selected
        );
        assert_eq!(runtime_group.matte.unwrap().tap.source, selected);

        // The member and donor move to new persisted positions. Both group
        // routes still name donor 42, and capture writes only the new position.
        let captured = runtime
            .capture(|layer_id| match layer_id.get() {
                10 => Some(position(2)),
                42 => Some(position(7)),
                _ => None,
            })
            .unwrap();
        let captured_group = captured.group(id(7)).unwrap();
        assert_eq!(
            captured_group.members.iter().collect::<Vec<_>>(),
            vec![position(2)]
        );
        let VisualNodeKind::Mask(MaskParams::Image(rack_matte)) =
            captured_group.rack.get(node_id).unwrap().kind
        else {
            panic!("test rack node must remain an image mask")
        };
        let expected_saved = SavedImageSource::SelectedLayer {
            layer_position: position(7),
            stage: LayerImageStage::PostLocalEffects,
        };
        assert_eq!(rack_matte.tap.source, expected_saved);
        assert_eq!(captured_group.matte.unwrap().tap.source, expected_saved);
        assert!(!serde_json::to_string(&captured)
            .unwrap()
            .contains("\"layer_id\""));

        let re_resolved = captured
            .resolve(|position| match position.get() {
                2 => Some(live(10)),
                7 => Some(live(42)),
                _ => None,
            })
            .unwrap();
        let re_resolved_group = re_resolved.group(id(7)).unwrap();
        assert!(matches!(
            re_resolved_group
                .rack
                .image_mask_route(node_id)
                .unwrap()
                .source,
            ResolvedImageSource::SelectedLayer { layer_id, .. } if layer_id == live(42)
        ));
        assert!(matches!(
            re_resolved_group.matte.unwrap().tap.source,
            ResolvedImageSource::SelectedLayer { layer_id, .. } if layer_id == live(42)
        ));
    }

    #[test]
    fn deleted_group_route_donor_stays_missing_through_capture_and_restore() {
        let (group, node_id) = group_with_selected_routes();
        let saved = CompositionTree::try_from_parts(
            vec![group],
            vec![RootItem::Group { group_id: id(7) }],
            Some(8),
            0.5,
        )
        .unwrap();
        let mut runtime = saved
            .resolve(|position| match position.get() {
                0 => Some(live(10)),
                1 => Some(live(42)),
                _ => None,
            })
            .unwrap();
        runtime.mark_layer_output_missing(live(42));
        let expected_runtime = ResolvedImageSource::MissingSelectedLayer {
            saved_position: position(1),
            stage: LayerImageStage::PostLocalEffects,
        };
        let runtime_group = runtime.group(id(7)).unwrap();
        assert_eq!(
            runtime_group.rack.image_mask_route(node_id).unwrap().source,
            expected_runtime
        );
        assert_eq!(runtime_group.matte.unwrap().tap.source, expected_runtime);

        // Even if a new layer appears at the old slot, the deleted identity is
        // not queried during capture and both routes persist as explicitly missing.
        let captured = runtime
            .capture(|layer_id| match layer_id.get() {
                10 => Some(position(2)),
                42 => Some(position(9)),
                _ => None,
            })
            .unwrap();
        let captured_group = captured.group(id(7)).unwrap();
        let VisualNodeKind::Mask(MaskParams::Image(rack_matte)) =
            captured_group.rack.get(node_id).unwrap().kind
        else {
            panic!("test rack node must remain an image mask")
        };
        let expected_saved = SavedImageSource::MissingSelectedLayer {
            saved_position: position(1),
            stage: LayerImageStage::PostLocalEffects,
        };
        assert_eq!(rack_matte.tap.source, expected_saved);
        assert_eq!(captured_group.matte.unwrap().tap.source, expected_saved);

        let restored = captured
            .resolve(|position| match position.get() {
                1 => Some(live(99)),
                2 => Some(live(10)),
                _ => None,
            })
            .unwrap();
        let restored_group = restored.group(id(7)).unwrap();
        assert_eq!(
            restored_group
                .rack
                .image_mask_route(node_id)
                .unwrap()
                .source,
            expected_runtime
        );
        assert_eq!(restored_group.matte.unwrap().tap.source, expected_runtime);
    }

    #[test]
    fn resolve_and_capture_fail_atomically_on_missing_or_duplicate_mappings() {
        let saved = CompositionTree::legacy_for_layers(&[position(0), position(1)]).unwrap();
        let unchanged = saved.clone();
        assert!(matches!(
            saved.resolve(|position| (position.get() == 0).then(|| live(1))),
            Err(CompositionResolveError::MissingSavedLayer(missing)) if missing == position(1)
        ));
        assert_eq!(saved, unchanged);
        assert!(matches!(
            saved.resolve(|_| Some(live(1))),
            Err(CompositionResolveError::DuplicateLiveLayer { .. })
        ));

        let runtime = saved
            .resolve(|position| Some(live(u64::from(position.get()) + 1)))
            .unwrap();
        let runtime_unchanged = runtime.clone();
        assert!(matches!(
            runtime.capture(|layer_id| (layer_id == live(1)).then(|| position(0))),
            Err(CompositionResolveError::MissingLiveLayer(missing)) if missing == live(2)
        ));
        assert_eq!(runtime, runtime_unchanged);
        assert!(matches!(
            runtime.capture(|_| Some(position(0))),
            Err(CompositionResolveError::DuplicateSavedPosition { .. })
        ));
    }

    #[test]
    fn direct_root_legacy_stack_is_not_subject_to_advanced_layer_cap() {
        let positions: Vec<_> = (0..300).map(position).collect();
        let saved = CompositionTree::legacy_for_layers(&positions).unwrap();
        assert_eq!(saved.root().len(), 300);
        let restored: CompositionTree =
            serde_json::from_str(&serde_json::to_string(&saved).unwrap()).unwrap();
        assert_eq!(restored.root().len(), 300);

        // Huge sparse identities remain BTreeMap keys; no ID-sized storage is
        // created while resolving or validating the real 300-item stack.
        let runtime = restored
            .resolve(|saved| Some(live(u64::MAX - u64::from(saved.get()))))
            .unwrap();
        let expected: Vec<_> = positions
            .iter()
            .map(|saved| live(u64::MAX - u64::from(saved.get())))
            .collect();
        runtime.validate_for_layers(&expected).unwrap();
        assert_eq!(runtime.root().len(), 300);
        let captured = runtime
            .capture(|id| {
                u32::try_from(u64::MAX - id.get())
                    .ok()
                    .and_then(SavedLayerPosition::new)
            })
            .unwrap();
        assert_eq!(captured.root(), restored.root());
    }

    #[test]
    fn runtime_group_delete_marks_missing_output_and_never_reuses_id() {
        let mut receiver = group(10, &[]);
        receiver.matte = Some(ImageMatte {
            tap: crate::visual_rack::SavedImageTap {
                source: SavedImageSource::GroupOutput { group_id: id(20) },
                ..Default::default()
            },
            ..Default::default()
        });
        let donor = group(20, &[0]);
        let saved = CompositionTree::try_from_parts(
            vec![receiver, donor],
            vec![
                RootItem::Group { group_id: id(10) },
                RootItem::Group { group_id: id(20) },
            ],
            Some(21),
            0.5,
        )
        .unwrap();
        let mut runtime = saved.resolve(|_| Some(live(77))).unwrap();
        runtime.remove_group_ungroup(id(20)).unwrap();
        assert_eq!(
            runtime.group_output_status(id(20)),
            GroupOutputStatus::Missing
        );
        assert_eq!(
            runtime.group(id(10)).unwrap().matte.unwrap().tap.source,
            ResolvedImageSource::MissingGroupOutput(id(20))
        );
        assert!(matches!(
            runtime.root(),
            [RuntimeRootItem::Group { group_id }, RuntimeRootItem::Layer { layer_id, .. }]
                if *group_id == id(10) && *layer_id == live(77)
        ));
        let replacement = runtime
            .insert_empty_group(GroupName::new("replacement").unwrap(), 2)
            .unwrap();
        assert_eq!(replacement, id(21));
        let captured = runtime
            .capture(|layer_id| (layer_id == live(77)).then(|| position(0)))
            .unwrap();
        assert_eq!(captured.next_group_id_raw(), 22);
        assert_eq!(
            captured.group(id(10)).unwrap().matte.unwrap().tap.source,
            SavedImageSource::MissingGroupOutput { group_id: id(20) }
        );
    }
}
