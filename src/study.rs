//! Trusted, data-only Study document and instruction ABI.
//!
//! A Study is declarative data evaluated only through this fixed vocabulary.
//! It cannot carry native code, shader source, paths, URLs with authority,
//! network requests, processes, devices, or host mutations. A Study's own
//! license notice applies to that data document only; it does not grant or
//! imply a license to upstream portions of the host application.

#![allow(
    dead_code,
    reason = "M6-A freezes a data-only ABI before any trusted Study evaluator exists"
)]

use std::collections::BTreeSet;
use std::fmt;

use serde::{de, Deserialize, Deserializer, Serialize};

pub const STUDY_SCHEMA_VERSION: u16 = 1;
pub const STUDY_ABI_MAJOR: u16 = 1;
/// Highest append-only minor understood by this build. ABI 1.0 remains the
/// serialization default so constructing an old document cannot silently
/// opt into new opcode semantics.
pub const STUDY_ABI_MINOR: u16 = 1;
pub const STUDY_LEGACY_ABI_MINOR: u16 = 0;
pub const STUDY_MAX_DOCUMENT_BYTES: usize = 1024 * 1024;
pub const STUDY_MAX_INSTRUCTIONS: usize = 256;
pub const STUDY_MAX_REGISTERS: usize = 64;
pub const STUDY_MAX_CAPABILITIES: usize = 16;
pub const STUDY_MAX_NAME_BYTES: usize = 80;
pub const STUDY_MAX_AUTHOR_BYTES: usize = 80;
pub const STUDY_MAX_DESCRIPTION_BYTES: usize = 512;
pub const STUDY_MAX_LICENSE_ID_BYTES: usize = 64;
pub const STUDY_MAX_LICENSE_NOTICE_BYTES: usize = 1024;
/// Upper bound on `LoadHistoryColor { age }`, validated as `1..=23` with age
/// 0 rejected. **Ruled by the operator (R1, 2026-08-19): a Study age
/// addresses the committed clean-history ring's stored layer directly** —
/// age N is ring age N, so the cap is the ring's max age, derived from
/// `TEMPORAL_HISTORY_LEN - 1` exactly as `SYMMETRY_MAX_HISTORY_AGE` derives
/// it rather than restated as a literal. The previous literal `24` was the
/// ring *length* set where the ring *max age* was meant; it was narrowed
/// while no evaluator had ever given a document pixels, the only moment the
/// exact-equality ABI gate below makes that cheap. `LoadCurrentColor`
/// remains the only spelling of "now" — one meaning, one spelling — and the
/// alternative (age 1 duplicating it via an `age - 1` mapping) was
/// deliberately rejected. The validator bounds the authored age only; an
/// evaluator must additionally guard every age against the valid-sample
/// count exactly as `temporal_originals.wgsl` does, or a young program
/// reads unwritten texture content.
pub const STUDY_MAX_HISTORY_AGE: u8 = (crate::temporal::TEMPORAL_HISTORY_LEN - 1) as u8;
pub const STUDY_MAX_AUDIO_BANDS: u8 = 8;
pub const STUDY_MAX_FINITE_VALUE: f32 = 65_504.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StudyAuthority {
    pub native_code: bool,
    pub shader_source: bool,
    pub filesystem: bool,
    pub network: bool,
    pub process: bool,
    pub device: bool,
    pub host_mutation: bool,
}

pub const DATA_ONLY_STUDY_AUTHORITY: StudyAuthority = StudyAuthority {
    native_code: false,
    shader_source: false,
    filesystem: false,
    network: false,
    process: false,
    device: false,
    host_mutation: false,
};

/// The ABI gate is a **backward window on minor only**, per the operator's
/// R3 ruling (2026-08-19): validation accepts `major == STUDY_ABI_MAJOR &&
/// minor <= STUDY_ABI_MINOR` and rejects everything else. The window landed
/// with the CPU reference evaluator (`study_eval.rs`), the commit R3 named.
/// ABI 1.0 retains the R1 history-age and R2 determinism rulings byte for
/// byte; ABI 1.1 is the first append-only extension. Opcode codes are never
/// renumbered or reused (the `NodeKindTag` discipline). An evaluator that
/// executes 1.N can execute 1.0 by construction when growth is append-only;
/// a major bump stays a hard break.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudyAbiVersion {
    pub major: u16,
    pub minor: u16,
}

impl Default for StudyAbiVersion {
    fn default() -> Self {
        Self {
            major: STUDY_ABI_MAJOR,
            minor: STUDY_LEGACY_ABI_MINOR,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StudyPublicationBoundary {
    /// The declared license covers only the Study data. It does not grant,
    /// replace, or characterize the license of the host or its upstream code.
    StudyDataOnlyDoesNotLicenseHost,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudyLicenseNotice {
    pub identifier: String,
    pub notice: String,
    pub publication_boundary: StudyPublicationBoundary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudyMetadata {
    pub name: String,
    pub author: String,
    pub description: String,
    pub license: StudyLicenseNotice,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StudyCapability {
    CurrentColor,
    HistoryRead,
    MotionFieldRead,
    AudioFeatures,
    BeatPhase,
    DeterministicRandom,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StudyRegister(u8);

impl fmt::Debug for StudyRegister {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("StudyRegister")
            .field(&self.0)
            .finish()
    }
}

impl StudyRegister {
    pub const fn new(value: u8) -> Option<Self> {
        if (value as usize) < STUDY_MAX_REGISTERS {
            Some(Self(value))
        } else {
            None
        }
    }

    pub const fn get(self) -> u8 {
        self.0
    }
}

impl Serialize for StudyRegister {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u8(self.0)
    }
}

impl<'de> Deserialize<'de> for StudyRegister {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u8::deserialize(deserializer)?;
        Self::new(value).ok_or_else(|| {
            de::Error::custom(format_args!(
                "Study register {value} exceeds {} registers",
                STUDY_MAX_REGISTERS
            ))
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StudyValueType {
    Scalar,
    Vector2,
    Color,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum StudyInstruction {
    LoadCurrentColor {
        dst: StudyRegister,
    },
    LoadHistoryColor {
        dst: StudyRegister,
        age: u8,
    },
    /// Loads a `Vector2`. It remains a provable dead-end type in ABI 1.0;
    /// ABI 1.1 can project it through the explicitly appended vector ops.
    LoadMotionVector {
        dst: StudyRegister,
    },
    LoadAudioBand {
        dst: StudyRegister,
        band: u8,
    },
    LoadBeatPhase {
        dst: StudyRegister,
    },
    /// Carries only a `domain`; no seed field exists anywhere in the
    /// document. **Ruled by the operator (R2, 2026-08-19): the value is a
    /// domain-separated SplitMix64-style counter hash over (the Study ABI
    /// version, the document's canonical content digest, `domain`) — one
    /// constant per document, frame-independent.** Same document, same
    /// value, live and offline, forever; two domains can never perturb each
    /// other; nothing process-lifetime and no wall time may enter the hash
    /// (the Symmetry sector-table law). Time-varying noise is deliberately
    /// out of this opcode's meaning — that is what the LFO, audio, and beat
    /// sources are for — and an animated variant would be a future opcode
    /// under the R3 versioning law, never a redefinition of this one. The
    /// evaluator implements this ruling; until it lands the opcode remains
    /// valid, capability-gated data.
    LoadDeterministicRandom {
        dst: StudyRegister,
        domain: u32,
    },
    ConstantScalar {
        dst: StudyRegister,
        value: f32,
    },
    ConstantVector2 {
        dst: StudyRegister,
        value: [f32; 2],
    },
    ConstantColor {
        dst: StudyRegister,
        value: [f32; 4],
    },
    Add {
        dst: StudyRegister,
        left: StudyRegister,
        right: StudyRegister,
    },
    Subtract {
        dst: StudyRegister,
        left: StudyRegister,
        right: StudyRegister,
    },
    Multiply {
        dst: StudyRegister,
        left: StudyRegister,
        right: StudyRegister,
    },
    Mix {
        dst: StudyRegister,
        a: StudyRegister,
        b: StudyRegister,
        amount: StudyRegister,
    },
    Clamp01 {
        dst: StudyRegister,
        input: StudyRegister,
    },
    HueRotate {
        dst: StudyRegister,
        color: StudyRegister,
        turns: StudyRegister,
    },
    /// ABI 1.1: project a motion/vector lane onto its X component.
    VectorX {
        dst: StudyRegister,
        input: StudyRegister,
    },
    /// ABI 1.1: project a motion/vector lane onto its Y component.
    VectorY {
        dst: StudyRegister,
        input: StudyRegister,
    },
    /// ABI 1.1: bounded Euclidean magnitude.
    VectorMagnitude {
        dst: StudyRegister,
        input: StudyRegister,
    },
    /// ABI 1.1: unit direction; the zero vector maps to `[0, 0]`.
    VectorNormalizedDirection {
        dst: StudyRegister,
        input: StudyRegister,
    },
    /// ABI 1.1: dot a vector against one finite authored constant vector.
    VectorDotConstant {
        dst: StudyRegister,
        input: StudyRegister,
        vector: [f32; 2],
    },
    /// ABI 1.1: clamp a scalar to `[0,1]` and mix two typed colors.
    ScalarToColor {
        dst: StudyRegister,
        scalar: StudyRegister,
        low: StudyRegister,
        high: StudyRegister,
    },
    OutputColor {
        color: StudyRegister,
    },
}

impl StudyInstruction {
    fn destination(&self) -> Option<StudyRegister> {
        match self {
            Self::LoadCurrentColor { dst }
            | Self::LoadHistoryColor { dst, .. }
            | Self::LoadMotionVector { dst }
            | Self::LoadAudioBand { dst, .. }
            | Self::LoadBeatPhase { dst }
            | Self::LoadDeterministicRandom { dst, .. }
            | Self::ConstantScalar { dst, .. }
            | Self::ConstantVector2 { dst, .. }
            | Self::ConstantColor { dst, .. }
            | Self::Add { dst, .. }
            | Self::Subtract { dst, .. }
            | Self::Multiply { dst, .. }
            | Self::Mix { dst, .. }
            | Self::Clamp01 { dst, .. }
            | Self::HueRotate { dst, .. }
            | Self::VectorX { dst, .. }
            | Self::VectorY { dst, .. }
            | Self::VectorMagnitude { dst, .. }
            | Self::VectorNormalizedDirection { dst, .. }
            | Self::VectorDotConstant { dst, .. }
            | Self::ScalarToColor { dst, .. } => Some(*dst),
            Self::OutputColor { .. } => None,
        }
    }

    /// The capability an instruction consumes. This match is deliberately
    /// exhaustive with no wildcard: `destination()` and the validator both
    /// force an arm for every new opcode, and this — the authority boundary —
    /// must force one too. A wildcard here was the exact mechanism by which
    /// a future `Load*` opcode could read a host input while the canonical
    /// capability law silently derived "none declared, none required";
    /// authority that depends on someone remembering is not a boundary.
    fn capability(&self) -> Option<StudyCapability> {
        match self {
            Self::LoadCurrentColor { .. } => Some(StudyCapability::CurrentColor),
            Self::LoadHistoryColor { .. } => Some(StudyCapability::HistoryRead),
            Self::LoadMotionVector { .. } => Some(StudyCapability::MotionFieldRead),
            Self::LoadAudioBand { .. } => Some(StudyCapability::AudioFeatures),
            Self::LoadBeatPhase { .. } => Some(StudyCapability::BeatPhase),
            Self::LoadDeterministicRandom { .. } => Some(StudyCapability::DeterministicRandom),
            Self::ConstantScalar { .. }
            | Self::ConstantVector2 { .. }
            | Self::ConstantColor { .. }
            | Self::Add { .. }
            | Self::Subtract { .. }
            | Self::Multiply { .. }
            | Self::Mix { .. }
            | Self::Clamp01 { .. }
            | Self::HueRotate { .. }
            | Self::VectorX { .. }
            | Self::VectorY { .. }
            | Self::VectorMagnitude { .. }
            | Self::VectorNormalizedDirection { .. }
            | Self::VectorDotConstant { .. }
            | Self::ScalarToColor { .. }
            | Self::OutputColor { .. } => None,
        }
    }

    const fn minimum_abi_minor(&self) -> u16 {
        match self {
            Self::VectorX { .. }
            | Self::VectorY { .. }
            | Self::VectorMagnitude { .. }
            | Self::VectorNormalizedDirection { .. }
            | Self::VectorDotConstant { .. }
            | Self::ScalarToColor { .. } => 1,
            Self::LoadCurrentColor { .. }
            | Self::LoadHistoryColor { .. }
            | Self::LoadMotionVector { .. }
            | Self::LoadAudioBand { .. }
            | Self::LoadBeatPhase { .. }
            | Self::LoadDeterministicRandom { .. }
            | Self::ConstantScalar { .. }
            | Self::ConstantVector2 { .. }
            | Self::ConstantColor { .. }
            | Self::Add { .. }
            | Self::Subtract { .. }
            | Self::Multiply { .. }
            | Self::Mix { .. }
            | Self::Clamp01 { .. }
            | Self::HueRotate { .. }
            | Self::OutputColor { .. } => 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StudyDocument {
    pub schema_version: u16,
    pub abi: StudyAbiVersion,
    pub metadata: StudyMetadata,
    pub capabilities: Vec<StudyCapability>,
    pub instructions: Vec<StudyInstruction>,
}

impl StudyDocument {
    pub fn validate(&self) -> Result<(), StudyError> {
        if self.schema_version != STUDY_SCHEMA_VERSION {
            return Err(StudyError::UnsupportedSchemaVersion(self.schema_version));
        }
        if self.abi.major != STUDY_ABI_MAJOR || self.abi.minor > STUDY_ABI_MINOR {
            return Err(StudyError::UnsupportedAbi(self.abi));
        }
        validate_metadata(&self.metadata)?;
        if self.capabilities.len() > STUDY_MAX_CAPABILITIES {
            return Err(StudyError::TooManyCapabilities(self.capabilities.len()));
        }
        if self.instructions.is_empty() || self.instructions.len() > STUDY_MAX_INSTRUCTIONS {
            return Err(StudyError::InstructionCount(self.instructions.len()));
        }

        let mut register_types = [None; STUDY_MAX_REGISTERS];
        let mut derived_capabilities = BTreeSet::new();
        let mut output_count = 0_usize;
        for (index, instruction) in self.instructions.iter().enumerate() {
            let required_minor = instruction.minimum_abi_minor();
            if self.abi.minor < required_minor {
                return Err(StudyError::OpcodeRequiresAbi {
                    instruction: index,
                    required_minor,
                    declared_minor: self.abi.minor,
                });
            }
            if let Some(capability) = instruction.capability() {
                derived_capabilities.insert(capability);
            }
            if let Some(destination) = instruction.destination() {
                if register_types[usize::from(destination.get())].is_some() {
                    return Err(StudyError::RegisterReassigned(destination));
                }
            }
            match instruction {
                StudyInstruction::LoadCurrentColor { dst }
                | StudyInstruction::LoadHistoryColor { dst, .. }
                | StudyInstruction::ConstantColor { dst, .. } => {
                    if let StudyInstruction::LoadHistoryColor { age, .. } = instruction {
                        if *age == 0 || *age > STUDY_MAX_HISTORY_AGE {
                            return Err(StudyError::HistoryAge(*age));
                        }
                    }
                    if let StudyInstruction::ConstantColor { value, .. } = instruction {
                        validate_finite_values(value)?;
                    }
                    define_register(&mut register_types, *dst, StudyValueType::Color);
                }
                StudyInstruction::LoadMotionVector { dst }
                | StudyInstruction::ConstantVector2 { dst, .. } => {
                    if let StudyInstruction::ConstantVector2 { value, .. } = instruction {
                        validate_finite_values(value)?;
                    }
                    define_register(&mut register_types, *dst, StudyValueType::Vector2);
                }
                StudyInstruction::LoadAudioBand { dst, band } => {
                    if *band >= STUDY_MAX_AUDIO_BANDS {
                        return Err(StudyError::AudioBand(*band));
                    }
                    define_register(&mut register_types, *dst, StudyValueType::Scalar);
                }
                StudyInstruction::LoadBeatPhase { dst }
                | StudyInstruction::LoadDeterministicRandom { dst, .. } => {
                    define_register(&mut register_types, *dst, StudyValueType::Scalar);
                }
                StudyInstruction::ConstantScalar { dst, value } => {
                    validate_finite_values(&[*value])?;
                    define_register(&mut register_types, *dst, StudyValueType::Scalar);
                }
                StudyInstruction::Add { dst, left, right }
                | StudyInstruction::Subtract { dst, left, right }
                | StudyInstruction::Multiply { dst, left, right } => {
                    let left_type = read_register(&register_types, *left)?;
                    let right_type = read_register(&register_types, *right)?;
                    if left_type != right_type {
                        return Err(StudyError::TypeMismatch {
                            instruction: index,
                            expected: left_type,
                            observed: right_type,
                        });
                    }
                    define_register(&mut register_types, *dst, left_type);
                }
                StudyInstruction::Mix { dst, a, b, amount } => {
                    let a_type = read_register(&register_types, *a)?;
                    let b_type = read_register(&register_types, *b)?;
                    if a_type != b_type {
                        return Err(StudyError::TypeMismatch {
                            instruction: index,
                            expected: a_type,
                            observed: b_type,
                        });
                    }
                    require_type(&register_types, *amount, StudyValueType::Scalar, index)?;
                    define_register(&mut register_types, *dst, a_type);
                }
                StudyInstruction::Clamp01 { dst, input } => {
                    let input_type = read_register(&register_types, *input)?;
                    define_register(&mut register_types, *dst, input_type);
                }
                StudyInstruction::HueRotate { dst, color, turns } => {
                    require_type(&register_types, *color, StudyValueType::Color, index)?;
                    require_type(&register_types, *turns, StudyValueType::Scalar, index)?;
                    define_register(&mut register_types, *dst, StudyValueType::Color);
                }
                StudyInstruction::VectorX { dst, input }
                | StudyInstruction::VectorY { dst, input }
                | StudyInstruction::VectorMagnitude { dst, input } => {
                    require_type(&register_types, *input, StudyValueType::Vector2, index)?;
                    define_register(&mut register_types, *dst, StudyValueType::Scalar);
                }
                StudyInstruction::VectorNormalizedDirection { dst, input } => {
                    require_type(&register_types, *input, StudyValueType::Vector2, index)?;
                    define_register(&mut register_types, *dst, StudyValueType::Vector2);
                }
                StudyInstruction::VectorDotConstant { dst, input, vector } => {
                    require_type(&register_types, *input, StudyValueType::Vector2, index)?;
                    validate_finite_values(vector)?;
                    define_register(&mut register_types, *dst, StudyValueType::Scalar);
                }
                StudyInstruction::ScalarToColor {
                    dst,
                    scalar,
                    low,
                    high,
                } => {
                    require_type(&register_types, *scalar, StudyValueType::Scalar, index)?;
                    require_type(&register_types, *low, StudyValueType::Color, index)?;
                    require_type(&register_types, *high, StudyValueType::Color, index)?;
                    define_register(&mut register_types, *dst, StudyValueType::Color);
                }
                StudyInstruction::OutputColor { color } => {
                    output_count += 1;
                    if index + 1 != self.instructions.len() {
                        return Err(StudyError::OutputNotFinal(index));
                    }
                    require_type(&register_types, *color, StudyValueType::Color, index)?;
                }
            }
        }
        if output_count != 1 {
            return Err(StudyError::OutputCount(output_count));
        }

        let declared = self.capabilities.iter().copied().collect::<BTreeSet<_>>();
        if declared.len() != self.capabilities.len()
            || self
                .capabilities
                .iter()
                .copied()
                .ne(declared.iter().copied())
            || declared != derived_capabilities
        {
            return Err(StudyError::CapabilitiesNotCanonical);
        }
        Ok(())
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>, StudyError> {
        self.validate()?;
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| StudyError::Serialization(error.to_string()))?;
        validate_document_bytes(bytes.len())?;
        Ok(bytes)
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, StudyError> {
        validate_document_bytes(bytes.len())?;
        serde_json::from_slice(bytes).map_err(|error| StudyError::Serialization(error.to_string()))
    }

    pub fn to_yaml(&self) -> Result<String, StudyError> {
        self.validate()?;
        let yaml = serde_yaml::to_string(self)
            .map_err(|error| StudyError::Serialization(error.to_string()))?;
        validate_document_bytes(yaml.len())?;
        Ok(yaml)
    }

    pub fn from_yaml(yaml: &str) -> Result<Self, StudyError> {
        validate_document_bytes(yaml.len())?;
        serde_yaml::from_str(yaml).map_err(|error| StudyError::Serialization(error.to_string()))
    }

    pub const fn authority(&self) -> StudyAuthority {
        DATA_ONLY_STUDY_AUTHORITY
    }
}

fn validate_document_bytes(bytes: usize) -> Result<(), StudyError> {
    if bytes == 0 || bytes > STUDY_MAX_DOCUMENT_BYTES {
        Err(StudyError::DocumentBytes(bytes))
    } else {
        Ok(())
    }
}

fn validate_metadata(metadata: &StudyMetadata) -> Result<(), StudyError> {
    validate_text("name", &metadata.name, STUDY_MAX_NAME_BYTES, false)?;
    validate_text("author", &metadata.author, STUDY_MAX_AUTHOR_BYTES, false)?;
    validate_text(
        "description",
        &metadata.description,
        STUDY_MAX_DESCRIPTION_BYTES,
        true,
    )?;
    validate_text(
        "license identifier",
        &metadata.license.identifier,
        STUDY_MAX_LICENSE_ID_BYTES,
        false,
    )?;
    validate_text(
        "license notice",
        &metadata.license.notice,
        STUDY_MAX_LICENSE_NOTICE_BYTES,
        true,
    )?;
    Ok(())
}

fn validate_text(
    field: &'static str,
    value: &str,
    max_bytes: usize,
    empty_allowed: bool,
) -> Result<(), StudyError> {
    if (!empty_allowed && value.is_empty())
        || value.len() > max_bytes
        || value.trim() != value
        || value
            .chars()
            .any(|character| character.is_control() && character != '\n' && character != '\t')
    {
        return Err(StudyError::InvalidText { field, max_bytes });
    }
    Ok(())
}

fn validate_finite_values(values: &[f32]) -> Result<(), StudyError> {
    if values
        .iter()
        .any(|value| !value.is_finite() || value.abs() > STUDY_MAX_FINITE_VALUE)
    {
        Err(StudyError::NonFiniteOrUnboundedConstant)
    } else {
        Ok(())
    }
}

fn define_register(
    registers: &mut [Option<StudyValueType>; STUDY_MAX_REGISTERS],
    register: StudyRegister,
    value_type: StudyValueType,
) {
    registers[usize::from(register.get())] = Some(value_type);
}

fn read_register(
    registers: &[Option<StudyValueType>; STUDY_MAX_REGISTERS],
    register: StudyRegister,
) -> Result<StudyValueType, StudyError> {
    registers[usize::from(register.get())].ok_or(StudyError::RegisterReadBeforeDefinition(register))
}

fn require_type(
    registers: &[Option<StudyValueType>; STUDY_MAX_REGISTERS],
    register: StudyRegister,
    expected: StudyValueType,
    instruction: usize,
) -> Result<(), StudyError> {
    let observed = read_register(registers, register)?;
    if observed == expected {
        Ok(())
    } else {
        Err(StudyError::TypeMismatch {
            instruction,
            expected,
            observed,
        })
    }
}

struct BoundedCapabilities(Vec<StudyCapability>);

impl<'de> Deserialize<'de> for BoundedCapabilities {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct Visitor;
        impl<'de> de::Visitor<'de> for Visitor {
            type Value = BoundedCapabilities;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    formatter,
                    "at most {STUDY_MAX_CAPABILITIES} Study capabilities"
                )
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                let mut values = Vec::with_capacity(
                    sequence
                        .size_hint()
                        .unwrap_or(0)
                        .min(STUDY_MAX_CAPABILITIES),
                );
                while values.len() < STUDY_MAX_CAPABILITIES {
                    let Some(value) = sequence.next_element()? else {
                        return Ok(BoundedCapabilities(values));
                    };
                    values.push(value);
                }
                if sequence.next_element::<de::IgnoredAny>()?.is_some() {
                    return Err(de::Error::custom("too many Study capabilities"));
                }
                Ok(BoundedCapabilities(values))
            }
        }
        deserializer.deserialize_seq(Visitor)
    }
}

struct BoundedInstructions(Vec<StudyInstruction>);

impl<'de> Deserialize<'de> for BoundedInstructions {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct Visitor;
        impl<'de> de::Visitor<'de> for Visitor {
            type Value = BoundedInstructions;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    formatter,
                    "at most {STUDY_MAX_INSTRUCTIONS} Study instructions"
                )
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                let mut values = Vec::with_capacity(
                    sequence
                        .size_hint()
                        .unwrap_or(0)
                        .min(STUDY_MAX_INSTRUCTIONS),
                );
                while values.len() < STUDY_MAX_INSTRUCTIONS {
                    let Some(value) = sequence.next_element()? else {
                        return Ok(BoundedInstructions(values));
                    };
                    values.push(value);
                }
                if sequence.next_element::<de::IgnoredAny>()?.is_some() {
                    return Err(de::Error::custom("too many Study instructions"));
                }
                Ok(BoundedInstructions(values))
            }
        }
        deserializer.deserialize_seq(Visitor)
    }
}

impl<'de> Deserialize<'de> for StudyDocument {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Raw {
            schema_version: u16,
            abi: StudyAbiVersion,
            metadata: StudyMetadata,
            capabilities: BoundedCapabilities,
            instructions: BoundedInstructions,
        }

        let raw = Raw::deserialize(deserializer)?;
        let document = Self {
            schema_version: raw.schema_version,
            abi: raw.abi,
            metadata: raw.metadata,
            capabilities: raw.capabilities.0,
            instructions: raw.instructions.0,
        };
        document.validate().map_err(de::Error::custom)?;
        Ok(document)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum StudyError {
    UnsupportedSchemaVersion(u16),
    UnsupportedAbi(StudyAbiVersion),
    DocumentBytes(usize),
    InvalidText {
        field: &'static str,
        max_bytes: usize,
    },
    TooManyCapabilities(usize),
    CapabilitiesNotCanonical,
    InstructionCount(usize),
    OpcodeRequiresAbi {
        instruction: usize,
        required_minor: u16,
        declared_minor: u16,
    },
    RegisterReassigned(StudyRegister),
    RegisterReadBeforeDefinition(StudyRegister),
    TypeMismatch {
        instruction: usize,
        expected: StudyValueType,
        observed: StudyValueType,
    },
    HistoryAge(u8),
    AudioBand(u8),
    NonFiniteOrUnboundedConstant,
    OutputNotFinal(usize),
    OutputCount(usize),
    Serialization(String),
}

impl fmt::Display for StudyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion(version) => {
                write!(formatter, "unsupported Study schema version {version}")
            }
            Self::UnsupportedAbi(version) => write!(
                formatter,
                "unsupported Study ABI {}.{}",
                version.major, version.minor
            ),
            Self::DocumentBytes(bytes) => write!(
                formatter,
                "Study document is {bytes} bytes; limit is {STUDY_MAX_DOCUMENT_BYTES}"
            ),
            Self::InvalidText { field, max_bytes } => {
                write!(
                    formatter,
                    "invalid Study {field}; limit is {max_bytes} bytes"
                )
            }
            Self::TooManyCapabilities(count) => write!(
                formatter,
                "Study declares {count} capabilities; limit is {STUDY_MAX_CAPABILITIES}"
            ),
            Self::CapabilitiesNotCanonical => formatter.write_str(
                "Study capabilities must be unique, sorted, and exactly match instruction use",
            ),
            Self::InstructionCount(count) => write!(
                formatter,
                "Study has {count} instructions; valid range is 1..={STUDY_MAX_INSTRUCTIONS}"
            ),
            Self::OpcodeRequiresAbi {
                instruction,
                required_minor,
                declared_minor,
            } => write!(
                formatter,
                "Study instruction {instruction} requires ABI 1.{required_minor}, document declares 1.{declared_minor}"
            ),
            Self::RegisterReassigned(register) => {
                write!(
                    formatter,
                    "Study register {} is assigned twice",
                    register.get()
                )
            }
            Self::RegisterReadBeforeDefinition(register) => write!(
                formatter,
                "Study register {} is read before definition",
                register.get()
            ),
            Self::TypeMismatch {
                instruction,
                expected,
                observed,
            } => write!(
                formatter,
                "Study instruction {instruction} expects {expected:?}, observed {observed:?}"
            ),
            Self::HistoryAge(age) => write!(
                formatter,
                "Study history age {age} is outside 1..={STUDY_MAX_HISTORY_AGE}"
            ),
            Self::AudioBand(band) => write!(
                formatter,
                "Study audio band {band} is outside 0..{STUDY_MAX_AUDIO_BANDS}"
            ),
            Self::NonFiniteOrUnboundedConstant => formatter.write_str(
                "Study constants must be finite and within the RGBA16Float numeric envelope",
            ),
            Self::OutputNotFinal(index) => {
                write!(
                    formatter,
                    "Study output at instruction {index} is not final"
                )
            }
            Self::OutputCount(count) => {
                write!(
                    formatter,
                    "Study must contain exactly one output; observed {count}"
                )
            }
            Self::Serialization(error) => write!(formatter, "decode Study document: {error}"),
        }
    }
}

impl std::error::Error for StudyError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn register(value: u8) -> StudyRegister {
        StudyRegister::new(value).unwrap()
    }

    /// Pins the complete opcode-to-capability table, one sample per opcode.
    /// The compiler now refuses a new opcode with no `capability()` arm; this
    /// test makes the *value* of every arm a reviewed fact as well, so an
    /// authority change can never ride in as an incidental diff.
    #[test]
    fn every_opcode_declares_its_capability_explicitly() {
        let dst = register(0);
        let table: [(StudyInstruction, Option<StudyCapability>); 22] = [
            (
                StudyInstruction::LoadCurrentColor { dst },
                Some(StudyCapability::CurrentColor),
            ),
            (
                StudyInstruction::LoadHistoryColor { dst, age: 1 },
                Some(StudyCapability::HistoryRead),
            ),
            (
                StudyInstruction::LoadMotionVector { dst },
                Some(StudyCapability::MotionFieldRead),
            ),
            (
                StudyInstruction::LoadAudioBand { dst, band: 0 },
                Some(StudyCapability::AudioFeatures),
            ),
            (
                StudyInstruction::LoadBeatPhase { dst },
                Some(StudyCapability::BeatPhase),
            ),
            (
                StudyInstruction::LoadDeterministicRandom { dst, domain: 0 },
                Some(StudyCapability::DeterministicRandom),
            ),
            (StudyInstruction::ConstantScalar { dst, value: 0.0 }, None),
            (
                StudyInstruction::ConstantVector2 {
                    dst,
                    value: [0.0, 0.0],
                },
                None,
            ),
            (
                StudyInstruction::ConstantColor {
                    dst,
                    value: [0.0; 4],
                },
                None,
            ),
            (
                StudyInstruction::Add {
                    dst,
                    left: dst,
                    right: dst,
                },
                None,
            ),
            (
                StudyInstruction::Subtract {
                    dst,
                    left: dst,
                    right: dst,
                },
                None,
            ),
            (
                StudyInstruction::Multiply {
                    dst,
                    left: dst,
                    right: dst,
                },
                None,
            ),
            (
                StudyInstruction::Mix {
                    dst,
                    a: dst,
                    b: dst,
                    amount: dst,
                },
                None,
            ),
            (StudyInstruction::Clamp01 { dst, input: dst }, None),
            (
                StudyInstruction::HueRotate {
                    dst,
                    color: dst,
                    turns: dst,
                },
                None,
            ),
            (StudyInstruction::VectorX { dst, input: dst }, None),
            (StudyInstruction::VectorY { dst, input: dst }, None),
            (StudyInstruction::VectorMagnitude { dst, input: dst }, None),
            (
                StudyInstruction::VectorNormalizedDirection { dst, input: dst },
                None,
            ),
            (
                StudyInstruction::VectorDotConstant {
                    dst,
                    input: dst,
                    vector: [1.0, 0.0],
                },
                None,
            ),
            (
                StudyInstruction::ScalarToColor {
                    dst,
                    scalar: dst,
                    low: dst,
                    high: dst,
                },
                None,
            ),
            (StudyInstruction::OutputColor { color: dst }, None),
        ];
        for (instruction, expected) in table {
            assert_eq!(
                instruction.capability(),
                expected,
                "capability table changed for {instruction:?}"
            );
        }
    }

    fn valid_document() -> StudyDocument {
        StudyDocument {
            schema_version: STUDY_SCHEMA_VERSION,
            abi: StudyAbiVersion::default(),
            metadata: StudyMetadata {
                name: "Field tint".into(),
                author: "Study author".into(),
                description: "Data-only fixture".into(),
                license: StudyLicenseNotice {
                    identifier: "CC0-1.0".into(),
                    notice: "This notice covers the Study data only.".into(),
                    publication_boundary: StudyPublicationBoundary::StudyDataOnlyDoesNotLicenseHost,
                },
            },
            capabilities: vec![
                StudyCapability::CurrentColor,
                StudyCapability::AudioFeatures,
            ],
            instructions: vec![
                StudyInstruction::LoadCurrentColor { dst: register(0) },
                StudyInstruction::LoadAudioBand {
                    dst: register(1),
                    band: 0,
                },
                StudyInstruction::ConstantColor {
                    dst: register(2),
                    value: [0.2, 0.1, 0.8, 1.0],
                },
                StudyInstruction::Mix {
                    dst: register(3),
                    a: register(0),
                    b: register(2),
                    amount: register(1),
                },
                StudyInstruction::OutputColor { color: register(3) },
            ],
        }
    }

    #[test]
    fn trusted_document_round_trips_closed_json_and_yaml() {
        let document = valid_document();
        document.validate().unwrap();
        let json = document.to_json_bytes().unwrap();
        assert_eq!(StudyDocument::from_json_bytes(&json).unwrap(), document);
        let yaml = document.to_yaml().unwrap();
        assert_eq!(StudyDocument::from_yaml(&yaml).unwrap(), document);
        assert_eq!(document.authority(), DATA_ONLY_STUDY_AUTHORITY);
    }

    #[test]
    fn authority_and_capability_vocabulary_cannot_grant_external_power() {
        let authority = DATA_ONLY_STUDY_AUTHORITY;
        assert!(!authority.native_code);
        assert!(!authority.shader_source);
        assert!(!authority.filesystem);
        assert!(!authority.network);
        assert!(!authority.process);
        assert!(!authority.device);
        assert!(!authority.host_mutation);

        let capabilities = serde_json::to_string(&[
            StudyCapability::CurrentColor,
            StudyCapability::HistoryRead,
            StudyCapability::MotionFieldRead,
            StudyCapability::AudioFeatures,
            StudyCapability::BeatPhase,
            StudyCapability::DeterministicRandom,
        ])
        .unwrap();
        for forbidden in ["filesystem", "network", "native", "shader", "process"] {
            assert!(!capabilities.contains(forbidden));
        }
    }

    #[test]
    fn unknown_fields_versions_and_publication_boundary_fail_closed() {
        let document = valid_document();
        let mut json = serde_json::to_value(&document).unwrap();
        json["unexpected"] = serde_json::json!(true);
        assert!(serde_json::from_value::<StudyDocument>(json).is_err());

        let mut nested = serde_json::to_value(&document).unwrap();
        nested["metadata"]["license"]["unexpected"] = serde_json::json!(true);
        assert!(serde_json::from_value::<StudyDocument>(nested).is_err());

        let mut missing_boundary = serde_json::to_value(&document).unwrap();
        missing_boundary["metadata"]["license"]
            .as_object_mut()
            .unwrap()
            .remove("publication_boundary");
        assert!(serde_json::from_value::<StudyDocument>(missing_boundary).is_err());

        let mut wrong_abi = document;
        wrong_abi.abi.major = 2;
        assert!(matches!(
            wrong_abi.validate(),
            Err(StudyError::UnsupportedAbi(_))
        ));
    }

    #[test]
    fn the_abi_gate_is_a_backward_minor_window_by_the_operator_ruling() {
        // R3: accept major == current && minor <= current; reject a newer
        // minor (forward) and any other major (both directions hard).
        let mut document = valid_document();
        document.validate().unwrap();
        document.abi.minor = STUDY_ABI_MINOR + 1;
        assert!(matches!(
            document.validate(),
            Err(StudyError::UnsupportedAbi(_))
        ));
        document.abi = StudyAbiVersion {
            major: STUDY_ABI_MAJOR + 1,
            minor: 0,
        };
        assert!(matches!(
            document.validate(),
            Err(StudyError::UnsupportedAbi(_))
        ));
        // ABI 1.0 remains the construction/serialization default and is still
        // accepted byte-for-byte; 1.1 is an explicit opt-in.
        assert_eq!(StudyAbiVersion::default().minor, STUDY_LEGACY_ABI_MINOR);
        document.abi = StudyAbiVersion::default();
        document.validate().unwrap();
        document.abi = StudyAbiVersion {
            major: STUDY_ABI_MAJOR,
            minor: STUDY_ABI_MINOR,
        };
        document.validate().unwrap();

        document.instructions.insert(
            document.instructions.len() - 1,
            StudyInstruction::VectorMagnitude {
                dst: register(12),
                input: register(12),
            },
        );
        document.abi.minor = STUDY_LEGACY_ABI_MINOR;
        assert!(matches!(
            document.validate(),
            Err(StudyError::OpcodeRequiresAbi {
                required_minor: 1,
                declared_minor: 0,
                ..
            })
        ));
    }

    #[test]
    fn bounded_deserializer_rejects_instruction_bombs_before_acceptance() {
        let mut json = serde_json::to_value(valid_document()).unwrap();
        json["instructions"] = serde_json::Value::Array(
            (0..=STUDY_MAX_INSTRUCTIONS)
                .map(|index| {
                    serde_json::json!({
                        "op":"constant_scalar",
                        "dst": index % STUDY_MAX_REGISTERS,
                        "value": 0.0
                    })
                })
                .collect(),
        );
        assert!(serde_json::from_value::<StudyDocument>(json).is_err());
        assert!(matches!(
            StudyDocument::from_json_bytes(&vec![b' '; STUDY_MAX_DOCUMENT_BYTES + 1]),
            Err(StudyError::DocumentBytes(_))
        ));
    }

    #[test]
    fn ssa_types_capabilities_and_output_are_strict() {
        let mut document = valid_document();
        document.capabilities.reverse();
        assert_eq!(
            document.validate(),
            Err(StudyError::CapabilitiesNotCanonical)
        );

        document = valid_document();
        document.instructions[3] = StudyInstruction::Add {
            dst: register(3),
            left: register(0),
            right: register(1),
        };
        assert!(matches!(
            document.validate(),
            Err(StudyError::TypeMismatch { instruction: 3, .. })
        ));

        document = valid_document();
        document.instructions.swap(3, 4);
        assert_eq!(document.validate(), Err(StudyError::OutputNotFinal(3)));

        document = valid_document();
        document.instructions[2] = StudyInstruction::ConstantColor {
            dst: register(2),
            value: [f32::NAN, 0.0, 0.0, 1.0],
        };
        assert_eq!(
            document.validate(),
            Err(StudyError::NonFiniteOrUnboundedConstant)
        );
    }

    #[test]
    fn the_history_age_cap_is_the_ring_max_age_by_the_operator_ruling() {
        // R1: a Study age addresses the committed ring's stored layer
        // directly, so the cap derives from the ring and 23 is the last
        // valid age — 24, the old ring-length literal, is rejected.
        assert_eq!(STUDY_MAX_HISTORY_AGE, 23);
        let mut document = valid_document();
        document.capabilities = vec![StudyCapability::HistoryRead];
        document.instructions = vec![
            StudyInstruction::LoadHistoryColor {
                dst: register(0),
                age: STUDY_MAX_HISTORY_AGE,
            },
            StudyInstruction::OutputColor { color: register(0) },
        ];
        document.validate().unwrap();
        document.instructions[0] = StudyInstruction::LoadHistoryColor {
            dst: register(0),
            age: 24,
        };
        assert_eq!(document.validate(), Err(StudyError::HistoryAge(24)));
    }

    #[test]
    fn history_audio_and_register_caps_are_enforced() {
        let mut document = valid_document();
        document.capabilities = vec![StudyCapability::HistoryRead];
        document.instructions = vec![
            StudyInstruction::LoadHistoryColor {
                dst: register(0),
                age: STUDY_MAX_HISTORY_AGE + 1,
            },
            StudyInstruction::OutputColor { color: register(0) },
        ];
        assert_eq!(
            document.validate(),
            Err(StudyError::HistoryAge(STUDY_MAX_HISTORY_AGE + 1))
        );
        assert!(serde_json::from_str::<StudyRegister>("64").is_err());

        document.capabilities = vec![StudyCapability::AudioFeatures];
        document.instructions = vec![
            StudyInstruction::LoadAudioBand {
                dst: register(0),
                band: STUDY_MAX_AUDIO_BANDS,
            },
            StudyInstruction::ConstantColor {
                dst: register(1),
                value: [0.0, 0.0, 0.0, 1.0],
            },
            StudyInstruction::OutputColor { color: register(1) },
        ];
        assert_eq!(
            document.validate(),
            Err(StudyError::AudioBand(STUDY_MAX_AUDIO_BANDS))
        );
    }
}
