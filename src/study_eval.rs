//! The pure CPU reference evaluator for data-only Studies.
//!
//! This module gives the Study ABI's opcodes their meaning under the
//! operator's rulings (R1, R2, R3 — 2026-08-19, recorded on the opcodes in
//! `study.rs`). It is the independent reference a future WGSL interpreter is
//! checked against, so it deliberately has no `wgpu`, clock, filesystem, or
//! UI dependency — the `gesture.rs` shape. Shader-source generation remains
//! permanently refused by `StudyAuthority`; the GPU half, when it lands, is
//! a fixed, pre-compiled interpreter over a bounded instruction buffer.
//!
//! Semantics are anchored to existing program law rather than invented:
//! - `LoadHistoryColor` guards ages against the valid-sample count exactly
//!   as `temporal_originals.wgsl` does — a depth clamp to
//!   `valid_history - 1`, with the virtual current image at depth zero — so
//!   a young program can never read unwritten texture content.
//! - `HueRotate` mirrors `rack_node.wgsl`'s HSL round trip line for line,
//!   with the shader's own `fract` wrap, so the eventual GPU interpreter can
//!   share literal math with the racks.
//! - Every computed component passes one bound law: a non-finite result
//!   lands on the documented neutral `0.0` (never a clamped extreme), and a
//!   finite one clamps to `±STUDY_MAX_FINITE_VALUE` — the same bound the
//!   validator already imposes on constants, applied uniformly so no
//!   instruction chain can escape the representable range the ABI promises.

// The S10b interpreter consumes this module (`renderer/study.rs` encodes
// compiled studies for the fixed WGSL interpreter and its fixtures prove
// CPU/GPU agreement), but the whole chain stays unreachable from the frame
// loop until a Study gains an authored audience surface — where it plugs
// into the composition is a product decision the operator has not yet
// opened. This allow is scoped to exactly that window and comes out with
// the authored-surface tranche.
#![allow(dead_code)]

use bytemuck::Zeroable;
use sha2::{Digest, Sha256};

use crate::study::{
    StudyCapability, StudyDocument, StudyError, StudyInstruction, StudyRegister, STUDY_ABI_MAJOR,
    STUDY_ABI_MINOR, STUDY_MAX_AUDIO_BANDS, STUDY_MAX_FINITE_VALUE, STUDY_MAX_HISTORY_AGE,
    STUDY_MAX_REGISTERS,
};

/// Append-only version of the evaluation semantics themselves. Changing any
/// law in this module — the bound, the guard, the hash layout, the hue math —
/// requires bumping this and revisiting the ABI's R3 window. Version 2 (the
/// S10b interpreter tranche, days after version 1 and before any consumer
/// existed) added the HueRotate unorm input clamp so the CPU and GPU halves
/// agree everywhere WGSL's non-finite handling is implementation-defined.
pub const STUDY_EVAL_ALGORITHM_VERSION: u16 = 2;

/// Domain-separation lanes for the R2 deterministic-random hash. The layout
/// is frozen: the first eight canonical-digest bytes little-endian, XOR the
/// ABI lane, XOR the domain lane, XOR the tag, through the SplitMix64
/// finalizer `symmetry.rs` already uses.
const STUDY_RANDOM_TAG: u64 = 0x5354_5544_5952_4e44; // "STUDYRND"
const STUDY_RANDOM_ABI_MIX: u64 = 0x9e37_79b9_7f4a_7c15;
const STUDY_RANDOM_DOMAIN_MIX: u64 = 0xa076_1d64_78bd_642f;

/// One typed SSA register value. `Vector2` remains ABI 1.0's recorded
/// dead-end type: it evaluates honestly but no opcode can carry it to the
/// output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StudyValue {
    Scalar(f32),
    Vector2([f32; 2]),
    Color([f32; 4]),
}

/// Per-frame inputs. Everything here is sanitized on evaluation: non-finite
/// lands on the documented neutral, bands and phase clamp to `0..=1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudyFrameContext {
    pub audio_bands: [f32; STUDY_MAX_AUDIO_BANDS as usize],
    pub beat_phase: f32,
    /// The committed clean-history ring's valid-sample count, exactly the
    /// `valid_history` counter the temporal pass consumes.
    pub valid_history: u32,
}

impl Default for StudyFrameContext {
    fn default() -> Self {
        Self {
            audio_bands: [0.0; STUDY_MAX_AUDIO_BANDS as usize],
            beat_phase: 0.0,
            valid_history: 0,
        }
    }
}

/// The caller's window onto one pixel's stored ring layers. The evaluator
/// applies the validity guard first and calls this only with an age it has
/// already bounded to `1..=min(STUDY_MAX_HISTORY_AGE, valid_history - 1)`;
/// age zero — the virtual current image — never reaches this trait.
pub trait StudyHistorySource {
    fn history_color(&self, age: u8) -> [f32; 4];
}

impl<F: Fn(u8) -> [f32; 4]> StudyHistorySource for F {
    fn history_color(&self, age: u8) -> [f32; 4] {
        self(age)
    }
}

/// One pixel's inputs.
pub struct StudyPixelInputs<'a> {
    pub current: [f32; 4],
    pub motion: [f32; 2],
    pub history: &'a dyn StudyHistorySource,
}

/// A validated document compiled for frame-rate evaluation: the R2 random
/// values are resolved to constants here (they depend only on the document,
/// so evaluation never hashes), and the history ages a frame must be able to
/// serve are listed once. Compile on change, never at frame rate.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledStudy {
    instructions: Vec<StudyInstruction>,
    /// SHA-256 over the document's canonical serialized form — the exact
    /// bytes `StudyDocument::to_json_bytes` emits. This is the identity the
    /// R2 hash consumes, so the same document is the same randomness, live
    /// and offline, forever.
    canonical_digest: [u8; 32],
    /// `deterministic_random` resolved per instruction index; `None` for
    /// every non-random instruction.
    resolved_random: Vec<Option<f32>>,
    /// The authored history ages this study can request, pre-guard.
    required_history_ages: Vec<u8>,
}

impl CompiledStudy {
    /// Validate and compile. Every structural refusal is `validate()`'s;
    /// this adds nothing an invalid document could slip through.
    pub fn compile(document: &StudyDocument) -> Result<Self, StudyError> {
        let canonical = document.to_json_bytes()?;
        let mut hasher = Sha256::new();
        hasher.update(&canonical);
        let canonical_digest: [u8; 32] = hasher.finalize().into();

        let mut resolved_random = Vec::with_capacity(document.instructions.len());
        let mut required_history_ages = Vec::new();
        for instruction in &document.instructions {
            match instruction {
                StudyInstruction::LoadDeterministicRandom { domain, .. } => {
                    resolved_random.push(Some(deterministic_random(&canonical_digest, *domain)));
                }
                StudyInstruction::LoadHistoryColor { age, .. } => {
                    if !required_history_ages.contains(age) {
                        required_history_ages.push(*age);
                    }
                    resolved_random.push(None);
                }
                _ => resolved_random.push(None),
            }
        }
        required_history_ages.sort_unstable();
        Ok(Self {
            instructions: document.instructions.clone(),
            canonical_digest,
            resolved_random,
            required_history_ages,
        })
    }

    pub fn canonical_digest(&self) -> &[u8; 32] {
        &self.canonical_digest
    }

    /// Authored ages this study reads, `1..=STUDY_MAX_HISTORY_AGE`,
    /// pre-guard and deduplicated.
    pub fn required_history_ages(&self) -> &[u8] {
        &self.required_history_ages
    }

    /// Evaluate one pixel. Infallible by construction: `compile` ran the
    /// validator, so every register read is typed and defined, and the bound
    /// law keeps every intermediate representable. The output is clamped to
    /// `0..=1` per component at the boundary — the audience image is unorm.
    pub fn evaluate_pixel(
        &self,
        frame: &StudyFrameContext,
        pixel: &StudyPixelInputs<'_>,
    ) -> [f32; 4] {
        let mut registers = [StudyValue::Scalar(0.0); STUDY_MAX_REGISTERS];
        let current = bound_color(pixel.current);
        let mut output = [0.0; 4];
        for (index, instruction) in self.instructions.iter().enumerate() {
            match instruction {
                StudyInstruction::LoadCurrentColor { dst } => {
                    set(&mut registers, *dst, StudyValue::Color(current));
                }
                StudyInstruction::LoadHistoryColor { dst, age } => {
                    let color = guarded_history(current, *age, frame.valid_history, pixel.history);
                    set(&mut registers, *dst, StudyValue::Color(bound_color(color)));
                }
                StudyInstruction::LoadMotionVector { dst } => {
                    set(
                        &mut registers,
                        *dst,
                        StudyValue::Vector2([bound(pixel.motion[0]), bound(pixel.motion[1])]),
                    );
                }
                StudyInstruction::LoadAudioBand { dst, band } => {
                    let raw = frame
                        .audio_bands
                        .get(usize::from(*band))
                        .copied()
                        .unwrap_or(0.0);
                    let value = if raw.is_finite() {
                        raw.clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    set(&mut registers, *dst, StudyValue::Scalar(value));
                }
                StudyInstruction::LoadBeatPhase { dst } => {
                    let value = if frame.beat_phase.is_finite() {
                        frame.beat_phase.clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    set(&mut registers, *dst, StudyValue::Scalar(value));
                }
                StudyInstruction::LoadDeterministicRandom { dst, .. } => {
                    let value = self.resolved_random[index].unwrap_or(0.0);
                    set(&mut registers, *dst, StudyValue::Scalar(value));
                }
                StudyInstruction::ConstantScalar { dst, value } => {
                    set(&mut registers, *dst, StudyValue::Scalar(bound(*value)));
                }
                StudyInstruction::ConstantVector2 { dst, value } => {
                    set(
                        &mut registers,
                        *dst,
                        StudyValue::Vector2([bound(value[0]), bound(value[1])]),
                    );
                }
                StudyInstruction::ConstantColor { dst, value } => {
                    set(&mut registers, *dst, StudyValue::Color(bound_color(*value)));
                }
                StudyInstruction::Add { dst, left, right } => {
                    let value = zip(get(&registers, *left), get(&registers, *right), |a, b| {
                        a + b
                    });
                    set(&mut registers, *dst, value);
                }
                StudyInstruction::Subtract { dst, left, right } => {
                    let value = zip(get(&registers, *left), get(&registers, *right), |a, b| {
                        a - b
                    });
                    set(&mut registers, *dst, value);
                }
                StudyInstruction::Multiply { dst, left, right } => {
                    let value = zip(get(&registers, *left), get(&registers, *right), |a, b| {
                        a * b
                    });
                    set(&mut registers, *dst, value);
                }
                StudyInstruction::Mix { dst, a, b, amount } => {
                    let StudyValue::Scalar(t) = get(&registers, *amount) else {
                        unreachable!("validation typed the mix amount as a scalar");
                    };
                    let value = zip(get(&registers, *a), get(&registers, *b), |a, b| {
                        a + (b - a) * t
                    });
                    set(&mut registers, *dst, value);
                }
                StudyInstruction::Clamp01 { dst, input } => {
                    let value = map(get(&registers, *input), |v| v.clamp(0.0, 1.0));
                    set(&mut registers, *dst, value);
                }
                StudyInstruction::HueRotate { dst, color, turns } => {
                    let StudyValue::Color(color) = get(&registers, *color) else {
                        unreachable!("validation typed the hue input as a color");
                    };
                    let StudyValue::Scalar(turns) = get(&registers, *turns) else {
                        unreachable!("validation typed the hue turns as a scalar");
                    };
                    set(
                        &mut registers,
                        *dst,
                        StudyValue::Color(bound_color(hue_rotate(color, turns))),
                    );
                }
                StudyInstruction::OutputColor { color } => {
                    let StudyValue::Color(color) = get(&registers, *color) else {
                        unreachable!("validation typed the output as a color");
                    };
                    output = [
                        color[0].clamp(0.0, 1.0),
                        color[1].clamp(0.0, 1.0),
                        color[2].clamp(0.0, 1.0),
                        color[3].clamp(0.0, 1.0),
                    ];
                }
            }
        }
        output
    }
}

/// The frozen R2 hash. Layout: first eight canonical-digest bytes
/// little-endian, XOR the ABI lane (`major << 16 | minor`, mixed), XOR the
/// domain lane (mixed), XOR the tag; SplitMix64 finalizer; top 24 bits over
/// `2^24` so the result is exactly representable and in `[0, 1)`.
pub fn deterministic_random(canonical_digest: &[u8; 32], domain: u32) -> f32 {
    let document_lane = u64::from_le_bytes(
        canonical_digest[..8]
            .try_into()
            .expect("a SHA-256 digest always has eight leading bytes"),
    );
    let abi_lane = (u64::from(STUDY_ABI_MAJOR) << 16) | u64::from(STUDY_ABI_MINOR);
    let mut value = document_lane
        ^ abi_lane.wrapping_mul(STUDY_RANDOM_ABI_MIX)
        ^ u64::from(domain).wrapping_mul(STUDY_RANDOM_DOMAIN_MIX)
        ^ STUDY_RANDOM_TAG;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    ((value >> 40) as f32) / (1u64 << 24) as f32
}

/// The R1 guard, exactly `temporal_originals.wgsl`'s law: depth clamps to
/// `valid_history - 1` (zero when nothing is committed), and depth zero is
/// the virtual current image rather than any stored layer.
fn guarded_history(
    current: [f32; 4],
    age: u8,
    valid_history: u32,
    history: &dyn StudyHistorySource,
) -> [f32; 4] {
    debug_assert!((1..=STUDY_MAX_HISTORY_AGE).contains(&age));
    let max_depth = valid_history
        .saturating_sub(1)
        .min(u32::from(STUDY_MAX_HISTORY_AGE));
    let effective = u32::from(age).min(max_depth);
    if effective == 0 {
        current
    } else {
        history.history_color(effective as u8)
    }
}

fn set(registers: &mut [StudyValue; STUDY_MAX_REGISTERS], dst: StudyRegister, value: StudyValue) {
    registers[usize::from(dst.get())] = value;
}

fn get(registers: &[StudyValue; STUDY_MAX_REGISTERS], src: StudyRegister) -> StudyValue {
    registers[usize::from(src.get())]
}

/// The bound law: non-finite lands on the documented neutral `0.0`, finite
/// clamps into the ABI's representable range.
fn bound(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(-STUDY_MAX_FINITE_VALUE, STUDY_MAX_FINITE_VALUE)
    } else {
        0.0
    }
}

fn bound_color(color: [f32; 4]) -> [f32; 4] {
    [
        bound(color[0]),
        bound(color[1]),
        bound(color[2]),
        bound(color[3]),
    ]
}

fn map(value: StudyValue, f: impl Fn(f32) -> f32) -> StudyValue {
    match value {
        StudyValue::Scalar(a) => StudyValue::Scalar(bound(f(a))),
        StudyValue::Vector2(a) => StudyValue::Vector2([bound(f(a[0])), bound(f(a[1]))]),
        StudyValue::Color(a) => StudyValue::Color([
            bound(f(a[0])),
            bound(f(a[1])),
            bound(f(a[2])),
            bound(f(a[3])),
        ]),
    }
}

/// Componentwise combination of two same-typed values. Validation guarantees
/// the types match; the mismatch arms are structurally unreachable.
fn zip(left: StudyValue, right: StudyValue, f: impl Fn(f32, f32) -> f32) -> StudyValue {
    match (left, right) {
        (StudyValue::Scalar(a), StudyValue::Scalar(b)) => StudyValue::Scalar(bound(f(a, b))),
        (StudyValue::Vector2(a), StudyValue::Vector2(b)) => {
            StudyValue::Vector2([bound(f(a[0], b[0])), bound(f(a[1], b[1]))])
        }
        (StudyValue::Color(a), StudyValue::Color(b)) => StudyValue::Color([
            bound(f(a[0], b[0])),
            bound(f(a[1], b[1])),
            bound(f(a[2], b[2])),
            bound(f(a[3], b[3])),
        ]),
        _ => unreachable!("validation typed both operands identically"),
    }
}

// --- The rack hue law, mirrored line for line from rack_node.wgsl ---------

fn rgb_to_hsl(c: [f32; 3]) -> [f32; 3] {
    let max_c = c[0].max(c[1]).max(c[2]);
    let min_c = c[0].min(c[1]).min(c[2]);
    let lightness = (max_c + min_c) * 0.5;
    let delta = max_c - min_c;
    if delta < 0.001 {
        return [0.0, 0.0, lightness];
    }
    let saturation = if lightness > 0.5 {
        delta / (2.0 - max_c - min_c)
    } else {
        delta / (max_c + min_c)
    };
    let hue = if max_c == c[0] {
        (c[1] - c[2]) / delta + if c[1] < c[2] { 6.0 } else { 0.0 }
    } else if max_c == c[1] {
        (c[2] - c[0]) / delta + 2.0
    } else {
        (c[0] - c[1]) / delta + 4.0
    };
    [hue / 6.0, saturation, lightness]
}

fn hue_to_rgb(p: f32, q: f32, initial: f32) -> f32 {
    let mut t = initial;
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 0.5 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    p
}

fn hsl_to_rgb(hsl: [f32; 3]) -> [f32; 3] {
    if hsl[1] < 0.001 {
        return [hsl[2], hsl[2], hsl[2]];
    }
    let q = if hsl[2] < 0.5 {
        hsl[2] * (1.0 + hsl[1])
    } else {
        hsl[2] + hsl[1] - hsl[2] * hsl[1]
    };
    let p = 2.0 * hsl[2] - q;
    [
        hue_to_rgb(p, q, hsl[0] + 1.0 / 3.0),
        hue_to_rgb(p, q, hsl[0]),
        hue_to_rgb(p, q, hsl[0] - 1.0 / 3.0),
    ]
}

fn fract(value: f32) -> f32 {
    value - value.floor()
}

fn hue_rotate(color: [f32; 4], turns: f32) -> [f32; 4] {
    // The S10b domain clamp: the HSL round trip is defined on unorm colors —
    // outside that range its divisions can reach non-finite values whose
    // handling WGSL leaves implementation-defined, so both evaluators clamp
    // the rgb operand first and the halves stay bit-agreeable everywhere.
    // Alpha passes through untouched.
    let mut hsl = rgb_to_hsl([
        color[0].clamp(0.0, 1.0),
        color[1].clamp(0.0, 1.0),
        color[2].clamp(0.0, 1.0),
    ]);
    // `fract` on a non-finite turn is meaningless; the neutral is no
    // rotation, never a clamped extreme.
    if turns.is_finite() {
        hsl[0] = fract(hsl[0] + turns);
    }
    let rgb = hsl_to_rgb(hsl);
    [rgb[0], rgb[1], rgb[2], color[3]]
}

/// Whether a compiled study consumes a capability — a convenience mirror of
/// the document's own canonical table for callers wiring frame inputs.
pub fn consumes(document: &StudyDocument, capability: StudyCapability) -> bool {
    document.capabilities.contains(&capability)
}

// --- The frozen GPU instruction encoding -----------------------------------

/// GPU opcode numbering, append-only exactly like `NodeKindTag` codes:
/// 0 LoadCurrentColor, 1 LoadHistoryColor, 2 LoadMotionVector,
/// 3 LoadAudioBand, 4 LoadBeatPhase, 5 LoadDeterministicRandom (resolved to
/// an immediate at compile — the GPU never hashes), 6 ConstantScalar,
/// 7 ConstantVector2, 8 ConstantColor, 9 Add, 10 Subtract, 11 Multiply,
/// 12 Mix, 13 Clamp01, 14 HueRotate, 15 OutputColor. Codes are never
/// renumbered or reused; growth is an R3 minor bump.
pub const STUDY_GPU_MAX_INSTRUCTIONS: usize = crate::study::STUDY_MAX_INSTRUCTIONS;

/// One encoded instruction, 32 bytes, uniform-array stride safe.
/// `words[0]` carries the opcode in its low 16 bits and the auxiliary
/// operand — Mix's amount register, HueRotate's turns register, the history
/// age, the audio band — in its high 16; `words[1..=3]` are dst, src a,
/// src b. The immediate carries constants and resolved random values.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StudyGpuOp {
    pub words: [u32; 4],
    pub immediate: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<StudyGpuOp>() == 32);

impl CompiledStudy {
    /// Encode this study for the fixed WGSL interpreter. The full uniform
    /// array is always `STUDY_GPU_MAX_INSTRUCTIONS` long — unused slots are
    /// zero and the shader walks only `instruction_count` — so one buffer
    /// size serves every study and a swap never reallocates.
    pub fn encode_gpu_program(&self) -> Vec<StudyGpuOp> {
        let mut ops = vec![StudyGpuOp::zeroed(); STUDY_GPU_MAX_INSTRUCTIONS];
        for (index, instruction) in self.instructions.iter().enumerate() {
            let op = &mut ops[index];
            let mut encode = |opcode: u32, aux: u32, dst: u8, a: u8, b: u8, immediate: [f32; 4]| {
                op.words = [
                    opcode | (aux << 16),
                    u32::from(dst),
                    u32::from(a),
                    u32::from(b),
                ];
                op.immediate = immediate;
            };
            match instruction {
                StudyInstruction::LoadCurrentColor { dst } => {
                    encode(0, 0, dst.get(), 0, 0, [0.0; 4]);
                }
                StudyInstruction::LoadHistoryColor { dst, age } => {
                    encode(1, u32::from(*age), dst.get(), 0, 0, [0.0; 4]);
                }
                StudyInstruction::LoadMotionVector { dst } => {
                    encode(2, 0, dst.get(), 0, 0, [0.0; 4]);
                }
                StudyInstruction::LoadAudioBand { dst, band } => {
                    encode(3, u32::from(*band), dst.get(), 0, 0, [0.0; 4]);
                }
                StudyInstruction::LoadBeatPhase { dst } => {
                    encode(4, 0, dst.get(), 0, 0, [0.0; 4]);
                }
                StudyInstruction::LoadDeterministicRandom { dst, .. } => {
                    let value = self.resolved_random[index]
                        .expect("compile resolved every random instruction");
                    encode(5, 0, dst.get(), 0, 0, [value, 0.0, 0.0, 0.0]);
                }
                StudyInstruction::ConstantScalar { dst, value } => {
                    encode(6, 0, dst.get(), 0, 0, [*value, 0.0, 0.0, 0.0]);
                }
                StudyInstruction::ConstantVector2 { dst, value } => {
                    encode(7, 0, dst.get(), 0, 0, [value[0], value[1], 0.0, 0.0]);
                }
                StudyInstruction::ConstantColor { dst, value } => {
                    encode(8, 0, dst.get(), 0, 0, *value);
                }
                StudyInstruction::Add { dst, left, right } => {
                    encode(9, 0, dst.get(), left.get(), right.get(), [0.0; 4]);
                }
                StudyInstruction::Subtract { dst, left, right } => {
                    encode(10, 0, dst.get(), left.get(), right.get(), [0.0; 4]);
                }
                StudyInstruction::Multiply { dst, left, right } => {
                    encode(11, 0, dst.get(), left.get(), right.get(), [0.0; 4]);
                }
                StudyInstruction::Mix { dst, a, b, amount } => {
                    encode(
                        12,
                        u32::from(amount.get()),
                        dst.get(),
                        a.get(),
                        b.get(),
                        [0.0; 4],
                    );
                }
                StudyInstruction::Clamp01 { dst, input } => {
                    encode(13, 0, dst.get(), input.get(), 0, [0.0; 4]);
                }
                StudyInstruction::HueRotate { dst, color, turns } => {
                    encode(
                        14,
                        u32::from(turns.get()),
                        dst.get(),
                        color.get(),
                        0,
                        [0.0; 4],
                    );
                }
                StudyInstruction::OutputColor { color } => {
                    encode(15, 0, 0, color.get(), 0, [0.0; 4]);
                }
            }
        }
        ops
    }

    /// The number of live instructions the shader must walk.
    pub fn instruction_count(&self) -> u32 {
        self.instructions.len() as u32
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::study::{
        StudyAbiVersion, StudyLicenseNotice, StudyMetadata, StudyPublicationBoundary,
        STUDY_SCHEMA_VERSION,
    };

    pub(crate) fn register(value: u8) -> StudyRegister {
        StudyRegister::new(value).unwrap()
    }

    pub(crate) fn document(
        capabilities: Vec<StudyCapability>,
        instructions: Vec<StudyInstruction>,
    ) -> StudyDocument {
        StudyDocument {
            schema_version: STUDY_SCHEMA_VERSION,
            abi: StudyAbiVersion::default(),
            metadata: StudyMetadata {
                name: "Reference fixture".into(),
                author: "Evaluator tests".into(),
                description: String::new(),
                license: StudyLicenseNotice {
                    identifier: "CC0-1.0".into(),
                    notice: String::new(),
                    publication_boundary: StudyPublicationBoundary::StudyDataOnlyDoesNotLicenseHost,
                },
            },
            capabilities,
            instructions,
        }
    }

    struct NoHistory;
    impl StudyHistorySource for NoHistory {
        fn history_color(&self, _age: u8) -> [f32; 4] {
            panic!("this study must not reach the history source");
        }
    }

    fn pixel<'a>(current: [f32; 4], history: &'a dyn StudyHistorySource) -> StudyPixelInputs<'a> {
        StudyPixelInputs {
            current,
            motion: [0.0, 0.0],
            history,
        }
    }

    #[test]
    fn constants_arithmetic_mix_and_clamp_follow_their_analytic_laws() {
        let compiled = CompiledStudy::compile(&document(
            vec![],
            vec![
                StudyInstruction::ConstantColor {
                    dst: register(0),
                    value: [0.2, 0.4, 0.6, 1.0],
                },
                StudyInstruction::ConstantColor {
                    dst: register(1),
                    value: [0.6, 0.2, 1.4, 1.0],
                },
                StudyInstruction::ConstantScalar {
                    dst: register(2),
                    value: 0.5,
                },
                // (a + b) * mix(a, b, 0.5), clamped: exercises every
                // arithmetic opcode with exact binary-fraction expectations.
                StudyInstruction::Add {
                    dst: register(3),
                    left: register(0),
                    right: register(1),
                },
                StudyInstruction::Mix {
                    dst: register(4),
                    a: register(0),
                    b: register(1),
                    amount: register(2),
                },
                StudyInstruction::Multiply {
                    dst: register(5),
                    left: register(3),
                    right: register(4),
                },
                StudyInstruction::Subtract {
                    dst: register(6),
                    left: register(5),
                    right: register(0),
                },
                StudyInstruction::Clamp01 {
                    dst: register(7),
                    input: register(6),
                },
                StudyInstruction::OutputColor { color: register(7) },
            ],
        ))
        .unwrap();
        let out =
            compiled.evaluate_pixel(&StudyFrameContext::default(), &pixel([0.0; 4], &NoHistory));
        // add = [0.8, 0.6, 2.0, 2.0]; mix = [0.4, 0.3, 1.0, 1.0];
        // mul = [0.32, 0.18, 2.0, 2.0]; sub = [0.12, -0.22, 1.4, 1.0];
        // clamp01 = [0.12, 0.0, 1.0, 1.0].
        assert!((out[0] - 0.12).abs() < 1e-6);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 1.0);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn hue_rotation_matches_the_rack_shader_law_analytically() {
        let rotate = |turns: f32| {
            let compiled = CompiledStudy::compile(&document(
                vec![],
                vec![
                    StudyInstruction::ConstantColor {
                        dst: register(0),
                        value: [1.0, 0.0, 0.0, 1.0],
                    },
                    StudyInstruction::ConstantScalar {
                        dst: register(1),
                        value: turns,
                    },
                    StudyInstruction::HueRotate {
                        dst: register(2),
                        color: register(0),
                        turns: register(1),
                    },
                    StudyInstruction::OutputColor { color: register(2) },
                ],
            ))
            .unwrap();
            compiled.evaluate_pixel(&StudyFrameContext::default(), &pixel([0.0; 4], &NoHistory))
        };
        // Red a third of a turn forward is green; a third backward is blue
        // (the shader's fract wrap); a full turn is the identity; alpha
        // never rotates. Compared at f32 epsilon because 1/3 sits on a
        // hue_to_rgb branch boundary.
        let close = |observed: [f32; 4], expected: [f32; 4]| {
            for (o, e) in observed.iter().zip(expected.iter()) {
                assert!((o - e).abs() < 1e-6, "{observed:?} != {expected:?}");
            }
        };
        close(rotate(1.0 / 3.0), [0.0, 1.0, 0.0, 1.0]);
        close(rotate(-1.0 / 3.0), [0.0, 0.0, 1.0, 1.0]);
        close(rotate(1.0), [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn history_reads_are_guarded_by_the_valid_sample_count_like_the_temporal_shader() {
        let study_with_age = |age: u8| {
            CompiledStudy::compile(&document(
                vec![StudyCapability::HistoryRead],
                vec![
                    StudyInstruction::LoadHistoryColor {
                        dst: register(0),
                        age,
                    },
                    StudyInstruction::OutputColor { color: register(0) },
                ],
            ))
            .unwrap()
        };
        struct AgeEncodes;
        impl StudyHistorySource for AgeEncodes {
            fn history_color(&self, age: u8) -> [f32; 4] {
                [f32::from(age) / 100.0, 0.0, 0.0, 1.0]
            }
        }
        let current = [0.77, 0.0, 0.0, 1.0];

        // Nothing committed: every age is the virtual current image and the
        // stored layers are never touched.
        let empty = StudyFrameContext {
            valid_history: 0,
            ..StudyFrameContext::default()
        };
        assert_eq!(
            study_with_age(9).evaluate_pixel(&empty, &pixel(current, &NoHistory)),
            current
        );
        // One committed sample: max depth is zero, still the current image.
        let one = StudyFrameContext {
            valid_history: 1,
            ..StudyFrameContext::default()
        };
        assert_eq!(
            study_with_age(1).evaluate_pixel(&one, &pixel(current, &NoHistory)),
            current
        );
        // Five committed samples: an in-range age reads exactly itself, and
        // a deeper request clamps to the oldest valid layer — depth 4 —
        // exactly the temporal shader's max-depth law.
        let five = StudyFrameContext {
            valid_history: 5,
            ..StudyFrameContext::default()
        };
        let exact = study_with_age(3).evaluate_pixel(&five, &pixel(current, &AgeEncodes));
        assert!((exact[0] - 0.03).abs() < 1e-6);
        let clamped = study_with_age(9).evaluate_pixel(&five, &pixel(current, &AgeEncodes));
        assert!((clamped[0] - 0.04).abs() < 1e-6);
    }

    #[test]
    fn deterministic_random_is_a_document_constant_with_independent_domains() {
        let build = |name: &str, domain: u32| {
            let mut doc = document(
                vec![StudyCapability::DeterministicRandom],
                vec![
                    StudyInstruction::LoadDeterministicRandom {
                        dst: register(0),
                        domain,
                    },
                    StudyInstruction::ConstantColor {
                        dst: register(1),
                        value: [1.0, 1.0, 1.0, 1.0],
                    },
                    StudyInstruction::ConstantColor {
                        dst: register(2),
                        value: [0.0, 0.0, 0.0, 1.0],
                    },
                    StudyInstruction::Mix {
                        dst: register(3),
                        a: register(2),
                        b: register(1),
                        amount: register(0),
                    },
                    StudyInstruction::OutputColor { color: register(3) },
                ],
            );
            doc.metadata.name = name.into();
            CompiledStudy::compile(&doc).unwrap()
        };
        let evaluate = |compiled: &CompiledStudy, frame: &StudyFrameContext| {
            compiled.evaluate_pixel(frame, &pixel([0.0; 4], &NoHistory))[0]
        };

        let a = build("Domain study", 7);
        // The value is a compile-time document constant: identical across
        // recompiles and across arbitrarily different frame contexts.
        assert_eq!(
            a.canonical_digest(),
            build("Domain study", 7).canonical_digest()
        );
        let loud = StudyFrameContext {
            audio_bands: [1.0; STUDY_MAX_AUDIO_BANDS as usize],
            beat_phase: 0.9,
            valid_history: 12,
        };
        let value = evaluate(&a, &StudyFrameContext::default());
        assert_eq!(value, evaluate(&a, &loud));
        assert_eq!(value, evaluate(&build("Domain study", 7), &loud));
        assert!((0.0..1.0).contains(&value));

        // A different domain and a different document each reroll; nothing
        // else does.
        let other_domain = evaluate(&build("Domain study", 8), &loud);
        let other_document = evaluate(&build("Domain study renamed", 7), &loud);
        assert_ne!(value, other_domain);
        assert_ne!(value, other_document);

        // The raw hash law is pinned directly too: domain-separated, [0, 1).
        let digest = *a.canonical_digest();
        assert_eq!(
            deterministic_random(&digest, 7),
            deterministic_random(&digest, 7)
        );
        assert_ne!(
            deterministic_random(&digest, 7),
            deterministic_random(&digest, 8)
        );
    }

    #[test]
    fn the_bound_law_keeps_every_intermediate_inside_the_abi_range() {
        // 65504^2 is finite in f32, so without the bound the subtraction
        // below would leave ~4.29e9 and the final scale would saturate the
        // output at 1.0; with the bound the product clamps to 65504, the
        // subtraction leaves exactly 1.0, and the scale makes it vanish.
        let compiled = CompiledStudy::compile(&document(
            vec![],
            vec![
                StudyInstruction::ConstantColor {
                    dst: register(0),
                    value: [STUDY_MAX_FINITE_VALUE, 0.0, 0.0, 1.0],
                },
                StudyInstruction::Multiply {
                    dst: register(1),
                    left: register(0),
                    right: register(0),
                },
                StudyInstruction::ConstantColor {
                    dst: register(2),
                    value: [STUDY_MAX_FINITE_VALUE - 1.0, 0.0, 0.0, 0.0],
                },
                StudyInstruction::Subtract {
                    dst: register(3),
                    left: register(1),
                    right: register(2),
                },
                StudyInstruction::ConstantColor {
                    dst: register(4),
                    value: [1.0e-6, 1.0, 1.0, 1.0],
                },
                StudyInstruction::Multiply {
                    dst: register(5),
                    left: register(3),
                    right: register(4),
                },
                StudyInstruction::OutputColor { color: register(5) },
            ],
        ))
        .unwrap();
        let out =
            compiled.evaluate_pixel(&StudyFrameContext::default(), &pixel([0.0; 4], &NoHistory));
        assert!(
            out[0] < 1.0e-5,
            "the product must clamp at the ABI bound before the subtraction, got {}",
            out[0]
        );
    }

    #[test]
    fn frame_inputs_sanitize_to_documented_neutrals_and_vector2_stays_a_dead_end() {
        let compiled = CompiledStudy::compile(&document(
            vec![
                StudyCapability::MotionFieldRead,
                StudyCapability::AudioFeatures,
                StudyCapability::BeatPhase,
            ],
            vec![
                // The dead-end lane: loaded and combined, provably unable to
                // reach the output, still evaluated honestly.
                StudyInstruction::LoadMotionVector { dst: register(0) },
                StudyInstruction::Add {
                    dst: register(1),
                    left: register(0),
                    right: register(0),
                },
                StudyInstruction::LoadAudioBand {
                    dst: register(2),
                    band: 2,
                },
                StudyInstruction::LoadBeatPhase { dst: register(3) },
                StudyInstruction::ConstantColor {
                    dst: register(4),
                    value: [1.0, 1.0, 1.0, 1.0],
                },
                StudyInstruction::ConstantColor {
                    dst: register(5),
                    value: [0.0, 0.0, 0.0, 1.0],
                },
                StudyInstruction::Mix {
                    dst: register(6),
                    a: register(5),
                    b: register(4),
                    amount: register(2),
                },
                StudyInstruction::HueRotate {
                    dst: register(7),
                    color: register(6),
                    turns: register(3),
                },
                StudyInstruction::OutputColor { color: register(7) },
            ],
        ))
        .unwrap();
        // A NaN band lands on the documented neutral 0 (black), never on a
        // clamped extreme; an out-of-range beat phase clamps.
        let mut frame = StudyFrameContext::default();
        frame.audio_bands[2] = f32::NAN;
        frame.beat_phase = 2.5;
        let out = compiled.evaluate_pixel(
            &frame,
            &StudyPixelInputs {
                current: [0.0; 4],
                motion: [64.0, -64.0],
                history: &NoHistory,
            },
        );
        assert_eq!(out, [0.0, 0.0, 0.0, 1.0]);
    }

    /// A document exercising all sixteen opcodes with valid SSA, shared by
    /// the encoding golden below and the GPU agreement fixtures.
    pub(crate) fn every_opcode_document() -> StudyDocument {
        document(
            vec![
                StudyCapability::CurrentColor,
                StudyCapability::HistoryRead,
                StudyCapability::MotionFieldRead,
                StudyCapability::AudioFeatures,
                StudyCapability::BeatPhase,
                StudyCapability::DeterministicRandom,
            ],
            vec![
                StudyInstruction::LoadCurrentColor { dst: register(0) },
                StudyInstruction::LoadHistoryColor {
                    dst: register(1),
                    age: 7,
                },
                StudyInstruction::LoadMotionVector { dst: register(2) },
                StudyInstruction::LoadAudioBand {
                    dst: register(3),
                    band: 5,
                },
                StudyInstruction::LoadBeatPhase { dst: register(4) },
                StudyInstruction::LoadDeterministicRandom {
                    dst: register(5),
                    domain: 9,
                },
                StudyInstruction::ConstantScalar {
                    dst: register(6),
                    value: 0.25,
                },
                StudyInstruction::ConstantVector2 {
                    dst: register(7),
                    value: [0.5, -0.5],
                },
                StudyInstruction::ConstantColor {
                    dst: register(8),
                    value: [0.1, 0.2, 0.3, 0.4],
                },
                StudyInstruction::Add {
                    dst: register(9),
                    left: register(0),
                    right: register(1),
                },
                StudyInstruction::Subtract {
                    dst: register(10),
                    left: register(9),
                    right: register(8),
                },
                StudyInstruction::Multiply {
                    dst: register(11),
                    left: register(10),
                    right: register(8),
                },
                StudyInstruction::Mix {
                    dst: register(12),
                    a: register(11),
                    b: register(8),
                    amount: register(6),
                },
                StudyInstruction::Clamp01 {
                    dst: register(13),
                    input: register(12),
                },
                StudyInstruction::HueRotate {
                    dst: register(14),
                    color: register(13),
                    turns: register(4),
                },
                StudyInstruction::OutputColor {
                    color: register(14),
                },
            ],
        )
    }

    #[test]
    fn the_gpu_encoding_is_the_frozen_layout() {
        let compiled = CompiledStudy::compile(&every_opcode_document()).unwrap();
        let ops = compiled.encode_gpu_program();
        assert_eq!(ops.len(), STUDY_GPU_MAX_INSTRUCTIONS);
        assert_eq!(compiled.instruction_count(), 16);

        let words = |index: usize| ops[index].words;
        let imm = |index: usize| ops[index].immediate;
        // opcode | aux << 16, dst, a, b — the append-only numbering.
        assert_eq!(words(0), [0, 0, 0, 0]); // LoadCurrentColor -> r0
        assert_eq!(words(1), [1 | (7 << 16), 1, 0, 0]); // history age 7
        assert_eq!(words(2), [2, 2, 0, 0]); // motion (dead lane)
        assert_eq!(words(3), [3 | (5 << 16), 3, 0, 0]); // audio band 5
        assert_eq!(words(4), [4, 4, 0, 0]); // beat phase
        assert_eq!(words(5), [5, 5, 0, 0]); // random, resolved immediate
        assert_eq!(
            imm(5)[0],
            deterministic_random(compiled.canonical_digest(), 9)
        );
        assert_eq!(words(6), [6, 6, 0, 0]);
        assert_eq!(imm(6), [0.25, 0.0, 0.0, 0.0]);
        assert_eq!(words(7), [7, 7, 0, 0]);
        assert_eq!(imm(7), [0.5, -0.5, 0.0, 0.0]);
        assert_eq!(words(8), [8, 8, 0, 0]);
        assert_eq!(imm(8), [0.1, 0.2, 0.3, 0.4]);
        assert_eq!(words(9), [9, 9, 0, 1]); // add r0 r1
        assert_eq!(words(10), [10, 10, 9, 8]);
        assert_eq!(words(11), [11, 11, 10, 8]);
        assert_eq!(words(12), [12 | (6 << 16), 12, 11, 8]); // mix amount r6
        assert_eq!(words(13), [13, 13, 12, 0]);
        assert_eq!(words(14), [14 | (4 << 16), 14, 13, 0]); // hue turns r4
        assert_eq!(words(15), [15, 0, 14, 0]); // output reads r14
                                               // Unused slots are zero, so one fixed-size buffer serves every study.
        assert_eq!(ops[16], StudyGpuOp::zeroed());
        assert_eq!(ops[STUDY_GPU_MAX_INSTRUCTIONS - 1], StudyGpuOp::zeroed());
    }

    #[test]
    fn hue_rotation_clamps_its_operand_to_the_unorm_domain() {
        // The S10b domain clamp (semantics version 2): an out-of-range color
        // rotates exactly as its unorm clamp does, on both evaluators.
        let rotate = |red: f32| {
            let compiled = CompiledStudy::compile(&document(
                vec![],
                vec![
                    StudyInstruction::ConstantColor {
                        dst: register(0),
                        value: [red, -3.0, 0.0, 1.0],
                    },
                    StudyInstruction::ConstantScalar {
                        dst: register(1),
                        value: 0.5,
                    },
                    StudyInstruction::HueRotate {
                        dst: register(2),
                        color: register(0),
                        turns: register(1),
                    },
                    StudyInstruction::OutputColor { color: register(2) },
                ],
            ))
            .unwrap();
            compiled.evaluate_pixel(&StudyFrameContext::default(), &pixel([0.0; 4], &NoHistory))
        };
        assert_eq!(rotate(2.0), rotate(1.0));
        assert_eq!(STUDY_EVAL_ALGORITHM_VERSION, 2);
    }

    #[test]
    fn the_interpreter_shader_shares_the_rack_hue_law_character_for_character() {
        const INTERPRETER: &str = include_str!("shaders/study_interpreter.wgsl");
        const RACK: &str = include_str!("shaders/rack_node.wgsl");
        fn wgsl_function<'a>(source: &'a str, signature: &str) -> &'a str {
            let start = source
                .find(signature)
                .unwrap_or_else(|| panic!("{signature} is missing"));
            let body = &source[start..];
            let end = body
                .find("\n}\n")
                .expect("a top level body ends at column zero");
            &body[..end]
        }
        for signature in ["fn rgb_to_hsl(", "fn hue_to_rgb(", "fn hsl_to_rgb("] {
            assert_eq!(
                wgsl_function(INTERPRETER, signature),
                wgsl_function(RACK, signature),
                "{signature} must stay one law"
            );
        }
        // The interpreter walks at most the frozen capacity, the bound is
        // the ABI's, and the guard uses the committed validity counter.
        assert_eq!(
            STUDY_GPU_MAX_INSTRUCTIONS,
            crate::study::STUDY_MAX_INSTRUCTIONS
        );
        assert!(INTERPRETER.contains("const STUDY_GPU_MAX_INSTRUCTIONS: u32 = 256u;"));
        assert!(INTERPRETER.contains("const STUDY_BOUND: f32 = 65504.0;"));
        assert!(INTERPRETER.contains("frame.valid_history - 1u"));
    }

    #[test]
    fn compile_lists_required_history_ages_and_refuses_invalid_documents() {
        let compiled = CompiledStudy::compile(&document(
            vec![StudyCapability::HistoryRead],
            vec![
                StudyInstruction::LoadHistoryColor {
                    dst: register(0),
                    age: 5,
                },
                StudyInstruction::LoadHistoryColor {
                    dst: register(1),
                    age: 2,
                },
                StudyInstruction::LoadHistoryColor {
                    dst: register(2),
                    age: 5,
                },
                StudyInstruction::Add {
                    dst: register(3),
                    left: register(0),
                    right: register(1),
                },
                StudyInstruction::Add {
                    dst: register(4),
                    left: register(3),
                    right: register(2),
                },
                StudyInstruction::OutputColor { color: register(4) },
            ],
        ))
        .unwrap();
        assert_eq!(compiled.required_history_ages(), &[2, 5]);

        // Compilation is validation: an undeclared capability refuses, so a
        // compiled study can never consume an input its document hid.
        let undeclared = document(
            vec![],
            vec![
                StudyInstruction::LoadCurrentColor { dst: register(0) },
                StudyInstruction::OutputColor { color: register(0) },
            ],
        );
        assert!(CompiledStudy::compile(&undeclared).is_err());
    }
}
