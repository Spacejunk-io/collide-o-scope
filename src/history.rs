//! Bounded transactional history for explicitly manual authored mutations.
//!
//! The container is deliberately generic. The application owns the exact
//! authored world that is checkpointed, while this module owns gesture
//! coalescing, origin filtering, memory/count bounds, and single-use
//! two-phase undo/redo tokens. A checkpoint may contain authored stable IDs,
//! but must never contain GPU resources or hidden runtime pixels.

use std::collections::VecDeque;
use std::fmt;
use std::num::NonZeroU64;
use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::gesture::{GestureOrigin, GesturePhase, MAX_ACTIVE_STROKES};

pub const HISTORY_MAX_ENTRIES: usize = 128;
pub const HISTORY_MAX_BYTES: u64 = 64 * 1024 * 1024;
pub const HISTORY_MAX_CHECKPOINT_BYTES: u64 = 8 * 1024 * 1024;
pub const HISTORY_MAX_LABEL_BYTES: usize = 96;
pub const HISTORY_MAX_CATEGORY_BYTES: usize = 32;

pub type HistoryFingerprint = [u8; 32];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "non-manual origins are an explicit exclusion contract exercised by history tests"
    )
)]
pub enum MutationOrigin {
    BrowserManual,
    NativeManual,
    Midi,
    Lfo,
    Audio,
    Osc,
    ClipTrigger,
    CollisionScore,
    Automation,
}

impl MutationOrigin {
    pub const fn is_manual(self) -> bool {
        matches!(self, Self::BrowserManual | Self::NativeManual)
    }

    /// Manual-history class of a normalized gesture surface.
    ///
    /// Provenance decides whether a stroke was authored by a person at all: a
    /// native pointer and a phone are hands, while MIDI and OSC are automation
    /// and are excluded from history by exactly the same law that already
    /// excludes every other automation origin. The mapping lives here rather
    /// than in `crate::gesture` so the portable event contract keeps no
    /// knowledge of the history store.
    pub const fn for_gesture_origin(origin: GestureOrigin) -> Self {
        match origin {
            GestureOrigin::NativePointer => Self::NativeManual,
            GestureOrigin::Phone => Self::BrowserManual,
            GestureOrigin::Midi => Self::Midi,
            GestureOrigin::Osc => Self::Osc,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HistoryGestureId(NonZeroU64);

impl HistoryGestureId {
    pub const fn new(raw: u64) -> Option<Self> {
        match NonZeroU64::new(raw) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// What one observed gesture sample asks the manual-history store to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GestureHistoryStep {
    /// Nothing. A motion sample, an automation origin, an out-of-range stroke
    /// identity, an unmatched phase, or a stroke that joined a gesture already
    /// being recorded all land here.
    Idle,
    /// Open exactly one manual-history entry for the gesture now beginning.
    Open(HistoryGestureId),
    /// Close the one entry this gesture opened.
    Close(HistoryGestureId),
}

/// Router that turns a normalized gesture stroke set into **exactly one**
/// manual-history entry.
///
/// The law is deliberately narrow. An entry opens at the first `Begin`
/// observed while the router holds nothing open, and closes at the `End` that
/// closes the last still-open stroke, so a multi-touch gesture is one authored
/// transaction rather than one per finger. Every `Move` in between is
/// invisible: a five-hundred-sample stroke must cost the bounded stack exactly
/// one checkpoint, not five hundred. An automation-driven origin is refused
/// before any identity is allocated, so MIDI and OSC gestures record nothing at
/// all.
///
/// The router allocates the identity and reports the boundary; it never touches
/// the store, because the authored world a checkpoint captures belongs to the
/// application and not to this module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GestureHistoryRouter {
    /// One bit per stroke identity. `MAX_ACTIVE_STROKES` is exactly `u16::BITS`
    /// and `crate::gesture` asserts that at compile time, so this mask covers
    /// the whole identity space by construction.
    open_strokes: u16,
    entry: Option<HistoryGestureId>,
    next_id: u64,
}

impl Default for GestureHistoryRouter {
    fn default() -> Self {
        Self {
            open_strokes: 0,
            entry: None,
            next_id: 1,
        }
    }
}

impl GestureHistoryRouter {
    /// Identity of the one entry currently open, if any.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the open entry is read by the §4 router goldens; the host tracks its own accepted identity"
        )
    )]
    pub const fn entry(&self) -> Option<HistoryGestureId> {
        self.entry
    }

    /// Strokes the router believes are still down. This is history bookkeeping
    /// only; the recorded track keeps its own authoritative open-stroke set.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the open-stroke set is asserted by the §4 multi-touch golden"
        )
    )]
    pub const fn open_stroke_count(&self) -> u32 {
        self.open_strokes.count_ones()
    }

    /// Observe one normalized sample and report the manual-history boundary it
    /// crosses, if any.
    pub fn observe(
        &mut self,
        origin: GestureOrigin,
        phase: GesturePhase,
        stroke: u8,
    ) -> GestureHistoryStep {
        // Automation records nothing. Returning before the mask is even
        // computed keeps an automation stroke out of the open-stroke set, so a
        // controller sweep can never close an entry a person opened.
        if origin.is_automation() {
            return GestureHistoryStep::Idle;
        }
        let Some(mask) = Self::stroke_mask(stroke) else {
            return GestureHistoryStep::Idle;
        };
        match phase {
            GesturePhase::Begin => {
                if self.open_strokes & mask != 0 {
                    return GestureHistoryStep::Idle;
                }
                self.open_strokes |= mask;
                if self.entry.is_some() {
                    return GestureHistoryStep::Idle;
                }
                let id = self.allocate();
                self.entry = Some(id);
                GestureHistoryStep::Open(id)
            }
            // The whole point of §4. Motion between the boundaries never
            // reaches the bounded stack.
            GesturePhase::Move => GestureHistoryStep::Idle,
            GesturePhase::End => {
                if self.open_strokes & mask == 0 {
                    return GestureHistoryStep::Idle;
                }
                self.open_strokes &= !mask;
                if self.open_strokes != 0 {
                    return GestureHistoryStep::Idle;
                }
                match self.entry.take() {
                    Some(id) => GestureHistoryStep::Close(id),
                    None => GestureHistoryStep::Idle,
                }
            }
        }
    }

    /// Abandon whatever this router holds open and report the entry identity so
    /// the caller can cancel it. A generation barrier replaces the authored
    /// world the open checkpoint described, so the transaction is abandoned
    /// rather than committed against a program that no longer exists.
    pub fn abandon(&mut self) -> Option<HistoryGestureId> {
        self.open_strokes = 0;
        self.entry.take()
    }

    fn stroke_mask(stroke: u8) -> Option<u16> {
        (usize::from(stroke) < MAX_ACTIVE_STROKES).then(|| 1_u16 << stroke)
    }

    /// Allocate the next non-zero identity. The counter wraps rather than
    /// saturating, and zero is skipped, so a very long session keeps producing
    /// usable identities instead of failing closed on a boundary nobody can
    /// reach in practice.
    fn allocate(&mut self) -> HistoryGestureId {
        loop {
            let raw = self.next_id;
            self.next_id = raw.wrapping_add(1);
            if let Some(id) = HistoryGestureId::new(raw) {
                return id;
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistoryLimits {
    pub max_entries: usize,
    pub max_bytes: u64,
    pub max_checkpoint_bytes: u64,
}

impl Default for HistoryLimits {
    fn default() -> Self {
        Self {
            max_entries: HISTORY_MAX_ENTRIES,
            max_bytes: HISTORY_MAX_BYTES,
            max_checkpoint_bytes: HISTORY_MAX_CHECKPOINT_BYTES,
        }
    }
}

impl HistoryLimits {
    pub fn bounded(self) -> Result<Self, HistoryError> {
        if self.max_entries == 0
            || self.max_entries > HISTORY_MAX_ENTRIES
            || self.max_bytes == 0
            || self.max_bytes > HISTORY_MAX_BYTES
            || self.max_checkpoint_bytes == 0
            || self.max_checkpoint_bytes > HISTORY_MAX_CHECKPOINT_BYTES
            || self.max_checkpoint_bytes > self.max_bytes
        {
            return Err(HistoryError::InvalidLimits);
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HistoryError {
    InvalidLimits,
    CheckpointTooLarge {
        bytes: u64,
        limit: u64,
    },
    EmptyLabel,
    LabelTooLong {
        bytes: usize,
        limit: usize,
    },
    InvalidLabel,
    EmptyCategory,
    CategoryTooLong {
        bytes: usize,
        limit: usize,
    },
    InvalidCategory,
    GestureAlreadyOpen {
        open: HistoryGestureId,
        requested: HistoryGestureId,
    },
    GestureNotOpen,
    GestureMismatch {
        open: HistoryGestureId,
        requested: HistoryGestureId,
    },
    RestorePending,
    NoRestorePending,
    StaleRestoreToken,
    RestoreTokenExhausted,
    RestoreDescriptorMismatch,
    RestoreStackChanged,
}

impl fmt::Display for HistoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimits => formatter.write_str("history limits exceed the hard bounds"),
            Self::CheckpointTooLarge { bytes, limit } => {
                write!(
                    formatter,
                    "history checkpoint is {bytes} bytes; limit is {limit}"
                )
            }
            Self::EmptyLabel => formatter.write_str("history label must not be empty"),
            Self::LabelTooLong { bytes, limit } => {
                write!(
                    formatter,
                    "history label is {bytes} bytes; limit is {limit}"
                )
            }
            Self::InvalidLabel => {
                formatter.write_str("history label has surrounding whitespace or control text")
            }
            Self::EmptyCategory => formatter.write_str("history category must not be empty"),
            Self::CategoryTooLong { bytes, limit } => write!(
                formatter,
                "history category is {bytes} bytes; limit is {limit}"
            ),
            Self::InvalidCategory => formatter.write_str(
                "history category must contain lowercase ASCII letters, digits, '-' or '_'",
            ),
            Self::GestureAlreadyOpen { open, requested } => write!(
                formatter,
                "history gesture {} is already open; cannot nest gesture {}",
                open.get(),
                requested.get()
            ),
            Self::GestureNotOpen => formatter.write_str("no history gesture is open"),
            Self::GestureMismatch { open, requested } => write!(
                formatter,
                "history gesture {} cannot close open gesture {}",
                requested.get(),
                open.get()
            ),
            Self::RestorePending => {
                formatter.write_str("an undo/redo restore candidate is already pending")
            }
            Self::NoRestorePending => formatter.write_str("no undo/redo restore is pending"),
            Self::StaleRestoreToken => {
                formatter.write_str("undo/redo restore token is stale or already consumed")
            }
            Self::RestoreTokenExhausted => {
                formatter.write_str("undo/redo restore token identity space is exhausted")
            }
            Self::RestoreDescriptorMismatch => formatter
                .write_str("current undo/redo checkpoint does not describe the prepared mutation"),
            Self::RestoreStackChanged => {
                formatter.write_str("undo/redo stack changed after restore preflight")
            }
        }
    }
}

impl std::error::Error for HistoryError {}

fn validate_label(label: &str) -> Result<(), HistoryError> {
    if label.is_empty() {
        return Err(HistoryError::EmptyLabel);
    }
    if label.len() > HISTORY_MAX_LABEL_BYTES {
        return Err(HistoryError::LabelTooLong {
            bytes: label.len(),
            limit: HISTORY_MAX_LABEL_BYTES,
        });
    }
    if label.trim() != label || label.chars().any(char::is_control) {
        return Err(HistoryError::InvalidLabel);
    }
    Ok(())
}

fn validate_category(category: &str) -> Result<(), HistoryError> {
    if category.is_empty() {
        return Err(HistoryError::EmptyCategory);
    }
    if category.len() > HISTORY_MAX_CATEGORY_BYTES {
        return Err(HistoryError::CategoryTooLong {
            bytes: category.len(),
            limit: HISTORY_MAX_CATEGORY_BYTES,
        });
    }
    if !category.bytes().all(|byte| {
        byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_')
    }) {
        return Err(HistoryError::InvalidCategory);
    }
    Ok(())
}

pub fn fingerprint_canonical(bytes: &[u8]) -> HistoryFingerprint {
    Sha256::digest(bytes).into()
}

pub struct HistoryCheckpoint<T> {
    state: Arc<T>,
    fingerprint: HistoryFingerprint,
    bytes: u64,
    label: String,
    category: String,
}

impl<T> Clone for HistoryCheckpoint<T> {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
            fingerprint: self.fingerprint,
            bytes: self.bytes,
            label: self.label.clone(),
            category: self.category.clone(),
        }
    }
}

impl<T> fmt::Debug for HistoryCheckpoint<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HistoryCheckpoint")
            .field("fingerprint", &self.fingerprint)
            .field("bytes", &self.bytes)
            .field("label", &self.label)
            .field("category", &self.category)
            .finish_non_exhaustive()
    }
}

impl<T> HistoryCheckpoint<T> {
    pub fn from_canonical(
        state: T,
        canonical: &[u8],
        label: impl Into<String>,
        category: impl Into<String>,
    ) -> Result<Self, HistoryError> {
        let bytes = u64::try_from(canonical.len()).unwrap_or(u64::MAX);
        if bytes > HISTORY_MAX_CHECKPOINT_BYTES {
            return Err(HistoryError::CheckpointTooLarge {
                bytes,
                limit: HISTORY_MAX_CHECKPOINT_BYTES,
            });
        }
        let label = label.into();
        let category = category.into();
        validate_label(&label)?;
        validate_category(&category)?;
        Ok(Self {
            state: Arc::new(state),
            fingerprint: fingerprint_canonical(canonical),
            bytes,
            label,
            category,
        })
    }

    pub fn state(&self) -> &T {
        &self.state
    }

    pub const fn fingerprint(&self) -> HistoryFingerprint {
        self.fingerprint
    }

    pub fn label(&self) -> &str {
        &self.label
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryRecordOutcome {
    Recorded,
    NoChange,
    IgnoredNonManual,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryRestoreDirection {
    Undo,
    Redo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HistoryRestoreToken(NonZeroU64);

impl HistoryRestoreToken {
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "token identity is exposed for stale-token goldens"
        )
    )]
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

pub struct HistoryRestoreCandidate<T> {
    token: HistoryRestoreToken,
    direction: HistoryRestoreDirection,
    target: HistoryCheckpoint<T>,
}

impl<T> Clone for HistoryRestoreCandidate<T> {
    fn clone(&self) -> Self {
        Self {
            token: self.token,
            direction: self.direction,
            target: self.target.clone(),
        }
    }
}

impl<T> fmt::Debug for HistoryRestoreCandidate<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HistoryRestoreCandidate")
            .field("token", &self.token)
            .field("direction", &self.direction)
            .field("target", &self.target)
            .finish()
    }
}

impl<T> HistoryRestoreCandidate<T> {
    pub const fn token(&self) -> HistoryRestoreToken {
        self.token
    }

    pub fn target(&self) -> &T {
        self.target.state()
    }

    pub fn label(&self) -> &str {
        self.target.label()
    }

    pub fn checkpoint_current(
        &self,
        state: T,
        canonical: &[u8],
    ) -> Result<HistoryCheckpoint<T>, HistoryError> {
        HistoryCheckpoint::from_canonical(
            state,
            canonical,
            self.target.label.clone(),
            self.target.category.clone(),
        )
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HistoryMetrics {
    pub can_undo: bool,
    pub can_redo: bool,
    pub undo_depth: usize,
    pub redo_depth: usize,
    pub bytes: u64,
    pub max_entries: usize,
    pub max_bytes: u64,
    pub gesture_open: bool,
    pub restore_pending: bool,
    pub generation: u64,
}

struct OpenGesture<T> {
    id: HistoryGestureId,
    origin: MutationOrigin,
    before: HistoryCheckpoint<T>,
}

struct PendingRestore {
    token: HistoryRestoreToken,
    direction: HistoryRestoreDirection,
    target_fingerprint: HistoryFingerprint,
}

pub struct ManualHistory<T> {
    limits: HistoryLimits,
    undo: VecDeque<HistoryCheckpoint<T>>,
    redo: VecDeque<HistoryCheckpoint<T>>,
    undo_bytes: u64,
    redo_bytes: u64,
    open: Option<OpenGesture<T>>,
    pending: Option<PendingRestore>,
    next_restore_token: u64,
    generation: u64,
}

impl<T> Default for ManualHistory<T> {
    fn default() -> Self {
        Self::with_limits(HistoryLimits::default()).expect("default history limits are valid")
    }
}

impl<T> ManualHistory<T> {
    pub fn with_limits(limits: HistoryLimits) -> Result<Self, HistoryError> {
        Ok(Self {
            limits: limits.bounded()?,
            undo: VecDeque::new(),
            redo: VecDeque::new(),
            undo_bytes: 0,
            redo_bytes: 0,
            open: None,
            pending: None,
            next_restore_token: 1,
            generation: 1,
        })
    }

    pub fn begin_gesture(
        &mut self,
        id: HistoryGestureId,
        origin: MutationOrigin,
        before: HistoryCheckpoint<T>,
    ) -> Result<HistoryRecordOutcome, HistoryError> {
        if !origin.is_manual() {
            return Ok(HistoryRecordOutcome::IgnoredNonManual);
        }
        if self.pending.is_some() {
            return Err(HistoryError::RestorePending);
        }
        if let Some(open) = &self.open {
            return Err(HistoryError::GestureAlreadyOpen {
                open: open.id,
                requested: id,
            });
        }
        self.validate_checkpoint(&before)?;
        self.make_room_for_open(before.bytes);
        self.open = Some(OpenGesture { id, origin, before });
        self.bump_generation();
        Ok(HistoryRecordOutcome::Recorded)
    }

    pub fn finish_gesture(
        &mut self,
        id: HistoryGestureId,
        after_fingerprint: HistoryFingerprint,
    ) -> Result<HistoryRecordOutcome, HistoryError> {
        if self.pending.is_some() {
            return Err(HistoryError::RestorePending);
        }
        let Some(open) = self.open.as_ref() else {
            return Err(HistoryError::GestureNotOpen);
        };
        if open.id != id {
            return Err(HistoryError::GestureMismatch {
                open: open.id,
                requested: id,
            });
        }
        let open = self.open.take().expect("open gesture was checked");
        let outcome = if open.before.fingerprint == after_fingerprint {
            HistoryRecordOutcome::NoChange
        } else {
            self.clear_redo();
            self.push_undo(open.before);
            self.trim_to_limits();
            HistoryRecordOutcome::Recorded
        };
        self.bump_generation();
        Ok(outcome)
    }

    pub fn cancel_gesture(
        &mut self,
        id: HistoryGestureId,
    ) -> Result<HistoryRecordOutcome, HistoryError> {
        if self.pending.is_some() {
            return Err(HistoryError::RestorePending);
        }
        let Some(open) = self.open.as_ref() else {
            return Err(HistoryError::GestureNotOpen);
        };
        if open.id != id {
            return Err(HistoryError::GestureMismatch {
                open: open.id,
                requested: id,
            });
        }
        self.open = None;
        self.bump_generation();
        Ok(HistoryRecordOutcome::Cancelled)
    }

    pub fn record_manual(
        &mut self,
        origin: MutationOrigin,
        before: HistoryCheckpoint<T>,
        after_fingerprint: HistoryFingerprint,
    ) -> Result<HistoryRecordOutcome, HistoryError> {
        if !origin.is_manual() {
            return Ok(HistoryRecordOutcome::IgnoredNonManual);
        }
        if self.pending.is_some() {
            return Err(HistoryError::RestorePending);
        }
        if let Some(open) = &self.open {
            return Err(HistoryError::GestureAlreadyOpen {
                open: open.id,
                requested: open.id,
            });
        }
        self.validate_checkpoint(&before)?;
        if before.fingerprint == after_fingerprint {
            return Ok(HistoryRecordOutcome::NoChange);
        }
        self.clear_redo();
        self.push_undo(before);
        self.trim_to_limits();
        self.bump_generation();
        Ok(HistoryRecordOutcome::Recorded)
    }

    pub fn prepare_undo(&mut self) -> Result<Option<HistoryRestoreCandidate<T>>, HistoryError> {
        self.prepare_restore(HistoryRestoreDirection::Undo)
    }

    pub fn prepare_redo(&mut self) -> Result<Option<HistoryRestoreCandidate<T>>, HistoryError> {
        self.prepare_restore(HistoryRestoreDirection::Redo)
    }

    fn prepare_restore(
        &mut self,
        direction: HistoryRestoreDirection,
    ) -> Result<Option<HistoryRestoreCandidate<T>>, HistoryError> {
        if self.pending.is_some() {
            return Err(HistoryError::RestorePending);
        }
        if let Some(open) = &self.open {
            return Err(HistoryError::GestureAlreadyOpen {
                open: open.id,
                requested: open.id,
            });
        }
        let target = match direction {
            HistoryRestoreDirection::Undo => self.undo.back(),
            HistoryRestoreDirection::Redo => self.redo.back(),
        };
        let Some(target) = target.cloned() else {
            return Ok(None);
        };
        let token = self.allocate_restore_token()?;
        self.pending = Some(PendingRestore {
            token,
            direction,
            target_fingerprint: target.fingerprint,
        });
        self.bump_generation();
        Ok(Some(HistoryRestoreCandidate {
            token,
            direction,
            target,
        }))
    }

    pub fn reject_restore(&mut self, token: HistoryRestoreToken) -> Result<(), HistoryError> {
        let pending = self
            .pending
            .as_ref()
            .ok_or(HistoryError::NoRestorePending)?;
        if pending.token != token {
            return Err(HistoryError::StaleRestoreToken);
        }
        self.pending = None;
        self.bump_generation();
        Ok(())
    }

    pub fn commit_restore(
        &mut self,
        token: HistoryRestoreToken,
        current: HistoryCheckpoint<T>,
    ) -> Result<(), HistoryError> {
        let pending = self
            .pending
            .as_ref()
            .ok_or(HistoryError::NoRestorePending)?;
        if pending.token != token {
            return Err(HistoryError::StaleRestoreToken);
        }
        self.validate_checkpoint(&current)?;
        let target = match pending.direction {
            HistoryRestoreDirection::Undo => self.undo.back(),
            HistoryRestoreDirection::Redo => self.redo.back(),
        }
        .ok_or(HistoryError::RestoreStackChanged)?;
        if target.fingerprint != pending.target_fingerprint {
            return Err(HistoryError::RestoreStackChanged);
        }
        if target.label != current.label || target.category != current.category {
            return Err(HistoryError::RestoreDescriptorMismatch);
        }

        let direction = pending.direction;
        self.pending = None;
        match direction {
            HistoryRestoreDirection::Undo => {
                let removed = self.undo.pop_back().expect("prepared undo target exists");
                self.undo_bytes = self.undo_bytes.saturating_sub(removed.bytes);
                self.redo_bytes = self.redo_bytes.saturating_add(current.bytes);
                self.redo.push_back(current);
            }
            HistoryRestoreDirection::Redo => {
                let removed = self.redo.pop_back().expect("prepared redo target exists");
                self.redo_bytes = self.redo_bytes.saturating_sub(removed.bytes);
                self.undo_bytes = self.undo_bytes.saturating_add(current.bytes);
                self.undo.push_back(current);
            }
        }
        self.trim_to_limits();
        self.bump_generation();
        Ok(())
    }

    pub fn metrics(&self) -> HistoryMetrics {
        let blocked = self.open.is_some() || self.pending.is_some();
        HistoryMetrics {
            can_undo: !blocked && !self.undo.is_empty(),
            can_redo: !blocked && !self.redo.is_empty(),
            undo_depth: self.undo.len(),
            redo_depth: self.redo.len(),
            bytes: self
                .undo_bytes
                .saturating_add(self.redo_bytes)
                .saturating_add(self.open.as_ref().map_or(0, |open| open.before.bytes)),
            max_entries: self.limits.max_entries,
            max_bytes: self.limits.max_bytes,
            gesture_open: self.open.is_some(),
            restore_pending: self.pending.is_some(),
            generation: self.generation,
        }
    }

    /// Label of the next undo target, if one is currently retained. This is
    /// read-only operator telemetry; it does not expose or clone the authored
    /// world stored by the checkpoint.
    pub fn undo_label(&self) -> Option<&str> {
        self.undo.back().map(HistoryCheckpoint::label)
    }

    /// Label of the next redo target, if one is currently retained.
    pub fn redo_label(&self) -> Option<&str> {
        self.redo.back().map(HistoryCheckpoint::label)
    }

    /// Identity and controller class of the one open gesture. Hosts use this
    /// to prevent simultaneous native/browser edits from being folded into
    /// another controller's transaction.
    pub fn open_gesture(&self) -> Option<(HistoryGestureId, MutationOrigin)> {
        self.open.as_ref().map(|open| (open.id, open.origin))
    }

    fn validate_checkpoint(&self, checkpoint: &HistoryCheckpoint<T>) -> Result<(), HistoryError> {
        if checkpoint.bytes > self.limits.max_checkpoint_bytes {
            return Err(HistoryError::CheckpointTooLarge {
                bytes: checkpoint.bytes,
                limit: self.limits.max_checkpoint_bytes,
            });
        }
        Ok(())
    }

    fn allocate_restore_token(&mut self) -> Result<HistoryRestoreToken, HistoryError> {
        let raw =
            NonZeroU64::new(self.next_restore_token).ok_or(HistoryError::RestoreTokenExhausted)?;
        self.next_restore_token = raw.get().checked_add(1).unwrap_or(0);
        Ok(HistoryRestoreToken(raw))
    }

    fn clear_redo(&mut self) {
        self.redo.clear();
        self.redo_bytes = 0;
    }

    fn push_undo(&mut self, checkpoint: HistoryCheckpoint<T>) {
        self.undo_bytes = self.undo_bytes.saturating_add(checkpoint.bytes);
        self.undo.push_back(checkpoint);
    }

    fn trim_to_limits(&mut self) {
        while self.undo.len() + self.redo.len() > self.limits.max_entries
            || self.undo_bytes.saturating_add(self.redo_bytes) > self.limits.max_bytes
        {
            if let Some(removed) = self.undo.pop_front() {
                self.undo_bytes = self.undo_bytes.saturating_sub(removed.bytes);
            } else if let Some(removed) = self.redo.pop_front() {
                self.redo_bytes = self.redo_bytes.saturating_sub(removed.bytes);
            } else {
                break;
            }
        }
    }

    fn make_room_for_open(&mut self, open_bytes: u64) {
        while self.undo.len() + self.redo.len() + 1 > self.limits.max_entries
            || self
                .undo_bytes
                .saturating_add(self.redo_bytes)
                .saturating_add(open_bytes)
                > self.limits.max_bytes
        {
            if let Some(removed) = self.undo.pop_front() {
                self.undo_bytes = self.undo_bytes.saturating_sub(removed.bytes);
            } else if let Some(removed) = self.redo.pop_front() {
                self.redo_bytes = self.redo_bytes.saturating_sub(removed.bytes);
            } else {
                break;
            }
        }
    }

    fn bump_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1).max(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint(value: u64, label: &str) -> HistoryCheckpoint<u64> {
        HistoryCheckpoint::from_canonical(value, &value.to_le_bytes(), label, "transform").unwrap()
    }

    fn fingerprint(value: u64) -> HistoryFingerprint {
        fingerprint_canonical(&value.to_le_bytes())
    }

    #[test]
    fn one_gesture_records_one_before_checkpoint_and_no_change_records_nothing() {
        let mut history = ManualHistory::default();
        let id = HistoryGestureId::new(7).unwrap();
        assert_eq!(
            history
                .begin_gesture(id, MutationOrigin::BrowserManual, checkpoint(10, "Move X"))
                .unwrap(),
            HistoryRecordOutcome::Recorded
        );
        // Any number of scalar updates may happen between these boundaries;
        // only the final authored fingerprint decides the one transaction.
        assert_eq!(
            history.finish_gesture(id, fingerprint(42)).unwrap(),
            HistoryRecordOutcome::Recorded
        );
        assert_eq!(history.metrics().undo_depth, 1);

        let next = HistoryGestureId::new(8).unwrap();
        history
            .begin_gesture(
                next,
                MutationOrigin::BrowserManual,
                checkpoint(42, "Move X"),
            )
            .unwrap();
        assert_eq!(
            history.finish_gesture(next, fingerprint(42)).unwrap(),
            HistoryRecordOutcome::NoChange
        );
        assert_eq!(history.metrics().undo_depth, 1);
    }

    #[test]
    fn nested_and_mismatched_gestures_reject_without_changing_open_transaction() {
        let mut history = ManualHistory::default();
        let first = HistoryGestureId::new(1).unwrap();
        let second = HistoryGestureId::new(2).unwrap();
        history
            .begin_gesture(first, MutationOrigin::NativeManual, checkpoint(1, "Scale"))
            .unwrap();
        let generation = history.metrics().generation;
        assert!(matches!(
            history.begin_gesture(
                second,
                MutationOrigin::BrowserManual,
                checkpoint(2, "Scale")
            ),
            Err(HistoryError::GestureAlreadyOpen { .. })
        ));
        assert!(matches!(
            history.finish_gesture(second, fingerprint(3)),
            Err(HistoryError::GestureMismatch { .. })
        ));
        assert_eq!(history.metrics().generation, generation);
        history.finish_gesture(first, fingerprint(3)).unwrap();
        assert_eq!(history.metrics().undo_depth, 1);
    }

    #[test]
    fn automation_origins_never_enter_history_or_clear_redo() {
        let mut history = ManualHistory::default();
        history
            .record_manual(
                MutationOrigin::BrowserManual,
                checkpoint(1, "Opacity"),
                fingerprint(2),
            )
            .unwrap();
        let candidate = history.prepare_undo().unwrap().unwrap();
        let current = candidate
            .checkpoint_current(2, &2_u64.to_le_bytes())
            .unwrap();
        history.commit_restore(candidate.token(), current).unwrap();
        assert_eq!(history.metrics().redo_depth, 1);

        for origin in [
            MutationOrigin::Midi,
            MutationOrigin::Lfo,
            MutationOrigin::Audio,
            MutationOrigin::Osc,
            MutationOrigin::ClipTrigger,
            MutationOrigin::CollisionScore,
            MutationOrigin::Automation,
        ] {
            assert_eq!(
                history
                    .record_manual(origin, checkpoint(9, "Automation"), fingerprint(10))
                    .unwrap(),
                HistoryRecordOutcome::IgnoredNonManual
            );
        }
        assert_eq!(history.metrics().redo_depth, 1);
    }

    #[test]
    fn two_phase_restore_tokens_are_single_use_and_stale_candidates_reject() {
        let mut history = ManualHistory::default();
        history
            .record_manual(
                MutationOrigin::BrowserManual,
                checkpoint(1001, "Stable layer"),
                fingerprint(2002),
            )
            .unwrap();
        let undo = history.prepare_undo().unwrap().unwrap();
        assert_eq!(*undo.target(), 1001);
        let current = undo
            .checkpoint_current(2002, &2002_u64.to_le_bytes())
            .unwrap();
        history.commit_restore(undo.token(), current).unwrap();
        assert!(matches!(
            history.commit_restore(
                undo.token(),
                undo.checkpoint_current(2002, &2002_u64.to_le_bytes())
                    .unwrap()
            ),
            Err(HistoryError::NoRestorePending)
        ));

        let redo = history.prepare_redo().unwrap().unwrap();
        let wrong = HistoryRestoreToken(NonZeroU64::new(redo.token().get() + 1).unwrap());
        assert!(matches!(
            history.reject_restore(wrong),
            Err(HistoryError::StaleRestoreToken)
        ));
        history.reject_restore(redo.token()).unwrap();
        assert!(matches!(
            history.reject_restore(redo.token()),
            Err(HistoryError::NoRestorePending)
        ));

        history.next_restore_token = u64::MAX;
        let final_token = history.prepare_redo().unwrap().unwrap();
        assert_eq!(final_token.token().get(), u64::MAX);
        history.reject_restore(final_token.token()).unwrap();
        assert!(matches!(
            history.prepare_redo(),
            Err(HistoryError::RestoreTokenExhausted)
        ));
    }

    #[test]
    fn redo_clears_only_after_a_changed_manual_commit() {
        let mut history = ManualHistory::default();
        history
            .record_manual(
                MutationOrigin::BrowserManual,
                checkpoint(1, "Opacity"),
                fingerprint(2),
            )
            .unwrap();
        let undo = history.prepare_undo().unwrap().unwrap();
        history
            .commit_restore(
                undo.token(),
                undo.checkpoint_current(2, &2_u64.to_le_bytes()).unwrap(),
            )
            .unwrap();
        assert_eq!(history.metrics().redo_depth, 1);
        assert_eq!(
            history
                .record_manual(
                    MutationOrigin::NativeManual,
                    checkpoint(1, "No-op"),
                    fingerprint(1),
                )
                .unwrap(),
            HistoryRecordOutcome::NoChange
        );
        assert_eq!(history.metrics().redo_depth, 1);
        history
            .record_manual(
                MutationOrigin::NativeManual,
                checkpoint(1, "New value"),
                fingerprint(3),
            )
            .unwrap();
        assert_eq!(history.metrics().redo_depth, 0);
    }

    #[test]
    fn count_and_byte_limits_evict_the_oldest_checkpoints() {
        let limits = HistoryLimits {
            max_entries: 2,
            max_bytes: 16,
            max_checkpoint_bytes: 8,
        };
        let mut history = ManualHistory::with_limits(limits).unwrap();
        for value in 1..=3 {
            history
                .record_manual(
                    MutationOrigin::BrowserManual,
                    checkpoint(value, "Edit"),
                    fingerprint(value + 1),
                )
                .unwrap();
        }
        assert_eq!(history.metrics().undo_depth, 2);
        assert_eq!(history.metrics().bytes, 16);
        let newest = history.prepare_undo().unwrap().unwrap();
        assert_eq!(*newest.target(), 3);
    }

    #[test]
    fn open_gesture_is_included_in_aggregate_count_and_byte_caps() {
        let limits = HistoryLimits {
            max_entries: 2,
            max_bytes: 16,
            max_checkpoint_bytes: 8,
        };
        let mut history = ManualHistory::with_limits(limits).unwrap();
        for value in 1..=2 {
            history
                .record_manual(
                    MutationOrigin::BrowserManual,
                    checkpoint(value, "Edit"),
                    fingerprint(value + 1),
                )
                .unwrap();
        }
        let gesture = HistoryGestureId::new(17).unwrap();
        history
            .begin_gesture(
                gesture,
                MutationOrigin::BrowserManual,
                checkpoint(3, "Drag"),
            )
            .unwrap();
        let metrics = history.metrics();
        assert_eq!(metrics.undo_depth, 1);
        assert_eq!(metrics.bytes, limits.max_bytes);
        assert!(metrics.gesture_open);
        history.cancel_gesture(gesture).unwrap();
    }

    /// Drive one router step into a real store, exactly as the host does: an
    /// `Open` takes a before-checkpoint, a `Close` finishes against the value
    /// the world now holds, and an `Idle` is genuinely nothing.
    fn route(
        router: &mut GestureHistoryRouter,
        history: &mut ManualHistory<u64>,
        origin: GestureOrigin,
        phase: GesturePhase,
        stroke: u8,
        before: u64,
        after: u64,
    ) -> GestureHistoryStep {
        let step = router.observe(origin, phase, stroke);
        match step {
            GestureHistoryStep::Idle => {}
            GestureHistoryStep::Open(id) => {
                history
                    .begin_gesture(
                        id,
                        MutationOrigin::for_gesture_origin(origin),
                        checkpoint(before, "Etch gesture"),
                    )
                    .unwrap();
            }
            GestureHistoryStep::Close(id) => {
                history.finish_gesture(id, fingerprint(after)).unwrap();
            }
        }
        step
    }

    #[test]
    fn a_five_hundred_sample_authored_stroke_yields_exactly_one_undo_entry() {
        let mut router = GestureHistoryRouter::default();
        let mut history = ManualHistory::<u64>::default();

        assert!(matches!(
            route(
                &mut router,
                &mut history,
                GestureOrigin::NativePointer,
                GesturePhase::Begin,
                0,
                1,
                2,
            ),
            GestureHistoryStep::Open(_)
        ));
        assert!(history.metrics().gesture_open);

        // Four hundred and ninety-eight motion samples plus the opening Begin
        // and the closing End are five hundred events in one stroke.
        for _ in 0..498 {
            assert_eq!(
                route(
                    &mut router,
                    &mut history,
                    GestureOrigin::NativePointer,
                    GesturePhase::Move,
                    0,
                    1,
                    2,
                ),
                GestureHistoryStep::Idle,
                "pointer motion must never reach the bounded history stack"
            );
        }
        assert_eq!(history.metrics().undo_depth, 0);

        assert!(matches!(
            route(
                &mut router,
                &mut history,
                GestureOrigin::NativePointer,
                GesturePhase::End,
                0,
                1,
                2,
            ),
            GestureHistoryStep::Close(_)
        ));

        let metrics = history.metrics();
        assert_eq!(
            metrics.undo_depth, 1,
            "one completed authored gesture is exactly one manual-history entry"
        );
        assert!(!metrics.gesture_open);
        assert_eq!(router.open_stroke_count(), 0);
        assert_eq!(router.entry(), None);
    }

    #[test]
    fn automation_origin_gestures_open_no_entry_and_never_close_an_authored_one() {
        let mut router = GestureHistoryRouter::default();
        let mut history = ManualHistory::<u64>::default();

        // A complete MIDI and a complete OSC stroke each record nothing.
        for origin in [GestureOrigin::Midi, GestureOrigin::Osc] {
            for phase in [GesturePhase::Begin, GesturePhase::Move, GesturePhase::End] {
                assert_eq!(
                    route(&mut router, &mut history, origin, phase, 3, 1, 2),
                    GestureHistoryStep::Idle,
                    "{origin} is automation and must record no manual history"
                );
            }
            assert!(!MutationOrigin::for_gesture_origin(origin).is_manual());
        }
        assert_eq!(history.metrics().undo_depth, 0);
        assert!(!history.metrics().gesture_open);
        assert_eq!(router.open_stroke_count(), 0);

        // An automation stroke sharing a person's stroke identity cannot close
        // the entry the person opened.
        route(
            &mut router,
            &mut history,
            GestureOrigin::Phone,
            GesturePhase::Begin,
            3,
            1,
            2,
        );
        let owned = router.entry().expect("the phone opened an entry");
        assert_eq!(
            router.observe(GestureOrigin::Midi, GesturePhase::End, 3),
            GestureHistoryStep::Idle
        );
        assert_eq!(router.entry(), Some(owned));
        assert!(history.metrics().gesture_open);
        assert_eq!(
            route(
                &mut router,
                &mut history,
                GestureOrigin::Phone,
                GesturePhase::End,
                3,
                1,
                2,
            ),
            GestureHistoryStep::Close(owned)
        );
        assert_eq!(history.metrics().undo_depth, 1);

        assert_eq!(
            MutationOrigin::for_gesture_origin(GestureOrigin::NativePointer),
            MutationOrigin::NativeManual
        );
        assert_eq!(
            MutationOrigin::for_gesture_origin(GestureOrigin::Phone),
            MutationOrigin::BrowserManual
        );
    }

    #[test]
    fn a_multi_touch_gesture_is_one_entry_opened_at_the_first_begin_and_closed_at_the_last_end() {
        let mut router = GestureHistoryRouter::default();
        let mut history = ManualHistory::<u64>::default();

        let GestureHistoryStep::Open(first) = route(
            &mut router,
            &mut history,
            GestureOrigin::Phone,
            GesturePhase::Begin,
            0,
            1,
            2,
        ) else {
            panic!("the first Begin opens the one entry");
        };
        for stroke in 1..u8::try_from(MAX_ACTIVE_STROKES).unwrap() {
            assert_eq!(
                route(
                    &mut router,
                    &mut history,
                    GestureOrigin::Phone,
                    GesturePhase::Begin,
                    stroke,
                    1,
                    2,
                ),
                GestureHistoryStep::Idle,
                "a stroke joining an open gesture never opens a second entry"
            );
        }
        assert_eq!(
            router.open_stroke_count(),
            u32::try_from(MAX_ACTIVE_STROKES).unwrap()
        );
        // A duplicate Begin and an out-of-range identity are both inert.
        assert_eq!(
            router.observe(GestureOrigin::Phone, GesturePhase::Begin, 0),
            GestureHistoryStep::Idle
        );
        assert_eq!(
            router.observe(
                GestureOrigin::Phone,
                GesturePhase::Begin,
                u8::try_from(MAX_ACTIVE_STROKES).unwrap()
            ),
            GestureHistoryStep::Idle
        );
        assert_eq!(
            router.observe(GestureOrigin::Phone, GesturePhase::Begin, 255),
            GestureHistoryStep::Idle
        );
        assert_eq!(
            router.open_stroke_count(),
            u32::try_from(MAX_ACTIVE_STROKES).unwrap()
        );

        for stroke in 0..u8::try_from(MAX_ACTIVE_STROKES).unwrap() - 1 {
            assert_eq!(
                route(
                    &mut router,
                    &mut history,
                    GestureOrigin::Phone,
                    GesturePhase::End,
                    stroke,
                    1,
                    2,
                ),
                GestureHistoryStep::Idle,
                "the entry stays open while any stroke is still down"
            );
        }
        assert!(history.metrics().gesture_open);
        assert_eq!(
            route(
                &mut router,
                &mut history,
                GestureOrigin::Phone,
                GesturePhase::End,
                u8::try_from(MAX_ACTIVE_STROKES).unwrap() - 1,
                1,
                2,
            ),
            GestureHistoryStep::Close(first)
        );
        assert_eq!(history.metrics().undo_depth, 1);

        // An orphan End and a Move with nothing open are both inert.
        assert_eq!(
            router.observe(GestureOrigin::Phone, GesturePhase::End, 0),
            GestureHistoryStep::Idle
        );
        assert_eq!(
            router.observe(GestureOrigin::Phone, GesturePhase::Move, 0),
            GestureHistoryStep::Idle
        );
        assert_eq!(router.entry(), None);
    }

    #[test]
    fn an_unchanged_gesture_records_nothing_and_a_barrier_abandons_an_open_one() {
        let mut router = GestureHistoryRouter::default();
        let mut history = ManualHistory::<u64>::default();

        // A stroke that leaves the authored world exactly as it found it is
        // NoChange, not an empty undo step.
        let GestureHistoryStep::Open(id) = route(
            &mut router,
            &mut history,
            GestureOrigin::NativePointer,
            GesturePhase::Begin,
            2,
            7,
            7,
        ) else {
            panic!("Begin opens an entry");
        };
        assert_eq!(
            history.finish_gesture(id, fingerprint(7)).unwrap(),
            HistoryRecordOutcome::NoChange
        );
        assert_eq!(
            router.observe(GestureOrigin::NativePointer, GesturePhase::End, 2),
            GestureHistoryStep::Close(id)
        );
        assert_eq!(history.metrics().undo_depth, 0);

        // A generation barrier abandons the open transaction rather than
        // committing it against a program that no longer exists.
        let GestureHistoryStep::Open(open) = route(
            &mut router,
            &mut history,
            GestureOrigin::Phone,
            GesturePhase::Begin,
            5,
            11,
            12,
        ) else {
            panic!("Begin opens an entry");
        };
        router.observe(GestureOrigin::Phone, GesturePhase::Begin, 6);
        assert_eq!(router.abandon(), Some(open));
        assert_eq!(router.open_stroke_count(), 0);
        assert_eq!(router.entry(), None);
        history.cancel_gesture(open).unwrap();
        assert_eq!(history.metrics().undo_depth, 0);
        assert!(!history.metrics().gesture_open);
        assert_eq!(router.abandon(), None);
    }

    #[test]
    fn router_identities_are_non_zero_distinct_and_survive_a_counter_wrap() {
        let mut router = GestureHistoryRouter::default();
        let mut seen = Vec::new();
        for stroke in 0..4_u8 {
            let GestureHistoryStep::Open(id) =
                router.observe(GestureOrigin::NativePointer, GesturePhase::Begin, stroke)
            else {
                panic!("each completed gesture opens its own entry");
            };
            assert_eq!(
                router.observe(GestureOrigin::NativePointer, GesturePhase::End, stroke),
                GestureHistoryStep::Close(id)
            );
            assert!(!seen.contains(&id.get()));
            seen.push(id.get());
        }
        assert_eq!(seen, vec![1, 2, 3, 4]);

        router.next_id = u64::MAX;
        let GestureHistoryStep::Open(last) =
            router.observe(GestureOrigin::Phone, GesturePhase::Begin, 0)
        else {
            panic!("Begin opens an entry");
        };
        assert_eq!(last.get(), u64::MAX);
        router.observe(GestureOrigin::Phone, GesturePhase::End, 0);
        let GestureHistoryStep::Open(wrapped) =
            router.observe(GestureOrigin::Phone, GesturePhase::Begin, 0)
        else {
            panic!("Begin opens an entry");
        };
        assert_eq!(wrapped.get(), 1, "zero is skipped rather than allocated");
    }

    #[test]
    fn labels_categories_and_checkpoint_bytes_are_strictly_bounded() {
        assert!(matches!(
            HistoryCheckpoint::from_canonical((), b"x", "", "rack"),
            Err(HistoryError::EmptyLabel)
        ));
        assert!(matches!(
            HistoryCheckpoint::from_canonical((), b"x", "Edit\nnow", "rack"),
            Err(HistoryError::InvalidLabel)
        ));
        assert!(matches!(
            HistoryCheckpoint::from_canonical((), b"x", "Edit", "Rack Values"),
            Err(HistoryError::InvalidCategory)
        ));
        let huge = vec![0_u8; HISTORY_MAX_CHECKPOINT_BYTES as usize + 1];
        assert!(matches!(
            HistoryCheckpoint::from_canonical((), &huge, "Edit", "rack"),
            Err(HistoryError::CheckpointTooLarge { .. })
        ));
    }
}
