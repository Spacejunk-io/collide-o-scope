//! Pure Scene-only beat Autopilot scheduler.
//!
//! Async preparation, GPU commits, clocks, and visible output stay with the
//! host. This state machine owns only deterministic sequencing. The host must
//! snapshot preparation readiness before a frame begins and pass that snapshot
//! in [`AutopilotFrameInput::readiness_before_frame`]. A completion first
//! observed later in the frame is supplied on the next call, so it can never
//! consume a beat crossing from the frame in which it arrived.

use std::fmt;

use super::{AutopilotPlan, AutopilotRepeat, AutopilotSceneReferenceIssue, SceneId, Scenes};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AutopilotState {
    #[default]
    Stopped,
    Starting,
    Running,
    Paused,
    Stalled,
    Faulted,
    Complete,
}

/// Epoch- and sequence-qualified identity for one preparation/commit request.
/// Stable Scene IDs are not sufficient because duplicates and one-step loops
/// are legal; `sequence` rejects a late result from an earlier occurrence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AutopilotTarget {
    pub epoch: u64,
    pub sequence: u64,
    pub step_index: usize,
    pub scene_id: SceneId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutopilotCommand {
    /// Cancel only work owned by the named, now-obsolete Autopilot epoch.
    CancelOwned { epoch: u64 },
    /// Prepare exactly one lookahead Scene without changing visible output.
    Prepare(AutopilotTarget),
    /// Atomically make the already-prepared Scene visible.
    Commit(AutopilotTarget),
}

pub type AutopilotCommands = Vec<AutopilotCommand>;

/// Preparation truth captured before this program frame began.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AutopilotReadinessBeforeFrame {
    #[default]
    Pending,
    Ready(AutopilotTarget),
    #[allow(
        dead_code,
        reason = "hosts may report preparation failure here or through fault()"
    )]
    Faulted(AutopilotTarget),
}

/// One accepted program-frame observation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AutopilotFrameInput {
    /// All integer media-beat boundaries crossed since the prior observation.
    /// Every crossing advances the ordinal, but no frame can emit more than
    /// one commit and overshoot never causes catch-up cuts.
    pub crossed_beats: u32,
    /// False for freeze, pause, or any other clock-reanchor frame. Crossings
    /// are discarded in that case; preparation readiness may still advance.
    pub media_running: bool,
    pub readiness_before_frame: AutopilotReadinessBeforeFrame,
}

#[allow(
    dead_code,
    reason = "convenience constructors keep embedders and pure fixtures explicit"
)]
impl AutopilotFrameInput {
    pub const fn running(
        crossed_beats: u32,
        readiness_before_frame: AutopilotReadinessBeforeFrame,
    ) -> Self {
        Self {
            crossed_beats,
            media_running: true,
            readiness_before_frame,
        }
    }

    pub const fn reanchor(readiness_before_frame: AutopilotReadinessBeforeFrame) -> Self {
        Self {
            crossed_beats: 0,
            media_running: false,
            readiness_before_frame,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutopilotPlayError {
    EmptyPlan,
    MissingScene(AutopilotSceneReferenceIssue),
}

impl fmt::Display for AutopilotPlayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPlan => formatter.write_str("Autopilot requires at least one Scene step"),
            Self::MissingScene(issue) => issue.fmt(formatter),
        }
    }
}

impl std::error::Error for AutopilotPlayError {}

/// A rejected non-empty Play can still cancel an obsolete epoch, so callers
/// receive both the semantic error and any required cancellation commands.
/// The special empty-plan rejection leaves runtime state wholly untouched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutopilotPlayFailure {
    pub error: AutopilotPlayError,
    pub commands: AutopilotCommands,
}

impl fmt::Display for AutopilotPlayFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(formatter)
    }
}

impl std::error::Error for AutopilotPlayFailure {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutopilotFault {
    EmptyPlan,
    MissingScene(AutopilotSceneReferenceIssue),
    Preparation(AutopilotTarget),
    Commit(AutopilotTarget),
    Reference { scene_id: SceneId },
    Invariant,
}

/// Runtime-only scheduler state. It intentionally implements no serialization
/// traits; only [`AutopilotPlan`] belongs in a patch.
#[derive(Debug, Clone)]
pub struct AutopilotScheduler {
    state: AutopilotState,
    resume_state: Option<AutopilotState>,
    epoch: u64,
    next_sequence: u64,
    plan: Option<AutopilotPlan>,
    cursor: usize,
    active_cursor: Option<usize>,
    last_commit: Option<(AutopilotTarget, Option<usize>)>,
    accepted_media_beats: u64,
    due_media_beat: Option<u64>,
    expected: Option<AutopilotTarget>,
    expected_ready: bool,
    suppress_first_release_frame: bool,
    fault: Option<AutopilotFault>,
}

impl Default for AutopilotScheduler {
    fn default() -> Self {
        Self {
            state: AutopilotState::Stopped,
            resume_state: None,
            epoch: 0,
            next_sequence: 0,
            plan: None,
            cursor: 0,
            active_cursor: None,
            last_commit: None,
            accepted_media_beats: 0,
            due_media_beat: None,
            expected: None,
            expected_ready: false,
            suppress_first_release_frame: false,
            fault: None,
        }
    }
}

impl AutopilotScheduler {
    pub fn state(&self) -> AutopilotState {
        self.state
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    #[allow(dead_code, reason = "pure scheduler reset/cursor inspection API")]
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn active_step_index(&self) -> Option<usize> {
        self.active_cursor
    }

    #[allow(dead_code, reason = "pure scheduler ordinal inspection API")]
    pub fn accepted_media_beats(&self) -> u64 {
        self.accepted_media_beats
    }

    #[allow(dead_code, reason = "pure scheduler due-beat inspection API")]
    pub fn due_media_beat(&self) -> Option<u64> {
        self.due_media_beat
    }

    pub fn remaining_hold_beats(&self) -> Option<u64> {
        self.due_media_beat
            .map(|due| due.saturating_sub(self.accepted_media_beats))
    }

    #[allow(dead_code, reason = "pure scheduler visible-Scene inspection API")]
    pub fn active_scene_id(&self) -> Option<SceneId> {
        self.plan
            .as_ref()?
            .steps
            .get(self.active_cursor?)
            .map(|step| step.scene_id)
    }

    pub fn expected_target(&self) -> Option<AutopilotTarget> {
        self.expected
    }

    #[allow(dead_code, reason = "pure scheduler fault inspection API")]
    pub fn fault_reason(&self) -> Option<AutopilotFault> {
        self.fault
    }

    /// Validate and arm a fresh sequence. Visible output is unchanged: the
    /// first command only prepares Scene 0, and a release is forbidden until
    /// a later media-running frame crosses a beat.
    pub fn play(
        &mut self,
        plan: &AutopilotPlan,
        scenes: &Scenes,
    ) -> Result<AutopilotCommands, AutopilotPlayFailure> {
        if plan.is_empty() {
            return Err(AutopilotPlayFailure {
                error: AutopilotPlayError::EmptyPlan,
                commands: Vec::new(),
            });
        }
        let validation = plan
            .validate_scenes(scenes)
            .map_err(AutopilotPlayError::MissingScene);

        let old_epoch = self.advance_epoch();
        self.clear_sequence_state();
        let mut commands = vec![AutopilotCommand::CancelOwned { epoch: old_epoch }];

        if let Err(error) = validation {
            self.state = AutopilotState::Faulted;
            self.fault = Some(match error {
                AutopilotPlayError::EmptyPlan => AutopilotFault::EmptyPlan,
                AutopilotPlayError::MissingScene(issue) => AutopilotFault::MissingScene(issue),
            });
            return Err(AutopilotPlayFailure { error, commands });
        }

        self.plan = Some(plan.clone());
        self.state = AutopilotState::Starting;
        self.suppress_first_release_frame = true;
        if let Some(command) = self.prepare_step(0) {
            commands.push(command);
        }
        Ok(commands)
    }

    /// Stop and forget runtime sequencing without changing the visible Scene.
    /// Authored plan data remains owned by the caller/PatchState.
    pub fn reset(&mut self) -> AutopilotCommands {
        let old_epoch = self.advance_epoch();
        self.clear_sequence_state();
        self.state = AutopilotState::Stopped;
        vec![AutopilotCommand::CancelOwned { epoch: old_epoch }]
    }

    pub fn pause(&mut self) -> bool {
        if matches!(
            self.state,
            AutopilotState::Starting | AutopilotState::Running | AutopilotState::Stalled
        ) {
            self.resume_state = Some(self.state);
            self.state = AutopilotState::Paused;
            true
        } else {
            false
        }
    }

    pub fn resume(&mut self) -> bool {
        if self.state != AutopilotState::Paused {
            return false;
        }
        self.state = self.resume_state.take().unwrap_or(AutopilotState::Running);
        true
    }

    /// Advance one accepted program frame. `readiness_before_frame` must be a
    /// snapshot taken before this frame's async poll. Readiness is latched, so
    /// callers may pass `Pending` on later frames after one matching `Ready`.
    pub fn advance_frame(&mut self, input: AutopilotFrameInput) -> AutopilotCommands {
        // The common frame emits nothing (including every frame while
        // stopped), so keep that path allocation-free. A command-bearing
        // boundary may allocate for its permanently bounded one-or-two items.
        let mut commands = Vec::new();
        if let Some(command) = self.accept_readiness(input.readiness_before_frame) {
            commands.push(command);
            return commands;
        }

        if !matches!(
            self.state,
            AutopilotState::Starting
                | AutopilotState::Running
                | AutopilotState::Paused
                | AutopilotState::Stalled
        ) {
            return commands;
        }

        if self.state == AutopilotState::Paused {
            return commands;
        }

        // Play may be requested after the host has already measured this
        // frame's crossings. Suppressing one scheduler frame makes the first
        // release unambiguously future-facing even when Scene 0 was cached.
        if self.suppress_first_release_frame {
            self.suppress_first_release_frame = false;
            return commands;
        }

        if !input.media_running || input.crossed_beats == 0 {
            return commands;
        }
        self.accepted_media_beats = self
            .accepted_media_beats
            .saturating_add(u64::from(input.crossed_beats));

        match self.state {
            AutopilotState::Starting => {
                if self.expected_ready {
                    self.commit_expected(&mut commands);
                }
            }
            AutopilotState::Running => {
                let Some(due) = self.due_media_beat else {
                    return self.enter_fault(AutopilotFault::Invariant);
                };
                if self.accepted_media_beats < due {
                    return commands;
                }
                if self.expected.is_none() {
                    // Only Once on its final Scene has no lookahead target.
                    self.state = AutopilotState::Complete;
                    self.due_media_beat = None;
                } else if self.expected_ready {
                    self.commit_expected(&mut commands);
                } else {
                    self.state = AutopilotState::Stalled;
                }
            }
            AutopilotState::Stalled => {
                if self.expected_ready {
                    self.commit_expected(&mut commands);
                }
            }
            AutopilotState::Stopped
            | AutopilotState::Paused
            | AutopilotState::Faulted
            | AutopilotState::Complete => {}
        }
        commands
    }

    /// Enter the no-skip fault law after a host-side preparation, reference,
    /// or GPU commit failure. The host keeps its current visible output.
    pub fn fault(&mut self, fault: AutopilotFault) -> AutopilotCommands {
        if let AutopilotFault::Commit(target) = fault {
            if let Some((last_target, prior_cursor)) = self.last_commit {
                if last_target == target {
                    self.active_cursor = prior_cursor;
                    self.cursor = prior_cursor.unwrap_or(0);
                }
            }
        }
        self.enter_fault(fault)
    }

    fn accept_readiness(
        &mut self,
        readiness: AutopilotReadinessBeforeFrame,
    ) -> Option<AutopilotCommand> {
        match readiness {
            AutopilotReadinessBeforeFrame::Pending => None,
            AutopilotReadinessBeforeFrame::Ready(target) => {
                if self.expected == Some(target) {
                    self.expected_ready = true;
                }
                None
            }
            AutopilotReadinessBeforeFrame::Faulted(target) => {
                if self.expected == Some(target) {
                    self.enter_fault(AutopilotFault::Preparation(target))
                        .into_iter()
                        .next()
                } else {
                    None
                }
            }
        }
    }

    fn commit_expected(&mut self, commands: &mut AutopilotCommands) {
        let Some(target) = self.expected.take() else {
            return;
        };
        let Some(plan) = self.plan.as_ref() else {
            return;
        };
        let Some(step) = plan.steps.get(target.step_index).copied() else {
            return;
        };

        let prior_cursor = self.active_cursor;
        self.cursor = target.step_index;
        self.active_cursor = Some(target.step_index);
        self.last_commit = Some((target, prior_cursor));
        self.expected_ready = false;
        self.state = AutopilotState::Running;
        self.due_media_beat = Some(
            self.accepted_media_beats
                .saturating_add(u64::from(step.hold_beats.get())),
        );
        commands.push(AutopilotCommand::Commit(target));

        let next_index = if self.cursor + 1 < plan.len() {
            Some(self.cursor + 1)
        } else if plan.repeat == AutopilotRepeat::Loop {
            Some(0)
        } else {
            None
        };
        if let Some(next_index) = next_index {
            if let Some(command) = self.prepare_step(next_index) {
                commands.push(command);
            }
        }
    }

    fn prepare_step(&mut self, step_index: usize) -> Option<AutopilotCommand> {
        let scene_id = self.plan.as_ref()?.steps.get(step_index)?.scene_id;
        let target = AutopilotTarget {
            epoch: self.epoch,
            sequence: self.next_sequence,
            step_index,
            scene_id,
        };
        self.next_sequence = self.next_sequence.wrapping_add(1);
        self.expected = Some(target);
        self.expected_ready = false;
        Some(AutopilotCommand::Prepare(target))
    }

    fn enter_fault(&mut self, fault: AutopilotFault) -> AutopilotCommands {
        // Invalidate every unprocessed Prepare that may have followed a failed
        // Commit in the same command batch before asking the host to cancel
        // the old epoch's async ownership.
        let old_epoch = self.advance_epoch();
        self.state = AutopilotState::Faulted;
        self.resume_state = None;
        self.fault = Some(fault);
        // Faulted is a terminal hold of the last visible Scene, not a ghost
        // scheduling phase. A Commit command may already have installed its
        // lookahead before the host rejects the cut, so clear every countdown
        // and expected-target field while preserving the rolled-back cursor.
        self.due_media_beat = None;
        self.expected = None;
        self.expected_ready = false;
        self.last_commit = None;
        vec![AutopilotCommand::CancelOwned { epoch: old_epoch }]
    }

    fn advance_epoch(&mut self) -> u64 {
        let old = self.epoch;
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.epoch = 1;
        }
        old
    }

    fn clear_sequence_state(&mut self) {
        self.resume_state = None;
        self.next_sequence = 0;
        self.plan = None;
        self.cursor = 0;
        self.active_cursor = None;
        self.last_commit = None;
        self.accepted_media_beats = 0;
        self.due_media_beat = None;
        self.expected = None;
        self.expected_ready = false;
        self.suppress_first_release_frame = false;
        self.fault = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::performance::{AutopilotHoldBeats, AutopilotStep, Scene, SceneBindings};
    use crate::transport::TriggerMode;

    fn scenes(ids: &[u16]) -> Scenes {
        Scenes::try_from_vec(
            ids.iter()
                .map(|id| Scene {
                    id: SceneId::new(*id).unwrap(),
                    name: format!("Scene {id}"),
                    trigger_mode: TriggerMode::Immediate,
                    bindings: SceneBindings::default(),
                })
                .collect(),
        )
        .unwrap()
    }

    fn plan(repeat: AutopilotRepeat, steps: &[(u16, u16)]) -> AutopilotPlan {
        AutopilotPlan::try_new(
            repeat,
            steps
                .iter()
                .map(|(scene_id, beats)| AutopilotStep {
                    scene_id: SceneId::new(*scene_id).unwrap(),
                    hold_beats: AutopilotHoldBeats::new(*beats).unwrap(),
                })
                .collect(),
        )
        .unwrap()
    }

    fn prepared(commands: &[AutopilotCommand]) -> AutopilotTarget {
        commands
            .iter()
            .find_map(|command| match command {
                AutopilotCommand::Prepare(target) => Some(*target),
                _ => None,
            })
            .expect("prepare command")
    }

    fn committed(commands: &[AutopilotCommand]) -> Option<AutopilotTarget> {
        commands.iter().find_map(|command| match command {
            AutopilotCommand::Commit(target) => Some(*target),
            _ => None,
        })
    }

    #[test]
    fn stopped_frame_command_path_retains_zero_vector_capacity() {
        let mut scheduler = AutopilotScheduler::default();
        let commands = scheduler.advance_frame(AutopilotFrameInput::running(
            0,
            AutopilotReadinessBeforeFrame::Pending,
        ));
        assert!(commands.is_empty());
        assert_eq!(
            commands.capacity(),
            0,
            "the common stopped frame must not allocate"
        );
    }

    #[test]
    fn play_validates_then_prepares_without_changing_output() {
        let mut scheduler = AutopilotScheduler::default();
        let all_scenes = scenes(&[1, 2]);

        let failure = scheduler
            .play(&AutopilotPlan::default(), &all_scenes)
            .unwrap_err();
        assert_eq!(failure.error, AutopilotPlayError::EmptyPlan);
        assert!(failure.commands.is_empty());
        assert_eq!(scheduler.state(), AutopilotState::Stopped);

        let missing = plan(AutopilotRepeat::Loop, &[(9, 4)]);
        let failure = scheduler.play(&missing, &all_scenes).unwrap_err();
        assert!(matches!(
            failure.error,
            AutopilotPlayError::MissingScene(issue) if issue.scene_id.get() == 9
        ));

        let commands = scheduler
            .play(&plan(AutopilotRepeat::Loop, &[(1, 4), (2, 4)]), &all_scenes)
            .unwrap();
        assert_eq!(scheduler.state(), AutopilotState::Starting);
        assert!(committed(&commands).is_none());
        assert_eq!(prepared(&commands).scene_id.get(), 1);
        assert_eq!(scheduler.active_scene_id(), None);
    }

    #[test]
    fn first_release_is_on_a_later_future_beat_even_when_cached() {
        let mut scheduler = AutopilotScheduler::default();
        let commands = scheduler
            .play(&plan(AutopilotRepeat::Once, &[(1, 2)]), &scenes(&[1]))
            .unwrap();
        let first = prepared(&commands);

        let suppressed = scheduler.advance_frame(AutopilotFrameInput::running(
            1,
            AutopilotReadinessBeforeFrame::Ready(first),
        ));
        assert!(committed(&suppressed).is_none());
        assert_eq!(scheduler.accepted_media_beats(), 0);
        assert!(
            committed(&scheduler.advance_frame(AutopilotFrameInput::running(
                0,
                AutopilotReadinessBeforeFrame::Pending,
            )))
            .is_none()
        );

        let release = scheduler.advance_frame(AutopilotFrameInput::running(
            1,
            AutopilotReadinessBeforeFrame::Pending,
        ));
        assert_eq!(committed(&release), Some(first));
        assert_eq!(scheduler.due_media_beat(), Some(3));
    }

    #[test]
    fn exact_dwell_prearms_one_lookahead_and_once_completes_on_final_hold() {
        let mut scheduler = AutopilotScheduler::default();
        let commands = scheduler
            .play(
                &plan(AutopilotRepeat::Once, &[(1, 2), (2, 3)]),
                &scenes(&[1, 2]),
            )
            .unwrap();
        let first = prepared(&commands);
        scheduler.advance_frame(AutopilotFrameInput::running(
            0,
            AutopilotReadinessBeforeFrame::Ready(first),
        ));
        let first_cut = scheduler.advance_frame(AutopilotFrameInput::running(
            1,
            AutopilotReadinessBeforeFrame::Pending,
        ));
        let second = prepared(&first_cut);
        assert_eq!(committed(&first_cut), Some(first));
        assert_eq!(scheduler.remaining_hold_beats(), Some(2));

        assert!(
            committed(&scheduler.advance_frame(AutopilotFrameInput::running(
                1,
                AutopilotReadinessBeforeFrame::Ready(second),
            )))
            .is_none()
        );
        let second_cut = scheduler.advance_frame(AutopilotFrameInput::running(
            1,
            AutopilotReadinessBeforeFrame::Pending,
        ));
        assert_eq!(committed(&second_cut), Some(second));
        assert_eq!(
            second_cut.len(),
            1,
            "Once does not prearm after its last cut"
        );
        assert_eq!(scheduler.remaining_hold_beats(), Some(3));

        scheduler.advance_frame(AutopilotFrameInput::running(
            2,
            AutopilotReadinessBeforeFrame::Pending,
        ));
        assert_eq!(scheduler.state(), AutopilotState::Running);
        let completion = scheduler.advance_frame(AutopilotFrameInput::running(
            1,
            AutopilotReadinessBeforeFrame::Pending,
        ));
        assert!(completion.is_empty());
        assert_eq!(scheduler.state(), AutopilotState::Complete);
        assert_eq!(scheduler.active_scene_id().unwrap().get(), 2);
    }

    #[test]
    fn missed_due_beat_stalls_and_ready_waits_for_another_future_beat() {
        let mut scheduler = AutopilotScheduler::default();
        let commands = scheduler
            .play(
                &plan(AutopilotRepeat::Loop, &[(1, 1), (2, 4)]),
                &scenes(&[1, 2]),
            )
            .unwrap();
        let first = prepared(&commands);
        scheduler.advance_frame(AutopilotFrameInput::running(
            0,
            AutopilotReadinessBeforeFrame::Ready(first),
        ));
        let cut = scheduler.advance_frame(AutopilotFrameInput::running(
            1,
            AutopilotReadinessBeforeFrame::Pending,
        ));
        let second = prepared(&cut);

        let due = scheduler.advance_frame(AutopilotFrameInput::running(
            1,
            AutopilotReadinessBeforeFrame::Pending,
        ));
        assert!(committed(&due).is_none());
        assert_eq!(scheduler.state(), AutopilotState::Stalled);

        let readiness_only = scheduler.advance_frame(AutopilotFrameInput::running(
            0,
            AutopilotReadinessBeforeFrame::Ready(second),
        ));
        assert!(committed(&readiness_only).is_none());
        assert_eq!(scheduler.state(), AutopilotState::Stalled);

        let later = scheduler.advance_frame(AutopilotFrameInput::running(
            1,
            AutopilotReadinessBeforeFrame::Pending,
        ));
        assert_eq!(committed(&later), Some(second));
        assert_eq!(scheduler.state(), AutopilotState::Running);
    }

    #[test]
    fn multi_crossings_advance_all_beats_but_never_catch_up_more_than_one_cut() {
        let mut scheduler = AutopilotScheduler::default();
        let commands = scheduler
            .play(
                &plan(AutopilotRepeat::Loop, &[(1, 2), (2, 3), (3, 4)]),
                &scenes(&[1, 2, 3]),
            )
            .unwrap();
        let first = prepared(&commands);
        scheduler.advance_frame(AutopilotFrameInput::running(
            0,
            AutopilotReadinessBeforeFrame::Ready(first),
        ));
        let cut = scheduler.advance_frame(AutopilotFrameInput::running(
            1,
            AutopilotReadinessBeforeFrame::Pending,
        ));
        let second = prepared(&cut);

        let leap = scheduler.advance_frame(AutopilotFrameInput::running(
            10,
            AutopilotReadinessBeforeFrame::Ready(second),
        ));
        assert_eq!(
            leap.iter()
                .filter(|command| matches!(command, AutopilotCommand::Commit(_)))
                .count(),
            1
        );
        assert_eq!(scheduler.cursor(), 1);
        assert_eq!(scheduler.accepted_media_beats(), 11);
        assert_eq!(scheduler.due_media_beat(), Some(14));
    }

    #[test]
    fn freeze_and_pause_keep_remaining_dwell_and_allow_preparation() {
        let mut scheduler = AutopilotScheduler::default();
        let commands = scheduler
            .play(
                &plan(AutopilotRepeat::Loop, &[(1, 2), (2, 2)]),
                &scenes(&[1, 2]),
            )
            .unwrap();
        let first = prepared(&commands);
        scheduler.advance_frame(AutopilotFrameInput::running(
            0,
            AutopilotReadinessBeforeFrame::Ready(first),
        ));
        let cut = scheduler.advance_frame(AutopilotFrameInput::running(
            1,
            AutopilotReadinessBeforeFrame::Pending,
        ));
        let second = prepared(&cut);
        assert_eq!(scheduler.remaining_hold_beats(), Some(2));

        let frozen = scheduler.advance_frame(AutopilotFrameInput {
            crossed_beats: 99,
            media_running: false,
            readiness_before_frame: AutopilotReadinessBeforeFrame::Ready(second),
        });
        assert!(frozen.is_empty());
        assert_eq!(scheduler.remaining_hold_beats(), Some(2));

        assert!(scheduler.pause());
        scheduler.advance_frame(AutopilotFrameInput::running(
            99,
            AutopilotReadinessBeforeFrame::Pending,
        ));
        assert_eq!(scheduler.remaining_hold_beats(), Some(2));
        assert!(scheduler.resume());
        scheduler.advance_frame(AutopilotFrameInput::running(
            1,
            AutopilotReadinessBeforeFrame::Pending,
        ));
        assert_eq!(scheduler.remaining_hold_beats(), Some(1));
        let next = scheduler.advance_frame(AutopilotFrameInput::running(
            1,
            AutopilotReadinessBeforeFrame::Pending,
        ));
        assert_eq!(committed(&next), Some(second));
    }

    #[test]
    fn one_step_loop_retriggers_and_tokens_never_alias_occurrences() {
        let mut scheduler = AutopilotScheduler::default();
        let commands = scheduler
            .play(&plan(AutopilotRepeat::Loop, &[(7, 1)]), &scenes(&[7]))
            .unwrap();
        let first = prepared(&commands);
        scheduler.advance_frame(AutopilotFrameInput::running(
            0,
            AutopilotReadinessBeforeFrame::Ready(first),
        ));
        let first_cut = scheduler.advance_frame(AutopilotFrameInput::running(
            1,
            AutopilotReadinessBeforeFrame::Pending,
        ));
        let second = prepared(&first_cut);
        assert_ne!(first, second);
        assert_eq!(first.scene_id, second.scene_id);

        let retrigger = scheduler.advance_frame(AutopilotFrameInput::running(
            1,
            AutopilotReadinessBeforeFrame::Ready(second),
        ));
        assert_eq!(committed(&retrigger), Some(second));
    }

    #[test]
    fn matching_failure_faults_without_skip_and_reset_invalidates_epoch() {
        let mut scheduler = AutopilotScheduler::default();
        let commands = scheduler
            .play(&plan(AutopilotRepeat::Loop, &[(1, 2)]), &scenes(&[1]))
            .unwrap();
        let first = prepared(&commands);
        let fault_commands = scheduler.advance_frame(AutopilotFrameInput::reanchor(
            AutopilotReadinessBeforeFrame::Faulted(first),
        ));
        assert_eq!(scheduler.state(), AutopilotState::Faulted);
        assert_eq!(scheduler.active_scene_id(), None);
        assert_eq!(
            fault_commands,
            vec![AutopilotCommand::CancelOwned { epoch: first.epoch }]
        );

        let fault_epoch = scheduler.epoch();
        let reset = scheduler.reset();
        assert_eq!(scheduler.state(), AutopilotState::Stopped);
        assert_ne!(scheduler.epoch(), fault_epoch);
        assert_eq!(
            reset,
            vec![AutopilotCommand::CancelOwned { epoch: fault_epoch }]
        );
        assert_eq!(scheduler.cursor(), 0);
    }

    #[test]
    fn rejected_gpu_commit_restores_the_last_visible_scene_before_faulting() {
        let mut scheduler = AutopilotScheduler::default();
        let commands = scheduler
            .play(
                &plan(AutopilotRepeat::Loop, &[(1, 1), (2, 1)]),
                &scenes(&[1, 2]),
            )
            .unwrap();
        let first = prepared(&commands);
        scheduler.advance_frame(AutopilotFrameInput::running(
            0,
            AutopilotReadinessBeforeFrame::Ready(first),
        ));
        let first_cut = scheduler.advance_frame(AutopilotFrameInput::running(
            1,
            AutopilotReadinessBeforeFrame::Pending,
        ));
        let second = prepared(&first_cut);
        assert_eq!(scheduler.active_scene_id().unwrap().get(), 1);

        let second_cut = scheduler.advance_frame(AutopilotFrameInput::running(
            1,
            AutopilotReadinessBeforeFrame::Ready(second),
        ));
        assert_eq!(committed(&second_cut), Some(second));
        assert_eq!(scheduler.active_scene_id().unwrap().get(), 2);

        scheduler.fault(AutopilotFault::Commit(second));
        assert_eq!(scheduler.state(), AutopilotState::Faulted);
        assert_eq!(scheduler.active_scene_id().unwrap().get(), 1);
        assert_eq!(scheduler.cursor(), 0);
        assert_eq!(scheduler.expected_target(), None);
        assert_eq!(scheduler.remaining_hold_beats(), None);
        assert_eq!(scheduler.due_media_beat(), None);
    }
}
