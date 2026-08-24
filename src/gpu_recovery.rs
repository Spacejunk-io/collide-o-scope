//! P7 GPU-loss fault model and bounded Phase-A supervised-restart contract.
//!
//! The live application currently uses the Phase-A path only.  The richer
//! state machine is executable so a later in-process rebuild cannot invent
//! looser epoch, audience, retry, or operator-resume semantics.

#![allow(
    dead_code,
    reason = "Phase B is an executable fault-model contract but remains disabled until its recovery gate is proven"
)]

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use serde::{Deserialize, Serialize};

pub const SUPERVISED_GPU_RESTART_EXIT_CODE: i32 = 75;
pub const MAX_IN_PROCESS_REBUILD_ATTEMPTS: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum GpuLossPoint {
    Unattributed = 1,
    PreSubmit = 2,
    PostSubmit = 3,
    ReadbackPending = 4,
    ProxyAdoption = 5,
    MoshVhsHandoff = 6,
    RecorderActive = 7,
    OutputResize = 8,
    StageMapCreation = 9,
}

impl GpuLossPoint {
    const fn from_u8(value: u8) -> Self {
        match value {
            2 => Self::PreSubmit,
            3 => Self::PostSubmit,
            4 => Self::ReadbackPending,
            5 => Self::ProxyAdoption,
            6 => Self::MoshVhsHandoff,
            7 => Self::RecorderActive,
            8 => Self::OutputResize,
            9 => Self::StageMapCreation,
            _ => Self::Unattributed,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GpuEpoch(u64);

impl GpuEpoch {
    pub const INITIAL: Self = Self(1);

    pub const fn get(self) -> u64 {
        self.0
    }

    fn next(self) -> Option<Self> {
        self.0.checked_add(1).map(Self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuRecoveryState {
    Healthy,
    ClosingAudience,
    RetiringGpuEpoch,
    Rebuilding,
    AwaitingOperator,
    SupervisedRestartRequired,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuRecoveryError {
    InvalidTransition,
    AudienceNotClosed,
    OldEpochNotRetired,
    RebuildAttemptCap,
    SecondLoss,
    OperatorResumeBeforeValidation,
    StaleCompletion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuRecoveryReceipt {
    pub schema_version: u16,
    pub state: GpuRecoveryState,
    pub retired_epoch: u64,
    pub candidate_epoch: u64,
    pub loss_point: GpuLossPoint,
    pub rebuild_attempts: u8,
    pub audience_closed: bool,
    pub required_claims_valid: bool,
    pub operator_resume_required: bool,
    pub supervised_exit_code: Option<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuRecoveryMachine {
    state: GpuRecoveryState,
    retired_epoch: GpuEpoch,
    candidate_epoch: GpuEpoch,
    loss_point: GpuLossPoint,
    rebuild_attempts: u8,
    audience_closed: bool,
    old_epoch_retired: bool,
    required_claims_valid: bool,
}

impl Default for GpuRecoveryMachine {
    fn default() -> Self {
        Self {
            state: GpuRecoveryState::Healthy,
            retired_epoch: GpuEpoch::INITIAL,
            candidate_epoch: GpuEpoch::INITIAL,
            loss_point: GpuLossPoint::Unattributed,
            rebuild_attempts: 0,
            audience_closed: false,
            old_epoch_retired: false,
            required_claims_valid: false,
        }
    }
}

impl GpuRecoveryMachine {
    pub const fn state(self) -> GpuRecoveryState {
        self.state
    }

    pub const fn live_epoch(self) -> GpuEpoch {
        self.candidate_epoch
    }

    pub fn begin_loss(&mut self, point: GpuLossPoint) -> Result<(), GpuRecoveryError> {
        if self.state != GpuRecoveryState::Healthy {
            self.state = GpuRecoveryState::Failed;
            return Err(GpuRecoveryError::SecondLoss);
        }
        self.loss_point = point;
        self.retired_epoch = self.candidate_epoch;
        self.audience_closed = false;
        self.old_epoch_retired = false;
        self.required_claims_valid = false;
        self.state = GpuRecoveryState::ClosingAudience;
        Ok(())
    }

    /// "Closed" means every reachable endpoint has been explicitly closed or
    /// marked failed. It never means that a lost device rendered a new black
    /// frame successfully.
    pub fn audience_closed(&mut self) -> Result<(), GpuRecoveryError> {
        if self.state != GpuRecoveryState::ClosingAudience {
            return Err(GpuRecoveryError::InvalidTransition);
        }
        self.audience_closed = true;
        self.state = GpuRecoveryState::RetiringGpuEpoch;
        Ok(())
    }

    pub fn gpu_epoch_retired(&mut self) -> Result<(), GpuRecoveryError> {
        if self.state != GpuRecoveryState::RetiringGpuEpoch || !self.audience_closed {
            return Err(GpuRecoveryError::AudienceNotClosed);
        }
        self.old_epoch_retired = true;
        Ok(())
    }

    /// Phase A intentionally stops here. A launcher may restart once, but the
    /// existing explicit journal-recovery surface remains operator-owned.
    pub fn require_supervised_restart(&mut self) -> Result<i32, GpuRecoveryError> {
        if self.state != GpuRecoveryState::RetiringGpuEpoch || !self.old_epoch_retired {
            return Err(GpuRecoveryError::OldEpochNotRetired);
        }
        self.state = GpuRecoveryState::SupervisedRestartRequired;
        Ok(SUPERVISED_GPU_RESTART_EXIT_CODE)
    }

    /// Phase-B research seam. It is not called by the live application.
    pub fn begin_in_process_rebuild(&mut self) -> Result<GpuEpoch, GpuRecoveryError> {
        if self.state != GpuRecoveryState::RetiringGpuEpoch || !self.old_epoch_retired {
            return Err(GpuRecoveryError::OldEpochNotRetired);
        }
        if self.rebuild_attempts >= MAX_IN_PROCESS_REBUILD_ATTEMPTS {
            self.state = GpuRecoveryState::Failed;
            return Err(GpuRecoveryError::RebuildAttemptCap);
        }
        self.candidate_epoch = self
            .retired_epoch
            .next()
            .ok_or(GpuRecoveryError::RebuildAttemptCap)?;
        self.rebuild_attempts += 1;
        self.state = GpuRecoveryState::Rebuilding;
        Ok(self.candidate_epoch)
    }

    pub fn rebuild_validated(
        &mut self,
        epoch: GpuEpoch,
        required_claims_valid: bool,
    ) -> Result<(), GpuRecoveryError> {
        if self.state != GpuRecoveryState::Rebuilding || epoch != self.candidate_epoch {
            return Err(GpuRecoveryError::StaleCompletion);
        }
        self.required_claims_valid = required_claims_valid;
        self.state = if required_claims_valid {
            GpuRecoveryState::AwaitingOperator
        } else {
            GpuRecoveryState::Failed
        };
        Ok(())
    }

    pub fn operator_resume(&mut self) -> Result<(), GpuRecoveryError> {
        if self.state != GpuRecoveryState::AwaitingOperator || !self.required_claims_valid {
            return Err(GpuRecoveryError::OperatorResumeBeforeValidation);
        }
        self.state = GpuRecoveryState::Healthy;
        Ok(())
    }

    pub const fn accepts_completion(self, epoch: GpuEpoch) -> bool {
        matches!(
            self.state,
            GpuRecoveryState::Healthy | GpuRecoveryState::Rebuilding
        ) && epoch.0 == self.candidate_epoch.0
    }

    pub const fn receipt(self) -> GpuRecoveryReceipt {
        GpuRecoveryReceipt {
            schema_version: 1,
            state: self.state,
            retired_epoch: self.retired_epoch.get(),
            candidate_epoch: self.candidate_epoch.get(),
            loss_point: self.loss_point,
            rebuild_attempts: self.rebuild_attempts,
            audience_closed: self.audience_closed,
            required_claims_valid: self.required_claims_valid,
            operator_resume_required: matches!(self.state, GpuRecoveryState::AwaitingOperator),
            supervised_exit_code: if matches!(
                self.state,
                GpuRecoveryState::SupervisedRestartRequired
            ) {
                Some(SUPERVISED_GPU_RESTART_EXIT_CODE)
            } else {
                None
            },
        }
    }
}

static PHASE_A_RESTART_REQUESTED: AtomicBool = AtomicBool::new(false);
static PHASE_A_LOSS_POINT: AtomicU8 = AtomicU8::new(GpuLossPoint::Unattributed as u8);

/// Idempotent process-wide latch used by all existing device-loss readers.
/// It performs no allocation and preserves the first attribution.
pub fn request_phase_a_restart(point: GpuLossPoint) {
    if PHASE_A_RESTART_REQUESTED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        PHASE_A_LOSS_POINT.store(point as u8, Ordering::Release);
    }
}

pub fn phase_a_restart_request() -> Option<GpuLossPoint> {
    PHASE_A_RESTART_REQUESTED
        .load(Ordering::Acquire)
        .then(|| GpuLossPoint::from_u8(PHASE_A_LOSS_POINT.load(Ordering::Acquire)))
}

#[cfg(test)]
fn reset_phase_a_latch_for_test() {
    PHASE_A_LOSS_POINT.store(GpuLossPoint::Unattributed as u8, Ordering::Release);
    PHASE_A_RESTART_REQUESTED.store(false, Ordering::Release);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_injected_loss_point_reaches_the_same_bounded_phase_a_exit() {
        for point in [
            GpuLossPoint::PreSubmit,
            GpuLossPoint::PostSubmit,
            GpuLossPoint::ReadbackPending,
            GpuLossPoint::ProxyAdoption,
            GpuLossPoint::MoshVhsHandoff,
            GpuLossPoint::RecorderActive,
            GpuLossPoint::OutputResize,
            GpuLossPoint::StageMapCreation,
        ] {
            let mut machine = GpuRecoveryMachine::default();
            machine.begin_loss(point).unwrap();
            assert!(!machine.accepts_completion(GpuEpoch::INITIAL));
            machine.audience_closed().unwrap();
            machine.gpu_epoch_retired().unwrap();
            assert_eq!(
                machine.require_supervised_restart().unwrap(),
                SUPERVISED_GPU_RESTART_EXIT_CODE
            );
            let receipt = machine.receipt();
            assert_eq!(receipt.loss_point, point);
            assert_eq!(receipt.supervised_exit_code, Some(75));
            assert!(receipt.audience_closed);
            assert!(!receipt.operator_resume_required);
        }
    }

    #[test]
    fn phase_b_refuses_stale_work_and_requires_explicit_operator_resume() {
        let mut machine = GpuRecoveryMachine::default();
        let old = machine.live_epoch();
        machine.begin_loss(GpuLossPoint::PostSubmit).unwrap();
        machine.audience_closed().unwrap();
        machine.gpu_epoch_retired().unwrap();
        let new = machine.begin_in_process_rebuild().unwrap();
        assert_ne!(new, old);
        assert!(!machine.accepts_completion(old));
        assert!(machine.accepts_completion(new));
        assert_eq!(
            machine.rebuild_validated(old, true),
            Err(GpuRecoveryError::StaleCompletion)
        );
        machine.rebuild_validated(new, true).unwrap();
        assert_eq!(machine.state(), GpuRecoveryState::AwaitingOperator);
        machine.operator_resume().unwrap();
        assert_eq!(machine.state(), GpuRecoveryState::Healthy);
    }

    #[test]
    fn second_loss_during_rebuild_fails_instead_of_recursing() {
        let mut machine = GpuRecoveryMachine::default();
        machine.begin_loss(GpuLossPoint::PreSubmit).unwrap();
        machine.audience_closed().unwrap();
        machine.gpu_epoch_retired().unwrap();
        machine.begin_in_process_rebuild().unwrap();
        assert_eq!(
            machine.begin_loss(GpuLossPoint::ReadbackPending),
            Err(GpuRecoveryError::SecondLoss)
        );
        assert_eq!(machine.state(), GpuRecoveryState::Failed);
    }

    #[test]
    fn process_latch_is_idempotent_and_keeps_first_attribution() {
        reset_phase_a_latch_for_test();
        request_phase_a_restart(GpuLossPoint::RecorderActive);
        request_phase_a_restart(GpuLossPoint::OutputResize);
        assert_eq!(
            phase_a_restart_request(),
            Some(GpuLossPoint::RecorderActive)
        );
        reset_phase_a_latch_for_test();
    }
}
