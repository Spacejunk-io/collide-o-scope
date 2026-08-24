//! Pure latest-only publication contract for fallible asynchronous workers.
//!
//! Decoder intent, proxy adoption, temporal/Mosh/VHS work, recorder reports,
//! and retired-device callbacks all need the same rule: work may finish after
//! it was superseded or cancelled, but stale work may never become visible.
//! This state machine owns no thread or resource, which lets Loom explore the
//! scheduling boundary without constructing GPU or FFmpeg objects.

#![allow(
    dead_code,
    reason = "P8 freezes the shared contract before each worker migrates to it"
)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PublicationToken {
    epoch: u64,
    generation: u64,
}

impl PublicationToken {
    pub const fn generation(self) -> u64 {
        self.generation
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LatestOnlyPublicationGate {
    epoch: u64,
    next_generation: u64,
    requested: Option<PublicationToken>,
    published_generation: Option<u64>,
}

impl LatestOnlyPublicationGate {
    pub fn request(&mut self) -> PublicationToken {
        self.next_generation = self.next_generation.wrapping_add(1).max(1);
        let token = PublicationToken {
            epoch: self.epoch,
            generation: self.next_generation,
        };
        self.requested = Some(token);
        token
    }

    /// A worker claims only the newest outstanding request. Repeated claims
    /// are harmless; publication is still guarded by the exact token.
    pub const fn claim_latest(&self) -> Option<PublicationToken> {
        self.requested
    }

    /// Invalidate every token already handed to a worker.
    pub fn cancel_all(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        self.requested = None;
    }

    /// Publish iff no newer request or cancellation barrier has intervened.
    pub fn try_publish(&mut self, token: PublicationToken) -> bool {
        if self.requested != Some(token) || token.epoch != self.epoch {
            return false;
        }
        if self
            .published_generation
            .is_some_and(|published| published >= token.generation)
        {
            return false;
        }
        self.published_generation = Some(token.generation);
        self.requested = None;
        true
    }

    pub const fn published_generation(&self) -> Option<u64> {
        self.published_generation
    }
}
