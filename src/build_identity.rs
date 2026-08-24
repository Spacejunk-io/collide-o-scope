//! One compile-time identity shared by every operator and provenance surface.
//!
//! The build script generates only immutable, path-free facts. Runtime code
//! may observe a different external FFmpeg tool, but it must never rewrite the
//! identity of the executable that is already running.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const BUILD_IDENTITY_DOMAIN: &str = "collide-o-scope build identity v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuildIdentity {
    pub schema_version: u16,
    pub package_name: &'static str,
    pub version: &'static str,
    pub git_sha: &'static str,
    pub git_dirty: bool,
    pub profile: &'static str,
    pub target: &'static str,
    pub enabled_features: &'static str,
    pub rustc_vv: &'static str,
    pub cargo_version: &'static str,
    pub linker_identity: &'static str,
    pub sdk_identity: &'static str,
    pub ffmpeg_libraries: &'static str,
    pub ffmpeg_binary_version: &'static str,
    pub ffmpeg_binary_sha256: &'static str,
    pub ffprobe_binary_version: &'static str,
    pub ffprobe_binary_sha256: &'static str,
    pub shader_bundle_sha256: &'static str,
    pub cargo_lock_sha256: &'static str,
    pub identity_sha256: &'static str,
    pub published_artifact: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildIdentitySnapshot {
    pub schema_version: u16,
    pub package_name: String,
    pub version: String,
    pub git_sha: String,
    pub git_dirty: bool,
    pub profile: String,
    pub target: String,
    pub enabled_features: String,
    pub rustc_vv: String,
    pub cargo_version: String,
    pub linker_identity: String,
    pub sdk_identity: String,
    pub ffmpeg_libraries: String,
    pub ffmpeg_binary_version: String,
    pub ffmpeg_binary_sha256: String,
    pub ffprobe_binary_version: String,
    pub ffprobe_binary_sha256: String,
    pub shader_bundle_sha256: String,
    pub cargo_lock_sha256: String,
    pub identity_sha256: String,
    pub published_artifact: bool,
}

include!(concat!(env!("OUT_DIR"), "/build_identity.rs"));

#[cfg(test)]
#[path = "../build_identity_policy.rs"]
mod build_identity_policy;

pub const fn current() -> &'static BuildIdentity {
    &GENERATED_BUILD_IDENTITY
}

impl BuildIdentity {
    pub fn snapshot(self) -> BuildIdentitySnapshot {
        BuildIdentitySnapshot {
            schema_version: self.schema_version,
            package_name: self.package_name.to_owned(),
            version: self.version.to_owned(),
            git_sha: self.git_sha.to_owned(),
            git_dirty: self.git_dirty,
            profile: self.profile.to_owned(),
            target: self.target.to_owned(),
            enabled_features: self.enabled_features.to_owned(),
            rustc_vv: self.rustc_vv.to_owned(),
            cargo_version: self.cargo_version.to_owned(),
            linker_identity: self.linker_identity.to_owned(),
            sdk_identity: self.sdk_identity.to_owned(),
            ffmpeg_libraries: self.ffmpeg_libraries.to_owned(),
            ffmpeg_binary_version: self.ffmpeg_binary_version.to_owned(),
            ffmpeg_binary_sha256: self.ffmpeg_binary_sha256.to_owned(),
            ffprobe_binary_version: self.ffprobe_binary_version.to_owned(),
            ffprobe_binary_sha256: self.ffprobe_binary_sha256.to_owned(),
            shader_bundle_sha256: self.shader_bundle_sha256.to_owned(),
            cargo_lock_sha256: self.cargo_lock_sha256.to_owned(),
            identity_sha256: self.identity_sha256.to_owned(),
            published_artifact: self.published_artifact,
        }
    }

    pub fn canonical_payload(self) -> String {
        let mut payload = String::new();
        for (key, value) in [
            ("domain", BUILD_IDENTITY_DOMAIN.to_owned()),
            ("package_name", self.package_name.to_owned()),
            ("version", self.version.to_owned()),
            ("git_sha", self.git_sha.to_owned()),
            ("git_dirty", self.git_dirty.to_string()),
            ("profile", self.profile.to_owned()),
            ("target", self.target.to_owned()),
            ("enabled_features", self.enabled_features.to_owned()),
            ("rustc_vv", self.rustc_vv.to_owned()),
            ("cargo_version", self.cargo_version.to_owned()),
            ("linker_identity", self.linker_identity.to_owned()),
            ("sdk_identity", self.sdk_identity.to_owned()),
            ("ffmpeg_libraries", self.ffmpeg_libraries.to_owned()),
            (
                "ffmpeg_binary_version",
                self.ffmpeg_binary_version.to_owned(),
            ),
            ("ffmpeg_binary_sha256", self.ffmpeg_binary_sha256.to_owned()),
            (
                "ffprobe_binary_version",
                self.ffprobe_binary_version.to_owned(),
            ),
            (
                "ffprobe_binary_sha256",
                self.ffprobe_binary_sha256.to_owned(),
            ),
            ("shader_bundle_sha256", self.shader_bundle_sha256.to_owned()),
            ("cargo_lock_sha256", self.cargo_lock_sha256.to_owned()),
            ("published_artifact", self.published_artifact.to_string()),
        ] {
            let _ = writeln!(payload, "{key}={value}");
        }
        payload
    }

    pub fn digest_is_valid(self) -> bool {
        format!("{:x}", Sha256::digest(self.canonical_payload().as_bytes())) == self.identity_sha256
    }

    pub fn human_version(self) -> String {
        let dirty = if self.git_dirty { " dirty" } else { "" };
        let badge = if self.published_artifact {
            " published"
        } else {
            " local"
        };
        format!(
            "{} {} ({}{}; {}; features {}; shaders {}; identity {};{})",
            self.package_name,
            self.version,
            self.git_sha,
            dirty,
            self.target,
            self.enabled_features,
            self.shader_bundle_sha256,
            self.identity_sha256,
            badge
        )
    }

    pub fn pretty_json(self) -> String {
        serde_json::to_string_pretty(&self.snapshot())
            .expect("the static build identity is JSON serializable")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_identity_is_complete_and_self_authenticating() {
        let identity = *current();
        assert_eq!(identity.schema_version, 1);
        assert_eq!(identity.package_name, "collide-o-scope");
        assert_eq!(identity.identity_sha256.len(), 64);
        assert_eq!(identity.shader_bundle_sha256.len(), 64);
        assert_eq!(identity.cargo_lock_sha256.len(), 64);
        assert!(identity.digest_is_valid());
    }

    #[test]
    fn a_dirty_build_can_never_present_the_published_badge() {
        let identity = *current();
        assert!(!identity.git_dirty || !identity.published_artifact);
    }

    #[test]
    fn owned_snapshot_round_trips_without_losing_any_field() {
        let snapshot = current().snapshot();
        let json = serde_json::to_vec(&snapshot).unwrap();
        let restored: BuildIdentitySnapshot = serde_json::from_slice(&json).unwrap();
        assert_eq!(restored, snapshot);
    }
}
