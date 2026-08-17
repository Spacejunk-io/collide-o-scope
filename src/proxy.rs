//! Pure, content-addressed playback assessment and proxy-cache planning.
//!
//! This module does not encode media, select a proprietary codec, inspect a
//! path, or mutate a cache. It turns an already verified content identity,
//! fixed proxy settings, measured playback facts, and a bounded cache index
//! into deterministic decisions that a later worker may execute atomically.

use std::collections::BTreeSet;
use std::fmt;

use serde::{de, Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};

use crate::media_source::ContentIdentity;

pub const PROXY_SCHEMA_VERSION: u16 = 1;
pub const PROXY_ALGORITHM_VERSION: u16 = 1;
pub const PROXY_CACHE_KEY_DOMAIN: &[u8] = b"collide-o-scope.proxy-cache-key.v1\0";
pub const PROXY_MAX_FIXED_FPS_NUMERATOR: u32 = 240_000;
pub const PROXY_MAX_FIXED_FPS_DENOMINATOR: u32 = 10_000;
pub const PROXY_MAX_OBSERVATION_FRAMES: u32 = 10_000_000;
pub const PROXY_MAX_VISIBLE_LAYERS: u16 = 1_024;
#[allow(
    dead_code,
    reason = "reserved for the explicitly deferred cache worker"
)]
pub const PROXY_CACHE_HARD_MAX_ENTRIES: usize = 4_096;
#[allow(
    dead_code,
    reason = "reserved for the explicitly deferred cache worker"
)]
pub const PROXY_CACHE_HARD_MAX_ENTRY_BYTES: u64 = 64 * 1024 * 1024 * 1024;
#[allow(
    dead_code,
    reason = "reserved for the explicitly deferred cache worker"
)]
pub const PROXY_CACHE_HARD_MAX_TOTAL_BYTES: u64 = 512 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProxyFormat {
    /// Open, lossless FFV1 video in a Matroska container. This is a cache-key
    /// vocabulary only; its presence does not claim an encoder is installed.
    Ffv1Matroska,
}

impl ProxyFormat {
    const fn code(self) -> u8 {
        match self {
            Self::Ffv1Matroska => 1,
        }
    }

    #[allow(
        dead_code,
        reason = "artifact naming belongs to the deferred cache worker"
    )]
    pub const fn file_extension(self) -> &'static str {
        match self {
            Self::Ffv1Matroska => "mkv",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProxyScale {
    Original,
    Half,
    Quarter,
}

impl ProxyScale {
    const fn code(self) -> u8 {
        match self {
            Self::Original => 1,
            Self::Half => 2,
            Self::Quarter => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ProxyFrameRate {
    Source,
    Fixed { numerator: u32, denominator: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ProxySettings {
    pub schema_version: u16,
    pub algorithm_version: u16,
    pub format: ProxyFormat,
    pub scale: ProxyScale,
    pub frame_rate: ProxyFrameRate,
    pub include_audio: bool,
}

impl Default for ProxySettings {
    fn default() -> Self {
        Self {
            schema_version: PROXY_SCHEMA_VERSION,
            algorithm_version: PROXY_ALGORITHM_VERSION,
            format: ProxyFormat::Ffv1Matroska,
            scale: ProxyScale::Half,
            frame_rate: ProxyFrameRate::Source,
            include_audio: true,
        }
    }
}

impl ProxySettings {
    pub fn validate(self) -> Result<Self, ProxyError> {
        if self.schema_version != PROXY_SCHEMA_VERSION {
            return Err(ProxyError::UnsupportedSchemaVersion(self.schema_version));
        }
        if self.algorithm_version != PROXY_ALGORITHM_VERSION {
            return Err(ProxyError::UnsupportedAlgorithmVersion(
                self.algorithm_version,
            ));
        }
        if let ProxyFrameRate::Fixed {
            numerator,
            denominator,
        } = self.frame_rate
        {
            if numerator == 0
                || numerator > PROXY_MAX_FIXED_FPS_NUMERATOR
                || denominator == 0
                || denominator > PROXY_MAX_FIXED_FPS_DENOMINATOR
                || u64::from(numerator) > 240 * u64::from(denominator)
            {
                return Err(ProxyError::InvalidFrameRate {
                    numerator,
                    denominator,
                });
            }
        }
        Ok(self)
    }

    fn update_cache_key(self, hasher: &mut Sha256) {
        hasher.update([self.format.code(), self.scale.code()]);
        match self.frame_rate {
            ProxyFrameRate::Source => hasher.update([0]),
            ProxyFrameRate::Fixed {
                numerator,
                denominator,
            } => {
                hasher.update([1]);
                hasher.update(numerator.to_le_bytes());
                hasher.update(denominator.to_le_bytes());
            }
        }
        hasher.update([u8::from(self.include_audio)]);
        hasher.update(self.schema_version.to_le_bytes());
        hasher.update(self.algorithm_version.to_le_bytes());
    }
}

impl<'de> Deserialize<'de> for ProxySettings {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Raw {
            schema_version: u16,
            algorithm_version: u16,
            format: ProxyFormat,
            scale: ProxyScale,
            frame_rate: ProxyFrameRate,
            include_audio: bool,
        }

        let raw = Raw::deserialize(deserializer)?;
        Self {
            schema_version: raw.schema_version,
            algorithm_version: raw.algorithm_version,
            format: raw.format,
            scale: raw.scale,
            frame_rate: raw.frame_rate,
            include_audio: raw.include_audio,
        }
        .validate()
        .map_err(de::Error::custom)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProxyCacheKey([u8; 32]);

impl fmt::Debug for ProxyCacheKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("ProxyCacheKey")
            .field(&self.to_hex())
            .finish()
    }
}

impl ProxyCacheKey {
    pub fn derive(source: &ContentIdentity, settings: ProxySettings) -> Result<Self, ProxyError> {
        let settings = settings.validate()?;
        validate_content_identity(source)?;
        let digest = decode_sha256(&source.sha256).ok_or(ProxyError::InvalidContentIdentity)?;
        let mut hasher = Sha256::new();
        hasher.update(PROXY_CACHE_KEY_DOMAIN);
        hasher.update(digest);
        hasher.update(source.byte_len.to_le_bytes());
        settings.update_cache_key(&mut hasher);
        Ok(Self(hasher.finalize().into()))
    }

    pub fn from_hex(value: &str) -> Result<Self, ProxyError> {
        decode_sha256(value)
            .map(Self)
            .ok_or(ProxyError::InvalidCacheKey)
    }

    pub fn to_hex(self) -> String {
        encode_hex(&self.0)
    }

    #[allow(
        dead_code,
        reason = "artifact naming belongs to the deferred cache worker"
    )]
    pub fn artifact_file_name(self, format: ProxyFormat) -> String {
        format!("{}.{}", self.to_hex(), format.file_extension())
    }
}

impl Serialize for ProxyCacheKey {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for ProxyCacheKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::from_hex(&value).map_err(de::Error::custom)
    }
}

fn validate_content_identity(source: &ContentIdentity) -> Result<(), ProxyError> {
    if source.byte_len == 0
        || source.sha256.len() != 64
        || !source
            .sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ProxyError::InvalidContentIdentity);
    }
    Ok(())
}

fn decode_sha256(value: &str) -> Option<[u8; 32]> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return None;
    }
    let mut result = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = hex_nibble(pair[0])?;
        let low = hex_nibble(pair[1])?;
        result[index] = (high << 4) | low;
    }
    Some(result)
}

const fn hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut value = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        value.push(char::from(HEX[usize::from(byte >> 4)]));
        value.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    value
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ProxyPlaybackObservation {
    pub sampled_frames: u32,
    pub visible_layers: u16,
    pub frame_budget_micros: u32,
    pub decode_p95_micros: u32,
    pub upload_p95_micros: u32,
    pub frame_age_p95_micros: u32,
    pub dropped_frames: u32,
    pub pending_frames_peak: u16,
    pub hardware_decode_active: bool,
    pub zero_copy_active: bool,
}

impl ProxyPlaybackObservation {
    pub fn validate(self) -> Result<Self, ProxyError> {
        if self.sampled_frames == 0 || self.sampled_frames > PROXY_MAX_OBSERVATION_FRAMES {
            return Err(ProxyError::InvalidObservation("sampled_frames"));
        }
        if self.visible_layers == 0 || self.visible_layers > PROXY_MAX_VISIBLE_LAYERS {
            return Err(ProxyError::InvalidObservation("visible_layers"));
        }
        if self.frame_budget_micros == 0
            || self.frame_budget_micros > 60_000_000
            || self.decode_p95_micros > 60_000_000
            || self.upload_p95_micros > 60_000_000
            || self.frame_age_p95_micros > 60_000_000
            || self.dropped_frames > self.sampled_frames
            || self.pending_frames_peak > 4_096
            || (self.zero_copy_active && !self.hardware_decode_active)
        {
            return Err(ProxyError::InvalidObservation("playback facts"));
        }
        Ok(self)
    }
}

impl<'de> Deserialize<'de> for ProxyPlaybackObservation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Raw {
            sampled_frames: u32,
            visible_layers: u16,
            frame_budget_micros: u32,
            decode_p95_micros: u32,
            upload_p95_micros: u32,
            frame_age_p95_micros: u32,
            dropped_frames: u32,
            pending_frames_peak: u16,
            hardware_decode_active: bool,
            zero_copy_active: bool,
        }

        let raw = Raw::deserialize(deserializer)?;
        Self {
            sampled_frames: raw.sampled_frames,
            visible_layers: raw.visible_layers,
            frame_budget_micros: raw.frame_budget_micros,
            decode_p95_micros: raw.decode_p95_micros,
            upload_p95_micros: raw.upload_p95_micros,
            frame_age_p95_micros: raw.frame_age_p95_micros,
            dropped_frames: raw.dropped_frames,
            pending_frames_peak: raw.pending_frames_peak,
            hardware_decode_active: raw.hardware_decode_active,
            zero_copy_active: raw.zero_copy_active,
        }
        .validate()
        .map_err(de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProxyRecommendationReason {
    DecodeExceedsFrameBudget,
    UploadExceedsFrameBudget,
    FrameAgeExceedsFrameBudget,
    DroppedFramesObserved,
    DecoderQueuePressure,
    MultiLayerPressure,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProxyAssessment {
    MeasurementRequired,
    OriginalSufficient,
    ProxyRecommended(Box<[ProxyRecommendationReason]>),
}

/// Binds measured playback facts to the same source-bytes/settings identity
/// that names a future cache artifact. No host path can enter this record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentAddressedProxyAssessment {
    pub cache_key: ProxyCacheKey,
    pub source_byte_len: u64,
    pub settings: ProxySettings,
    pub assessment: ProxyAssessment,
}

pub fn assess_content_addressed_proxy(
    source: &ContentIdentity,
    settings: ProxySettings,
    observation: ProxyPlaybackObservation,
) -> Result<ContentAddressedProxyAssessment, ProxyError> {
    let settings = settings.validate()?;
    Ok(ContentAddressedProxyAssessment {
        cache_key: ProxyCacheKey::derive(source, settings)?,
        source_byte_len: source.byte_len,
        settings,
        assessment: assess_proxy(observation)?,
    })
}

pub fn assess_proxy(observation: ProxyPlaybackObservation) -> Result<ProxyAssessment, ProxyError> {
    let observation = observation.validate()?;
    if observation.sampled_frames < 60 {
        return Ok(ProxyAssessment::MeasurementRequired);
    }
    let mut reasons = BTreeSet::new();
    if observation.decode_p95_micros > observation.frame_budget_micros {
        reasons.insert(ProxyRecommendationReason::DecodeExceedsFrameBudget);
    }
    if observation.upload_p95_micros > observation.frame_budget_micros / 2 {
        reasons.insert(ProxyRecommendationReason::UploadExceedsFrameBudget);
    }
    if observation.frame_age_p95_micros > observation.frame_budget_micros * 2 {
        reasons.insert(ProxyRecommendationReason::FrameAgeExceedsFrameBudget);
    }
    if observation.dropped_frames != 0 {
        reasons.insert(ProxyRecommendationReason::DroppedFramesObserved);
    }
    if observation.pending_frames_peak > 2 {
        reasons.insert(ProxyRecommendationReason::DecoderQueuePressure);
    }
    if observation.visible_layers >= 8
        && observation.decode_p95_micros > observation.frame_budget_micros / 2
    {
        reasons.insert(ProxyRecommendationReason::MultiLayerPressure);
    }
    if reasons.is_empty() {
        Ok(ProxyAssessment::OriginalSufficient)
    } else {
        Ok(ProxyAssessment::ProxyRecommended(
            reasons.into_iter().collect(),
        ))
    }
}

#[allow(
    dead_code,
    reason = "bounded admission is retained for the deferred cache worker"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProxyCacheLimits {
    pub max_entries: usize,
    pub max_entry_bytes: u64,
    pub max_total_bytes: u64,
}

impl Default for ProxyCacheLimits {
    fn default() -> Self {
        Self {
            max_entries: 128,
            max_entry_bytes: 8 * 1024 * 1024 * 1024,
            max_total_bytes: 64 * 1024 * 1024 * 1024,
        }
    }
}

#[allow(
    dead_code,
    reason = "bounded admission is retained for the deferred cache worker"
)]
impl ProxyCacheLimits {
    pub fn validate(self) -> Result<Self, ProxyError> {
        if self.max_entries == 0 || self.max_entries > PROXY_CACHE_HARD_MAX_ENTRIES {
            return Err(ProxyError::InvalidCacheLimits("max_entries"));
        }
        if self.max_entry_bytes == 0
            || self.max_entry_bytes > PROXY_CACHE_HARD_MAX_ENTRY_BYTES
            || self.max_total_bytes == 0
            || self.max_total_bytes > PROXY_CACHE_HARD_MAX_TOTAL_BYTES
            || self.max_entry_bytes > self.max_total_bytes
        {
            return Err(ProxyError::InvalidCacheLimits("byte limits"));
        }
        Ok(self)
    }
}

#[allow(
    dead_code,
    reason = "cache index entries are retained for the deferred cache worker"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProxyCacheEntry {
    pub key: ProxyCacheKey,
    pub artifact_bytes: u64,
    pub last_used_ordinal: u64,
}

#[allow(
    dead_code,
    reason = "atomic publication law is retained for the deferred cache worker"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomicCacheCommitLaw {
    pub create_new_temp: bool,
    pub sync_temp_before_publish: bool,
    pub atomic_replace: bool,
    pub sync_parent_after_publish: bool,
}

#[allow(
    dead_code,
    reason = "atomic publication law is retained for the deferred cache worker"
)]
pub const ATOMIC_PROXY_CACHE_COMMIT_LAW: AtomicCacheCommitLaw = AtomicCacheCommitLaw {
    create_new_temp: true,
    sync_temp_before_publish: true,
    atomic_replace: true,
    sync_parent_after_publish: true,
};

#[allow(
    dead_code,
    reason = "bounded admission is retained for the deferred cache worker"
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProxyCachePlan {
    pub key: ProxyCacheKey,
    pub artifact_bytes: u64,
    pub evict: Box<[ProxyCacheKey]>,
    pub retained_bytes_before_stage: u64,
    pub peak_bytes_during_stage: u64,
    pub committed_bytes: u64,
    pub committed_entries: usize,
    pub commit_law: AtomicCacheCommitLaw,
}

#[allow(
    dead_code,
    reason = "bounded admission is retained for the deferred cache worker"
)]
impl ProxyCachePlan {
    pub fn preflight(
        key: ProxyCacheKey,
        artifact_bytes: u64,
        entries: &[ProxyCacheEntry],
        limits: ProxyCacheLimits,
    ) -> Result<Self, ProxyError> {
        let limits = limits.validate()?;
        if artifact_bytes == 0 || artifact_bytes > limits.max_entry_bytes {
            return Err(ProxyError::CacheEntryBytes {
                bytes: artifact_bytes,
                limit: limits.max_entry_bytes,
            });
        }
        if entries.len() > PROXY_CACHE_HARD_MAX_ENTRIES {
            return Err(ProxyError::CacheIndexEntries {
                count: entries.len(),
                limit: PROXY_CACHE_HARD_MAX_ENTRIES,
            });
        }

        let mut keys = BTreeSet::new();
        let mut retained_bytes = 0_u64;
        let mut existing_same_bytes = 0_u64;
        for entry in entries {
            if !keys.insert(entry.key) {
                return Err(ProxyError::DuplicateCacheKey(entry.key));
            }
            if entry.artifact_bytes == 0 || entry.artifact_bytes > limits.max_entry_bytes {
                return Err(ProxyError::CacheEntryBytes {
                    bytes: entry.artifact_bytes,
                    limit: limits.max_entry_bytes,
                });
            }
            retained_bytes = retained_bytes
                .checked_add(entry.artifact_bytes)
                .ok_or(ProxyError::ArithmeticOverflow)?;
            if entry.key == key {
                existing_same_bytes = entry.artifact_bytes;
            }
        }
        if retained_bytes > limits.max_total_bytes {
            return Err(ProxyError::ExistingCacheBytes {
                bytes: retained_bytes,
                limit: limits.max_total_bytes,
            });
        }

        let mut candidates = entries
            .iter()
            .filter(|entry| entry.key != key)
            .copied()
            .collect::<Vec<_>>();
        candidates.sort_by_key(|entry| (entry.last_used_ordinal, entry.key));
        let mut retained_entries = entries.len();
        let mut evict = Vec::new();
        let mut cursor = 0;
        loop {
            let peak = retained_bytes
                .checked_add(artifact_bytes)
                .ok_or(ProxyError::ArithmeticOverflow)?;
            let existing_same_entries = if existing_same_bytes == 0 { 0 } else { 1 };
            let committed_entries = retained_entries
                .checked_sub(existing_same_entries)
                .and_then(|value| value.checked_add(1))
                .ok_or(ProxyError::ArithmeticOverflow)?;
            if peak <= limits.max_total_bytes && committed_entries <= limits.max_entries {
                let committed_bytes = retained_bytes
                    .checked_sub(existing_same_bytes)
                    .and_then(|value| value.checked_add(artifact_bytes))
                    .ok_or(ProxyError::ArithmeticOverflow)?;
                return Ok(Self {
                    key,
                    artifact_bytes,
                    evict: evict.into_boxed_slice(),
                    retained_bytes_before_stage: retained_bytes,
                    peak_bytes_during_stage: peak,
                    committed_bytes,
                    committed_entries,
                    commit_law: ATOMIC_PROXY_CACHE_COMMIT_LAW,
                });
            }
            let Some(candidate) = candidates.get(cursor) else {
                return Err(ProxyError::CacheAdmission {
                    peak_bytes: peak,
                    byte_limit: limits.max_total_bytes,
                    entries: committed_entries,
                    entry_limit: limits.max_entries,
                });
            };
            cursor += 1;
            retained_bytes = retained_bytes
                .checked_sub(candidate.artifact_bytes)
                .ok_or(ProxyError::ArithmeticOverflow)?;
            retained_entries = retained_entries
                .checked_sub(1)
                .ok_or(ProxyError::ArithmeticOverflow)?;
            evict.push(candidate.key);
        }
    }
}

#[allow(
    dead_code,
    reason = "cache-specific errors remain part of the deferred worker contract"
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProxyError {
    UnsupportedSchemaVersion(u16),
    UnsupportedAlgorithmVersion(u16),
    InvalidFrameRate {
        numerator: u32,
        denominator: u32,
    },
    InvalidContentIdentity,
    InvalidCacheKey,
    InvalidObservation(&'static str),
    InvalidCacheLimits(&'static str),
    CacheIndexEntries {
        count: usize,
        limit: usize,
    },
    CacheEntryBytes {
        bytes: u64,
        limit: u64,
    },
    ExistingCacheBytes {
        bytes: u64,
        limit: u64,
    },
    DuplicateCacheKey(ProxyCacheKey),
    CacheAdmission {
        peak_bytes: u64,
        byte_limit: u64,
        entries: usize,
        entry_limit: usize,
    },
    ArithmeticOverflow,
}

impl fmt::Display for ProxyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion(version) => {
                write!(formatter, "unsupported proxy schema version {version}")
            }
            Self::UnsupportedAlgorithmVersion(version) => {
                write!(formatter, "unsupported proxy algorithm version {version}")
            }
            Self::InvalidFrameRate {
                numerator,
                denominator,
            } => write!(formatter, "invalid fixed proxy frame rate {numerator}/{denominator}"),
            Self::InvalidContentIdentity => formatter.write_str(
                "proxy source identity must be canonical lowercase SHA-256 plus non-zero bytes",
            ),
            Self::InvalidCacheKey => formatter.write_str("invalid canonical proxy cache key"),
            Self::InvalidObservation(field) => {
                write!(formatter, "invalid proxy playback observation: {field}")
            }
            Self::InvalidCacheLimits(field) => {
                write!(formatter, "invalid proxy cache limits: {field}")
            }
            Self::CacheIndexEntries { count, limit } => {
                write!(formatter, "proxy cache index has {count} entries; hard limit is {limit}")
            }
            Self::CacheEntryBytes { bytes, limit } => {
                write!(formatter, "proxy artifact is {bytes} bytes; limit is {limit}")
            }
            Self::ExistingCacheBytes { bytes, limit } => write!(
                formatter,
                "existing proxy cache is {bytes} bytes; configured limit is {limit}"
            ),
            Self::DuplicateCacheKey(key) => write!(formatter, "duplicate proxy cache key {key:?}"),
            Self::CacheAdmission {
                peak_bytes,
                byte_limit,
                entries,
                entry_limit,
            } => write!(
                formatter,
                "proxy cache admission needs {peak_bytes}/{byte_limit} peak bytes and {entries}/{entry_limit} entries"
            ),
            Self::ArithmeticOverflow => {
                formatter.write_str("proxy cache resource arithmetic overflowed")
            }
        }
    }
}

impl std::error::Error for ProxyError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(digit: char, bytes: u64) -> ContentIdentity {
        ContentIdentity::new(digit.to_string().repeat(64), bytes).unwrap()
    }

    fn key(digit: char) -> ProxyCacheKey {
        ProxyCacheKey::derive(&identity(digit, 4_096), ProxySettings::default()).unwrap()
    }

    #[test]
    fn cache_key_is_path_independent_settings_sensitive_and_has_a_golden() {
        let source = identity('a', 12_345);
        let first = ProxyCacheKey::derive(&source, ProxySettings::default()).unwrap();
        let renamed_source = ContentIdentity::new(source.sha256.clone(), source.byte_len).unwrap();
        let renamed = ProxyCacheKey::derive(&renamed_source, ProxySettings::default()).unwrap();
        assert_eq!(first, renamed);
        assert_eq!(
            first.to_hex(),
            "5fcb91222a0163213508b5109b4e5a456ad0e4b7addd95e2f849ce6c8e43607e"
        );

        let changed = ProxySettings {
            scale: ProxyScale::Quarter,
            ..ProxySettings::default()
        };
        assert_ne!(first, ProxyCacheKey::derive(&source, changed).unwrap());
        assert_ne!(
            first,
            ProxyCacheKey::derive(&identity('b', 12_345), ProxySettings::default()).unwrap()
        );
        assert_eq!(
            first.artifact_file_name(ProxyFormat::Ffv1Matroska),
            format!("{}.mkv", first.to_hex())
        );
    }

    #[test]
    fn settings_and_observations_reject_hostile_serde() {
        let mut settings = serde_json::to_value(ProxySettings::default()).unwrap();
        settings["unexpected"] = serde_json::json!(true);
        assert!(serde_json::from_value::<ProxySettings>(settings).is_err());
        let hostile = r#"{
            "schema_version":1,"algorithm_version":1,"format":"ffv1_matroska",
            "scale":"half","frame_rate":{"fixed":{"numerator":241,"denominator":1}},
            "include_audio":true
        }"#;
        assert!(serde_json::from_str::<ProxySettings>(hostile).is_err());
        assert!(ProxyCacheKey::from_hex(&"A".repeat(64)).is_err());

        let observation = ProxyPlaybackObservation {
            sampled_frames: 1,
            visible_layers: 1,
            frame_budget_micros: 16_667,
            decode_p95_micros: 1,
            upload_p95_micros: 1,
            frame_age_p95_micros: 1,
            dropped_frames: 2,
            pending_frames_peak: 0,
            hardware_decode_active: false,
            zero_copy_active: false,
        };
        assert!(observation.validate().is_err());

        let mut hostile_budget = observation;
        hostile_budget.dropped_frames = 0;
        hostile_budget.frame_budget_micros = u32::MAX;
        assert!(hostile_budget.validate().is_err());
        assert!(assess_proxy(hostile_budget).is_err());

        let mut serialized = serde_json::to_value(hostile_budget).unwrap();
        serialized["frame_budget_micros"] = serde_json::json!(u32::MAX);
        assert!(serde_json::from_value::<ProxyPlaybackObservation>(serialized).is_err());
    }

    #[test]
    fn playback_assessment_requires_measurement_then_reports_objective_reasons() {
        let mut observation = ProxyPlaybackObservation {
            sampled_frames: 59,
            visible_layers: 1,
            frame_budget_micros: 16_667,
            decode_p95_micros: 4_000,
            upload_p95_micros: 2_000,
            frame_age_p95_micros: 8_000,
            dropped_frames: 0,
            pending_frames_peak: 1,
            hardware_decode_active: false,
            zero_copy_active: false,
        };
        assert_eq!(
            assess_proxy(observation).unwrap(),
            ProxyAssessment::MeasurementRequired
        );
        observation.sampled_frames = 600;
        assert_eq!(
            assess_proxy(observation).unwrap(),
            ProxyAssessment::OriginalSufficient
        );
        observation.decode_p95_micros = 20_000;
        observation.dropped_frames = 1;
        assert_eq!(
            assess_proxy(observation).unwrap(),
            ProxyAssessment::ProxyRecommended(
                vec![
                    ProxyRecommendationReason::DecodeExceedsFrameBudget,
                    ProxyRecommendationReason::DroppedFramesObserved,
                ]
                .into_boxed_slice()
            )
        );
        let keyed = assess_content_addressed_proxy(
            &identity('a', 123),
            ProxySettings::default(),
            observation,
        )
        .unwrap();
        assert_eq!(keyed.source_byte_len, 123);
        assert_eq!(
            keyed.cache_key,
            ProxyCacheKey::derive(&identity('a', 123), ProxySettings::default()).unwrap()
        );
    }

    #[test]
    fn cache_preflight_evicts_deterministically_and_counts_atomic_staging() {
        let limits = ProxyCacheLimits {
            max_entries: 2,
            max_entry_bytes: 100,
            max_total_bytes: 200,
        };
        let entries = [
            ProxyCacheEntry {
                key: key('b'),
                artifact_bytes: 60,
                last_used_ordinal: 1,
            },
            ProxyCacheEntry {
                key: key('a'),
                artifact_bytes: 60,
                last_used_ordinal: 1,
            },
        ];
        let incoming = key('c');
        let plan = ProxyCachePlan::preflight(incoming, 100, &entries, limits).unwrap();
        let deterministic_first = entries.iter().map(|entry| entry.key).min().unwrap();
        assert_eq!(plan.evict.as_ref(), &[deterministic_first]);
        assert_eq!(plan.retained_bytes_before_stage, 60);
        assert_eq!(plan.peak_bytes_during_stage, 160);
        assert_eq!(plan.committed_bytes, 160);
        assert_eq!(plan.committed_entries, 2);
        assert_eq!(plan.commit_law, ATOMIC_PROXY_CACHE_COMMIT_LAW);

        let replacement = ProxyCachePlan::preflight(entries[0].key, 100, &entries, limits).unwrap();
        assert_eq!(replacement.evict.len(), 1);
        assert_eq!(replacement.peak_bytes_during_stage, 160);
        assert_eq!(replacement.committed_bytes, 100);
        assert_eq!(replacement.committed_entries, 1);
    }

    #[test]
    fn cache_preflight_rejects_duplicate_overflow_and_impossible_staging() {
        let duplicate = ProxyCacheEntry {
            key: key('a'),
            artifact_bytes: 1,
            last_used_ordinal: 1,
        };
        assert!(matches!(
            ProxyCachePlan::preflight(
                key('b'),
                1,
                &[duplicate, duplicate],
                ProxyCacheLimits::default()
            ),
            Err(ProxyError::DuplicateCacheKey(_))
        ));
        assert!(ProxyCacheLimits {
            max_entries: usize::MAX,
            ..ProxyCacheLimits::default()
        }
        .validate()
        .is_err());
        assert!(matches!(
            ProxyCachePlan::preflight(
                key('b'),
                101,
                &[],
                ProxyCacheLimits {
                    max_entries: 1,
                    max_entry_bytes: 100,
                    max_total_bytes: 100,
                }
            ),
            Err(ProxyError::CacheEntryBytes { .. })
        ));
    }
}
