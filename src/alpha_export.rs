//! Exact straight-alpha export contracts and crash-safe artifact publishers.
//!
//! `pre_opaque_straight_alpha_v1` is the renderer's straight RGBA8/sRGB
//! programme image after creative composition, Temporal, and display effects,
//! but before the existing black opaque audience pass.  This module never
//! changes pixels in the ordinary MP4 path: callers explicitly opt in, read
//! that seam, and submit the resulting bytes to one of these publishers.

use crate::durable_file::{
    publish_bytes, publish_directory_noreplace, sync_directory, PublishMode,
};
use image::{codecs::png::PngEncoder, ExtendedColorType, ImageEncoder};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};

/// Stable renderer/readback seam name. Changing its pixel law requires a new
/// name and schema rather than silently changing this contract.
pub const PRE_OPAQUE_STRAIGHT_ALPHA_V1: &str = "pre_opaque_straight_alpha_v1";
pub const ALPHA_EXPORT_RECEIPT_SCHEMA: u16 = 1;

const MAX_DIMENSION: u32 = 16_384;
const MAX_FRAME_BYTES: u64 = 256 * 1024 * 1024;
const MAX_FRAME_COUNT: u64 = 1_000_000;
const MAX_FPS: u64 = 240;
const STDERR_LIMIT: u64 = 64 * 1024;

/// Integer-linear thresholds between neighboring sRGB codes. Each threshold
/// is `ceil(srgb_decode((code + 0.5) / 255) * 65535)`, pinning the inverse of
/// the shared Q0.16 decode table without runtime floating point.
const LINEAR_Q_TO_SRGB8_THRESHOLDS: [u32; 255] = [
    10, 30, 50, 70, 90, 110, 130, 150, 170, 189, 209, 230, 253, 276, 301, 327, 354, 382, 412, 443,
    475, 509, 544, 580, 618, 657, 698, 740, 783, 828, 875, 923, 972, 1_023, 1_075, 1_129, 1_185,
    1_242, 1_300, 1_360, 1_422, 1_486, 1_551, 1_617, 1_685, 1_755, 1_827, 1_900, 1_975, 2_052,
    2_130, 2_210, 2_292, 2_376, 2_461, 2_548, 2_637, 2_727, 2_820, 2_914, 3_010, 3_108, 3_208,
    3_309, 3_412, 3_518, 3_625, 3_734, 3_844, 3_957, 4_072, 4_188, 4_307, 4_427, 4_550, 4_674,
    4_800, 4_928, 5_059, 5_191, 5_325, 5_461, 5_599, 5_740, 5_882, 6_026, 6_173, 6_321, 6_471,
    6_624, 6_778, 6_935, 7_094, 7_255, 7_418, 7_583, 7_750, 7_919, 8_091, 8_265, 8_440, 8_618,
    8_798, 8_981, 9_165, 9_352, 9_541, 9_732, 9_925, 10_121, 10_318, 10_518, 10_720, 10_925,
    11_132, 11_341, 11_552, 11_765, 11_981, 12_199, 12_420, 12_643, 12_868, 13_095, 13_325, 13_557,
    13_791, 14_028, 14_267, 14_508, 14_752, 14_998, 15_247, 15_498, 15_751, 16_007, 16_265, 16_525,
    16_788, 17_054, 17_321, 17_592, 17_864, 18_139, 18_417, 18_697, 18_980, 19_264, 19_552, 19_842,
    20_134, 20_429, 20_727, 21_027, 21_329, 21_634, 21_942, 22_252, 22_564, 22_880, 23_197, 23_518,
    23_840, 24_166, 24_494, 24_824, 25_158, 25_493, 25_832, 26_173, 26_516, 26_862, 27_211, 27_563,
    27_917, 28_273, 28_633, 28_995, 29_359, 29_727, 30_097, 30_469, 30_845, 31_223, 31_603, 31_987,
    32_373, 32_762, 33_153, 33_547, 33_944, 34_344, 34_747, 35_152, 35_560, 35_970, 36_384, 36_800,
    37_219, 37_640, 38_065, 38_492, 38_922, 39_355, 39_790, 40_229, 40_670, 41_114, 41_561, 42_011,
    42_463, 42_918, 43_377, 43_838, 44_301, 44_768, 45_238, 45_710, 46_185, 46_663, 47_144, 47_628,
    48_115, 48_605, 49_097, 49_593, 50_091, 50_592, 51_096, 51_604, 52_114, 52_627, 53_142, 53_661,
    54_183, 54_708, 55_235, 55_766, 56_300, 56_836, 57_376, 57_918, 58_464, 59_012, 59_564, 60_118,
    60_675, 61_236, 61_799, 62_366, 62_935, 63_508, 64_083, 64_662, 65_244,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlphaArtifactKind {
    StraightPngSequence,
    FillKeyPngSequence,
    StraightPngAndFillKey,
    Ffv1Rgba,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlphaEffectState {
    pub codec_mosh: bool,
    pub final_program_vhs: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlphaExportPlan {
    pub width: u32,
    pub height: u32,
    pub frame_count: u64,
    pub fps_numerator: u32,
    pub fps_denominator: u32,
    pub artifact: AlphaArtifactKind,
    pub effects: AlphaEffectState,
}

impl AlphaExportPlan {
    fn frame_bytes(self) -> Result<usize, AlphaExportError> {
        let bytes = u64::from(self.width)
            .checked_mul(u64::from(self.height))
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or(AlphaExportError::InvalidPlan("frame byte count overflows"))?;
        if bytes > MAX_FRAME_BYTES {
            return Err(AlphaExportError::InvalidPlan(
                "frame exceeds the 256 MiB admission bound",
            ));
        }
        usize::try_from(bytes)
            .map_err(|_| AlphaExportError::InvalidPlan("frame does not fit address space"))
    }

    pub fn validate(self) -> Result<(), AlphaExportError> {
        if self.width == 0
            || self.height == 0
            || self.width > MAX_DIMENSION
            || self.height > MAX_DIMENSION
        {
            return Err(AlphaExportError::InvalidPlan(
                "dimensions must be within 1..=16384",
            ));
        }
        self.frame_bytes()?;
        if self.frame_count == 0 || self.frame_count > MAX_FRAME_COUNT {
            return Err(AlphaExportError::InvalidPlan(
                "frame_count must be within 1..=1000000",
            ));
        }
        if self.fps_numerator == 0
            || self.fps_denominator == 0
            || u64::from(self.fps_numerator)
                > MAX_FPS.saturating_mul(u64::from(self.fps_denominator))
        {
            return Err(AlphaExportError::InvalidPlan(
                "frame rate must be positive and no greater than 240 fps",
            ));
        }
        if self.effects.codec_mosh {
            return Err(AlphaExportError::EffectAmbiguity("Codec-Mosh"));
        }
        if self.effects.final_program_vhs {
            return Err(AlphaExportError::EffectAmbiguity("final-program VHS"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlphaExportReceipt {
    pub schema_version: u16,
    pub seam: String,
    pub artifact: AlphaArtifactKind,
    pub width: u32,
    pub height: u32,
    pub frame_count: u64,
    pub fps_numerator: u32,
    pub fps_denominator: u32,
    pub source_pixel_law: String,
    pub premultiplication_law: String,
    pub black_fill_law: String,
    pub key_law: String,
    pub effect_law: String,
    pub storage_pixel_format: String,
    pub raw_straight_rgba_sha256: String,
    pub file_patterns: Vec<String>,
}

impl AlphaExportReceipt {
    fn new(plan: AlphaExportPlan, raw_hash: &Sha256, storage_pixel_format: &str) -> Self {
        let file_patterns = match plan.artifact {
            AlphaArtifactKind::StraightPngSequence => vec!["rgba/frame_%08d.png".to_owned()],
            AlphaArtifactKind::FillKeyPngSequence => vec![
                "fill/frame_%08d.png".to_owned(),
                "key/frame_%08d.png".to_owned(),
            ],
            AlphaArtifactKind::StraightPngAndFillKey => vec![
                "rgba/frame_%08d.png".to_owned(),
                "fill/frame_%08d.png".to_owned(),
                "key/frame_%08d.png".to_owned(),
            ],
            AlphaArtifactKind::Ffv1Rgba => vec!["plate.mkv".to_owned()],
        };
        Self {
            schema_version: ALPHA_EXPORT_RECEIPT_SCHEMA,
            seam: PRE_OPAQUE_STRAIGHT_ALPHA_V1.to_owned(),
            artifact: plan.artifact,
            width: plan.width,
            height: plan.height,
            frame_count: plan.frame_count,
            fps_numerator: plan.fps_numerator,
            fps_denominator: plan.fps_denominator,
            source_pixel_law: "RGBA8_UNORM_SRGB_straight_alpha_hidden_RGB_preserved".to_owned(),
            premultiplication_law:
                "source_is_never_premultiplied; fill_only_is_linear_RGB_times_alpha".to_owned(),
            black_fill_law:
                "decode_sRGB_Q0.16; round(linear*alpha/255); threshold_encode_sRGB; alpha=255"
                    .to_owned(),
            key_law: "RGB=source_alpha; alpha=255".to_owned(),
            effect_law: "Codec-Mosh_and_final-program_VHS_are_refused".to_owned(),
            storage_pixel_format: storage_pixel_format.to_owned(),
            raw_straight_rgba_sha256: upper_hex(raw_hash.clone().finalize().as_slice()),
            file_patterns,
        }
    }
}

#[derive(Debug)]
pub enum AlphaExportError {
    InvalidPlan(&'static str),
    EffectAmbiguity(&'static str),
    UnexpectedArtifact,
    FrameOrdinal { expected: u64, actual: u64 },
    FrameLength { expected: usize, actual: usize },
    IncompleteSequence { expected: u64, actual: u64 },
    Io(io::Error),
    Image(image::ImageError),
    Receipt(serde_json::Error),
    EncoderFailed { code: Option<i32>, stderr: String },
}

impl fmt::Display for AlphaExportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPlan(message) => write!(formatter, "invalid alpha-export plan: {message}"),
            Self::EffectAmbiguity(effect) => write!(
                formatter,
                "{effect} has no proven straight-alpha propagation law; alpha export refused"
            ),
            Self::UnexpectedArtifact => write!(formatter, "publisher does not match artifact kind"),
            Self::FrameOrdinal { expected, actual } => {
                write!(formatter, "expected frame {expected}, received {actual}")
            }
            Self::FrameLength { expected, actual } => {
                write!(
                    formatter,
                    "expected {expected} RGBA bytes, received {actual}"
                )
            }
            Self::IncompleteSequence { expected, actual } => {
                write!(formatter, "expected {expected} frames, received {actual}")
            }
            Self::Io(error) => write!(formatter, "alpha-export IO: {error}"),
            Self::Image(error) => write!(formatter, "alpha-export PNG: {error}"),
            Self::Receipt(error) => write!(formatter, "alpha-export receipt: {error}"),
            Self::EncoderFailed { code, stderr } => {
                write!(formatter, "FFV1 encoder failed ({code:?}): {stderr}")
            }
        }
    }
}

impl Error for AlphaExportError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Image(error) => Some(error),
            Self::Receipt(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for AlphaExportError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<image::ImageError> for AlphaExportError {
    fn from(error: image::ImageError) -> Self {
        Self::Image(error)
    }
}

impl From<serde_json::Error> for AlphaExportError {
    fn from(error: serde_json::Error) -> Self {
        Self::Receipt(error)
    }
}

#[derive(Debug)]
struct StagingDirectory {
    path: Option<PathBuf>,
}

impl StagingDirectory {
    fn create(destination: &Path) -> Result<Self, AlphaExportError> {
        let file_name = destination
            .file_name()
            .ok_or(AlphaExportError::InvalidPlan(
                "destination must have a final directory name",
            ))?;
        let parent = destination
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        if !parent.is_dir() {
            return Err(AlphaExportError::InvalidPlan(
                "destination parent must already exist",
            ));
        }
        if destination.exists() {
            return Err(AlphaExportError::InvalidPlan(
                "destination already exists; alpha export never replaces it",
            ));
        }
        for _ in 0..16 {
            let mut nonce = [0_u8; 16];
            getrandom::fill(&mut nonce).map_err(|error| {
                AlphaExportError::Io(io::Error::other(format!("staging entropy: {error}")))
            })?;
            let path = parent.join(format!(
                ".alpha-export-{}-{}-{}.part",
                std::process::id(),
                file_name.to_string_lossy(),
                upper_hex(&nonce)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path: Some(path) }),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(AlphaExportError::Io(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not reserve alpha-export staging directory",
        )))
    }

    fn path(&self) -> &Path {
        self.path
            .as_deref()
            .expect("active alpha publisher owns its staging directory")
    }

    fn publish(&mut self, destination: &Path) -> Result<(), AlphaExportError> {
        let path = self
            .path
            .as_deref()
            .expect("active alpha publisher owns its staging directory");
        publish_directory_noreplace(path, destination)?;
        self.path = None;
        Ok(())
    }
}

impl Drop for StagingDirectory {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            let _ = fs::remove_dir_all(path);
        }
    }
}

/// Transactional PNG-generation publisher. Until `finish`, no destination
/// name is visible; Drop cancels and removes the private sibling directory.
#[derive(Debug)]
pub struct AlphaPngSequencePublisher {
    plan: AlphaExportPlan,
    destination: PathBuf,
    staging: StagingDirectory,
    next_frame: u64,
    raw_hash: Sha256,
}

impl AlphaPngSequencePublisher {
    pub fn begin(
        destination: impl AsRef<Path>,
        plan: AlphaExportPlan,
    ) -> Result<Self, AlphaExportError> {
        plan.validate()?;
        if plan.artifact == AlphaArtifactKind::Ffv1Rgba {
            return Err(AlphaExportError::UnexpectedArtifact);
        }
        let destination = destination.as_ref().to_path_buf();
        let staging = StagingDirectory::create(&destination)?;
        for directory in png_directories(plan.artifact) {
            fs::create_dir(staging.path().join(directory))?;
        }
        Ok(Self {
            plan,
            destination,
            staging,
            next_frame: 0,
            raw_hash: Sha256::new(),
        })
    }

    pub fn write_frame(&mut self, ordinal: u64, rgba: &[u8]) -> Result<(), AlphaExportError> {
        validate_frame(self.plan, self.next_frame, ordinal, rgba)?;
        let file_name = format!("frame_{ordinal:08}.png");
        if matches!(
            self.plan.artifact,
            AlphaArtifactKind::StraightPngSequence | AlphaArtifactKind::StraightPngAndFillKey
        ) {
            write_png(
                &self.staging.path().join("rgba").join(&file_name),
                rgba,
                self.plan.width,
                self.plan.height,
            )?;
        }
        if matches!(
            self.plan.artifact,
            AlphaArtifactKind::FillKeyPngSequence | AlphaArtifactKind::StraightPngAndFillKey
        ) {
            let mut derived = vec![0_u8; rgba.len()];
            derive_black_fill_rgba(rgba, &mut derived)?;
            write_png(
                &self.staging.path().join("fill").join(&file_name),
                &derived,
                self.plan.width,
                self.plan.height,
            )?;
            derive_key_rgba(rgba, &mut derived)?;
            write_png(
                &self.staging.path().join("key").join(&file_name),
                &derived,
                self.plan.width,
                self.plan.height,
            )?;
        }
        self.raw_hash.update(rgba);
        self.next_frame += 1;
        Ok(())
    }

    pub fn finish(mut self) -> Result<AlphaExportReceipt, AlphaExportError> {
        ensure_complete(self.plan, self.next_frame)?;
        let receipt = AlphaExportReceipt::new(self.plan, &self.raw_hash, "PNG_RGBA8");
        write_receipt(self.staging.path(), &receipt)?;
        for directory in png_directories(self.plan.artifact) {
            sync_directory(&self.staging.path().join(directory))?;
        }
        sync_directory(self.staging.path())?;
        self.staging.publish(&self.destination)?;
        Ok(receipt)
    }
}

/// Transactional FFV1 publisher. The public byte contract is packed straight
/// RGBA8; FFmpeg stores the lossless stream as planar `gbrap` inside Matroska.
/// Exact decode-to-RGBA round-trip is a required keep gate.
#[derive(Debug)]
pub struct AlphaFfv1Publisher {
    plan: AlphaExportPlan,
    destination: PathBuf,
    staging: StagingDirectory,
    stderr_path: PathBuf,
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    next_frame: u64,
    raw_hash: Sha256,
}

impl AlphaFfv1Publisher {
    pub fn begin(
        ffmpeg: impl AsRef<Path>,
        destination: impl AsRef<Path>,
        plan: AlphaExportPlan,
    ) -> Result<Self, AlphaExportError> {
        plan.validate()?;
        if plan.artifact != AlphaArtifactKind::Ffv1Rgba {
            return Err(AlphaExportError::UnexpectedArtifact);
        }
        let ffmpeg = ffmpeg.as_ref();
        if !ffmpeg.is_file() {
            return Err(AlphaExportError::InvalidPlan(
                "ffmpeg executable is not a regular file",
            ));
        }
        let destination = destination.as_ref().to_path_buf();
        let staging = StagingDirectory::create(&destination)?;
        let output_path = staging.path().join("plate.mkv");
        let stderr_path = staging.path().join("ffmpeg.stderr.log");
        let stderr = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&stderr_path)?;
        let frame_rate = format!("{}/{}", plan.fps_numerator, plan.fps_denominator);
        let video_size = format!("{}x{}", plan.width, plan.height);
        let mut child = Command::new(ffmpeg)
            .args([
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-n",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgba",
                "-video_size",
                &video_size,
                "-framerate",
                &frame_rate,
                "-i",
                "pipe:0",
                "-map",
                "0:v:0",
                "-an",
                "-c:v",
                "ffv1",
                "-level",
                "3",
                "-coder",
                "1",
                "-context",
                "1",
                "-g",
                "1",
                "-pix_fmt",
                "gbrap",
                "-color_range",
                "pc",
                "-colorspace",
                "bt709",
                "-color_trc",
                "iec61966-2-1",
                "-color_primaries",
                "bt709",
                "-map_metadata",
                "-1",
                "-f",
                "matroska",
            ])
            .arg(&output_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::from(stderr))
            .spawn()?;
        let stdin = child.stdin.take().ok_or_else(|| {
            AlphaExportError::Io(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "FFmpeg did not expose stdin",
            ))
        })?;
        Ok(Self {
            plan,
            destination,
            staging,
            stderr_path,
            child: Some(child),
            stdin: Some(stdin),
            next_frame: 0,
            raw_hash: Sha256::new(),
        })
    }

    pub fn write_frame(&mut self, ordinal: u64, rgba: &[u8]) -> Result<(), AlphaExportError> {
        validate_frame(self.plan, self.next_frame, ordinal, rgba)?;
        self.stdin
            .as_mut()
            .ok_or_else(|| {
                AlphaExportError::Io(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "FFV1 publisher stdin is closed",
                ))
            })?
            .write_all(rgba)?;
        self.raw_hash.update(rgba);
        self.next_frame += 1;
        Ok(())
    }

    pub fn finish(mut self) -> Result<AlphaExportReceipt, AlphaExportError> {
        ensure_complete(self.plan, self.next_frame)?;
        drop(self.stdin.take());
        let status = self
            .child
            .take()
            .expect("active FFV1 publisher owns its child")
            .wait()?;
        if !status.success() {
            return Err(AlphaExportError::EncoderFailed {
                code: status.code(),
                stderr: bounded_stderr(&self.stderr_path),
            });
        }
        fs::remove_file(&self.stderr_path)
            .map_err(|error| contextual_io("remove successful FFmpeg stderr log", error))?;
        let output_path = self.staging.path().join("plate.mkv");
        // Windows requires a writable handle for FlushFileBuffers/sync_all.
        let output = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&output_path)
            .map_err(|error| contextual_io("open completed FFV1 artifact", error))?;
        if output
            .metadata()
            .map_err(|error| contextual_io("inspect completed FFV1 artifact", error))?
            .len()
            == 0
        {
            return Err(AlphaExportError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                "FFV1 encoder produced an empty artifact",
            )));
        }
        output
            .sync_all()
            .map_err(|error| contextual_io("sync completed FFV1 artifact", error))?;
        // Windows cannot rename the staging directory while this child handle
        // remains open, even though the file itself is fully flushed.
        drop(output);
        let receipt = AlphaExportReceipt::new(self.plan, &self.raw_hash, "FFV1_v3_GBRAP");
        write_receipt(self.staging.path(), &receipt)?;
        sync_directory(self.staging.path())?;
        self.staging.publish(&self.destination)?;
        Ok(receipt)
    }
}

impl Drop for AlphaFfv1Publisher {
    fn drop(&mut self) {
        drop(self.stdin.take());
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Produce the paired fill image over black. RGB is decoded with the pinned
/// sRGB table, multiplied by alpha in Q0.16, and re-encoded with pinned
/// thresholds. Output alpha is opaque. Hidden RGB at alpha zero becomes black.
pub fn derive_black_fill_rgba(
    straight_rgba: &[u8],
    output: &mut [u8],
) -> Result<(), AlphaExportError> {
    validate_derived_buffers(straight_rgba, output)?;
    for (source, destination) in straight_rgba
        .as_chunks::<4>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<4>().0)
    {
        let alpha = u32::from(source[3]);
        for channel in 0..3 {
            let linear = crate::photosensitivity_advisor::srgb_code_to_linear_q(source[channel]);
            let over_black = linear.saturating_mul(alpha).saturating_add(127) / 255;
            destination[channel] = linear_q_to_srgb8(over_black);
        }
        destination[3] = 255;
    }
    Ok(())
}

/// Produce the paired grayscale key; RGB is the exact source alpha and the
/// PNG itself is opaque so generic players do not apply alpha a second time.
pub fn derive_key_rgba(straight_rgba: &[u8], output: &mut [u8]) -> Result<(), AlphaExportError> {
    validate_derived_buffers(straight_rgba, output)?;
    for (source, destination) in straight_rgba
        .as_chunks::<4>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<4>().0)
    {
        destination[..3].fill(source[3]);
        destination[3] = 255;
    }
    Ok(())
}

fn validate_derived_buffers(input: &[u8], output: &[u8]) -> Result<(), AlphaExportError> {
    if !input.len().is_multiple_of(4) || input.len() != output.len() {
        return Err(AlphaExportError::FrameLength {
            expected: input.len().div_ceil(4) * 4,
            actual: output.len(),
        });
    }
    Ok(())
}

fn linear_q_to_srgb8(linear_q: u32) -> u8 {
    LINEAR_Q_TO_SRGB8_THRESHOLDS.partition_point(|threshold| *threshold <= linear_q) as u8
}

fn validate_frame(
    plan: AlphaExportPlan,
    expected_ordinal: u64,
    ordinal: u64,
    rgba: &[u8],
) -> Result<(), AlphaExportError> {
    if ordinal != expected_ordinal || ordinal >= plan.frame_count {
        return Err(AlphaExportError::FrameOrdinal {
            expected: expected_ordinal,
            actual: ordinal,
        });
    }
    let expected = plan.frame_bytes()?;
    if rgba.len() != expected {
        return Err(AlphaExportError::FrameLength {
            expected,
            actual: rgba.len(),
        });
    }
    Ok(())
}

fn ensure_complete(plan: AlphaExportPlan, actual: u64) -> Result<(), AlphaExportError> {
    if actual != plan.frame_count {
        return Err(AlphaExportError::IncompleteSequence {
            expected: plan.frame_count,
            actual,
        });
    }
    Ok(())
}

fn png_directories(artifact: AlphaArtifactKind) -> &'static [&'static str] {
    match artifact {
        AlphaArtifactKind::StraightPngSequence => &["rgba"],
        AlphaArtifactKind::FillKeyPngSequence => &["fill", "key"],
        AlphaArtifactKind::StraightPngAndFillKey => &["rgba", "fill", "key"],
        AlphaArtifactKind::Ffv1Rgba => &[],
    }
}

fn write_png(path: &Path, rgba: &[u8], width: u32, height: u32) -> Result<(), AlphaExportError> {
    let (publication, mut file) =
        crate::durable_file::StagedPublication::create(path, "alpha-png")?;
    PngEncoder::new(&mut file).write_image(rgba, width, height, ExtendedColorType::Rgba8)?;
    publication.commit(file, PublishMode::NoReplace)?;
    Ok(())
}

fn write_receipt(
    staging_directory: &Path,
    receipt: &AlphaExportReceipt,
) -> Result<(), AlphaExportError> {
    let mut bytes = serde_json::to_vec_pretty(receipt)?;
    bytes.push(b'\n');
    publish_bytes(
        &staging_directory.join("alpha-export-receipt.json"),
        &bytes,
        "alpha-receipt",
        PublishMode::NoReplace,
    )?;
    Ok(())
}

fn bounded_stderr(path: &Path) -> String {
    let Ok(file) = File::open(path) else {
        return "stderr unavailable".to_owned();
    };
    let mut bytes = Vec::new();
    let _ = file.take(STDERR_LIMIT).read_to_end(&mut bytes);
    String::from_utf8_lossy(&bytes).trim().to_owned()
}

fn contextual_io(context: &str, error: io::Error) -> AlphaExportError {
    AlphaExportError::Io(io::Error::new(error.kind(), format!("{context}: {error}")))
}

fn upper_hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use fmt::Write as _;
        let _ = write!(&mut output, "{byte:02X}");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(label: &str) -> Self {
            for _ in 0..16 {
                let mut nonce = [0_u8; 8];
                getrandom::fill(&mut nonce).unwrap();
                let path = std::env::temp_dir().join(format!(
                    "collide-o-scope-{label}-{}-{}",
                    std::process::id(),
                    upper_hex(&nonce)
                ));
                match fs::create_dir(&path) {
                    Ok(()) => return Self(path),
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                    Err(error) => panic!("test directory: {error}"),
                }
            }
            panic!("could not reserve test directory");
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn plan(artifact: AlphaArtifactKind, frames: u64) -> AlphaExportPlan {
        AlphaExportPlan {
            width: 4,
            height: 2,
            frame_count: frames,
            fps_numerator: 24_000,
            fps_denominator: 1_001,
            artifact,
            effects: AlphaEffectState {
                codec_mosh: false,
                final_program_vhs: false,
            },
        }
    }

    fn hostile_fixture() -> Vec<u8> {
        vec![
            201, 17, 99, 0, // hidden RGB
            255, 64, 0, 127, // soft key
            9, 200, 41, 0, // cellular gap
            23, 77, 222, 191, // partial group matte
            255, 255, 255, 1, // transform edge low coverage
            2, 3, 4, 254, // transform edge high coverage
            0, 0, 0, 255, // black fill
            61, 127, 253, 255, // opaque identity
        ]
    }

    #[test]
    fn d5_effect_ambiguity_is_refused_before_any_output_exists() {
        let directory = TestDirectory::new("d5-refusal");
        for effects in [
            AlphaEffectState {
                codec_mosh: true,
                final_program_vhs: false,
            },
            AlphaEffectState {
                codec_mosh: false,
                final_program_vhs: true,
            },
            AlphaEffectState {
                codec_mosh: true,
                final_program_vhs: true,
            },
        ] {
            let mut request = plan(AlphaArtifactKind::StraightPngSequence, 1);
            request.effects = effects;
            let destination = directory.0.join(format!("refused-{effects:?}"));
            let error = AlphaPngSequencePublisher::begin(&destination, request).unwrap_err();
            assert!(matches!(error, AlphaExportError::EffectAmbiguity(_)));
            assert!(!destination.exists());
        }
    }

    #[test]
    fn d5_black_fill_and_key_laws_cover_hostile_alpha_edges() {
        let fixture = hostile_fixture();
        let mut fill = vec![0; fixture.len()];
        derive_black_fill_rgba(&fixture, &mut fill).unwrap();
        let mut key = vec![0; fixture.len()];
        derive_key_rgba(&fixture, &mut key).unwrap();

        assert_eq!(&fill[0..4], &[0, 0, 0, 255]);
        assert_eq!(&fill[8..12], &[0, 0, 0, 255]);
        assert_eq!(&fill[24..28], &[0, 0, 0, 255]);
        assert_eq!(&fill[28..32], &[61, 127, 253, 255]);
        for (source, output) in fixture
            .as_chunks::<4>()
            .0
            .iter()
            .zip(key.as_chunks::<4>().0)
        {
            assert_eq!(output, &[source[3], source[3], source[3], 255]);
        }
    }

    #[test]
    fn d5_pinned_srgb_inverse_is_identity_at_opaque_alpha() {
        for code in 0_u8..=u8::MAX {
            let linear = crate::photosensitivity_advisor::srgb_code_to_linear_q(code);
            assert_eq!(linear_q_to_srgb8(linear), code);
        }
    }

    #[test]
    fn d5_png_generation_is_atomic_and_round_trips_hidden_rgb() {
        let directory = TestDirectory::new("d5-png");
        let destination = directory.0.join("generation");
        let fixture = hostile_fixture();
        let mut publisher = AlphaPngSequencePublisher::begin(
            &destination,
            plan(AlphaArtifactKind::StraightPngAndFillKey, 1),
        )
        .unwrap();
        publisher.write_frame(0, &fixture).unwrap();
        assert!(!destination.exists(), "partial generation became visible");
        let receipt = publisher.finish().unwrap();

        let rgba = image::open(destination.join("rgba/frame_00000000.png"))
            .unwrap()
            .into_rgba8()
            .into_raw();
        assert_eq!(rgba, fixture);
        let key = image::open(destination.join("key/frame_00000000.png"))
            .unwrap()
            .into_rgba8()
            .into_raw();
        assert_eq!(&key[0..4], &[0, 0, 0, 255]);
        assert_eq!(&key[4..8], &[127, 127, 127, 255]);
        assert_eq!(receipt.seam, PRE_OPAQUE_STRAIGHT_ALPHA_V1);
        assert_eq!(
            receipt.raw_straight_rgba_sha256,
            "39757AEAF173C4675FD703D990348E55EE5D8B6C5DDFF3497998C78545564C96"
        );
        let parsed: AlphaExportReceipt = serde_json::from_slice(
            &fs::read(destination.join("alpha-export-receipt.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(parsed, receipt);
    }

    #[test]
    fn d5_cancel_and_sequence_errors_publish_nothing() {
        let directory = TestDirectory::new("d5-cancel");
        let destination = directory.0.join("cancelled");
        let fixture = hostile_fixture();
        {
            let mut publisher = AlphaPngSequencePublisher::begin(
                &destination,
                plan(AlphaArtifactKind::StraightPngSequence, 2),
            )
            .unwrap();
            let error = publisher.write_frame(1, &fixture).unwrap_err();
            assert!(matches!(error, AlphaExportError::FrameOrdinal { .. }));
            publisher.write_frame(0, &fixture).unwrap();
        }
        assert!(!destination.exists());
        assert!(fs::read_dir(&directory.0).unwrap().all(|entry| !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".alpha-export-")));
    }

    #[test]
    fn d5_renderer_source_names_the_seam_before_the_opaque_boundary() {
        let export_source = include_str!("render_export.rs");
        let export_bind = export_source
            .find("pre_opaque_straight_alpha_v1_view(&composite_views)")
            .expect("offline renderer binds the named straight-alpha seam");
        let run = export_source
            .find("fn run_export(")
            .expect("offline export function");
        let opaque = export_source[run..]
            .find("crate::renderer::state::encode_opaque_output(")
            .map(|offset| run + offset)
            .expect("opaque audience boundary");
        assert!(export_bind < opaque);
        let helper = export_source
            .find("fn pre_opaque_straight_alpha_v1_view")
            .expect("named seam resolver");
        let helper_end = export_source[helper..]
            .find("\n}")
            .map(|offset| helper + offset)
            .expect("named seam resolver end");
        assert!(export_source[helper..helper_end].contains("&views[0]"));
        assert!(export_source.contains("fn readback_pre_opaque_straight_alpha_v1("));

        let live_source = include_str!("renderer/state.rs");
        let live_helper = live_source
            .find("fn pre_opaque_straight_alpha_v1_view")
            .expect("live named seam resolver");
        let live_helper_end = live_source[live_helper..]
            .find("\n}")
            .map(|offset| live_helper + offset)
            .expect("live named seam resolver end");
        assert!(live_source[live_helper..live_helper_end].contains("&views[0]"));
        assert!(live_source.contains("pre_opaque_straight_alpha_v1_view(&composite_views)"));
    }

    #[test]
    #[ignore = "requires the pinned external FFmpeg executable"]
    fn d5_ffv1_gbrap_round_trips_exact_rgba() {
        let ffmpeg_dir = std::env::var_os("FFMPEG_DIR").expect("FFMPEG_DIR");
        let ffmpeg = PathBuf::from(ffmpeg_dir).join("bin/ffmpeg.exe");
        let directory = TestDirectory::new("d5-ffv1");
        let destination = directory.0.join("generation");
        let first = hostile_fixture();
        let mut second = first.clone();
        second.rotate_left(4);
        let mut publisher =
            AlphaFfv1Publisher::begin(&ffmpeg, &destination, plan(AlphaArtifactKind::Ffv1Rgba, 2))
                .unwrap();
        publisher.write_frame(0, &first).unwrap();
        publisher.write_frame(1, &second).unwrap();
        assert!(!destination.exists());
        let receipt = publisher.finish().unwrap();
        assert_eq!(receipt.storage_pixel_format, "FFV1_v3_GBRAP");

        let decoded = Command::new(&ffmpeg)
            .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-i"])
            .arg(destination.join("plate.mkv"))
            .args([
                "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1",
            ])
            .output()
            .unwrap();
        assert!(
            decoded.status.success(),
            "{}",
            String::from_utf8_lossy(&decoded.stderr)
        );
        let expected = [first, second].concat();
        assert_eq!(decoded.stdout, expected);
    }
}
