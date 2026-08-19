//! The evaluation-only hardware decode session — Gate 4's first tranche.
//!
//! This module is the D3D11VA backend integration the capability evaluator
//! has been waiting for, in exactly the Gate 6 shape: **measurement-only**.
//! No production decode path constructs a session; the only constructor
//! calls are the opt-in interoperability probe (which regenerates the
//! tracked `docs/evidence/hw-decode-interop-receipt.json`) and the hosted
//! refusal tests. Landing it flips `backend_integrated` for hardware decode
//! on Windows, which moves the capability from
//! `Deferred(BackendNotIntegrated)` to
//! `EvaluationRequired(InteroperabilityProof)` — and deliberately no
//! further: `Available`, live usage, and the zero-copy path are separate
//! product decisions taken after reading the receipt. The HUD's
//! `hardware_decode_active` claim stays false throughout, derived through
//! the evaluator rather than edited here.
//!
//! The session decodes through FFmpeg's D3D11VA hwaccel — the same library
//! path a live integration would use, never the CLI — and downloads every
//! hardware surface to system memory with `av_hwframe_transfer_data`. That
//! download is the point for an evaluation session: the probe compares the
//! downloaded pixels against the pure software decode of the same stream,
//! so the receipt measures whether this host's hardware decoder agrees with
//! the reference before anyone builds product on it. A frame the decoder
//! declined to route through hardware is counted honestly as a software
//! fallback rather than folded into the hardware count.

// The session's only constructors are the opt-in interoperability probe and
// the hosted refusal tests — the S10a discipline: name the consumer honestly
// rather than fake a premature integration. `hardware_decode_backend_exists_
// on_this_platform` is the module's one production-alive item, consumed by
// `precision::probe_capability_evidence`.
#![allow(
    dead_code,
    reason = "probe-only until the operator opens the live hardware-decode tranche"
)]

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

#[cfg(target_os = "windows")]
use ffmpeg_next as ffmpeg;

/// Typed refusals. `PlatformUnsupported` is the entire non-Windows story:
/// the backend is D3D11VA and does not pretend otherwise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HardwareDecodeError {
    PlatformUnsupported,
    Open(String),
    NoVideoStream,
    /// `av_hwdevice_ctx_create` failed (no adapter, no video engine, remote
    /// session, …). Carries FFmpeg's negative status for the receipt.
    DeviceUnavailable(i32),
    DecoderOpen(String),
    Decode(String),
    /// A hardware surface could not be downloaded to system memory.
    Transfer(i32),
    /// The stream ended before a single frame decoded — malformed input,
    /// mirroring the software decoder's zero-frame refusal.
    NoFrames,
}

impl std::fmt::Display for HardwareDecodeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PlatformUnsupported => {
                formatter.write_str("hardware decode backend is D3D11VA and requires Windows")
            }
            Self::Open(error) => write!(formatter, "hardware decode open: {error}"),
            Self::NoVideoStream => formatter.write_str("hardware decode: no video stream"),
            Self::DeviceUnavailable(status) => write!(
                formatter,
                "D3D11VA device unavailable (ffmpeg status {status})"
            ),
            Self::DecoderOpen(error) => {
                write!(formatter, "hardware decoder open: {error}")
            }
            Self::Decode(error) => write!(formatter, "hardware decode: {error}"),
            Self::Transfer(status) => write!(
                formatter,
                "hardware frame download failed (ffmpeg status {status})"
            ),
            Self::NoFrames => {
                formatter.write_str("hardware decode produced no frames before end of stream")
            }
        }
    }
}

impl std::error::Error for HardwareDecodeError {}

/// A hostile stream cannot make the probe scan packets forever — the same
/// practical bound the software decoder holds.
#[cfg(target_os = "windows")]
const MAX_PACKETS_WITHOUT_FRAME: u32 = 4_096;

/// FFmpeg calls this to pick the decode pixel format. Choosing
/// `AV_PIX_FMT_D3D11` is what routes decoding through the hardware device;
/// when the list does not offer it the first entry keeps the codec on its
/// ordinary software path, which the session then counts as fallback.
///
/// SAFETY contract (FFmpeg's, upheld here): `formats` is a non-null,
/// `AV_PIX_FMT_NONE`-terminated array valid for the duration of the call.
#[cfg(target_os = "windows")]
unsafe extern "C" fn pick_d3d11_format(
    _context: *mut ffmpeg::ffi::AVCodecContext,
    formats: *const ffmpeg::ffi::AVPixelFormat,
) -> ffmpeg::ffi::AVPixelFormat {
    let mut cursor = formats;
    let mut first = ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_NONE;
    // SAFETY: bounded by the documented NONE terminator.
    unsafe {
        while *cursor != ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_NONE {
            if *cursor == ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_D3D11 {
                return ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_D3D11;
            }
            if first == ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_NONE {
                first = *cursor;
            }
            cursor = cursor.add(1);
        }
    }
    first
}

/// Our own reference on the hardware device context. The codec context holds
/// its own independent reference; this one exists so an open that fails
/// after device creation cannot leak, and it is released on drop.
#[cfg(target_os = "windows")]
struct HardwareDeviceRef(*mut ffmpeg::ffi::AVBufferRef);

#[cfg(target_os = "windows")]
impl Drop for HardwareDeviceRef {
    fn drop(&mut self) {
        // SAFETY: the pointer came from `av_hwdevice_ctx_create` and is
        // released exactly once here; FFmpeg tolerates the by-then-null ptr.
        unsafe { ffmpeg::ffi::av_buffer_unref(&mut self.0) }
    }
}

#[cfg(target_os = "windows")]
pub struct HardwareDecodeSession {
    input: ffmpeg::format::context::Input,
    stream_index: usize,
    decoder: ffmpeg::decoder::Video,
    _device: HardwareDeviceRef,
    /// Frames that decoded on the D3D11VA device and were downloaded.
    pub hardware_frames: u64,
    /// Frames the codec declined to route through hardware. Counted
    /// separately and honestly — never folded into the hardware count.
    pub software_fallback_frames: u64,
    flushed: bool,
}

#[cfg(target_os = "windows")]
impl HardwareDecodeSession {
    /// Open `path`'s best video stream with a freshly created D3D11VA
    /// device. Uses the same bounded, cancellation-aware open the software
    /// decoder uses; a host without a usable device is a typed refusal
    /// before any decode work.
    pub fn open(path: &str, cancel: Arc<AtomicBool>) -> Result<Self, HardwareDecodeError> {
        let input = super::decoder::open_input(path, cancel, "hardware decode probe")
            .map_err(HardwareDecodeError::Open)?;
        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or(HardwareDecodeError::NoVideoStream)?;
        let stream_index = stream.index();
        let mut context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|error| HardwareDecodeError::DecoderOpen(error.to_string()))?;

        // SAFETY: `Context` owns a live, uniquely borrowed AVCodecContext.
        // `hw_device_ctx` and `get_format` are public FFmpeg ABI written
        // before decoder open — the same narrow-ffi discipline as the
        // decoder core's EXPORT_MVS flag.
        let device = unsafe {
            let mut device: *mut ffmpeg::ffi::AVBufferRef = std::ptr::null_mut();
            let status = ffmpeg::ffi::av_hwdevice_ctx_create(
                &mut device,
                ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA,
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
            );
            if status < 0 || device.is_null() {
                return Err(HardwareDecodeError::DeviceUnavailable(status));
            }
            let device = HardwareDeviceRef(device);
            let raw = context.as_mut_ptr();
            (*raw).hw_device_ctx = ffmpeg::ffi::av_buffer_ref(device.0);
            if (*raw).hw_device_ctx.is_null() {
                return Err(HardwareDecodeError::DeviceUnavailable(status));
            }
            (*raw).get_format = Some(pick_d3d11_format);
            device
        };

        let decoder = context
            .decoder()
            .video()
            .map_err(|error| HardwareDecodeError::DecoderOpen(error.to_string()))?;
        Ok(Self {
            input,
            stream_index,
            decoder,
            _device: device,
            hardware_frames: 0,
            software_fallback_frames: 0,
            flushed: false,
        })
    }

    /// Decode the next frame, downloading hardware surfaces to system
    /// memory. `Ok(None)` is a clean end of stream.
    pub fn next_frame(
        &mut self,
    ) -> Result<Option<ffmpeg::util::frame::video::Video>, HardwareDecodeError> {
        let mut packets_without_frame = 0_u32;
        loop {
            let mut decoded = ffmpeg::util::frame::video::Video::empty();
            if self.decoder.receive_frame(&mut decoded).is_ok() {
                return self.finish_frame(decoded).map(Some);
            }
            if self.flushed {
                return Ok(None);
            }
            let mut sent = false;
            for (stream, packet) in self.input.packets() {
                if stream.index() != self.stream_index {
                    continue;
                }
                self.decoder
                    .send_packet(&packet)
                    .map_err(|error| HardwareDecodeError::Decode(error.to_string()))?;
                sent = true;
                break;
            }
            if sent {
                packets_without_frame += 1;
                if packets_without_frame > MAX_PACKETS_WITHOUT_FRAME {
                    return Err(HardwareDecodeError::Decode(
                        "no frame within the packet bound".to_owned(),
                    ));
                }
                continue;
            }
            // End of stream: flush once, then drain remaining frames.
            self.decoder.send_eof().ok();
            self.flushed = true;
        }
    }

    fn finish_frame(
        &mut self,
        frame: ffmpeg::util::frame::video::Video,
    ) -> Result<ffmpeg::util::frame::video::Video, HardwareDecodeError> {
        // SAFETY: reading the owned frame's public `format` field.
        let is_hardware = unsafe {
            (*frame.as_ptr()).format == ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_D3D11 as i32
        };
        if !is_hardware {
            self.software_fallback_frames += 1;
            return Ok(frame);
        }
        let mut transferred = ffmpeg::util::frame::video::Video::empty();
        // SAFETY: source is a live hardware frame, target an empty frame the
        // call allocates; both uniquely owned here.
        let status = unsafe {
            ffmpeg::ffi::av_hwframe_transfer_data(transferred.as_mut_ptr(), frame.as_ptr(), 0)
        };
        if status < 0 {
            return Err(HardwareDecodeError::Transfer(status));
        }
        // SAFETY: copies timestamps/side data between the two live frames.
        unsafe {
            ffmpeg::ffi::av_frame_copy_props(transferred.as_mut_ptr(), frame.as_ptr());
        }
        self.hardware_frames += 1;
        Ok(transferred)
    }
}

/// The one cross-platform door. Non-Windows answers with the typed platform
/// refusal instead of pretending an absent backend exists — the same honesty
/// the capability evaluator's per-platform reason table publishes.
#[cfg(not(target_os = "windows"))]
pub struct HardwareDecodeSession;

#[cfg(not(target_os = "windows"))]
impl HardwareDecodeSession {
    pub fn open(_path: &str, _cancel: Arc<AtomicBool>) -> Result<Self, HardwareDecodeError> {
        Err(HardwareDecodeError::PlatformUnsupported)
    }
}

/// The one producer of the "does a hardware decode backend exist in this
/// tree, on this platform" fact. `precision::probe_capability_evidence`
/// answers `backend_integrated` for hardware decode from here, so the tree
/// change that integrated the backend is the same change that flipped it —
/// the seam that section always promised. Zero-copy deliberately does not
/// consume this: downloading frames is not a zero-copy path.
pub fn hardware_decode_backend_exists_on_this_platform() -> bool {
    cfg!(target_os = "windows")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_file_is_a_typed_open_refusal_or_a_platform_refusal() {
        let result = HardwareDecodeSession::open(
            "definitely-missing-hardware-probe-input.mp4",
            Arc::new(AtomicBool::new(false)),
        );
        match result {
            Err(HardwareDecodeError::Open(_)) if cfg!(target_os = "windows") => {}
            Err(HardwareDecodeError::PlatformUnsupported) if !cfg!(target_os = "windows") => {}
            other => panic!("expected a typed refusal, got {:?}", other.err()),
        }
    }

    /// Every frame of the audit clip decoded in software — the pure
    /// reference the hardware session is judged against, as raw 4:2:0 YUV
    /// planes so no format conversion can launder or invent a difference.
    #[cfg(target_os = "windows")]
    fn software_reference_frames(path: &str) -> Vec<YuvPlanes> {
        use ffmpeg::media::Type;

        let mut input = super::super::decoder::open_input(
            path,
            Arc::new(AtomicBool::new(false)),
            "hardware interop software reference",
        )
        .expect("software reference open");
        let stream = input.streams().best(Type::Video).expect("video stream");
        let stream_index = stream.index();
        let context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .expect("software reference context");
        let mut decoder = context.decoder().video().expect("software decoder");
        let mut frames = Vec::new();
        let drain = |decoder: &mut ffmpeg::decoder::Video, frames: &mut Vec<YuvPlanes>| loop {
            let mut decoded = ffmpeg::util::frame::video::Video::empty();
            if decoder.receive_frame(&mut decoded).is_err() {
                break;
            }
            frames.push(yuv_planes(&decoded));
        };
        for (stream, packet) in input.packets() {
            if stream.index() != stream_index {
                continue;
            }
            decoder.send_packet(&packet).expect("software send");
            drain(&mut decoder, &mut frames);
        }
        decoder.send_eof().ok();
        drain(&mut decoder, &mut frames);
        frames
    }

    #[cfg(target_os = "windows")]
    struct YuvPlanes {
        y: Vec<u8>,
        u: Vec<u8>,
        v: Vec<u8>,
    }

    /// Tightly packed 4:2:0 planes from either layout the two paths produce:
    /// planar `yuv420p` from the software decoder, interleaved-chroma `nv12`
    /// from the hardware download. Stride padding is dropped; nothing else
    /// is touched.
    #[cfg(target_os = "windows")]
    fn yuv_planes(frame: &ffmpeg::util::frame::video::Video) -> YuvPlanes {
        let width = frame.width() as usize;
        let height = frame.height() as usize;
        let chroma_width = width.div_ceil(2);
        let chroma_height = height.div_ceil(2);
        let packed = |data: &[u8], stride: usize, row_bytes: usize, rows: usize| {
            let mut bytes = Vec::with_capacity(row_bytes * rows);
            for row in 0..rows {
                let start = row * stride;
                bytes.extend_from_slice(&data[start..start + row_bytes]);
            }
            bytes
        };
        let y = packed(frame.data(0), frame.stride(0), width, height);
        match frame.format() {
            ffmpeg::format::Pixel::NV12 => {
                let uv = packed(
                    frame.data(1),
                    frame.stride(1),
                    chroma_width * 2,
                    chroma_height,
                );
                let mut u = Vec::with_capacity(chroma_width * chroma_height);
                let mut v = Vec::with_capacity(chroma_width * chroma_height);
                for pair in uv.chunks_exact(2) {
                    u.push(pair[0]);
                    v.push(pair[1]);
                }
                YuvPlanes { y, u, v }
            }
            ffmpeg::format::Pixel::YUV420P => YuvPlanes {
                y,
                u: packed(frame.data(1), frame.stride(1), chroma_width, chroma_height),
                v: packed(frame.data(2), frame.stride(2), chroma_width, chroma_height),
            },
            other => panic!("unexpected interop frame format {other:?}"),
        }
    }

    #[cfg(target_os = "windows")]
    fn plane_delta(observed: &[u8], reference: &[u8]) -> (u8, u64) {
        assert_eq!(observed.len(), reference.len());
        let mut max = 0_u8;
        let mut differing = 0_u64;
        for (observed, reference) in observed.iter().zip(reference) {
            let delta = observed.abs_diff(*reference);
            max = max.max(delta);
            differing += u64::from(delta != 0);
        }
        (max, differing)
    }

    /// Gate 4's interoperability measurement: the D3D11VA session against
    /// the pure software decode of the same stream, frame for frame, through
    /// one RGBA conversion — regenerating the tracked
    /// `docs/evidence/hw-decode-interop-receipt.json` in place (the
    /// S2-receipt law: a changed receipt after an opt-in run is a new
    /// measurement on new hardware; commit it). The wall timings are
    /// fixture-local smoke observations with a full download per frame, not
    /// a renderer-throughput comparison.
    #[cfg(target_os = "windows")]
    #[test]
    #[ignore = "requires a Windows D3D11VA device and videos/audit.mp4; emits the Gate 4 interop receipt"]
    fn hw_decode_interop_probe_measures_agreement_and_writes_the_receipt() {
        use std::time::Instant;

        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );

        let software_start = Instant::now();
        let software = software_reference_frames("videos/audit.mp4");
        let software_ns = software_start.elapsed().as_nanos() as u64;
        assert!(!software.is_empty(), "software reference decoded no frames");

        let mut session =
            HardwareDecodeSession::open("videos/audit.mp4", Arc::new(AtomicBool::new(false)))
                .expect("D3D11VA session open");
        let hardware_start = Instant::now();
        let mut hardware = Vec::new();
        while let Some(frame) = session.next_frame().expect("hardware decode") {
            hardware.push(yuv_planes(&frame));
        }
        let hardware_ns = hardware_start.elapsed().as_nanos() as u64;

        // The session must have actually decoded on the device — a receipt
        // recording an all-software run would be a lie with a green check.
        assert!(
            session.hardware_frames > 0,
            "no frame decoded through D3D11VA: fallbacks={}",
            session.software_fallback_frames
        );
        assert_eq!(
            hardware.len(),
            software.len(),
            "hardware and software paths decoded different frame counts"
        );

        // Raw decoded 4:2:0 planes, no conversion in between: this is the
        // decoder-agreement claim itself. H.264 decoding is spec-exact, so a
        // conforming device must match the software reference byte for byte;
        // a deviation is a finding the receipt must not paper over.
        let mut max_luma_delta = 0_u8;
        let mut max_chroma_delta = 0_u8;
        let mut differing_samples = 0_u64;
        let mut total_samples = 0_u64;
        for (hardware_frame, software_frame) in hardware.iter().zip(&software) {
            let (y_max, y_diff) = plane_delta(&hardware_frame.y, &software_frame.y);
            let (u_max, u_diff) = plane_delta(&hardware_frame.u, &software_frame.u);
            let (v_max, v_diff) = plane_delta(&hardware_frame.v, &software_frame.v);
            max_luma_delta = max_luma_delta.max(y_max);
            max_chroma_delta = max_chroma_delta.max(u_max).max(v_max);
            differing_samples += y_diff + u_diff + v_diff;
            total_samples +=
                (hardware_frame.y.len() + hardware_frame.u.len() + hardware_frame.v.len()) as u64;
        }
        assert_eq!(
            (max_luma_delta, max_chroma_delta),
            (0, 0),
            "hardware decode disagreed with the spec-exact software reference \
             (luma max {max_luma_delta}, chroma max {max_chroma_delta}, \
             {differing_samples} of {total_samples} samples)"
        );

        let git = |args: &[&str]| {
            std::process::Command::new("git")
                .args(args)
                .output()
                .ok()
                .filter(|output| output.status.success())
                .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
                .unwrap_or_else(|| "unknown".to_string())
        };
        let receipt = serde_json::json!({
            "schema": "collide-o-scope-hw-decode-interop-receipt/1",
            "command": "cargo test --locked hw_decode_interop_probe_measures_agreement_and_writes_the_receipt -- --ignored --nocapture",
            "measured_at": {
                "commit": git(&["rev-parse", "HEAD"]),
                "branch": git(&["rev-parse", "--abbrev-ref", "HEAD"]),
                "tree": match git(&["status", "--porcelain"]).as_str() {
                    "unknown" => "unknown",
                    "" => "clean",
                    _ => "dirty",
                },
            },
            "host": {
                "os": std::env::consts::OS,
                "arch": std::env::consts::ARCH,
                "backend": "D3D11VA via FFmpeg av_hwdevice_ctx_create on the default adapter",
                "avcodec_version": ffmpeg::codec::version(),
                "avformat_version": ffmpeg::format::version(),
            },
            "scope": "evaluation-only: the session is constructed by this probe alone; the capability stops at EvaluationRequired, hardware_decode_active stays false, and live usage plus zero-copy remain separate operator-decided tranches",
            "source": "videos/audit.mp4 (untracked local clip; identity travels via frame counts and agreement, not bytes)",
            "frames": {
                "software_reference": software.len(),
                "hardware_decoded": session.hardware_frames,
                "software_fallback_in_hardware_session": session.software_fallback_frames,
            },
            "agreement": {
                "comparison": "raw decoded 4:2:0 planes, hardware download (NV12, de-interleaved) against software decode (yuv420p), no format conversion in between",
                "max_luma_delta": max_luma_delta,
                "max_chroma_delta": max_chroma_delta,
                "differing_samples": differing_samples,
                "total_samples": total_samples,
            },
            "timing_scope": "fixture-local wall time; the hardware loop downloads every frame to system memory, which a live integration would avoid — smoke evidence only",
            "software_wall_ns_per_frame": software_ns / software.len() as u64,
            "hardware_download_wall_ns_per_frame": hardware_ns / hardware.len().max(1) as u64,
        });
        std::fs::write(
            "docs/evidence/hw-decode-interop-receipt.json",
            format!("{}\n", serde_json::to_string_pretty(&receipt).unwrap()),
        )
        .expect("write the hardware decode interop receipt");
        println!(
            "HW_DECODE_INTEROP_RECEIPT={}",
            serde_json::to_string_pretty(&receipt).unwrap()
        );
    }
}
