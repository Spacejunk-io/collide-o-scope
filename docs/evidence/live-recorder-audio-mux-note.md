# Live recorder audio mux: evidence note

The most operator-visible deferral in the tree — live recordings were silent
and their receipts said `audio_not_muxed=true` — closed on the exact ladder
`docs/campaigns/known-open-work.md` ruled in advance: one owned Program PCM
clock, a bounded ring with an explicit underrun law, bounded drift
correction, and transactional cancel/failure publication, covering missing,
short, long, discontinuous, and device-loss audio.

## The design, compressed

**The clock is the recorder's, not the analyzer's.** `ProgramAudioTap`
(`src/program_recorder.rs`) is a bounded interleaved f32 ring whose
`delivered_frames` counter advances only when the capture device actually
delivers samples. The `AudioAnalyzer` callback tees the raw interleaved
device samples into an armed tap *before* its mono analysis downmix, so
analysis audio keeps its own independent ring and law — the capture stream is
only the source device, exactly as the ruling required. Arming without a
live stream is a refusal: there is no honest clock to fabricate.

**Every video frame is stamped with the clock.** The dormant
`RecorderFrameMetadata::audio_clock` field — retained since the recorder
foundation landed — is now fed: `frame_intent` stamps each capture with the
tap's delivered-frame position at render time. The worker anchors audio file
position zero to the first accepted frame's stamp, discarding pre-anchor
audio, so container time zero means the same instant on both timelines by
construction.

**The underrun law is explicit, ordered, and counted.** Ring overflow drops
the *oldest* frames and advances the ring head, so the single reader observes
the exact discarded span as a gap and writes that many frames of explicit
silence in the correct position — never a silent timeline shift. An armed
source that never delivers publishes a fully explicit zeroed PCM timeline
(the mux never sees an empty raw input). Device loss — stream error, stream
stop, stream restart, or channel-layout change — marks the tap lost; the
recording completes with the remainder padded and the report saying so.

**Drift correction is bounded and measured at the source.** The stamp taken
at frame `k`'s capture intent is compared against the program capture
cadence (`expected = k · rate / fps`); because both quantities are
render-thread facts, the measurement is immune to worker or encoder lag. A
slip past a quarter second fires one bounded correction — counted silence
insertion for a slow device clock, a counted drop for a fast one — and the
terminal boundary is exact regardless: the mux pads and trims to the exact
rational CFR duration of the encoded video.

**The mux is the export audio law inside the recorder's transaction.** Audio
PCM stages into a sibling temp beside the video temp; at finish one bounded,
supervised ffmpeg invocation copies the video stream (`-c:v copy`) and
encodes AAC with export's own
`asetpts=PTS-STARTPTS,apad,atrim=end=<duration>` + `-t <duration>` law into a
mux temp, which enters the existing linearized no-replace commit pair.
Cancel kills the helper and the temp guard removes every staged file; a mux
failure fails the recording with the bounded captured stderr — never a
silently degraded artifact. A recording with no audio source keeps the exact
prior video-only pipeline (`-an`, no mux process, `audio_not_muxed=true`).

**The report is the receipt.** Schema version 3 adds an `audio` block —
device, rate, channels, anchor position, captured frames, silence-gap
frames, both drift counters, `device_lost`, `capture_truncated`, and the
muxed duration — and `audio_not_muxed` is now computed, not constant.
`AppSnapshot`'s existing `audio_not_muxed` field turned truthful with zero
wire-protocol change.

## Bounds

| Bound | Value |
|---|---:|
| Ring retention | min(4 s at device rate, 2,097,152 samples) |
| Staged PCM temp cap | 2 GiB (reaching it marks the capture truncated) |
| Drift-correction threshold | 0.25 s, one bounded correction per trip |
| Finish tail grace | 500 ms, skipped on device loss |
| Mux helper deadline | 30 s absolute, cancel-aware, killed and reaped |

## The measurement

Hosted (all platforms, CLI-free): tap clock/overflow/anchor/sanitize laws,
the exact mux argument shape, anchored-PCM delivery into a fixture mux with
the exact rational duration, the video-only path proven mux-free, device
loss publishing with honest padding truth, armed-but-silent publishing
explicit full silence, cancel removing all four staged temps, mux failure
failing the recording with clean temps, and the drift law's direction,
bounds, and self-accounting — 10 new tests beside the existing recorder
battery, all green.

Opt-in, run on this host (Windows, FFmpeg 9.0.1 CLI):
`recorder_audio_mux_end_to_end_duration_and_offset_verified_by_ffprobe` — a
real 3-second 90-frame recording of a clocked ramp signal through the real
libx264 sink and the real AAC mux, verified by ffprobe: an `aac` audio
stream present, video duration within 0.05 s of 3.0, audio duration within
0.2 s of 3.0 (AAC priming tolerance), and A/V start offset under 0.06 s.
Passed.

What is deliberately not claimed: no resampling drift correction (the
bounded insert/drop law plus exact terminal trim is the whole claim, and the
report counts every correction); no physical audio-interface proof — a real
device driving a live recording belongs to the hardware-matrix campaign; no
change to offline export's independent audio policy; and layer/group/still
captures carry the same law with no target-specific claims.

Capability registry: `live_recorder_audio_mux` moves Deferred → Implemented
with the truthful limitation that audio is muxed exactly when the live audio
capture stream is running at recording start. The campaign row moves
`deferred/owned_program_pcm` → `implemented/complete`.

Gate: fmt, both node checks, check, tests, clippy `-D warnings` — the exact
CI form, run on the final tree before commit.
