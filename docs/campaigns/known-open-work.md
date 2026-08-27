# Known-open evidence campaigns

These campaigns are deliberately separate from the post-v1.6 improvement
tranches. Their status cannot be upgraded by a unit test or a new label.

## Exact PTS-driven VFR live transport

Build a bounded timestamp timeline with explicit repeated/non-monotonic PTS,
loop, reverse, and source-time selection laws. Preserve CFR hashes. Admit only
when irregular-PTS frame identities match an independent FFmpeg reference in
live and export. Current status: Deferred; average-FPS cadence remains truthful.

## Live recorder audio mux

The ruled ladder landed whole: one owned Program PCM clock (the recorder's
`ProgramAudioTap`, teed from the capture callback before the analysis
downmix — analysis audio remains its own ring), a bounded ring whose overflow
is recovered by the reader as an explicit counted silence gap, bounded
drift correction against the per-frame clock stamps, and the mux inside the
existing transactional no-replace publication with cancel/failure cleanup.
Missing audio keeps the exact video-only path; short and long audio are
padded/trimmed to the exact CFR duration by the export audio law; discontinuous
and device-loss audio become counted silence in the durable report. Current
status: Implemented; evidence in
`docs/evidence/live-recorder-audio-mux-note.md`, physical audio-interface proof
remains with the hardware matrix below.

## Physical venue and hardware matrix

Run real phone gyro/touch/Wi-Fi, MIDI feedback, audio input, Spout sender and
receiver, multi-monitor StageMap/fullscreen, refresh mismatch/VRR, and the one
hour mixed-effect soak. Each receipt records build identity, host facts,
adapter/driver, display path, device identity digest, fixture, duration, seed,
pass/fail, and bounded redacted failure. Current status: External proof required.

## Capture/NDI/Syphon/zero-copy/mesh/full-16

Reopen only per named platform, SDK/license/network authorization,
interoperability, resource, pixel, and performance gates. D3D11VA and full-16
history retain their measured non-promotion. No general RAM/VRAM increase,
automatic proxy/quality degradation, per-layer codec/VHS workers, speculative
thread increase, shader approximation, or variant explosion is authorized.
