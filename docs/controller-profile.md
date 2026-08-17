# Controller profile JSON

Controller profiles are separate from artistic patches. They describe bounded
MIDI input, typed application targets, and optional MIDI feedback. The default
operator document is `controller_profile.json` in the collide-o-scope per-user
state directory.

Use **Import JSON** / **Export JSON** in the Controller runtime panel, or use
**Ctrl+Shift+I** / **Ctrl+Shift+X** in the native window. Native commands own
their file pickers. Browser transfer uses the authenticated
`POST /controller-profile` endpoint with the closed pathless action shapes
`{"action":"import","document":{...}}` and `{"action":"export"}`; it can
never supply a host path or URL. Import validates the complete document,
resolves saved layer positions to the current stable layer IDs, durably
publishes the candidate, and installs the matching MIDI runtime as one manual
history transaction. A rejected import changes neither active protocol state
nor the saved document.

## Example

```json
{
  "version": 1,
  "name": "Venue controller",
  "input": { "mode": "exact", "name": "Venue MIDI In" },
  "output": { "mode": "exact", "name": "Venue MIDI Out" },
  "channel": { "mode": "omni" },
  "bindings": [
    {
      "id": 1,
      "source": { "kind": "control_change", "controller": 74 },
      "channel": { "mode": "exact", "channel": 1 },
      "encoding": "absolute",
      "button_mode": null,
      "press_threshold": 64,
      "relative_step": 0.007874016,
      "target": {
        "scope": "layer",
        "position": 0,
        "parameter": "opacity"
      },
      "feedback": {
        "kind": "control_change",
        "channel": 1,
        "controller": 74
      }
    }
  ]
}
```

`input` and `output` accept `{"mode":"first_available"}` or the exact form
shown above. The document-wide channel and an optional binding-local channel
accept `omni` or `exact` with channels 1–16.

Input sources are `control_change` or `note`, each with a 0–127 data number.
Encodings are `absolute`, `relative_twos_complement`,
`relative_binary_offset`, and `relative_sign_magnitude`. Notes require a
`button_mode` of `momentary`, `toggle`, or `gate`. Feedback is optional and is
either a three-byte Control Change or Note message with a 1–16 channel and
0–127 data number.

## Typed targets

Targets are internally tagged by `scope`:

- `legacy_midi_slot` with slot 0–3;
- `master` with a closed parameter;
- `layer` with a zero-based saved `position` and parameter;
- `group` with a nonzero `group_id` and parameter;
- `node` with a typed `node_scope`, nonzero `node_id`, and parameter;
- `transport` with a transport parameter.

Layer positions are resolved exactly once when the profile is installed.
Thereafter both input and feedback retain the resolved `StableLayerId` through
ordinary reorder; a missing saved position rejects the whole import rather
than retargeting.

The closed parameter vocabulary is:

```text
value amount wet bypass enabled opacity speed rate position_x position_y
scale_x scale_y rotation brightness contrast saturation hue threshold softness
visibility paused solo bus_crossfade program_freeze media_freeze blackout play
seek_normalized bpm tap_tempo downbeat clear_motion_memory clear_temporal_memory
```

Not every parameter is meaningful for every scope. Unsupported typed
scope/parameter pairs are diagnosed and do not fall through to arbitrary
string dispatch.

## Bounds and hostile input

- schema version: exactly 1;
- document: at most 256 KiB;
- browser action envelope: at most 257 KiB (document cap plus 1 KiB);
- profile name: 1–96 UTF-8 bytes, no control characters;
- bindings: at most 256 with unique nonzero IDs;
- exact device name: at most 256 UTF-8 bytes, no control characters;
- relative step: finite, greater than zero for relative encodings, at most 1;
- unknown fields, malformed MIDI data, duplicate IDs, missing layer positions,
  and unsupported enum values reject the complete document.

MIDI callbacks remain nonblocking. Raw/event/feedback queues and feedback rates
are bounded, malformed wire messages increment diagnostics without learning or
changing controls, and input-origin feedback is suppressed to avoid loops.
