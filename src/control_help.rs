//! B15 per-control help: one static table, two consumers.
//!
//! The panel needs help text to search over and to show; the native patch
//! parameter editor needs the same sentences as hover tooltips. Two hand-kept
//! copies would drift the first time a law changed, so the table lives here
//! and the browser's copy is *generated* from it by
//! [`panel_javascript`] and served as `help.js` — the shared-parse-table law
//! the wire vocabularies already follow, applied to prose.
//!
//! House voice: what the control does, and why it behaves the way it does.
//! The second half is the part that is hard to recover from the code, so it
//! is the part worth writing down. Where a law is stated exactly by the module
//! that owns it, the text states it exactly; where behaviour is a matter of
//! degree it stays
//! general rather than inventing a precision the engine does not promise.
//!
//! The table is keyed by the **wire parameter name**, which is also the key
//! the native editor's rows carry, so a single lookup serves both surfaces.

/// The authoring surface a help entry belongs to. These are the panel's
/// `data-*` row families, so a row can look up its own help without a
/// translation step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelpScope {
    /// `data-param` — the master effect block (and the identically named
    /// per-layer controls, which share every law except the master-only
    /// optics).
    Master,
    /// `data-temporal` — the temporal section, including the B3 rig, B4
    /// display physics, B8 melting edge, B5 codec mosh, and B14 sync latch.
    Temporal,
    /// `data-ntsc` — the ntsc-rs VHS emulation block.
    Ntsc,
}

impl HelpScope {
    /// The key this scope uses in the generated panel table.
    pub fn key(self) -> &'static str {
        match self {
            Self::Master => "master",
            Self::Temporal => "temporal",
            Self::Ntsc => "ntsc",
        }
    }

    pub const ALL: [Self; 3] = [Self::Master, Self::Temporal, Self::Ntsc];
}

pub struct HelpEntry {
    pub scope: HelpScope,
    pub param: &'static str,
    pub text: &'static str,
}

const fn entry(scope: HelpScope, param: &'static str, text: &'static str) -> HelpEntry {
    HelpEntry { scope, param, text }
}

/// Look one control's help up by scope and wire name.
pub fn help_for(scope: HelpScope, param: &str) -> Option<&'static str> {
    CONTROL_HELP
        .iter()
        .find(|entry| entry.scope == scope && entry.param == param)
        .map(|entry| entry.text)
}

/// Look a control up by wire name alone, trying every scope in order. The
/// native editor's rows carry a bare key, and the master block is the one it
/// edits, so this resolves the way that editor needs.
pub fn help_for_any(param: &str) -> Option<&'static str> {
    HelpScope::ALL
        .iter()
        .find_map(|scope| help_for(*scope, param))
}

/// Emit the browser's copy of the table as a frozen JavaScript object. This
/// is served as `help.js` and is the *only* copy the panel sees, so the two
/// surfaces cannot disagree about what a control does.
pub fn panel_javascript() -> String {
    let mut out = String::from(
        "// Generated from src/control_help.rs at request time. Do not edit by\n\
         // hand: the Rust table is the single source, and the native patch\n\
         // editor's tooltips read the same entries.\n\
         window.CONTROL_HELP = Object.freeze({\n",
    );
    for scope in HelpScope::ALL {
        out.push_str("  ");
        out.push_str(scope.key());
        out.push_str(": Object.freeze({\n");
        for item in CONTROL_HELP.iter().filter(|entry| entry.scope == scope) {
            out.push_str("    \"");
            out.push_str(item.param);
            out.push_str("\": \"");
            escape_into(item.text, &mut out);
            out.push_str("\",\n");
        }
        out.push_str("  }),\n");
    }
    out.push_str("});\n");
    out
}

/// Escape a help sentence for a double-quoted JavaScript string literal.
/// Control characters are escaped rather than passed through, so a stray
/// newline in a future entry cannot break the served asset.
fn escape_into(text: &str, out: &mut String) {
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '<' => out.push_str("\\u003c"),
            '>' => out.push_str("\\u003e"),
            '&' => out.push_str("\\u0026"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
}

use HelpScope::{Master, Ntsc, Temporal};

pub static CONTROL_HELP: &[HelpEntry] = &[
    // ---------------------------------------------------------------- master
    entry(
        Master,
        "pixelate",
        "Quantises the image into square blocks, sized in source pixels. Because the size is in source pixels rather than output ones, a scaled or downsampled layer blocks at a different visual size than the master does.",
    ),
    entry(
        Master,
        "rgb_split",
        "Slides red and blue apart in opposite directions by a pixel distance while green stays put. Luminance therefore survives almost intact and only the edges fringe.",
    ),
    entry(
        Master,
        "hue_shift",
        "Rotates every colour around the wheel, in degrees. The value wraps, so +180 and -180 are the same place, and a Morph between two hues takes the shortest arc rather than the long way round.",
    ),
    entry(
        Master,
        "saturation",
        "Pushes colour away from grey or toward it. The maths runs in linear light, so pushing saturation up brightens saturated areas less than a gamma-space control would.",
    ),
    entry(
        Master,
        "brightness",
        "Adds or subtracts light. It is an offset rather than a multiply, so it lifts shadows as much as highlights and will flatten contrast if pushed far.",
    ),
    entry(
        Master,
        "contrast",
        "Expands or compresses the range around mid grey. Values beyond the middle clip, and clipping in linear light crushes colour toward the primaries rather than toward white.",
    ),
    entry(
        Master,
        "posterize",
        "Snaps each channel to a limited number of levels. Because the channels quantise independently, flat areas band into colour steps rather than grey ones.",
    ),
    entry(
        Master,
        "invert",
        "Flips the image to its photographic negative. See Negative for the modes that invert luma or hue alone.",
    ),
    entry(
        Master,
        "downsample",
        "Renders through a reduced-resolution sample before scaling back up. Unlike Pixelate this softens as well as blocks, because the upscale filters.",
    ),
    entry(
        Master,
        "shift_amount",
        "How far banded rows slide sideways. Zero takes an explicitly exact bypass branch in the shader, so an unshifted image is the historical sample byte for byte.",
    ),
    entry(
        Master,
        "shift_block_size",
        "The height of a shift band in output pixels, from 2 to 256. Bands are cut in output space, so the same setting bands identically whatever the source resolution.",
    ),
    entry(
        Master,
        "shift_density",
        "How many bands are displaced on any given step. Which bands fire is a hash of the band, the time epoch, and the master seed, so the arrangement is deterministic and repeats exactly on export.",
    ),
    entry(
        Master,
        "shift_speed",
        "How often the band pattern is redrawn. Time enters as a discrete epoch rather than continuously, so bands hold still between steps instead of sliding.",
    ),
    entry(
        Master,
        "cellular_amount",
        "Warps the image along a Worley cell field. At exactly zero the 3x3 cell search is skipped entirely rather than computed and discarded, so an unused Cellular costs nothing.",
    ),
    entry(
        Master,
        "cellular_scale",
        "How many cells span the frame. Feature points stay bounded inside their own cell, so raising the scale adds detail without letting cells overlap unpredictably.",
    ),
    entry(
        Master,
        "cellular_warp",
        "How strongly the cell field bends the sampled coordinate, as distinct from how much of the warp is mixed in.",
    ),
    entry(
        Master,
        "cellular_speed",
        "How fast the cell field drifts. Its clock is anchored to patch generation rather than wall time, so a reload replays the same drift.",
    ),
    entry(
        Master,
        "cellular_gap_amount",
        "Opens the cell boundaries into transparent gaps. This is a real coverage cut, not a darkening: at layer scope it reveals the stack beneath, and at master scope it resolves over black.",
    ),
    entry(
        Master,
        "cellular_gap_threshold",
        "How close to a cell wall counts as gap. Together with softness this sets how wide the cut runs.",
    ),
    entry(
        Master,
        "cellular_gap_softness",
        "Feathers the gap edge. Coverage stays straight-alpha through the whole chain and is flattened exactly once at the end, so a soft gap composites correctly rather than fringing.",
    ),
    entry(
        Master,
        "grain_intensity",
        "Adds film-style noise. The pattern follows the master seed, so two renders of the same patch grain identically.",
    ),
    entry(
        Master,
        "grain_size",
        "How coarse each grain is. Larger grains read as film stock, smaller ones as sensor noise.",
    ),
    entry(
        Master,
        "grain_algo",
        "Chooses the noise generator. It is a discrete choice, so a Morph recalls one endpoint at the midpoint rather than blending two noise fields into a third.",
    ),
    entry(
        Master,
        "color_grain",
        "Lets the grain carry colour instead of moving all three channels together. Monochrome grain reads as exposure; coloured grain reads as electronics.",
    ),
    entry(
        Master,
        "vignette",
        "Darkens toward the corners. It is applied in linear light, so it dims without shifting hue the way a gamma-space vignette does.",
    ),
    entry(
        Master,
        "color_drift",
        "Slowly rotates colour balance over time. A slow drift keeps a long take from looking static without reading as an effect.",
    ),
    entry(
        Master,
        "breathe_scale",
        "Gently pulses the image size. Breathing is a coordinate effect, so it resamples rather than re-renders and stays cheap.",
    ),
    entry(
        Master,
        "breathe_rotation",
        "Gently rocks the image angle. The rotation is conjugated through output aspect, so the physical angle is correct on non-square outputs.",
    ),
    entry(
        Master,
        "breathe_position",
        "Gently drifts the image position. Breathing moves the frame without touching the spatial transform, so it composes on top of an authored position rather than fighting it.",
    ),
    entry(
        Master,
        "key_mode",
        "Chooses what the static key removes: nothing, bright, dark, a chroma match, or everything but a chroma match. The shader outputs straight RGB with a modified alpha and the compositor does the premultiply, so keyed edges do not darken.",
    ),
    entry(
        Master,
        "key_threshold",
        "Where the luminance key opens. Only the luminance modes use it; the chroma modes use target and tolerance instead.",
    ),
    entry(
        Master,
        "key_softness",
        "Feathers the key edge in both the luminance and chroma modes.",
    ),
    entry(
        Master,
        "key_color_r",
        "The red component of the chroma key target.",
    ),
    entry(
        Master,
        "key_color_g",
        "The green component of the chroma key target. The default is pure green because that is the conventional screen colour.",
    ),
    entry(
        Master,
        "key_color_b",
        "The blue component of the chroma key target.",
    ),
    entry(
        Master,
        "key_tolerance",
        "How far a pixel may sit from the chroma target and still key. Widen it for uneven lighting; narrow it to protect colours near the screen hue.",
    ),
    entry(
        Master,
        "key_border",
        "Grows a fill border out of the key's own matte, the way a broadcast border generator adds fill to a key. A layer has no composite underneath it, so the border joins the key signal rather than compositing over anything.",
    ),
    entry(
        Master,
        "key_border_color",
        "Which bench colour the key border fills with. A closed eight-colour table, so it recalls an endpoint under Morph rather than interpolating.",
    ),
    entry(
        Master,
        "key_shadow",
        "Drops a darkened offset copy of the matte behind the key, giving the keyed shape a lift off whatever it sits on.",
    ),
    entry(
        Master,
        "contour",
        "Draws isolines between smoothed luma bands, so the image reads as a contour map. The line distance is measured with screen-space derivatives, which keeps line weight even as the image scales.",
    ),
    entry(
        Master,
        "contour_bands",
        "How many luma bands the contour lines separate. More bands means more lines, not thinner ones.",
    ),
    entry(
        Master,
        "contour_width",
        "How thick the contour lines are drawn. Width is measured in screen space, so lines keep their weight as the image scales.",
    ),
    entry(
        Master,
        "contour_hue",
        "Colours the contour lines. Near hue phase zero the lines go white, so a hue sweep passes through a bright band rather than wrapping abruptly.",
    ),
    entry(
        Master,
        "contour_fill",
        "How much of the banded fill shows between the lines, as opposed to the original image.",
    ),
    entry(
        Master,
        "flatten",
        "Quantises luma into solid fields, discarding the gradient between them.",
    ),
    entry(
        Master,
        "flatten_levels",
        "How many solid luma fields Flatten collapses to.",
    ),
    entry(
        Master,
        "contour_dither",
        "Applies an ordered 4x4 Bayer dither to the flattened fields, trading hard banding for structured texture.",
    ),
    entry(
        Master,
        "solarize",
        "Folds exposure back on itself past a point, so highlights reverse. This is the darkroom effect, not an inversion.",
    ),
    entry(
        Master,
        "negative",
        "Mixes toward the inverted image. What 'inverted' means is set by Negative Mode.",
    ),
    entry(
        Master,
        "negative_mode",
        "Chooses the inversion law: plain RGB, luma-only (which keeps hue), or hue-flip (which keeps luminance). A discrete law, so Morph recalls an endpoint at the midpoint.",
    ),
    entry(
        Master,
        "colourpass",
        "Keeps one hue window in colour and sends everything else to mono. The window is measured in YIQ, so it follows perceived hue rather than raw RGB proximity.",
    ),
    entry(
        Master,
        "colourpass_hue",
        "The centre of the hue window that survives, in degrees. It wraps, so a Morph takes the shortest arc.",
    ),
    entry(
        Master,
        "colourpass_width",
        "How wide the surviving hue window is. Narrow it to isolate a single colour; widen it until almost nothing goes mono.",
    ),
    entry(
        Master,
        "edge_amount",
        "Mixes in a Sobel outline taken from source luma. Because it reads luma rather than colour, it finds shape edges instead of colour boundaries.",
    ),
    entry(
        Master,
        "edge_hue",
        "Colours the detected edges, in degrees around the wheel.",
    ),
    entry(
        Master,
        "emboss",
        "Lights the image from one side using a directional difference, so it reads as relief.",
    ),
    entry(
        Master,
        "emboss_angle",
        "Which direction the emboss light comes from, in degrees.",
    ),
    entry(
        Master,
        "halftone",
        "Sizes dots by brightness on a rotatable screen. It runs before the colour adjustments, so the dots receive hue, saturation, and contrast rather than being stamped on afterwards.",
    ),
    entry(
        Master,
        "halftone_pitch",
        "How far apart the halftone dots sit. Pitch sets only the spacing; the dots themselves are sized by brightness.",
    ),
    entry(
        Master,
        "halftone_angle",
        "The screen angle of the halftone grid, in degrees.",
    ),
    entry(
        Master,
        "moire",
        "Interferes the image against a virtual grid. Its clock is the established effect time, so it holds still under Pause and replays on export.",
    ),
    entry(
        Master,
        "moire_freq",
        "How fine the interfering grid is. Small changes here move the pattern a long way, which is the nature of interference.",
    ),
    entry(
        Master,
        "row_smear",
        "Shears each row by a wrong-predictor offset, the way a corrupt scanline decoder does. It is a coordinate effect, so it resamples rather than smudging.",
    ),
    entry(
        Master,
        "bitcrush",
        "Quantises to a small number of monochrome levels with an ordered dither.",
    ),
    entry(
        Master,
        "bitcrush_levels",
        "How many levels survive the crush. Two is the classic one-bit look.",
    ),
    entry(
        Master,
        "bitcrush_dither",
        "How much ordered dither is mixed into the crush. Dither trades hard steps for texture.",
    ),
    entry(
        Master,
        "multi_grid_x",
        "Tiles the image horizontally, one to eight. Odd cells are mirrored so tiles meet cleanly at their seams.",
    ),
    entry(
        Master,
        "multi_grid_y",
        "Tiles the image vertically, one to eight, with the same mirrored-cell rule as the horizontal count.",
    ),
    entry(
        Master,
        "barrel",
        "Bends the frame outward or inward like a lens. Master only: a layer has no lens of its own, and the optic is refused at every layer authoring seam.",
    ),
    entry(
        Master,
        "chroma_aberration",
        "Scales each primary by a slightly different amount from the centre, so colour separates toward the edges the way a real lens separates it. Master only.",
    ),
    entry(
        Master,
        "anamorphic_streak",
        "Throws a horizontal flare off the highlights. It is blue because the coatings that cause the real thing are blue. Master only.",
    ),
    // -------------------------------------------------------------- temporal
    entry(
        Temporal,
        "feedback",
        "How much of the previous processed frame is fed back into this one. The value is retention per 1/30 second, so the trail length is the same whether the program runs at 24, 30, or 60 fps.",
    ),
    entry(
        Temporal,
        "fb_zoom",
        "Zooms the fed-back image each tick. Values above one push the trail outward into a tunnel; below one it collapses inward.",
    ),
    entry(
        Temporal,
        "fb_rotate",
        "Rotates the fed-back image each tick, in degrees. A steady rotation locks an impulse into arms; detuning it shears them off.",
    ),
    entry(
        Temporal,
        "fb_offset_x",
        "Shifts the fed-back image horizontally each tick, so the trail drifts sideways.",
    ),
    entry(
        Temporal,
        "fb_offset_y",
        "Shifts the fed-back image vertically each tick.",
    ),
    entry(
        Temporal,
        "fb_reflect_x",
        "Mirrors the fed-back sample horizontally. This is a regime no amount of rotation can reach: it produces a two-cycle alternation rather than a spin.",
    ),
    entry(
        Temporal,
        "fb_reflect_y",
        "Mirrors the fed-back sample vertically, with the same two-cycle character as the horizontal reflection.",
    ),
    entry(
        Temporal,
        "fb_hue_rotate",
        "Rotates hue inside the feedback loop, so each pass shifts colour further and the trail rainbows.",
    ),
    entry(
        Temporal,
        "fb_saturation",
        "Multiplies saturation inside the loop. Because it compounds every tick, small departures from one go a long way.",
    ),
    entry(
        Temporal,
        "fb_gain_r",
        "Multiplies the red channel inside the loop. Gains above one are allowed and are what the servo exists to contain.",
    ),
    entry(
        Temporal,
        "fb_gain_g",
        "Multiplies the green channel inside the loop.",
    ),
    entry(
        Temporal,
        "fb_gain_b",
        "Multiplies the blue channel inside the loop.",
    ),
    entry(
        Temporal,
        "fb_chroma_displace",
        "Displaces the loop's colour lookup away from its luma lookup, so colour trails at a different rate than brightness.",
    ),
    entry(
        Temporal,
        "fb_blur",
        "Softens the fed-back sample over fixed cross taps. Paired with Sharpen it forms an activator-inhibitor pair, which is what grows structure rather than mush.",
    ),
    entry(
        Temporal,
        "fb_sharpen",
        "Sharpens the fed-back sample over the same fixed taps Blur uses. Alone it hardens the trail; against Blur it generates pattern.",
    ),
    entry(
        Temporal,
        "fb_shape",
        "The waveshaper applied to the looped value: clamp, soft, wrap, or fold. Clamp at drive one is the exact identity, so the default loop is untouched.",
    ),
    entry(
        Temporal,
        "fb_drive",
        "How hard the loop is driven into the waveshaper before it folds or clips.",
    ),
    entry(
        Temporal,
        "fb_pivot",
        "The point the waveshaper folds or clips around.",
    ),
    entry(
        Temporal,
        "fb_threshold",
        "Decays light below this level out of the loop, so dim residue clears instead of accumulating into haze.",
    ),
    entry(
        Temporal,
        "fb_noise",
        "Adds deterministic noise inside the loop, keyed by pixel and reference tick. Deterministic because live and export must agree.",
    ),
    entry(
        Temporal,
        "fb_edge",
        "What the loop reads past the frame edge: transparent, mirror, wrap, or hold. Transparent is the exact historical inside test.",
    ),
    entry(
        Temporal,
        "fb_servo",
        "Engages a compressive auto-level on the loop, so gains above one settle instead of running away. It is deliberately per-pixel and deterministic rather than measured from the frame, because a measured servo would give live and export different dynamics.",
    ),
    entry(
        Temporal,
        "fb_servo_defeated",
        "Defeats the servo even while it is engaged. This is a failure switch: defeated, the loop may run to white or black and stay there. A model that always recovers cannot actually break.",
    ),
    entry(
        Temporal,
        "slitscan",
        "Replaces the current frame with samples taken from different depths of the history ring, so the picture is assembled from several moments at once.",
    ),
    entry(
        Temporal,
        "slit_angle",
        "The direction the time gradient runs across the frame, in degrees. Zero scans along Y and ninety along X; older row-and-column patches map onto those two.",
    ),
    entry(
        Temporal,
        "slit_map",
        "How image position becomes a history age: a plain ramp, image brightness, distance from centre, a per-scanline sawtooth, or a slow horizontal sweep. Ramp is the exact historical path.",
    ),
    entry(
        Temporal,
        "slit_interp",
        "Interpolates between the two adjacent history layers instead of snapping to one. Off is the exact banded floor law and costs one fewer history read.",
    ),
    entry(
        Temporal,
        "key_mode",
        "Which change the temporal key keeps: motion, stillness, brightening, or darkening. It compares the current clean composite against a chosen history frame.",
    ),
    entry(
        Temporal,
        "key_threshold",
        "How much change is needed before the temporal key opens.",
    ),
    entry(
        Temporal,
        "key_softness",
        "Feathers the temporal key's edge, so a moving subject enters and leaves the mask gradually instead of popping in.",
    ),
    entry(
        Temporal,
        "key_history",
        "How far back the comparison frame sits, in 30 Hz history frames. The ring holds twenty-four, so the reachable past is about eight tenths of a second.",
    ),
    entry(
        Temporal,
        "long_exposure_amount",
        "Blends in a photographic average of the clean current frame and its recent 30 Hz history. Zero is an exact bypass.",
    ),
    entry(
        Temporal,
        "long_exposure_frames",
        "Sets the shutter span from two to twenty-four frames. Spans through eight are exact; longer spans keep the full trail extent with eight bounded samples.",
    ),
    entry(
        Temporal,
        "loom_amount",
        "How strongly the Loom rewrites the history read into its own topology.",
    ),
    entry(Temporal, "loom_topology", "The geometry the Loom folds history through: linear, radial, spiral, contour, folded, or kaleidoscopic."),
    entry(Temporal, "loom_interpolation", "Whether the Loom's history read snaps to a layer or interpolates between two."),
    entry(Temporal, "loom_depth", "How deep into the history ring the Loom reaches."),
    entry(Temporal, "loom_phase", "Where the Loom's pattern sits along its own cycle."),
    entry(Temporal, "loom_scale", "How large the Loom's geometry is relative to the frame."),
    entry(Temporal, "loom_angle", "The Loom geometry's rotation, in degrees."),
    entry(Temporal, "loom_folds", "How many repeats the Loom's fold produces."),
    entry(Temporal, "loom_quantization", "Snaps the Loom's history read to discrete steps; zero leaves it continuous."),
    entry(Temporal, "atlas_amount", "How strongly the Atlas partitions the frame into independently-timed territories."),
    entry(Temporal, "atlas_seed", "Selects the Atlas territory arrangement. It is authored identity, not a value: Morph recalls it as an endpoint rather than interpolating an RNG."),
    entry(Temporal, "atlas_territories", "How many territories the Atlas divides the frame into."),
    entry(Temporal, "atlas_collision", "How much neighbouring territories bleed into one another at their borders."),
    entry(Temporal, "garden_amount", "How strongly the Refresh Garden holds and releases regions of the image."),
    entry(Temporal, "garden_gate", "What decides a Garden region refreshes: temporal delta, luma, chroma, a cellular ridge, audio energy or onset, a matte, or motion."),
    entry(Temporal, "garden_threshold", "How much of the gate signal is needed before a region refreshes."),
    entry(Temporal, "garden_softness", "Feathers the Garden's gate decision, so regions ease between held and refreshing rather than switching hard."),
    entry(Temporal, "garden_decay", "How quickly a held Garden region gives way once its gate closes."),
    entry(Temporal, "garden_max_hold_ticks", "A hard ceiling on how long any region may hold, so nothing freezes permanently."),
    entry(Temporal, "score_enabled", "Arms the Collision Score, which advances temporal state through a sequence of discrete states."),
    entry(Temporal, "score_seed", "Selects the Score's state arrangement. Authored identity, recalled as an endpoint rather than blended."),
    entry(Temporal, "score_state_count", "How many states the Score cycles through."),
    entry(Temporal, "score_trigger", "What advances the Score: a boundary, a downbeat, an audio onset, or a manual trigger."),
    entry(Temporal, "score_loop_driver", "Which live layer's loop boundary drives the Score, or none."),
    entry(Temporal, "reset_loop_boundary", "What the Score clears when a loop boundary passes."),
    entry(Temporal, "reset_downbeat", "What the Score clears on a downbeat. This is an imperative choice rather than a value, so nothing modulates it."),
    entry(
        Temporal,
        "disp_il_amount",
        "How much real interlacing the display stage applies. Everything the program renders is watched through something; this is the first part of that something.",
    ),
    entry(
        Temporal,
        "disp_il_mode",
        "How the two fields are reconciled: weave interleaves them, bob fills from the current image's neighbours, and blend ghosts one into the other.",
    ),
    entry(
        Temporal,
        "disp_il_order",
        "Swaps which field is dominant. This is the field-order fault, and getting it wrong is exactly what a mis-set deck looks like.",
    ),
    entry(
        Temporal,
        "disp_il_twitter",
        "Flips high vertical detail between fields, producing the shimmer fine horizontal lines get on an interlaced display.",
    ),
    entry(
        Temporal,
        "disp_il_judder",
        "Applies a 3:2 film cadence, holding two frames of every five. This is the judder film gets when it is pulled onto a 60 Hz display.",
    ),
    entry(
        Temporal,
        "disp_phosphor",
        "How long the phosphor holds light after the beam has passed. The store decays and the display reads the previous trail, which is what an accumulator actually does.",
    ),
    entry(
        Temporal,
        "disp_phos_r",
        "Red phosphor persistence. The defaults are the P22 signature, where green outlasts red and red outlasts blue.",
    ),
    entry(Temporal, "disp_phos_g", "Green phosphor persistence, the longest of the three in the P22 default."),
    entry(Temporal, "disp_phos_b", "Blue phosphor persistence, the shortest of the three."),
    entry(
        Temporal,
        "disp_model",
        "Which screen the program is watched through: flat, aperture grille, slot mask, shadow mask, LCD stripe, mono, or green screen. Scanlines, beam profile, and mask act only under a non-flat model.",
    ),
    entry(Temporal, "disp_scanlines", "How pronounced the scanline gaps are. It acts only under a non-flat display model, because a flat screen has no scan."),
    entry(
        Temporal,
        "disp_beam_width",
        "How wide the electron beam draws. The profile widens with brightness, so highlights bloom into their neighbouring lines the way a real beam does.",
    ),
    entry(Temporal, "disp_beam_shape", "How sharply the beam falls off from its centre."),
    entry(Temporal, "disp_mask_strength", "How strongly the selected mask pattern shows."),
    entry(Temporal, "disp_mask_dark", "How dark the gaps between mask elements sit."),
    entry(Temporal, "disp_bloom", "How much bright areas spill into their surroundings."),
    entry(Temporal, "disp_bloom_radius", "How far the bloom spreads, over a fixed twelve-tap gather ring."),
    entry(Temporal, "disp_halation", "Adds the faceplate tint real glass gives to bloomed highlights."),
    entry(Temporal, "disp_defocus", "Softens the whole picture the way a misconverged or tired tube does."),
    entry(Temporal, "disp_sag", "Bows the picture geometry, measured at the picture centre, the way high-voltage sag pulls a CRT raster."),
    entry(
        Temporal,
        "melt_amount",
        "How far the image drags along its own coverage boundaries. The matte is the composite's own alpha, so static key alpha, cellular gaps, and group mattes all melt through this one mechanism.",
    ),
    entry(Temporal, "melt_width", "How wide a band around each coverage boundary the melt acts in."),
    entry(
        Temporal,
        "melt_hold",
        "How much of the stage's own previous output is dissolved back into the band. This is what makes the smear stay put and creep further rather than washing out.",
    ),
    entry(Temporal, "melt_swirl", "Rotates the drag direction away from the boundary normal, up to a quarter turn either way."),
    entry(Temporal, "melt_chroma", "Lets colour run further off the edge than luma, by taking its chroma from a farther tap."),
    entry(Temporal, "melt_creep", "Biases the melt onto the uncovered side, so a keyed shape bleeds outward into the background rather than the background eating the shape."),
    entry(
        Temporal,
        "mosh_amount",
        "How much of the real encode-break-decode round trip is mixed in. At zero no encoder is alive at all: the round trip is lossy even at rest, so bypass is a true bypass rather than a hopeful identity.",
    ),
    entry(
        Temporal,
        "mosh_key_removal",
        "How often whole keyframes are thrown away. The first key after any reset always passes, because the decoder needs one complete picture to damage, and at full removal the picture never recovers.",
    ),
    entry(
        Temporal,
        "mosh_wipe",
        "Restricts the damaged codec picture to motion. At zero Mosh keeps its original full-program blend; toward one, moving objects wipe open corridors of stale pixels while still using the same codec round trip.",
    ),
    entry(
        Temporal,
        "mosh_smear",
        "Pulls damaged pixels backward along the MPEG motion vectors already exported by the mosh decoder. It adds no codec pass or frame of latency; the displaced sample is folded into the existing wet/dry blend.",
    ),
    entry(
        Temporal,
        "mosh_trail",
        "Retains the motion corridor on the fixed 30 Hz program clock, so an object leaves a fading pixel wake after it moves. At zero only the current motion wake remains.",
    ),
    entry(Temporal, "mosh_hold", "Re-applies the same delta several times under fresh timestamps, so motion keeps smearing in one direction."),
    entry(Temporal, "mosh_drop", "Starves the decoder of chunks. The last decoded picture is held, so a dropped chunk smears rather than flashing."),
    entry(Temporal, "mosh_shuffle", "Re-injects an older chunk out of order. Only chunks at least six deep are eligible, so the result is stale motion rather than a stutter."),
    entry(Temporal, "mosh_rate", "Scales how often hold, drop, and shuffle fire. It deliberately does not scale key removal."),
    entry(Temporal, "mosh_bitrate_starve", "Squeezes the encoder's bitrate. Every reconfigure forces a full re-acquire, so the picture snaps back and starts falling apart again."),
    entry(Temporal, "mosh_resync", "How often the encoder is forced to send a fresh key. At zero the picture never recovers on its own."),
    entry(Temporal, "mosh_recycle", "Feeds the encoder its own previous blended output instead of the clean image, so every pass builds on the last one's wreckage."),
    entry(
        Temporal,
        "sync_amount",
        "How far a scanline band slips sideways when it loses horizontal sync. Zero is the exact prior path, and the stage encodes no pass at all.",
    ),
    entry(
        Temporal,
        "sync_rate",
        "How often a band loses sync. At full rate about half the bands slip on every 30 Hz tick; at zero nothing ever fires and the stage stays dormant.",
    ),
    entry(
        Temporal,
        "sync_spread",
        "How tall a slipping band is, from a single line up to sixty-four. Every line in a band carries the identical offset, which is what makes a tear read as a tear rather than as static.",
    ),
    entry(
        Temporal,
        "sync_bias",
        "Pushes slips toward one side while keeping their size. At the extremes every slip carries the same sign, so a latched picture leans steadily instead of shredding around centre.",
    ),
    entry(
        Temporal,
        "sync_latched",
        "The failure switch. Off, a slip lives for its own tick and the picture heals. On, every slip is written into a bounded per-line table and stays there, accumulating, until you release the switch and the whole displacement unwinds at once. Pulling the shear to zero stops new damage but does not repair what is done.",
    ),
    // ------------------------------------------------------------------ ntsc
    entry(
        Ntsc,
        "enabled",
        "Runs composite VHS once over the finished programme, independently of per-layer Master bypass. It follows Codec Mosh in their shared bounded worker hop (Mosh then VHS), so it costs one frame of latency rather than adding another hop.",
    ),
    entry(Ntsc, "tape_speed", "Which tape speed is being emulated. Slower speeds carry less bandwidth and degrade more."),
    entry(Ntsc, "composite_noise_intensity", "How much noise rides on the composite signal as a whole."),
    entry(Ntsc, "composite_sharpening", "The sharpening a composite decoder applies, which is also what gives composite its characteristic ringing."),
    entry(Ntsc, "luma_noise_intensity", "How much noise sits on the brightness signal."),
    entry(Ntsc, "luma_smear", "How far brightness smears to the right of an edge, the way limited luma bandwidth smears it."),
    entry(Ntsc, "chroma_noise_intensity", "How much noise sits on the colour signal."),
    entry(Ntsc, "chroma_loss", "How often colour is lost outright, leaving stretches of the line monochrome."),
    entry(Ntsc, "snow_intensity", "Sparse dropout speckle across the picture."),
    entry(Ntsc, "head_switching_enabled", "Arms the tear at the bottom of the frame where a helical-scan head hands over to the next."),
    entry(Ntsc, "head_switching_height", "How many lines the head-switching tear occupies."),
    entry(Ntsc, "head_switching_shift", "How far the head-switching band is displaced sideways."),
    entry(Ntsc, "tracking_noise_enabled", "Arms the mistracking band, the noise a deck shows when the tape path is off."),
    entry(Ntsc, "tracking_noise_height", "How tall the mistracking band is. It sits low in the frame because that is where a deck loses lock on the control track first."),
    entry(Ntsc, "tracking_noise_wave", "How much the mistracking band waves rather than sitting straight."),
    entry(Ntsc, "tracking_noise_snow", "How much speckle the mistracking band carries."),
    entry(Ntsc, "edge_wave_enabled", "Arms a horizontal wobble on the whole picture, as an unstable timebase produces."),
    entry(Ntsc, "edge_wave_intensity", "How far the edge wave displaces each line."),
    entry(Ntsc, "edge_wave_speed", "How quickly the edge wave travels down the frame."),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_entry_is_unique_within_its_scope() {
        for scope in HelpScope::ALL {
            let mut seen = std::collections::HashSet::new();
            for item in CONTROL_HELP.iter().filter(|entry| entry.scope == scope) {
                assert!(
                    seen.insert(item.param),
                    "duplicate help entry {}/{}",
                    scope.key(),
                    item.param
                );
            }
        }
    }

    #[test]
    fn every_entry_is_a_usable_sentence() {
        for item in CONTROL_HELP {
            assert!(
                !item.param.is_empty() && !item.text.is_empty(),
                "empty help entry"
            );
            // House voice: what it does, and why it behaves that way. A
            // fragment shorter than this is a label, not help.
            assert!(
                item.text.len() >= 40,
                "{}/{} is too short to be help: {:?}",
                item.scope.key(),
                item.param,
                item.text
            );
            assert!(
                item.text.ends_with('.'),
                "{}/{} should read as prose: {:?}",
                item.scope.key(),
                item.param,
                item.text
            );
            assert!(
                item.text.chars().next().is_some_and(char::is_uppercase),
                "{}/{} should start with a capital: {:?}",
                item.scope.key(),
                item.param,
                item.text
            );
        }
    }

    #[test]
    fn lookup_resolves_by_scope_and_by_bare_name() {
        assert!(help_for(Master, "pixelate").is_some());
        assert!(help_for(Temporal, "sync_latched").is_some());
        assert!(help_for(Ntsc, "tape_speed").is_some());
        // A master name must not resolve inside the temporal scope.
        assert!(help_for(Temporal, "pixelate").is_none());
        assert!(help_for(Master, "nonexistent_control").is_none());
        // The native editor's bare-key lookup finds the master block.
        assert_eq!(help_for_any("pixelate"), help_for(Master, "pixelate"));
        assert!(help_for_any("no_such_param").is_none());
    }

    #[test]
    fn the_generated_panel_table_is_well_formed_and_escaped() {
        let js = panel_javascript();
        assert!(js.starts_with("// Generated from src/control_help.rs"));
        assert!(js.contains("window.CONTROL_HELP = Object.freeze({"));
        for scope in HelpScope::ALL {
            assert!(
                js.contains(&format!("  {}: Object.freeze({{", scope.key())),
                "missing scope {}",
                scope.key()
            );
        }
        assert!(js.contains("\"sync_latched\": \"The failure switch."));
        // Nothing that could close the script tag or inject markup survives.
        assert!(!js.contains('<'), "raw angle bracket in generated help");
        assert!(!js.contains('>'), "raw angle bracket in generated help");
        assert!(js.ends_with("});\n"));
        // Every entry appears exactly once.
        for item in CONTROL_HELP {
            assert!(
                js.contains(&format!("\"{}\": \"", item.param)),
                "missing generated entry {}",
                item.param
            );
        }
    }

    /// Coverage is proven against the shipped panel rather than asserted: every
    /// control row the operator can actually see must have help, and no entry
    /// may describe a control that no longer exists. A tranche that adds a row
    /// without writing its sentence fails here.
    #[test]
    fn every_panel_control_row_has_help_and_no_entry_is_an_orphan() {
        let html = include_str!("../static/index.html");

        // Pull the row identities straight out of the served markup.
        let scrape = |attribute: &str| -> std::collections::BTreeSet<String> {
            let mut found = std::collections::BTreeSet::new();
            let needle = format!("{attribute}=\"");
            let mut rest = html;
            while let Some(at) = rest.find(&needle) {
                rest = &rest[at + needle.len()..];
                if let Some(end) = rest.find('"') {
                    found.insert(rest[..end].to_string());
                    rest = &rest[end..];
                }
            }
            found
        };

        for (scope, attribute) in [
            (Master, "data-param"),
            (Temporal, "data-temporal"),
            (Ntsc, "data-ntsc"),
        ] {
            let rows = scrape(attribute);
            assert!(
                !rows.is_empty(),
                "{attribute} rows vanished from the panel; this test would pass vacuously"
            );
            for param in &rows {
                assert!(
                    help_for(scope, param).is_some(),
                    "no help for {}/{param} — every visible control needs a sentence",
                    scope.key()
                );
            }
            for item in CONTROL_HELP.iter().filter(|entry| entry.scope == scope) {
                assert!(
                    rows.contains(item.param),
                    "help entry {}/{} describes a control the panel no longer shows",
                    scope.key(),
                    item.param
                );
            }
        }
    }

    /// The panel must actually load the generated table, and load it before the
    /// script that indexes it.
    #[test]
    fn the_panel_loads_the_generated_help_before_the_app() {
        let html = include_str!("../static/index.html");
        let help_at = html
            .find("src=\"help.js\"")
            .expect("the panel must load the generated help table");
        let app_at = html
            .find("src=\"app.js\"")
            .expect("the panel must load app.js");
        assert!(
            help_at < app_at,
            "help.js must load before app.js, which reads window.CONTROL_HELP at startup"
        );
    }

    #[test]
    fn escaping_is_faithful_for_hostile_text() {
        let mut out = String::new();
        escape_into("a\"b\\c<d>e&f\ng\th", &mut out);
        assert_eq!(out, "a\\\"b\\\\c\\u003cd\\u003ee\\u0026f\\ng\\th");
    }
}
