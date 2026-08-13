//! Patch morphing: a crossfader between two captured parameter states.
//!
//! Slots A and B hold full snapshots of the continuous performance state
//! (master effects, NTSC, temporal, per-layer opacity/speed/key). While
//! both are set, the crossfader value writes the *base* parameters each
//! frame as the interpolation of A and B — the UI sliders visibly follow,
//! and the modulation matrix still breathes on top of the morphed bases.
//! Discrete values (blend toggles, algorithm selects) switch sides at the
//! midpoint. The morph amount is itself a modulation target ("morph"),
//! so an LFO or a MIDI knob can sweep between two entire worlds.

use crate::effects::params::TemporalParams;
use crate::effects::EffectUniforms;
use crate::layers::Layer;
use crate::ntsc::NtscParams;

#[derive(Clone)]
pub struct MorphSlot {
    master: EffectUniforms,
    ntsc: NtscParams,
    temporal: TemporalParams,
    layers: Vec<LayerMorph>,
}

#[derive(Clone, Copy)]
struct LayerMorph {
    opacity: f32,
    speed: f32,
    key_threshold: f32,
}

impl MorphSlot {
    pub fn capture(
        master: &EffectUniforms,
        ntsc: &NtscParams,
        temporal: &TemporalParams,
        layers: &[Layer],
    ) -> Self {
        Self {
            master: *master,
            ntsc: ntsc.clone(),
            temporal: temporal.clone(),
            layers: layers
                .iter()
                .map(|l| LayerMorph {
                    opacity: l.opacity,
                    speed: l.speed,
                    key_threshold: l.effects.key_threshold,
                })
                .collect(),
        }
    }
}

#[derive(Default)]
pub struct Morph {
    pub a: Option<MorphSlot>,
    pub b: Option<MorphSlot>,
    /// Crossfader position, 0 = A, 1 = B.
    pub t: f32,
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn pick(a: f32, b: f32, t: f32) -> f32 {
    if t < 0.5 {
        a
    } else {
        b
    }
}

impl Morph {
    pub fn active(&self) -> bool {
        self.a.is_some() && self.b.is_some()
    }

    pub fn clear(&mut self) {
        self.a = None;
        self.b = None;
    }

    /// Write the interpolated state into the live base parameters.
    /// resolution/time on the master uniforms are left to the render loop.
    pub fn apply(
        &self,
        t: f32,
        master: &mut EffectUniforms,
        ntsc: &mut NtscParams,
        temporal: &mut TemporalParams,
        layers: &mut [Layer],
    ) {
        let (Some(a), Some(b)) = (&self.a, &self.b) else {
            return;
        };

        // Master effects: continuous fields lerp, discrete ones switch.
        let (ma, mb) = (&a.master, &b.master);
        master.pixelate_size = lerp(ma.pixelate_size, mb.pixelate_size, t);
        master.rgb_split = lerp(ma.rgb_split, mb.rgb_split, t);
        master.hue_shift = lerp(ma.hue_shift, mb.hue_shift, t);
        master.saturation = lerp(ma.saturation, mb.saturation, t);
        master.brightness = lerp(ma.brightness, mb.brightness, t);
        master.contrast = lerp(ma.contrast, mb.contrast, t);
        master.posterize = lerp(ma.posterize, mb.posterize, t);
        master.invert = pick(ma.invert, mb.invert, t);
        master.downsample = lerp(ma.downsample, mb.downsample, t);
        master.grain_intensity = lerp(ma.grain_intensity, mb.grain_intensity, t);
        master.grain_size = lerp(ma.grain_size, mb.grain_size, t);
        master.grain_algo = pick(ma.grain_algo, mb.grain_algo, t);
        master.color_grain = pick(ma.color_grain, mb.color_grain, t);
        master.breathe_scale = lerp(ma.breathe_scale, mb.breathe_scale, t);
        master.breathe_rotation = lerp(ma.breathe_rotation, mb.breathe_rotation, t);
        master.breathe_position = lerp(ma.breathe_position, mb.breathe_position, t);
        master.vignette = lerp(ma.vignette, mb.vignette, t);
        master.color_drift = lerp(ma.color_drift, mb.color_drift, t);
        master.key_mode = pick(ma.key_mode, mb.key_mode, t);
        master.key_threshold = lerp(ma.key_threshold, mb.key_threshold, t);
        master.key_softness = lerp(ma.key_softness, mb.key_softness, t);

        // NTSC: floats lerp, switches and tape speed flip at midpoint.
        let (na, nb) = (&a.ntsc, &b.ntsc);
        ntsc.enabled = if t < 0.5 { na.enabled } else { nb.enabled };
        ntsc.tape_speed = if t < 0.5 { na.tape_speed } else { nb.tape_speed };
        ntsc.chroma_loss = lerp(na.chroma_loss, nb.chroma_loss, t);
        ntsc.edge_wave_enabled = if t < 0.5 { na.edge_wave_enabled } else { nb.edge_wave_enabled };
        ntsc.edge_wave_intensity = lerp(na.edge_wave_intensity, nb.edge_wave_intensity, t);
        ntsc.edge_wave_speed = lerp(na.edge_wave_speed, nb.edge_wave_speed, t);
        ntsc.head_switching_enabled =
            if t < 0.5 { na.head_switching_enabled } else { nb.head_switching_enabled };
        ntsc.head_switching_height = lerp(
            na.head_switching_height as f32,
            nb.head_switching_height as f32,
            t,
        )
        .round() as i32;
        ntsc.head_switching_shift = lerp(na.head_switching_shift, nb.head_switching_shift, t);
        ntsc.tracking_noise_enabled =
            if t < 0.5 { na.tracking_noise_enabled } else { nb.tracking_noise_enabled };
        ntsc.tracking_noise_height = lerp(
            na.tracking_noise_height as f32,
            nb.tracking_noise_height as f32,
            t,
        )
        .round() as i32;
        ntsc.tracking_noise_wave = lerp(na.tracking_noise_wave, nb.tracking_noise_wave, t);
        ntsc.tracking_noise_snow = lerp(na.tracking_noise_snow, nb.tracking_noise_snow, t);
        ntsc.snow_intensity = lerp(na.snow_intensity, nb.snow_intensity, t);
        ntsc.composite_noise_intensity =
            lerp(na.composite_noise_intensity, nb.composite_noise_intensity, t);
        ntsc.luma_noise_intensity = lerp(na.luma_noise_intensity, nb.luma_noise_intensity, t);
        ntsc.chroma_noise_intensity =
            lerp(na.chroma_noise_intensity, nb.chroma_noise_intensity, t);
        ntsc.luma_smear = lerp(na.luma_smear, nb.luma_smear, t);
        ntsc.composite_sharpening = lerp(na.composite_sharpening, nb.composite_sharpening, t);

        // Temporal: floats lerp, the slit axis flips at midpoint.
        let (ta, tb) = (&a.temporal, &b.temporal);
        temporal.feedback = lerp(ta.feedback, tb.feedback, t);
        temporal.fb_zoom = lerp(ta.fb_zoom, tb.fb_zoom, t);
        temporal.fb_rotate = lerp(ta.fb_rotate, tb.fb_rotate, t);
        temporal.slitscan = lerp(ta.slitscan, tb.slitscan, t);
        temporal.slit_axis = pick(ta.slit_axis, tb.slit_axis, t);

        // Layers present in both snapshots morph; extras keep their state.
        for (i, layer) in layers.iter_mut().enumerate() {
            let (Some(la), Some(lb)) = (a.layers.get(i), b.layers.get(i)) else {
                continue;
            };
            layer.opacity = lerp(la.opacity, lb.opacity, t);
            layer.speed = lerp(la.speed, lb.speed, t);
            layer.effects.key_threshold = lerp(la.key_threshold, lb.key_threshold, t);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morph_lerps_continuous_and_switches_discrete() {
        let mut a_fx = EffectUniforms::default();
        a_fx.pixelate_size = 1.0;
        a_fx.invert = 0.0;
        let mut b_fx = EffectUniforms::default();
        b_fx.pixelate_size = 31.0;
        b_fx.invert = 1.0;

        let a = MorphSlot {
            master: a_fx,
            ntsc: NtscParams::default(),
            temporal: TemporalParams::default(),
            layers: vec![],
        };
        let mut nb = NtscParams::default();
        nb.snow_intensity = 1.0;
        let b = MorphSlot {
            master: b_fx,
            ntsc: nb,
            temporal: TemporalParams::default(),
            layers: vec![],
        };

        let morph = Morph { a: Some(a), b: Some(b), t: 0.0 };
        let mut master = EffectUniforms::default();
        let mut ntsc = NtscParams::default();
        let mut temporal = TemporalParams::default();

        morph.apply(0.5, &mut master, &mut ntsc, &mut temporal, &mut []);
        assert!((master.pixelate_size - 16.0).abs() < 1e-6);
        assert_eq!(master.invert, 1.0, "discrete switches to B at midpoint");
        assert!((ntsc.snow_intensity - 0.5).abs() < 1e-6);

        morph.apply(0.25, &mut master, &mut ntsc, &mut temporal, &mut []);
        assert!((master.pixelate_size - 8.5).abs() < 1e-6);
        assert_eq!(master.invert, 0.0, "discrete stays A below midpoint");
    }
}
