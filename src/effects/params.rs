/// Temporal (frame-history) effect parameters: feedback trails and
/// slit-scan, driven by the renderer's ring buffer of past output frames.
#[derive(Debug, Clone, Copy)]
pub struct TemporalParams {
    pub feedback: f32,  // 0 = off, up to 0.95 = long light trails
    pub fb_zoom: f32,   // per-frame zoom of the fed-back image (1.0 = none)
    pub fb_rotate: f32, // per-frame rotation of the fed-back image, degrees
    pub slitscan: f32,  // 0 = off, 1 = deepest time-warp across the frame
    pub slit_axis: f32, // 0 = rows scan time (vertical), 1 = columns
}

impl Default for TemporalParams {
    fn default() -> Self {
        Self {
            feedback: 0.0,
            fb_zoom: 1.0,
            fb_rotate: 0.0,
            slitscan: 0.0,
            slit_axis: 0.0,
        }
    }
}

impl TemporalParams {
    pub fn is_active(&self) -> bool {
        self.feedback > 0.0 || self.slitscan > 0.0
    }
}

/// GPU-side effect parameters, uploaded as a uniform buffer each frame.
/// Must be 16-byte aligned (96 bytes total = 6 × vec4).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EffectUniforms {
    // vec4 #1
    pub pixelate_size: f32,  // 1.0 = off, 2..32 = block size in pixels
    pub rgb_split: f32,      // 0.0 = off, 1..30 = horizontal pixel offset
    pub resolution: [f32; 2],
    // vec4 #2
    pub hue_shift: f32,      // -180..180 degrees
    pub saturation: f32,     // -1..1 (0 = no change)
    pub brightness: f32,     // -1..1 (0 = no change)
    pub contrast: f32,       // -1..1 (0 = no change)
    // vec4 #3
    pub posterize: f32,      // 0 = off, 2..16 = color levels
    pub invert: f32,         // 0.0 = off, 1.0 = full invert
    pub downsample: f32,     // 1.0 = full res, 0.05..1.0 = fraction (lower = blurrier)
    pub time: f32,           // elapsed seconds (for animated noise)
    // vec4 #4 — Analog: grain
    pub grain_intensity: f32, // 0.0 = off, 0.01..0.3
    pub grain_size: f32,      // 1.0 = fine, 2..4 = coarse
    pub grain_algo: f32,      // 0=gaussian, 1=perlin, 2=salt_pepper, 3=blue
    pub color_grain: f32,     // 0=mono, 1=chromatic
    // vec4 #5 — Analog: breathing + vignette
    pub breathe_scale: f32,    // 0.0 = off, 0.0..0.05 (±zoom)
    pub breathe_rotation: f32, // 0.0 = off, 0.0..2.0 (degrees)
    pub breathe_position: f32, // 0.0 = off, 0.0..0.02 (drift)
    pub vignette: f32,         // 0.0 = off, 0.0..1.5
    // vec4 #6 — color drift + luma key
    pub color_drift: f32,     // 0.0 = off, 0.0..0.02 (per-frame random aberration)
    pub key_mode: f32,        // 0=off, 1=keep bright, 2=keep dark
    pub key_threshold: f32,   // 0..1 luminance cut point
    pub key_softness: f32,    // 0..0.5 smoothstep half-width around threshold
}

impl Default for EffectUniforms {
    fn default() -> Self {
        Self {
            pixelate_size: 1.0,
            rgb_split: 0.0,
            resolution: [1280.0, 720.0],
            hue_shift: 0.0,
            saturation: 0.0,
            brightness: 0.0,
            contrast: 0.0,
            posterize: 0.0,
            invert: 0.0,
            downsample: 1.0,
            time: 0.0,
            grain_intensity: 0.0,
            grain_size: 1.0,
            grain_algo: 0.0,
            color_grain: 0.0,
            breathe_scale: 0.0,
            breathe_rotation: 0.0,
            breathe_position: 0.0,
            vignette: 0.0,
            color_drift: 0.0,
            key_mode: 0.0,
            key_threshold: 0.5,
            key_softness: 0.1,
        }
    }
}

impl EffectUniforms {
    pub fn increase_pixelate(&mut self) {
        self.pixelate_size = (self.pixelate_size * 2.0).min(32.0).max(2.0);
    }

    pub fn decrease_pixelate(&mut self) {
        self.pixelate_size = (self.pixelate_size / 2.0).max(1.0);
    }

    pub fn increase_rgb_split(&mut self) {
        self.rgb_split = (self.rgb_split + 5.0).min(30.0);
    }

    pub fn decrease_rgb_split(&mut self) {
        self.rgb_split = (self.rgb_split - 5.0).max(0.0);
    }

    pub fn reset(&mut self) {
        let res = self.resolution;
        *self = Self::default();
        self.resolution = res;
    }
}
