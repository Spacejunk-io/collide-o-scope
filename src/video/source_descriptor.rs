//! Bounded, provenance-carrying source color and display truth.
//!
//! These descriptors describe the source that produced the legacy RGBA8
//! mailbox. They are deliberately fixed-size: no tag string, ICC profile, or
//! arbitrary side-data blob can enter decoder telemetry, Stage Health, proxy
//! receipts, or export provenance through this module.

use ffmpeg_next as ffmpeg;
use serde::{Deserialize, Serialize};

const MAX_RATIONAL_TERM: u32 = 1_000_000;
const DISPLAY_MATRIX_BYTES: usize = 9 * std::mem::size_of::<i32>();

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DescriptorProvenance {
    #[default]
    Unspecified,
    ContainerDeclared,
    CodecDeclared,
    FrameDeclared,
    StillHeaderDeclared,
    StillExifDeclared,
    PixelFormatDerived,
    DeclaredDerived,
    InferredFallback,
}

impl DescriptorProvenance {
    pub const fn is_declared(self) -> bool {
        matches!(
            self,
            Self::ContainerDeclared
                | Self::CodecDeclared
                | Self::FrameDeclared
                | Self::StillHeaderDeclared
                | Self::StillExifDeclared
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct DescriptorValue<T> {
    pub value: T,
    pub provenance: DescriptorProvenance,
}

impl<T: Default> Default for DescriptorValue<T> {
    fn default() -> Self {
        Self {
            value: T::default(),
            provenance: DescriptorProvenance::Unspecified,
        }
    }
}

impl<T> DescriptorValue<T> {
    pub const fn new(value: T, provenance: DescriptorProvenance) -> Self {
        Self { value, provenance }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PixelFamily {
    #[default]
    Unspecified,
    Yuv,
    Yuva,
    Rgb,
    Rgba,
    Gray,
    GrayAlpha,
    Paletted,
    Hardware,
    Other,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BitDepth {
    #[default]
    Unspecified,
    Bits(u8),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceColorRange {
    #[default]
    Unspecified,
    Limited,
    Full,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatrixCoefficients {
    #[default]
    Unspecified,
    Rgb,
    Bt709,
    Fcc,
    Bt470Bg,
    Smpte170M,
    Smpte240M,
    Ycgco,
    Bt2020Ncl,
    Bt2020Cl,
    Smpte2085,
    ChromaDerivedNcl,
    ChromaDerivedCl,
    Ictcp,
    Other,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColorPrimaries {
    #[default]
    Unspecified,
    Bt709,
    Bt470M,
    Bt470Bg,
    Smpte170M,
    Smpte240M,
    Film,
    Bt2020,
    Smpte428,
    Smpte431,
    Smpte432,
    Ebu3213,
    Other,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransferCharacteristic {
    #[default]
    Unspecified,
    Bt709,
    Gamma22,
    Gamma28,
    Smpte170M,
    Smpte240M,
    Linear,
    Log,
    LogSqrt,
    Iec61966_2_4,
    Bt1361Ecg,
    Srgb,
    Bt2020_10,
    Bt2020_12,
    Pq,
    Smpte428,
    Hlg,
    Other,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChromaLocation {
    #[default]
    Unspecified,
    Left,
    Center,
    TopLeft,
    Top,
    BottomLeft,
    Bottom,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChromaSubsampling {
    pub horizontal_log2: u8,
    pub vertical_log2: u8,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SourceColorDescriptor {
    pub pixel_family: DescriptorValue<PixelFamily>,
    pub bit_depth: DescriptorValue<BitDepth>,
    pub range: DescriptorValue<SourceColorRange>,
    pub matrix: DescriptorValue<MatrixCoefficients>,
    pub primaries: DescriptorValue<ColorPrimaries>,
    pub transfer: DescriptorValue<TransferCharacteristic>,
    pub chroma_location: DescriptorValue<ChromaLocation>,
    pub chroma_subsampling: DescriptorValue<ChromaSubsampling>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PixelDimensions {
    pub width: u32,
    pub height: u32,
}

impl PixelDimensions {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct BoundedRational {
    pub numerator: u32,
    pub denominator: u32,
}

impl Default for BoundedRational {
    fn default() -> Self {
        Self {
            numerator: 0,
            denominator: 1,
        }
    }
}

impl BoundedRational {
    pub fn new(numerator: i64, denominator: i64) -> Option<Self> {
        if numerator <= 0 || denominator <= 0 {
            return None;
        }
        let numerator = u64::try_from(numerator).ok()?;
        let denominator = u64::try_from(denominator).ok()?;
        let divisor = gcd_u64(numerator, denominator);
        let numerator = numerator / divisor;
        let denominator = denominator / divisor;
        if numerator > u64::from(MAX_RATIONAL_TERM) || denominator > u64::from(MAX_RATIONAL_TERM) {
            return None;
        }
        Some(Self {
            numerator: numerator as u32,
            denominator: denominator as u32,
        })
    }

    pub const fn square() -> Self {
        Self {
            numerator: 1,
            denominator: 1,
        }
    }

    pub const fn is_specified(self) -> bool {
        self.numerator > 0 && self.denominator > 0
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CleanAperture {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
}

impl CleanAperture {
    pub fn validate(self, coded: PixelDimensions) -> Option<Self> {
        let horizontal = self.left.checked_add(self.right)?;
        let vertical = self.top.checked_add(self.bottom)?;
        (horizontal < coded.width && vertical < coded.height).then_some(self)
    }

    pub fn raster(self, coded: PixelDimensions) -> Option<PixelDimensions> {
        self.validate(coded).map(|clean| PixelDimensions {
            width: coded.width - clean.left - clean.right,
            height: coded.height - clean.top - clean.bottom,
        })
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rotation {
    #[default]
    Unspecified,
    Degrees0,
    Degrees90,
    Degrees180,
    Degrees270,
    OtherMilliDegrees(i32),
}

impl Rotation {
    pub const fn swaps_axes(self) -> bool {
        matches!(self, Self::Degrees90 | Self::Degrees270)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mirror {
    #[default]
    Unspecified,
    None,
    Horizontal,
    Vertical,
    Both,
}

/// The eight lossless orientation laws used by EXIF and by axis-aligned
/// FFmpeg display matrices. Pixel storage is never rewritten to realize it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormalizedOrientation {
    #[default]
    Unspecified,
    Identity,
    MirrorHorizontal,
    Rotate180,
    MirrorVertical,
    Transpose,
    Rotate90,
    Transverse,
    Rotate270,
}

impl NormalizedOrientation {
    pub const fn rotation(self) -> Rotation {
        match self {
            Self::Unspecified => Rotation::Unspecified,
            Self::Identity | Self::MirrorHorizontal => Rotation::Degrees0,
            Self::Rotate180 | Self::MirrorVertical => Rotation::Degrees180,
            Self::Transpose | Self::Rotate90 => Rotation::Degrees90,
            Self::Transverse | Self::Rotate270 => Rotation::Degrees270,
        }
    }

    pub const fn mirror(self) -> Mirror {
        match self {
            Self::Unspecified => Mirror::Unspecified,
            Self::Identity | Self::Rotate90 | Self::Rotate180 | Self::Rotate270 => Mirror::None,
            Self::MirrorHorizontal | Self::Transpose | Self::Transverse => Mirror::Horizontal,
            Self::MirrorVertical => Mirror::Vertical,
        }
    }

    pub const fn swaps_axes(self) -> bool {
        matches!(
            self,
            Self::Transpose | Self::Rotate90 | Self::Transverse | Self::Rotate270
        )
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct DisplayMatrix {
    pub elements: [i32; 9],
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceFieldOrder {
    #[default]
    Unspecified,
    Progressive,
    TopCodedTopDisplayed,
    BottomCodedBottomDisplayed,
    TopCodedBottomDisplayed,
    BottomCodedTopDisplayed,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SourceDisplayDescriptor {
    pub coded_dimensions: DescriptorValue<PixelDimensions>,
    /// Clean-aperture-adjusted raster after the normalized orientation. SAR
    /// remains separate and exact, so this field never rounds display aspect.
    pub display_dimensions: DescriptorValue<PixelDimensions>,
    pub sample_aspect_ratio: DescriptorValue<BoundedRational>,
    pub clean_aperture: DescriptorValue<CleanAperture>,
    pub rotation: DescriptorValue<Rotation>,
    pub mirror: DescriptorValue<Mirror>,
    pub orientation: DescriptorValue<NormalizedOrientation>,
    pub display_matrix: DescriptorValue<DisplayMatrix>,
    pub field_order: DescriptorValue<SourceFieldOrder>,
}

impl SourceDisplayDescriptor {
    pub fn refresh_display_dimensions(&mut self) {
        let coded = self.coded_dimensions.value;
        if coded.width == 0 || coded.height == 0 {
            self.display_dimensions = DescriptorValue::default();
            return;
        }
        let aperture = if self.clean_aperture.provenance == DescriptorProvenance::Unspecified {
            coded
        } else {
            self.clean_aperture.value.raster(coded).unwrap_or(coded)
        };
        let swaps_axes = if self.orientation.provenance != DescriptorProvenance::Unspecified {
            self.orientation.value.swaps_axes()
        } else {
            self.rotation.value.swaps_axes()
        };
        let value = if swaps_axes {
            PixelDimensions::new(aperture.height, aperture.width)
        } else {
            aperture
        };
        let has_declared_input = [
            self.coded_dimensions.provenance,
            self.clean_aperture.provenance,
            self.orientation.provenance,
            self.rotation.provenance,
        ]
        .into_iter()
        .any(DescriptorProvenance::is_declared);
        let provenance = if has_declared_input {
            DescriptorProvenance::DeclaredDerived
        } else {
            DescriptorProvenance::InferredFallback
        };
        self.display_dimensions = DescriptorValue::new(value, provenance);
    }

    /// Exact display aspect as a bounded rational. `None` means either source
    /// dimensions or SAR were not trustworthy enough to publish.
    pub fn display_aspect_ratio(self) -> Option<BoundedRational> {
        let raster = self.display_dimensions.value;
        let sar = self.sample_aspect_ratio.value;
        if raster.width == 0 || raster.height == 0 || !sar.is_specified() {
            return None;
        }
        let swaps_axes = if self.orientation.provenance != DescriptorProvenance::Unspecified {
            self.orientation.value.swaps_axes()
        } else {
            self.rotation.value.swaps_axes()
        };
        let (sar_width, sar_height) = if swaps_axes {
            (sar.denominator, sar.numerator)
        } else {
            (sar.numerator, sar.denominator)
        };
        BoundedRational::new(
            i64::from(raster.width) * i64::from(sar_width),
            i64::from(raster.height) * i64::from(sar_height),
        )
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConversionPolicyKind {
    #[default]
    Unspecified,
    /// Exact historical `sws_getContext(..., BILINEAR)` behavior. No color
    /// details setter was called.
    LegacyUnspecified,
    /// The source declared both a supported matrix and range and libswscale
    /// accepted those exact details before the first byte conversion.
    ExplicitDeclaredMatrixRange,
    /// Declared metadata was present but cannot be represented by the current
    /// software converter; the actual path remained legacy and says so.
    LegacyUnsupportedDeclared,
    /// libswscale rejected an otherwise supported request. Pixels came from
    /// the legacy context and the failure remains visible.
    LegacyConfigurationRejected,
    /// image-rs decoded directly to the existing RGBA8 mailbox representation.
    StillImageLegacyRgba8,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SourceConversionPolicy {
    pub kind: ConversionPolicyKind,
    pub input_matrix: MatrixCoefficients,
    pub input_range: SourceColorRange,
    pub output_family: PixelFamily,
    pub output_bit_depth: BitDepth,
    pub output_range: SourceColorRange,
}

impl SourceConversionPolicy {
    pub const fn legacy_video() -> Self {
        Self {
            kind: ConversionPolicyKind::LegacyUnspecified,
            input_matrix: MatrixCoefficients::Unspecified,
            input_range: SourceColorRange::Unspecified,
            output_family: PixelFamily::Rgba,
            output_bit_depth: BitDepth::Bits(8),
            output_range: SourceColorRange::Full,
        }
    }

    pub const fn still_image() -> Self {
        Self {
            kind: ConversionPolicyKind::StillImageLegacyRgba8,
            input_matrix: MatrixCoefficients::Unspecified,
            input_range: SourceColorRange::Unspecified,
            output_family: PixelFamily::Rgba,
            output_bit_depth: BitDepth::Bits(8),
            output_range: SourceColorRange::Full,
        }
    }
}

/// Fixed-size reference mapping from displayed normalized coordinates into
/// the coded raster. This is the P4b stop-gate artifact: it proves the law
/// without mutating source bytes or introducing a second full-frame pass.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(
    dead_code,
    reason = "P4b stop-gate reference awaits the audited renderer UV integration seam"
)]
pub struct SourceUvReference {
    pub coded_origin: [f32; 2],
    pub coded_extent: [f32; 2],
    pub orientation: NormalizedOrientation,
}

#[allow(
    dead_code,
    reason = "P4b stop-gate reference awaits the audited renderer UV integration seam"
)]
impl SourceUvReference {
    pub fn from_descriptor(descriptor: SourceDisplayDescriptor) -> Option<Self> {
        let coded = descriptor.coded_dimensions.value;
        if coded.width == 0 || coded.height == 0 {
            return None;
        }
        let clean = if descriptor.clean_aperture.provenance == DescriptorProvenance::Unspecified {
            CleanAperture::default()
        } else {
            descriptor.clean_aperture.value.validate(coded)?
        };
        let orientation = descriptor.orientation.value;
        if orientation == NormalizedOrientation::Unspecified {
            return None;
        }
        Some(Self {
            coded_origin: [
                clean.left as f32 / coded.width as f32,
                clean.top as f32 / coded.height as f32,
            ],
            coded_extent: [
                (coded.width - clean.left - clean.right) as f32 / coded.width as f32,
                (coded.height - clean.top - clean.bottom) as f32 / coded.height as f32,
            ],
            orientation,
        })
    }

    pub fn map_display_to_coded(self, display_uv: [f32; 2]) -> [f32; 2] {
        let [u, v] = display_uv;
        let local = match self.orientation {
            NormalizedOrientation::Unspecified | NormalizedOrientation::Identity => [u, v],
            NormalizedOrientation::MirrorHorizontal => [1.0 - u, v],
            NormalizedOrientation::Rotate180 => [1.0 - u, 1.0 - v],
            NormalizedOrientation::MirrorVertical => [u, 1.0 - v],
            NormalizedOrientation::Transpose => [v, u],
            NormalizedOrientation::Rotate90 => [v, 1.0 - u],
            NormalizedOrientation::Transverse => [1.0 - v, 1.0 - u],
            NormalizedOrientation::Rotate270 => [1.0 - v, u],
        };
        [
            self.coded_origin[0] + local[0] * self.coded_extent[0],
            self.coded_origin[1] + local[1] * self.coded_extent[1],
        ]
    }
}

pub(crate) fn descriptors_from_ffmpeg(
    stream: &ffmpeg::Stream<'_>,
    decoder: &ffmpeg::decoder::Video,
) -> (SourceColorDescriptor, SourceDisplayDescriptor) {
    let mut color = SourceColorDescriptor::default();
    merge_pixel_format(&mut color, decoder.format());
    let parameters = stream.parameters();
    let raw_parameters = unsafe { &*parameters.as_ptr() };
    merge_ffmpeg_color(
        &mut color,
        ffmpeg::color::Range::from(raw_parameters.color_range),
        ffmpeg::color::Space::from(raw_parameters.color_space),
        ffmpeg::color::Primaries::from(raw_parameters.color_primaries),
        ffmpeg::color::TransferCharacteristic::from(raw_parameters.color_trc),
        ffmpeg::chroma::Location::from(raw_parameters.chroma_location),
        DescriptorProvenance::CodecDeclared,
    );

    let coded = PixelDimensions::new(decoder.width(), decoder.height());
    let coded_provenance = if raw_parameters.width > 0
        && raw_parameters.height > 0
        && u32::try_from(raw_parameters.width).ok() == Some(coded.width)
        && u32::try_from(raw_parameters.height).ok() == Some(coded.height)
    {
        DescriptorProvenance::CodecDeclared
    } else {
        DescriptorProvenance::InferredFallback
    };
    let mut display = SourceDisplayDescriptor {
        coded_dimensions: DescriptorValue::new(coded, coded_provenance),
        sample_aspect_ratio: rational_from_ffmpeg(ffmpeg::Rational::from(
            raw_parameters.sample_aspect_ratio,
        ))
        .map_or_else(DescriptorValue::default, |value| {
            DescriptorValue::new(value, DescriptorProvenance::CodecDeclared)
        }),
        field_order: ffmpeg_field_order(raw_parameters.field_order)
            .map_or_else(DescriptorValue::default, |value| {
                DescriptorValue::new(value, DescriptorProvenance::CodecDeclared)
            }),
        ..SourceDisplayDescriptor::default()
    };

    for side_data in stream.side_data() {
        match side_data.kind() {
            ffmpeg::codec::packet::side_data::Type::DisplayMatrix => {
                if let Some(matrix) = parse_display_matrix(side_data.data()) {
                    apply_display_matrix(
                        &mut display,
                        matrix,
                        DescriptorProvenance::ContainerDeclared,
                    );
                }
            }
            ffmpeg::codec::packet::side_data::Type::FRAME_CROPPING => {
                if let Some(clean) = parse_frame_cropping(side_data.data(), coded) {
                    display.clean_aperture =
                        DescriptorValue::new(clean, DescriptorProvenance::ContainerDeclared);
                }
            }
            _ => {}
        }
    }
    if matches!(
        display.sample_aspect_ratio.provenance,
        DescriptorProvenance::Unspecified | DescriptorProvenance::InferredFallback
    ) {
        display.sample_aspect_ratio = DescriptorValue::new(
            BoundedRational::square(),
            DescriptorProvenance::InferredFallback,
        );
    }
    if display.orientation.provenance == DescriptorProvenance::Unspecified {
        display.orientation = DescriptorValue::new(
            NormalizedOrientation::Identity,
            DescriptorProvenance::InferredFallback,
        );
        display.rotation =
            DescriptorValue::new(Rotation::Degrees0, DescriptorProvenance::InferredFallback);
        display.mirror = DescriptorValue::new(Mirror::None, DescriptorProvenance::InferredFallback);
    }
    display.refresh_display_dimensions();
    (color, display)
}

pub(crate) fn merge_first_ffmpeg_frame(
    color: &mut SourceColorDescriptor,
    display: &mut SourceDisplayDescriptor,
    frame: &ffmpeg::frame::Video,
) {
    merge_pixel_format(color, frame.format());
    merge_ffmpeg_color(
        color,
        frame.color_range(),
        frame.color_space(),
        frame.color_primaries(),
        frame.color_transfer_characteristic(),
        frame.chroma_location(),
        DescriptorProvenance::FrameDeclared,
    );
    if matches!(
        display.sample_aspect_ratio.provenance,
        DescriptorProvenance::Unspecified | DescriptorProvenance::InferredFallback
    ) {
        if let Some(value) = rational_from_ffmpeg(frame.aspect_ratio()) {
            display.sample_aspect_ratio =
                DescriptorValue::new(value, DescriptorProvenance::FrameDeclared);
        }
    }
    let coded = display.coded_dimensions.value;
    // AVFrame crop fields are bounded integers set by the decoder. Treat a
    // zero crop as absence, not as an authored clean-aperture claim.
    let crop = unsafe {
        let raw = frame.as_ptr();
        CleanAperture {
            left: (*raw).crop_left.min(u32::MAX as usize) as u32,
            top: (*raw).crop_top.min(u32::MAX as usize) as u32,
            right: (*raw).crop_right.min(u32::MAX as usize) as u32,
            bottom: (*raw).crop_bottom.min(u32::MAX as usize) as u32,
        }
    };
    if crop != CleanAperture::default() {
        if let Some(clean) = crop.validate(coded) {
            display.clean_aperture =
                DescriptorValue::new(clean, DescriptorProvenance::FrameDeclared);
        }
    }
    if let Some(side_data) = frame.side_data(ffmpeg::frame::side_data::Type::DisplayMatrix) {
        if let Some(matrix) = parse_display_matrix(side_data.data()) {
            apply_display_matrix(display, matrix, DescriptorProvenance::FrameDeclared);
        }
    }
    if display.field_order.provenance == DescriptorProvenance::Unspecified {
        let order = if frame.is_interlaced() {
            if frame.is_top_first() {
                SourceFieldOrder::TopCodedTopDisplayed
            } else {
                SourceFieldOrder::BottomCodedBottomDisplayed
            }
        } else {
            SourceFieldOrder::Progressive
        };
        display.field_order = DescriptorValue::new(order, DescriptorProvenance::FrameDeclared);
    }
    display.refresh_display_dimensions();
}

pub(crate) fn configure_sws_conversion(
    scaler: &mut ffmpeg::software::scaling::Context,
    color: SourceColorDescriptor,
) -> SourceConversionPolicy {
    let mut policy = SourceConversionPolicy::legacy_video();
    policy.input_matrix = color.matrix.value;
    policy.input_range = color.range.value;
    let declared = color.matrix.provenance.is_declared() && color.range.provenance.is_declared();
    if !declared {
        return policy;
    }
    let Some(coefficients) = sws_coefficients(color.matrix.value) else {
        policy.kind = ConversionPolicyKind::LegacyUnsupportedDeclared;
        return policy;
    };
    let source_full_range = match color.range.value {
        SourceColorRange::Limited => 0,
        SourceColorRange::Full => 1,
        SourceColorRange::Unspecified => return policy,
    };
    let result = unsafe {
        let table = ffmpeg::ffi::sws_getCoefficients(coefficients);
        if table.is_null() {
            -1
        } else {
            ffmpeg::ffi::sws_setColorspaceDetails(
                scaler.as_mut_ptr(),
                table,
                source_full_range,
                table,
                1,
                0,
                1 << 16,
                1 << 16,
            )
        }
    };
    policy.kind = if result >= 0 {
        ConversionPolicyKind::ExplicitDeclaredMatrixRange
    } else {
        ConversionPolicyKind::LegacyConfigurationRejected
    };
    policy
}

pub(crate) fn still_descriptors(
    width: u32,
    height: u32,
    color_type: image::ColorType,
    orientation: Option<image::metadata::Orientation>,
) -> (
    SourceColorDescriptor,
    SourceDisplayDescriptor,
    SourceConversionPolicy,
) {
    let (family, depth) = match color_type {
        image::ColorType::L8 => (PixelFamily::Gray, 8),
        image::ColorType::La8 => (PixelFamily::GrayAlpha, 8),
        image::ColorType::Rgb8 => (PixelFamily::Rgb, 8),
        image::ColorType::Rgba8 => (PixelFamily::Rgba, 8),
        image::ColorType::L16 => (PixelFamily::Gray, 16),
        image::ColorType::La16 => (PixelFamily::GrayAlpha, 16),
        image::ColorType::Rgb16 => (PixelFamily::Rgb, 16),
        image::ColorType::Rgba16 => (PixelFamily::Rgba, 16),
        image::ColorType::Rgb32F => (PixelFamily::Rgb, 32),
        image::ColorType::Rgba32F => (PixelFamily::Rgba, 32),
        _ => (PixelFamily::Other, 0),
    };
    let mut color = SourceColorDescriptor {
        pixel_family: DescriptorValue::new(family, DescriptorProvenance::StillHeaderDeclared),
        ..SourceColorDescriptor::default()
    };
    if depth > 0 {
        color.bit_depth = DescriptorValue::new(
            BitDepth::Bits(depth),
            DescriptorProvenance::StillHeaderDeclared,
        );
    }

    let normalized = orientation
        .map(normalized_image_orientation)
        .unwrap_or(NormalizedOrientation::Identity);
    let provenance = if orientation.is_some() {
        DescriptorProvenance::StillExifDeclared
    } else {
        DescriptorProvenance::InferredFallback
    };
    let mut display = SourceDisplayDescriptor {
        coded_dimensions: DescriptorValue::new(
            PixelDimensions::new(width, height),
            DescriptorProvenance::StillHeaderDeclared,
        ),
        sample_aspect_ratio: DescriptorValue::new(
            BoundedRational::square(),
            DescriptorProvenance::InferredFallback,
        ),
        rotation: DescriptorValue::new(normalized.rotation(), provenance),
        mirror: DescriptorValue::new(normalized.mirror(), provenance),
        orientation: DescriptorValue::new(normalized, provenance),
        field_order: DescriptorValue::new(
            SourceFieldOrder::Progressive,
            DescriptorProvenance::InferredFallback,
        ),
        ..SourceDisplayDescriptor::default()
    };
    display.refresh_display_dimensions();
    (color, display, SourceConversionPolicy::still_image())
}

fn merge_pixel_format(color: &mut SourceColorDescriptor, format: ffmpeg::format::Pixel) {
    let Some(descriptor) = format.descriptor() else {
        return;
    };
    let raw = unsafe { &*descriptor.as_ptr() };
    let flags = raw.flags;
    let alpha = flags & ffmpeg::ffi::AV_PIX_FMT_FLAG_ALPHA as u64 != 0;
    let family = if flags & ffmpeg::ffi::AV_PIX_FMT_FLAG_HWACCEL as u64 != 0 {
        PixelFamily::Hardware
    } else if flags & ffmpeg::ffi::AV_PIX_FMT_FLAG_PAL as u64 != 0 {
        PixelFamily::Paletted
    } else if flags & ffmpeg::ffi::AV_PIX_FMT_FLAG_RGB as u64 != 0 {
        if alpha {
            PixelFamily::Rgba
        } else {
            PixelFamily::Rgb
        }
    } else if raw.nb_components <= 2 {
        if alpha {
            PixelFamily::GrayAlpha
        } else {
            PixelFamily::Gray
        }
    } else if alpha {
        PixelFamily::Yuva
    } else {
        PixelFamily::Yuv
    };
    color.pixel_family = DescriptorValue::new(family, DescriptorProvenance::PixelFormatDerived);
    let depth = raw.comp[..usize::from(raw.nb_components.min(4))]
        .iter()
        .filter_map(|component| u8::try_from(component.depth).ok())
        .max()
        .filter(|depth| (1..=32).contains(depth));
    if let Some(depth) = depth {
        color.bit_depth = DescriptorValue::new(
            BitDepth::Bits(depth),
            DescriptorProvenance::PixelFormatDerived,
        );
    }
    if matches!(family, PixelFamily::Yuv | PixelFamily::Yuva) {
        color.chroma_subsampling = DescriptorValue::new(
            ChromaSubsampling {
                horizontal_log2: raw.log2_chroma_w.min(4),
                vertical_log2: raw.log2_chroma_h.min(4),
            },
            DescriptorProvenance::PixelFormatDerived,
        );
        if color.range.provenance == DescriptorProvenance::Unspecified
            && descriptor.name().starts_with("yuvj")
        {
            color.range = DescriptorValue::new(
                SourceColorRange::Full,
                DescriptorProvenance::PixelFormatDerived,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn merge_ffmpeg_color(
    color: &mut SourceColorDescriptor,
    range: ffmpeg::color::Range,
    matrix: ffmpeg::color::Space,
    primaries: ffmpeg::color::Primaries,
    transfer: ffmpeg::color::TransferCharacteristic,
    chroma: ffmpeg::chroma::Location,
    provenance: DescriptorProvenance,
) {
    if range != ffmpeg::color::Range::Unspecified {
        color.range = DescriptorValue::new(
            match range {
                ffmpeg::color::Range::MPEG => SourceColorRange::Limited,
                ffmpeg::color::Range::JPEG => SourceColorRange::Full,
                ffmpeg::color::Range::Unspecified => SourceColorRange::Unspecified,
            },
            provenance,
        );
    }
    if matrix != ffmpeg::color::Space::Unspecified {
        color.matrix = DescriptorValue::new(matrix_from_ffmpeg(matrix), provenance);
    }
    if primaries != ffmpeg::color::Primaries::Unspecified {
        color.primaries = DescriptorValue::new(primaries_from_ffmpeg(primaries), provenance);
    }
    if transfer != ffmpeg::color::TransferCharacteristic::Unspecified {
        color.transfer = DescriptorValue::new(transfer_from_ffmpeg(transfer), provenance);
    }
    if chroma != ffmpeg::chroma::Location::Unspecified {
        color.chroma_location = DescriptorValue::new(chroma_from_ffmpeg(chroma), provenance);
    }
}

fn matrix_from_ffmpeg(value: ffmpeg::color::Space) -> MatrixCoefficients {
    use ffmpeg::color::Space as F;
    match value {
        F::RGB => MatrixCoefficients::Rgb,
        F::BT709 => MatrixCoefficients::Bt709,
        F::FCC => MatrixCoefficients::Fcc,
        F::BT470BG => MatrixCoefficients::Bt470Bg,
        F::SMPTE170M => MatrixCoefficients::Smpte170M,
        F::SMPTE240M => MatrixCoefficients::Smpte240M,
        F::YCGCO => MatrixCoefficients::Ycgco,
        F::BT2020NCL => MatrixCoefficients::Bt2020Ncl,
        F::BT2020CL => MatrixCoefficients::Bt2020Cl,
        F::SMPTE2085 => MatrixCoefficients::Smpte2085,
        F::ChromaDerivedNCL => MatrixCoefficients::ChromaDerivedNcl,
        F::ChromaDerivedCL => MatrixCoefficients::ChromaDerivedCl,
        F::ICTCP => MatrixCoefficients::Ictcp,
        F::Unspecified => MatrixCoefficients::Unspecified,
        _ => MatrixCoefficients::Other,
    }
}

fn primaries_from_ffmpeg(value: ffmpeg::color::Primaries) -> ColorPrimaries {
    use ffmpeg::color::Primaries as F;
    match value {
        F::BT709 => ColorPrimaries::Bt709,
        F::BT470M => ColorPrimaries::Bt470M,
        F::BT470BG => ColorPrimaries::Bt470Bg,
        F::SMPTE170M => ColorPrimaries::Smpte170M,
        F::SMPTE240M => ColorPrimaries::Smpte240M,
        F::Film => ColorPrimaries::Film,
        F::BT2020 => ColorPrimaries::Bt2020,
        F::SMPTE428 => ColorPrimaries::Smpte428,
        F::SMPTE431 => ColorPrimaries::Smpte431,
        F::SMPTE432 => ColorPrimaries::Smpte432,
        F::EBU3213 => ColorPrimaries::Ebu3213,
        F::Unspecified => ColorPrimaries::Unspecified,
        _ => ColorPrimaries::Other,
    }
}

fn transfer_from_ffmpeg(value: ffmpeg::color::TransferCharacteristic) -> TransferCharacteristic {
    use ffmpeg::color::TransferCharacteristic as F;
    match value {
        F::BT709 => TransferCharacteristic::Bt709,
        F::GAMMA22 => TransferCharacteristic::Gamma22,
        F::GAMMA28 => TransferCharacteristic::Gamma28,
        F::SMPTE170M => TransferCharacteristic::Smpte170M,
        F::SMPTE240M => TransferCharacteristic::Smpte240M,
        F::Linear => TransferCharacteristic::Linear,
        F::Log => TransferCharacteristic::Log,
        F::LogSqrt => TransferCharacteristic::LogSqrt,
        F::IEC61966_2_4 => TransferCharacteristic::Iec61966_2_4,
        F::BT1361_ECG => TransferCharacteristic::Bt1361Ecg,
        F::IEC61966_2_1 => TransferCharacteristic::Srgb,
        F::BT2020_10 => TransferCharacteristic::Bt2020_10,
        F::BT2020_12 => TransferCharacteristic::Bt2020_12,
        F::SMPTE2084 => TransferCharacteristic::Pq,
        F::SMPTE428 => TransferCharacteristic::Smpte428,
        F::ARIB_STD_B67 => TransferCharacteristic::Hlg,
        F::Unspecified => TransferCharacteristic::Unspecified,
        _ => TransferCharacteristic::Other,
    }
}

fn chroma_from_ffmpeg(value: ffmpeg::chroma::Location) -> ChromaLocation {
    use ffmpeg::chroma::Location as F;
    match value {
        F::Left => ChromaLocation::Left,
        F::Center => ChromaLocation::Center,
        F::TopLeft => ChromaLocation::TopLeft,
        F::Top => ChromaLocation::Top,
        F::BottomLeft => ChromaLocation::BottomLeft,
        F::Bottom => ChromaLocation::Bottom,
        F::Unspecified => ChromaLocation::Unspecified,
    }
}

fn ffmpeg_field_order(raw: ffmpeg::ffi::AVFieldOrder) -> Option<SourceFieldOrder> {
    match ffmpeg::FieldOrder::from(raw) {
        ffmpeg::FieldOrder::Unknown => None,
        ffmpeg::FieldOrder::Progressive => Some(SourceFieldOrder::Progressive),
        ffmpeg::FieldOrder::TT => Some(SourceFieldOrder::TopCodedTopDisplayed),
        ffmpeg::FieldOrder::BB => Some(SourceFieldOrder::BottomCodedBottomDisplayed),
        ffmpeg::FieldOrder::TB => Some(SourceFieldOrder::TopCodedBottomDisplayed),
        ffmpeg::FieldOrder::BT => Some(SourceFieldOrder::BottomCodedTopDisplayed),
    }
}

fn rational_from_ffmpeg(value: ffmpeg::Rational) -> Option<BoundedRational> {
    BoundedRational::new(i64::from(value.numerator()), i64::from(value.denominator()))
}

fn parse_display_matrix(data: &[u8]) -> Option<DisplayMatrix> {
    if data.len() != DISPLAY_MATRIX_BYTES {
        return None;
    }
    let mut elements = [0_i32; 9];
    for (index, chunk) in data.chunks_exact(4).enumerate() {
        elements[index] = i32::from_ne_bytes(chunk.try_into().ok()?);
    }
    // The homogeneous term is 2.30 fixed point and must be nonzero. Reject
    // the all-zero/malformed matrix before any trigonometry or UV law sees it.
    if elements[..4].iter().all(|value| *value == 0) || elements[8] == 0 {
        return None;
    }
    Some(DisplayMatrix { elements })
}

fn parse_frame_cropping(data: &[u8], coded: PixelDimensions) -> Option<CleanAperture> {
    if data.len() != 16 {
        return None;
    }
    let read = |index: usize| {
        u32::from_le_bytes(data[index..index + 4].try_into().expect("fixed crop chunk"))
    };
    // FFmpeg contract order: top, bottom, left, right.
    CleanAperture {
        top: read(0),
        bottom: read(4),
        left: read(8),
        right: read(12),
    }
    .validate(coded)
}

fn apply_display_matrix(
    display: &mut SourceDisplayDescriptor,
    matrix: DisplayMatrix,
    provenance: DescriptorProvenance,
) {
    display.display_matrix = DescriptorValue::new(matrix, provenance);
    let orientation = orientation_from_matrix(matrix.elements);
    if orientation != NormalizedOrientation::Unspecified {
        display.orientation = DescriptorValue::new(orientation, provenance);
        display.rotation = DescriptorValue::new(orientation.rotation(), provenance);
        display.mirror = DescriptorValue::new(orientation.mirror(), provenance);
        return;
    }
    let angle = unsafe { ffmpeg::ffi::av_display_rotation_get(matrix.elements.as_ptr()) };
    if angle.is_finite() {
        let normalized = angle.rem_euclid(360.0);
        let milli = (normalized * 1_000.0).round();
        if milli >= i32::MIN as f64 && milli <= i32::MAX as f64 {
            display.rotation =
                DescriptorValue::new(Rotation::OtherMilliDegrees(milli as i32), provenance);
        }
    }
    let determinant = i64::from(matrix.elements[0]) * i64::from(matrix.elements[4])
        - i64::from(matrix.elements[1]) * i64::from(matrix.elements[3]);
    display.mirror = DescriptorValue::new(
        if determinant < 0 {
            Mirror::Horizontal
        } else {
            Mirror::None
        },
        provenance,
    );
}

fn orientation_from_matrix(matrix: [i32; 9]) -> NormalizedOrientation {
    let a = sign_component(matrix[0]);
    let b = sign_component(matrix[1]);
    let c = sign_component(matrix[3]);
    let d = sign_component(matrix[4]);
    match (a, b, c, d) {
        (1, 0, 0, 1) => NormalizedOrientation::Identity,
        (-1, 0, 0, 1) => NormalizedOrientation::MirrorHorizontal,
        (-1, 0, 0, -1) => NormalizedOrientation::Rotate180,
        (1, 0, 0, -1) => NormalizedOrientation::MirrorVertical,
        (0, 1, 1, 0) => NormalizedOrientation::Transpose,
        (0, -1, 1, 0) => NormalizedOrientation::Rotate90,
        (0, -1, -1, 0) => NormalizedOrientation::Transverse,
        (0, 1, -1, 0) => NormalizedOrientation::Rotate270,
        _ => NormalizedOrientation::Unspecified,
    }
}

fn sign_component(value: i32) -> i8 {
    const NOISE: i32 = 64;
    if value > NOISE {
        1
    } else if value < -NOISE {
        -1
    } else {
        0
    }
}

fn normalized_image_orientation(value: image::metadata::Orientation) -> NormalizedOrientation {
    use image::metadata::Orientation as I;
    match value {
        I::NoTransforms => NormalizedOrientation::Identity,
        I::FlipHorizontal => NormalizedOrientation::MirrorHorizontal,
        I::Rotate180 => NormalizedOrientation::Rotate180,
        I::FlipVertical => NormalizedOrientation::MirrorVertical,
        I::Rotate90FlipH => NormalizedOrientation::Transpose,
        I::Rotate90 => NormalizedOrientation::Rotate90,
        I::Rotate270FlipH => NormalizedOrientation::Transverse,
        I::Rotate270 => NormalizedOrientation::Rotate270,
    }
}

fn sws_coefficients(matrix: MatrixCoefficients) -> Option<i32> {
    match matrix {
        MatrixCoefficients::Bt709 => Some(ffmpeg::ffi::SWS_CS_ITU709),
        MatrixCoefficients::Fcc => Some(ffmpeg::ffi::SWS_CS_FCC),
        MatrixCoefficients::Bt470Bg | MatrixCoefficients::Smpte170M => {
            Some(ffmpeg::ffi::SWS_CS_ITU601)
        }
        MatrixCoefficients::Smpte240M => Some(ffmpeg::ffi::SWS_CS_SMPTE240M),
        MatrixCoefficients::Bt2020Ncl | MatrixCoefficients::Bt2020Cl => {
            Some(ffmpeg::ffi::SWS_CS_BT2020)
        }
        _ => None,
    }
}

fn gcd_u64(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left.max(1)
}

/// Deterministic CPU upload reference for color-bar, chroma-edge, and ramp
/// fixtures. The GPU samples these resulting RGBA8 bytes; P4b does not enter
/// P4c's planar GPU conversion path.
#[cfg(test)]
fn yuv_to_rgba8_reference(
    y: u16,
    u: u16,
    v: u16,
    depth: u8,
    range: SourceColorRange,
    matrix: MatrixCoefficients,
) -> [u8; 4] {
    let max = ((1_u32 << depth) - 1) as f64;
    let scale = (1_u32 << depth.saturating_sub(8)) as f64;
    let (y_min, y_max, c_mid, c_span) = match range {
        SourceColorRange::Limited => (16.0 * scale, 235.0 * scale, 128.0 * scale, 224.0 * scale),
        SourceColorRange::Full | SourceColorRange::Unspecified => (0.0, max, max / 2.0, max),
    };
    let y = ((f64::from(y) - y_min) / (y_max - y_min)).clamp(0.0, 1.0);
    let cb = (f64::from(u) - c_mid) / c_span;
    let cr = (f64::from(v) - c_mid) / c_span;
    let (kr, kb) = match matrix {
        MatrixCoefficients::Bt709 => (0.2126, 0.0722),
        MatrixCoefficients::Bt2020Ncl | MatrixCoefficients::Bt2020Cl => (0.2627, 0.0593),
        _ => (0.299, 0.114),
    };
    let kg = 1.0 - kr - kb;
    let r = y + 2.0 * (1.0 - kr) * cr;
    let b = y + 2.0 * (1.0 - kb) * cb;
    let g = y - 2.0 * kb * (1.0 - kb) / kg * cb - 2.0 * kr * (1.0 - kr) / kg * cr;
    let byte = |value: f64| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
    [byte(r), byte(g), byte(b), 255]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn display_fixture(orientation: NormalizedOrientation) -> SourceDisplayDescriptor {
        let mut descriptor = SourceDisplayDescriptor {
            coded_dimensions: DescriptorValue::new(
                PixelDimensions::new(720, 480),
                DescriptorProvenance::CodecDeclared,
            ),
            sample_aspect_ratio: DescriptorValue::new(
                BoundedRational::new(40, 33).unwrap(),
                DescriptorProvenance::CodecDeclared,
            ),
            clean_aperture: DescriptorValue::new(
                CleanAperture {
                    left: 8,
                    top: 4,
                    right: 8,
                    bottom: 4,
                },
                DescriptorProvenance::ContainerDeclared,
            ),
            orientation: DescriptorValue::new(orientation, DescriptorProvenance::ContainerDeclared),
            rotation: DescriptorValue::new(
                orientation.rotation(),
                DescriptorProvenance::ContainerDeclared,
            ),
            mirror: DescriptorValue::new(
                orientation.mirror(),
                DescriptorProvenance::ContainerDeclared,
            ),
            ..Default::default()
        };
        descriptor.refresh_display_dimensions();
        descriptor
    }

    #[test]
    fn rational_display_law_preserves_sar_and_clean_aperture_exactly() {
        let normal = display_fixture(NormalizedOrientation::Identity);
        assert_eq!(
            normal.display_dimensions.value,
            PixelDimensions::new(704, 472)
        );
        assert_eq!(
            normal.display_aspect_ratio(),
            BoundedRational::new(28_160, 15_576)
        );
        let rotated = display_fixture(NormalizedOrientation::Rotate90);
        assert_eq!(
            rotated.display_dimensions.value,
            PixelDimensions::new(472, 704)
        );
        assert_eq!(
            rotated.display_aspect_ratio(),
            BoundedRational::new(15_576, 28_160)
        );
        assert!(BoundedRational::new(1_000_001, 1).is_none());
        assert!(BoundedRational::new(1, 0).is_none());
    }

    #[test]
    fn all_exif_orientations_are_normalized_without_rewriting_pixels() {
        let expected = [
            NormalizedOrientation::Identity,
            NormalizedOrientation::MirrorHorizontal,
            NormalizedOrientation::Rotate180,
            NormalizedOrientation::MirrorVertical,
            NormalizedOrientation::Transpose,
            NormalizedOrientation::Rotate90,
            NormalizedOrientation::Transverse,
            NormalizedOrientation::Rotate270,
        ];
        for (index, expected) in expected.into_iter().enumerate() {
            let image = image::metadata::Orientation::from_exif(index as u8 + 1).unwrap();
            assert_eq!(normalized_image_orientation(image), expected);
            let descriptor = display_fixture(expected);
            let reference = SourceUvReference::from_descriptor(descriptor).unwrap();
            for corner in [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]] {
                let mapped = reference.map_display_to_coded(corner);
                assert!(mapped.into_iter().all(|value| (0.0..=1.0).contains(&value)));
            }
        }
    }

    #[test]
    fn display_matrix_parser_is_exact_bounded_and_classifies_rotation_mirror() {
        let matrices = [
            (
                [65_536, 0, 0, 0, 65_536, 0, 0, 0, 1 << 30],
                NormalizedOrientation::Identity,
            ),
            (
                [0, -65_536, 0, 65_536, 0, 0, 0, 0, 1 << 30],
                NormalizedOrientation::Rotate90,
            ),
            (
                [-65_536, 0, 0, 0, -65_536, 0, 0, 0, 1 << 30],
                NormalizedOrientation::Rotate180,
            ),
            (
                [0, 65_536, 0, -65_536, 0, 0, 0, 0, 1 << 30],
                NormalizedOrientation::Rotate270,
            ),
            (
                [-65_536, 0, 0, 0, 65_536, 0, 0, 0, 1 << 30],
                NormalizedOrientation::MirrorHorizontal,
            ),
            (
                [0, 65_536, 0, 65_536, 0, 0, 0, 0, 1 << 30],
                NormalizedOrientation::Transpose,
            ),
        ];
        for (elements, expected) in matrices {
            let bytes = elements
                .into_iter()
                .flat_map(i32::to_ne_bytes)
                .collect::<Vec<_>>();
            let parsed = parse_display_matrix(&bytes).unwrap();
            assert_eq!(orientation_from_matrix(parsed.elements), expected);
        }
        assert!(parse_display_matrix(&[0; DISPLAY_MATRIX_BYTES - 1]).is_none());
        assert!(parse_display_matrix(&[0; DISPLAY_MATRIX_BYTES]).is_none());
    }

    #[test]
    fn color_bars_chroma_edges_and_8_10_bit_ramps_are_deterministic() {
        for depth in [8, 10] {
            let shift = depth - 8;
            for (matrix, range) in [
                (MatrixCoefficients::Smpte170M, SourceColorRange::Limited),
                (MatrixCoefficients::Bt709, SourceColorRange::Limited),
                (MatrixCoefficients::Bt709, SourceColorRange::Full),
                (MatrixCoefficients::Bt2020Ncl, SourceColorRange::Limited),
            ] {
                let black = yuv_to_rgba8_reference(
                    16 << shift,
                    128 << shift,
                    128 << shift,
                    depth,
                    range,
                    matrix,
                );
                let white = yuv_to_rgba8_reference(
                    235 << shift,
                    128 << shift,
                    128 << shift,
                    depth,
                    range,
                    matrix,
                );
                assert_eq!(black[3], 255);
                assert_eq!(white[3], 255);
                assert!(black[0] <= white[0]);
                let blue_edge = yuv_to_rgba8_reference(
                    128 << shift,
                    240 << shift,
                    16 << shift,
                    depth,
                    range,
                    matrix,
                );
                let red_edge = yuv_to_rgba8_reference(
                    128 << shift,
                    16 << shift,
                    240 << shift,
                    depth,
                    range,
                    matrix,
                );
                assert_ne!(blue_edge, red_edge);
                let ramp = (16_u16..=235)
                    .step_by(17)
                    .map(|y| {
                        yuv_to_rgba8_reference(
                            y << shift,
                            128 << shift,
                            128 << shift,
                            depth,
                            range,
                            matrix,
                        )[0]
                    })
                    .collect::<Vec<_>>();
                assert!(ramp.windows(2).all(|pair| pair[0] <= pair[1]));
            }
        }
    }

    #[test]
    fn descriptor_serde_defaults_are_backward_compatible_and_do_not_lie() {
        let restored: SourceColorDescriptor = serde_json::from_str("{}").unwrap();
        assert_eq!(restored, SourceColorDescriptor::default());
        assert!(!restored.range.provenance.is_declared());
        let declared = DescriptorValue::new(
            SourceColorRange::Limited,
            DescriptorProvenance::CodecDeclared,
        );
        assert!(declared.provenance.is_declared());
        let inferred = DescriptorValue::new(
            SourceColorRange::Limited,
            DescriptorProvenance::InferredFallback,
        );
        assert!(!inferred.provenance.is_declared());
    }

    #[test]
    fn unspecified_policy_is_the_byte_exact_legacy_scaler_path() {
        ffmpeg::init().unwrap();
        let mut source = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::YUV420P, 4, 4);
        source.data_mut(0).fill(81);
        source.data_mut(1).fill(90);
        source.data_mut(2).fill(240);
        let make_scaler = || {
            ffmpeg::software::scaling::Context::get(
                ffmpeg::format::Pixel::YUV420P,
                4,
                4,
                ffmpeg::format::Pixel::RGBA,
                4,
                4,
                ffmpeg::software::scaling::flag::Flags::BILINEAR,
            )
            .unwrap()
        };
        let mut historical = make_scaler();
        let mut candidate = make_scaler();
        let policy = configure_sws_conversion(&mut candidate, SourceColorDescriptor::default());
        assert_eq!(policy.kind, ConversionPolicyKind::LegacyUnspecified);
        let mut historical_rgba = ffmpeg::frame::Video::empty();
        let mut candidate_rgba = ffmpeg::frame::Video::empty();
        historical.run(&source, &mut historical_rgba).unwrap();
        candidate.run(&source, &mut candidate_rgba).unwrap();
        assert_eq!(historical_rgba.stride(0), candidate_rgba.stride(0));
        assert_eq!(historical_rgba.data(0), candidate_rgba.data(0));
    }

    #[test]
    fn declared_601_709_and_2020_limited_full_policies_are_explicit() {
        ffmpeg::init().unwrap();
        for (matrix, range) in [
            (MatrixCoefficients::Smpte170M, SourceColorRange::Limited),
            (MatrixCoefficients::Bt709, SourceColorRange::Limited),
            (MatrixCoefficients::Bt709, SourceColorRange::Full),
            (MatrixCoefficients::Bt2020Ncl, SourceColorRange::Limited),
        ] {
            let mut scaler = ffmpeg::software::scaling::Context::get(
                ffmpeg::format::Pixel::YUV420P10LE,
                4,
                4,
                ffmpeg::format::Pixel::RGBA,
                4,
                4,
                ffmpeg::software::scaling::flag::Flags::BILINEAR,
            )
            .unwrap();
            let descriptor = SourceColorDescriptor {
                matrix: DescriptorValue::new(matrix, DescriptorProvenance::CodecDeclared),
                range: DescriptorValue::new(range, DescriptorProvenance::CodecDeclared),
                ..Default::default()
            };
            let policy = configure_sws_conversion(&mut scaler, descriptor);
            assert_eq!(
                policy.kind,
                ConversionPolicyKind::ExplicitDeclaredMatrixRange,
                "{matrix:?}/{range:?} must not silently use converter defaults"
            );
            assert_eq!(policy.input_matrix, matrix);
            assert_eq!(policy.input_range, range);
        }
    }

    struct TemporaryDescriptorFixture {
        root: std::path::PathBuf,
        video: std::path::PathBuf,
    }

    impl TemporaryDescriptorFixture {
        fn create() -> Result<Self, String> {
            let root = std::env::temp_dir().join(format!(
                "collideoscope-source-descriptor-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_err(|error| error.to_string())?
                    .as_nanos()
            ));
            std::fs::create_dir(&root).map_err(|error| error.to_string())?;
            let video = root.join("bt709-limited-10bit-sar.mkv");
            let output = std::process::Command::new(crate::host_paths::ffmpeg())
                .args([
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "smptebars=size=64x48:rate=2:duration=1",
                    "-vf",
                    "setsar=40/33",
                    "-c:v",
                    "ffv1",
                    "-pix_fmt",
                    "yuv420p10le",
                    "-color_range",
                    "tv",
                    "-colorspace",
                    "bt709",
                    "-color_primaries",
                    "bt709",
                    "-color_trc",
                    "bt709",
                    "-y",
                ])
                .arg(&video)
                .output()
                .map_err(|error| error.to_string())?;
            if !output.status.success() {
                let error = String::from_utf8_lossy(&output.stderr).into_owned();
                let _ = std::fs::remove_dir_all(&root);
                return Err(error);
            }
            Ok(Self { root, video })
        }
    }

    impl Drop for TemporaryDescriptorFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    #[ignore = "requires an ffmpeg executable; creates one bounded temporary metadata fixture"]
    fn live_and_offline_freeze_the_same_descriptor_policy_and_pixels() {
        let fixture = TemporaryDescriptorFixture::create().unwrap();
        let path = fixture.video.to_string_lossy();
        let mut live = crate::video::VideoDecoder::open(&path).unwrap();
        let live_frame = live.next_timed_frame_result(7).unwrap();
        let mut offline = crate::video::VideoDecoder::open(&path).unwrap();
        let offline_frame = offline.seek_decode_for_generation(0.0, 7).unwrap();
        assert_eq!(
            live.source_color_descriptor(),
            offline.source_color_descriptor()
        );
        assert_eq!(
            live.source_display_descriptor(),
            offline.source_display_descriptor()
        );
        assert_eq!(live.conversion_policy(), offline.conversion_policy());
        assert_eq!(
            live.conversion_policy().kind,
            ConversionPolicyKind::ExplicitDeclaredMatrixRange
        );
        assert_eq!(
            live.source_color_descriptor().bit_depth.value,
            BitDepth::Bits(10)
        );
        assert_eq!(
            live.source_display_descriptor().sample_aspect_ratio.value,
            BoundedRational::new(40, 33).unwrap()
        );
        assert_eq!(live_frame.rgba, offline_frame.rgba);
    }
}
