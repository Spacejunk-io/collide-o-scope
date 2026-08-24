//! Bounded, one-shot decoding for still-image layer sources.
//!
//! Still images are decoded once into straight-alpha RGBA and then held by the
//! layer texture. They deliberately have no transport clock: live playback and
//! frame-indexed export therefore sample the exact same pixels on every frame.

use image::{ImageDecoder as _, ImageFormat, ImageReader, Limits};
use std::path::Path;

use crate::media_safety::{
    MediaAllocationPlan, MediaDeviceLimits, MediaReservation, MediaSafetyPolicy, MediaSourceKind,
    ABSOLUTE_MEDIA_MAX_EDGE,
};
#[cfg(test)]
use crate::video::source_descriptor::DescriptorProvenance;
use crate::video::source_descriptor::{
    still_descriptors, SourceColorDescriptor, SourceConversionPolicy, SourceDisplayDescriptor,
};

/// An image decoded into the engine's source-texture representation.
#[derive(Debug)]
pub struct DecodedStillImage {
    pub width: u32,
    pub height: u32,
    /// Tightly packed, straight-alpha RGBA8 in top-to-bottom row order.
    pub rgba: Vec<u8>,
    pub source_color_descriptor: SourceColorDescriptor,
    pub source_display_descriptor: SourceDisplayDescriptor,
    pub conversion_policy: SourceConversionPolicy,
    media_reservation: MediaReservation,
}

impl DecodedStillImage {
    /// Construct already-decoded pixels under the exact legacy Safe policy.
    /// Primarily useful for deterministic synthetic/test sources.
    #[cfg(test)]
    pub fn from_rgba(width: u32, height: u32, rgba: Vec<u8>) -> Result<Self, String> {
        let policy = MediaSafetyPolicy::safe();
        let reservation = policy
            .reserve_source(
                MediaSourceKind::Still,
                width,
                height,
                MediaDeviceLimits::none(),
            )
            .map_err(|error| error.to_string())?;
        validate_rgba_length(width, height, rgba.len())?;
        let (mut source_color_descriptor, mut source_display_descriptor, conversion_policy) =
            still_descriptors(width, height, image::ColorType::Rgba8, None);
        source_color_descriptor.pixel_family.provenance = DescriptorProvenance::InferredFallback;
        source_color_descriptor.bit_depth.provenance = DescriptorProvenance::InferredFallback;
        source_display_descriptor.coded_dimensions.provenance =
            DescriptorProvenance::InferredFallback;
        Ok(Self {
            width,
            height,
            rgba,
            source_color_descriptor,
            source_display_descriptor,
            conversion_policy,
            media_reservation: reservation,
        })
    }

    pub fn media_allocation_plan(&self) -> &MediaAllocationPlan {
        self.media_reservation.plan()
    }
}

impl PartialEq for DecodedStillImage {
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width
            && self.height == other.height
            && self.rgba == other.rgba
            && self.source_color_descriptor == other.source_color_descriptor
            && self.source_display_descriptor == other.source_display_descriptor
            && self.conversion_policy == other.conversion_policy
    }
}

impl Eq for DecodedStillImage {}

/// A live still source publishes its pixels exactly once for texture upload.
/// Keeping this mailbox local avoids a decoder thread for immutable content.
pub struct StillImage {
    frame: Option<Vec<u8>>,
    source_color_descriptor: SourceColorDescriptor,
    source_display_descriptor: SourceDisplayDescriptor,
    conversion_policy: SourceConversionPolicy,
    _media_reservation: MediaReservation,
}

impl StillImage {
    pub fn from_decoded(decoded: DecodedStillImage) -> Self {
        let DecodedStillImage {
            rgba,
            source_color_descriptor,
            source_display_descriptor,
            conversion_policy,
            media_reservation,
            ..
        } = decoded;
        Self {
            frame: Some(rgba),
            source_color_descriptor,
            source_display_descriptor,
            conversion_policy,
            _media_reservation: media_reservation,
        }
    }

    pub fn take_frame(&mut self) -> Option<Vec<u8>> {
        self.frame.take()
    }

    pub const fn source_color_descriptor(&self) -> SourceColorDescriptor {
        self.source_color_descriptor
    }

    pub const fn source_display_descriptor(&self) -> SourceDisplayDescriptor {
        self.source_display_descriptor
    }

    pub const fn conversion_policy(&self) -> SourceConversionPolicy {
        self.conversion_policy
    }

    /// A recoverable GPU upload failure must not consume the immutable source
    /// forever. Restore only into the empty one-slot mailbox so a successful
    /// or newer publication can never be overwritten.
    pub fn restore_frame_after_failed_upload(&mut self, frame: Vec<u8>) {
        if self.frame.is_none() {
            self.frame = Some(frame);
        }
    }
}

/// Decode one supported still image with strict edge, aggregate-pixel, and
/// allocation limits. Format is detected from file contents after the caller
/// has classified the library extension.
/// Safe-policy compatibility wrapper retained for embedders and tests.
#[allow(dead_code)]
pub fn decode_still_image(
    path: &Path,
    adapter_max_dimension: Option<u32>,
) -> Result<DecodedStillImage, String> {
    let media_policy = MediaSafetyPolicy::safe();
    decode_still_image_with_media_policy(
        path,
        &media_policy,
        MediaDeviceLimits {
            max_texture_dimension_2d: adapter_max_dimension,
            max_buffer_size: None,
        },
    )
}

/// Policy-aware still decode. A reservation is acquired before the decoder is
/// allowed to allocate and remains attached to the returned image until a
/// live [`StillImage`] drops it.
pub fn decode_still_image_with_media_policy(
    path: &Path,
    media_policy: &MediaSafetyPolicy,
    device_limits: MediaDeviceLimits,
) -> Result<DecodedStillImage, String> {
    let reader = open_reader(path)?;
    let format = reader
        .format()
        .ok_or_else(|| format!("cannot determine still-image format for {}", path.display()))?;
    if !is_supported_format(format) {
        return Err(format!(
            "unsupported still-image data in {} (expected PNG, JPEG, BMP, or WebP)",
            path.display()
        ));
    }

    let (width, height) = reader.into_dimensions().map_err(|error| {
        format!(
            "cannot read still-image dimensions from {}: {error}",
            path.display()
        )
    })?;
    let media_reservation = media_policy
        .reserve_source(MediaSourceKind::Still, width, height, device_limits)
        .map_err(|error| format!("still image {} rejected: {error}", path.display()))?;
    let media_plan = media_reservation.plan();

    // Reopen after probing because `into_dimensions` consumes the reader.
    // Strict image limits guard a file replacement between the two opens; the
    // post-decode equality check makes that race a deterministic error.
    let mut reader = open_reader(path)?;
    let mut limits = Limits::default();
    let edge_limit = device_limits
        .max_texture_dimension_2d
        .unwrap_or(ABSOLUTE_MEDIA_MAX_EDGE)
        .min(ABSOLUTE_MEDIA_MAX_EDGE);
    limits.max_image_width = Some(edge_limit);
    limits.max_image_height = Some(edge_limit);
    limits.max_alloc = Some(
        media_plan
            .still_decoder_allocation_limit()
            .map_err(|error| format!("still image {} rejected: {error}", path.display()))?,
    );
    reader.limits(limits);

    // Inspect bounded header/EXIF truth without applying it to the raster.
    // The historical byte path remains in coded orientation; display intent
    // travels only in the normalized descriptor.
    let (source_color_type, source_orientation) = {
        let mut decoder = reader.into_decoder().map_err(|error| {
            format!(
                "cannot inspect still-image metadata for {} within safety limits: {error}",
                path.display()
            )
        })?;
        let color_type = decoder.color_type();
        let orientation = decoder
            .exif_metadata()
            .ok()
            .flatten()
            .as_deref()
            .and_then(image::metadata::Orientation::from_exif_chunk);
        (color_type, orientation)
    };

    // Metadata inspection consumes the reader. Reopen with the identical
    // bounds; the dimension equality below rejects a replacement race.
    let mut reader = open_reader(path)?;
    let mut limits = Limits::default();
    limits.max_image_width = Some(edge_limit);
    limits.max_image_height = Some(edge_limit);
    limits.max_alloc = Some(
        media_plan
            .still_decoder_allocation_limit()
            .map_err(|error| format!("still image {} rejected: {error}", path.display()))?,
    );
    reader.limits(limits);

    let image = reader.decode().map_err(|error| {
        format!(
            "cannot decode still image {} within safety limits: {error}",
            path.display()
        )
    })?;
    if image.width() != width || image.height() != height {
        return Err(format!(
            "still image {} changed while opening: expected {width}x{height}, decoded {}x{}",
            path.display(),
            image.width(),
            image.height()
        ));
    }
    let rgba = image.into_rgba8().into_raw();
    validate_rgba_length(width, height, rgba.len())
        .map_err(|error| format!("still image {} rejected: {error}", path.display()))?;

    let (source_color_descriptor, source_display_descriptor, conversion_policy) =
        still_descriptors(width, height, source_color_type, source_orientation);

    Ok(DecodedStillImage {
        width,
        height,
        rgba,
        source_color_descriptor,
        source_display_descriptor,
        conversion_policy,
        media_reservation,
    })
}

/// Probe and plan a still source without decoding or reserving it. This is
/// suitable for previews/status; constructors must use the decode function
/// above so admission and reservation are atomic.
#[allow(dead_code)]
pub fn probe_still_image_dimensions_with_media_policy(
    path: &Path,
    media_policy: &MediaSafetyPolicy,
    device_limits: MediaDeviceLimits,
) -> Result<MediaAllocationPlan, String> {
    let reader = open_reader(path)?;
    let format = reader
        .format()
        .ok_or_else(|| format!("cannot determine still-image format for {}", path.display()))?;
    if !is_supported_format(format) {
        return Err(format!(
            "unsupported still-image data in {} (expected PNG, JPEG, BMP, or WebP)",
            path.display()
        ));
    }
    let (width, height) = reader.into_dimensions().map_err(|error| {
        format!(
            "cannot read still-image dimensions from {}: {error}",
            path.display()
        )
    })?;
    media_policy
        .plan(MediaSourceKind::Still, width, height, device_limits)
        .map_err(|error| format!("still image {} rejected: {error}", path.display()))
}

fn validate_rgba_length(width: u32, height: u32, actual_len: usize) -> Result<(), String> {
    let expected_u64 = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| format!("still-image RGBA size overflows for {width}x{height}"))?;
    let expected_len = usize::try_from(expected_u64)
        .map_err(|_| format!("still-image RGBA size does not fit memory for {width}x{height}"))?;
    if actual_len != expected_len {
        return Err(format!(
            "decoded to {actual_len} RGBA bytes; expected {expected_len} for {width}x{height}"
        ));
    }
    Ok(())
}

fn open_reader(path: &Path) -> Result<ImageReader<std::io::BufReader<std::fs::File>>, String> {
    ImageReader::open(path)
        .map_err(|error| format!("cannot open still image {}: {error}", path.display()))?
        .with_guessed_format()
        .map_err(|error| format!("cannot inspect still image {}: {error}", path.display()))
}

fn is_supported_format(format: ImageFormat) -> bool {
    matches!(
        format,
        ImageFormat::Png | ImageFormat::Jpeg | ImageFormat::Bmp | ImageFormat::WebP
    )
}

#[cfg(test)]
mod tests {
    use super::{decode_still_image, DecodedStillImage, StillImage};
    use image::{DynamicImage, ImageBuffer, ImageEncoder as _, ImageFormat, Rgba};
    use std::io::Cursor;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_path(extension: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "collide-o-scope-still-{}-{nonce}.{extension}",
            std::process::id()
        ))
    }

    fn png_fixture() -> Vec<u8> {
        let pixels = vec![
            255, 0, 0, 255, // opaque red
            0, 64, 255, 37, // translucent blue
        ];
        let image = ImageBuffer::<Rgba<u8>, _>::from_raw(2, 1, pixels).unwrap();
        let mut bytes = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(image)
            .write_to(&mut bytes, ImageFormat::Png)
            .unwrap();
        bytes.into_inner()
    }

    fn minimal_exif_orientation(value: u8) -> Vec<u8> {
        vec![
            b'I', b'I', 42, 0, // little-endian TIFF header
            8, 0, 0, 0, // first IFD offset
            1, 0, // one directory entry
            0x12, 0x01, // orientation tag
            3, 0, // SHORT
            1, 0, 0, 0, // one value
            value, 0, 0, 0, // inline SHORT + padding
            0, 0, 0, 0, // no next IFD
        ]
    }

    #[test]
    fn png_decodes_once_to_exact_straight_alpha_rgba() {
        let path = unique_temp_path("png");
        std::fs::write(&path, png_fixture()).unwrap();

        let decoded = decode_still_image(&path, Some(8192)).unwrap();
        std::fs::remove_file(path).unwrap();

        assert_eq!((decoded.width, decoded.height), (2, 1));
        assert_eq!(decoded.rgba, [255, 0, 0, 255, 0, 64, 255, 37]);
    }

    #[test]
    fn still_source_publishes_once_and_can_retry_a_failed_gpu_upload() {
        let mut source = StillImage::from_decoded(
            DecodedStillImage::from_rgba(1, 1, vec![10, 20, 30, 40]).unwrap(),
        );
        let frame = source.take_frame().unwrap();
        assert_eq!(source.take_frame(), None);
        source.restore_frame_after_failed_upload(frame);
        assert_eq!(source.take_frame(), Some(vec![10, 20, 30, 40]));
        assert_eq!(source.take_frame(), None);
    }

    #[test]
    fn jpeg_exif_orientation_is_normalized_without_rotating_legacy_pixels() {
        let path = unique_temp_path("jpg");
        let file = std::fs::File::create(&path).unwrap();
        let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(file, 100);
        encoder
            .set_exif_metadata(minimal_exif_orientation(6))
            .unwrap();
        encoder
            .write_image(
                &[255, 0, 0, 0, 255, 0],
                2,
                1,
                image::ExtendedColorType::Rgb8,
            )
            .unwrap();

        let decoded = decode_still_image(&path, Some(8192)).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!((decoded.width, decoded.height), (2, 1));
        assert_eq!(
            decoded.source_display_descriptor.orientation.value,
            crate::video::NormalizedOrientation::Rotate90
        );
        assert_eq!(
            decoded.source_display_descriptor.orientation.provenance,
            crate::video::DescriptorProvenance::StillExifDeclared
        );
        assert_eq!(
            decoded.source_display_descriptor.display_dimensions.value,
            crate::video::PixelDimensions::new(1, 2)
        );
    }

    #[test]
    fn gpu_edge_limit_is_applied_before_pixel_decode() {
        let path = unique_temp_path("png");
        std::fs::write(&path, png_fixture()).unwrap();

        let error = decode_still_image(&path, Some(1)).unwrap_err();
        std::fs::remove_file(path).unwrap();

        assert!(error.contains("2x1"));
        assert!(error.contains("GPU"));
    }

    #[test]
    fn extension_cannot_make_unsupported_bytes_decodable() {
        let path = unique_temp_path("png");
        std::fs::write(&path, b"not an image").unwrap();

        let error = decode_still_image(&path, Some(8192)).unwrap_err();
        std::fs::remove_file(path).unwrap();

        let error = error.to_ascii_lowercase();
        assert!(error.contains("still"));
        assert!(error.contains("dimension") || error.contains("decode"));
    }
}
