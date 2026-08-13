//! Bounded, one-shot decoding for still-image layer sources.
//!
//! Still images are decoded once into straight-alpha RGBA and then held by the
//! layer texture. They deliberately have no transport clock: live playback and
//! frame-indexed export therefore sample the exact same pixels on every frame.

use image::{ImageFormat, ImageReader, Limits};
use std::path::Path;

use super::decoder::{validate_media_dimensions, MAX_MEDIA_EDGE, MAX_MEDIA_RGBA_BYTES};

/// Extra headroom for decoder working buffers. The decoded RGBA output is
/// independently bounded by `validate_media_dimensions` to UHD/33 MiB.
const MAX_STILL_DECODE_ALLOC_BYTES: u64 = MAX_MEDIA_RGBA_BYTES * 4;

/// An image decoded into the engine's source-texture representation.
#[derive(Debug, PartialEq, Eq)]
pub struct DecodedStillImage {
    pub width: u32,
    pub height: u32,
    /// Tightly packed, straight-alpha RGBA8 in top-to-bottom row order.
    pub rgba: Vec<u8>,
}

/// A live still source publishes its pixels exactly once for texture upload.
/// Keeping this mailbox local avoids a decoder thread for immutable content.
pub struct StillImage {
    frame: Option<Vec<u8>>,
}

impl StillImage {
    pub fn from_decoded(decoded: DecodedStillImage) -> Self {
        Self {
            frame: Some(decoded.rgba),
        }
    }

    pub fn take_frame(&mut self) -> Option<Vec<u8>> {
        self.frame.take()
    }
}

/// Decode one supported still image with strict edge, aggregate-pixel, and
/// allocation limits. Format is detected from file contents after the caller
/// has classified the library extension.
pub fn decode_still_image(
    path: &Path,
    adapter_max_dimension: Option<u32>,
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
    validate_media_dimensions(width, height, adapter_max_dimension)
        .map_err(|error| format!("still image {} rejected: {error}", path.display()))?;

    // Reopen after probing because `into_dimensions` consumes the reader.
    // Strict image limits guard a file replacement between the two opens; the
    // post-decode equality check makes that race a deterministic error.
    let mut reader = open_reader(path)?;
    let mut limits = Limits::default();
    let edge_limit = adapter_max_dimension
        .unwrap_or(MAX_MEDIA_EDGE)
        .min(MAX_MEDIA_EDGE);
    limits.max_image_width = Some(edge_limit);
    limits.max_image_height = Some(edge_limit);
    limits.max_alloc = Some(MAX_STILL_DECODE_ALLOC_BYTES);
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
    validate_media_dimensions(image.width(), image.height(), adapter_max_dimension)
        .map_err(|error| format!("still image {} rejected: {error}", path.display()))?;

    let rgba = image.into_rgba8().into_raw();
    let expected_len = usize::try_from(u64::from(width) * u64::from(height) * 4)
        .map_err(|_| format!("still-image RGBA size does not fit memory for {width}x{height}"))?;
    if rgba.len() != expected_len {
        return Err(format!(
            "still image {} decoded to {} RGBA bytes; expected {expected_len}",
            path.display(),
            rgba.len()
        ));
    }

    Ok(DecodedStillImage {
        width,
        height,
        rgba,
    })
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
    use image::{DynamicImage, ImageBuffer, ImageFormat, Rgba};
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
    fn still_source_publishes_once_then_holds_the_uploaded_texture() {
        let mut source = StillImage::from_decoded(DecodedStillImage {
            width: 1,
            height: 1,
            rgba: vec![10, 20, 30, 40],
        });
        assert_eq!(source.take_frame(), Some(vec![10, 20, 30, 40]));
        assert_eq!(source.take_frame(), None);
        assert_eq!(source.take_frame(), None);
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
