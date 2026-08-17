pub mod codec_motion;
pub mod decoder;
pub mod indexed;
pub mod still;
pub mod threaded;

#[allow(
    unused_imports,
    reason = "M4 host/GPU consumers land after the frozen decoder product"
)]
pub use codec_motion::{
    CodecMotionFrame, CodecMotionFrameType, CodecMotionProvenance, CodecMotionRejectReason,
    CodecMotionStatus,
};
pub use decoder::VideoDecoder;
pub use indexed::{DecodeWorkError, DecodedVideoFrame, FrameMetadata};
#[allow(unused_imports)]
pub use still::{
    decode_still_image, decode_still_image_with_media_policy,
    probe_still_image_dimensions_with_media_policy, DecodedStillImage, StillImage,
};
pub use threaded::ThreadedDecoder;
