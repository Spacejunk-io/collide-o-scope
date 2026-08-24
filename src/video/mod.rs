pub mod codec_motion;
pub mod codec_motion_sequence;
pub mod decoder;
pub mod frame_selection;
pub mod hw_decode;
pub mod indexed;
pub mod payload;
pub mod planar;
pub mod retirement;
pub mod source_descriptor;
pub mod still;
pub mod threaded;

#[allow(
    unused_imports,
    reason = "M4 host/GPU consumers land after the frozen decoder product"
)]
pub use codec_motion::{
    AdjacentReferencePolicy, CodecFrameIdentity, CodecMotionFrame, CodecMotionFrameType,
    CodecMotionProvenance, CodecMotionRejectReason, CodecMotionStatus, CodecPastReferenceProof,
    CodecTimeBase,
};
pub use codec_motion_sequence::{
    CodecMotionProduct, CodecMotionProductIdentity, CodecMotionSequence,
};
pub use decoder::VideoDecoder;
pub use indexed::{DecodeWorkError, DecodedVideoFrame, FrameMetadata};
#[allow(unused_imports)]
pub use payload::{
    DecodedImagePayload, DecodedPayloadOwner, DecodedPayloadOwnerSnapshot, DecodedPixelFormat,
    DecodedRasterLayout,
};
#[allow(
    unused_imports,
    reason = "P4c planar delivery is an evidence-gated public prototype, not a promoted renderer path"
)]
pub use planar::{
    prototype_delivery_decision, CpuPlanarConversion, CpuPlanarConversionLaw,
    CpuPlanarConversionPolicy, DecodedDeliveryFrame, DecodedImageDelivery, PlanarAllocationBudget,
    PlanarAllocationSnapshot, PlanarConversionError, PlanarDeliveryDecision, PlanarDeliveryPolicy,
    PlanarDeliverySettings, PlanarFallbackReason, PlanarImageError, PlanarImageLayout,
    PlanarImagePayload, PlanarPixelFormat, PlanarPlane, PlanarPlaneInput, PlanarPlaneInputs,
    PlanarPlaneKind, PlanarPlaneLayout, MAX_CPU_REFERENCE_RGBA_BYTES, MAX_PLANAR_BUDGET_BYTES,
    MAX_PLANAR_FRAME_BYTES, MAX_PLANAR_PLANES,
};
#[allow(
    unused_imports,
    reason = "decoder retirement diagnostics are consumed by host status and shutdown seams"
)]
pub use retirement::{
    decoder_retirement_snapshot, drain_decoder_retirements_with_deadline,
    DecoderRetirementDrainReceipt, DecoderRetirementHealth, DecoderRetirementIdentity,
    DecoderRetirementSnapshot, DecoderSourceFingerprint, DECODER_RETIREMENT_CHURN_CAP,
    DECODER_RETIREMENT_SHUTDOWN_DEADLINE, DECODER_WORKER_HARD_CAP,
};
#[allow(
    unused_imports,
    reason = "the bounded descriptor vocabulary is the public video telemetry/provenance contract"
)]
pub use source_descriptor::{
    BitDepth, BoundedRational, ChromaLocation, ChromaSubsampling, CleanAperture, ColorPrimaries,
    ConversionPolicyKind, DescriptorProvenance, DescriptorValue, DisplayMatrix, MatrixCoefficients,
    Mirror, NormalizedOrientation, PixelDimensions, PixelFamily, Rotation, SourceColorDescriptor,
    SourceColorRange, SourceConversionPolicy, SourceDisplayDescriptor, SourceFieldOrder,
    SourceUvReference, TransferCharacteristic,
};
#[allow(unused_imports)]
pub use still::{
    decode_still_image, decode_still_image_with_media_policy,
    probe_still_image_dimensions_with_media_policy, DecodedStillImage, StillImage,
};
pub use threaded::{SeedSelectError, ThreadedDecoder};
