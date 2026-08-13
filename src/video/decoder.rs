use ffmpeg_next as ffmpeg;
use ffmpeg_next::format::{context::Input, input};
use ffmpeg_next::media::Type;
use ffmpeg_next::software::scaling::{context::Context as ScalerContext, flag::Flags};
use ffmpeg_next::util::frame::video::Video as VideoFrame;

pub struct VideoDecoder {
    path: String,
    input_ctx: Input,
    stream_index: usize,
    decoder: ffmpeg::decoder::Video,
    scaler: ScalerContext,
    pub width: u32,
    pub height: u32,
    frame_count: u64,
    total_frames: u64,
}

impl VideoDecoder {
    pub fn open(path: &str) -> Result<Self, String> {
        ffmpeg::init().map_err(|e| format!("ffmpeg init failed: {e}"))?;

        let input_ctx = input(&path).map_err(|e| format!("Cannot open {path}: {e}"))?;

        let stream = input_ctx
            .streams()
            .best(Type::Video)
            .ok_or("No video stream found")?;
        let stream_index = stream.index();

        let codec_ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| format!("Codec params: {e}"))?;
        let decoder = codec_ctx
            .decoder()
            .video()
            .map_err(|e| format!("Decoder: {e}"))?;

        let width = decoder.width();
        let height = decoder.height();

        // Estimate total frames from stream duration and frame rate
        let total_frames = {
            let frames = stream.frames() as u64;
            if frames > 0 {
                frames
            } else {
                // Fallback: compute from duration and fps
                let duration_secs = input_ctx.duration() as f64 / f64::from(ffmpeg::ffi::AV_TIME_BASE);
                let fps = f64::from(stream.avg_frame_rate());
                let estimated = (duration_secs * fps) as u64;
                estimated.max(1)
            }
        };

        let scaler = ScalerContext::get(
            decoder.format(),
            width,
            height,
            ffmpeg::format::Pixel::RGBA,
            width,
            height,
            Flags::BILINEAR,
        )
        .map_err(|e| format!("Scaler: {e}"))?;

        Ok(Self {
            path: path.to_string(),
            input_ctx,
            stream_index,
            decoder,
            scaler,
            width,
            height,
            frame_count: 0,
            total_frames,
        })
    }

    /// Get the next decoded RGBA frame. Returns None only on unrecoverable error.
    /// Loops back to the start when the file ends.
    pub fn next_frame(&mut self) -> Option<Vec<u8>> {
        loop {
            // Try to receive already-decoded frames first
            let mut decoded = VideoFrame::empty();
            if self.decoder.receive_frame(&mut decoded).is_ok() {
                self.frame_count += 1;
                return Some(self.scale_frame(&decoded));
            }

            // Feed more packets to the decoder
            match self.next_video_packet() {
                Some(packet) => {
                    if self.decoder.send_packet(&packet).is_err() {
                        continue;
                    }
                    let mut decoded = VideoFrame::empty();
                    if self.decoder.receive_frame(&mut decoded).is_ok() {
                        self.frame_count += 1;
                        return Some(self.scale_frame(&decoded));
                    }
                }
                None => {
                    // EOF — flush decoder then loop the file
                    self.decoder.send_eof().ok();
                    let mut decoded = VideoFrame::empty();
                    if self.decoder.receive_frame(&mut decoded).is_ok() {
                        self.frame_count += 1;
                        return Some(self.scale_frame(&decoded));
                    }
                    // Reopen file to loop
                    if self.reopen().is_err() {
                        return None;
                    }
                }
            }
        }
    }

    /// Returns loop progress as 0.0..1.0
    pub fn progress(&self) -> f32 {
        if self.total_frames == 0 {
            return 0.0;
        }
        (self.frame_count % self.total_frames) as f32 / self.total_frames as f32
    }

    fn scale_frame(&mut self, frame: &VideoFrame) -> Vec<u8> {
        let mut rgb_frame = VideoFrame::empty();
        self.scaler.run(frame, &mut rgb_frame).unwrap();
        rgb_frame.data(0).to_vec()
    }

    fn next_video_packet(&mut self) -> Option<ffmpeg::Packet> {
        for (stream, packet) in self.input_ctx.packets() {
            if stream.index() == self.stream_index {
                return Some(packet);
            }
        }
        None
    }

    fn reopen(&mut self) -> Result<(), String> {
        self.input_ctx =
            input(&self.path).map_err(|e| format!("Cannot reopen {}: {e}", self.path))?;

        let stream = self
            .input_ctx
            .streams()
            .best(Type::Video)
            .ok_or("No video stream on reopen")?;

        let codec_ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| format!("Codec params on reopen: {e}"))?;
        self.decoder = codec_ctx
            .decoder()
            .video()
            .map_err(|e| format!("Decoder on reopen: {e}"))?;

        self.frame_count = 0;

        Ok(())
    }
}
