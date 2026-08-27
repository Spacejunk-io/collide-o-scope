struct PlanarUniforms {
    size: vec2<u32>,
    format: u32,
    bit_depth: u32,
    range_full: u32,
    _pad0: u32,
    chroma_offset: vec2<f32>,
    kr: f32,
    kb: f32,
    _pad1: vec2<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: PlanarUniforms;
@group(0) @binding(1) var luma_plane: texture_2d<u32>;
@group(0) @binding(2) var chroma_a: texture_2d<u32>;
@group(0) @binding(3) var chroma_b: texture_2d<u32>;

@vertex
fn vs_convert(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn luma_code(pixel: vec2<u32>) -> f32 {
    let word = textureLoad(luma_plane, vec2<i32>(pixel), 0).r;
    if uniforms.format == 2u {
        return f32(word >> 6u);
    }
    return f32(word);
}

fn chroma_pair_at(texel: vec2<i32>) -> vec2<f32> {
    if uniforms.format == 0u {
        return vec2<f32>(
            f32(textureLoad(chroma_a, texel, 0).r),
            f32(textureLoad(chroma_b, texel, 0).r),
        );
    }
    let pair = textureLoad(chroma_a, texel, 0).rg;
    if uniforms.format == 2u {
        return vec2<f32>(f32(pair.x >> 6u), f32(pair.y >> 6u));
    }
    return vec2<f32>(f32(pair.x), f32(pair.y));
}

fn chroma_code(pixel: vec2<u32>) -> vec2<f32> {
    let plane_size = textureDimensions(chroma_a);
    let plane_w = f32(plane_size.x);
    let plane_h = f32(plane_size.y);
    let x = clamp(
        (f32(pixel.x) - uniforms.chroma_offset.x) / 2.0,
        0.0,
        plane_w - 1.0,
    );
    let y = clamp(
        (f32(pixel.y) - uniforms.chroma_offset.y) / 2.0,
        0.0,
        plane_h - 1.0,
    );
    let x0 = floor(x);
    let y0 = floor(y);
    let x1 = min(x0 + 1.0, plane_w - 1.0);
    let y1 = min(y0 + 1.0, plane_h - 1.0);
    let tx = x - x0;
    let ty = y - y0;
    let c00 = chroma_pair_at(vec2<i32>(i32(x0), i32(y0)));
    let c10 = chroma_pair_at(vec2<i32>(i32(x1), i32(y0)));
    let c01 = chroma_pair_at(vec2<i32>(i32(x0), i32(y1)));
    let c11 = chroma_pair_at(vec2<i32>(i32(x1), i32(y1)));
    let top = c00 * (1.0 - tx) + c10 * tx;
    let bottom = c01 * (1.0 - tx) + c11 * tx;
    let max_code = f32((1u << uniforms.bit_depth) - 1u);
    return clamp(
        top * (1.0 - ty) + bottom * ty,
        vec2<f32>(0.0, 0.0),
        vec2<f32>(max_code, max_code),
    );
}

@fragment
fn fs_convert(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<u32>(u32(position.x), u32(position.y));
    let y_code = luma_code(pixel);
    let chroma = chroma_code(pixel);
    let bd = uniforms.bit_depth;
    let scale = f32(1u << (bd - 8u));
    let max_code = f32((1u << bd) - 1u);
    var y: f32;
    var cb: f32;
    var cr: f32;
    if uniforms.range_full == 1u {
        y = y_code / max_code;
        cb = (chroma.x - f32(1u << (bd - 1u))) / max_code;
        cr = (chroma.y - f32(1u << (bd - 1u))) / max_code;
    } else {
        y = (y_code - 16.0 * scale) / (219.0 * scale);
        cb = (chroma.x - 128.0 * scale) / (224.0 * scale);
        cr = (chroma.y - 128.0 * scale) / (224.0 * scale);
    }
    let kr = uniforms.kr;
    let kb = uniforms.kb;
    let kg = 1.0 - kr - kb;
    let red = y + (2.0 - 2.0 * kr) * cr;
    let blue = y + (2.0 - 2.0 * kb) * cb;
    let green = y - kb * (2.0 - 2.0 * kb) / kg * cb - kr * (2.0 - 2.0 * kr) / kg * cr;
    return vec4<f32>(
        clamp(vec3<f32>(red, green, blue), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0)),
        1.0,
    );
}
