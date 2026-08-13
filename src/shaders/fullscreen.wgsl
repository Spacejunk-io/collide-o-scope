// Fullscreen triangle vertex shader — no vertex buffer needed.
// Draws a triangle that covers the entire screen; UVs calculated from vertex index.

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    // Generate 3 vertices that form a triangle covering clip space [-1,1]
    let x = f32(i32(idx & 1u)) * 4.0 - 1.0;
    let y = f32(i32(idx >> 1u)) * 4.0 - 1.0;

    var out: VertexOutput;
    out.position = vec4f(x, y, 0.0, 1.0);
    // Map from clip space to UV space (0,0 top-left to 1,1 bottom-right)
    out.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}
