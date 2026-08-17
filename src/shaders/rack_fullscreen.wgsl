// Collision Rack fullscreen triangle. Kept local to the rack executor so its
// fixed pipeline can be constructed and tested without the legacy renderer.

struct RackVertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> RackVertexOutput {
    let x = f32(i32(index & 1u)) * 4.0 - 1.0;
    let y = f32(i32(index >> 1u)) * 4.0 - 1.0;
    var output: RackVertexOutput;
    output.position = vec4f(x, y, 0.0, 1.0);
    output.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return output;
}
