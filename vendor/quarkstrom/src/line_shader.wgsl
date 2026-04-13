struct View {
    position: vec2<f32>,
    scale: f32,
    xy: u32,
};

@group(0)
@binding(0)
var<uniform> view: View;

struct VertexOutput {
    @builtin(position) clip_space: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
) -> VertexOutput {
    var result: VertexOutput;

    let x = (view.xy       ) & 0xffffu;
    let y = (view.xy >> 16u) & 0xffffu;
    let aspect = f32(y) / f32(x);

    let view_space = (position - view.position) / view.scale;
    let clip_space = vec4(view_space.x * aspect, view_space.y, 0.0, 1.0);

    result.clip_space = clip_space;
    result.color = vec4(pow(color.rgb, vec3(2.2)), color.a);
    return result;
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    return vertex.color;
}