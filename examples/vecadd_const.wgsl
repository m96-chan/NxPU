@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

var<private> bias: array<f32, 4> = array(0.1, 0.2, 0.3, 0.4);

struct Params { N: u32 }
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.N) { return; }
    output[i] = input[i] + bias[i];
}
