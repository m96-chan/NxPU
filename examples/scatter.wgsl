// Scatter: output = data; output[indices[i]] = updates[i]
@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> updates: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

struct Params { N: u32, data_len: u32 }
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.N) { return; }
  output[indices[idx]] = updates[idx];
}
