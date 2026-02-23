// Split a 1D array into two outputs at a given index.
//
// out_a[i] = input[i]              if i < split_at
// out_b[i - split_at] = input[i]   otherwise

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_a: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_b: array<f32>;

struct Params { N: u32, split_at: u32 }
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.N) { return; }

  if (idx < params.split_at) {
    out_a[idx] = input[idx];
  } else {
    out_b[idx - params.split_at] = input[idx];
  }
}
