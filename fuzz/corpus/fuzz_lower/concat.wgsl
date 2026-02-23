// Concatenate two 1D arrays along axis 0.
//
// output[i] = a[i]           if i < N_a
// output[i] = b[i - N_a]    otherwise

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params { N_a: u32, N_b: u32 }
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.N_a + params.N_b;
  if (idx >= total) { return; }

  if (idx < params.N_a) {
    output[idx] = a[idx];
  } else {
    output[idx] = b[idx - params.N_a];
  }
}
