// Concatenate two 2D arrays along axis 1 (channel dimension).
//
// For a tensor with shape [N, C], the boundary is at params.C1:
//   result[n, c] = a[n, c]           if c < C1
//   result[n, c] = b[n, c - C1]     otherwise

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Params { N: u32, C1: u32, C2: u32 }
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x / (params.C1 + params.C2);
  let c = gid.x % (params.C1 + params.C2);
  if c < params.C1 {
    result[gid.x] = a[n * params.C1 + c];
  } else {
    result[gid.x] = b[n * params.C2 + (c - params.C1)];
  }
}
