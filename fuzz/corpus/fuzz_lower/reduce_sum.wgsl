// Reduction sum over rows — target for NPU transpilation

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> c: array<f32>;

struct Params {
  N: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.N) {
    return;
  }
  var sum: f32 = 0.0;
  for (var i: u32 = 0u; i < params.N; i = i + 1u) {
    sum = sum + a[row * params.N + i];
  }
  c[row] = sum;
}
