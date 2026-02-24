// Layer normalization over the last axis
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

struct Params { N: u32, C: u32 }
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= params.N) { return; }
  let C = params.C;
  let base = n * C;

  var mean: f32 = 0.0;
  for (var c: u32 = 0u; c < C; c = c + 1u) {
    mean = mean + input[base + c];
  }
  mean = mean / f32(C);

  var variance: f32 = 0.0;
  for (var c: u32 = 0u; c < C; c = c + 1u) {
    let diff = input[base + c] - mean;
    variance = variance + diff * diff;
  }
  variance = variance / f32(C);

  let eps: f32 = 1e-5;
  let inv_std = 1.0 / sqrt(variance + eps);
  for (var c: u32 = 0u; c < C; c = c + 1u) {
    let normed = (input[base + c] - mean) * inv_std;
    output[base + c] = normed * scale[c] + bias[c];
  }
}
