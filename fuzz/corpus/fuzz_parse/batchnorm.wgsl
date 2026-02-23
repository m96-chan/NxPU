// Batch normalization — target for NPU transpilation
// y = gamma * (x - mean) / sqrt(variance + epsilon) + beta

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

struct Params {
  N: u32,
  C: u32,
  HW: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.N * params.C * params.HW;
  if (idx >= total) {
    return;
  }
  let c = (idx / params.HW) % params.C;

  // Compute channel mean
  var mean: f32 = 0.0;
  let ch_start = (idx / (params.C * params.HW)) * params.C * params.HW + c * params.HW;
  for (var i: u32 = 0u; i < params.HW; i = i + 1u) {
    mean = mean + x[ch_start + i];
  }
  mean = mean / f32(params.HW);

  // Compute channel variance
  var variance: f32 = 0.0;
  for (var i: u32 = 0u; i < params.HW; i = i + 1u) {
    let diff = x[ch_start + i] - mean;
    variance = variance + diff * diff;
  }
  variance = variance / f32(params.HW);

  // Normalize
  let epsilon: f32 = 1e-5;
  let normalized = (x[idx] - mean) / sqrt(variance + epsilon);
  y[idx] = gamma[c] * normalized + beta[c];
}
