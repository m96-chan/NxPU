// Scaled dot-product attention.
//
// attn = softmax(Q * K^T / sqrt(d_k)) * V
//
// For each query position, computes attention scores against all key positions,
// applies numerically stable softmax, then weights values.

@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

struct Params { seq_len: u32, d_k: u32 }
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let col = gid.y;
  if (row >= params.seq_len || col >= params.d_k) { return; }

  let scale = 1.0 / sqrt(f32(params.d_k));

  // Compute attention scores: Q[row,:] dot K[j,:] for all j
  // Then softmax over j dimension
  var max_score: f32 = -1e30;
  for (var j: u32 = 0u; j < params.seq_len; j = j + 1u) {
    var score: f32 = 0.0;
    for (var k: u32 = 0u; k < params.d_k; k = k + 1u) {
      score = score + query[row * params.d_k + k] * key[j * params.d_k + k];
    }
    score = score * scale;
    max_score = max(max_score, score);
  }

  // Compute softmax denominator and weighted sum
  var sum_exp: f32 = 0.0;
  var result: f32 = 0.0;
  for (var j: u32 = 0u; j < params.seq_len; j = j + 1u) {
    var score: f32 = 0.0;
    for (var k: u32 = 0u; k < params.d_k; k = k + 1u) {
      score = score + query[row * params.d_k + k] * key[j * params.d_k + k];
    }
    score = score * scale;
    let w = exp(score - max_score);
    sum_exp = sum_exp + w;
    result = result + w * value[j * params.d_k + col];
  }

  output[row * params.d_k + col] = result / sum_exp;
}
