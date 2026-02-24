// Causal (masked) scaled dot-product attention.
//
// Applies a triangular causal mask: positions j > i are masked out
// with a large negative value before softmax, preventing attention
// to future tokens.

@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

struct Params { seq_len: u32, d_k: u32 }
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let j = gid.y;
  if i >= params.seq_len || j >= params.seq_len { return; }

  let scale = 1.0 / sqrt(f32(params.d_k));

  var score: f32 = 0.0;
  for (var d: u32 = 0u; d < params.d_k; d = d + 1u) {
    score = score + query[i * params.d_k + d] * key[j * params.d_k + d];
  }
  score = score * scale;

  if j > i {
    score = -1e30;
  }

  let attn = exp(score);

  for (var d: u32 = 0u; d < params.d_k; d = d + 1u) {
    output[i * params.d_k + d] = output[i * params.d_k + d] + attn * value[j * params.d_k + d];
  }
}
