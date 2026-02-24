// Multi-head scaled dot-product attention.
//
// Splits Q/K/V across num_heads attention heads,
// computes scaled attention for each head, then recombines.

@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

struct Params { seq_len: u32, d_model: u32, num_heads: u32 }
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let j = gid.y;
  if i >= params.seq_len || j >= params.seq_len { return; }

  let head_dim = params.d_model / params.num_heads;
  let scale = 1.0 / sqrt(f32(head_dim));

  for (var h: u32 = 0u; h < params.num_heads; h = h + 1u) {
    var score: f32 = 0.0;
    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
      let q_idx = i * params.d_model + h * head_dim + d;
      let k_idx = j * params.d_model + h * head_dim + d;
      score = score + query[q_idx] * key[k_idx];
    }
    score = score * scale;

    let attn = exp(score);

    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
      let v_idx = j * params.d_model + h * head_dim + d;
      let o_idx = i * params.d_model + h * head_dim + d;
      output[o_idx] = output[o_idx] + attn * value[v_idx];
    }
  }
}
