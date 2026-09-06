// A matrix-vector product against int8 weights with one scale per row.
//
//   out[i] = (sum_k unpack(weight[i, k]) * vector[k]) * scale[i]
//
// `weight` is bound as `array<u32>` and holds four two's-complement codes per
// word, least-significant byte first — which is the byte layout of a
// contiguous `i8` row, so the emitted graph declares it `int8 [N, K]` and the
// packing stays where it belongs, in the kernel. `extractBits` on a *signed*
// base sign-extends the field, which is what makes `0xff` read as -1.
//
// The row scale multiplies the finished sum rather than each of its terms:
// it does not depend on the column, so it comes out of the reduction. Both
// emitted graphs keep that hoist — the scale is applied after the contraction,
// where it broadcasts over the result's last axis.
//
// Reduced from vendor/web-xpu-ops `ops/matvec/wgsl/q8.wgsl`, which streams the
// row across 256 lanes and reduces through workgroup memory. None of that
// changes what the kernel computes, and this form is what an example is for.

struct Params {
  N: u32,
  K: u32,
}

@group(0) @binding(0) var<storage, read> weight: array<u32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> vector: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn unpack_i8(word: u32, lane: u32) -> f32 {
  return f32(extractBits(bitcast<i32>(word), lane * 8u, 8u));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.N) {
    return;
  }
  let words_per_row = (params.K + 3u) / 4u;
  let row_word_offset = row * words_per_row;

  var partial: f32 = 0.0;
  for (var word_index = 0u; word_index < words_per_row; word_index += 1u) {
    let word = weight[row_word_offset + word_index];
    let base_col = word_index * 4u;
    for (var lane = 0u; lane < 4u; lane += 1u) {
      let col = base_col + lane;
      if (col >= params.K) {
        break;
      }
      partial += unpack_i8(word, lane) * vector[col];
    }
  }

  output[row] = partial * scale[row];
}
