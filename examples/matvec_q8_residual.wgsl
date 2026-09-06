// `matvec_q8.wgsl` with a residual added once per row:
//
//   out[i] = (sum_k unpack(weight[i, k]) * vector[k]) * scale[i] + residual[i]
//
// Two buffers are read at `row` here — the scale and the residual — and what
// tells them apart is what the kernel does with them: one multiplies the
// finished contraction and the other is added to it. Reading them the other
// way round would scale by the residual and add the scale, which is a graph
// that runs and computes something else, so the recogniser keys on the
// operator and the emitted graph carries the residual as the matmul's bias.
//
// At `-O1` the store is `fma(partial, scale[row], residual[row])` rather than
// `partial * scale[row] + residual[row]`; both spellings have to be read the
// same way, which `e2e_opt_invariance.rs` holds to.
//
// Reduced from vendor/web-xpu-ops `ops/matvec/wgsl/q8_residual.wgsl`, which
// fuses the add onto the projection to save a dispatch.

struct Params {
  N: u32,
  K: u32,
}

@group(0) @binding(0) var<storage, read> weight: array<u32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> vector: array<f32>;
@group(0) @binding(3) var<storage, read> residual: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

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

  output[row] = partial * scale[row] + residual[row];
}
