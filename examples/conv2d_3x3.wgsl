// 2D convolution — 3×3 kernel, stride 1, no padding.
//
// The corpus had no convolution with a literal window and no padding, which is
// the most ordinary convolution there is: conv2d takes its window from the
// params struct, and conv2d_5x5 pads. The gap mattered. A driver's answer
// about the two we had could not separate "the window is symbolic" from any
// other difference, because both differences were present in every model.

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
  N: u32,
  IC: u32,
  IH: u32,
  IW: u32,
  OC: u32,
  KH: u32,
  KW: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let oc = gid.x;
  let oh = gid.y;
  if (oc >= params.OC) {
    return;
  }

  let ow_max = params.IW - params.KW + 1u;
  for (var ow: u32 = 0u; ow < ow_max; ow = ow + 1u) {
    var sum: f32 = 0.0;
    // Literal bounds, so the emitted weight says 3x3 rather than taking
    // whatever --symbolic-dim supplies.
    for (var kh: u32 = 0u; kh < 3u; kh = kh + 1u) {
      for (var kw: u32 = 0u; kw < 3u; kw = kw + 1u) {
        for (var ic: u32 = 0u; ic < params.IC; ic = ic + 1u) {
          let ih = oh + kh;
          let iw = ow + kw;
          let in_idx = ic * params.IH * params.IW + ih * params.IW + iw;
          // Flattened through the params rather than through literal factors:
          // the stride extractor reads multiplication literals, and a literal
          // that flattens an index is not a stride.
          let w_idx = oc * params.IC * params.KH * params.KW
                    + ic * params.KH * params.KW + kh * params.KW + kw;
          sum = sum + input[in_idx] * weight[w_idx];
        }
      }
    }
    let out_idx = oc * (params.IH - params.KH + 1u) * ow_max + oh * ow_max + ow;
    output[out_idx] = sum;
  }
}
