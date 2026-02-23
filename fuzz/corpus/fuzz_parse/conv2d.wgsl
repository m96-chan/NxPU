// 2D convolution — target for NPU transpilation

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
    for (var kh: u32 = 0u; kh < params.KH; kh = kh + 1u) {
      for (var kw: u32 = 0u; kw < params.KW; kw = kw + 1u) {
        for (var ic: u32 = 0u; ic < params.IC; ic = ic + 1u) {
          let ih = oh + kh;
          let iw = ow + kw;
          let in_idx = ic * params.IH * params.IW + ih * params.IW + iw;
          let w_idx = oc * params.IC * params.KH * params.KW + ic * params.KH * params.KW + kh * params.KW + kw;
          sum = sum + input[in_idx] * weight[w_idx];
        }
      }
    }
    let out_idx = oc * (params.IH - params.KH + 1u) * ow_max + oh * ow_max + ow;
    output[out_idx] = sum;
  }
}
