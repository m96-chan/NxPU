// 2D convolution — 5×5 kernel, stride 2, padding 1

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
  if (oc >= params.OC) { return; }

  for (var ow: u32 = 0u; ow < params.IW; ow = ow + 1u) {
    var sum: f32 = 0.0;
    for (var kh: u32 = 0u; kh < 5u; kh = kh + 1u) {
      for (var kw: u32 = 0u; kw < 5u; kw = kw + 1u) {
        let ih = oh * 2u + kh - 1u;
        let iw = ow * 2u + kw - 1u;
        if (ih < params.IH && iw < params.IW) {
          for (var ic: u32 = 0u; ic < params.IC; ic = ic + 1u) {
            let in_idx = ic * params.IH * params.IW + ih * params.IW + iw;
            let w_idx = oc * params.IC * params.KH * params.KW
                      + ic * params.KH * params.KW + kh * params.KW + kw;
            sum = sum + input[in_idx] * weight[w_idx];
          }
        }
      }
    }
    let out_idx = oc * params.IH * params.IW + oh * params.IW + ow;
    output[out_idx] = sum;
  }
}
