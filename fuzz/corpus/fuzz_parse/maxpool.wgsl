// Max pooling 2D (2x2 kernel, stride 2) — target for NPU transpilation

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
  C: u32,
  IH: u32,
  IW: u32,
  OH: u32,
  OW: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  let oh = gid.y;
  if (c >= params.C || oh >= params.OH) {
    return;
  }

  for (var ow: u32 = 0u; ow < params.OW; ow = ow + 1u) {
    var max_val: f32 = -3.402823e+38;
    for (var kh: u32 = 0u; kh < 2u; kh = kh + 1u) {
      for (var kw: u32 = 0u; kw < 2u; kw = kw + 1u) {
        let ih = oh * 2u + kh;
        let iw = ow * 2u + kw;
        let val = input[c * params.IH * params.IW + ih * params.IW + iw];
        max_val = max(max_val, val);
      }
    }
    output[c * params.OH * params.OW + oh * params.OW + ow] = max_val;
  }
}
