// Depthwise convolution 2D (groups == channels_in)
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params { N: u32, C: u32, H: u32, W: u32, KH: u32, KW: u32, groups: u32 }
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let OH = params.H - params.KH + 1u;
  let OW = params.W - params.KW + 1u;
  let total = params.N * params.C * OH * OW;
  if (idx >= total) { return; }

  let ow = idx % OW;
  let oh = (idx / OW) % OH;
  let c  = (idx / (OW * OH)) % params.C;
  let n  = idx / (OW * OH * params.C);

  var sum: f32 = 0.0;
  for (var kh: u32 = 0u; kh < params.KH; kh = kh + 1u) {
    for (var kw: u32 = 0u; kw < params.KW; kw = kw + 1u) {
      let ih = oh + kh;
      let iw = ow + kw;
      let in_idx = ((n * params.C + c) * params.H + ih) * params.W + iw;
      let w_idx  = (c * params.KH + kh) * params.KW + kw;
      sum = sum + input[in_idx] * weight[w_idx];
    }
  }
  output[idx] = sum;
}
