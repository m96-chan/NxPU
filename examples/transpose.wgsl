// Matrix transpose — target for NPU transpilation

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> c: array<f32>;

struct Params {
  rows: u32,
  cols: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let col = gid.y;
  if (row >= params.rows || col >= params.cols) {
    return;
  }
  c[col * params.rows + row] = a[row * params.cols + col];
}
