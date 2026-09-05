//! Kernels whose addresses come out of a buffer rather than out of the thread
//! id, and what the classifier is allowed to say about them.
//!
//! All four sources below are reduced from vendor/web-xpu-ops — enough of each
//! kernel to carry the evidence, so the tests do not need the submodule. Every
//! one of them used to be reported as an operator it is not: the row gather as
//! `Concat`, the scatter-add and the MoE dispatch as `MatMul`. Two inputs, a
//! loop and a params struct with three names were all it took, and none of
//! those is evidence of a matmul.
//!
//! The assertions are about the outcome, not the label: for the one kernel
//! that is recognised, which tensor is the data and which the indices; for the
//! three that are refused, that the reason names the shape that was found.

use nxpu_analysis::analyze::{self, KernelPattern};
use nxpu_backend_core::{Backend, BackendOptions};
use nxpu_backend_onnx::OnnxBackend;
use nxpu_backend_tflite::TfLiteBackend;
use nxpu_opt::{OptLevel, PassManager};

/// Classify the single entry point of `source` after optimizing at `level`.
fn classify_at(source: &str, level: OptLevel) -> KernelPattern {
    let mut module = nxpu_parser::parse(source).expect("WGSL parse failed");
    PassManager::for_level(level).run(&mut module);
    analyze::classify_entry_point(&module, 0).expect("classification failed")
}

/// Classify at `-O1`, and check the answer does not move with the level.
///
/// FMA fusion rewrites `idx * D + d` into `fma(idx, D, d)`, which is exactly
/// the expression these recognisers walk, so "same answer at every level" is
/// not a formality here.
fn classify(source: &str) -> KernelPattern {
    let o0 = format!("{:?}", classify_at(source, OptLevel::O0));
    let o1 = classify_at(source, OptLevel::O1);
    let o2 = format!("{:?}", classify_at(source, OptLevel::O2));
    assert_eq!(
        o0,
        format!("{o1:?}"),
        "classification moved between -O0/-O1"
    );
    assert_eq!(
        o2,
        format!("{o1:?}"),
        "classification moved between -O2/-O1"
    );
    o1
}

/// The `Unknown` reason, or a panic naming what was returned instead.
fn refusal(source: &str) -> String {
    match classify(source) {
        KernelPattern::Unknown { reason } => reason,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------

/// `output[n, :] = table[indices[n], :]` — web-xpu-ops `ops/gather`, an
/// embedding lookup. The index selects a row `D` wide, so it is multiplied by
/// `D` on its way to becoming an address.
const ROW_GATHER: &str = "
struct Params { N: u32, D: u32, rows: u32 }
@group(0) @binding(0) var<storage, read> table: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.N * params.D;
  if (gid.x >= total) { return; }
  let n = gid.x / params.D;
  let d = gid.x % params.D;
  let row = indices[n];
  if (u32(row) >= params.rows) { output[gid.x] = 0.0; return; }
  output[gid.x] = table[u32(row) * params.D + d];
}";

/// `output[i] = table[indices[i]]` — one element per index, over a flat table.
/// This is the shape `KernelPattern::Gather` describes, and the reason the row
/// gather above is told apart from it rather than refused with it.
const ELEMENT_GATHER: &str = "
struct Params { N: u32 }
@group(0) @binding(0) var<storage, read> table: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.N) { return; }
  output[i] = table[u32(indices[i])];
}";

/// `output[row][indices[row][slot]] += src[row][slot]` — web-xpu-ops
/// `ops/scatter`. Colliding indices accumulate, which is what the atomic
/// output buffer is for; the compare-exchange loop is an f32 atomic add.
const SCATTER_ADD: &str = "
struct Params { N: u32, S: u32, D: u32 }
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: Params;

fn add_f32(slot: u32, value: f32) {
  var old = atomicLoad(&output[slot]);
  loop {
    let attempt = atomicCompareExchangeWeak(&output[slot], old, bitcast<u32>(bitcast<f32>(old) + value));
    if (attempt.exchanged) { break; }
    old = attempt.old_value;
  }
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let row = wg_id.x;
  if (row >= params.N) { return; }
  for (var slot = local_id.x; slot < params.S; slot += 256u) {
    let column = indices[row * params.S + slot];
    if (column < 0 || column >= i32(params.D)) { continue; }
    add_f32(row * params.D + u32(column), src[row * params.S + slot]);
  }
}";

/// `buffer[e, pos, :] = x[t, :]` where `expert[t] == e` — web-xpu-ops
/// `ops/moe/dispatch`, reduced to one rank per token and a serial scan. Where
/// a row lands is decided by the routing data, and the positions themselves
/// are a second output.
const MOE_DISPATCH: &str = "
struct Params { T: u32, k: u32, E: u32, C: u32, D: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> expert: array<i32>;
@group(0) @binding(2) var<storage, read_write> buffer: array<f32>;
@group(0) @binding(3) var<storage, read_write> pos: array<i32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let token = gid.x;
  if (token >= params.T) { return; }
  let chosen = expert[token];
  if (chosen < 0 || chosen >= i32(params.E)) { pos[token] = -1; return; }
  var earlier = 0;
  for (var p = 0u; p < token; p += 1u) {
    if (expert[p] == chosen) { earlier += 1; }
  }
  pos[token] = earlier;
  let destination = (u32(chosen) * params.C + u32(earlier)) * params.D;
  for (var d = 0u; d < params.D; d += 1u) {
    buffer[destination + d] = x[token * params.D + d];
  }
}";

// ---------------------------------------------------------------------------
// The one that is recognised
// ---------------------------------------------------------------------------

/// A flat gather is a `Gather`, and which buffer is which matters: swapping
/// data and indices produces a graph that indexes the table with the table.
#[test]
fn element_gather_names_its_data_and_indices() {
    match classify(ELEMENT_GATHER) {
        KernelPattern::Gather {
            data,
            indices,
            output,
            axis,
        } => {
            assert_eq!(data.name, "table");
            assert_eq!(indices.name, "indices");
            assert_eq!(output.name, "output");
            assert_eq!(axis, 0);
        }
        other => panic!("expected Gather, got {other:?}"),
    }
}

/// The evidence is the indexing, not the binding order: a kernel that binds
/// the index buffer first is the same operator and must be read the same way.
#[test]
fn element_gather_indices_may_be_bound_first() {
    const SWAPPED: &str = "
struct Params { N: u32 }
@group(0) @binding(0) var<storage, read> indices: array<i32>;
@group(0) @binding(1) var<storage, read> table: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.N) { return; }
  output[i] = table[u32(indices[i])];
}";
    match classify(SWAPPED) {
        KernelPattern::Gather { data, indices, .. } => {
            assert_eq!(data.name, "table");
            assert_eq!(indices.name, "indices");
        }
        other => panic!("expected Gather, got {other:?}"),
    }
}

/// A recognised pattern has to reach a backend, or the refusal has only been
/// moved somewhere less informative.
#[test]
fn element_gather_lowers() {
    let mut module = nxpu_parser::parse(ELEMENT_GATHER).expect("WGSL parse failed");
    PassManager::for_level(OptLevel::O1).run(&mut module);
    let options = BackendOptions {
        opt_level: 1,
        symbolic_extent: Some(64),
        ..Default::default()
    };
    OnnxBackend
        .compile(&module, &options)
        .expect("gather does not lower to ONNX");
    TfLiteBackend
        .compile(&module, &options)
        .expect("gather does not lower to TFLite");
}

// ---------------------------------------------------------------------------
// The three that are refused
// ---------------------------------------------------------------------------

/// A row gather is not a `Gather`: the pattern carries no row width, and its
/// lowering indexes a flat buffer, so it would select `N` single elements out
/// of a table whose rows are `D` wide. Refusing says so; `Concat`, which is
/// what two inputs and an `if` used to produce, says nothing true at all.
#[test]
fn row_gather_is_refused_for_the_missing_row_width() {
    let reason = refusal(ROW_GATHER);
    assert!(
        reason.contains("row gather"),
        "the reason should name the shape found: {reason}"
    );
    assert!(
        reason.contains("row width"),
        "the reason should name what is missing: {reason}"
    );
}

/// An atomic output is the accumulation, and `Scatter` has no reduction mode.
/// Reported as a matmul, this computed a product of two tensors that are a
/// value buffer and an index buffer.
#[test]
fn scatter_add_is_refused_for_the_accumulation() {
    let reason = refusal(SCATTER_ADD);
    assert!(
        reason.contains("atomics"),
        "the reason should name the atomic output: {reason}"
    );
    assert!(
        reason.contains("accumulat"),
        "the reason should say the writes accumulate: {reason}"
    );
}

/// Where each row lands is read out of the routing buffer, and the positions
/// are a second output that no two-input pattern here can carry.
#[test]
fn moe_dispatch_is_refused_for_the_data_dependent_placement() {
    let reason = refusal(MOE_DISPATCH);
    assert!(
        reason.contains("loaded from an input buffer"),
        "the reason should say the write address was loaded: {reason}"
    );
    assert!(
        reason.contains("two output tensors"),
        "the reason should name the second output: {reason}"
    );
}

// ---------------------------------------------------------------------------
// What the new evidence must not swallow
// ---------------------------------------------------------------------------

/// The real matmul stages both operands through workgroup tiles, so its inner
/// loop reads neither global directly. It is here because a two-input
/// recogniser that rejects it has failed however good its reasoning looked:
/// this must stay a `MatMul`.
const TILED_MATMUL: &str = "
struct Params { M: u32, N: u32, K: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = gid.y;
  let col = gid.x;
  var acc = 0.0;
  for (var t = 0u; t < params.K; t += 16u) {
    tile_a[lid.y * 16u + lid.x] = a[row * params.K + t + lid.x];
    tile_b[lid.y * 16u + lid.x] = b[(t + lid.y) * params.N + col];
    workgroupBarrier();
    for (var k = 0u; k < 16u; k += 1u) {
      acc = acc + tile_a[lid.y * 16u + k] * tile_b[k * 16u + lid.x];
    }
    workgroupBarrier();
  }
  output[row * params.N + col] = acc;
}";

#[test]
fn tiled_matmul_still_classifies_as_matmul() {
    match classify(TILED_MATMUL) {
        KernelPattern::MatMul { shape, .. } => {
            assert_eq!(shape.m, "M");
            assert_eq!(shape.n, "N");
            assert_eq!(shape.k, "K");
        }
        other => panic!("expected MatMul, got {other:?}"),
    }
}

/// A two-input convolution's window comes from the params, and its addresses
/// are arithmetic on the thread id — no buffer feeds them. The addressing
/// checks run ahead of the conv arm, so this pins that they let it through.
const CONV_NO_BIAS: &str = "
struct Params { H: u32, W: u32, KH: u32, KW: u32, C: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x % params.W;
  let y = gid.x / params.W;
  var acc = 0.0;
  for (var kh = 0u; kh < params.KH; kh += 1u) {
    for (var kw = 0u; kw < params.KW; kw += 1u) {
      acc = acc + input[(y + kh) * params.W + x + kw] * weight[kh * params.KW + kw];
    }
  }
  output[y * params.W + x] = acc;
}";

#[test]
fn two_input_conv_still_classifies_as_conv() {
    match classify(CONV_NO_BIAS) {
        KernelPattern::Conv2D { bias, .. } => assert!(bias.is_none()),
        other => panic!("expected Conv2D, got {other:?}"),
    }
}

/// An element-wise add reads both inputs at the thread id. Nothing about it is
/// index-dependent, and the checks above must not read `a[idx]` as an address
/// that came out of a buffer.
#[test]
fn elementwise_add_is_untouched() {
    const ADD: &str = "
struct Params { N: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.N) { return; }
  output[idx] = a[idx] + b[idx];
}";
    match classify(ADD) {
        KernelPattern::ElementWise { .. } => {}
        other => panic!("expected ElementWise, got {other:?}"),
    }
}
