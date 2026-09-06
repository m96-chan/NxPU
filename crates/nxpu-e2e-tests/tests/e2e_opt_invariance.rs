//! Classification must not depend on the optimization level.
//!
//! `nxpu-analysis` classifies an entry point *after* `nxpu-opt` has rewritten
//! it, so every rewrite the optimizer makes is a chance for the classifier to
//! give a different answer about the same source. FMA fusion (`nxpu-opt`'s
//! `FmaFusion`, on from `--opt-level 1`) is the case that actually bit: it
//! turns `y + a * x` into `fma(a, x, y)`, and the element-wise matcher only
//! looked at `Expression::Binary`.
//!
//! The invariant asserted here is deliberately about *agreement*, not about
//! any particular answer: the table below pins no expected pattern, so it goes
//! on holding as classification improves. What it forbids is a kernel that
//! compiles at `-O0` and stops compiling at `-O2`, or classifies as one op at
//! one level and another op at another.

use nxpu_analysis::classify_entry_point;
use nxpu_opt::{OptLevel, PassManager};

/// Classify every entry point of `source` after optimizing at `level`.
///
/// Errors are folded into the returned strings rather than unwrapped: a source
/// that fails to classify must fail the same way at every level, and that is
/// as much a part of the invariant as a source that succeeds.
fn classify_at(source: &str, level: OptLevel) -> Vec<String> {
    let mut module = nxpu_parser::parse(source).expect("WGSL parse failed");
    PassManager::for_level(level).run(&mut module);
    (0..module.entry_points.len())
        .map(|i| match classify_entry_point(&module, i) {
            Ok(pattern) => format!("{pattern:?}"),
            Err(e) => format!("Err({e})"),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Kernels. The multiply-add ones are reduced from vendor/web-xpu-ops so the
// test does not need the submodule checked out.
// ---------------------------------------------------------------------------

/// `out[i] = y[i] + a * x[i]` — vendor/web-xpu-ops `ops/axpy`. The multiply-add
/// that FMA fusion rewrites, and the reason this file exists.
const AXPY: &str = r#"
struct Params { N: u32, a: f32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.N) { return; }
  output[idx] = y[idx] + params.a * x[idx];
}
"#;

/// `y[i] += a * x[i]` — vendor/web-xpu-ops `ops/axpy/inplace.wgsl`. The same
/// arithmetic as `AXPY` with `y` bound `read_write`, so it is both read and
/// written and the classifier sees one input rather than two. FMA fusion
/// rewrites it identically, and it has to classify identically too.
const AXPY_INPLACE: &str = r#"
struct Params { N: u32, a: f32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.N) { return; }
  y[idx] = y[idx] + params.a * x[idx];
}
"#;

/// `out[i] = f32(input[i]) * s1 * s2` — vendor/web-xpu-ops `ops/dequantize`.
/// Two multiplies over dispatch-time scalars with a conversion under them; no
/// multiply-add, but the same chain matcher reads it, so it belongs in the
/// same table.
const DEQUANTIZE: &str = r#"
struct Params { N: u32 }
@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> weight_scale: f32;
@group(0) @binding(3) var<uniform> input_scale: f32;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.N) { return; }
  output[idx] = f32(input[idx]) * weight_scale * input_scale;
}
"#;

/// `out = scores + slope * distance` — vendor/web-xpu-ops `ops/alibi`. Same
/// shape as axpy but with the coefficient read from a second storage buffer.
const ALIBI: &str = r#"
struct Params { num_heads: u32, M: u32, N: u32, pos_offset: u32 }
@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read> slopes: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.num_heads * params.M * params.N;
  let index = gid.x;
  if (index >= total) { return; }
  let key = index % params.N;
  let rest = index / params.N;
  let query = rest % params.M;
  let head = rest / params.M;
  let distance = f32(key) - (f32(query) + f32(params.pos_offset));
  output[index] = scores[index] + slopes[head] * distance;
}
"#;

/// `out = x + (1/(a + eps)) * sin(a*x)^2` — vendor/web-xpu-ops `ops/snake`. A
/// multiply-add whose multiply is itself a non-trivial subtree.
const SNAKE: &str = r#"
struct Params { N: u32, C: u32, L: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> alpha: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.N * params.C * params.L) { return; }
  let c = (idx / params.L) % params.C;
  let a = alpha[c];
  let x = input[idx];
  let s = sin(a * x);
  output[idx] = x + (1.0 / (a + 1e-9)) * (s * s);
}
"#;

/// A multiply-add whose multiply has two users, so `FmaFusion` declines to
/// fuse it. The classifier must still call it the same thing as the fusable
/// spelling — invariance across levels is not the same claim as "the optimizer
/// always fires".
const SHARED_MULTIPLY: &str = r#"
struct Params { N: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.N) { return; }
  let m = a[idx] * b[idx];
  output[idx] = m + m;
}
"#;

/// A genuine two-operand add. Must keep classifying as one.
const ELEMENTWISE_ADD: &str = r#"
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
}
"#;

/// `x0*c - x1*s` — a subtract over two multiplies. Not a multiply-add, and
/// nothing fuses it; here to pin that the multiply-add check did not widen
/// into every arithmetic expression that contains a `*`.
const SUB_OF_MULTIPLIES: &str = r#"
struct Params { N: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.N) { return; }
  output[idx] = a[idx] * 0.5 - b[idx] * 0.25;
}
"#;

/// A loop with an `acc = acc + a*b` accumulation — the multiply-add FMA fusion
/// finds inside every matmul.
const MATMUL: &str = r#"
struct Params { M: u32, N: u32, K: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let col = gid.y;
  if (row >= params.M) { return; }
  var acc = 0.0;
  for (var k = 0u; k < params.K; k = k + 1u) {
    acc = acc + a[row * params.K + k] * b[k * params.N + col];
  }
  output[row * params.N + col] = acc;
}
"#;

/// `max(x, 0)` — an activation, reached before the element-wise matcher runs.
const RELU: &str = r#"
struct Params { N: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.N) { return; }
  output[idx] = max(input[idx], 0.0);
}
"#;

/// GELU's tanh approximation, with the outer half-and-half factored as
/// `0.5 + 0.5 * tanh(...)` rather than `0.5 * (1.0 + tanh(...))`. The two are
/// the same arithmetic, but the first is a multiply-add: fusion rewrites it to
/// `fma(0.5, tanh(...), 0.5)` and buries the `tanh` in the *second* operand of
/// a `Math` expression — a place the activation walk did not look.
const GELU: &str = r#"
struct Params { N: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.N) { return; }
  let x = input[idx];
  let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
  output[idx] = x * (0.5 + 0.5 * tanh(inner));
}
"#;

/// A row softmax, written as two reduction loops and a normalising pass rather
/// than the workgroup tree `vendor/web-xpu-ops` uses. Recognition looks for an
/// `exp` accumulated in a loop and an `exp` stored over a divide, and both
/// halves are things the optimizer rewrites around: the accumulation is an add
/// in a loop, and `exp(..) / total` here is the reciprocal-multiply spelling's
/// twin.
const SOFTMAX: &str = r#"
struct Params { N: u32, D: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.N) { return; }
  let base = row * params.D;
  var row_max = input[base];
  for (var c = 1u; c < params.D; c += 1u) { row_max = max(row_max, input[base + c]); }
  var total = 0.0;
  for (var c = 0u; c < params.D; c += 1u) { total += exp(input[base + c] - row_max); }
  for (var c = 0u; c < params.D; c += 1u) {
    output[base + c] = exp(input[base + c] - row_max) / total;
  }
}
"#;

/// One input, two outputs, and neither of them a slice of it — an activation
/// quantizer's codes and per-row scales, cut down. The multi-output arm has to
/// refuse it identically at every level, reason included.
const QUANTIZE: &str = r#"
struct Params { N: u32, D: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;
@group(0) @binding(2) var<storage, read_write> scales: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.N) { return; }
  let base = row * params.D;
  var absmax = 0.0;
  for (var c = 0u; c < params.D; c += 1u) { absmax = max(absmax, abs(input[base + c])); }
  scales[row] = absmax / 127.0;
  let inv_scale = 127.0 / absmax;
  for (var c = 0u; c < params.D; c += 1u) {
    output[base + c] = clamp(i32(round(input[base + c] * inv_scale)), -127, 127);
  }
}
"#;

/// The split `examples/split.wgsl` compiles, kept here so that tightening the
/// multi-output arm around the quantizer above is checked against the op it
/// still has to accept.
const SPLIT: &str = r#"
struct Params { N: u32, split_at: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_a: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_b: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.N) { return; }
  if (idx < params.split_at) {
    out_a[idx] = input[idx];
  } else {
    out_b[idx - params.split_at] = input[idx];
  }
}
"#;

const KERNELS: &[(&str, &str)] = &[
    ("axpy", AXPY),
    ("axpy_inplace", AXPY_INPLACE),
    ("dequantize", DEQUANTIZE),
    ("alibi", ALIBI),
    ("snake", SNAKE),
    ("shared_multiply", SHARED_MULTIPLY),
    ("elementwise_add", ELEMENTWISE_ADD),
    ("sub_of_multiplies", SUB_OF_MULTIPLIES),
    ("matmul", MATMUL),
    ("relu", RELU),
    ("gelu", GELU),
    ("softmax", SOFTMAX),
    ("quantize", QUANTIZE),
    ("split", SPLIT),
];

#[test]
fn classification_is_invariant_across_opt_levels() {
    for (name, source) in KERNELS {
        let o0 = classify_at(source, OptLevel::O0);
        let o1 = classify_at(source, OptLevel::O1);
        let o2 = classify_at(source, OptLevel::O2);

        assert_eq!(
            o0, o1,
            "{name}: classification changed between -O0 and -O1\n  -O0: {o0:#?}\n  -O1: {o1:#?}"
        );
        assert_eq!(
            o0, o2,
            "{name}: classification changed between -O0 and -O2\n  -O0: {o0:#?}\n  -O2: {o2:#?}"
        );
    }
}

/// The table above says only that the answers agree. This says what the answer
/// for a multiply-add now *is*, because the choice was deliberate and a silent
/// change back would be a regression.
///
/// `y + a * x` is not an add. Classifying it as `ElementWise(Add)` — which is
/// what `-O0` did before this — drops the multiply on the floor and hands the
/// backend a kernel that computes something else. `axpy` is now read as an
/// element-wise chain instead; the two below are multiply-adds that are *not*
/// chains, and the honest answer for them is still `Unknown` with a reason
/// that names what was found.
///
/// `alibi` multiplies a slope read at `slopes[head]` — a subscript derived
/// from the thread id, not the id — by a distance computed from the id and
/// read out of no buffer at all. `snake` multiplies a reciprocal by a squared
/// sine, two subtrees neither of which is a leaf. Both would need operands a
/// chain cannot carry.
#[test]
fn multiply_add_is_unknown_and_says_why() {
    for (name, source) in [("alibi", ALIBI), ("snake", SNAKE)] {
        for level in [OptLevel::O0, OptLevel::O2] {
            let patterns = classify_at(source, level);
            assert_eq!(patterns.len(), 1, "{name}: expected one entry point");
            let p = &patterns[0];
            assert!(
                p.contains("Unknown"),
                "{name} at {level:?}: expected Unknown, got {p}"
            );
            assert!(
                p.contains("multiply-add"),
                "{name} at {level:?}: reason should name the multiply-add, got {p}"
            );
        }
    }
}

/// And this says what the answer for the chains now is, at every level.
///
/// `-O1` rewrites `y + a * x` into `fma(a, x, y)`, which is why the chain
/// matcher reads `Math { fun: Fma }` as well as `Binary { op: Add }`. Pinning
/// the variant here rather than only its agreement with itself is what stops a
/// future rewrite from making both levels agree on `Unknown` again.
#[test]
fn a_chain_is_recognized_at_every_level() {
    for (name, source) in [
        ("axpy", AXPY),
        ("axpy_inplace", AXPY_INPLACE),
        ("dequantize", DEQUANTIZE),
    ] {
        for level in [OptLevel::O0, OptLevel::O1, OptLevel::O2] {
            let patterns = classify_at(source, level);
            assert_eq!(patterns.len(), 1, "{name}: expected one entry point");
            let p = &patterns[0];
            assert!(
                p.contains("ElementWiseChain"),
                "{name} at {level:?}: expected a chain, got {p}"
            );
        }
    }
}
