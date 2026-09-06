//! Kernels whose stored value is a small expression tree.
//!
//! `ElementWise` holds two operands and one operation, so a kernel that scales
//! one tensor by a dispatch-time scalar before adding another had nowhere to
//! go: `axpy` was refused as a "fused multiply-add", `dequantize` as "single
//! input, no recognized activation". Both are now read as
//! `KernelPattern::ElementWiseChain`, which carries one tensor, an optional
//! conversion, and a sequence of steps.
//!
//! What is asserted here is the outcome — which tensor the chain starts from,
//! which operation each step applies, which operand it applies it to, and what
//! the ONNX and TFLite backends then compute — rather than that a particular
//! branch was taken. The rejections matter as much: four predicates were tried
//! on this shape and each was wrong in one direction, so every kernel that
//! must *not* become a chain is listed below with the reason it does not.
//!
//! The sources are reduced from `vendor/web-xpu-ops` so the test does not need
//! the submodule checked out.

mod common;

use nxpu_analysis::analyze::{
    self, ChainOperand, ElementWiseOp, KernelPattern, TensorRole, data_type,
};
use nxpu_backend_core::{Backend, BackendOptions, OutputContent};
use nxpu_backend_onnx::OnnxBackend;
use nxpu_backend_tflite::TfLiteBackend;
use nxpu_opt::{OptLevel, PassManager};
use tract_onnx::prelude::*;

// ---------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------

/// `ops/axpy/wgsl/kernel.wgsl`: `out[i] = y[i] + a * x[i]`, `a` a uniform
/// scalar rewritten every diffusion step.
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

/// `ops/axpy/wgsl/inplace.wgsl`: the same arithmetic with `y` bound
/// `read_write`, so the buffer added to is the buffer written.
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

/// `ops/dequantize/wgsl/kernel.wgsl`: an i32 accumulator converted and scaled
/// twice, both scales dispatch-time uniforms of their own.
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn classify_at(source: &str, level: OptLevel) -> KernelPattern {
    let mut module = nxpu_parser::parse(source).expect("WGSL parse failed");
    PassManager::for_level(level).run(&mut module);
    analyze::classify_entry_point(&module, 0).expect("classification failed")
}

/// Classify at `-O0` and `-O2` and require the same answer, returning it.
///
/// Every assertion below goes through here rather than through one level:
/// `y + a * x` is `fma(a, x, y)` from `-O1`, and a matcher that reads only one
/// spelling gives different answers about the same source.
fn classify(source: &str) -> KernelPattern {
    let o0 = format!("{:?}", classify_at(source, OptLevel::O0));
    let o2 = classify_at(source, OptLevel::O2);
    assert_eq!(
        o0,
        format!("{o2:?}"),
        "classification changed between -O0 and -O2"
    );
    o2
}

/// The chain's fields, flattened for assertion: the base tensor's name, the
/// cast, and one `(op, operand name, is-scalar)` per step.
fn chain_of(pattern: &KernelPattern) -> (String, Option<i32>, Vec<(ElementWiseOp, String, bool)>) {
    match pattern {
        KernelPattern::ElementWiseChain {
            base,
            cast,
            steps,
            output,
            ..
        } => {
            assert_eq!(base.role, TensorRole::Input, "the base is an input");
            assert_eq!(output.role, TensorRole::Output, "the result is an output");
            let steps = steps
                .iter()
                .map(|s| match &s.operand {
                    ChainOperand::Tensor(t) => (s.op, t.name.clone(), false),
                    ChainOperand::Scalar(sc) => (s.op, sc.name.clone(), true),
                })
                .collect();
            (base.name.clone(), *cast, steps)
        }
        other => panic!("expected an element-wise chain, got {other:?}"),
    }
}

fn output_name(pattern: &KernelPattern) -> String {
    match pattern {
        KernelPattern::ElementWiseChain { output, .. } => output.name.clone(),
        other => panic!("expected an element-wise chain, got {other:?}"),
    }
}

/// The reason a source is refused, having checked it is refused identically at
/// every optimization level.
fn refusal(source: &str) -> String {
    let mut reasons = Vec::new();
    for level in [OptLevel::O0, OptLevel::O1, OptLevel::O2] {
        match classify_at(source, level) {
            KernelPattern::Unknown { reason } => reasons.push(reason),
            other => panic!("expected Unknown at {level:?}, got {other:?}"),
        }
    }
    assert_eq!(reasons[0], reasons[1], "reason changed between -O0 and -O1");
    assert_eq!(reasons[0], reasons[2], "reason changed between -O0 and -O2");
    reasons.remove(0)
}

// ---------------------------------------------------------------------------
// What the chains say
// ---------------------------------------------------------------------------

/// `x` is scaled by `a` and the result added to `y` — in that order, because
/// the order is the graph. Reporting `Add` here, which `-O0` used to, keeps
/// the addition and silently drops the scale.
#[test]
fn axpy_scales_x_by_the_uniform_then_adds_y() {
    let pattern = classify(AXPY);
    let (base, cast, steps) = chain_of(&pattern);

    assert_eq!(base, "x");
    assert_eq!(cast, None, "both buffers are f32; nothing is converted");
    assert_eq!(
        steps,
        vec![
            (ElementWiseOp::Mul, "a".to_string(), true),
            (ElementWiseOp::Add, "y".to_string(), false),
        ]
    );
    assert_eq!(output_name(&pattern), "output");
}

/// The in-place entry point computes the same function. Its `y` is bound
/// `read_write`, so the classifier counts it as an output and sees one input;
/// the chain still reads it as the tensor added to, and the result takes a
/// different name because a graph needs two names for a value before and after
/// a write.
#[test]
fn axpy_in_place_is_the_same_arithmetic_with_a_renamed_result() {
    let pattern = classify(AXPY_INPLACE);
    let (base, cast, steps) = chain_of(&pattern);

    assert_eq!(base, "x");
    assert_eq!(cast, None);
    assert_eq!(
        steps,
        vec![
            (ElementWiseOp::Mul, "a".to_string(), true),
            (ElementWiseOp::Add, "y".to_string(), false),
        ]
    );
    assert_eq!(
        output_name(&pattern),
        "y_out",
        "the buffer is read and written, so the two values need two names"
    );
}

/// The conversion is part of the operator, not a detail under it: `input` is
/// `array<i32>` and `output` is `array<f32>`, and a chain that started from
/// the i32 buffer without a `Cast` would reinterpret the bits.
#[test]
fn dequantize_converts_then_applies_both_scales() {
    let pattern = classify(DEQUANTIZE);
    let (base, cast, steps) = chain_of(&pattern);

    assert_eq!(base, "input");
    assert_eq!(cast, Some(data_type::FLOAT));
    assert_eq!(
        steps,
        vec![
            (ElementWiseOp::Mul, "weight_scale".to_string(), true),
            (ElementWiseOp::Mul, "input_scale".to_string(), true),
        ]
    );
    assert_eq!(output_name(&pattern), "output");
}

/// The scales are graph inputs, not constants folded into the graph. Their
/// values are written per dispatch — `axpy`'s `a` changes every step — so a
/// constant would bake in whatever they were at compile time.
#[test]
fn a_dispatch_time_scalar_is_an_input_not_a_constant() {
    let output = common::compile_wgsl(AXPY, &OnnxBackend, 1);
    let bytes = onnx_bytes(&output);
    let model = load_proto(&bytes);
    let graph = model.graph.expect("graph");

    assert!(
        graph.initializer.is_empty(),
        "nothing about a per-dispatch scalar is known at compile time"
    );
    let inputs: Vec<&str> = graph.input.iter().map(|i| i.name.as_str()).collect();
    assert!(
        inputs.contains(&"a"),
        "the scalar is a graph input: {inputs:?}"
    );
}

// ---------------------------------------------------------------------------
// What must not become a chain
// ---------------------------------------------------------------------------

/// A broadcast: `b` is read at the column, not at the index written. It is an
/// `ElementWise` and has to stay one — a chain step over `b` would claim an
/// operand of the output's shape, which `b` is not.
#[test]
fn a_row_broadcast_stays_element_wise() {
    const ROWS: &str = r#"
struct Params { S: u32, D: u32, op: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.S * params.D) { return; }
  let col = idx % params.D;
  let scalar = b[col];
  if (params.op == 0u) { output[idx] = a[idx] + scalar; }
  else { output[idx] = a[idx] * scalar; }
}
"#;
    assert!(
        matches!(
            classify(ROWS),
            KernelPattern::ElementWise {
                op: ElementWiseOp::Add,
                ..
            }
        ),
        "a broadcast add is an Add"
    );
}

/// A plain two-tensor add is a plain two-tensor add. A chain could describe it
/// — one step, one tensor operand — and must not, because two names for one
/// operator is how a pattern starts being emitted where it is not meant.
#[test]
fn a_two_tensor_add_stays_element_wise() {
    const ADD: &str = r#"
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
    assert!(
        matches!(
            classify(ADD),
            KernelPattern::ElementWise {
                op: ElementWiseOp::Add,
                ..
            }
        ),
        "a two-tensor add is an Add, not a one-step chain"
    );
}

/// `alibi`: `scores + slopes[head] * distance`. The slope is read at a
/// subscript derived from the index rather than at the index, and the distance
/// is computed from the index and read out of no buffer at all. Neither is an
/// operand a chain can carry.
#[test]
fn alibi_is_refused_because_its_operands_are_not_tensors_at_the_index() {
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
    assert!(refusal(ALIBI).contains("not a chain"));
}

/// `snake`: `x + (1/(a + eps)) * sin(a*x)^2`. The multiply has a reciprocal on
/// one side and a squared sine on the other — two subtrees, neither a leaf —
/// so the expression is a tree and not a chain.
#[test]
fn snake_is_refused_because_its_multiply_has_two_subtrees() {
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
    assert!(refusal(SNAKE).contains("not a chain"));
}

/// `snake`'s SnakeBeta, with α and β as separate per-channel buffers. Three
/// inputs, so it reaches a different arm — and the chain matcher runs there
/// too, because the rule is about the shape of the expression and not about
/// how many buffers are bound. It fails for the same two reasons: the multiply
/// is a tree, and the parameters are read at `(i / L) % C`.
#[test]
fn snake_beta_is_refused_for_the_same_reasons_with_three_inputs() {
    const SNAKE_BETA: &str = r#"
struct Params { N: u32, C: u32, L: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> alpha: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.N * params.C * params.L) { return; }
  let c = (idx / params.L) % params.C;
  let a = alpha[c];
  let b = beta[c];
  let x = input[idx];
  let s = sin(a * x);
  output[idx] = x + (1.0 / (b + 1e-9)) * (s * s);
}
"#;
    assert!(refusal(SNAKE_BETA).contains("not as a chain"));
}

/// A multiply with two users, which FMA fusion declines to fuse: `m + m` where
/// `m = a[i] * b[i]`. Both operands of the add are the same non-leaf, so there
/// is no side that carries the chain and no side that is an operand.
#[test]
fn a_shared_multiply_is_refused() {
    const SHARED: &str = r#"
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
    assert!(refusal(SHARED).contains("not a chain"));
}

/// `s - x[i]`: the accumulator would land on the right of a `Sub`, and every
/// chain step is applied as `acc = acc <op> operand`. Reordering it into
/// `x - s` computes the negation of the answer, so it is refused instead.
#[test]
fn a_scalar_minus_a_tensor_is_refused() {
    const REVERSED: &str = r#"
struct Params { N: u32, s: f32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.N) { return; }
  output[idx] = params.s - x[idx];
}
"#;
    assert!(
        refusal(REVERSED).contains("no recognized activation"),
        "a reversed subtract is not a chain"
    );
}

/// `x[i] - s`, the same kernel the other way round, *is* a chain. Here to pin
/// that the rule above is about operand order and not about `Sub`.
#[test]
fn a_tensor_minus_a_scalar_is_a_chain() {
    const FORWARD: &str = r#"
struct Params { N: u32, s: f32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.N) { return; }
  output[idx] = x[idx] - params.s;
}
"#;
    let pattern = classify(FORWARD);
    let (base, cast, steps) = chain_of(&pattern);
    assert_eq!(base, "x");
    assert_eq!(cast, None);
    assert_eq!(steps, vec![(ElementWiseOp::Sub, "s".to_string(), true)]);
}

/// The anchors. Four predicates were tried on this file and each broke one of
/// these; the chain matcher must leave every one of them alone.
#[test]
fn the_anchors_keep_their_classification() {
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
    const PERMUTE: &str = r#"
struct Params { D0: u32, D1: u32, D2: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.D0 * params.D1 * params.D2) { return; }
  let i2 = idx % params.D2;
  let rest = idx / params.D2;
  let i1 = rest % params.D1;
  let i0 = rest / params.D1;
  output[(i2 * params.D1 + i1) * params.D0 + i0] = input[idx];
}
"#;
    assert!(matches!(classify(MATMUL), KernelPattern::MatMul { .. }));
    assert!(matches!(classify(PERMUTE), KernelPattern::Transpose { .. }));
}

// ---------------------------------------------------------------------------
// What the backends make of it
// ---------------------------------------------------------------------------

fn onnx_bytes(output: &nxpu_backend_core::BackendOutput) -> Vec<u8> {
    match &output.files[0].content {
        OutputContent::Binary(b) => b.clone(),
        other => panic!("expected binary ONNX output, got {other:?}"),
    }
}

fn load_proto(bytes: &[u8]) -> nxpu_backend_onnx::proto::ModelProto {
    use prost::Message;
    nxpu_backend_onnx::proto::ModelProto::decode(bytes).expect("ONNX decode failed")
}

/// Compile to ONNX, fix `N` to the length given, and run it through tract.
fn run_onnx(source: &str, n: usize, inputs: Vec<Tensor>) -> Vec<f32> {
    let output = common::compile_wgsl(source, &OnnxBackend, 1);
    let bytes = onnx_bytes(&output);
    let model = onnx()
        .model_for_read(&mut std::io::Cursor::new(bytes))
        .expect("tract could not load the model");
    let mut model = model.into_typed().expect("typing failed");

    for (i, tensor) in inputs.iter().enumerate() {
        let id = model.inputs[i];
        let dt = model.outlet_fact(id).unwrap().datum_type;
        let shape: Vec<usize> = if tensor.rank() == 0 { vec![] } else { vec![n] };
        model
            .set_input_fact(i, TypedFact::dt_shape(dt, shape))
            .expect("could not fix the input shape");
    }

    let result = model
        .into_optimized()
        .expect("optimization failed")
        .into_runnable()
        .expect("could not make the model runnable")
        .run(inputs.into_iter().map(Into::into).collect::<TVec<_>>())
        .expect("inference failed");

    result[0]
        .try_as_plain()
        .expect("output is not plainly stored")
        .to_array_view::<f32>()
        .expect("output is not f32")
        .iter()
        .copied()
        .collect()
}

/// The compiled graph has to compute what the kernel computes. Two nodes with
/// the right names is not the same claim as `y + a*x`.
#[test]
fn axpy_lowers_to_onnx_that_computes_y_plus_a_times_x() {
    let x = vec![1.0f32, 2.0, 3.0, 4.0];
    let y = vec![10.0f32, 20.0, 30.0, 40.0];
    let a = 0.5f32;

    let got = run_onnx(AXPY, 4, vec![tensor1(&x), Tensor::from(a), tensor1(&y)]);

    let want: Vec<f32> = x.iter().zip(&y).map(|(x, y)| y + a * x).collect();
    assert_eq!(got, want);
}

/// Same graph, and the scale must land on `x` rather than on `y`: with
/// `a = 0.5` and the operands swapped the answer would be `x + 0.5*y`, which
/// is a different tensor for these values and equal for symmetric ones.
#[test]
fn axpy_scales_the_right_operand() {
    let x = vec![2.0f32, 2.0];
    let y = vec![8.0f32, 8.0];
    let got = run_onnx(
        AXPY,
        2,
        vec![tensor1(&x), Tensor::from(0.5f32), tensor1(&y)],
    );
    assert_eq!(got, vec![9.0, 9.0], "0.5*x + y, not x + 0.5*y");
}

/// The conversion has to happen, and it has to be a conversion: `f32(i)`, not
/// the bits of `i` read as an f32.
#[test]
fn dequantize_lowers_to_onnx_that_converts_and_scales() {
    let codes = vec![1i32, -2, 3, -4];
    let (s1, s2) = (0.25f32, 2.0f32);

    let got = run_onnx(
        DEQUANTIZE,
        4,
        vec![
            Tensor::from_shape(&[4], &codes).unwrap(),
            Tensor::from(s1),
            Tensor::from(s2),
        ],
    );

    let want: Vec<f32> = codes.iter().map(|c| *c as f32 * s1 * s2).collect();
    assert_eq!(got, want);
}

fn tensor1(v: &[f32]) -> Tensor {
    Tensor::from_shape(&[v.len()], v).unwrap()
}

/// Both backends have to accept it. A pattern only one of them can lower is a
/// refusal moved further away from the kernel that caused it.
#[test]
fn both_backends_lower_every_chain() {
    for (name, source) in [
        ("axpy", AXPY),
        ("axpy_inplace", AXPY_INPLACE),
        ("dequantize", DEQUANTIZE),
    ] {
        for level in [0u8, 2] {
            let onnx = common::compile_wgsl(source, &OnnxBackend, level);
            assert_eq!(onnx.files.len(), 1, "{name}: one ONNX file");
            let tflite = common::compile_wgsl(source, &TfLiteBackend, level);
            assert_eq!(tflite.files.len(), 1, "{name}: one TFLite file");
            let bytes = common::first_binary(&tflite);
            assert!(bytes.len() > 8 && &bytes[4..8] == b"TFL3", "{name}: TFL3");
            // A `-1` reaching a `Tensor.shape` overflows `BytesRequired` and no
            // interpreter can be built from the file, on any device. The
            // rank-0 scalars are the new shape here, so this is the check that
            // matters for them.
            assert!(
                !bytes.windows(4).any(|w| w == [0xff, 0xff, 0xff, 0xff]),
                "{name}: a negative extent reached the model"
            );
        }
    }
}

/// The diagnostic names the operators emitted, in order, so a reader can check
/// it against the source. "ElementWiseChain" would name a category and say
/// nothing about what the kernel computes.
#[test]
fn the_diagnostic_names_the_operators() {
    for (source, expected) in [(AXPY, "Mul+Add"), (DEQUANTIZE, "Cast+Mul+Mul")] {
        for backend in [&OnnxBackend as &dyn Backend, &TfLiteBackend] {
            let mut module = nxpu_parser::parse(source).unwrap();
            PassManager::for_level(OptLevel::O1).run(&mut module);
            let out = backend
                .compile(&module, &BackendOptions::default())
                .unwrap();
            assert!(
                out.diagnostics
                    .iter()
                    .any(|d| d.message.contains(&format!("classified as {expected}"))),
                "{}: expected `classified as {expected}` in {:?}",
                backend.name(),
                out.diagnostics
            );
        }
    }
}

/// The ONNX graph is the chain, node for node: a `Cast` when the kernel
/// converts, then one node per step, threaded through named intermediates.
#[test]
fn the_onnx_graph_is_one_node_per_step() {
    let bytes = onnx_bytes(&common::compile_wgsl(DEQUANTIZE, &OnnxBackend, 1));
    let graph = load_proto(&bytes).graph.expect("graph");

    let ops: Vec<&str> = graph.node.iter().map(|n| n.op_type.as_str()).collect();
    assert_eq!(ops, vec!["Cast", "Mul", "Mul"]);

    // The value each node produces is the value the next one consumes.
    for pair in graph.node.windows(2) {
        assert_eq!(
            pair[0].output[0], pair[1].input[0],
            "the chain has to be threaded, not left dangling"
        );
    }
    assert_eq!(
        graph.node.last().unwrap().output[0],
        graph.output[0].name,
        "the last node writes the graph's output"
    );
}

/// A vendor support matrix is asked about each operator the chain emits, not
/// about the joined name. Asking about `Mul+Add` reported it unsupported on
/// hardware that has both a multiply and an add.
#[test]
fn vendor_validation_asks_about_each_operator() {
    let mut module = nxpu_parser::parse(AXPY).unwrap();
    PassManager::for_level(OptLevel::O1).run(&mut module);
    // f32, where XDNA emulates both operators and so has something to say
    // about each. At its preferred int8 both are native and silent.
    let opts = BackendOptions {
        precision: nxpu_backend_core::PrecisionPolicy::Keep,
        ..Default::default()
    };
    let out = nxpu_backend_amd::AmdBackend
        .compile(&module, &opts)
        .expect("the chain still compiles through the ONNX path");

    let messages: Vec<&str> = out.diagnostics.iter().map(|d| d.message.as_str()).collect();
    assert!(
        messages.iter().any(|m| m.contains("'Mul'")),
        "expected a verdict on Mul, got {messages:?}"
    );
    assert!(
        messages.iter().any(|m| m.contains("'Add'")),
        "expected a verdict on Add, got {messages:?}"
    );
    assert!(
        !messages.iter().any(|m| m.contains("'Mul+Add' at")),
        "the matrix is not asked about a fused name it does not list: {messages:?}"
    );
}

/// The backends that cannot express a chain say so rather than emitting the
/// first step. A model that runs and computes something else is worse than a
/// refusal that names the shape.
#[test]
fn the_backends_that_cannot_lower_it_refuse() {
    let mut module = nxpu_parser::parse(AXPY).unwrap();
    PassManager::for_level(OptLevel::O1).run(&mut module);

    for backend in [
        &nxpu_backend_coreml::CoreMlBackend as &dyn Backend,
        &nxpu_backend_stablehlo::StableHloBackend,
    ] {
        let err = backend
            .compile(&module, &BackendOptions::default())
            .expect_err("expected a refusal");
        let message = err.to_string();
        assert!(
            message.contains("Mul+Add"),
            "{}: the refusal should name what it could not lower, got {message}",
            backend.name()
        );
    }
}
