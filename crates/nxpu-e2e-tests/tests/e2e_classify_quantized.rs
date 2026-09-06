//! Matrix multiplication against integer weight codes.
//!
//! Six kernels in `vendor/web-xpu-ops` are quantized matmuls, and they were the
//! largest single group the classifier refused: three or more inputs and no
//! recogniser, because no pattern could carry a weight that arrives as codes
//! plus the scale that turns them back into numbers.
//!
//! These pin what the recogniser keys on, in both directions. What it must
//! accept: an int8 weight with one scale per row, with and without a residual
//! fused on. What it must refuse, and with what reason: a scale per *block* of
//! contracted columns, four-bit codes, and two weights in one kernel. And what
//! it must leave alone: a dense matmul, which unpacks nothing.
//!
//! The kernels are reduced from the vendored ones so the file does not need
//! the submodule; the vendored originals stream a row across 256 lanes and
//! reduce through workgroup memory, which changes none of the evidence read
//! here.

mod common;

use nxpu_analysis::analyze::{KernelPattern, classify_entry_point, data_type};
use nxpu_backend_core::OutputContent;
use nxpu_backend_onnx::OnnxBackend;
use nxpu_backend_tflite::TfLiteBackend;
use nxpu_opt::{OptLevel, PassManager};

/// `out[i] = (sum_k unpack(weight[i,k]) * vector[k]) * scale[i]` —
/// `ops/matvec/wgsl/q8.wgsl`.
const MATVEC_Q8: &str = r#"
struct Params { N: u32, K: u32 }
@group(0) @binding(0) var<storage, read> weight: array<u32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> vector: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
fn unpack_i8(word: u32, lane: u32) -> f32 {
  return f32(extractBits(bitcast<i32>(word), lane * 8u, 8u));
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>) {
  let row = wg_id.x;
  let words_per_row = (params.K + 3u) / 4u;
  let row_word_offset = row * words_per_row;
  var partial: f32 = 0.0;
  for (var word_index = 0u; word_index < words_per_row; word_index += 1u) {
    let word = weight[row_word_offset + word_index];
    let base_col = word_index * 4u;
    for (var lane = 0u; lane < 4u; lane += 1u) {
      let col = base_col + lane;
      if (col >= params.K) { break; }
      partial += unpack_i8(word, lane) * vector[col];
    }
  }
  output[row] = partial * scale[row];
}
"#;

/// The same with a residual added after the row scale —
/// `ops/matvec/wgsl/q8_residual.wgsl`.
const MATVEC_Q8_RESIDUAL: &str = r#"
struct Params { N: u32, K: u32 }
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
fn main(@builtin(workgroup_id) wg_id: vec3<u32>) {
  let row = wg_id.x;
  let words_per_row = (params.K + 3u) / 4u;
  let row_word_offset = row * words_per_row;
  var partial: f32 = 0.0;
  for (var word_index = 0u; word_index < words_per_row; word_index += 1u) {
    let word = weight[row_word_offset + word_index];
    let base_col = word_index * 4u;
    for (var lane = 0u; lane < 4u; lane += 1u) {
      let col = base_col + lane;
      if (col >= params.K) { break; }
      partial += unpack_i8(word, lane) * vector[col];
    }
  }
  output[row] = partial * scale[row] + residual[row];
}
"#;

/// `output[n, m] = sum_k a[n, k] * unpack(weight[m, k]) * scale[m]` —
/// `ops/matmul/wgsl/q8.wgsl`, untiled. Its params are `(N, M, K)`, with `N` the
/// rows of `a` and `M` the output channels, so the axis names are *not* in the
/// order the dense matmul's positional reading would give them.
const MATMUL_Q8: &str = r#"
struct Params { N: u32, M: u32, K: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
fn unpack_i8(word: u32, lane: u32) -> f32 {
  return f32(extractBits(bitcast<i32>(word), lane * 8u, 8u));
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.y;
  let col = gid.x;
  if (row >= params.N || col >= params.M) { return; }
  let words_per_row = (params.K + 3u) / 4u;
  var acc: f32 = 0.0;
  for (var k = 0u; k < params.K; k += 1u) {
    let word = weight[col * words_per_row + (k >> 2u)];
    let value = unpack_i8(word, k & 3u) * scale[col];
    acc = acc + a[row * params.K + k] * value;
  }
  output[row * params.M + col] = acc;
}
"#;

/// Four-bit codes with a scale per group of 128 contracted columns —
/// `ops/matvec/wgsl/q4_g128.wgsl`.
const MATVEC_Q4_G128: &str = r#"
struct Params { N: u32, K: u32 }
@group(0) @binding(0) var<storage, read> weight: array<u32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> vector: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
fn unpack_i4(word: u32, lane: u32) -> f32 {
  return f32(extractBits(bitcast<i32>(word), lane * 4u, 4u));
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>) {
  let row = wg_id.x;
  let words_per_row = (params.K + 7u) / 8u;
  let groups_per_row = (params.K + 127u) / 128u;
  let row_word_offset = row * words_per_row;
  let row_group_offset = row * groups_per_row;
  var partial: f32 = 0.0;
  for (var word_index = 0u; word_index < words_per_row; word_index += 1u) {
    let word = weight[row_word_offset + word_index];
    let base_col = word_index * 8u;
    let group_scale = scale[row_group_offset + base_col / 128u];
    var word_sum: f32 = 0.0;
    for (var lane = 0u; lane < 8u; lane += 1u) {
      let col = base_col + lane;
      if (col >= params.K) { break; }
      word_sum += unpack_i4(word, lane) * vector[col];
    }
    partial += word_sum * group_scale;
  }
  output[row] = partial;
}
"#;

/// A gate and an up projection sharing one activation, with a SiLU and a
/// multiply on top — `ops/matvec/wgsl/q8_ffn.wgsl`.
const MATVEC_Q8_FFN: &str = r#"
struct Params { N: u32, K: u32 }
@group(0) @binding(0) var<storage, read> weight_gate: array<u32>;
@group(0) @binding(1) var<storage, read> scale_gate: array<f32>;
@group(0) @binding(2) var<storage, read> weight_up: array<u32>;
@group(0) @binding(3) var<storage, read> scale_up: array<f32>;
@group(0) @binding(4) var<storage, read> vector: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;
fn unpack_i8(word: u32, lane: u32) -> f32 {
  return f32(extractBits(bitcast<i32>(word), lane * 8u, 8u));
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>) {
  let row = wg_id.x;
  let words_per_row = (params.K + 3u) / 4u;
  let row_word_offset = row * words_per_row;
  var partial_gate: f32 = 0.0;
  var partial_up: f32 = 0.0;
  for (var word_index = 0u; word_index < words_per_row; word_index += 1u) {
    let wg = weight_gate[row_word_offset + word_index];
    let wu = weight_up[row_word_offset + word_index];
    let base_col = word_index * 4u;
    for (var lane = 0u; lane < 4u; lane += 1u) {
      let col = base_col + lane;
      if (col >= params.K) { break; }
      let v = vector[col];
      partial_gate += unpack_i8(wg, lane) * v;
      partial_up += unpack_i8(wu, lane) * v;
    }
  }
  let g = partial_gate * scale_gate[row];
  let u = partial_up * scale_up[row];
  output[row] = (g / (1.0 + exp(-g))) * u;
}
"#;

/// A dense matmul, and a dense matmul with a bias. Neither unpacks anything, so
/// the quantized recogniser has to leave both alone — it was added at the top
/// of the three-or-more-input arm, ahead of Attention and Normalization, and a
/// recogniser that answered for every looping kernel with three buffers would
/// swallow those too.
const DENSE_MATMUL: &str = r#"
struct Params { M: u32, N: u32, K: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.y;
  let col = gid.x;
  if (row >= params.M || col >= params.N) { return; }
  var acc = 0.0;
  for (var k = 0u; k < params.K; k = k + 1u) {
    acc = acc + a[row * params.K + k] * b[k * params.N + col];
  }
  output[row * params.N + col] = acc;
}
"#;

const DENSE_MATMUL_WITH_BIAS: &str = r#"
struct Params { M: u32, N: u32, K: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.y;
  let col = gid.x;
  if (row >= params.M || col >= params.N) { return; }
  var acc = 0.0;
  for (var k = 0u; k < params.K; k = k + 1u) {
    acc = acc + a[row * params.K + k] * b[k * params.N + col];
  }
  output[row * params.N + col] = acc + bias[col];
}
"#;

fn classify(source: &str) -> KernelPattern {
    let mut module = nxpu_parser::parse(source).expect("WGSL parse failed");
    PassManager::for_level(OptLevel::O1).run(&mut module);
    classify_entry_point(&module, 0).expect("classification errored")
}

fn refusal(source: &str) -> String {
    match classify(source) {
        KernelPattern::Unknown { reason } => reason,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn a_row_scaled_int8_matvec_names_every_operand() {
    let KernelPattern::QuantizedMatMul {
        input,
        weight,
        scale,
        bias,
        output,
        shape,
    } = classify(MATVEC_Q8)
    else {
        panic!(
            "matvec/q8 is not a quantized matmul: {}",
            classify(MATVEC_Q8)
        );
    };
    assert_eq!(input.name, "vector");
    assert_eq!(weight.name, "weight");
    assert_eq!(scale.name, "scale");
    assert_eq!(output.name, "output");
    assert!(bias.is_none(), "matvec/q8 fuses nothing onto its result");
    // The buffer is `array<u32>` of packed words; what the graph carries is the
    // codes inside them, and every consumer has to be told that.
    assert_eq!(weight.elem_type, data_type::INT8);
    assert_eq!(input.elem_type, data_type::FLOAT);
    assert_eq!(scale.elem_type, data_type::FLOAT);
    // One row of results, `N` output channels, contracted over `K`.
    assert_eq!(shape.m, "1");
    assert_eq!(shape.n, "N");
    assert_eq!(shape.k, "K");
}

#[test]
fn a_residual_becomes_the_bias_and_the_scale_stays_the_scale() {
    let KernelPattern::QuantizedMatMul {
        scale, bias, shape, ..
    } = classify(MATVEC_Q8_RESIDUAL)
    else {
        panic!("matvec/q8_residual is not a quantized matmul");
    };
    // Both are read at `row`. What tells them apart is that one multiplies the
    // contraction and the other is added to it, and getting that backwards
    // would scale by the residual and add the scale.
    assert_eq!(scale.name, "scale");
    assert_eq!(
        bias.map(|b| b.name).as_deref(),
        Some("residual"),
        "the residual has to survive as the bias"
    );
    assert_eq!(
        (shape.m.as_str(), shape.n.as_str(), shape.k.as_str()),
        ("1", "N", "K")
    );
}

#[test]
fn the_axes_come_from_the_kernel_and_not_from_their_position() {
    let KernelPattern::QuantizedMatMul { shape, input, .. } = classify(MATMUL_Q8) else {
        panic!("matmul/q8 is not a quantized matmul");
    };
    assert_eq!(input.name, "a");
    // Reading the params positionally the way the dense matmul does would give
    // (M, N, K) = (N, M, K) — the rows and the channels swapped. `K` is named
    // by the weight's row stride and `M` by the result's, so both are read out
    // of the arithmetic instead.
    assert_eq!(shape.m, "N", "the rows of `a`");
    assert_eq!(shape.n, "M", "the output channels");
    assert_eq!(shape.k, "K", "the contracted extent");
}

#[test]
fn a_block_wise_scale_is_refused_by_name() {
    let reason = refusal(MATVEC_Q4_G128);
    assert!(
        reason.contains("block"),
        "the refusal has to say the scale is block-wise: {reason}"
    );
    assert!(
        reason.contains("per-channel") || reason.contains("per output channel"),
        "and say what it would have to be instead: {reason}"
    );
}

#[test]
fn two_packed_weights_are_refused_by_name() {
    let reason = refusal(MATVEC_Q8_FFN);
    assert!(
        reason.contains("packed weight"),
        "the refusal has to name the two weights: {reason}"
    );
}

#[test]
fn a_dense_matmul_is_still_a_dense_matmul() {
    assert!(
        matches!(classify(DENSE_MATMUL), KernelPattern::MatMul { .. }),
        "a kernel that unpacks nothing must not reach the quantized arm: {}",
        classify(DENSE_MATMUL)
    );
    // Three inputs, a loop, a per-output-channel operand added to the result —
    // the shape a quantized matmul has, minus the unpacking. It was refused
    // before this recogniser existed and has to be refused the same way now:
    // the quantized arm runs first and has to answer `None` for it.
    let reason = refusal(DENSE_MATMUL_WITH_BIAS);
    assert!(
        reason.contains("3+ inputs but no recognized pattern"),
        "a dense matmul with a bias must keep the refusal it already had, got: {reason}"
    );
}

/// The pattern's operator names are what a vendor support matrix is asked
/// about, and both backends emit the same four.
#[test]
fn the_operator_names_are_the_nodes_that_get_emitted() {
    let names = nxpu_analysis::analyze::pattern_op_names(&classify(MATVEC_Q8));
    assert_eq!(names, ["Transpose", "DequantizeLinear", "MatMul"]);
    let with_bias = nxpu_analysis::analyze::pattern_op_names(&classify(MATVEC_Q8_RESIDUAL));
    assert_eq!(
        with_bias,
        ["Transpose", "DequantizeLinear", "MatMul", "Add"]
    );
}

// ---------------------------------------------------------------------------
// What comes out the other end
// ---------------------------------------------------------------------------

fn compile(backend: &dyn nxpu_backend_core::Backend, source: &str, extent: u32) -> Vec<u8> {
    let mut module = nxpu_parser::parse(source).expect("WGSL parse failed");
    PassManager::for_level(OptLevel::O1).run(&mut module);
    let output = backend
        .compile(
            &module,
            &nxpu_backend_core::BackendOptions {
                symbolic_extent: Some(extent),
                ..Default::default()
            },
        )
        .expect("backend compilation failed");
    match &output.files[0].content {
        OutputContent::Binary(b) => b.clone(),
        OutputContent::Text(t) => t.clone().into_bytes(),
    }
}

fn contains(haystack: &[u8], needle: &str) -> bool {
    haystack
        .windows(needle.len())
        .any(|w| w == needle.as_bytes())
}

#[test]
fn tflite_carries_the_scale_and_no_unresolved_dimension() {
    let bytes = compile(&TfLiteBackend, MATVEC_Q8_RESIDUAL, 8);
    assert_eq!(&bytes[4..8], b"TFL3");
    for name in ["weight", "scale", "vector", "residual", "output"] {
        assert!(contains(&bytes, name), "`{name}` is not in the model");
    }
    // A `-1` reaching a shape is how this backend shipped months of models
    // nothing could load, and the quantized graph writes four shapes of its own.
    assert!(
        !bytes.windows(4).any(|w| w == [0xff, 0xff, 0xff, 0xff]),
        "an unresolved dimension reached the model"
    );
}

/// The int8 tensor is the whole point, so it is asserted rather than assumed:
/// TFLite's `TensorType.INT8` is 9, and a tensor left FLOAT32 would be read as
/// four times as many bytes of something else.
#[test]
fn the_weight_goes_out_as_int8() {
    let types = tflite_tensor_types(&compile(&TfLiteBackend, MATVEC_Q8, 8));
    // vector, weight, scale, perm, transposed codes, dequantized codes,
    // unscaled result, output.
    assert_eq!(types.len(), 8, "unexpected tensor count: {types:?}");
    // 9 is `TensorType.INT8`, 2 is INT32, 0 is FLOAT32. Two int8 tensors: the
    // codes as bound, and the codes transposed. One int32: the permutation.
    assert_eq!(
        types.iter().filter(|t| **t == 9).count(),
        2,
        "the weight and its transpose have to stay int8: {types:?}"
    );
    assert_eq!(
        types.iter().filter(|t| **t == 2).count(),
        1,
        "the transpose permutation is an int32 constant: {types:?}"
    );
}

/// Read every tensor's `type` field out of the flatbuffer.
///
/// Hand-decoded: this crate writes TFLite with the low-level builder and has no
/// reader, and only the four hops from the root to a tensor's type slot are
/// needed. The slot numbers are `nxpu_backend_tflite::schema::vt`'s, repeated
/// here so that a wrong one in the writer cannot be confirmed by the same wrong
/// one in the test.
fn tflite_tensor_types(bytes: &[u8]) -> Vec<i8> {
    // A `uoffset` is unsigned and relative to its own position; the root's is
    // at 0, ahead of the "TFL3" identifier at 4.
    let follow = |o: usize| o + u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) as usize;
    let i16_at = |o: usize| i16::from_le_bytes(bytes[o..o + 2].try_into().unwrap());
    // A table starts with a *signed* offset back to its vtable, which lists one
    // 16-bit offset per slot and is truncated after the last one written.
    let field = |table: usize, slot: u16| -> Option<usize> {
        let back = i32::from_le_bytes(bytes[table..table + 4].try_into().unwrap());
        let vtable = (table as i64 - back as i64) as usize;
        if slot as usize + 2 > i16_at(vtable) as usize {
            return None;
        }
        let off = i16_at(vtable + slot as usize);
        (off != 0).then(|| table + off as usize)
    };
    // A vector field holds a uoffset to a length followed by that many
    // elements; a vector of tables holds one uoffset per element.
    let element = |vector: usize, i: usize| follow(vector + 4 + i * 4);

    let root = follow(0);
    let subgraphs = follow(field(root, 8).expect("model has no subgraphs"));
    let subgraph = element(subgraphs, 0);
    let tensors = follow(field(subgraph, 4).expect("subgraph has no tensors"));
    let count = u32::from_le_bytes(bytes[tensors..tensors + 4].try_into().unwrap()) as usize;
    (0..count)
        .map(|i| {
            // TFLite omits a field whose value equals its default, and
            // FLOAT32 is 0 — an absent type slot means f32, not a broken read.
            field(element(tensors, i), 6).map_or(0, |p| bytes[p] as i8)
        })
        .collect()
}

#[test]
fn onnx_dequantizes_along_the_axis_the_scales_are_per() {
    let bytes = compile(&OnnxBackend, MATVEC_Q8, 8);
    for node in ["Transpose", "DequantizeLinear", "MatMul"] {
        assert!(contains(&bytes, node), "no {node} node in the graph");
    }
    // The transpose runs first, so the output channels are axis 1 by the time
    // the dequantization sees them. Emitting the two the other way round is the
    // same arithmetic and a model onnxruntime rejects.
    let model = onnx_node_types(&bytes);
    assert_eq!(
        model,
        ["Transpose", "DequantizeLinear", "MatMul"],
        "the node order is what keeps onnxruntime's transpose optimizer out of it"
    );
}

/// The `op_type` of every node, in order.
fn onnx_node_types(bytes: &[u8]) -> Vec<String> {
    use nxpu_backend_onnx::proto::ModelProto;
    use prost::Message;
    let model = ModelProto::decode(bytes).expect("emitted bytes are not a ModelProto");
    model
        .graph
        .expect("no graph")
        .node
        .into_iter()
        .map(|n| n.op_type)
        .collect()
}
