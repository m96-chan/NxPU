mod common;

use nxpu_backend_onnx::OnnxBackend;
use nxpu_backend_onnx::proto::{ModelProto, data_type};
use prost::Message;

fn decode_onnx(output: &nxpu_backend_core::BackendOutput) -> ModelProto {
    let bytes = common::first_binary(output);
    ModelProto::decode(bytes).expect("failed to decode ONNX model")
}

// --- MatMul ---

#[test]
fn matmul_o0() {
    let source = common::load_example("matmul");
    let output = common::compile_wgsl(&source, &OnnxBackend, 0);
    let model = decode_onnx(&output);
    let graph = model.graph.as_ref().unwrap();
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "MatMul");
}

#[test]
fn matmul_o1() {
    let source = common::load_example("matmul");
    let output = common::compile_wgsl(&source, &OnnxBackend, 1);
    let model = decode_onnx(&output);
    let graph = model.graph.as_ref().unwrap();
    assert_eq!(graph.node[0].op_type, "MatMul");
    // Verify tensor data types are FLOAT.
    for input in &graph.input {
        let tt = input.r#type.as_ref().unwrap();
        let tv = tt.value.as_ref().unwrap();
        let nxpu_backend_onnx::proto::type_proto::Value::TensorType(t) = tv;
        assert_eq!(t.elem_type, data_type::FLOAT);
    }
}

#[test]
fn matmul_o2() {
    let source = common::load_example("matmul");
    let output = common::compile_wgsl(&source, &OnnxBackend, 2);
    let model = decode_onnx(&output);
    assert_eq!(model.graph.unwrap().node[0].op_type, "MatMul");
}

// --- VecAdd ---

#[test]
fn vecadd_produces_add_node() {
    let source = common::load_example("vecadd");
    let output = common::compile_wgsl(&source, &OnnxBackend, 1);
    let model = decode_onnx(&output);
    let graph = model.graph.unwrap();
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "Add");
}

// --- VecSub ---

#[test]
fn vecsub_produces_sub_node() {
    let source = common::load_example("vecsub");
    let output = common::compile_wgsl(&source, &OnnxBackend, 1);
    let model = decode_onnx(&output);
    assert_eq!(model.graph.unwrap().node[0].op_type, "Sub");
}

// --- VecMul ---

#[test]
fn vecmul_produces_mul_node() {
    let source = common::load_example("vecmul");
    let output = common::compile_wgsl(&source, &OnnxBackend, 1);
    let model = decode_onnx(&output);
    assert_eq!(model.graph.unwrap().node[0].op_type, "Mul");
}

// --- All opt levels for vecadd ---

#[test]
fn vecadd_all_opt_levels() {
    let source = common::load_example("vecadd");
    for level in [0, 1, 2] {
        let output = common::compile_wgsl(&source, &OnnxBackend, level);
        let model = decode_onnx(&output);
        assert_eq!(
            model.graph.unwrap().node[0].op_type,
            "Add",
            "failed at opt level {level}"
        );
    }
}

// --- Conv2D ---

#[test]
fn conv2d_produces_conv_node() {
    let source = common::load_example("conv2d");
    let output = common::compile_wgsl(&source, &OnnxBackend, 1);
    let model = decode_onnx(&output);
    let graph = model.graph.unwrap();
    assert_eq!(graph.node[0].op_type, "Conv");
}

// --- ReLU ---

#[test]
fn relu_produces_relu_node() {
    let source = common::load_example("relu");
    let output = common::compile_wgsl(&source, &OnnxBackend, 1);
    let model = decode_onnx(&output);
    let graph = model.graph.unwrap();
    assert_eq!(graph.node[0].op_type, "Relu");
}

// --- Tanh ---

#[test]
fn tanh_produces_tanh_node() {
    let source = common::load_example("tanh_act");
    let output = common::compile_wgsl(&source, &OnnxBackend, 1);
    let model = decode_onnx(&output);
    let graph = model.graph.unwrap();
    assert_eq!(graph.node[0].op_type, "Tanh");
}

// --- Reduce Sum ---

#[test]
fn reduce_sum_produces_reducesum_node() {
    let source = common::load_example("reduce_sum");
    let output = common::compile_wgsl(&source, &OnnxBackend, 1);
    let model = decode_onnx(&output);
    let graph = model.graph.unwrap();
    assert_eq!(graph.node[0].op_type, "ReduceSum");
}

// --- Transpose ---

#[test]
fn transpose_produces_unknown_pattern() {
    // Transpose is now classified as Unknown (no silent fallback — #64).
    let source = common::load_example("transpose");
    let result = common::try_compile_wgsl(&source, &OnnxBackend, 1);
    assert!(result.is_err(), "expected Unsupported error for transpose");
}

// --- BatchNorm ---

#[test]
fn batchnorm_produces_unknown_pattern() {
    // BatchNorm is now classified as Unknown (no silent fallback — #64).
    let source = common::load_example("batchnorm");
    let result = common::try_compile_wgsl(&source, &OnnxBackend, 1);
    assert!(result.is_err(), "expected Unsupported error for batchnorm");
}

// --- MaxPool ---

#[test]
fn maxpool_produces_maxpool_node() {
    let source = common::load_example("maxpool");
    let output = common::compile_wgsl(&source, &OnnxBackend, 1);
    let model = decode_onnx(&output);
    let graph = model.graph.unwrap();
    assert_eq!(graph.node[0].op_type, "MaxPool");
}

// --- Concat ---

#[test]
fn concat_produces_concat_node() {
    let source = common::load_example("concat");
    let output = common::compile_wgsl(&source, &OnnxBackend, 1);
    let model = decode_onnx(&output);
    let graph = model.graph.unwrap();
    assert_eq!(graph.node[0].op_type, "Concat");
}

// --- Split ---

#[test]
fn split_produces_split_node() {
    let source = common::load_example("split");
    let output = common::compile_wgsl(&source, &OnnxBackend, 1);
    let model = decode_onnx(&output);
    let graph = model.graph.unwrap();
    assert_eq!(graph.node[0].op_type, "Split");
}

// --- Embedded weights ---

#[test]
fn vecadd_const_has_embedded_weights() {
    let source = common::load_example("vecadd_const");
    let module = common::parse_wgsl(&source);
    let weights = nxpu_analysis::extract_embedded_weights(&module);
    assert!(!weights.is_empty(), "expected at least one embedded weight");
    let bias = weights
        .iter()
        .find(|w| w.name == "bias")
        .expect("expected 'bias' weight");
    assert_eq!(bias.dims, vec![4]);
    assert_eq!(bias.data.len(), 4);
    assert!((bias.data[0] - 0.1).abs() < 1e-6);
    assert!((bias.data[3] - 0.4).abs() < 1e-6);
}

// --- Attention ---

#[test]
fn attention_produces_attention_subgraph() {
    let source = common::load_example("attention");
    let output = common::compile_wgsl(&source, &OnnxBackend, 1);
    let model = decode_onnx(&output);
    let graph = model.graph.unwrap();
    // Attention subgraph: Transpose + MatMul + Shape + Gather + Cast + Sqrt + Div + Softmax + MatMul = 9 nodes
    assert_eq!(graph.node.len(), 9);
    let op_types: Vec<&str> = graph.node.iter().map(|n| n.op_type.as_str()).collect();
    assert!(op_types.contains(&"Transpose"));
    assert!(op_types.contains(&"MatMul"));
    assert!(op_types.contains(&"Shape"));
    assert!(op_types.contains(&"Sqrt"));
    assert!(op_types.contains(&"Div"));
    assert!(op_types.contains(&"Softmax"));
}
