mod common;

use nxpu_backend_onnx::OnnxBackend;
use nxpu_backend_onnx::proto::{ModelProto, data_type};
use prost::Message;

fn decode_onnx(output: &nxpu_backend_core::BackendOutput) -> ModelProto {
    let bytes = common::first_binary(output);
    ModelProto::decode(bytes.as_ref()).expect("failed to decode ONNX model")
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
fn transpose_produces_transpose_node() {
    let source = common::load_example("transpose");
    let output = common::compile_wgsl(&source, &OnnxBackend, 1);
    let model = decode_onnx(&output);
    let graph = model.graph.unwrap();
    assert_eq!(graph.node[0].op_type, "Transpose");
}

// --- BatchNorm ---

#[test]
fn batchnorm_produces_batchnorm_node() {
    let source = common::load_example("batchnorm");
    let output = common::compile_wgsl(&source, &OnnxBackend, 1);
    let model = decode_onnx(&output);
    let graph = model.graph.unwrap();
    assert_eq!(graph.node[0].op_type, "BatchNormalization");
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
