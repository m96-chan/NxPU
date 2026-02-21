mod common;

use nxpu_backend_onnx::OnnxBackend;
use nxpu_backend_onnx::proto::{ModelProto, data_type, type_proto};
use nxpu_backend_stablehlo::StableHloBackend;
use nxpu_opt::{F32ToF16, F32ToInt8};
use prost::Message;

fn decode_onnx(output: &nxpu_backend_core::BackendOutput) -> ModelProto {
    let bytes = common::first_binary(output);
    ModelProto::decode(bytes.as_ref()).expect("failed to decode ONNX model")
}

fn get_input_elem_types(model: &ModelProto) -> Vec<i32> {
    model
        .graph
        .as_ref()
        .unwrap()
        .input
        .iter()
        .map(|vi| {
            let tt = vi.r#type.as_ref().unwrap();
            let type_proto::Value::TensorType(t) = tt.value.as_ref().unwrap();
            t.elem_type
        })
        .collect()
}

#[test]
fn f32_to_f16_onnx_pipeline() {
    let source = common::load_example("vecadd");
    let output = common::compile_wgsl_with_pass(&source, &F32ToF16, &OnnxBackend, 1);
    let model = decode_onnx(&output);
    let types = get_input_elem_types(&model);
    for dt in &types {
        assert_eq!(*dt, data_type::FLOAT16, "expected FLOAT16, got {dt}");
    }
}

#[test]
fn f32_to_int8_onnx_pipeline() {
    let source = common::load_example("vecadd");
    let output = common::compile_wgsl_with_pass(&source, &F32ToInt8::default(), &OnnxBackend, 1);
    let model = decode_onnx(&output);
    let types = get_input_elem_types(&model);
    for dt in &types {
        assert_eq!(*dt, data_type::INT8, "expected INT8, got {dt}");
    }
}

#[test]
fn f32_to_f16_stablehlo_pipeline() {
    let source = common::load_example("matmul");
    let output = common::compile_wgsl_with_pass(&source, &F32ToF16, &StableHloBackend, 1);
    let mlir = common::first_text(&output);
    assert!(
        mlir.contains("f16"),
        "expected f16 types in MLIR output:\n{mlir}"
    );
}

#[test]
fn f32_to_int8_stablehlo_pipeline() {
    let source = common::load_example("matmul");
    let output =
        common::compile_wgsl_with_pass(&source, &F32ToInt8::default(), &StableHloBackend, 1);
    let mlir = common::first_text(&output);
    assert!(
        mlir.contains("i8"),
        "expected i8 types in MLIR output:\n{mlir}"
    );
}

#[test]
fn matmul_f32_to_f16_onnx() {
    let source = common::load_example("matmul");
    let output = common::compile_wgsl_with_pass(&source, &F32ToF16, &OnnxBackend, 1);
    let model = decode_onnx(&output);
    let types = get_input_elem_types(&model);
    for dt in &types {
        assert_eq!(*dt, data_type::FLOAT16);
    }
}
