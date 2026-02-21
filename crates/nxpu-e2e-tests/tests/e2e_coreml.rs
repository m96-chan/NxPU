mod common;

use nxpu_backend_coreml::CoreMlBackend;
use nxpu_backend_coreml::proto::{self, Model, model};
use prost::Message;

fn decode_coreml(output: &nxpu_backend_core::BackendOutput) -> Model {
    let bytes = common::first_binary(output);
    Model::decode(bytes.as_ref()).expect("failed to decode CoreML model")
}

fn get_mil_ops(model: &Model) -> &[proto::MlOperation] {
    let prog = match model.r#type.as_ref().unwrap() {
        model::Type::MlProgram(p) => p,
    };
    &prog.functions[0].block.as_ref().unwrap().operations
}

#[test]
fn matmul_coreml_structure() {
    let source = common::load_example("matmul");
    let output = common::compile_wgsl(&source, &CoreMlBackend, 1);
    let model = decode_coreml(&output);
    assert_eq!(model.specification_version, proto::SPECIFICATION_VERSION);
    let ops = get_mil_ops(&model);
    assert_eq!(ops.len(), 1);
    assert_eq!(ops[0].r#type, "matmul");
}

#[test]
fn vecadd_coreml_add_op() {
    let source = common::load_example("vecadd");
    let output = common::compile_wgsl(&source, &CoreMlBackend, 1);
    let model = decode_coreml(&output);
    let ops = get_mil_ops(&model);
    assert_eq!(ops[0].r#type, "add");
}

#[test]
fn vecsub_coreml_sub_op() {
    let source = common::load_example("vecsub");
    let output = common::compile_wgsl(&source, &CoreMlBackend, 1);
    let model = decode_coreml(&output);
    let ops = get_mil_ops(&model);
    assert_eq!(ops[0].r#type, "sub");
}

#[test]
fn vecmul_coreml_mul_op() {
    let source = common::load_example("vecmul");
    let output = common::compile_wgsl(&source, &CoreMlBackend, 1);
    let model = decode_coreml(&output);
    let ops = get_mil_ops(&model);
    assert_eq!(ops[0].r#type, "mul");
}

#[test]
fn matmul_all_opt_levels() {
    let source = common::load_example("matmul");
    for level in [0, 1, 2] {
        let output = common::compile_wgsl(&source, &CoreMlBackend, level);
        let model = decode_coreml(&output);
        let ops = get_mil_ops(&model);
        assert_eq!(ops[0].r#type, "matmul", "failed at opt level {level}");
    }
}
