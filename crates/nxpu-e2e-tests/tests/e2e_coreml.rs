mod common;

use nxpu_backend_coreml::CoreMlBackend;
use nxpu_backend_coreml::proto::{self, Model, model};
use prost::Message;

fn decode_coreml(output: &nxpu_backend_core::BackendOutput) -> Model {
    let bytes = common::first_binary(output);
    Model::decode(bytes).expect("failed to decode CoreML model")
}

fn get_mil_ops(model: &Model) -> &[proto::MlOperation] {
    let model::Type::MlProgram(prog) = model.r#type.as_ref().unwrap();
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

#[test]
fn conv2d_coreml_conv_op() {
    let source = common::load_example("conv2d");
    let output = common::compile_wgsl(&source, &CoreMlBackend, 1);
    let model = decode_coreml(&output);
    let ops = get_mil_ops(&model);
    assert_eq!(ops[0].r#type, "conv");
}

#[test]
fn relu_coreml_relu_op() {
    let source = common::load_example("relu");
    let output = common::compile_wgsl(&source, &CoreMlBackend, 1);
    let model = decode_coreml(&output);
    let ops = get_mil_ops(&model);
    assert_eq!(ops[0].r#type, "relu");
}

#[test]
fn tanh_coreml_tanh_op() {
    let source = common::load_example("tanh_act");
    let output = common::compile_wgsl(&source, &CoreMlBackend, 1);
    let model = decode_coreml(&output);
    let ops = get_mil_ops(&model);
    assert_eq!(ops[0].r#type, "tanh");
}

#[test]
fn reduce_sum_coreml_reduce_op() {
    let source = common::load_example("reduce_sum");
    let output = common::compile_wgsl(&source, &CoreMlBackend, 1);
    let model = decode_coreml(&output);
    let ops = get_mil_ops(&model);
    assert_eq!(ops[0].r#type, "reduce_sum");
}

#[test]
fn transpose_coreml_transpose_op() {
    let source = common::load_example("transpose");
    let output = common::compile_wgsl(&source, &CoreMlBackend, 1);
    let model = decode_coreml(&output);
    let ops = get_mil_ops(&model);
    assert_eq!(ops[0].r#type, "transpose");
}

#[test]
fn batchnorm_coreml_batchnorm_op() {
    let source = common::load_example("batchnorm");
    let output = common::compile_wgsl(&source, &CoreMlBackend, 1);
    let model = decode_coreml(&output);
    let ops = get_mil_ops(&model);
    assert_eq!(ops[0].r#type, "batch_norm");
}

#[test]
fn maxpool_coreml_maxpool_op() {
    let source = common::load_example("maxpool");
    let output = common::compile_wgsl(&source, &CoreMlBackend, 1);
    let model = decode_coreml(&output);
    let ops = get_mil_ops(&model);
    assert_eq!(ops[0].r#type, "max_pool");
}
