//! CoreML model construction from classified kernel patterns.
//!
//! Builds a CoreML ML Program model with MIL operations targeting
//! Apple Neural Engine (FP16 precision).

use nxpu_backend_onnx::analyze::{ElementWiseOp, KernelPattern, TensorBinding};
use nxpu_backend_onnx::proto::data_type;

use crate::proto::*;

/// Build a CoreML model from a classified kernel pattern.
pub fn build_model(pattern: &KernelPattern, ep_name: &str) -> Model {
    let (inputs, outputs, operations) = match pattern {
        KernelPattern::MatMul {
            inputs,
            output,
            shape,
        } => build_matmul(&inputs[0], &inputs[1], output, shape),
        KernelPattern::ElementWise {
            op, inputs, output, ..
        } => build_elementwise(*op, &inputs[0], &inputs[1], output),
    };

    let feature_inputs: Vec<FeatureDescription> = inputs
        .iter()
        .map(|b| {
            FeatureDescription::multi_array(&b.name, onnx_to_coreml_type(b.elem_type), &[-1, -1])
        })
        .collect();

    let feature_outputs: Vec<FeatureDescription> = outputs
        .iter()
        .map(|b| {
            FeatureDescription::multi_array(&b.name, onnx_to_coreml_type(b.elem_type), &[-1, -1])
        })
        .collect();

    let output_names = outputs.iter().map(|b| b.name.clone()).collect();

    Model {
        specification_version: SPECIFICATION_VERSION,
        description: Some(ModelDescription {
            input: feature_inputs,
            output: feature_outputs,
            metadata: Some(Metadata {
                author: "nxpu".into(),
                short_description: format!("NxPU {ep_name} kernel"),
            }),
        }),
        r#type: Some(model::Type::MlProgram(MlProgram {
            functions: vec![MlFunction {
                name: ep_name.into(),
                inputs: inputs
                    .iter()
                    .map(|b| NamedValueType {
                        name: b.name.clone(),
                        r#type: "fp16".into(),
                    })
                    .collect(),
                block: Some(MlBlock {
                    operations,
                    outputs: output_names,
                }),
            }],
        })),
    }
}

fn onnx_to_coreml_type(onnx_dt: i32) -> ArrayDataType {
    match onnx_dt {
        data_type::FLOAT16 => ArrayDataType::Float16,
        data_type::INT32 => ArrayDataType::Int32,
        // INT8 not natively supported on ANE — fall back to FP16.
        data_type::INT8 => ArrayDataType::Float16,
        // ANE operates at FP16; promote FP32 to FP16.
        _ => ArrayDataType::Float16,
    }
}

fn build_matmul<'a>(
    a: &'a TensorBinding,
    b: &'a TensorBinding,
    c: &'a TensorBinding,
    _shape: &nxpu_backend_onnx::analyze::MatMulShape,
) -> (
    Vec<&'a TensorBinding>,
    Vec<&'a TensorBinding>,
    Vec<MlOperation>,
) {
    let op = MlOperation {
        r#type: "matmul".into(),
        name: "matmul_0".into(),
        inputs: vec![
            MlOperand {
                name: a.name.clone(),
            },
            MlOperand {
                name: b.name.clone(),
            },
        ],
        outputs: vec![MlOperand {
            name: c.name.clone(),
        }],
    };
    (vec![a, b], vec![c], vec![op])
}

fn build_elementwise<'a>(
    ew_op: ElementWiseOp,
    a: &'a TensorBinding,
    b: &'a TensorBinding,
    c: &'a TensorBinding,
) -> (
    Vec<&'a TensorBinding>,
    Vec<&'a TensorBinding>,
    Vec<MlOperation>,
) {
    let mil_op = match ew_op {
        ElementWiseOp::Add => "add",
        ElementWiseOp::Sub => "sub",
        ElementWiseOp::Mul => "mul",
        ElementWiseOp::Div => "real_div",
    };
    let op = MlOperation {
        r#type: mil_op.into(),
        name: format!("{mil_op}_0"),
        inputs: vec![
            MlOperand {
                name: a.name.clone(),
            },
            MlOperand {
                name: b.name.clone(),
            },
        ],
        outputs: vec![MlOperand {
            name: c.name.clone(),
        }],
    };
    (vec![a, b], vec![c], vec![op])
}

#[cfg(test)]
mod tests {
    use super::*;
    use nxpu_backend_onnx::analyze::{MatMulShape, TensorRole};

    fn dummy_handle() -> nxpu_ir::Handle<nxpu_ir::GlobalVariable> {
        let mut arena = nxpu_ir::Arena::new();
        arena.append(nxpu_ir::GlobalVariable {
            name: None,
            space: nxpu_ir::AddressSpace::Uniform,
            binding: None,
            ty: {
                let mut types = nxpu_ir::UniqueArena::new();
                types.insert(nxpu_ir::Type {
                    name: None,
                    inner: nxpu_ir::TypeInner::Scalar(nxpu_ir::Scalar::F32),
                })
            },
            init: None,
        })
    }

    fn make_tensor(name: &str, role: TensorRole) -> TensorBinding {
        TensorBinding {
            handle: dummy_handle(),
            name: name.into(),
            elem_type: data_type::FLOAT,
            role,
        }
    }

    #[test]
    fn matmul_model_structure() {
        let pattern = KernelPattern::MatMul {
            inputs: [
                make_tensor("A", TensorRole::Input),
                make_tensor("B", TensorRole::Input),
            ],
            output: make_tensor("C", TensorRole::Output),
            shape: MatMulShape {
                m: "M".into(),
                n: "N".into(),
                k: "K".into(),
            },
        };

        let model = build_model(&pattern, "matmul_kernel");
        assert_eq!(model.specification_version, SPECIFICATION_VERSION);

        let prog = match model.r#type.as_ref().unwrap() {
            model::Type::MlProgram(p) => p,
        };
        assert_eq!(prog.functions.len(), 1);
        let block = prog.functions[0].block.as_ref().unwrap();
        assert_eq!(block.operations.len(), 1);
        assert_eq!(block.operations[0].r#type, "matmul");
    }

    #[test]
    fn elementwise_add_model() {
        let pattern = KernelPattern::ElementWise {
            op: ElementWiseOp::Add,
            inputs: [
                make_tensor("x", TensorRole::Input),
                make_tensor("y", TensorRole::Input),
            ],
            output: make_tensor("z", TensorRole::Output),
            dim_name: "N".into(),
        };

        let model = build_model(&pattern, "vecadd");
        let prog = match model.r#type.as_ref().unwrap() {
            model::Type::MlProgram(p) => p,
        };
        let block = prog.functions[0].block.as_ref().unwrap();
        assert_eq!(block.operations[0].r#type, "add");
    }
}
