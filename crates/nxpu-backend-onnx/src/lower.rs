//! ONNX graph construction from classified kernel patterns.
//!
//! Converts a [`KernelPattern`] into an ONNX [`ModelProto`] with the
//! appropriate graph topology.

use crate::analyze::{
    ActivationOp, Conv2DShape, ElementWiseOp, KernelPattern, MatMulShape, PoolKind, PoolShape,
    ReduceOp, TensorBinding,
};
use crate::proto::*;

/// Build an ONNX model from a classified kernel pattern.
pub fn build_model(pattern: &KernelPattern, ep_name: &str) -> ModelProto {
    let graph = match pattern {
        KernelPattern::MatMul {
            inputs,
            output,
            shape,
        } => build_matmul_graph(inputs, output, shape, ep_name),
        KernelPattern::ElementWise {
            op,
            inputs,
            output,
            dim_name,
        } => build_elementwise_graph(*op, inputs, output, dim_name, ep_name),
        KernelPattern::Conv2D {
            input,
            weight,
            output,
            shape,
        } => build_conv2d_graph(input, weight, output, shape, ep_name),
        KernelPattern::Pool {
            kind,
            input,
            output,
            shape,
        } => build_pool_graph(*kind, input, output, shape, ep_name),
        KernelPattern::Activation {
            op,
            input,
            output,
            dim_name,
        } => build_activation_graph(*op, input, output, dim_name, ep_name),
        KernelPattern::Reduce {
            op,
            input,
            output,
            axis,
        } => build_reduce_graph(*op, input, output, *axis, ep_name),
        KernelPattern::Transpose {
            input,
            output,
            perm,
        } => build_transpose_graph(input, output, perm, ep_name),
        KernelPattern::Reshape { input, output } => build_reshape_graph(input, output, ep_name),
        KernelPattern::Normalization {
            input,
            scale,
            bias,
            output,
            ..
        } => build_normalization_graph(input, scale, bias, output, ep_name),
    };

    ModelProto {
        ir_version: 7,
        producer_name: "nxpu".into(),
        producer_version: env!("CARGO_PKG_VERSION").into(),
        graph: Some(graph),
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
    }
}

fn build_matmul_graph(
    inputs: &[TensorBinding; 2],
    output: &TensorBinding,
    shape: &MatMulShape,
    ep_name: &str,
) -> GraphProto {
    let a_name = &inputs[0].name;
    let b_name = &inputs[1].name;
    let c_name = &output.name;

    GraphProto {
        name: ep_name.into(),
        node: vec![NodeProto::simple(
            "MatMul",
            "matmul_0",
            vec![a_name.clone(), b_name.clone()],
            vec![c_name.clone()],
        )],
        input: vec![
            ValueInfoProto::tensor(
                a_name,
                inputs[0].elem_type,
                vec![
                    TensorShapeDimension::symbolic(&shape.m),
                    TensorShapeDimension::symbolic(&shape.k),
                ],
            ),
            ValueInfoProto::tensor(
                b_name,
                inputs[1].elem_type,
                vec![
                    TensorShapeDimension::symbolic(&shape.k),
                    TensorShapeDimension::symbolic(&shape.n),
                ],
            ),
        ],
        output: vec![ValueInfoProto::tensor(
            c_name,
            output.elem_type,
            vec![
                TensorShapeDimension::symbolic(&shape.m),
                TensorShapeDimension::symbolic(&shape.n),
            ],
        )],
    }
}

fn build_elementwise_graph(
    op: ElementWiseOp,
    inputs: &[TensorBinding; 2],
    output: &TensorBinding,
    dim_name: &str,
    ep_name: &str,
) -> GraphProto {
    let a_name = &inputs[0].name;
    let b_name = &inputs[1].name;
    let c_name = &output.name;

    GraphProto {
        name: ep_name.into(),
        node: vec![NodeProto::simple(
            op.onnx_op_type(),
            format!("{}_0", op.onnx_op_type().to_lowercase()),
            vec![a_name.clone(), b_name.clone()],
            vec![c_name.clone()],
        )],
        input: vec![
            ValueInfoProto::tensor(
                a_name,
                inputs[0].elem_type,
                vec![TensorShapeDimension::symbolic(dim_name)],
            ),
            ValueInfoProto::tensor(
                b_name,
                inputs[1].elem_type,
                vec![TensorShapeDimension::symbolic(dim_name)],
            ),
        ],
        output: vec![ValueInfoProto::tensor(
            c_name,
            output.elem_type,
            vec![TensorShapeDimension::symbolic(dim_name)],
        )],
    }
}

fn build_conv2d_graph(
    input: &TensorBinding,
    weight: &TensorBinding,
    output: &TensorBinding,
    shape: &Conv2DShape,
    ep_name: &str,
) -> GraphProto {
    GraphProto {
        name: ep_name.into(),
        node: vec![NodeProto::with_attrs(
            "Conv",
            "conv_0",
            vec![input.name.clone(), weight.name.clone()],
            vec![output.name.clone()],
            vec![
                AttributeProto::ints(
                    "kernel_shape",
                    vec![shape.stride_h.max(1), shape.stride_w.max(1)],
                ),
                AttributeProto::ints(
                    "strides",
                    vec![shape.stride_h.max(1), shape.stride_w.max(1)],
                ),
                AttributeProto::ints(
                    "pads",
                    vec![shape.pad_h, shape.pad_w, shape.pad_h, shape.pad_w],
                ),
            ],
        )],
        input: vec![
            ValueInfoProto::tensor(
                &input.name,
                input.elem_type,
                vec![
                    TensorShapeDimension::symbolic(&shape.batch),
                    TensorShapeDimension::symbolic(&shape.channels_in),
                    TensorShapeDimension::symbolic(&shape.height),
                    TensorShapeDimension::symbolic(&shape.width),
                ],
            ),
            ValueInfoProto::tensor(
                &weight.name,
                weight.elem_type,
                vec![
                    TensorShapeDimension::symbolic(&shape.channels_out),
                    TensorShapeDimension::symbolic(&shape.channels_in),
                    TensorShapeDimension::symbolic(&shape.kernel_h),
                    TensorShapeDimension::symbolic(&shape.kernel_w),
                ],
            ),
        ],
        output: vec![ValueInfoProto::tensor(
            &output.name,
            output.elem_type,
            vec![
                TensorShapeDimension::symbolic(&shape.batch),
                TensorShapeDimension::symbolic(&shape.channels_out),
                TensorShapeDimension::symbolic("OH"),
                TensorShapeDimension::symbolic("OW"),
            ],
        )],
    }
}

fn build_pool_graph(
    kind: PoolKind,
    input: &TensorBinding,
    output: &TensorBinding,
    shape: &PoolShape,
    ep_name: &str,
) -> GraphProto {
    GraphProto {
        name: ep_name.into(),
        node: vec![NodeProto::with_attrs(
            kind.onnx_op_type(),
            format!("{}_0", kind.onnx_op_type().to_lowercase()),
            vec![input.name.clone()],
            vec![output.name.clone()],
            vec![
                AttributeProto::ints("kernel_shape", vec![shape.kernel_h, shape.kernel_w]),
                AttributeProto::ints("strides", vec![shape.stride_h, shape.stride_w]),
            ],
        )],
        input: vec![ValueInfoProto::tensor(
            &input.name,
            input.elem_type,
            vec![
                TensorShapeDimension::symbolic("N"),
                TensorShapeDimension::symbolic("C"),
                TensorShapeDimension::symbolic("H"),
                TensorShapeDimension::symbolic("W"),
            ],
        )],
        output: vec![ValueInfoProto::tensor(
            &output.name,
            output.elem_type,
            vec![
                TensorShapeDimension::symbolic("N"),
                TensorShapeDimension::symbolic("C"),
                TensorShapeDimension::symbolic("OH"),
                TensorShapeDimension::symbolic("OW"),
            ],
        )],
    }
}

fn build_activation_graph(
    op: ActivationOp,
    input: &TensorBinding,
    output: &TensorBinding,
    dim_name: &str,
    ep_name: &str,
) -> GraphProto {
    GraphProto {
        name: ep_name.into(),
        node: vec![NodeProto::simple(
            op.onnx_op_type(),
            format!("{}_0", op.onnx_op_type().to_lowercase()),
            vec![input.name.clone()],
            vec![output.name.clone()],
        )],
        input: vec![ValueInfoProto::tensor(
            &input.name,
            input.elem_type,
            vec![TensorShapeDimension::symbolic(dim_name)],
        )],
        output: vec![ValueInfoProto::tensor(
            &output.name,
            output.elem_type,
            vec![TensorShapeDimension::symbolic(dim_name)],
        )],
    }
}

fn build_reduce_graph(
    op: ReduceOp,
    input: &TensorBinding,
    output: &TensorBinding,
    axis: i64,
    ep_name: &str,
) -> GraphProto {
    GraphProto {
        name: ep_name.into(),
        node: vec![NodeProto::with_attrs(
            op.onnx_op_type(),
            format!("{}_0", op.onnx_op_type().to_lowercase()),
            vec![input.name.clone()],
            vec![output.name.clone()],
            vec![AttributeProto::ints("axes", vec![axis])],
        )],
        input: vec![ValueInfoProto::tensor(
            &input.name,
            input.elem_type,
            vec![
                TensorShapeDimension::symbolic("N"),
                TensorShapeDimension::symbolic("D"),
            ],
        )],
        output: vec![ValueInfoProto::tensor(
            &output.name,
            output.elem_type,
            vec![TensorShapeDimension::symbolic("N")],
        )],
    }
}

fn build_transpose_graph(
    input: &TensorBinding,
    output: &TensorBinding,
    perm: &[i64],
    ep_name: &str,
) -> GraphProto {
    let ndim = perm.len();
    let in_dims: Vec<_> = (0..ndim)
        .map(|i| TensorShapeDimension::symbolic(format!("d{i}")))
        .collect();
    let out_dims: Vec<_> = perm
        .iter()
        .map(|&p| TensorShapeDimension::symbolic(format!("d{p}")))
        .collect();

    GraphProto {
        name: ep_name.into(),
        node: vec![NodeProto::with_attrs(
            "Transpose",
            "transpose_0",
            vec![input.name.clone()],
            vec![output.name.clone()],
            vec![AttributeProto::ints("perm", perm.to_vec())],
        )],
        input: vec![ValueInfoProto::tensor(
            &input.name,
            input.elem_type,
            in_dims,
        )],
        output: vec![ValueInfoProto::tensor(
            &output.name,
            output.elem_type,
            out_dims,
        )],
    }
}

fn build_reshape_graph(input: &TensorBinding, output: &TensorBinding, ep_name: &str) -> GraphProto {
    GraphProto {
        name: ep_name.into(),
        node: vec![NodeProto::simple(
            "Reshape",
            "reshape_0",
            vec![input.name.clone()],
            vec![output.name.clone()],
        )],
        input: vec![ValueInfoProto::tensor(
            &input.name,
            input.elem_type,
            vec![TensorShapeDimension::symbolic("N")],
        )],
        output: vec![ValueInfoProto::tensor(
            &output.name,
            output.elem_type,
            vec![TensorShapeDimension::symbolic("N")],
        )],
    }
}

fn build_normalization_graph(
    input: &TensorBinding,
    scale: &TensorBinding,
    bias: &TensorBinding,
    output: &TensorBinding,
    ep_name: &str,
) -> GraphProto {
    // mean → empty string (runtime computed)
    // var → empty string (runtime computed)
    GraphProto {
        name: ep_name.into(),
        node: vec![NodeProto::with_attrs(
            "BatchNormalization",
            "batchnorm_0",
            vec![
                input.name.clone(),
                scale.name.clone(),
                bias.name.clone(),
                "running_mean".into(),
                "running_var".into(),
            ],
            vec![output.name.clone()],
            vec![AttributeProto::int("epsilon", 1065353216)], // IEEE float bit pattern for 1e-5 stored as int (convention)
        )],
        input: vec![
            ValueInfoProto::tensor(
                &input.name,
                input.elem_type,
                vec![
                    TensorShapeDimension::symbolic("N"),
                    TensorShapeDimension::symbolic("C"),
                    TensorShapeDimension::symbolic("H"),
                    TensorShapeDimension::symbolic("W"),
                ],
            ),
            ValueInfoProto::tensor(
                &scale.name,
                scale.elem_type,
                vec![TensorShapeDimension::symbolic("C")],
            ),
            ValueInfoProto::tensor(
                &bias.name,
                bias.elem_type,
                vec![TensorShapeDimension::symbolic("C")],
            ),
        ],
        output: vec![ValueInfoProto::tensor(
            &output.name,
            output.elem_type,
            vec![
                TensorShapeDimension::symbolic("N"),
                TensorShapeDimension::symbolic("C"),
                TensorShapeDimension::symbolic("H"),
                TensorShapeDimension::symbolic("W"),
            ],
        )],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze::TensorRole;
    use crate::proto::{data_type, tensor_shape_dimension, type_proto};
    use nxpu_ir::Handle;

    fn dummy_handle() -> Handle<nxpu_ir::GlobalVariable> {
        // We just need any handle for testing; the arena provides them.
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
            layout: None,
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

        assert_eq!(model.ir_version, 7);
        assert_eq!(model.producer_name, "nxpu");
        assert_eq!(model.opset_import.len(), 1);
        assert_eq!(model.opset_import[0].version, 13);

        let graph = model.graph.as_ref().unwrap();
        assert_eq!(graph.name, "matmul_kernel");
        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "MatMul");
        assert_eq!(graph.node[0].input, vec!["A", "B"]);
        assert_eq!(graph.node[0].output, vec!["C"]);

        assert_eq!(graph.input.len(), 2);
        assert_eq!(graph.output.len(), 1);

        // Verify A shape = [M, K].
        let a_type = graph.input[0].r#type.as_ref().unwrap();
        let a_tensor = match a_type.value.as_ref().unwrap() {
            type_proto::Value::TensorType(t) => t,
        };
        assert_eq!(a_tensor.elem_type, data_type::FLOAT);
        let a_dims = &a_tensor.shape.as_ref().unwrap().dim;
        assert_eq!(a_dims.len(), 2);
        assert_eq!(
            a_dims[0].value,
            Some(tensor_shape_dimension::Value::DimParam("M".into()))
        );
        assert_eq!(
            a_dims[1].value,
            Some(tensor_shape_dimension::Value::DimParam("K".into()))
        );

        // Verify C shape = [M, N].
        let c_type = graph.output[0].r#type.as_ref().unwrap();
        let c_tensor = match c_type.value.as_ref().unwrap() {
            type_proto::Value::TensorType(t) => t,
        };
        let c_dims = &c_tensor.shape.as_ref().unwrap().dim;
        assert_eq!(c_dims.len(), 2);
        assert_eq!(
            c_dims[0].value,
            Some(tensor_shape_dimension::Value::DimParam("M".into()))
        );
        assert_eq!(
            c_dims[1].value,
            Some(tensor_shape_dimension::Value::DimParam("N".into()))
        );
    }

    #[test]
    fn elementwise_add_model_structure() {
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
        let graph = model.graph.as_ref().unwrap();

        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "Add");
        assert_eq!(graph.node[0].name, "add_0");
        assert_eq!(graph.node[0].input, vec!["x", "y"]);
        assert_eq!(graph.node[0].output, vec!["z"]);

        // All tensors are 1D with symbolic dim "N".
        for vi in graph.input.iter().chain(graph.output.iter()) {
            let tensor = match vi.r#type.as_ref().unwrap().value.as_ref().unwrap() {
                type_proto::Value::TensorType(t) => t,
            };
            let dims = &tensor.shape.as_ref().unwrap().dim;
            assert_eq!(dims.len(), 1);
            assert_eq!(
                dims[0].value,
                Some(tensor_shape_dimension::Value::DimParam("N".into()))
            );
        }
    }

    #[test]
    fn elementwise_ops() {
        for (op, expected) in [
            (ElementWiseOp::Sub, "Sub"),
            (ElementWiseOp::Mul, "Mul"),
            (ElementWiseOp::Div, "Div"),
        ] {
            let pattern = KernelPattern::ElementWise {
                op,
                inputs: [
                    make_tensor("a", TensorRole::Input),
                    make_tensor("b", TensorRole::Input),
                ],
                output: make_tensor("c", TensorRole::Output),
                dim_name: "N".into(),
            };
            let model = build_model(&pattern, "test");
            let graph = model.graph.as_ref().unwrap();
            assert_eq!(graph.node[0].op_type, expected);
        }
    }

    #[test]
    fn conv2d_model_structure() {
        let pattern = KernelPattern::Conv2D {
            input: make_tensor("input", TensorRole::Input),
            weight: make_tensor("weight", TensorRole::Input),
            output: make_tensor("output", TensorRole::Output),
            shape: Conv2DShape {
                batch: "N".into(),
                channels_in: "IC".into(),
                channels_out: "OC".into(),
                height: "H".into(),
                width: "W".into(),
                kernel_h: "KH".into(),
                kernel_w: "KW".into(),
                stride_h: 1,
                stride_w: 1,
                pad_h: 0,
                pad_w: 0,
            },
        };
        let model = build_model(&pattern, "conv2d");
        let graph = model.graph.as_ref().unwrap();
        assert_eq!(graph.node[0].op_type, "Conv");
    }

    #[test]
    fn pool_model_structure() {
        let pattern = KernelPattern::Pool {
            kind: PoolKind::Max,
            input: make_tensor("input", TensorRole::Input),
            output: make_tensor("output", TensorRole::Output),
            shape: PoolShape {
                kernel_h: 2,
                kernel_w: 2,
                stride_h: 2,
                stride_w: 2,
            },
        };
        let model = build_model(&pattern, "maxpool");
        let graph = model.graph.as_ref().unwrap();
        assert_eq!(graph.node[0].op_type, "MaxPool");
    }

    #[test]
    fn activation_model_structure() {
        for (op, expected) in [
            (ActivationOp::Relu, "Relu"),
            (ActivationOp::Sigmoid, "Sigmoid"),
            (ActivationOp::Tanh, "Tanh"),
        ] {
            let pattern = KernelPattern::Activation {
                op,
                input: make_tensor("x", TensorRole::Input),
                output: make_tensor("y", TensorRole::Output),
                dim_name: "N".into(),
            };
            let model = build_model(&pattern, "test");
            let graph = model.graph.as_ref().unwrap();
            assert_eq!(graph.node[0].op_type, expected);
        }
    }

    #[test]
    fn reduce_model_structure() {
        let pattern = KernelPattern::Reduce {
            op: ReduceOp::Sum,
            input: make_tensor("x", TensorRole::Input),
            output: make_tensor("y", TensorRole::Output),
            axis: 1,
        };
        let model = build_model(&pattern, "reduce");
        let graph = model.graph.as_ref().unwrap();
        assert_eq!(graph.node[0].op_type, "ReduceSum");
    }

    #[test]
    fn transpose_model_structure() {
        let pattern = KernelPattern::Transpose {
            input: make_tensor("x", TensorRole::Input),
            output: make_tensor("y", TensorRole::Output),
            perm: vec![1, 0],
        };
        let model = build_model(&pattern, "transpose");
        let graph = model.graph.as_ref().unwrap();
        assert_eq!(graph.node[0].op_type, "Transpose");
    }

    #[test]
    fn normalization_model_structure() {
        let pattern = KernelPattern::Normalization {
            input: make_tensor("x", TensorRole::Input),
            scale: make_tensor("gamma", TensorRole::Input),
            bias: make_tensor("beta", TensorRole::Input),
            output: make_tensor("y", TensorRole::Output),
            epsilon: 1e-5,
        };
        let model = build_model(&pattern, "batchnorm");
        let graph = model.graph.as_ref().unwrap();
        assert_eq!(graph.node[0].op_type, "BatchNormalization");
    }
}
