//! ONNX graph construction from classified kernel patterns.
//!
//! Converts a [`KernelPattern`] into an ONNX [`ModelProto`] with the
//! appropriate graph topology (MatMul or element-wise op).

use crate::analyze::{ElementWiseOp, KernelPattern, MatMulShape, TensorBinding};
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
        node: vec![NodeProto {
            input: vec![a_name.clone(), b_name.clone()],
            output: vec![c_name.clone()],
            name: "matmul_0".into(),
            op_type: "MatMul".into(),
            domain: String::new(),
        }],
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
        node: vec![NodeProto {
            input: vec![a_name.clone(), b_name.clone()],
            output: vec![c_name.clone()],
            name: format!("{}_0", op.onnx_op_type().to_lowercase()),
            op_type: op.onnx_op_type().into(),
            domain: String::new(),
        }],
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
}
