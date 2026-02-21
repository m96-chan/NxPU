//! TFLite FlatBuffer model construction from classified kernel patterns.
//!
//! Builds a TFLite model using the `flatbuffers` crate's builder API
//! with manual table construction (no generated code, no .fbs schema).

use flatbuffers::FlatBufferBuilder;
use nxpu_backend_onnx::analyze::{ElementWiseOp, KernelPattern, TensorBinding};
use nxpu_backend_onnx::proto::data_type;

use crate::schema::{builtin_op, tensor_type, vt};

/// File identifier for TFLite FlatBuffer files.
const TFLITE_FILE_ID: &str = "TFL3";

/// Build a TFLite FlatBuffer model from a classified kernel pattern.
pub fn build_model(pattern: &KernelPattern) -> Vec<u8> {
    match pattern {
        KernelPattern::MatMul {
            inputs,
            output,
            shape,
        } => {
            let shapes = [
                vec![-1i32, -1], // A[M,K]
                vec![-1, -1],    // B[K,N]
                vec![-1, -1],    // C[M,N]
            ];
            build_tflite(
                &inputs[0],
                &inputs[1],
                output,
                &shapes,
                builtin_op::BATCH_MATMUL,
                &format!("matmul_{}x{}x{}", shape.m, shape.n, shape.k),
            )
        }
        KernelPattern::ElementWise {
            op, inputs, output, ..
        } => {
            let shapes = [vec![-1i32], vec![-1], vec![-1]];
            let opcode = match op {
                ElementWiseOp::Add => builtin_op::ADD,
                ElementWiseOp::Sub => builtin_op::SUB,
                ElementWiseOp::Mul => builtin_op::MUL,
                ElementWiseOp::Div => builtin_op::DIV,
            };
            build_tflite(
                &inputs[0],
                &inputs[1],
                output,
                &shapes,
                opcode,
                &format!("{}_1d", op.onnx_op_type().to_lowercase()),
            )
        }
    }
}

/// Convert ONNX data type to TFLite TensorType.
fn onnx_to_tflite_type(onnx_dt: i32) -> i8 {
    match onnx_dt {
        data_type::FLOAT => tensor_type::FLOAT32,
        data_type::FLOAT16 => tensor_type::FLOAT16,
        data_type::INT32 => tensor_type::INT32,
        data_type::UINT32 => tensor_type::UINT32,
        data_type::BOOL => tensor_type::BOOL,
        data_type::INT8 => tensor_type::INT8,
        _ => tensor_type::FLOAT32,
    }
}

fn build_tflite(
    input_a: &TensorBinding,
    input_b: &TensorBinding,
    output: &TensorBinding,
    shapes: &[Vec<i32>; 3],
    opcode: i32,
    graph_name: &str,
) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::with_capacity(1024);

    // --- Leaf objects (bottom-up) ---

    // Strings
    let name_a = fbb.create_string(&input_a.name);
    let name_b = fbb.create_string(&input_b.name);
    let name_c = fbb.create_string(&output.name);
    let desc = fbb.create_string("nxpu");
    let sg_name = fbb.create_string(graph_name);

    // Shape vectors
    let shape_a = fbb.create_vector(&shapes[0]);
    let shape_b = fbb.create_vector(&shapes[1]);
    let shape_c = fbb.create_vector(&shapes[2]);

    // Operator input/output index vectors
    let op_inputs = fbb.create_vector(&[0i32, 1]);
    let op_outputs = fbb.create_vector(&[2i32]);
    let sg_inputs = fbb.create_vector(&[0i32, 1]);
    let sg_outputs = fbb.create_vector(&[2i32]);

    // --- Tables ---

    // Buffers (4 empty: sentinel + 3 tensors)
    let mut buffer_offsets = Vec::new();
    for _ in 0..4 {
        let start = fbb.start_table();
        buffer_offsets.push(fbb.end_table(start));
    }
    let buffers = fbb.create_vector(&buffer_offsets);

    // Tensors
    let type_a = onnx_to_tflite_type(input_a.elem_type);
    let type_b = onnx_to_tflite_type(input_b.elem_type);
    let type_c = onnx_to_tflite_type(output.elem_type);

    let tensor_a = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_a);
        fbb.push_slot::<i8>(vt::tensor::TYPE, type_a, 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 1, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_a);
        fbb.end_table(start)
    };

    let tensor_b = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_b);
        fbb.push_slot::<i8>(vt::tensor::TYPE, type_b, 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 2, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_b);
        fbb.end_table(start)
    };

    let tensor_c = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_c);
        fbb.push_slot::<i8>(vt::tensor::TYPE, type_c, 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 3, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_c);
        fbb.end_table(start)
    };

    let tensors = fbb.create_vector(&[tensor_a, tensor_b, tensor_c]);

    // OperatorCode
    let deprecated_code = if opcode <= 127 { opcode as i8 } else { 127 };
    let opcode_table = {
        let start = fbb.start_table();
        fbb.push_slot::<i8>(
            vt::operator_code::DEPRECATED_BUILTIN_CODE,
            deprecated_code,
            0,
        );
        fbb.push_slot::<i32>(vt::operator_code::VERSION, 1, 1);
        fbb.push_slot::<i32>(vt::operator_code::BUILTIN_CODE, opcode, 0);
        fbb.end_table(start)
    };
    let operator_codes = fbb.create_vector(&[opcode_table]);

    // Operator
    let operator = {
        let start = fbb.start_table();
        fbb.push_slot::<u32>(vt::operator::OPCODE_INDEX, 0, 0);
        fbb.push_slot_always(vt::operator::INPUTS, op_inputs);
        fbb.push_slot_always(vt::operator::OUTPUTS, op_outputs);
        fbb.end_table(start)
    };
    let operators = fbb.create_vector(&[operator]);

    // SubGraph
    let subgraph = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::sub_graph::TENSORS, tensors);
        fbb.push_slot_always(vt::sub_graph::INPUTS, sg_inputs);
        fbb.push_slot_always(vt::sub_graph::OUTPUTS, sg_outputs);
        fbb.push_slot_always(vt::sub_graph::OPERATORS, operators);
        fbb.push_slot_always(vt::sub_graph::NAME, sg_name);
        fbb.end_table(start)
    };
    let subgraphs = fbb.create_vector(&[subgraph]);

    // Model (root table)
    let model = {
        let start = fbb.start_table();
        fbb.push_slot::<u32>(vt::model::VERSION, 3, 0);
        fbb.push_slot_always(vt::model::OPERATOR_CODES, operator_codes);
        fbb.push_slot_always(vt::model::SUBGRAPHS, subgraphs);
        fbb.push_slot_always(vt::model::DESCRIPTION, desc);
        fbb.push_slot_always(vt::model::BUFFERS, buffers);
        fbb.end_table(start)
    };

    fbb.finish(model, Some(TFLITE_FILE_ID));
    fbb.finished_data().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nxpu_backend_onnx::analyze::{MatMulShape, TensorRole};
    use nxpu_backend_onnx::proto::data_type;

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
    fn matmul_produces_valid_flatbuffer() {
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

        let bytes = build_model(&pattern);

        // TFLite files start with root offset (4 bytes) + file identifier "TFL3"
        assert!(bytes.len() > 8);
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn elementwise_add_produces_valid_flatbuffer() {
        let pattern = KernelPattern::ElementWise {
            op: ElementWiseOp::Add,
            inputs: [
                make_tensor("x", TensorRole::Input),
                make_tensor("y", TensorRole::Input),
            ],
            output: make_tensor("z", TensorRole::Output),
            dim_name: "N".into(),
        };

        let bytes = build_model(&pattern);
        assert!(bytes.len() > 8);
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn all_elementwise_ops() {
        for op in [
            ElementWiseOp::Add,
            ElementWiseOp::Sub,
            ElementWiseOp::Mul,
            ElementWiseOp::Div,
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
            let bytes = build_model(&pattern);
            assert_eq!(&bytes[4..8], b"TFL3", "failed for {:?}", op);
        }
    }
}
