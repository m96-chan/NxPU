//! TFLite FlatBuffer model construction from classified kernel patterns.
//!
//! Builds a TFLite model using the `flatbuffers` crate's builder API
//! with manual table construction (no generated code, no .fbs schema).

use flatbuffers::FlatBufferBuilder;
use nxpu_backend_onnx::analyze::{
    ActivationOp, ElementWiseOp, KernelPattern, PoolKind, ReduceOp, TensorBinding,
};
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
            let shapes = [vec![-1i32, -1], vec![-1, -1], vec![-1, -1]];
            build_tflite(
                &[&inputs[0], &inputs[1]],
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
                &[&inputs[0], &inputs[1]],
                output,
                &shapes,
                opcode,
                &format!("{}_1d", op.onnx_op_type().to_lowercase()),
            )
        }
        KernelPattern::Conv2D {
            input,
            weight,
            output,
            ..
        } => {
            let shapes = [
                vec![-1, -1, -1, -1],
                vec![-1, -1, -1, -1],
                vec![-1, -1, -1, -1],
            ];
            build_tflite(
                &[input, weight],
                output,
                &shapes,
                builtin_op::CONV_2D,
                "conv2d",
            )
        }
        KernelPattern::Pool {
            kind,
            input,
            output,
            ..
        } => {
            let shapes = [vec![-1, -1, -1, -1], vec![], vec![-1, -1, -1, -1]];
            let opcode = match kind {
                PoolKind::Max => builtin_op::MAX_POOL_2D,
                PoolKind::Avg => builtin_op::AVERAGE_POOL_2D,
            };
            build_tflite_unary(input, output, &shapes[0], &shapes[2], opcode, "pool")
        }
        KernelPattern::Activation {
            op, input, output, ..
        } => {
            let shapes = [vec![-1i32], vec![-1]];
            let opcode = match op {
                ActivationOp::Relu => builtin_op::RELU,
                ActivationOp::Sigmoid => builtin_op::LOGISTIC,
                ActivationOp::Tanh => builtin_op::TANH,
                ActivationOp::Softmax => builtin_op::SOFTMAX,
            };
            build_tflite_unary(
                input,
                output,
                &shapes[0],
                &shapes[1],
                opcode,
                &format!("{}_1d", op.onnx_op_type().to_lowercase()),
            )
        }
        KernelPattern::Reduce {
            op, input, output, ..
        } => {
            let shapes = [vec![-1, -1], vec![-1]];
            let opcode = match op {
                ReduceOp::Sum => builtin_op::SUM,
                ReduceOp::Mean => builtin_op::MEAN,
                ReduceOp::Max => builtin_op::REDUCE_MAX,
                ReduceOp::Min => builtin_op::REDUCE_MIN,
            };
            build_tflite_unary(
                input,
                output,
                &shapes[0],
                &shapes[1],
                opcode,
                &format!("{}_reduce", op.onnx_op_type().to_lowercase()),
            )
        }
        KernelPattern::Transpose { input, output, .. } => {
            let shapes = [vec![-1, -1], vec![-1, -1]];
            build_tflite_unary(
                input,
                output,
                &shapes[0],
                &shapes[1],
                builtin_op::TRANSPOSE,
                "transpose",
            )
        }
        KernelPattern::Reshape { input, output, .. } => {
            let shapes = [vec![-1i32], vec![-1]];
            build_tflite_unary(
                input,
                output,
                &shapes[0],
                &shapes[1],
                builtin_op::RESHAPE,
                "reshape",
            )
        }
        KernelPattern::Normalization {
            input,
            scale,
            output,
            ..
        } => {
            // TFLite doesn't have a direct BatchNorm op; approximate with MUL(input, scale)
            let shapes = [vec![-1, -1, -1, -1], vec![-1], vec![-1, -1, -1, -1]];
            build_tflite(
                &[input, scale],
                output,
                &shapes,
                builtin_op::MUL,
                "batchnorm_approx",
            )
        }
        KernelPattern::Concat { inputs, output, .. } => {
            let input_refs: Vec<&TensorBinding> = inputs.iter().collect();
            let shapes = [vec![-1i32], vec![-1], vec![-1]];
            build_tflite(
                &input_refs,
                output,
                &shapes,
                builtin_op::CONCATENATION,
                "concat",
            )
        }
        KernelPattern::Split { input, outputs, .. } => {
            let out = outputs.first().expect("split must have outputs");
            let shapes = [vec![-1i32], vec![-1]];
            build_tflite_unary(
                input,
                out,
                &shapes[0],
                &shapes[1],
                builtin_op::SPLIT,
                "split",
            )
        }
        KernelPattern::Attention {
            query, key, output, ..
        } => {
            // Approximate attention as BATCH_MATMUL (Q * K^T).
            let shapes = [vec![-1i32, -1], vec![-1, -1], vec![-1, -1]];
            build_tflite(
                &[query, key],
                output,
                &shapes,
                builtin_op::BATCH_MATMUL,
                "attention",
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

/// Build a TFLite model with N inputs and 1 output.
fn build_tflite(
    inputs: &[&TensorBinding],
    output: &TensorBinding,
    shapes: &[Vec<i32>; 3],
    opcode: i32,
    graph_name: &str,
) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::with_capacity(1024);

    // Strings
    let names: Vec<_> = inputs.iter().map(|i| fbb.create_string(&i.name)).collect();
    let name_out = fbb.create_string(&output.name);
    let desc = fbb.create_string("nxpu");
    let sg_name = fbb.create_string(graph_name);

    // Shape vectors
    let shape_vecs: Vec<_> = shapes.iter().map(|s| fbb.create_vector(s)).collect();

    // Operator input/output index vectors
    let input_indices: Vec<i32> = (0..inputs.len() as i32).collect();
    let op_inputs = fbb.create_vector(&input_indices);
    let op_outputs = fbb.create_vector(&[inputs.len() as i32]);
    let sg_inputs = fbb.create_vector(&input_indices);
    let sg_outputs = fbb.create_vector(&[inputs.len() as i32]);

    // Buffers (sentinel + tensors)
    let num_tensors = inputs.len() + 1;
    let mut buffer_offsets = Vec::new();
    for _ in 0..=num_tensors {
        let start = fbb.start_table();
        buffer_offsets.push(fbb.end_table(start));
    }
    let buffers = fbb.create_vector(&buffer_offsets);

    // Tensors
    let mut tensor_offsets = Vec::new();
    for (i, inp) in inputs.iter().enumerate() {
        let t = {
            let start = fbb.start_table();
            fbb.push_slot_always(vt::tensor::SHAPE, shape_vecs[i]);
            fbb.push_slot::<i8>(vt::tensor::TYPE, onnx_to_tflite_type(inp.elem_type), 0);
            fbb.push_slot::<u32>(vt::tensor::BUFFER, (i + 1) as u32, 0);
            fbb.push_slot_always(vt::tensor::NAME, names[i]);
            fbb.end_table(start)
        };
        tensor_offsets.push(t);
    }
    // Output tensor
    let out_tensor = {
        let start = fbb.start_table();
        fbb.push_slot_always(
            vt::tensor::SHAPE,
            shape_vecs[inputs.len().min(shapes.len() - 1)],
        );
        fbb.push_slot::<i8>(vt::tensor::TYPE, onnx_to_tflite_type(output.elem_type), 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, num_tensors as u32, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_out);
        fbb.end_table(start)
    };
    tensor_offsets.push(out_tensor);
    let tensors = fbb.create_vector(&tensor_offsets);

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

/// Build a TFLite model with a single input and single output.
fn build_tflite_unary(
    input: &TensorBinding,
    output: &TensorBinding,
    in_shape: &[i32],
    out_shape: &[i32],
    opcode: i32,
    graph_name: &str,
) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::with_capacity(1024);

    let name_in = fbb.create_string(&input.name);
    let name_out = fbb.create_string(&output.name);
    let desc = fbb.create_string("nxpu");
    let sg_name = fbb.create_string(graph_name);

    let shape_in = fbb.create_vector(in_shape);
    let shape_out = fbb.create_vector(out_shape);

    let op_inputs = fbb.create_vector(&[0i32]);
    let op_outputs = fbb.create_vector(&[1i32]);
    let sg_inputs = fbb.create_vector(&[0i32]);
    let sg_outputs = fbb.create_vector(&[1i32]);

    // 3 buffers: sentinel + input + output
    let mut buffer_offsets = Vec::new();
    for _ in 0..3 {
        let start = fbb.start_table();
        buffer_offsets.push(fbb.end_table(start));
    }
    let buffers = fbb.create_vector(&buffer_offsets);

    let tensor_in = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_in);
        fbb.push_slot::<i8>(vt::tensor::TYPE, onnx_to_tflite_type(input.elem_type), 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 1, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_in);
        fbb.end_table(start)
    };
    let tensor_out = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_out);
        fbb.push_slot::<i8>(vt::tensor::TYPE, onnx_to_tflite_type(output.elem_type), 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 2, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_out);
        fbb.end_table(start)
    };
    let tensors = fbb.create_vector(&[tensor_in, tensor_out]);

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

    let operator = {
        let start = fbb.start_table();
        fbb.push_slot::<u32>(vt::operator::OPCODE_INDEX, 0, 0);
        fbb.push_slot_always(vt::operator::INPUTS, op_inputs);
        fbb.push_slot_always(vt::operator::OUTPUTS, op_outputs);
        fbb.end_table(start)
    };
    let operators = fbb.create_vector(&[operator]);

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
    use nxpu_backend_onnx::analyze::{
        ActivationOp, Conv2DShape, MatMulShape, PoolKind, PoolShape, ReduceOp, TensorRole,
    };
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

    #[test]
    fn conv2d_produces_valid_flatbuffer() {
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
        let bytes = build_model(&pattern);
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn activation_produces_valid_flatbuffer() {
        for op in [
            ActivationOp::Relu,
            ActivationOp::Sigmoid,
            ActivationOp::Tanh,
        ] {
            let pattern = KernelPattern::Activation {
                op,
                input: make_tensor("x", TensorRole::Input),
                output: make_tensor("y", TensorRole::Output),
                dim_name: "N".into(),
            };
            let bytes = build_model(&pattern);
            assert_eq!(&bytes[4..8], b"TFL3", "failed for {:?}", op);
        }
    }

    #[test]
    fn pool_produces_valid_flatbuffer() {
        for kind in [PoolKind::Max, PoolKind::Avg] {
            let pattern = KernelPattern::Pool {
                kind,
                input: make_tensor("x", TensorRole::Input),
                output: make_tensor("y", TensorRole::Output),
                shape: PoolShape {
                    kernel_h: 2,
                    kernel_w: 2,
                    stride_h: 2,
                    stride_w: 2,
                },
            };
            let bytes = build_model(&pattern);
            assert_eq!(&bytes[4..8], b"TFL3", "failed for {:?}", kind);
        }
    }

    #[test]
    fn reduce_produces_valid_flatbuffer() {
        let pattern = KernelPattern::Reduce {
            op: ReduceOp::Sum,
            input: make_tensor("x", TensorRole::Input),
            output: make_tensor("y", TensorRole::Output),
            axis: 1,
        };
        let bytes = build_model(&pattern);
        assert_eq!(&bytes[4..8], b"TFL3");
    }
}
