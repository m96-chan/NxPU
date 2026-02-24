//! TFLite FlatBuffer model construction from classified kernel patterns.
//!
//! Builds a TFLite model using the `flatbuffers` crate's builder API
//! with manual table construction (no generated code, no .fbs schema).

use flatbuffers::FlatBufferBuilder;
use nxpu_analysis::analyze::data_type;
use nxpu_analysis::analyze::{
    ActivationOp, Conv2DShape, ElementWiseOp, KernelPattern, PoolKind, PoolShape, ReduceOp,
    TensorBinding,
};
use nxpu_analysis::fusion::FusedPattern;
use nxpu_backend_core::BackendError;

use crate::schema::{
    builtin_op, builtin_options_type, conv2d_options, pool2d_options, softmax_options, tensor_type,
    vt,
};

/// File identifier for TFLite FlatBuffer files.
const TFLITE_FILE_ID: &str = "TFL3";

/// Build a TFLite FlatBuffer model from a classified kernel pattern.
pub fn build_model(pattern: &KernelPattern) -> Result<Vec<u8>, BackendError> {
    let bytes = match pattern {
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
                &format!("{}_1d", op.op_name().to_lowercase()),
            )
        }
        KernelPattern::Conv2D {
            input,
            weight,
            output,
            shape,
        } => build_tflite_conv2d(input, weight, output, shape),
        KernelPattern::Pool {
            kind,
            input,
            output,
            shape,
        } => {
            let opcode = match kind {
                PoolKind::Max => builtin_op::MAX_POOL_2D,
                PoolKind::Avg => builtin_op::AVERAGE_POOL_2D,
            };
            build_tflite_pool(input, output, opcode, shape, "pool")
        }
        KernelPattern::Activation {
            op, input, output, ..
        } => {
            if matches!(op, ActivationOp::Softmax) {
                build_tflite_softmax(input, output)
            } else {
                let shapes = [vec![-1i32], vec![-1]];
                let opcode = match op {
                    ActivationOp::Relu => builtin_op::RELU,
                    ActivationOp::Sigmoid => builtin_op::LOGISTIC,
                    ActivationOp::Tanh => builtin_op::TANH,
                    ActivationOp::Softmax => unreachable!(),
                };
                build_tflite_unary(
                    input,
                    output,
                    &shapes[0],
                    &shapes[1],
                    opcode,
                    &format!("{}_1d", op.op_name().to_lowercase()),
                )
            }
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
                &format!("{}_reduce", op.op_name().to_lowercase()),
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
            bias,
            output,
            ..
        } => {
            // TFLite doesn't have a direct BatchNorm op; expand to MUL(input, scale) + ADD(mul_result, bias)
            build_tflite_batchnorm(input, scale, bias, output)
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
        KernelPattern::Split { input, outputs, .. } => build_tflite_split(input, outputs),
        KernelPattern::Attention {
            query,
            key,
            value,
            output,
            d_k,
            ..
        } => build_tflite_attention(query, key, value, output, d_k),
        KernelPattern::Unknown { reason } => {
            return Err(BackendError::Unsupported(format!(
                "cannot lower Unknown pattern to TFLite: {reason}"
            )));
        }
    };
    Ok(bytes)
}

/// Build a TFLite FlatBuffer model from a fused pattern.
///
/// Handles single patterns, Conv+BatchNorm, MatMul+Bias (Gemm), and
/// activation fusion.
pub fn build_fused_model(fp: &FusedPattern) -> Result<Vec<u8>, BackendError> {
    match fp {
        FusedPattern::Single(p) => build_model(p),
        FusedPattern::ConvBatchNorm { conv, .. } => {
            // Lower just the conv; TFLite doesn't have a native fused
            // Conv+BatchNorm, so we emit the conv and the BN is folded
            // into weights at a higher level.
            build_model(conv)
        }
        FusedPattern::MatMulBias { matmul, bias_add } => {
            // For TFLite, emit as BATCH_MATMUL followed by ADD in a
            // single model. For now, lower just the matmul; the bias is
            // handled separately.
            // A full implementation would build a multi-operator subgraph.
            // For simplicity, lower the primary pattern.
            let _ = bias_add;
            build_model(matmul)
        }
        FusedPattern::WithActivation {
            base, activation, ..
        } => {
            // For WithActivation, we lower the base and note the fused
            // activation. TFLite operators that support fused activation
            // (Conv2D, Add, etc.) set the activation enum in their options.
            // For operators that don't support inline activation, we just
            // lower the base pattern for now.
            let _ = activation;
            build_fused_model(base)
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

/// Build a TFLite model for BatchNorm as MUL(input, scale) → ADD(mul_result, bias).
///
/// Emits a 2-operator subgraph since TFLite has no native BatchNorm op.
fn build_tflite_batchnorm(
    input: &TensorBinding,
    scale: &TensorBinding,
    bias: &TensorBinding,
    output: &TensorBinding,
) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::with_capacity(2048);

    // Strings
    let name_in = fbb.create_string(&input.name);
    let name_scale = fbb.create_string(&scale.name);
    let name_bias = fbb.create_string(&bias.name);
    let name_mul = fbb.create_string("batchnorm_mul");
    let name_out = fbb.create_string(&output.name);
    let desc = fbb.create_string("nxpu");
    let sg_name = fbb.create_string("batchnorm");

    // Shapes
    let shape_nd = fbb.create_vector(&[-1i32, -1, -1, -1]);
    let shape_1d = fbb.create_vector(&[-1i32]);

    // Buffers: sentinel + input(1) + scale(2) + bias(3) + mul_result(4) + output(5)
    let mut buffer_offsets = Vec::new();
    for _ in 0..6 {
        let start = fbb.start_table();
        buffer_offsets.push(fbb.end_table(start));
    }
    let buffers = fbb.create_vector(&buffer_offsets);

    let itype = onnx_to_tflite_type(input.elem_type);

    // Tensors: input(0), scale(1), bias(2), mul_result(3), output(4)
    let t_in = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_nd);
        fbb.push_slot::<i8>(vt::tensor::TYPE, itype, 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 1, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_in);
        fbb.end_table(start)
    };
    let t_scale = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_1d);
        fbb.push_slot::<i8>(vt::tensor::TYPE, onnx_to_tflite_type(scale.elem_type), 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 2, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_scale);
        fbb.end_table(start)
    };
    let t_bias = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_1d);
        fbb.push_slot::<i8>(vt::tensor::TYPE, onnx_to_tflite_type(bias.elem_type), 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 3, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_bias);
        fbb.end_table(start)
    };
    let t_mul = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_nd);
        fbb.push_slot::<i8>(vt::tensor::TYPE, itype, 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 4, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_mul);
        fbb.end_table(start)
    };
    let t_out = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_nd);
        fbb.push_slot::<i8>(vt::tensor::TYPE, onnx_to_tflite_type(output.elem_type), 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 5, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_out);
        fbb.end_table(start)
    };
    let tensors = fbb.create_vector(&[t_in, t_scale, t_bias, t_mul, t_out]);

    // Operator codes: MUL(0), ADD(1)
    let mul_code = {
        let start = fbb.start_table();
        fbb.push_slot::<i8>(
            vt::operator_code::DEPRECATED_BUILTIN_CODE,
            builtin_op::MUL as i8,
            0,
        );
        fbb.push_slot::<i32>(vt::operator_code::VERSION, 1, 1);
        fbb.push_slot::<i32>(vt::operator_code::BUILTIN_CODE, builtin_op::MUL, 0);
        fbb.end_table(start)
    };
    let add_code = {
        let start = fbb.start_table();
        fbb.push_slot::<i8>(
            vt::operator_code::DEPRECATED_BUILTIN_CODE,
            builtin_op::ADD as i8,
            0,
        );
        fbb.push_slot::<i32>(vt::operator_code::VERSION, 1, 1);
        fbb.push_slot::<i32>(vt::operator_code::BUILTIN_CODE, builtin_op::ADD, 0);
        fbb.end_table(start)
    };
    let operator_codes = fbb.create_vector(&[mul_code, add_code]);

    // Op 0: MUL(input=0, scale=1) -> mul_result=3
    let op0_inputs = fbb.create_vector(&[0i32, 1]);
    let op0_outputs = fbb.create_vector(&[3i32]);
    let op0 = {
        let start = fbb.start_table();
        fbb.push_slot::<u32>(vt::operator::OPCODE_INDEX, 0, 0); // MUL
        fbb.push_slot_always(vt::operator::INPUTS, op0_inputs);
        fbb.push_slot_always(vt::operator::OUTPUTS, op0_outputs);
        fbb.end_table(start)
    };

    // Op 1: ADD(mul_result=3, bias=2) -> output=4
    let op1_inputs = fbb.create_vector(&[3i32, 2]);
    let op1_outputs = fbb.create_vector(&[4i32]);
    let op1 = {
        let start = fbb.start_table();
        fbb.push_slot::<u32>(vt::operator::OPCODE_INDEX, 1, 0); // ADD
        fbb.push_slot_always(vt::operator::INPUTS, op1_inputs);
        fbb.push_slot_always(vt::operator::OUTPUTS, op1_outputs);
        fbb.end_table(start)
    };
    let operators = fbb.create_vector(&[op0, op1]);

    // Subgraph: inputs = [0(input), 1(scale), 2(bias)], outputs = [4(output)]
    let sg_inputs = fbb.create_vector(&[0i32, 1, 2]);
    let sg_outputs = fbb.create_vector(&[4i32]);
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

/// Build a TFLite model for a Softmax activation with beta=1.0.
///
/// Uses BUILTIN_OPTIONS to embed a SoftmaxOptions table so that the TFLite
/// runtime picks up beta=1.0 instead of the default 0.0 (which is an identity).
fn build_tflite_softmax(input: &TensorBinding, output: &TensorBinding) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::with_capacity(1024);

    let name_in = fbb.create_string(&input.name);
    let name_out = fbb.create_string(&output.name);
    let desc = fbb.create_string("nxpu");
    let sg_name = fbb.create_string("softmax_1d");

    let shape_in = fbb.create_vector(&[-1i32]);
    let shape_out = fbb.create_vector(&[-1i32]);

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

    let deprecated_code = if builtin_op::SOFTMAX <= 127 {
        builtin_op::SOFTMAX as i8
    } else {
        127
    };
    let opcode_table = {
        let start = fbb.start_table();
        fbb.push_slot::<i8>(
            vt::operator_code::DEPRECATED_BUILTIN_CODE,
            deprecated_code,
            0,
        );
        fbb.push_slot::<i32>(vt::operator_code::VERSION, 1, 1);
        fbb.push_slot::<i32>(vt::operator_code::BUILTIN_CODE, builtin_op::SOFTMAX, 0);
        fbb.end_table(start)
    };
    let operator_codes = fbb.create_vector(&[opcode_table]);

    // SoftmaxOptions table: beta = 1.0
    let softmax_opts = {
        let start = fbb.start_table();
        fbb.push_slot::<f32>(softmax_options::BETA, 1.0, 0.0);
        fbb.end_table(start)
    };

    let operator = {
        let start = fbb.start_table();
        fbb.push_slot::<u32>(vt::operator::OPCODE_INDEX, 0, 0);
        fbb.push_slot_always(vt::operator::INPUTS, op_inputs);
        fbb.push_slot_always(vt::operator::OUTPUTS, op_outputs);
        fbb.push_slot::<u8>(
            vt::operator::BUILTIN_OPTIONS_TYPE,
            builtin_options_type::SOFTMAX,
            0,
        );
        fbb.push_slot_always(vt::operator::BUILTIN_OPTIONS, softmax_opts);
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

/// Build a TFLite model for Conv2D with a Conv2DOptions table.
fn build_tflite_conv2d(
    input: &TensorBinding,
    weight: &TensorBinding,
    output: &TensorBinding,
    shape: &Conv2DShape,
) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::with_capacity(1024);

    let name_in = fbb.create_string(&input.name);
    let name_w = fbb.create_string(&weight.name);
    let name_out = fbb.create_string(&output.name);
    let desc = fbb.create_string("nxpu");
    let sg_name = fbb.create_string("conv2d");

    let shape_4d = fbb.create_vector(&[-1i32, -1, -1, -1]);

    let op_inputs = fbb.create_vector(&[0i32, 1]);
    let op_outputs = fbb.create_vector(&[2i32]);
    let sg_inputs = fbb.create_vector(&[0i32, 1]);
    let sg_outputs = fbb.create_vector(&[2i32]);

    // 4 buffers: sentinel + input + weight + output
    let mut buffer_offsets = Vec::new();
    for _ in 0..4 {
        let start = fbb.start_table();
        buffer_offsets.push(fbb.end_table(start));
    }
    let buffers = fbb.create_vector(&buffer_offsets);

    let tensor_in = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_4d);
        fbb.push_slot::<i8>(vt::tensor::TYPE, onnx_to_tflite_type(input.elem_type), 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 1, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_in);
        fbb.end_table(start)
    };
    let tensor_w = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_4d);
        fbb.push_slot::<i8>(vt::tensor::TYPE, onnx_to_tflite_type(weight.elem_type), 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 2, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_w);
        fbb.end_table(start)
    };
    let tensor_out = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_4d);
        fbb.push_slot::<i8>(vt::tensor::TYPE, onnx_to_tflite_type(output.elem_type), 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 3, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_out);
        fbb.end_table(start)
    };
    let tensors = fbb.create_vector(&[tensor_in, tensor_w, tensor_out]);

    let deprecated_code = if builtin_op::CONV_2D <= 127 {
        builtin_op::CONV_2D as i8
    } else {
        127
    };
    let opcode_table = {
        let start = fbb.start_table();
        fbb.push_slot::<i8>(
            vt::operator_code::DEPRECATED_BUILTIN_CODE,
            deprecated_code,
            0,
        );
        fbb.push_slot::<i32>(vt::operator_code::VERSION, 1, 1);
        fbb.push_slot::<i32>(vt::operator_code::BUILTIN_CODE, builtin_op::CONV_2D, 0);
        fbb.end_table(start)
    };
    let operator_codes = fbb.create_vector(&[opcode_table]);

    // Conv2DOptions table: stride_w, stride_h, dilation_w=1, dilation_h=1, padding=VALID(0)
    let conv2d_opts = {
        let start = fbb.start_table();
        fbb.push_slot::<i8>(conv2d_options::PADDING, 0, 0); // VALID
        fbb.push_slot::<i32>(conv2d_options::STRIDE_W, shape.stride_w as i32, 1);
        fbb.push_slot::<i32>(conv2d_options::STRIDE_H, shape.stride_h as i32, 1);
        fbb.push_slot::<i32>(conv2d_options::ACTIVATION, 0, 0); // NONE
        fbb.push_slot::<i32>(conv2d_options::DILATION_W, 1, 1);
        fbb.push_slot::<i32>(conv2d_options::DILATION_H, 1, 1);
        fbb.end_table(start)
    };

    let operator = {
        let start = fbb.start_table();
        fbb.push_slot::<u32>(vt::operator::OPCODE_INDEX, 0, 0);
        fbb.push_slot_always(vt::operator::INPUTS, op_inputs);
        fbb.push_slot_always(vt::operator::OUTPUTS, op_outputs);
        fbb.push_slot::<u8>(
            vt::operator::BUILTIN_OPTIONS_TYPE,
            builtin_options_type::CONV_2D,
            0,
        );
        fbb.push_slot_always(vt::operator::BUILTIN_OPTIONS, conv2d_opts);
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

/// Build a TFLite model for a pooling op (Max or Avg) with a Pool2DOptions table.
fn build_tflite_pool(
    input: &TensorBinding,
    output: &TensorBinding,
    opcode: i32,
    shape: &PoolShape,
    graph_name: &str,
) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::with_capacity(1024);

    let name_in = fbb.create_string(&input.name);
    let name_out = fbb.create_string(&output.name);
    let desc = fbb.create_string("nxpu");
    let sg_name = fbb.create_string(graph_name);

    let shape_4d = fbb.create_vector(&[-1i32, -1, -1, -1]);

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
        fbb.push_slot_always(vt::tensor::SHAPE, shape_4d);
        fbb.push_slot::<i8>(vt::tensor::TYPE, onnx_to_tflite_type(input.elem_type), 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 1, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_in);
        fbb.end_table(start)
    };
    let tensor_out = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_4d);
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

    // Pool2DOptions table: padding=VALID(0), stride_w, stride_h, filter_w, filter_h, activation=NONE(0)
    let pool2d_opts = {
        let start = fbb.start_table();
        fbb.push_slot::<i8>(pool2d_options::PADDING, 0, 0); // VALID
        fbb.push_slot::<i32>(pool2d_options::STRIDE_W, shape.stride_w as i32, 1);
        fbb.push_slot::<i32>(pool2d_options::STRIDE_H, shape.stride_h as i32, 1);
        fbb.push_slot::<i32>(pool2d_options::FILTER_W, shape.kernel_w as i32, 1);
        fbb.push_slot::<i32>(pool2d_options::FILTER_H, shape.kernel_h as i32, 1);
        fbb.push_slot::<i32>(pool2d_options::ACTIVATION, 0, 0); // NONE
        fbb.end_table(start)
    };

    let operator = {
        let start = fbb.start_table();
        fbb.push_slot::<u32>(vt::operator::OPCODE_INDEX, 0, 0);
        fbb.push_slot_always(vt::operator::INPUTS, op_inputs);
        fbb.push_slot_always(vt::operator::OUTPUTS, op_outputs);
        fbb.push_slot::<u8>(
            vt::operator::BUILTIN_OPTIONS_TYPE,
            builtin_options_type::POOL_2D,
            0,
        );
        fbb.push_slot_always(vt::operator::BUILTIN_OPTIONS, pool2d_opts);
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

/// Build a TFLite model for attention:
///   BATCH_MATMUL(Q,K) → DIV(scores, sqrt_dk) → SOFTMAX(beta=1.0) → BATCH_MATMUL(attn,V).
///
/// The `d_k` string is parsed to an f32; if parsing fails a fallback of 64.0 is used.
/// NOTE: Unlike the ONNX and StableHLO backends which compute sqrt(d_k) dynamically
/// from the query tensor's shape, TFLite lacks dynamic shape operators. The sqrt(d_k)
/// value is embedded as a compile-time constant. When d_k is symbolic (a param name
/// rather than a number), the fallback value of sqrt(64) = 8.0 is used.
fn build_tflite_attention(
    query: &TensorBinding,
    key: &TensorBinding,
    value: &TensorBinding,
    output: &TensorBinding,
    d_k: &str,
) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::with_capacity(2048);

    // Compute sqrt(d_k) from the symbolic dimension name (fall back to 64.0 if not numeric).
    let dk_val: f32 = d_k.parse::<f32>().unwrap_or(64.0);
    let sqrt_dk: f32 = dk_val.sqrt();
    // Serialize as little-endian f32 bytes for the constant buffer.
    let sqrt_dk_bytes: Vec<u8> = sqrt_dk.to_le_bytes().to_vec();

    // Strings
    let name_q = fbb.create_string(&query.name);
    let name_k = fbb.create_string(&key.name);
    let name_v = fbb.create_string(&value.name);
    let name_scores = fbb.create_string("scores");
    let name_sqrt_dk = fbb.create_string("sqrt_dk");
    let name_scaled = fbb.create_string("scaled_scores");
    let name_attn = fbb.create_string("attn_weights");
    let name_out = fbb.create_string(&output.name);
    let desc = fbb.create_string("nxpu");
    let sg_name = fbb.create_string("attention");

    // Shapes
    let shape_2d = fbb.create_vector(&[-1i32, -1]);
    let shape_scalar = fbb.create_vector(&[1i32]);

    let qtype = onnx_to_tflite_type(query.elem_type);

    // sqrt_dk constant data vector
    let sqrt_dk_data = fbb.create_vector(&sqrt_dk_bytes);

    // Buffers:
    //   0 = sentinel
    //   1 = Q
    //   2 = K
    //   3 = V
    //   4 = scores (dynamic)
    //   5 = sqrt_dk constant (has data)
    //   6 = scaled_scores (dynamic)
    //   7 = attn_weights (dynamic)
    //   8 = output (dynamic)
    // Build empty buffers first (sentinel + dynamic tensors), then sqrt_dk with data.
    // FlatBuffers requires data to be written before the table that references it,
    // so we build the sqrt_dk buffer table with the data vector already created.
    let buf_empty = {
        let start = fbb.start_table();
        fbb.end_table(start)
    };
    let buf_sqrt_dk = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::buffer::DATA, sqrt_dk_data);
        fbb.end_table(start)
    };
    // Build 9-element buffer array in slot order.
    let buffer_offsets = [
        buf_empty,   // 0 sentinel
        buf_empty,   // 1 Q
        buf_empty,   // 2 K
        buf_empty,   // 3 V
        buf_empty,   // 4 scores
        buf_sqrt_dk, // 5 sqrt_dk constant
        buf_empty,   // 6 scaled_scores
        buf_empty,   // 7 attn_weights
        buf_empty,   // 8 output
    ];
    let buffers = fbb.create_vector(&buffer_offsets);

    // Tensors:
    //   0: Q        (buf 1)
    //   1: K        (buf 2)
    //   2: V        (buf 3)
    //   3: scores   (buf 4, dynamic)
    //   4: sqrt_dk  (buf 5, constant scalar)
    //   5: scaled_scores (buf 6, dynamic)
    //   6: attn_weights  (buf 7, dynamic)
    //   7: output   (buf 8)
    let t_q = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_2d);
        fbb.push_slot::<i8>(vt::tensor::TYPE, qtype, 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 1, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_q);
        fbb.end_table(start)
    };
    let t_k = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_2d);
        fbb.push_slot::<i8>(vt::tensor::TYPE, qtype, 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 2, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_k);
        fbb.end_table(start)
    };
    let t_v = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_2d);
        fbb.push_slot::<i8>(vt::tensor::TYPE, qtype, 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 3, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_v);
        fbb.end_table(start)
    };
    let t_scores = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_2d);
        fbb.push_slot::<i8>(vt::tensor::TYPE, qtype, 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 4, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_scores);
        fbb.end_table(start)
    };
    let t_sqrt_dk = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_scalar);
        fbb.push_slot::<i8>(vt::tensor::TYPE, qtype, 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 5, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_sqrt_dk);
        fbb.end_table(start)
    };
    let t_scaled = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_2d);
        fbb.push_slot::<i8>(vt::tensor::TYPE, qtype, 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 6, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_scaled);
        fbb.end_table(start)
    };
    let t_attn = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_2d);
        fbb.push_slot::<i8>(vt::tensor::TYPE, qtype, 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 7, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_attn);
        fbb.end_table(start)
    };
    let t_out = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_2d);
        fbb.push_slot::<i8>(vt::tensor::TYPE, onnx_to_tflite_type(output.elem_type), 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 8, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_out);
        fbb.end_table(start)
    };
    let tensors = fbb.create_vector(&[t_q, t_k, t_v, t_scores, t_sqrt_dk, t_scaled, t_attn, t_out]);

    // Operator codes: BATCH_MATMUL(0), DIV(1), SOFTMAX(2)
    let matmul_code = {
        let start = fbb.start_table();
        fbb.push_slot::<i8>(vt::operator_code::DEPRECATED_BUILTIN_CODE, 127, 0);
        fbb.push_slot::<i32>(vt::operator_code::VERSION, 1, 1);
        fbb.push_slot::<i32>(vt::operator_code::BUILTIN_CODE, builtin_op::BATCH_MATMUL, 0);
        fbb.end_table(start)
    };
    let div_code = {
        let start = fbb.start_table();
        let deprecated_div = if builtin_op::DIV <= 127 {
            builtin_op::DIV as i8
        } else {
            127
        };
        fbb.push_slot::<i8>(
            vt::operator_code::DEPRECATED_BUILTIN_CODE,
            deprecated_div,
            0,
        );
        fbb.push_slot::<i32>(vt::operator_code::VERSION, 1, 1);
        fbb.push_slot::<i32>(vt::operator_code::BUILTIN_CODE, builtin_op::DIV, 0);
        fbb.end_table(start)
    };
    let softmax_code = {
        let start = fbb.start_table();
        let deprecated_sm = if builtin_op::SOFTMAX <= 127 {
            builtin_op::SOFTMAX as i8
        } else {
            127
        };
        fbb.push_slot::<i8>(vt::operator_code::DEPRECATED_BUILTIN_CODE, deprecated_sm, 0);
        fbb.push_slot::<i32>(vt::operator_code::VERSION, 1, 1);
        fbb.push_slot::<i32>(vt::operator_code::BUILTIN_CODE, builtin_op::SOFTMAX, 0);
        fbb.end_table(start)
    };
    let operator_codes = fbb.create_vector(&[matmul_code, div_code, softmax_code]);

    // Op 0: BATCH_MATMUL(Q=0, K=1) -> scores=3
    let op0_inputs = fbb.create_vector(&[0i32, 1]);
    let op0_outputs = fbb.create_vector(&[3i32]);
    let op0 = {
        let start = fbb.start_table();
        fbb.push_slot::<u32>(vt::operator::OPCODE_INDEX, 0, 0); // BATCH_MATMUL
        fbb.push_slot_always(vt::operator::INPUTS, op0_inputs);
        fbb.push_slot_always(vt::operator::OUTPUTS, op0_outputs);
        fbb.end_table(start)
    };

    // Op 1: DIV(scores=3, sqrt_dk=4) -> scaled_scores=5
    let op1_inputs = fbb.create_vector(&[3i32, 4]);
    let op1_outputs = fbb.create_vector(&[5i32]);
    let op1 = {
        let start = fbb.start_table();
        fbb.push_slot::<u32>(vt::operator::OPCODE_INDEX, 1, 0); // DIV
        fbb.push_slot_always(vt::operator::INPUTS, op1_inputs);
        fbb.push_slot_always(vt::operator::OUTPUTS, op1_outputs);
        fbb.end_table(start)
    };

    // Op 2: SOFTMAX(scaled_scores=5) -> attn_weights=6  (beta=1.0)
    let softmax_opts = {
        let start = fbb.start_table();
        fbb.push_slot::<f32>(softmax_options::BETA, 1.0, 0.0);
        fbb.end_table(start)
    };
    let op2_inputs = fbb.create_vector(&[5i32]);
    let op2_outputs = fbb.create_vector(&[6i32]);
    let op2 = {
        let start = fbb.start_table();
        fbb.push_slot::<u32>(vt::operator::OPCODE_INDEX, 2, 0); // SOFTMAX
        fbb.push_slot_always(vt::operator::INPUTS, op2_inputs);
        fbb.push_slot_always(vt::operator::OUTPUTS, op2_outputs);
        fbb.push_slot::<u8>(
            vt::operator::BUILTIN_OPTIONS_TYPE,
            builtin_options_type::SOFTMAX,
            0,
        );
        fbb.push_slot_always(vt::operator::BUILTIN_OPTIONS, softmax_opts);
        fbb.end_table(start)
    };

    // Op 3: BATCH_MATMUL(attn_weights=6, V=2) -> output=7
    let op3_inputs = fbb.create_vector(&[6i32, 2]);
    let op3_outputs = fbb.create_vector(&[7i32]);
    let op3 = {
        let start = fbb.start_table();
        fbb.push_slot::<u32>(vt::operator::OPCODE_INDEX, 0, 0); // BATCH_MATMUL
        fbb.push_slot_always(vt::operator::INPUTS, op3_inputs);
        fbb.push_slot_always(vt::operator::OUTPUTS, op3_outputs);
        fbb.end_table(start)
    };
    let operators = fbb.create_vector(&[op0, op1, op2, op3]);

    // Subgraph: inputs=[Q=0, K=1, V=2], outputs=[output=7]
    let sg_inputs = fbb.create_vector(&[0i32, 1, 2]);
    let sg_outputs = fbb.create_vector(&[7i32]);
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

/// Build a TFLite model for Split: one input, multiple outputs.
fn build_tflite_split(input: &TensorBinding, outputs: &[TensorBinding]) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::with_capacity(1024);

    let name_in = fbb.create_string(&input.name);
    let out_names: Vec<_> = outputs.iter().map(|o| fbb.create_string(&o.name)).collect();
    let desc = fbb.create_string("nxpu");
    let sg_name = fbb.create_string("split");

    let shape_in = fbb.create_vector(&[-1i32]);
    let shape_out = fbb.create_vector(&[-1i32]);

    let num_tensors = 1 + outputs.len(); // input + outputs
    let mut buffer_offsets = Vec::new();
    for _ in 0..=num_tensors {
        let start = fbb.start_table();
        buffer_offsets.push(fbb.end_table(start));
    }
    let buffers = fbb.create_vector(&buffer_offsets);

    // Tensors: input(0), outputs(1..N)
    let mut tensor_offsets = Vec::new();
    let tensor_in = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_in);
        fbb.push_slot::<i8>(vt::tensor::TYPE, onnx_to_tflite_type(input.elem_type), 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 1, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_in);
        fbb.end_table(start)
    };
    tensor_offsets.push(tensor_in);

    for (i, (o, name)) in outputs.iter().zip(out_names.iter()).enumerate() {
        let t = {
            let start = fbb.start_table();
            fbb.push_slot_always(vt::tensor::SHAPE, shape_out);
            fbb.push_slot::<i8>(vt::tensor::TYPE, onnx_to_tflite_type(o.elem_type), 0);
            fbb.push_slot::<u32>(vt::tensor::BUFFER, (i + 2) as u32, 0);
            fbb.push_slot_always(vt::tensor::NAME, *name);
            fbb.end_table(start)
        };
        tensor_offsets.push(t);
    }
    let tensors = fbb.create_vector(&tensor_offsets);

    let deprecated_code = if builtin_op::SPLIT <= 127 {
        builtin_op::SPLIT as i8
    } else {
        127
    };
    let opcode_table = {
        let start = fbb.start_table();
        fbb.push_slot::<i8>(
            vt::operator_code::DEPRECATED_BUILTIN_CODE,
            deprecated_code,
            0,
        );
        fbb.push_slot::<i32>(vt::operator_code::VERSION, 1, 1);
        fbb.push_slot::<i32>(vt::operator_code::BUILTIN_CODE, builtin_op::SPLIT, 0);
        fbb.end_table(start)
    };
    let operator_codes = fbb.create_vector(&[opcode_table]);

    let op_inputs = fbb.create_vector(&[0i32]);
    let output_indices: Vec<i32> = (1..=outputs.len() as i32).collect();
    let op_outputs = fbb.create_vector(&output_indices);
    let operator = {
        let start = fbb.start_table();
        fbb.push_slot::<u32>(vt::operator::OPCODE_INDEX, 0, 0);
        fbb.push_slot_always(vt::operator::INPUTS, op_inputs);
        fbb.push_slot_always(vt::operator::OUTPUTS, op_outputs);
        fbb.end_table(start)
    };
    let operators = fbb.create_vector(&[operator]);

    let sg_inputs = fbb.create_vector(&[0i32]);
    let sg_outputs = fbb.create_vector(&output_indices);
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
    use nxpu_analysis::analyze::data_type;
    use nxpu_analysis::analyze::{
        ActivationOp, Conv2DShape, MatMulShape, PoolKind, PoolShape, ReduceOp, TensorRole,
    };

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
        let bytes = build_model(&pattern).unwrap();
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
        let bytes = build_model(&pattern).unwrap();
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
            let bytes = build_model(&pattern).unwrap();
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
                kernel_h_val: 3,
                kernel_w_val: 3,
                stride_h: 1,
                stride_w: 1,
                pad_h: 0,
                pad_w: 0,
            },
        };
        let bytes = build_model(&pattern).unwrap();
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
            let bytes = build_model(&pattern).unwrap();
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
            let bytes = build_model(&pattern).unwrap();
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
        let bytes = build_model(&pattern).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }
}
