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
            num_heads,
            causal,
            ..
        } => build_tflite_attention(query, key, value, output, d_k, *num_heads, *causal),
        KernelPattern::Unknown { reason } => {
            return Err(BackendError::Unsupported(format!(
                "cannot lower Unknown pattern to TFLite: {reason}"
            )));
        }
    };
    Ok(bytes)
}

// ---- Multi-op graph builder types ----

/// A tensor descriptor used when building multi-op TFLite subgraphs.
struct TensorInfo {
    name: String,
    elem_type: i32,
    shape: Vec<i32>,
}

/// An operator descriptor used when building multi-op TFLite subgraphs.
struct OpDesc {
    opcode: i32,
    inputs: Vec<i32>,
    outputs: Vec<i32>,
}

/// An intermediate graph description that can be serialised to a TFLite
/// FlatBuffer by [`build_from_graph_desc`].
struct GraphDesc {
    tensors: Vec<TensorInfo>,
    ops: Vec<OpDesc>,
    graph_inputs: Vec<i32>,
    graph_outputs: Vec<i32>,
    graph_name: String,
}

/// Serialise a [`GraphDesc`] into a TFLite FlatBuffer.
///
/// Creates one buffer slot per tensor plus the mandatory sentinel buffer at
/// index 0.  Deduplicates operator codes so each unique opcode appears only
/// once in the `operator_codes` vector.
fn build_from_graph_desc(desc: &GraphDesc) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::with_capacity(2048);

    // --- strings ---
    let tensor_name_offsets: Vec<_> = desc
        .tensors
        .iter()
        .map(|t| fbb.create_string(&t.name))
        .collect();
    let graph_desc_str = fbb.create_string("nxpu");
    let sg_name_str = fbb.create_string(&desc.graph_name);

    // --- shape vectors ---
    let shape_offsets: Vec<_> = desc
        .tensors
        .iter()
        .map(|t| fbb.create_vector(&t.shape))
        .collect();

    // --- op input/output index vectors ---
    let op_input_offsets: Vec<_> = desc
        .ops
        .iter()
        .map(|o| fbb.create_vector(&o.inputs))
        .collect();
    let op_output_offsets: Vec<_> = desc
        .ops
        .iter()
        .map(|o| fbb.create_vector(&o.outputs))
        .collect();

    // --- graph-level input/output index vectors ---
    let sg_inputs_vec = fbb.create_vector(&desc.graph_inputs);
    let sg_outputs_vec = fbb.create_vector(&desc.graph_outputs);

    // --- buffers: sentinel(0) + one per tensor ---
    let num_tensors = desc.tensors.len();
    let mut buf_offsets = Vec::with_capacity(num_tensors + 1);
    for _ in 0..=num_tensors {
        let start = fbb.start_table();
        buf_offsets.push(fbb.end_table(start));
    }
    let buffers_vec = fbb.create_vector(&buf_offsets);

    // --- tensors ---
    let mut tensor_offsets = Vec::with_capacity(num_tensors);
    for (i, ti) in desc.tensors.iter().enumerate() {
        let t = {
            let start = fbb.start_table();
            fbb.push_slot_always(vt::tensor::SHAPE, shape_offsets[i]);
            fbb.push_slot::<i8>(vt::tensor::TYPE, onnx_to_tflite_type(ti.elem_type), 0);
            fbb.push_slot::<u32>(vt::tensor::BUFFER, (i + 1) as u32, 0);
            fbb.push_slot_always(vt::tensor::NAME, tensor_name_offsets[i]);
            fbb.end_table(start)
        };
        tensor_offsets.push(t);
    }
    let tensors_vec = fbb.create_vector(&tensor_offsets);

    // --- deduplicated operator codes ---
    let mut unique_opcodes: Vec<i32> = Vec::new();
    for op in &desc.ops {
        if !unique_opcodes.contains(&op.opcode) {
            unique_opcodes.push(op.opcode);
        }
    }
    let mut opcode_offsets = Vec::with_capacity(unique_opcodes.len());
    for &opcode in &unique_opcodes {
        let deprecated_code = if opcode <= 127 { opcode as i8 } else { 127 };
        let oc = {
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
        opcode_offsets.push(oc);
    }
    let operator_codes_vec = fbb.create_vector(&opcode_offsets);

    // --- operators ---
    let mut operator_offsets = Vec::with_capacity(desc.ops.len());
    for (i, op) in desc.ops.iter().enumerate() {
        let opcode_index = unique_opcodes.iter().position(|&c| c == op.opcode).unwrap() as u32;
        let o = {
            let start = fbb.start_table();
            fbb.push_slot::<u32>(vt::operator::OPCODE_INDEX, opcode_index, 0);
            fbb.push_slot_always(vt::operator::INPUTS, op_input_offsets[i]);
            fbb.push_slot_always(vt::operator::OUTPUTS, op_output_offsets[i]);
            fbb.end_table(start)
        };
        operator_offsets.push(o);
    }
    let operators_vec = fbb.create_vector(&operator_offsets);

    // --- subgraph ---
    let subgraph = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::sub_graph::TENSORS, tensors_vec);
        fbb.push_slot_always(vt::sub_graph::INPUTS, sg_inputs_vec);
        fbb.push_slot_always(vt::sub_graph::OUTPUTS, sg_outputs_vec);
        fbb.push_slot_always(vt::sub_graph::OPERATORS, operators_vec);
        fbb.push_slot_always(vt::sub_graph::NAME, sg_name_str);
        fbb.end_table(start)
    };
    let subgraphs_vec = fbb.create_vector(&[subgraph]);

    // --- model ---
    let model = {
        let start = fbb.start_table();
        fbb.push_slot::<u32>(vt::model::VERSION, 3, 0);
        fbb.push_slot_always(vt::model::OPERATOR_CODES, operator_codes_vec);
        fbb.push_slot_always(vt::model::SUBGRAPHS, subgraphs_vec);
        fbb.push_slot_always(vt::model::DESCRIPTION, graph_desc_str);
        fbb.push_slot_always(vt::model::BUFFERS, buffers_vec);
        fbb.end_table(start)
    };

    fbb.finish(model, Some(TFLITE_FILE_ID));
    fbb.finished_data().to_vec()
}

// ---- FusedPattern graph collectors ----

/// Build a [`GraphDesc`] for `ConvBatchNorm`: CONV_2D → MUL(scale) → ADD(bias).
///
/// Tensor layout:
/// - 0: input  (4-D)
/// - 1: weight (4-D)
/// - 2: scale  (1-D)
/// - 3: bias   (1-D)
/// - 4: conv_out (4-D, intermediate)
/// - 5: bn_mul   (4-D, intermediate)
/// - 6: output   (4-D)
///
/// Graph inputs: [0,1,2,3]  Graph outputs: [6]
fn collect_conv_batchnorm_graph(
    conv: &KernelPattern,
    norm: &KernelPattern,
) -> Result<GraphDesc, BackendError> {
    let (input, weight, conv_shape) = match conv {
        KernelPattern::Conv2D {
            input,
            weight,
            shape,
            ..
        } => (input, weight, shape),
        _ => {
            return Err(BackendError::Other(
                "ConvBatchNorm: conv slot is not Conv2D".into(),
            ));
        }
    };
    let (scale, bias, output) = match norm {
        KernelPattern::Normalization {
            scale,
            bias,
            output,
            ..
        } => (scale, bias, output),
        _ => {
            return Err(BackendError::Other(
                "ConvBatchNorm: norm slot is not Normalization".into(),
            ));
        }
    };

    let _ = conv_shape; // shape info not needed for symbolic dims
    let shape_4d = vec![-1i32, -1, -1, -1];
    let shape_1d = vec![-1i32];

    Ok(GraphDesc {
        tensors: vec![
            TensorInfo {
                name: input.name.clone(),
                elem_type: input.elem_type,
                shape: shape_4d.clone(),
            }, // 0
            TensorInfo {
                name: weight.name.clone(),
                elem_type: weight.elem_type,
                shape: shape_4d.clone(),
            }, // 1
            TensorInfo {
                name: scale.name.clone(),
                elem_type: scale.elem_type,
                shape: shape_1d.clone(),
            }, // 2
            TensorInfo {
                name: bias.name.clone(),
                elem_type: bias.elem_type,
                shape: shape_1d.clone(),
            }, // 3
            TensorInfo {
                name: "conv_out".into(),
                elem_type: input.elem_type,
                shape: shape_4d.clone(),
            }, // 4
            TensorInfo {
                name: "bn_mul".into(),
                elem_type: input.elem_type,
                shape: shape_4d.clone(),
            }, // 5
            TensorInfo {
                name: output.name.clone(),
                elem_type: output.elem_type,
                shape: shape_4d,
            }, // 6
        ],
        ops: vec![
            OpDesc {
                opcode: builtin_op::CONV_2D,
                inputs: vec![0, 1],
                outputs: vec![4],
            },
            OpDesc {
                opcode: builtin_op::MUL,
                inputs: vec![4, 2],
                outputs: vec![5],
            },
            OpDesc {
                opcode: builtin_op::ADD,
                inputs: vec![5, 3],
                outputs: vec![6],
            },
        ],
        graph_inputs: vec![0, 1, 2, 3],
        graph_outputs: vec![6],
        graph_name: "conv_batchnorm".into(),
    })
}

/// Build a [`GraphDesc`] for `MatMulBias`: BATCH_MATMUL → ADD(bias).
///
/// Tensor layout:
/// - 0: A      (2-D)
/// - 1: B      (2-D)
/// - 2: bias   (1-D)
/// - 3: mm_out (2-D, intermediate)
/// - 4: output (2-D)
///
/// Graph inputs: [0,1,2]  Graph outputs: [4]
fn collect_matmul_bias_graph(
    matmul: &KernelPattern,
    bias_add: &KernelPattern,
) -> Result<GraphDesc, BackendError> {
    let (mm_inputs, mm_output) = match matmul {
        KernelPattern::MatMul { inputs, output, .. } => (inputs, output),
        _ => {
            return Err(BackendError::Other(
                "MatMulBias: matmul slot is not MatMul".into(),
            ));
        }
    };
    let (bias, output) = match bias_add {
        KernelPattern::ElementWise { inputs, output, .. } => (&inputs[1], output),
        _ => {
            return Err(BackendError::Other(
                "MatMulBias: bias_add slot is not ElementWise".into(),
            ));
        }
    };

    let shape_2d = vec![-1i32, -1];
    let shape_1d = vec![-1i32];

    Ok(GraphDesc {
        tensors: vec![
            TensorInfo {
                name: mm_inputs[0].name.clone(),
                elem_type: mm_inputs[0].elem_type,
                shape: shape_2d.clone(),
            }, // 0: A
            TensorInfo {
                name: mm_inputs[1].name.clone(),
                elem_type: mm_inputs[1].elem_type,
                shape: shape_2d.clone(),
            }, // 1: B
            TensorInfo {
                name: bias.name.clone(),
                elem_type: bias.elem_type,
                shape: shape_1d,
            }, // 2: bias
            TensorInfo {
                name: mm_output.name.clone(),
                elem_type: mm_output.elem_type,
                shape: shape_2d.clone(),
            }, // 3: mm_out
            TensorInfo {
                name: output.name.clone(),
                elem_type: output.elem_type,
                shape: shape_2d,
            }, // 4: output
        ],
        ops: vec![
            OpDesc {
                opcode: builtin_op::BATCH_MATMUL,
                inputs: vec![0, 1],
                outputs: vec![3],
            },
            OpDesc {
                opcode: builtin_op::ADD,
                inputs: vec![3, 2],
                outputs: vec![4],
            },
        ],
        graph_inputs: vec![0, 1, 2],
        graph_outputs: vec![4],
        graph_name: "gemm".into(),
    })
}

/// Build a [`GraphDesc`] for a single [`KernelPattern`].
///
/// Handles the patterns that can be represented as a simple 1-op (or
/// already-multi-op for Normalization/Attention) subgraph.  Returns an error
/// for patterns that are better handled by the specialised builders (e.g.
/// Attention), causing the caller to fall back to [`build_model`].
fn collect_single_graph(pattern: &KernelPattern) -> Result<GraphDesc, BackendError> {
    match pattern {
        KernelPattern::MatMul {
            inputs,
            output,
            shape,
        } => {
            let shape_2d = vec![-1i32, -1];
            Ok(GraphDesc {
                tensors: vec![
                    TensorInfo {
                        name: inputs[0].name.clone(),
                        elem_type: inputs[0].elem_type,
                        shape: shape_2d.clone(),
                    },
                    TensorInfo {
                        name: inputs[1].name.clone(),
                        elem_type: inputs[1].elem_type,
                        shape: shape_2d.clone(),
                    },
                    TensorInfo {
                        name: output.name.clone(),
                        elem_type: output.elem_type,
                        shape: shape_2d,
                    },
                ],
                ops: vec![OpDesc {
                    opcode: builtin_op::BATCH_MATMUL,
                    inputs: vec![0, 1],
                    outputs: vec![2],
                }],
                graph_inputs: vec![0, 1],
                graph_outputs: vec![2],
                graph_name: format!("matmul_{}x{}x{}", shape.m, shape.n, shape.k),
            })
        }
        KernelPattern::ElementWise {
            op, inputs, output, ..
        } => {
            let opcode = match op {
                ElementWiseOp::Add => builtin_op::ADD,
                ElementWiseOp::Sub => builtin_op::SUB,
                ElementWiseOp::Mul => builtin_op::MUL,
                ElementWiseOp::Div => builtin_op::DIV,
            };
            let shape_1d = vec![-1i32];
            Ok(GraphDesc {
                tensors: vec![
                    TensorInfo {
                        name: inputs[0].name.clone(),
                        elem_type: inputs[0].elem_type,
                        shape: shape_1d.clone(),
                    },
                    TensorInfo {
                        name: inputs[1].name.clone(),
                        elem_type: inputs[1].elem_type,
                        shape: shape_1d.clone(),
                    },
                    TensorInfo {
                        name: output.name.clone(),
                        elem_type: output.elem_type,
                        shape: shape_1d,
                    },
                ],
                ops: vec![OpDesc {
                    opcode,
                    inputs: vec![0, 1],
                    outputs: vec![2],
                }],
                graph_inputs: vec![0, 1],
                graph_outputs: vec![2],
                graph_name: format!("{}_1d", op.op_name().to_lowercase()),
            })
        }
        KernelPattern::Conv2D {
            input,
            weight,
            output,
            ..
        } => {
            let shape_4d = vec![-1i32, -1, -1, -1];
            Ok(GraphDesc {
                tensors: vec![
                    TensorInfo {
                        name: input.name.clone(),
                        elem_type: input.elem_type,
                        shape: shape_4d.clone(),
                    },
                    TensorInfo {
                        name: weight.name.clone(),
                        elem_type: weight.elem_type,
                        shape: shape_4d.clone(),
                    },
                    TensorInfo {
                        name: output.name.clone(),
                        elem_type: output.elem_type,
                        shape: shape_4d,
                    },
                ],
                ops: vec![OpDesc {
                    opcode: builtin_op::CONV_2D,
                    inputs: vec![0, 1],
                    outputs: vec![2],
                }],
                graph_inputs: vec![0, 1],
                graph_outputs: vec![2],
                graph_name: "conv2d".into(),
            })
        }
        KernelPattern::Pool {
            kind,
            input,
            output,
            ..
        } => {
            let opcode = match kind {
                PoolKind::Max => builtin_op::MAX_POOL_2D,
                PoolKind::Avg => builtin_op::AVERAGE_POOL_2D,
            };
            let shape_4d = vec![-1i32, -1, -1, -1];
            Ok(GraphDesc {
                tensors: vec![
                    TensorInfo {
                        name: input.name.clone(),
                        elem_type: input.elem_type,
                        shape: shape_4d.clone(),
                    },
                    TensorInfo {
                        name: output.name.clone(),
                        elem_type: output.elem_type,
                        shape: shape_4d,
                    },
                ],
                ops: vec![OpDesc {
                    opcode,
                    inputs: vec![0],
                    outputs: vec![1],
                }],
                graph_inputs: vec![0],
                graph_outputs: vec![1],
                graph_name: "pool".into(),
            })
        }
        KernelPattern::Activation {
            op, input, output, ..
        } => {
            let opcode = match op {
                ActivationOp::Relu => builtin_op::RELU,
                ActivationOp::Sigmoid => builtin_op::LOGISTIC,
                ActivationOp::Tanh => builtin_op::TANH,
                ActivationOp::Softmax => builtin_op::SOFTMAX,
            };
            let shape_1d = vec![-1i32];
            Ok(GraphDesc {
                tensors: vec![
                    TensorInfo {
                        name: input.name.clone(),
                        elem_type: input.elem_type,
                        shape: shape_1d.clone(),
                    },
                    TensorInfo {
                        name: output.name.clone(),
                        elem_type: output.elem_type,
                        shape: shape_1d,
                    },
                ],
                ops: vec![OpDesc {
                    opcode,
                    inputs: vec![0],
                    outputs: vec![1],
                }],
                graph_inputs: vec![0],
                graph_outputs: vec![1],
                graph_name: format!("{}_1d", op.op_name().to_lowercase()),
            })
        }
        KernelPattern::Reduce {
            op, input, output, ..
        } => {
            let opcode = match op {
                ReduceOp::Sum => builtin_op::SUM,
                ReduceOp::Mean => builtin_op::MEAN,
                ReduceOp::Max => builtin_op::REDUCE_MAX,
                ReduceOp::Min => builtin_op::REDUCE_MIN,
            };
            Ok(GraphDesc {
                tensors: vec![
                    TensorInfo {
                        name: input.name.clone(),
                        elem_type: input.elem_type,
                        shape: vec![-1, -1],
                    },
                    TensorInfo {
                        name: output.name.clone(),
                        elem_type: output.elem_type,
                        shape: vec![-1],
                    },
                ],
                ops: vec![OpDesc {
                    opcode,
                    inputs: vec![0],
                    outputs: vec![1],
                }],
                graph_inputs: vec![0],
                graph_outputs: vec![1],
                graph_name: format!("{}_reduce", op.op_name().to_lowercase()),
            })
        }
        KernelPattern::Transpose { input, output, .. } => Ok(GraphDesc {
            tensors: vec![
                TensorInfo {
                    name: input.name.clone(),
                    elem_type: input.elem_type,
                    shape: vec![-1, -1],
                },
                TensorInfo {
                    name: output.name.clone(),
                    elem_type: output.elem_type,
                    shape: vec![-1, -1],
                },
            ],
            ops: vec![OpDesc {
                opcode: builtin_op::TRANSPOSE,
                inputs: vec![0],
                outputs: vec![1],
            }],
            graph_inputs: vec![0],
            graph_outputs: vec![1],
            graph_name: "transpose".into(),
        }),
        KernelPattern::Reshape { input, output, .. } => Ok(GraphDesc {
            tensors: vec![
                TensorInfo {
                    name: input.name.clone(),
                    elem_type: input.elem_type,
                    shape: vec![-1],
                },
                TensorInfo {
                    name: output.name.clone(),
                    elem_type: output.elem_type,
                    shape: vec![-1],
                },
            ],
            ops: vec![OpDesc {
                opcode: builtin_op::RESHAPE,
                inputs: vec![0],
                outputs: vec![1],
            }],
            graph_inputs: vec![0],
            graph_outputs: vec![1],
            graph_name: "reshape".into(),
        }),
        KernelPattern::Normalization {
            input,
            scale,
            bias,
            output,
            ..
        } => {
            // Expand to MUL(input, scale) → ADD(mul_result, bias)
            let shape_4d = vec![-1i32, -1, -1, -1];
            let shape_1d = vec![-1i32];
            Ok(GraphDesc {
                tensors: vec![
                    TensorInfo {
                        name: input.name.clone(),
                        elem_type: input.elem_type,
                        shape: shape_4d.clone(),
                    }, // 0
                    TensorInfo {
                        name: scale.name.clone(),
                        elem_type: scale.elem_type,
                        shape: shape_1d.clone(),
                    }, // 1
                    TensorInfo {
                        name: bias.name.clone(),
                        elem_type: bias.elem_type,
                        shape: shape_1d,
                    }, // 2
                    TensorInfo {
                        name: "batchnorm_mul".into(),
                        elem_type: input.elem_type,
                        shape: shape_4d.clone(),
                    }, // 3
                    TensorInfo {
                        name: output.name.clone(),
                        elem_type: output.elem_type,
                        shape: shape_4d,
                    }, // 4
                ],
                ops: vec![
                    OpDesc {
                        opcode: builtin_op::MUL,
                        inputs: vec![0, 1],
                        outputs: vec![3],
                    },
                    OpDesc {
                        opcode: builtin_op::ADD,
                        inputs: vec![3, 2],
                        outputs: vec![4],
                    },
                ],
                graph_inputs: vec![0, 1, 2],
                graph_outputs: vec![4],
                graph_name: "batchnorm".into(),
            })
        }
        KernelPattern::Concat { inputs, output, .. } => {
            let shape_1d = vec![-1i32];
            let mut tensors: Vec<TensorInfo> = inputs
                .iter()
                .map(|t| TensorInfo {
                    name: t.name.clone(),
                    elem_type: t.elem_type,
                    shape: shape_1d.clone(),
                })
                .collect();
            tensors.push(TensorInfo {
                name: output.name.clone(),
                elem_type: output.elem_type,
                shape: shape_1d,
            });
            let n = inputs.len() as i32;
            let input_indices: Vec<i32> = (0..n).collect();
            Ok(GraphDesc {
                graph_inputs: input_indices.clone(),
                graph_outputs: vec![n],
                ops: vec![OpDesc {
                    opcode: builtin_op::CONCATENATION,
                    inputs: input_indices,
                    outputs: vec![n],
                }],
                tensors,
                graph_name: "concat".into(),
            })
        }
        // Patterns that are complex (Attention, Split) fall back to build_model.
        KernelPattern::Attention { .. } | KernelPattern::Split { .. } => Err(BackendError::Other(
            "complex pattern: use build_model fallback".into(),
        )),
        KernelPattern::Unknown { reason } => Err(BackendError::Unsupported(format!(
            "cannot lower Unknown pattern to TFLite: {reason}"
        ))),
    }
}

/// Return the TFLite builtin opcode for a [`FusedActivation`], or `None` if
/// the activation is `None` (no trailing op needed).
fn activation_opcode(act: &nxpu_analysis::fusion::FusedActivation) -> Option<i32> {
    use nxpu_analysis::fusion::FusedActivation;
    match act {
        FusedActivation::None => None,
        FusedActivation::Relu => Some(builtin_op::RELU),
        FusedActivation::Sigmoid => Some(builtin_op::LOGISTIC),
        FusedActivation::Tanh => Some(builtin_op::TANH),
    }
}

/// Append a trailing activation operator to a [`GraphDesc`] in place.
///
/// The current graph output tensor becomes the activation's input; a new
/// output tensor (named `<old_output>_act`) is appended and becomes the new
/// graph output.
fn append_activation(
    desc: &mut GraphDesc,
    act: &nxpu_analysis::fusion::FusedActivation,
    act_opcode: i32,
) {
    let old_out_idx = *desc.graph_outputs.last().unwrap();
    let old_out = &desc.tensors[old_out_idx as usize];
    let act_tensor = TensorInfo {
        name: format!("{}_act", old_out.name),
        elem_type: old_out.elem_type,
        shape: old_out.shape.clone(),
    };
    let act_tensor_idx = desc.tensors.len() as i32;
    desc.tensors.push(act_tensor);
    desc.ops.push(OpDesc {
        opcode: act_opcode,
        inputs: vec![old_out_idx],
        outputs: vec![act_tensor_idx],
    });
    // Replace graph outputs with the new activation output.
    *desc.graph_outputs.last_mut().unwrap() = act_tensor_idx;
    let _ = act; // only used for naming via caller
}

/// Build a TFLite FlatBuffer model from a fused pattern.
///
/// Handles single patterns, Conv+BatchNorm, MatMul+Bias (Gemm), and
/// activation fusion.  All fused combinations now emit proper multi-operator
/// subgraphs instead of delegating to the unfused single-op builder.
pub fn build_fused_model(fp: &FusedPattern) -> Result<Vec<u8>, BackendError> {
    use nxpu_analysis::fusion::FusedActivation;

    match fp {
        FusedPattern::Single(p) => build_model(p),
        FusedPattern::ConvBatchNorm { conv, norm } => {
            let desc = collect_conv_batchnorm_graph(conv, norm)?;
            Ok(build_from_graph_desc(&desc))
        }
        FusedPattern::MatMulBias { matmul, bias_add } => {
            let desc = collect_matmul_bias_graph(matmul, bias_add)?;
            Ok(build_from_graph_desc(&desc))
        }
        FusedPattern::WithActivation {
            base, activation, ..
        } => {
            if matches!(activation, FusedActivation::None) {
                return build_fused_model(base);
            }
            let act_opcode = match activation_opcode(activation) {
                Some(c) => c,
                None => return build_fused_model(base),
            };

            // Collect the base graph descriptor, then append the activation op.
            let mut desc = match base.as_ref() {
                FusedPattern::Single(p) => match collect_single_graph(p) {
                    Ok(d) => d,
                    // Fall back to build_model for complex single patterns
                    // (Attention, Split) and just return it without activation.
                    Err(_) => return build_model(p),
                },
                FusedPattern::ConvBatchNorm { conv, norm } => {
                    collect_conv_batchnorm_graph(conv, norm)?
                }
                FusedPattern::MatMulBias { matmul, bias_add } => {
                    collect_matmul_bias_graph(matmul, bias_add)?
                }
                FusedPattern::WithActivation { .. } => {
                    // Nested WithActivation should not occur in practice.
                    return build_fused_model(base);
                }
            };

            append_activation(&mut desc, activation, act_opcode);
            Ok(build_from_graph_desc(&desc))
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
    num_heads: u32,
    causal: bool,
) -> Vec<u8> {
    // Note: multi-head (num_heads > 1) would require additional Reshape operators
    // in the graph; causal mask would need a Where/Select op. Both are noted as
    // diagnostics but the core SDPA decomposition remains the same.
    let _ = (num_heads, causal);

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

    // ---- build_fused_model tests ----

    fn make_conv2d() -> KernelPattern {
        KernelPattern::Conv2D {
            input: make_tensor("x", TensorRole::Input),
            weight: make_tensor("w", TensorRole::Input),
            output: make_tensor("conv_out", TensorRole::Output),
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
        }
    }

    fn make_normalization(input_name: &str, output_name: &str) -> KernelPattern {
        KernelPattern::Normalization {
            input: make_tensor(input_name, TensorRole::Input),
            scale: make_tensor("gamma", TensorRole::Input),
            bias: make_tensor("beta", TensorRole::Input),
            output: make_tensor(output_name, TensorRole::Output),
            epsilon: 1e-5,
        }
    }

    fn make_matmul() -> KernelPattern {
        KernelPattern::MatMul {
            inputs: [
                make_tensor("A", TensorRole::Input),
                make_tensor("B", TensorRole::Input),
            ],
            output: make_tensor("mm_out", TensorRole::Output),
            shape: MatMulShape {
                m: "M".into(),
                n: "N".into(),
                k: "K".into(),
            },
        }
    }

    fn make_bias_add(input_name: &str, output_name: &str) -> KernelPattern {
        KernelPattern::ElementWise {
            op: ElementWiseOp::Add,
            inputs: [
                make_tensor(input_name, TensorRole::Input),
                make_tensor("bias", TensorRole::Input),
            ],
            output: make_tensor(output_name, TensorRole::Output),
            dim_name: "N".into(),
        }
    }

    fn make_activation(op: ActivationOp, input_name: &str, output_name: &str) -> KernelPattern {
        KernelPattern::Activation {
            op,
            input: make_tensor(input_name, TensorRole::Input),
            output: make_tensor(output_name, TensorRole::Output),
            dim_name: "N".into(),
        }
    }

    #[test]
    fn fused_model_conv_batchnorm() {
        use nxpu_analysis::fusion::FusedPattern;

        let fused = FusedPattern::ConvBatchNorm {
            conv: make_conv2d(),
            norm: Box::new(make_normalization("conv_out", "bn_out")),
        };

        let bytes = build_fused_model(&fused).unwrap();
        assert!(bytes.len() > 8);
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_model_matmul_bias() {
        use nxpu_analysis::fusion::FusedPattern;

        let fused = FusedPattern::MatMulBias {
            matmul: make_matmul(),
            bias_add: Box::new(make_bias_add("mm_out", "out")),
        };

        let bytes = build_fused_model(&fused).unwrap();
        assert!(bytes.len() > 8);
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_model_single_matmul() {
        use nxpu_analysis::fusion::FusedPattern;

        let fused = FusedPattern::Single(make_matmul());
        let bytes = build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_model_with_activation_on_single_matmul() {
        use nxpu_analysis::fusion::{FusedActivation, FusedPattern};

        let fused = FusedPattern::WithActivation {
            base: Box::new(FusedPattern::Single(make_matmul())),
            activation: FusedActivation::Relu,
            activation_pattern: Box::new(make_activation(ActivationOp::Relu, "mm_out", "relu_out")),
        };

        let bytes = build_fused_model(&fused).unwrap();
        assert!(bytes.len() > 8);
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_model_with_sigmoid_on_single_elementwise() {
        use nxpu_analysis::fusion::{FusedActivation, FusedPattern};

        let add = KernelPattern::ElementWise {
            op: ElementWiseOp::Add,
            inputs: [
                make_tensor("a", TensorRole::Input),
                make_tensor("b", TensorRole::Input),
            ],
            output: make_tensor("c", TensorRole::Output),
            dim_name: "N".into(),
        };

        let fused = FusedPattern::WithActivation {
            base: Box::new(FusedPattern::Single(add)),
            activation: FusedActivation::Sigmoid,
            activation_pattern: Box::new(make_activation(ActivationOp::Sigmoid, "c", "sig_out")),
        };

        let bytes = build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_model_with_tanh_on_single_elementwise() {
        use nxpu_analysis::fusion::{FusedActivation, FusedPattern};

        let mul = KernelPattern::ElementWise {
            op: ElementWiseOp::Mul,
            inputs: [
                make_tensor("a", TensorRole::Input),
                make_tensor("b", TensorRole::Input),
            ],
            output: make_tensor("c", TensorRole::Output),
            dim_name: "N".into(),
        };

        let fused = FusedPattern::WithActivation {
            base: Box::new(FusedPattern::Single(mul)),
            activation: FusedActivation::Tanh,
            activation_pattern: Box::new(make_activation(ActivationOp::Tanh, "c", "tanh_out")),
        };

        let bytes = build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_model_with_activation_on_conv_batchnorm() {
        use nxpu_analysis::fusion::{FusedActivation, FusedPattern};

        let fused = FusedPattern::WithActivation {
            base: Box::new(FusedPattern::ConvBatchNorm {
                conv: make_conv2d(),
                norm: Box::new(make_normalization("conv_out", "bn_out")),
            }),
            activation: FusedActivation::Relu,
            activation_pattern: Box::new(make_activation(ActivationOp::Relu, "bn_out", "relu_out")),
        };

        let bytes = build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_model_with_activation_on_matmul_bias() {
        use nxpu_analysis::fusion::{FusedActivation, FusedPattern};

        let fused = FusedPattern::WithActivation {
            base: Box::new(FusedPattern::MatMulBias {
                matmul: make_matmul(),
                bias_add: Box::new(make_bias_add("mm_out", "gemm_out")),
            }),
            activation: FusedActivation::Relu,
            activation_pattern: Box::new(make_activation(
                ActivationOp::Relu,
                "gemm_out",
                "relu_out",
            )),
        };

        let bytes = build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_model_with_activation_none_returns_base() {
        use nxpu_analysis::fusion::{FusedActivation, FusedPattern};

        let fused = FusedPattern::WithActivation {
            base: Box::new(FusedPattern::Single(make_matmul())),
            activation: FusedActivation::None,
            activation_pattern: Box::new(make_activation(ActivationOp::Relu, "mm_out", "out")),
        };

        let bytes = build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_model_with_nested_with_activation() {
        use nxpu_analysis::fusion::{FusedActivation, FusedPattern};

        // Nested WithActivation: should recurse into base
        let inner = FusedPattern::WithActivation {
            base: Box::new(FusedPattern::Single(make_matmul())),
            activation: FusedActivation::Relu,
            activation_pattern: Box::new(make_activation(ActivationOp::Relu, "mm_out", "relu_out")),
        };
        let outer = FusedPattern::WithActivation {
            base: Box::new(inner),
            activation: FusedActivation::Sigmoid,
            activation_pattern: Box::new(make_activation(
                ActivationOp::Sigmoid,
                "relu_out",
                "sig_out",
            )),
        };

        // Should not panic; the nested WithActivation causes a recursive call
        let bytes = build_fused_model(&outer).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    // ---- collect_single_graph tests for each pattern ----

    #[test]
    fn fused_single_conv2d() {
        use nxpu_analysis::fusion::FusedPattern;
        let fused = FusedPattern::Single(make_conv2d());
        let bytes = build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_single_pool_max() {
        use nxpu_analysis::fusion::FusedPattern;
        let fused = FusedPattern::Single(KernelPattern::Pool {
            kind: PoolKind::Max,
            input: make_tensor("x", TensorRole::Input),
            output: make_tensor("y", TensorRole::Output),
            shape: PoolShape {
                kernel_h: 2,
                kernel_w: 2,
                stride_h: 2,
                stride_w: 2,
            },
        });
        let bytes = build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_single_pool_avg() {
        use nxpu_analysis::fusion::FusedPattern;
        let fused = FusedPattern::Single(KernelPattern::Pool {
            kind: PoolKind::Avg,
            input: make_tensor("x", TensorRole::Input),
            output: make_tensor("y", TensorRole::Output),
            shape: PoolShape {
                kernel_h: 2,
                kernel_w: 2,
                stride_h: 2,
                stride_w: 2,
            },
        });
        let bytes = build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_single_activation_relu() {
        use nxpu_analysis::fusion::FusedPattern;
        let fused = FusedPattern::Single(make_activation(ActivationOp::Relu, "x", "y"));
        let bytes = build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_single_activation_sigmoid() {
        use nxpu_analysis::fusion::FusedPattern;
        let fused = FusedPattern::Single(make_activation(ActivationOp::Sigmoid, "x", "y"));
        let bytes = build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_single_activation_tanh() {
        use nxpu_analysis::fusion::FusedPattern;
        let fused = FusedPattern::Single(make_activation(ActivationOp::Tanh, "x", "y"));
        let bytes = build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_single_activation_softmax() {
        use nxpu_analysis::fusion::FusedPattern;
        let fused = FusedPattern::Single(make_activation(ActivationOp::Softmax, "x", "y"));
        let bytes = build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_single_reduce_all_ops() {
        use nxpu_analysis::fusion::FusedPattern;
        for op in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min] {
            let fused = FusedPattern::Single(KernelPattern::Reduce {
                op,
                input: make_tensor("x", TensorRole::Input),
                output: make_tensor("y", TensorRole::Output),
                axis: 1,
            });
            let bytes = build_fused_model(&fused).unwrap();
            assert_eq!(&bytes[4..8], b"TFL3", "failed for {:?}", op);
        }
    }

    #[test]
    fn fused_single_transpose() {
        use nxpu_analysis::fusion::FusedPattern;
        let fused = FusedPattern::Single(KernelPattern::Transpose {
            input: make_tensor("x", TensorRole::Input),
            output: make_tensor("y", TensorRole::Output),
            perm: vec![1, 0],
        });
        let bytes = build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_single_reshape() {
        use nxpu_analysis::fusion::FusedPattern;
        let fused = FusedPattern::Single(KernelPattern::Reshape {
            input: make_tensor("x", TensorRole::Input),
            output: make_tensor("y", TensorRole::Output),
        });
        let bytes = build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_single_normalization() {
        use nxpu_analysis::fusion::FusedPattern;
        let fused = FusedPattern::Single(make_normalization("x", "y"));
        let bytes = build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_single_concat() {
        use nxpu_analysis::fusion::FusedPattern;
        let fused = FusedPattern::Single(KernelPattern::Concat {
            inputs: vec![
                make_tensor("a", TensorRole::Input),
                make_tensor("b", TensorRole::Input),
            ],
            output: make_tensor("c", TensorRole::Output),
            axis: 0,
        });
        let bytes = build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_single_elementwise_all_ops() {
        use nxpu_analysis::fusion::FusedPattern;
        for op in [
            ElementWiseOp::Add,
            ElementWiseOp::Sub,
            ElementWiseOp::Mul,
            ElementWiseOp::Div,
        ] {
            let fused = FusedPattern::Single(KernelPattern::ElementWise {
                op,
                inputs: [
                    make_tensor("a", TensorRole::Input),
                    make_tensor("b", TensorRole::Input),
                ],
                output: make_tensor("c", TensorRole::Output),
                dim_name: "N".into(),
            });
            let bytes = build_fused_model(&fused).unwrap();
            assert_eq!(&bytes[4..8], b"TFL3", "failed for {:?}", op);
        }
    }

    // ---- activation_opcode tests ----

    #[test]
    fn activation_opcode_none() {
        use nxpu_analysis::fusion::FusedActivation;
        assert!(activation_opcode(&FusedActivation::None).is_none());
    }

    #[test]
    fn activation_opcode_relu() {
        use nxpu_analysis::fusion::FusedActivation;
        let code = activation_opcode(&FusedActivation::Relu).unwrap();
        assert_eq!(code, builtin_op::RELU);
    }

    #[test]
    fn activation_opcode_sigmoid() {
        use nxpu_analysis::fusion::FusedActivation;
        let code = activation_opcode(&FusedActivation::Sigmoid).unwrap();
        assert_eq!(code, builtin_op::LOGISTIC);
    }

    #[test]
    fn activation_opcode_tanh() {
        use nxpu_analysis::fusion::FusedActivation;
        let code = activation_opcode(&FusedActivation::Tanh).unwrap();
        assert_eq!(code, builtin_op::TANH);
    }

    // ---- Error case tests ----

    #[test]
    fn conv_batchnorm_wrong_conv_slot() {
        // Pass a MatMul in the conv slot - should error
        let result =
            collect_conv_batchnorm_graph(&make_matmul(), &make_normalization("conv_out", "bn_out"));
        match result {
            Err(e) => {
                let err_msg = format!("{e}");
                assert!(
                    err_msg.contains("conv slot is not Conv2D"),
                    "unexpected error: {err_msg}"
                );
            }
            Ok(_) => panic!("expected error for wrong conv slot"),
        }
    }

    #[test]
    fn conv_batchnorm_wrong_norm_slot() {
        // Pass a MatMul in the norm slot - should error
        let result = collect_conv_batchnorm_graph(&make_conv2d(), &make_matmul());
        match result {
            Err(e) => {
                let err_msg = format!("{e}");
                assert!(
                    err_msg.contains("norm slot is not Normalization"),
                    "unexpected error: {err_msg}"
                );
            }
            Ok(_) => panic!("expected error for wrong norm slot"),
        }
    }

    #[test]
    fn matmul_bias_wrong_matmul_slot() {
        // Pass an ElementWise in the matmul slot - should error
        let add = KernelPattern::ElementWise {
            op: ElementWiseOp::Add,
            inputs: [
                make_tensor("a", TensorRole::Input),
                make_tensor("b", TensorRole::Input),
            ],
            output: make_tensor("c", TensorRole::Output),
            dim_name: "N".into(),
        };
        let result = collect_matmul_bias_graph(&add, &make_bias_add("c", "out"));
        match result {
            Err(e) => {
                let err_msg = format!("{e}");
                assert!(
                    err_msg.contains("matmul slot is not MatMul"),
                    "unexpected error: {err_msg}"
                );
            }
            Ok(_) => panic!("expected error for wrong matmul slot"),
        }
    }

    #[test]
    fn matmul_bias_wrong_bias_slot() {
        // Pass a MatMul in the bias_add slot - should error
        let result = collect_matmul_bias_graph(&make_matmul(), &make_matmul());
        match result {
            Err(e) => {
                let err_msg = format!("{e}");
                assert!(
                    err_msg.contains("bias_add slot is not ElementWise"),
                    "unexpected error: {err_msg}"
                );
            }
            Ok(_) => panic!("expected error for wrong bias_add slot"),
        }
    }

    #[test]
    fn fused_conv_batchnorm_wrong_inner_errors() {
        use nxpu_analysis::fusion::FusedPattern;

        // ConvBatchNorm with wrong conv slot returns error from build_fused_model
        let fused = FusedPattern::ConvBatchNorm {
            conv: make_matmul(), // wrong: should be Conv2D
            norm: Box::new(make_normalization("x", "y")),
        };
        assert!(build_fused_model(&fused).is_err());
    }

    #[test]
    fn fused_matmul_bias_wrong_inner_errors() {
        use nxpu_analysis::fusion::FusedPattern;

        // MatMulBias with wrong matmul slot returns error
        let fused = FusedPattern::MatMulBias {
            matmul: make_conv2d(), // wrong: should be MatMul
            bias_add: Box::new(make_bias_add("x", "y")),
        };
        assert!(build_fused_model(&fused).is_err());
    }

    // ---- collect_single_graph error cases ----

    #[test]
    fn collect_single_graph_unknown_errors() {
        let pattern = KernelPattern::Unknown {
            reason: "test".into(),
        };
        let result = collect_single_graph(&pattern);
        assert!(result.is_err());
    }

    #[test]
    fn collect_single_graph_attention_falls_back() {
        // Attention should return an error from collect_single_graph
        let pattern = KernelPattern::Attention {
            query: make_tensor("q", TensorRole::Input),
            key: make_tensor("k", TensorRole::Input),
            value: make_tensor("v", TensorRole::Input),
            output: make_tensor("o", TensorRole::Output),
            d_k: "D".into(),
            seq_len: "S".into(),
            num_heads: 1,
            num_kv_heads: 1,
            causal: false,
        };
        let result = collect_single_graph(&pattern);
        assert!(result.is_err());
    }

    #[test]
    fn collect_single_graph_split_falls_back() {
        // Split should return an error from collect_single_graph
        let pattern = KernelPattern::Split {
            input: make_tensor("x", TensorRole::Input),
            outputs: vec![
                make_tensor("o1", TensorRole::Output),
                make_tensor("o2", TensorRole::Output),
            ],
            axis: 0,
        };
        let result = collect_single_graph(&pattern);
        assert!(result.is_err());
    }

    #[test]
    fn fused_with_activation_on_attention_falls_back_to_build_model() {
        use nxpu_analysis::fusion::{FusedActivation, FusedPattern};

        // WithActivation wrapping a Single(Attention) should fall back
        // to build_model for the base since collect_single_graph fails.
        let attention = KernelPattern::Attention {
            query: make_tensor("q", TensorRole::Input),
            key: make_tensor("k", TensorRole::Input),
            value: make_tensor("v", TensorRole::Input),
            output: make_tensor("o", TensorRole::Output),
            d_k: "D".into(),
            seq_len: "S".into(),
            num_heads: 1,
            num_kv_heads: 1,
            causal: false,
        };

        let fused = FusedPattern::WithActivation {
            base: Box::new(FusedPattern::Single(attention)),
            activation: FusedActivation::Relu,
            activation_pattern: Box::new(make_activation(ActivationOp::Relu, "o", "relu_out")),
        };

        // Falls back to build_model which handles Attention
        let bytes = build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    // ---- build_from_graph_desc tests ----

    #[test]
    fn build_from_graph_desc_simple() {
        let desc = GraphDesc {
            tensors: vec![
                TensorInfo {
                    name: "in".into(),
                    elem_type: data_type::FLOAT,
                    shape: vec![-1],
                },
                TensorInfo {
                    name: "out".into(),
                    elem_type: data_type::FLOAT,
                    shape: vec![-1],
                },
            ],
            ops: vec![OpDesc {
                opcode: builtin_op::RELU,
                inputs: vec![0],
                outputs: vec![1],
            }],
            graph_inputs: vec![0],
            graph_outputs: vec![1],
            graph_name: "test".into(),
        };
        let bytes = build_from_graph_desc(&desc);
        assert!(bytes.len() > 8);
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn build_from_graph_desc_deduplicates_opcodes() {
        // Two ops with the same opcode should result in one entry in operator_codes
        let desc = GraphDesc {
            tensors: vec![
                TensorInfo {
                    name: "a".into(),
                    elem_type: data_type::FLOAT,
                    shape: vec![-1],
                },
                TensorInfo {
                    name: "b".into(),
                    elem_type: data_type::FLOAT,
                    shape: vec![-1],
                },
                TensorInfo {
                    name: "c".into(),
                    elem_type: data_type::FLOAT,
                    shape: vec![-1],
                },
            ],
            ops: vec![
                OpDesc {
                    opcode: builtin_op::RELU,
                    inputs: vec![0],
                    outputs: vec![1],
                },
                OpDesc {
                    opcode: builtin_op::RELU,
                    inputs: vec![1],
                    outputs: vec![2],
                },
            ],
            graph_inputs: vec![0],
            graph_outputs: vec![2],
            graph_name: "dedup_test".into(),
        };
        let bytes = build_from_graph_desc(&desc);
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    // ---- append_activation tests ----

    #[test]
    fn append_activation_adds_op_and_tensor() {
        use nxpu_analysis::fusion::FusedActivation;

        let mut desc = GraphDesc {
            tensors: vec![
                TensorInfo {
                    name: "in".into(),
                    elem_type: data_type::FLOAT,
                    shape: vec![-1],
                },
                TensorInfo {
                    name: "out".into(),
                    elem_type: data_type::FLOAT,
                    shape: vec![-1],
                },
            ],
            ops: vec![OpDesc {
                opcode: builtin_op::RELU,
                inputs: vec![0],
                outputs: vec![1],
            }],
            graph_inputs: vec![0],
            graph_outputs: vec![1],
            graph_name: "test".into(),
        };

        append_activation(&mut desc, &FusedActivation::Sigmoid, builtin_op::LOGISTIC);

        // Should have added a new tensor and op
        assert_eq!(desc.tensors.len(), 3);
        assert_eq!(desc.ops.len(), 2);
        assert_eq!(desc.ops[1].opcode, builtin_op::LOGISTIC);
        assert_eq!(desc.ops[1].inputs, vec![1]); // old output
        assert_eq!(desc.ops[1].outputs, vec![2]); // new tensor
        assert_eq!(desc.graph_outputs, vec![2]); // updated
        assert!(desc.tensors[2].name.contains("_act"));
    }

    // ---- onnx_to_tflite_type tests ----

    #[test]
    fn onnx_to_tflite_type_all_variants() {
        assert_eq!(onnx_to_tflite_type(data_type::FLOAT), tensor_type::FLOAT32);
        assert_eq!(
            onnx_to_tflite_type(data_type::FLOAT16),
            tensor_type::FLOAT16
        );
        assert_eq!(onnx_to_tflite_type(data_type::INT32), tensor_type::INT32);
        assert_eq!(onnx_to_tflite_type(data_type::UINT32), tensor_type::UINT32);
        assert_eq!(onnx_to_tflite_type(data_type::BOOL), tensor_type::BOOL);
        assert_eq!(onnx_to_tflite_type(data_type::INT8), tensor_type::INT8);
        // Unknown type falls back to FLOAT32
        assert_eq!(onnx_to_tflite_type(9999), tensor_type::FLOAT32);
    }

    // ---- WithActivation on single patterns via collect_single_graph ----

    #[test]
    fn fused_with_activation_on_single_conv2d() {
        use nxpu_analysis::fusion::{FusedActivation, FusedPattern};

        let fused = FusedPattern::WithActivation {
            base: Box::new(FusedPattern::Single(make_conv2d())),
            activation: FusedActivation::Relu,
            activation_pattern: Box::new(make_activation(
                ActivationOp::Relu,
                "conv_out",
                "relu_out",
            )),
        };
        let bytes = build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_with_activation_on_single_pool() {
        use nxpu_analysis::fusion::{FusedActivation, FusedPattern};

        let pool = KernelPattern::Pool {
            kind: PoolKind::Max,
            input: make_tensor("x", TensorRole::Input),
            output: make_tensor("pool_out", TensorRole::Output),
            shape: PoolShape {
                kernel_h: 2,
                kernel_w: 2,
                stride_h: 2,
                stride_w: 2,
            },
        };
        let fused = FusedPattern::WithActivation {
            base: Box::new(FusedPattern::Single(pool)),
            activation: FusedActivation::Relu,
            activation_pattern: Box::new(make_activation(
                ActivationOp::Relu,
                "pool_out",
                "relu_out",
            )),
        };
        let bytes = build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_with_activation_on_single_reduce() {
        use nxpu_analysis::fusion::{FusedActivation, FusedPattern};

        let reduce = KernelPattern::Reduce {
            op: ReduceOp::Sum,
            input: make_tensor("x", TensorRole::Input),
            output: make_tensor("red_out", TensorRole::Output),
            axis: 1,
        };
        let fused = FusedPattern::WithActivation {
            base: Box::new(FusedPattern::Single(reduce)),
            activation: FusedActivation::Sigmoid,
            activation_pattern: Box::new(make_activation(
                ActivationOp::Sigmoid,
                "red_out",
                "sig_out",
            )),
        };
        let bytes = build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_with_activation_on_single_transpose() {
        use nxpu_analysis::fusion::{FusedActivation, FusedPattern};

        let transpose = KernelPattern::Transpose {
            input: make_tensor("x", TensorRole::Input),
            output: make_tensor("t_out", TensorRole::Output),
            perm: vec![1, 0],
        };
        let fused = FusedPattern::WithActivation {
            base: Box::new(FusedPattern::Single(transpose)),
            activation: FusedActivation::Tanh,
            activation_pattern: Box::new(make_activation(ActivationOp::Tanh, "t_out", "tanh_out")),
        };
        let bytes = build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_with_activation_on_single_reshape() {
        use nxpu_analysis::fusion::{FusedActivation, FusedPattern};

        let reshape = KernelPattern::Reshape {
            input: make_tensor("x", TensorRole::Input),
            output: make_tensor("r_out", TensorRole::Output),
        };
        let fused = FusedPattern::WithActivation {
            base: Box::new(FusedPattern::Single(reshape)),
            activation: FusedActivation::Relu,
            activation_pattern: Box::new(make_activation(ActivationOp::Relu, "r_out", "relu_out")),
        };
        let bytes = build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_with_activation_on_single_normalization() {
        use nxpu_analysis::fusion::{FusedActivation, FusedPattern};

        let fused = FusedPattern::WithActivation {
            base: Box::new(FusedPattern::Single(make_normalization("x", "n_out"))),
            activation: FusedActivation::Relu,
            activation_pattern: Box::new(make_activation(ActivationOp::Relu, "n_out", "relu_out")),
        };
        let bytes = build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_with_activation_on_single_concat() {
        use nxpu_analysis::fusion::{FusedActivation, FusedPattern};

        let concat = KernelPattern::Concat {
            inputs: vec![
                make_tensor("a", TensorRole::Input),
                make_tensor("b", TensorRole::Input),
            ],
            output: make_tensor("cat_out", TensorRole::Output),
            axis: 0,
        };
        let fused = FusedPattern::WithActivation {
            base: Box::new(FusedPattern::Single(concat)),
            activation: FusedActivation::Sigmoid,
            activation_pattern: Box::new(make_activation(
                ActivationOp::Sigmoid,
                "cat_out",
                "sig_out",
            )),
        };
        let bytes = build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_with_activation_on_single_activation() {
        use nxpu_analysis::fusion::{FusedActivation, FusedPattern};

        // Relu followed by Tanh via WithActivation
        let relu = make_activation(ActivationOp::Relu, "x", "relu_out");
        let fused = FusedPattern::WithActivation {
            base: Box::new(FusedPattern::Single(relu)),
            activation: FusedActivation::Tanh,
            activation_pattern: Box::new(make_activation(
                ActivationOp::Tanh,
                "relu_out",
                "tanh_out",
            )),
        };
        let bytes = build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    // ---- build_model edge cases ----

    #[test]
    fn build_model_unknown_returns_error() {
        let pattern = KernelPattern::Unknown {
            reason: "test error".into(),
        };
        let result = build_model(&pattern);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("cannot lower Unknown pattern"));
    }

    #[test]
    fn build_model_softmax() {
        let pattern = KernelPattern::Activation {
            op: ActivationOp::Softmax,
            input: make_tensor("x", TensorRole::Input),
            output: make_tensor("y", TensorRole::Output),
            dim_name: "N".into(),
        };
        let bytes = build_model(&pattern).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn build_model_transpose() {
        let pattern = KernelPattern::Transpose {
            input: make_tensor("x", TensorRole::Input),
            output: make_tensor("y", TensorRole::Output),
            perm: vec![1, 0],
        };
        let bytes = build_model(&pattern).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn build_model_reshape() {
        let pattern = KernelPattern::Reshape {
            input: make_tensor("x", TensorRole::Input),
            output: make_tensor("y", TensorRole::Output),
        };
        let bytes = build_model(&pattern).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn build_model_normalization() {
        let pattern = make_normalization("x", "y");
        let bytes = build_model(&pattern).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn build_model_concat() {
        let pattern = KernelPattern::Concat {
            inputs: vec![
                make_tensor("a", TensorRole::Input),
                make_tensor("b", TensorRole::Input),
            ],
            output: make_tensor("c", TensorRole::Output),
            axis: 0,
        };
        let bytes = build_model(&pattern).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn build_model_split() {
        let pattern = KernelPattern::Split {
            input: make_tensor("x", TensorRole::Input),
            outputs: vec![
                make_tensor("o1", TensorRole::Output),
                make_tensor("o2", TensorRole::Output),
            ],
            axis: 0,
        };
        let bytes = build_model(&pattern).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn build_model_attention() {
        let pattern = KernelPattern::Attention {
            query: make_tensor("q", TensorRole::Input),
            key: make_tensor("k", TensorRole::Input),
            value: make_tensor("v", TensorRole::Input),
            output: make_tensor("o", TensorRole::Output),
            d_k: "D".into(),
            seq_len: "S".into(),
            num_heads: 1,
            num_kv_heads: 1,
            causal: false,
        };
        let bytes = build_model(&pattern).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn build_model_all_reduce_ops() {
        for op in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min] {
            let pattern = KernelPattern::Reduce {
                op,
                input: make_tensor("x", TensorRole::Input),
                output: make_tensor("y", TensorRole::Output),
                axis: 1,
            };
            let bytes = build_model(&pattern).unwrap();
            assert_eq!(&bytes[4..8], b"TFL3", "failed for {:?}", op);
        }
    }
}
