//! TFLite FlatBuffer model construction from classified kernel patterns.
//!
//! Builds a TFLite model using the `flatbuffers` crate's builder API
//! with manual table construction (no generated code, no .fbs schema).

use flatbuffers::FlatBufferBuilder;
use nxpu_analysis::analyze::data_type;
use nxpu_analysis::analyze::{
    ActivationOp, ChainOperand, ChainStep, Conv2DShape, ElementWiseOp, KernelPattern, PoolKind,
    PoolShape, ReduceOp, TensorBinding,
};
use nxpu_analysis::fusion::FusedPattern;
use nxpu_backend_core::BackendError;

use crate::schema::{
    activation_function, builtin_op, builtin_options_type, concatenation_options, conv2d_options,
    padding, pool2d_options, softmax_options, split_options, tensor_type, vt,
};

/// File identifier for TFLite FlatBuffer files.
const TFLITE_FILE_ID: &str = "TFL3";

/// Create a `Tensor.shape` vector, replacing symbolic dimensions with
/// [`SYMBOLIC_EXTENT`].
///
/// Every shape written into a tensor goes through here, so a `-1` arriving
/// from any of the pattern builders is normalised in one place rather than at
/// each of the thirteen sites that build one.
fn shape_vector<'a>(
    fbb: &mut FlatBufferBuilder<'a>,
    dims: &[i32],
    extent: i32,
) -> flatbuffers::WIPOffset<flatbuffers::Vector<'a, i32>> {
    fbb.create_vector(&concrete_shape(dims, extent))
}

/// Substitute `extent` for every symbolic dimension, leaving known ones alone.
///
/// Separate from [`shape_vector`] so the rule can be asserted directly: every
/// pattern this backend lowers today produces entirely symbolic shapes, so the
/// branch that preserves a known dimension is never reached through them, and
/// a rule nothing exercises is a rule nothing protects.
fn concrete_shape(dims: &[i32], extent: i32) -> Vec<i32> {
    dims.iter()
        .map(|&d| if d < 0 { extent } else { d })
        .collect()
}

/// Build a TFLite FlatBuffer model from a classified kernel pattern.
pub fn build_model(pattern: &KernelPattern, extent: i32) -> Result<Vec<u8>, BackendError> {
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
                extent,
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
                extent,
            )
        }
        // Several operators, so these go through the multi-op builder rather
        // than through `build_tflite`, which emits exactly one.
        KernelPattern::ElementWiseChain { .. } | KernelPattern::QuantizedMatMul { .. } => {
            let desc = collect_single_graph(pattern, extent)?;
            build_from_graph_desc(&desc, extent)
        }
        KernelPattern::Conv2D {
            input,
            weight,
            output,
            shape,
            bias,
            activation,
        } => build_tflite_conv2d(
            input,
            weight,
            bias.as_ref(),
            output,
            shape,
            *activation,
            extent,
        ),
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
            build_tflite_pool(input, output, opcode, shape, "pool", extent)
        }
        KernelPattern::Activation {
            op, input, output, ..
        } => {
            if matches!(op, ActivationOp::Softmax) {
                build_tflite_softmax(input, output, extent)
            } else if matches!(
                op,
                ActivationOp::Gelu | ActivationOp::Silu | ActivationOp::Mish
            ) {
                // GELU is builtin 150. It went out as CUSTOM with no
                // custom_code, which no interpreter can register — the model
                // was rejected before anything looked at what it computed.
                // SiLU and Mish have no builtin and remain custom, and remain
                // unloadable for the same reason; naming them here is the
                // honest half.
                let shapes = [vec![-1i32], vec![-1]];
                let opcode = if matches!(op, ActivationOp::Gelu) {
                    builtin_op::GELU
                } else {
                    builtin_op::CUSTOM
                };
                build_tflite_unary(
                    input,
                    output,
                    &shapes[0],
                    &shapes[1],
                    opcode,
                    &format!("{}_1d", op.op_name().to_lowercase()),
                    extent,
                )
            } else {
                let shapes = [vec![-1i32], vec![-1]];
                let opcode = match op {
                    ActivationOp::Relu => builtin_op::RELU,
                    ActivationOp::Sigmoid => builtin_op::LOGISTIC,
                    ActivationOp::Tanh => builtin_op::TANH,
                    _ => unreachable!(),
                };
                build_tflite_unary(
                    input,
                    output,
                    &shapes[0],
                    &shapes[1],
                    opcode,
                    &format!("{}_1d", op.op_name().to_lowercase()),
                    extent,
                )
            }
        }
        KernelPattern::Reduce {
            op,
            input,
            output,
            axis,
        } => {
            let shapes = [vec![-1, -1], vec![-1]];
            let opcode = match op {
                ReduceOp::Sum => builtin_op::SUM,
                ReduceOp::Mean => builtin_op::MEAN,
                ReduceOp::Max => builtin_op::REDUCE_MAX,
                ReduceOp::Min => builtin_op::REDUCE_MIN,
            };
            build_tflite_reduce(
                input,
                output,
                &shapes[0],
                &shapes[1],
                opcode,
                &format!("{}_reduce", op.op_name().to_lowercase()),
                *axis,
                extent,
            )
        }
        KernelPattern::Transpose {
            input,
            output,
            perm,
        } => {
            let dims: Vec<i32> = vec![-1i32; perm.len().max(2)];
            build_tflite_transpose(input, output, &dims, &dims, perm, extent)
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
                extent,
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
            build_tflite_batchnorm(input, scale, bias, output, extent)
        }
        KernelPattern::Concat {
            inputs,
            output,
            axis,
        } => build_tflite_concat(inputs, output, *axis, extent),
        KernelPattern::Split {
            input,
            outputs,
            axis,
        } => build_tflite_split(input, outputs, *axis, extent),
        KernelPattern::Attention {
            query,
            key,
            value,
            output,
            d_k,
            num_heads,
            causal,
            ..
        } => build_tflite_attention(query, key, value, output, d_k, *num_heads, *causal, extent),
        KernelPattern::Gather {
            data,
            indices,
            output,
            ..
        } => {
            // TFLite refuses UINT32 positions. WGSL indexes with u32, and
            // for a non-negative index the bytes are identical, so what has
            // to change is the declared type rather than the data.
            let indices_i32 = TensorBinding {
                elem_type: data_type::INT32,
                ..indices.clone()
            };
            let shapes = [vec![-1i32], vec![-1], vec![-1]];
            build_tflite(
                &[data, &indices_i32],
                output,
                &shapes,
                builtin_op::GATHER,
                "gather",
                extent,
            )
        }
        KernelPattern::Scatter {
            indices,
            updates,
            output,
            ..
        } => {
            // TFLite's SCATTER_ND is `(indices, updates, shape)`. Emitting
            // `(data, indices, updates)` made the runtime read the indices as
            // the updates and refuse them for being UINT32 — the type error
            // was a symptom of the operands being in the wrong places.
            //
            // The base tensor is not dropped information: the kernel these
            // patterns come from never reads it. `output[indices[i]] =
            // updates[i]` scatters into a fresh tensor, which is what
            // SCATTER_ND does.
            //
            // It still does not load. SCATTER_ND checks its operands against
            // each other — `updates.DimensionsCount() - outer_dims ==
            // shape.Dims(0) - ix` — and that needs real ranks, while every
            // tensor this backend emits is rank-1 with a symbolic extent.
            // `TensorBinding` carries an element type and no shape, so there
            // is nothing here to satisfy the check with. The operand order and
            // the index type are fixed because they were wrong on their own
            // terms; the rest waits for shapes.
            let indices_i32 = TensorBinding {
                elem_type: data_type::INT32,
                ..indices.clone()
            };
            // SCATTER_ND checks its operands against each other:
            // `updates.rank - outer_dims == shape.Dims(0) - indices.Dims(last)`.
            // For a flat output that balances when the indices are `[N, 1]` —
            // N index vectors of length one — and the updates are `[N]`.
            build_tflite_scatter(&indices_i32, output, &[-1i32, 1], &[-1i32], updates, extent)
        }
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
    /// Contents, little-endian, for a tensor whose value is fixed at compile
    /// time — a `TRANSPOSE` permutation, say.
    ///
    /// A tensor is constant to TFLite exactly when its buffer carries data,
    /// and it is a graph input exactly when its index is in `graph_inputs`; a
    /// constant must be one and not the other, which is the caller's to get
    /// right. Before this field the only way to write a constant was a bespoke
    /// builder, which is why there are ten of them below.
    data: Option<Vec<u8>>,
}

impl TensorInfo {
    /// A tensor the caller supplies at inference time.
    fn input(name: impl Into<String>, elem_type: i32, shape: Vec<i32>) -> Self {
        Self {
            name: name.into(),
            elem_type,
            shape,
            data: None,
        }
    }

    /// A tensor whose contents are fixed now.
    fn constant(name: impl Into<String>, elem_type: i32, shape: Vec<i32>, data: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            elem_type,
            shape,
            data: Some(data),
        }
    }
}

/// Builtin options for one operator in a multi-op subgraph.
///
/// [`build_from_graph_desc`] wrote no options at all, which is survivable for
/// an operator that has none and fatal for one that does: TFLite's schema
/// default for a stride is 0, so a CONV_2D emitted through that path was
/// rejected by its own kernel with `params->stride_height > 0 was not true`
/// before any delegate saw it.
enum OpOptions {
    /// The operator has no options table, or needs none.
    None,
    Conv2D {
        padding: i8,
        stride_w: i32,
        stride_h: i32,
        dilation_w: i32,
        dilation_h: i32,
        /// Folded into the convolution rather than appended after it. A kernel
        /// writing `max(sum, 0.0)` is a convolution and a ReLU, and emitting
        /// only the convolution is a graph that runs and returns unclipped
        /// values.
        activation: i32,
    },
    Pool2D {
        padding: i8,
        stride_w: i32,
        stride_h: i32,
        filter_w: i32,
        filter_h: i32,
    },
    /// `PadOptions` is an empty table, but the union tag still has to be set
    /// or the operator carries a payload TFLite will not look for.
    Pad,
}

/// An operator descriptor used when building multi-op TFLite subgraphs.
struct OpDesc {
    opcode: i32,
    inputs: Vec<i32>,
    outputs: Vec<i32>,
    options: OpOptions,
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
fn build_from_graph_desc(desc: &GraphDesc, extent: i32) -> Vec<u8> {
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
        .map(|t| shape_vector(&mut fbb, &t.shape, extent))
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

    // --- constant contents, written before the buffer tables that hold them ---
    //
    // FlatBuffers cannot start a vector while a table is open, so every
    // constant's bytes have to exist before the loop below runs.
    let data_offsets: Vec<Option<_>> = desc
        .tensors
        .iter()
        .map(|t| t.data.as_ref().map(|d| fbb.create_vector(d)))
        .collect();

    // --- buffers: sentinel(0) + one per tensor ---
    let num_tensors = desc.tensors.len();
    let mut buf_offsets = Vec::with_capacity(num_tensors + 1);
    for i in 0..=num_tensors {
        let start = fbb.start_table();
        // Buffer 0 is the sentinel and holds nothing; buffer i + 1 belongs to
        // tensor i.
        if let Some(data) = i.checked_sub(1).and_then(|t| data_offsets[t]) {
            fbb.push_slot_always(vt::buffer::DATA, data);
        }
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

    // --- builtin options, before the operator tables that point at them ---
    //
    // FlatBuffers cannot start a table while another is open, so an operator's
    // options table has to be finished before the operator's own table begins.
    let option_offsets: Vec<
        Option<(
            u8,
            flatbuffers::WIPOffset<flatbuffers::TableFinishedWIPOffset>,
        )>,
    > = desc
        .ops
        .iter()
        .map(|op| match &op.options {
            OpOptions::None => None,
            OpOptions::Conv2D {
                padding: pad,
                stride_w,
                stride_h,
                dilation_w,
                dilation_h,
                activation,
            } => {
                let start = fbb.start_table();
                // Every default here is the value that must not be written
                // as absence: SAME for the padding, 0 for a stride.
                fbb.push_slot::<i8>(conv2d_options::PADDING, *pad, padding::SAME);
                fbb.push_slot::<i32>(conv2d_options::STRIDE_W, (*stride_w).max(1), 0);
                fbb.push_slot::<i32>(conv2d_options::STRIDE_H, (*stride_h).max(1), 0);
                fbb.push_slot::<i32>(
                    conv2d_options::ACTIVATION,
                    *activation,
                    activation_function::NONE,
                );
                fbb.push_slot::<i32>(conv2d_options::DILATION_W, (*dilation_w).max(1), 1);
                fbb.push_slot::<i32>(conv2d_options::DILATION_H, (*dilation_h).max(1), 1);
                Some((builtin_options_type::CONV_2D, fbb.end_table(start)))
            }
            OpOptions::Pool2D {
                padding: pad,
                stride_w,
                stride_h,
                filter_w,
                filter_h,
            } => {
                let start = fbb.start_table();
                fbb.push_slot::<i8>(pool2d_options::PADDING, *pad, padding::SAME);
                fbb.push_slot::<i32>(pool2d_options::STRIDE_W, (*stride_w).max(1), 0);
                fbb.push_slot::<i32>(pool2d_options::STRIDE_H, (*stride_h).max(1), 0);
                fbb.push_slot::<i32>(pool2d_options::FILTER_W, (*filter_w).max(1), 0);
                fbb.push_slot::<i32>(pool2d_options::FILTER_H, (*filter_h).max(1), 0);
                fbb.push_slot::<i32>(pool2d_options::ACTIVATION, 0, 0);
                Some((builtin_options_type::POOL_2D, fbb.end_table(start)))
            }
            OpOptions::Pad => {
                let start = fbb.start_table();
                Some((builtin_options_type::PAD, fbb.end_table(start)))
            }
        })
        .collect();

    // --- operators ---
    let mut operator_offsets = Vec::with_capacity(desc.ops.len());
    for (i, op) in desc.ops.iter().enumerate() {
        let opcode_index = unique_opcodes.iter().position(|&c| c == op.opcode).unwrap() as u32;
        let o = {
            let start = fbb.start_table();
            fbb.push_slot::<u32>(vt::operator::OPCODE_INDEX, opcode_index, 0);
            fbb.push_slot_always(vt::operator::INPUTS, op_input_offsets[i]);
            fbb.push_slot_always(vt::operator::OUTPUTS, op_output_offsets[i]);
            if let Some((kind, table)) = option_offsets[i] {
                fbb.push_slot::<u8>(vt::operator::BUILTIN_OPTIONS_TYPE, kind, 0);
                fbb.push_slot_always(vt::operator::BUILTIN_OPTIONS, table);
            }
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

/// Build a [`GraphDesc`] for `ConvBatchNorm`: an optional `PAD`, then
/// CONV_2D → MUL(scale) → ADD(shift).
///
/// Graph inputs are the convolution's input and weight, its bias when the
/// kernel has one, then the normalization's scale and shift; the graph output
/// is the ADD's result. Everything else is an intermediate or a constant, and
/// a constant is not a graph input.
///
/// The three defects [`conv2d_graph`] was written to fix all lived here too,
/// because the fused path builds its own convolution: no `Conv2DOptions`, two
/// inputs where the kernel demands three, and one `[-1,-1,-1,-1]` written to
/// the input, the weight and the output alike. The tests said `TFL3` and
/// passed throughout.
fn collect_conv_batchnorm_graph(
    conv: &KernelPattern,
    norm: &KernelPattern,
    extent: i32,
) -> Result<GraphDesc, BackendError> {
    let (input, weight, conv_bias, conv_out, conv_shape, conv_activation) = match conv {
        KernelPattern::Conv2D {
            input,
            weight,
            bias,
            output,
            shape,
            activation,
        } => (input, weight, bias.as_ref(), output, shape, *activation),
        _ => {
            return Err(BackendError::Other(
                "ConvBatchNorm: conv slot is not Conv2D".into(),
            ));
        }
    };
    let (scale, shift, output) = match norm {
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

    // The same derivation [`conv2d_graph`] makes, and it has to stay the same
    // one: a fused convolution and a standalone convolution that disagree
    // about the output's size are two models for one kernel. TFLite reads
    // these as NHWC with the weight as [out, kh, kw, in].
    let n = extent.max(1);
    let channels_in = extent.max(1);
    let channels_out = extent.max(1);
    let in_h = extent.max(1);
    let in_w = extent.max(1);

    // A window the kernel states as a literal is known exactly; one supplied
    // through the params struct is not, and falls back to the extent.
    let window = |literal: i64| {
        if literal > 0 {
            (literal as i32).max(1)
        } else {
            extent.max(1)
        }
    };
    let kernel_h = window(conv_shape.kernel_h_val);
    let kernel_w = window(conv_shape.kernel_w_val);

    let pad_h = conv_shape.pad_h.max(0) as i32;
    let pad_w = conv_shape.pad_w.max(0) as i32;
    let padded_h = in_h + 2 * pad_h;
    let padded_w = in_w + 2 * pad_w;

    // floor((in - reach) / stride) + 1, which is what TFLite's own kernel
    // computes for VALID, over the padded extent because that is what the
    // convolution now sees.
    let valid_out = |input: i32, k: i32, stride: i32, dilation: i32| {
        let reach = dilation.max(1) * (k - 1) + 1;
        ((input - reach) / stride.max(1) + 1).max(1)
    };
    let out_h = valid_out(
        padded_h,
        kernel_h,
        conv_shape.stride_h as i32,
        conv_shape.dilation_h as i32,
    );
    let out_w = valid_out(
        padded_w,
        kernel_w,
        conv_shape.stride_w as i32,
        conv_shape.dilation_w as i32,
    );
    let conv_out_shape = vec![n, out_h, out_w, channels_out];

    let mut tensors = vec![TensorInfo::input(
        input.name.clone(),
        input.elem_type,
        vec![n, in_h, in_w, channels_in],
    )];
    let mut ops = Vec::new();

    // The tensor the convolution reads: the input itself, or the padded copy.
    let conv_input = if pad_h > 0 || pad_w > 0 {
        let amounts: [i32; 8] = [0, 0, pad_h, pad_h, pad_w, pad_w, 0, 0];
        tensors.push(TensorInfo::constant(
            format!("{}_paddings", input.name),
            data_type::INT32,
            vec![4, 2],
            amounts.iter().flat_map(|d| d.to_le_bytes()).collect(),
        ));
        tensors.push(TensorInfo::input(
            format!("{}_padded", input.name),
            input.elem_type,
            vec![n, padded_h, padded_w, channels_in],
        ));
        ops.push(OpDesc {
            opcode: builtin_op::PAD,
            inputs: vec![0, 1],
            outputs: vec![2],
            options: OpOptions::Pad,
        });
        2
    } else {
        0
    };

    let weight_index = tensors.len() as i32;
    tensors.push(TensorInfo::input(
        weight.name.clone(),
        weight.elem_type,
        vec![channels_out, kernel_h, kernel_w, channels_in],
    ));

    // TFLite's CONV_2D kernel requires three inputs — `has_bias was not true`
    // is a hard failure — so a convolution whose source has no bias still gets
    // one, as a constant of zeros. Folding a batch norm onto a convolution
    // does not remove the convolution's own bias: it is added before the scale
    // multiplies, so dropping it is a different function, not a cheaper one.
    let bias_index = tensors.len() as i32;
    match conv_bias {
        Some(b) => tensors.push(TensorInfo::input(
            b.name.clone(),
            b.elem_type,
            vec![channels_out],
        )),
        None => tensors.push(TensorInfo::constant(
            "bias",
            data_type::FLOAT,
            vec![channels_out],
            vec![0u8; (channels_out as usize) * 4],
        )),
    }

    let conv_out_index = tensors.len() as i32;
    tensors.push(TensorInfo::input(
        conv_out.name.clone(),
        conv_out.elem_type,
        conv_out_shape.clone(),
    ));

    ops.push(OpDesc {
        opcode: builtin_op::CONV_2D,
        inputs: vec![conv_input, weight_index, bias_index],
        outputs: vec![conv_out_index],
        options: OpOptions::Conv2D {
            // VALID, because the padding is the PAD operator's business.
            padding: padding::VALID,
            stride_w: conv_shape.stride_w as i32,
            stride_h: conv_shape.stride_h as i32,
            dilation_w: conv_shape.dilation_w as i32,
            dilation_h: conv_shape.dilation_h as i32,
            // The kernel applies it to what the convolution stores, so it
            // belongs on the convolution and ahead of the normalisation that
            // reads that tensor -- not after the whole chain.
            activation: match conv_activation {
                Some(ActivationOp::Relu) => activation_function::RELU,
                Some(ActivationOp::Tanh) => activation_function::TANH,
                _ => activation_function::NONE,
            },
        },
    });

    // Scale and shift are per output channel, so they broadcast over NHWC.
    // They were [-1] here, which the extent turns into the same length as
    // every other dimension — right only when the channel count happens to be
    // the extent, and silently wrong the moment the window is a literal.
    let scale_index = tensors.len() as i32;
    tensors.push(TensorInfo::input(
        scale.name.clone(),
        scale.elem_type,
        vec![channels_out],
    ));
    let scaled_index = tensors.len() as i32;
    tensors.push(TensorInfo::input(
        format!("{}_scaled", conv_out.name),
        conv_out.elem_type,
        conv_out_shape.clone(),
    ));
    ops.push(OpDesc {
        opcode: builtin_op::MUL,
        inputs: vec![conv_out_index, scale_index],
        outputs: vec![scaled_index],
        options: OpOptions::None,
    });

    let shift_index = tensors.len() as i32;
    tensors.push(TensorInfo::input(
        shift.name.clone(),
        shift.elem_type,
        vec![channels_out],
    ));
    let output_index = tensors.len() as i32;
    tensors.push(TensorInfo::input(
        output.name.clone(),
        output.elem_type,
        conv_out_shape,
    ));
    ops.push(OpDesc {
        opcode: builtin_op::ADD,
        inputs: vec![scaled_index, shift_index],
        outputs: vec![output_index],
        options: OpOptions::None,
    });

    // A constant is not a graph input; the synthesised bias and the paddings
    // are constants, and listing either here makes the model invalid.
    let mut graph_inputs = vec![0, weight_index];
    if conv_bias.is_some() {
        graph_inputs.push(bias_index);
    }
    graph_inputs.push(scale_index);
    graph_inputs.push(shift_index);

    Ok(GraphDesc {
        tensors,
        ops,
        graph_inputs,
        graph_outputs: vec![output_index],
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
            TensorInfo::input(
                mm_inputs[0].name.clone(),
                mm_inputs[0].elem_type,
                shape_2d.clone(),
            ), // 0: A
            TensorInfo::input(
                mm_inputs[1].name.clone(),
                mm_inputs[1].elem_type,
                shape_2d.clone(),
            ), // 1: B
            TensorInfo::input(bias.name.clone(), bias.elem_type, shape_1d), // 2: bias
            TensorInfo::input(
                mm_output.name.clone(),
                mm_output.elem_type,
                shape_2d.clone(),
            ), // 3: mm_out
            TensorInfo::input(output.name.clone(), output.elem_type, shape_2d), // 4: output
        ],
        ops: vec![
            OpDesc {
                opcode: builtin_op::BATCH_MATMUL,
                inputs: vec![0, 1],
                outputs: vec![3],
                options: OpOptions::None,
            },
            OpDesc {
                opcode: builtin_op::ADD,
                inputs: vec![3, 2],
                outputs: vec![4],
                options: OpOptions::None,
            },
        ],
        graph_inputs: vec![0, 1, 2],
        graph_outputs: vec![4],
        graph_name: "gemm".into(),
    })
}

/// Build a [`GraphDesc`] for an element-wise chain: an optional `CAST`,
/// then one binary operator per step.
///
/// The scalars get rank-0 tensors — a shape of no dimensions, not a length-one
/// vector — because that is what they are, and TFLite broadcasts them over the
/// other operand. Giving them `[1]` would be a claim about a dimension the
/// kernel does not have.
fn collect_elementwise_chain_graph(
    base: &TensorBinding,
    cast: Option<i32>,
    steps: &[ChainStep],
    output: &TensorBinding,
) -> Result<GraphDesc, BackendError> {
    let vector = vec![-1i32];
    let scalar: Vec<i32> = Vec::new();

    let mut tensors = vec![TensorInfo::input(
        base.name.clone(),
        base.elem_type,
        vector.clone(),
    )];
    let mut graph_inputs = vec![0i32];
    let mut ops: Vec<OpDesc> = Vec::new();
    let mut acc = 0i32;
    // The type the accumulator carries: the base's, until a cast changes it.
    let mut acc_type = base.elem_type;

    if let Some(to) = cast {
        acc_type = to;
        let idx = tensors.len() as i32;
        tensors.push(TensorInfo::input(
            format!("{}_cast", base.name),
            to,
            vector.clone(),
        ));
        ops.push(OpDesc {
            opcode: builtin_op::CAST,
            inputs: vec![acc],
            outputs: vec![idx],
            options: OpOptions::None,
        });
        acc = idx;
    }

    for (i, step) in steps.iter().enumerate() {
        let opcode = match step.op {
            ElementWiseOp::Add => builtin_op::ADD,
            ElementWiseOp::Sub => builtin_op::SUB,
            ElementWiseOp::Mul => builtin_op::MUL,
            ElementWiseOp::Div => builtin_op::DIV,
        };
        let operand_idx = tensors.len() as i32;
        match &step.operand {
            ChainOperand::Tensor(t) => tensors.push(TensorInfo::input(
                t.name.clone(),
                t.elem_type,
                vector.clone(),
            )),
            ChainOperand::Scalar(s) => tensors.push(TensorInfo::input(
                s.name.clone(),
                s.elem_type,
                scalar.clone(),
            )),
        }
        graph_inputs.push(operand_idx);

        let last = i + 1 == steps.len();
        let result_idx = tensors.len() as i32;
        tensors.push(if last {
            TensorInfo::input(output.name.clone(), output.elem_type, vector.clone())
        } else {
            TensorInfo::input(format!("{}_step{i}", output.name), acc_type, vector.clone())
        });
        ops.push(OpDesc {
            opcode,
            inputs: vec![acc, operand_idx],
            outputs: vec![result_idx],
            options: OpOptions::None,
        });
        acc = result_idx;
    }

    if ops.is_empty() {
        return Err(BackendError::Other(
            "element-wise chain with no steps".into(),
        ));
    }

    Ok(GraphDesc {
        tensors,
        ops,
        graph_inputs,
        graph_outputs: vec![acc],
        graph_name: nxpu_analysis::analyze::chain_summary(cast, steps).to_lowercase(),
    })
}

/// Build a [`GraphDesc`] for a quantized matmul:
/// `TRANSPOSE → CAST → BATCH_MATMUL → MUL(scale) → ADD(bias)`.
///
/// **Why five operators and not one.** TFLite does have per-channel quantized
/// tensors, and this does not use them, because it cannot: a quantized
/// tensor's scales live in `Tensor.quantization`, which is model metadata
/// written when the file is written, and here the scales arrive in a storage
/// buffer the host fills per dispatch. Baking whatever `--symbolic-dim` says
/// into that field would be inventing the weights. So the dequantization is
/// spelled out — the codes are cast to f32 and multiplied by the scale tensor,
/// which is a graph input like any other.
///
/// **Why the scale multiplies after the contraction.** `out[m][n]` is
/// `sum_k a[m][k] * w[n][k] * scale[n]`, and `scale[n]` does not depend on `k`,
/// so it comes out of the sum. That is exactly the hoist `matvec/q8.wgsl` does
/// (`shared_sum[0] * scale[row]`), it saves a multiply per contracted position,
/// and it lets the scale broadcast over the result's last axis with no reshape:
/// `[m, n] * [n]` is a TFLite broadcast, `[n, k] * [n]` is not.
///
/// **Why the transpose, and why it comes first.** The weight's rows are output
/// channels, so the contraction runs along its second axis, and `BATCH_MATMUL`
/// contracts the second operand's *first* axis. `BatchMatMulOptions.adj_y`
/// would say so in one flag instead, but that options table is a new
/// `BuiltinOptions` union index and this file's own history says those get
/// written wrong; a `TRANSPOSE` with a constant permutation is the same graph
/// out of parts already proven to load. Transposing the codes rather than the
/// dequantized floats moves a quarter of the bytes, and it is the order the
/// ONNX graph has to use — onnxruntime's transpose optimizer mis-types a
/// transpose that sits after a `DequantizeLinear` — so the two backends emit
/// the same operators in the same order.
fn collect_quantized_matmul_graph(
    input: &TensorBinding,
    weight: &TensorBinding,
    scale: &TensorBinding,
    bias: Option<&TensorBinding>,
    output: &TensorBinding,
    shape: &nxpu_analysis::analyze::MatMulShape,
) -> GraphDesc {
    // A matvec is a matmul over one row, and the classifier says so by naming
    // the row count `1` rather than a dispatch parameter. That dimension is
    // known, so it is written rather than left to `--symbolic-dim`.
    let rows: i32 = if shape.m == "1" { 1 } else { -1 };
    let matrix = vec![-1i32, -1];
    let result = vec![rows, -1];

    let mut tensors = vec![
        TensorInfo::input(input.name.clone(), input.elem_type, vec![rows, -1]),
        TensorInfo::input(weight.name.clone(), weight.elem_type, matrix.clone()),
        TensorInfo::input(scale.name.clone(), scale.elem_type, vec![-1i32]),
    ];
    let mut graph_inputs = vec![0, 1, 2];
    if let Some(bias) = bias {
        tensors.push(TensorInfo::input(
            bias.name.clone(),
            bias.elem_type,
            vec![-1i32],
        ));
        graph_inputs.push(3);
    }

    let perm = tensors.len() as i32;
    tensors.push(TensorInfo::constant(
        format!("{}_perm", weight.name),
        data_type::INT32,
        vec![2],
        [1i32, 0].iter().flat_map(|d| d.to_le_bytes()).collect(),
    ));
    let transposed = tensors.len() as i32;
    tensors.push(TensorInfo::input(
        format!("{}_contraction_major", weight.name),
        weight.elem_type,
        matrix.clone(),
    ));
    let dequantized = tensors.len() as i32;
    tensors.push(TensorInfo::input(
        format!("{}_dequantized", weight.name),
        output.elem_type,
        matrix,
    ));
    let contracted = tensors.len() as i32;
    tensors.push(TensorInfo::input(
        format!("{}_unscaled", output.name),
        output.elem_type,
        result.clone(),
    ));

    let mut ops = vec![
        OpDesc {
            opcode: builtin_op::TRANSPOSE,
            inputs: vec![1, perm],
            outputs: vec![transposed],
            options: OpOptions::None,
        },
        OpDesc {
            opcode: builtin_op::CAST,
            inputs: vec![transposed],
            outputs: vec![dequantized],
            options: OpOptions::None,
        },
        OpDesc {
            opcode: builtin_op::BATCH_MATMUL,
            inputs: vec![0, dequantized],
            outputs: vec![contracted],
            options: OpOptions::None,
        },
    ];

    // The last operator writes the tensor the kernel's output buffer is named
    // after, so that whichever of the two it is, the graph's output keeps the
    // kernel's own name.
    let scaled = tensors.len() as i32;
    tensors.push(TensorInfo::input(
        if bias.is_some() {
            format!("{}_unbiased", output.name)
        } else {
            output.name.clone()
        },
        output.elem_type,
        result.clone(),
    ));
    ops.push(OpDesc {
        opcode: builtin_op::MUL,
        inputs: vec![contracted, 2],
        outputs: vec![scaled],
        options: OpOptions::None,
    });
    let mut last = scaled;
    if bias.is_some() {
        last = tensors.len() as i32;
        tensors.push(TensorInfo::input(
            output.name.clone(),
            output.elem_type,
            result,
        ));
        ops.push(OpDesc {
            opcode: builtin_op::ADD,
            inputs: vec![scaled, 3],
            outputs: vec![last],
            options: OpOptions::None,
        });
    }

    GraphDesc {
        tensors,
        ops,
        graph_inputs,
        graph_outputs: vec![last],
        graph_name: format!("quantized_matmul_{}x{}x{}", shape.m, shape.n, shape.k),
    }
}

/// Build a [`GraphDesc`] for a single [`KernelPattern`].
///
/// Handles the patterns that can be represented as a simple 1-op (or
/// already-multi-op for Normalization/Attention) subgraph.  Returns an error
/// for patterns that are better handled by the specialised builders (e.g.
/// Attention), causing the caller to fall back to [`build_model`].
fn collect_single_graph(pattern: &KernelPattern, extent: i32) -> Result<GraphDesc, BackendError> {
    match pattern {
        KernelPattern::MatMul {
            inputs,
            output,
            shape,
        } => {
            let shape_2d = vec![-1i32, -1];
            Ok(GraphDesc {
                tensors: vec![
                    TensorInfo::input(
                        inputs[0].name.clone(),
                        inputs[0].elem_type,
                        shape_2d.clone(),
                    ),
                    TensorInfo::input(
                        inputs[1].name.clone(),
                        inputs[1].elem_type,
                        shape_2d.clone(),
                    ),
                    TensorInfo::input(output.name.clone(), output.elem_type, shape_2d),
                ],
                ops: vec![OpDesc {
                    opcode: builtin_op::BATCH_MATMUL,
                    inputs: vec![0, 1],
                    outputs: vec![2],
                    options: OpOptions::None,
                }],
                graph_inputs: vec![0, 1],
                graph_outputs: vec![2],
                graph_name: format!("matmul_{}x{}x{}", shape.m, shape.n, shape.k),
            })
        }
        KernelPattern::QuantizedMatMul {
            input,
            weight,
            scale,
            bias,
            output,
            shape,
        } => Ok(collect_quantized_matmul_graph(
            input,
            weight,
            scale,
            bias.as_ref(),
            output,
            shape,
        )),
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
                    TensorInfo::input(
                        inputs[0].name.clone(),
                        inputs[0].elem_type,
                        shape_1d.clone(),
                    ),
                    TensorInfo::input(
                        inputs[1].name.clone(),
                        inputs[1].elem_type,
                        shape_1d.clone(),
                    ),
                    TensorInfo::input(output.name.clone(), output.elem_type, shape_1d),
                ],
                ops: vec![OpDesc {
                    opcode,
                    inputs: vec![0, 1],
                    outputs: vec![2],
                    options: OpOptions::None,
                }],
                graph_inputs: vec![0, 1],
                graph_outputs: vec![2],
                graph_name: format!("{}_1d", op.op_name().to_lowercase()),
            })
        }
        // The same graph the standalone builder emits, rather than a second
        // one written next to it. This arm used to omit the bias TFLite
        // requires, omit the options entirely -- a stride of 0 to the schema --
        // and give every tensor the same shape.
        KernelPattern::Conv2D {
            input,
            weight,
            output,
            shape,
            bias,
            activation,
        } => Ok(conv2d_graph(
            input,
            weight,
            bias.as_ref(),
            output,
            shape,
            *activation,
            extent,
        )),
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
            Ok(pool_graph(input, output, opcode, shape, "pool", extent))
        }
        KernelPattern::Activation {
            op, input, output, ..
        } => {
            let opcode = match op {
                ActivationOp::Relu => builtin_op::RELU,
                ActivationOp::Sigmoid => builtin_op::LOGISTIC,
                ActivationOp::Tanh => builtin_op::TANH,
                ActivationOp::Softmax => builtin_op::SOFTMAX,
                // GELU is builtin 150. It used to go out as CUSTOM with no
                // custom_code, which no interpreter can register; Silu and
                // Mish have no builtin and stay custom.
                ActivationOp::Gelu => builtin_op::GELU,
                ActivationOp::Silu | ActivationOp::Mish => builtin_op::CUSTOM,
            };
            let shape_1d = vec![-1i32];
            Ok(GraphDesc {
                tensors: vec![
                    TensorInfo::input(input.name.clone(), input.elem_type, shape_1d.clone()),
                    TensorInfo::input(output.name.clone(), output.elem_type, shape_1d),
                ],
                ops: vec![OpDesc {
                    opcode,
                    inputs: vec![0],
                    outputs: vec![1],
                    options: OpOptions::None,
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
                    TensorInfo::input(input.name.clone(), input.elem_type, vec![-1, -1]),
                    TensorInfo::input(output.name.clone(), output.elem_type, vec![-1]),
                ],
                ops: vec![OpDesc {
                    opcode,
                    inputs: vec![0],
                    outputs: vec![1],
                    options: OpOptions::None,
                }],
                graph_inputs: vec![0],
                graph_outputs: vec![1],
                graph_name: format!("{}_reduce", op.op_name().to_lowercase()),
            })
        }
        KernelPattern::Transpose { input, output, .. } => Ok(GraphDesc {
            tensors: vec![
                TensorInfo::input(input.name.clone(), input.elem_type, vec![-1, -1]),
                TensorInfo::input(output.name.clone(), output.elem_type, vec![-1, -1]),
            ],
            ops: vec![OpDesc {
                opcode: builtin_op::TRANSPOSE,
                inputs: vec![0],
                outputs: vec![1],
                options: OpOptions::None,
            }],
            graph_inputs: vec![0],
            graph_outputs: vec![1],
            graph_name: "transpose".into(),
        }),
        KernelPattern::Reshape { input, output, .. } => Ok(GraphDesc {
            tensors: vec![
                TensorInfo::input(input.name.clone(), input.elem_type, vec![-1]),
                TensorInfo::input(output.name.clone(), output.elem_type, vec![-1]),
            ],
            ops: vec![OpDesc {
                opcode: builtin_op::RESHAPE,
                inputs: vec![0],
                outputs: vec![1],
                options: OpOptions::None,
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
                    TensorInfo::input(input.name.clone(), input.elem_type, shape_4d.clone()), // 0
                    TensorInfo::input(scale.name.clone(), scale.elem_type, shape_1d.clone()), // 1
                    TensorInfo::input(bias.name.clone(), bias.elem_type, shape_1d),           // 2
                    TensorInfo::input("batchnorm_mul", input.elem_type, shape_4d.clone()),    // 3
                    TensorInfo::input(output.name.clone(), output.elem_type, shape_4d),       // 4
                ],
                ops: vec![
                    OpDesc {
                        opcode: builtin_op::MUL,
                        inputs: vec![0, 1],
                        outputs: vec![3],
                        options: OpOptions::None,
                    },
                    OpDesc {
                        opcode: builtin_op::ADD,
                        inputs: vec![3, 2],
                        outputs: vec![4],
                        options: OpOptions::None,
                    },
                ],
                graph_inputs: vec![0, 1, 2],
                graph_outputs: vec![4],
                graph_name: "batchnorm".into(),
            })
        }
        KernelPattern::Concat {
            inputs,
            output,
            axis,
        } => {
            let _ = axis; // axis is conveyed via ConcatenationOptions in build_model path
            let shape_1d = vec![-1i32];
            let mut tensors: Vec<TensorInfo> = inputs
                .iter()
                .map(|t| TensorInfo::input(t.name.clone(), t.elem_type, shape_1d.clone()))
                .collect();
            tensors.push(TensorInfo::input(
                output.name.clone(),
                output.elem_type,
                shape_1d,
            ));
            let n = inputs.len() as i32;
            let input_indices: Vec<i32> = (0..n).collect();
            Ok(GraphDesc {
                graph_inputs: input_indices.clone(),
                graph_outputs: vec![n],
                ops: vec![OpDesc {
                    opcode: builtin_op::CONCATENATION,
                    inputs: input_indices,
                    outputs: vec![n],
                    options: OpOptions::None,
                }],
                tensors,
                graph_name: "concat".into(),
            })
        }
        KernelPattern::Gather {
            data,
            indices,
            output,
            ..
        } => {
            let shape_1d = vec![-1i32];
            Ok(GraphDesc {
                tensors: vec![
                    TensorInfo::input(data.name.clone(), data.elem_type, shape_1d.clone()),
                    // TFLite refuses UINT32 positions. WGSL indexes with u32,
                    // and for a non-negative index the bytes are the same, so
                    // the declared type is what has to change.
                    TensorInfo::input(indices.name.clone(), data_type::INT32, shape_1d.clone()),
                    TensorInfo::input(output.name.clone(), output.elem_type, shape_1d),
                ],
                ops: vec![OpDesc {
                    opcode: builtin_op::GATHER,
                    inputs: vec![0, 1],
                    outputs: vec![2],
                    options: OpOptions::None,
                }],
                graph_inputs: vec![0, 1],
                graph_outputs: vec![2],
                graph_name: "gather".into(),
            })
        }
        KernelPattern::Scatter {
            data,
            indices,
            updates,
            output,
            ..
        } => {
            let shape_1d = vec![-1i32];
            Ok(GraphDesc {
                tensors: vec![
                    TensorInfo::input(data.name.clone(), data.elem_type, shape_1d.clone()),
                    TensorInfo::input(indices.name.clone(), indices.elem_type, shape_1d.clone()),
                    TensorInfo::input(updates.name.clone(), updates.elem_type, shape_1d.clone()),
                    TensorInfo::input(output.name.clone(), output.elem_type, shape_1d),
                ],
                ops: vec![OpDesc {
                    opcode: builtin_op::SCATTER_ND,
                    inputs: vec![0, 1, 2],
                    outputs: vec![3],
                    options: OpOptions::None,
                }],
                graph_inputs: vec![0, 1, 2],
                graph_outputs: vec![3],
                graph_name: "scatter_nd".into(),
            })
        }
        KernelPattern::ElementWiseChain {
            base,
            cast,
            steps,
            output,
            ..
        } => collect_elementwise_chain_graph(base, *cast, steps, output),
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
    let act_tensor = TensorInfo::input(
        format!("{}_act", old_out.name),
        old_out.elem_type,
        old_out.shape.clone(),
    );
    let act_tensor_idx = desc.tensors.len() as i32;
    desc.tensors.push(act_tensor);
    desc.ops.push(OpDesc {
        opcode: act_opcode,
        inputs: vec![old_out_idx],
        outputs: vec![act_tensor_idx],
        options: OpOptions::None,
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
pub fn build_fused_model(fp: &FusedPattern, extent: i32) -> Result<Vec<u8>, BackendError> {
    match fp {
        FusedPattern::Single(p) => build_model(p, extent),
        FusedPattern::ConvBatchNorm { conv, norm } => {
            let desc = collect_conv_batchnorm_graph(conv, norm, extent)?;
            Ok(build_from_graph_desc(&desc, extent))
        }
        FusedPattern::MatMulBias { matmul, bias_add } => {
            let desc = collect_matmul_bias_graph(matmul, bias_add)?;
            Ok(build_from_graph_desc(&desc, extent))
        }
        FusedPattern::WithActivation {
            base, activation, ..
        } => {
            // `activation_opcode` returns None for exactly one variant, and
            // it is the one this guard used to test for separately. Two places
            // deciding "there is no activation to append" is one place too
            // many: the second arm was unreachable, and an unreachable arm is
            // a rule that cannot be checked.
            let act_opcode = match activation_opcode(activation) {
                Some(c) => c,
                None => return build_fused_model(base, extent),
            };

            // Collect the base graph descriptor, then append the activation op.
            let mut desc = match base.as_ref() {
                FusedPattern::Single(p) => match collect_single_graph(p, extent) {
                    Ok(d) => d,
                    // Fall back to build_model for complex single patterns
                    // (Attention, Split) and just return it without activation.
                    Err(_) => return build_model(p, extent),
                },
                FusedPattern::ConvBatchNorm { conv, norm } => {
                    collect_conv_batchnorm_graph(conv, norm, extent)?
                }
                FusedPattern::MatMulBias { matmul, bias_add } => {
                    collect_matmul_bias_graph(matmul, bias_add)?
                }
                FusedPattern::WithActivation { .. } => {
                    // Nested WithActivation should not occur in practice.
                    return build_fused_model(base, extent);
                }
            };

            append_activation(&mut desc, activation, act_opcode);
            Ok(build_from_graph_desc(&desc, extent))
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
    extent: i32,
) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::with_capacity(1024);

    // Strings
    let names: Vec<_> = inputs.iter().map(|i| fbb.create_string(&i.name)).collect();
    let name_out = fbb.create_string(&output.name);
    let desc = fbb.create_string("nxpu");
    let sg_name = fbb.create_string(graph_name);

    // Shape vectors
    let shape_vecs: Vec<_> = shapes
        .iter()
        .map(|s| shape_vector(&mut fbb, s, extent))
        .collect();

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
    extent: i32,
) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::with_capacity(1024);

    let name_in = fbb.create_string(&input.name);
    let name_out = fbb.create_string(&output.name);
    let desc = fbb.create_string("nxpu");
    let sg_name = fbb.create_string(graph_name);

    let shape_in = shape_vector(&mut fbb, in_shape, extent);
    let shape_out = shape_vector(&mut fbb, out_shape, extent);

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
/// Build a reduction: `(input, axes) -> output`.
///
/// The parameter count follows `build_tflite_unary`, which this is a variant
/// of; the same allow is used elsewhere in this file for the same reason.
///
/// Separate from [`build_tflite_unary`] because TFLite's reducers are not
/// unary — the axes arrive as a second input tensor, and a SUM emitted with
/// one input is rejected before anything runs.
#[allow(clippy::too_many_arguments)]
fn build_tflite_reduce(
    input: &TensorBinding,
    output: &TensorBinding,
    in_shape: &[i32],
    out_shape: &[i32],
    opcode: i32,
    graph_name: &str,
    axis: i64,
    extent: i32,
) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::with_capacity(1024);

    let name_in = fbb.create_string(&input.name);
    let name_out = fbb.create_string(&output.name);
    let desc = fbb.create_string("nxpu");
    let sg_name = fbb.create_string(graph_name);

    let shape_in = shape_vector(&mut fbb, in_shape, extent);
    let shape_out = shape_vector(&mut fbb, out_shape, extent);
    // The axes tensor: one i32, held in a constant buffer.
    let shape_axes = shape_vector(&mut fbb, &[1i32], extent);
    let name_axes = fbb.create_string("reduce_axes");
    let axes_data = fbb.create_vector(&(axis as i32).to_le_bytes());

    // TFLite's reducers take (input, axes); a single-input SUM is rejected
    // with `NumInputs(node) != 2`. The axes tensor is a constant, so it is not
    // a graph input.
    let op_inputs = fbb.create_vector(&[0i32, 1]);
    let op_outputs = fbb.create_vector(&[2i32]);
    let sg_inputs = fbb.create_vector(&[0i32]);
    let sg_outputs = fbb.create_vector(&[2i32]);

    // 3 buffers: sentinel + input + output
    let mut buffer_offsets = Vec::new();
    for i in 0..4 {
        let start = fbb.start_table();
        // Buffer 2 holds the axes constant.
        if i == 2 {
            fbb.push_slot_always(vt::buffer::DATA, axes_data);
        }
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
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 3, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_out);
        fbb.end_table(start)
    };
    let tensor_axes = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_axes);
        fbb.push_slot::<i8>(vt::tensor::TYPE, tensor_type::INT32, 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 2, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_axes);
        fbb.end_table(start)
    };
    let tensors = fbb.create_vector(&[tensor_in, tensor_axes, tensor_out]);

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
/// Build a scatter: `(indices, updates, shape) -> output`.
///
/// SCATTER_ND builds its output rather than editing an existing tensor, so it
/// takes a shape and no base. The kernels this comes from never read their
/// `data` binding either — `output[indices[i]] = updates[i]` — so nothing is
/// lost by leaving it out.
/// Build a transpose: `(input, perm) -> output`.
///
/// The permutation is an operand, not an attribute — a TRANSPOSE emitted with
/// one input is rejected with `NumInputs(node) != 2`, which is how the first
/// kernel this backend ever recognised as a Transpose turned out to be
/// unloadable.
fn build_tflite_transpose(
    input: &TensorBinding,
    output: &TensorBinding,
    in_shape: &[i32],
    out_shape: &[i32],
    perm: &[i64],
    extent: i32,
) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::with_capacity(1024);

    let name_in = fbb.create_string(&input.name);
    let name_out = fbb.create_string(&output.name);
    let desc = fbb.create_string("nxpu");
    let sg_name = fbb.create_string("transpose");

    let shape_in = shape_vector(&mut fbb, in_shape, extent);
    let shape_out = shape_vector(&mut fbb, out_shape, extent);
    // The permutation, as an i32 constant. TRANSPOSE takes it as a second
    // input; without it the kernel is rejected with `NumInputs(node) != 2`.
    let shape_perm = shape_vector(&mut fbb, &[perm.len() as i32], extent);
    let name_perm = fbb.create_string("perm");
    let perm_bytes: Vec<u8> = perm
        .iter()
        .flat_map(|p| (*p as i32).to_le_bytes())
        .collect();
    let perm_data = fbb.create_vector(&perm_bytes);

    // (input, perm) -> output. The permutation is a constant, so the graph's
    // only input is the tensor being transposed.
    let op_inputs = fbb.create_vector(&[0i32, 1]);
    let op_outputs = fbb.create_vector(&[2i32]);
    let sg_inputs = fbb.create_vector(&[0i32]);
    let sg_outputs = fbb.create_vector(&[2i32]);

    // 3 buffers: sentinel + input + output
    let mut buffer_offsets = Vec::new();
    for i in 0..4 {
        let start = fbb.start_table();
        // Buffer 2 holds the permutation.
        if i == 2 {
            fbb.push_slot_always(vt::buffer::DATA, perm_data);
        }
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
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 3, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_out);
        fbb.end_table(start)
    };
    let tensor_perm = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_perm);
        fbb.push_slot::<i8>(vt::tensor::TYPE, tensor_type::INT32, 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 2, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_perm);
        fbb.end_table(start)
    };
    let tensors = fbb.create_vector(&[tensor_in, tensor_perm, tensor_out]);

    let opcode = builtin_op::TRANSPOSE;
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
/// Build a scatter: `(indices, updates, shape) -> output`.
///
/// SCATTER_ND builds its output rather than editing an existing tensor, so it
/// takes a shape and no base. The kernels this comes from never read their
/// `data` binding either — `output[indices[i]] = updates[i]` — so nothing is
/// lost by leaving it out.
fn build_tflite_scatter(
    input: &TensorBinding,
    output: &TensorBinding,
    in_shape: &[i32],
    out_shape: &[i32],
    updates: &TensorBinding,
    extent: i32,
) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::with_capacity(1024);

    let name_in = fbb.create_string(&input.name);
    let name_updates = fbb.create_string(&updates.name);
    let name_out = fbb.create_string(&output.name);
    let desc = fbb.create_string("nxpu");
    let sg_name = fbb.create_string("scatter_nd");

    let shape_in = shape_vector(&mut fbb, in_shape, extent);
    let shape_out = shape_vector(&mut fbb, out_shape, extent);
    // The shape operand: SCATTER_ND builds its output rather than editing
    // one, so it has to be told how big that output is. One i32, because
    // everything this backend emits is rank-1 at the extent it was given.
    let shape_shape = shape_vector(&mut fbb, &[1i32], extent);
    let name_shape = fbb.create_string("scatter_shape");
    let shape_data = fbb.create_vector(&extent.max(1).to_le_bytes());

    // (indices, updates, shape) -> output. The shape is a constant, so the
    // graph's inputs are the first two.
    let op_inputs = fbb.create_vector(&[0i32, 1, 2]);
    let op_outputs = fbb.create_vector(&[3i32]);
    let sg_inputs = fbb.create_vector(&[0i32, 1]);
    let sg_outputs = fbb.create_vector(&[3i32]);

    // 3 buffers: sentinel + input + output
    let mut buffer_offsets = Vec::new();
    for i in 0..5 {
        let start = fbb.start_table();
        // Buffer 3 holds the shape constant.
        if i == 3 {
            fbb.push_slot_always(vt::buffer::DATA, shape_data);
        }
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
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 4, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_out);
        fbb.end_table(start)
    };
    let tensor_updates = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_out);
        fbb.push_slot::<i8>(vt::tensor::TYPE, onnx_to_tflite_type(updates.elem_type), 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 2, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_updates);
        fbb.end_table(start)
    };
    let tensor_shape = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_shape);
        fbb.push_slot::<i8>(vt::tensor::TYPE, tensor_type::INT32, 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 3, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_shape);
        fbb.end_table(start)
    };
    let tensors = fbb.create_vector(&[tensor_in, tensor_updates, tensor_shape, tensor_out]);

    let opcode = builtin_op::SCATTER_ND;
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
    extent: i32,
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
    let shape_nd = shape_vector(&mut fbb, &[-1i32, -1, -1, -1], extent);
    let shape_1d = shape_vector(&mut fbb, &[-1i32], extent);

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
fn build_tflite_softmax(input: &TensorBinding, output: &TensorBinding, extent: i32) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::with_capacity(1024);

    let name_in = fbb.create_string(&input.name);
    let name_out = fbb.create_string(&output.name);
    let desc = fbb.create_string("nxpu");
    let sg_name = fbb.create_string("softmax_1d");

    let shape_in = shape_vector(&mut fbb, &[-1i32], extent);
    let shape_out = shape_vector(&mut fbb, &[-1i32], extent);

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
/// A convolution, with an explicit PAD in front of it when the source kernel
/// pads by an amount TFLite's two modes cannot express.
///
/// TFLite offers SAME and VALID and nothing else. A kernel that reads no
/// further than `in - k + 1` is VALID; one that offsets its reads by a literal
/// is padding by that literal, and SAME is only the same thing by coincidence.
/// `conv2d_5x5.wgsl` pads by 1 with a 5x5 window at stride 2, where its own
/// arithmetic gives 31 output pixels, SAME gives 32 and VALID gives 30 -- so
/// neither mode reproduces it, and emitting the nearer one would be a model
/// that runs and computes something else. A PAD operator says exactly what the
/// kernel says, and the convolution after it is VALID.
#[allow(clippy::too_many_arguments)]
fn build_tflite_conv2d(
    input: &TensorBinding,
    weight: &TensorBinding,
    bias: Option<&TensorBinding>,
    output: &TensorBinding,
    shape: &Conv2DShape,
    activation: Option<ActivationOp>,
    extent: i32,
) -> Vec<u8> {
    build_from_graph_desc(
        &conv2d_graph(input, weight, bias, output, shape, activation, extent),
        extent,
    )
}

/// The graph a convolution becomes, so that the fused path and the standalone
/// one cannot disagree about it.
///
/// They did. `collect_single_graph` built its own CONV_2D with two inputs, no
/// options and one shape vector for every tensor -- so a convolution followed
/// by an activation, which is the shape most real ones have, went out with a
/// stride of 0 and a missing bias and was refused by TFLite's own kernel.
#[allow(clippy::too_many_arguments)]
fn conv2d_graph(
    input: &TensorBinding,
    weight: &TensorBinding,
    bias: Option<&TensorBinding>,
    output: &TensorBinding,
    shape: &Conv2DShape,
    activation: Option<ActivationOp>,
    extent: i32,
) -> GraphDesc {
    // TFLite carries an activation inside the convolution's own options, and
    // has a form for exactly these. Sigmoid and the multiply-shaped ones
    // (GELU, SiLU, Mish) have none, and the classifier does not report them
    // here for that reason -- see `store_value_activation`.
    let fused_activation = match activation {
        None => activation_function::NONE,
        Some(ActivationOp::Relu) => activation_function::RELU,
        Some(ActivationOp::Tanh) => activation_function::TANH,
        Some(_) => activation_function::NONE,
    };
    // A convolution's tensors do not share a shape, and one vector used to be
    // written to all of them. At `--symbolic-dim 64` every tensor came out
    // [64, 64, 64, 64]: a 64x64 window with 64 channels each way over a 64x64
    // image, an im2col of 2^36 elements, past TFLite's 32-bit limit before any
    // delegate is asked. TFLite reads them as NHWC with the weight as
    // [out, kh, kw, in].
    let n = extent.max(1);
    let channels_in = extent.max(1);
    let channels_out = extent.max(1);
    let in_h = extent.max(1);
    let in_w = extent.max(1);

    // A window the kernel states as a literal is known exactly. One supplied
    // through the params struct is not, and falls back to the extent like any
    // other symbolic dimension -- which makes the window as wide as the image,
    // so the output is a single pixel. Degenerate, but coherent, and coherence
    // is what decides whether anything will load it.
    let window = |literal: i64| {
        if literal > 0 {
            (literal as i32).max(1)
        } else {
            extent.max(1)
        }
    };
    let kernel_h = window(shape.kernel_h_val);
    let kernel_w = window(shape.kernel_w_val);

    let pad_h = shape.pad_h.max(0) as i32;
    let pad_w = shape.pad_w.max(0) as i32;
    let padded_h = in_h + 2 * pad_h;
    let padded_w = in_w + 2 * pad_w;

    // floor((in - reach) / stride) + 1, which is what TFLite's own kernel
    // computes for VALID, over the padded extent because that is what the
    // convolution now sees.
    let valid_out = |input: i32, k: i32, stride: i32, dilation: i32| {
        let reach = dilation.max(1) * (k - 1) + 1;
        ((input - reach) / stride.max(1) + 1).max(1)
    };
    let out_h = valid_out(
        padded_h,
        kernel_h,
        shape.stride_h as i32,
        shape.dilation_h as i32,
    );
    let out_w = valid_out(
        padded_w,
        kernel_w,
        shape.stride_w as i32,
        shape.dilation_w as i32,
    );

    let mut tensors = vec![TensorInfo::input(
        input.name.clone(),
        input.elem_type,
        vec![n, in_h, in_w, channels_in],
    )];
    let mut ops = Vec::new();

    // The tensor the convolution reads: the input itself, or the padded copy.
    let conv_input = if pad_h > 0 || pad_w > 0 {
        // PAD takes the amounts as an int32 [rank, 2] constant, before and
        // after each dimension. Batch and channels are untouched.
        let amounts: [i32; 8] = [0, 0, pad_h, pad_h, pad_w, pad_w, 0, 0];
        tensors.push(TensorInfo::constant(
            format!("{}_paddings", input.name),
            data_type::INT32,
            vec![4, 2],
            amounts.iter().flat_map(|d| d.to_le_bytes()).collect(),
        ));
        tensors.push(TensorInfo::input(
            format!("{}_padded", input.name),
            input.elem_type,
            vec![n, padded_h, padded_w, channels_in],
        ));
        ops.push(OpDesc {
            opcode: builtin_op::PAD,
            inputs: vec![0, 1],
            outputs: vec![2],
            options: OpOptions::Pad,
        });
        2
    } else {
        0
    };

    let weight_index = tensors.len() as i32;
    tensors.push(TensorInfo::input(
        weight.name.clone(),
        weight.elem_type,
        vec![channels_out, kernel_h, kernel_w, channels_in],
    ));

    // TFLite's CONV_2D kernel requires three inputs — `has_bias was not true`
    // is a hard failure, not a fallback — so a convolution whose source has no
    // bias still gets one here, as a constant of zeros. A synthesised bias is
    // f32 whatever the operands are; TFLite requires it to match the
    // accumulator.
    let bias_index = tensors.len() as i32;
    match bias {
        Some(b) => tensors.push(TensorInfo::input(
            b.name.clone(),
            b.elem_type,
            vec![channels_out],
        )),
        None => tensors.push(TensorInfo::constant(
            "bias",
            data_type::FLOAT,
            vec![channels_out],
            vec![0u8; (channels_out as usize) * 4],
        )),
    }

    let output_index = tensors.len() as i32;
    tensors.push(TensorInfo::input(
        output.name.clone(),
        output.elem_type,
        vec![n, out_h, out_w, channels_out],
    ));

    ops.push(OpDesc {
        opcode: builtin_op::CONV_2D,
        inputs: vec![conv_input, weight_index, bias_index],
        outputs: vec![output_index],
        options: OpOptions::Conv2D {
            // VALID, because the padding is now the PAD operator's business.
            // `Padding`'s first member is SAME, so VALID is 1 and not 0; both
            // builders here pushed 0 under a comment saying VALID, and
            // `push_slot` then omitted the field for matching its own default,
            // so the mistake was not in the bytes to find.
            padding: padding::VALID,
            stride_w: shape.stride_w as i32,
            stride_h: shape.stride_h as i32,
            dilation_w: shape.dilation_w as i32,
            dilation_h: shape.dilation_h as i32,
            activation: fused_activation,
        },
    });

    // A constant is not a graph input; the synthesised bias and the paddings
    // are constants, and listing either here makes the model invalid.
    let mut graph_inputs = vec![0, weight_index];
    if bias.is_some() {
        graph_inputs.push(bias_index);
    }

    GraphDesc {
        tensors,
        ops,
        graph_inputs,
        graph_outputs: vec![output_index],
        graph_name: "conv2d".into(),
    }
}

fn build_tflite_pool(
    input: &TensorBinding,
    output: &TensorBinding,
    opcode: i32,
    shape: &PoolShape,
    graph_name: &str,
    extent: i32,
) -> Vec<u8> {
    build_from_graph_desc(
        &pool_graph(input, output, opcode, shape, graph_name, extent),
        extent,
    )
}

/// The graph a pool becomes. Shared with the fused path for the same reason
/// the convolution's is: that path built a POOL_2D with no options at all,
/// which is a stride of 0 and a window of 0 to TFLite's schema.
fn pool_graph(
    input: &TensorBinding,
    output: &TensorBinding,
    opcode: i32,
    shape: &PoolShape,
    graph_name: &str,
    extent: i32,
) -> GraphDesc {
    // Input and output do not have the same shape, and one vector was written
    // to both. A pool with a window narrows its input; saying otherwise is the
    // defect the convolution builder had, and it survived here only because
    // the padding was wrong in the direction that hid it.
    //
    // `PoolShape` carries no padding, so these are VALID and sized for VALID:
    // floor((in - k) / stride) + 1, which is what TFLite's own kernel computes.
    // Writing it as (in - k + 1) / stride is a different function that agrees
    // only at stride 1.
    let n = extent.max(1);
    let channels = extent.max(1);
    let in_h = extent.max(1);
    let in_w = extent.max(1);
    let valid_out =
        |input: i32, k: i32, stride: i32| ((input - k.max(1)) / stride.max(1) + 1).max(1);
    let out_h = valid_out(in_h, shape.kernel_h as i32, shape.stride_h as i32);
    let out_w = valid_out(in_w, shape.kernel_w as i32, shape.stride_w as i32);

    GraphDesc {
        tensors: vec![
            TensorInfo::input(
                input.name.clone(),
                input.elem_type,
                vec![n, in_h, in_w, channels],
            ),
            TensorInfo::input(
                output.name.clone(),
                output.elem_type,
                vec![n, out_h, out_w, channels],
            ),
        ],
        ops: vec![OpDesc {
            opcode,
            inputs: vec![0],
            outputs: vec![1],
            options: OpOptions::Pool2D {
                // SAME is the schema default and so was written as absence;
                // the comment here claimed VALID while the byte said SAME.
                padding: padding::VALID,
                stride_w: shape.stride_w as i32,
                stride_h: shape.stride_h as i32,
                filter_w: shape.kernel_w as i32,
                filter_h: shape.kernel_h as i32,
            },
        }],
        graph_inputs: vec![0],
        graph_outputs: vec![1],
        graph_name: graph_name.into(),
    }
}

// The extent joins seven existing parameters; the same allow is already used
// in the ONNX and StableHLO lowerings for the same reason.
#[allow(clippy::too_many_arguments)]
fn build_tflite_attention(
    query: &TensorBinding,
    key: &TensorBinding,
    value: &TensorBinding,
    output: &TensorBinding,
    d_k: &str,
    num_heads: u32,
    causal: bool,
    extent: i32,
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
    let shape_2d = shape_vector(&mut fbb, &[-1i32, -1], extent);
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
        // TFLite resolves an operator as `max(builtin_code, deprecated_builtin_code)`,
        // so 127 here does not mean "read the other field" — it wins, and 127
        // is PLACEHOLDER_FOR_GREATER_OP_CODES. Writing the real code in both
        // is what the reader expects for anything that fits in a byte.
        fbb.push_slot::<i8>(
            vt::operator_code::DEPRECATED_BUILTIN_CODE,
            builtin_op::BATCH_MATMUL as i8,
            0,
        );
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

/// Build a TFLite model for Concatenation with axis embedded via ConcatenationOptions.
fn build_tflite_concat(
    inputs: &[TensorBinding],
    output: &TensorBinding,
    axis: i64,
    extent: i32,
) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::with_capacity(1024);

    let in_names: Vec<_> = inputs.iter().map(|i| fbb.create_string(&i.name)).collect();
    let name_out = fbb.create_string(&output.name);
    let desc = fbb.create_string("nxpu");
    let sg_name = fbb.create_string("concat");

    // Rank has to cover the axis. Concatenating rank-1 tensors on axis 1 is
    // rejected — `axis < t0->dims->size was not true` — and the axis is known
    // right here, so the rank can follow it.
    let dims: Vec<i32> = vec![-1i32; (axis.max(0) as usize) + 1];
    let shape_1d = shape_vector(&mut fbb, &dims, extent);

    let num_tensors = inputs.len() + 1; // inputs + output
    let mut buffer_offsets = Vec::new();
    for _ in 0..=num_tensors {
        let start = fbb.start_table();
        buffer_offsets.push(fbb.end_table(start));
    }
    let buffers = fbb.create_vector(&buffer_offsets);

    // Tensors: inputs(0..N-1), output(N)
    let mut tensor_offsets = Vec::new();
    for (i, (inp, name)) in inputs.iter().zip(in_names.iter()).enumerate() {
        let t = {
            let start = fbb.start_table();
            fbb.push_slot_always(vt::tensor::SHAPE, shape_1d);
            fbb.push_slot::<i8>(vt::tensor::TYPE, onnx_to_tflite_type(inp.elem_type), 0);
            fbb.push_slot::<u32>(vt::tensor::BUFFER, (i + 1) as u32, 0);
            fbb.push_slot_always(vt::tensor::NAME, *name);
            fbb.end_table(start)
        };
        tensor_offsets.push(t);
    }
    let out_tensor = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_1d);
        fbb.push_slot::<i8>(vt::tensor::TYPE, onnx_to_tflite_type(output.elem_type), 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, num_tensors as u32, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_out);
        fbb.end_table(start)
    };
    tensor_offsets.push(out_tensor);
    let tensors = fbb.create_vector(&tensor_offsets);

    let deprecated_code = if builtin_op::CONCATENATION <= 127 {
        builtin_op::CONCATENATION as i8
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
        fbb.push_slot::<i32>(
            vt::operator_code::BUILTIN_CODE,
            builtin_op::CONCATENATION,
            0,
        );
        fbb.end_table(start)
    };
    let operator_codes = fbb.create_vector(&[opcode_table]);

    let input_indices: Vec<i32> = (0..inputs.len() as i32).collect();
    let op_inputs = fbb.create_vector(&input_indices);
    let op_outputs = fbb.create_vector(&[inputs.len() as i32]);

    // ConcatenationOptions table with axis
    let concat_opts = {
        let start = fbb.start_table();
        fbb.push_slot::<i32>(concatenation_options::AXIS, axis as i32, 0);
        fbb.push_slot::<i8>(concatenation_options::FUSED_ACTIVATION_FUNCTION, 0, 0); // NONE
        fbb.end_table(start)
    };

    let operator = {
        let start = fbb.start_table();
        fbb.push_slot::<u32>(vt::operator::OPCODE_INDEX, 0, 0);
        fbb.push_slot_always(vt::operator::INPUTS, op_inputs);
        fbb.push_slot_always(vt::operator::OUTPUTS, op_outputs);
        fbb.push_slot::<u8>(
            vt::operator::BUILTIN_OPTIONS_TYPE,
            builtin_options_type::CONCATENATION,
            0,
        );
        fbb.push_slot_always(vt::operator::BUILTIN_OPTIONS, concat_opts);
        fbb.end_table(start)
    };
    let operators = fbb.create_vector(&[operator]);

    let sg_inputs = fbb.create_vector(&input_indices);
    let sg_outputs = fbb.create_vector(&[inputs.len() as i32]);
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
///
/// In TFLite the Split op expects input 0 = axis (scalar int32 constant)
/// and input 1 = the tensor to split. SplitOptions carries `num_splits`.
fn build_tflite_split(
    input: &TensorBinding,
    outputs: &[TensorBinding],
    axis: i64,
    extent: i32,
) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::with_capacity(1024);

    let name_in = fbb.create_string(&input.name);
    let name_axis = fbb.create_string("split_axis");
    let out_names: Vec<_> = outputs.iter().map(|o| fbb.create_string(&o.name)).collect();
    let desc = fbb.create_string("nxpu");
    let sg_name = fbb.create_string("split");

    // Same as concat: splitting along axis 1 needs tensors that have one.
    let dims: Vec<i32> = vec![-1i32; (axis.max(0) as usize) + 1];
    let shape_in = shape_vector(&mut fbb, &dims, extent);
    let shape_out = shape_vector(&mut fbb, &dims, extent);
    let shape_scalar = fbb.create_vector::<i32>(&[]);

    // Buffer for axis constant: little-endian i32
    let axis_bytes = (axis as i32).to_le_bytes();
    let axis_data = fbb.create_vector(&axis_bytes);

    // num_tensors = 1 (axis) + 1 (input) + outputs
    let num_tensors = 2 + outputs.len();
    let mut buffer_offsets = Vec::new();
    // Buffer 0: sentinel (empty)
    let buf0 = {
        let start = fbb.start_table();
        fbb.end_table(start)
    };
    buffer_offsets.push(buf0);
    // Buffer 1: axis constant data
    let buf1 = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::buffer::DATA, axis_data);
        fbb.end_table(start)
    };
    buffer_offsets.push(buf1);
    // Remaining buffers: empty (for input tensor + output tensors)
    for _ in 0..num_tensors - 1 {
        let start = fbb.start_table();
        buffer_offsets.push(fbb.end_table(start));
    }
    let buffers = fbb.create_vector(&buffer_offsets);

    // Tensors: axis(0), input(1), outputs(2..N+1)
    let mut tensor_offsets = Vec::new();
    // Tensor 0: axis scalar (INT32, buffer 1)
    let tensor_axis = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_scalar);
        fbb.push_slot::<i8>(vt::tensor::TYPE, tensor_type::INT32, 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 1, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_axis);
        fbb.end_table(start)
    };
    tensor_offsets.push(tensor_axis);
    // Tensor 1: input
    let tensor_in = {
        let start = fbb.start_table();
        fbb.push_slot_always(vt::tensor::SHAPE, shape_in);
        fbb.push_slot::<i8>(vt::tensor::TYPE, onnx_to_tflite_type(input.elem_type), 0);
        fbb.push_slot::<u32>(vt::tensor::BUFFER, 2, 0);
        fbb.push_slot_always(vt::tensor::NAME, name_in);
        fbb.end_table(start)
    };
    tensor_offsets.push(tensor_in);

    for (i, (o, name)) in outputs.iter().zip(out_names.iter()).enumerate() {
        let t = {
            let start = fbb.start_table();
            fbb.push_slot_always(vt::tensor::SHAPE, shape_out);
            fbb.push_slot::<i8>(vt::tensor::TYPE, onnx_to_tflite_type(o.elem_type), 0);
            fbb.push_slot::<u32>(vt::tensor::BUFFER, (i + 3) as u32, 0);
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

    // TFLite Split: inputs = [axis_tensor(0), data_tensor(1)], outputs = [2..N+1]
    let op_inputs = fbb.create_vector(&[0i32, 1]);
    let output_indices: Vec<i32> = (2..2 + outputs.len() as i32).collect();
    let op_outputs = fbb.create_vector(&output_indices);

    // SplitOptions with num_splits
    let split_opts = {
        let start = fbb.start_table();
        fbb.push_slot::<i32>(split_options::NUM_SPLITS, outputs.len() as i32, 0);
        fbb.end_table(start)
    };

    let operator = {
        let start = fbb.start_table();
        fbb.push_slot::<u32>(vt::operator::OPCODE_INDEX, 0, 0);
        fbb.push_slot_always(vt::operator::INPUTS, op_inputs);
        fbb.push_slot_always(vt::operator::OUTPUTS, op_outputs);
        fbb.push_slot::<u8>(
            vt::operator::BUILTIN_OPTIONS_TYPE,
            builtin_options_type::SPLIT,
            0,
        );
        fbb.push_slot_always(vt::operator::BUILTIN_OPTIONS, split_opts);
        fbb.end_table(start)
    };
    let operators = fbb.create_vector(&[operator]);

    // Graph input is the data tensor (index 1); axis tensor (index 0) is a constant.
    let sg_inputs = fbb.create_vector(&[1i32]);
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

/// Build a standalone TFLite Transpose model for layout conversion.
///
/// If `perm` is a non-identity permutation, this builds a single-op TFLite
/// model containing a Transpose node.  Returns `None` if the permutation is
/// an identity (no transpose needed) or empty.
///
/// This is used by the TFLite backend when the source IR has a different
/// memory layout (e.g. NCHW) than TFLite's expected NHWC layout.
#[allow(dead_code)] // Infrastructure for layout conversion; wired in by backends as needed.
pub fn build_layout_transpose(
    input: &TensorBinding,
    output: &TensorBinding,
    perm: &[i64],
    extent: i32,
) -> Option<Vec<u8>> {
    // Identity check
    let is_identity = perm.iter().enumerate().all(|(i, &p)| p as usize == i);
    if is_identity || perm.is_empty() {
        return None;
    }

    let ndim = perm.len();
    let in_shape: Vec<i32> = (0..ndim).map(|_| -1i32).collect();
    let out_shape: Vec<i32> = (0..ndim).map(|_| -1i32).collect();

    Some(build_tflite_unary(
        input,
        output,
        &in_shape,
        &out_shape,
        builtin_op::TRANSPOSE,
        "layout_transpose",
        extent,
    ))
}

#[cfg(test)]
mod concrete_shape_tests {
    use super::concrete_shape;

    #[test]
    fn symbolic_dimensions_take_the_extent() {
        assert_eq!(concrete_shape(&[-1, -1], 1024), vec![1024, 1024]);
    }

    #[test]
    fn known_dimensions_are_preserved() {
        // Nothing lowers to this yet — every pattern is fully symbolic — but a
        // shape that knows one of its dimensions must not have it overwritten.
        assert_eq!(concrete_shape(&[-1, 3, -1, 224], 8), vec![8, 3, 8, 224]);
    }

    #[test]
    fn zero_is_a_known_dimension_not_a_symbolic_one() {
        // Only a negative value means "unknown"; 0 is a real, if useless,
        // extent and must survive rather than being substituted.
        assert_eq!(concrete_shape(&[0, -1], 4), vec![0, 4]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nxpu_analysis::analyze::data_type;
    use nxpu_analysis::analyze::{
        ActivationOp, Conv2DShape, MatMulShape, NormType, PoolKind, PoolShape, ReduceOp, TensorRole,
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
        let bytes = build_model(&pattern, 1).unwrap();
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
        let bytes = build_model(&pattern, 1).unwrap();
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
            let bytes = build_model(&pattern, 1).unwrap();
            assert_eq!(&bytes[4..8], b"TFL3", "failed for {:?}", op);
        }
    }

    /// Does the model contain this shape vector?
    ///
    /// A flatbuffer vector of i32 is a length followed by the elements, all
    /// little-endian, so a shape is a findable byte sequence. Low-level, but
    /// the alternative is asserting the file starts with `TFL3`, and that is
    /// what let a convolution give its input, its weight and its output the
    /// same shape for as long as this backend has existed.
    fn has_shape(bytes: &[u8], dims: &[i32]) -> bool {
        let mut needle = (dims.len() as i32).to_le_bytes().to_vec();
        for d in dims {
            needle.extend_from_slice(&d.to_le_bytes());
        }
        bytes.windows(needle.len()).any(|w| w == needle)
    }

    #[test]
    fn a_convolution_gives_each_tensor_its_own_shape() {
        // NHWC input, [out, kh, kw, in] weight, [out] bias, NHWC output. At an
        // extent of 8 with a 5x5 window and no padding the output is 4x4, and
        // none of the four is the same vector as another.
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
                kernel_h_val: 5,
                kernel_w_val: 5,
                stride_h: 1,
                stride_w: 1,
                pad_h: 0,
                pad_w: 0,
                groups: 1,
                dilation_h: 1,
                dilation_w: 1,
            },
            bias: None,
            activation: None,
        };
        let bytes = build_model(&pattern, 8).unwrap();
        assert!(has_shape(&bytes, &[8, 8, 8, 8]), "no input shape");
        assert!(
            has_shape(&bytes, &[8, 5, 5, 8]),
            "the weight is not [out, kh, kw, in]"
        );
        assert!(has_shape(&bytes, &[8]), "no per-output-channel bias shape");
        // The output shape is the assertion that catches a SAME padding
        // written under a comment claiming VALID: SAME would leave it at 8x8.
        assert!(
            has_shape(&bytes, &[8, 4, 4, 8]),
            "the output is not sized for VALID"
        );
    }

    #[test]
    fn a_convolution_whose_window_is_symbolic_still_narrows_its_output() {
        // A window supplied through the params struct is unknown, so it takes
        // the extent like any other symbolic dimension -- which makes it as
        // wide as the image and the output a single pixel. Degenerate, but
        // coherent; the incoherent version needed an im2col of 2^36 at an
        // extent of 64 and no interpreter would load it.
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
                kernel_h_val: 0,
                kernel_w_val: 0,
                stride_h: 1,
                stride_w: 1,
                pad_h: 0,
                pad_w: 0,
                groups: 1,
                dilation_h: 1,
                dilation_w: 1,
            },
            bias: None,
            activation: None,
        };
        let bytes = build_model(&pattern, 16).unwrap();
        assert!(has_shape(&bytes, &[16, 16, 16, 16]), "no input shape");
        assert!(
            has_shape(&bytes, &[16, 1, 1, 16]),
            "the output is not a single pixel"
        );
    }

    #[test]
    fn a_pool_narrows_its_input() {
        // The same defect as the convolution's, in the builder next door: one
        // shape vector written to the input and the output alike.
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
        let bytes = build_model(&pattern, 8).unwrap();
        assert!(has_shape(&bytes, &[8, 8, 8, 8]), "no input shape");
        assert!(
            has_shape(&bytes, &[8, 4, 4, 8]),
            "a 2x2 window at stride 2 over 8 pixels is not 8 pixels out"
        );
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
                groups: 1,
                dilation_h: 1,
                dilation_w: 1,
            },
            bias: None,
            activation: None,
        };
        let bytes = build_model(&pattern, 1).unwrap();
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
            let bytes = build_model(&pattern, 1).unwrap();
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
            let bytes = build_model(&pattern, 1).unwrap();
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
        let bytes = build_model(&pattern, 1).unwrap();
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
                groups: 1,
                dilation_h: 1,
                dilation_w: 1,
            },
            bias: None,
            activation: None,
        }
    }

    fn make_normalization(input_name: &str, output_name: &str) -> KernelPattern {
        KernelPattern::Normalization {
            input: make_tensor(input_name, TensorRole::Input),
            scale: make_tensor("gamma", TensorRole::Input),
            bias: make_tensor("beta", TensorRole::Input),
            output: make_tensor(output_name, TensorRole::Output),
            epsilon: 1e-5,
            norm_type: NormType::Batch,
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

        let bytes = build_fused_model(&fused, 1).unwrap();
        assert!(bytes.len() > 8);
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    /// The shape a tensor was given, by name, so a test can name the tensor it
    /// means instead of the index it happens to have.
    fn shape_of(desc: &GraphDesc, name: &str) -> Vec<i32> {
        desc.tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("no tensor named {name}"))
            .shape
            .clone()
    }

    #[test]
    fn a_fused_convolution_carries_its_options_its_bias_and_its_own_shapes() {
        // The three defects the standalone convolution had, asserted for the
        // fused one, which kept all three: no `Conv2DOptions` is read as
        // stride 0 and refused with `params->stride_height > 0 was not true`,
        // two inputs is refused with `has_bias was not true`, and one shape
        // vector for every tensor made a 3x3 window an 8x8 one.
        let desc = collect_conv_batchnorm_graph(
            &make_conv2d(),
            &make_normalization("conv_out", "bn_out"),
            8,
        )
        .unwrap();

        let conv = desc
            .ops
            .iter()
            .find(|o| o.opcode == builtin_op::CONV_2D)
            .expect("no convolution in the fused graph");
        assert_eq!(conv.inputs.len(), 3, "the convolution has no bias input");
        assert!(
            matches!(
                conv.options,
                OpOptions::Conv2D {
                    padding: padding::VALID,
                    stride_w: 1,
                    stride_h: 1,
                    dilation_w: 1,
                    dilation_h: 1,
                    // This fixture's convolution stores its accumulator
                    // directly, so there is nothing to fuse.
                    activation: activation_function::NONE,
                }
            ),
            "the convolution has no options table"
        );

        assert_eq!(shape_of(&desc, "x"), vec![8, 8, 8, 8], "the input is NHWC");
        assert_eq!(
            shape_of(&desc, "w"),
            vec![8, 3, 3, 8],
            "the weight is not [out, kh, kw, in]"
        );
        // 8 - 3 + 1 = 6, which is what the convolution computes for VALID and
        // the assertion that separates it from a SAME written under a comment
        // claiming VALID.
        assert_eq!(shape_of(&desc, "conv_out"), vec![8, 6, 6, 8]);
        assert_eq!(shape_of(&desc, "bn_out"), vec![8, 6, 6, 8]);
        // Per output channel, so they broadcast over NHWC. At this extent the
        // channel count and the spatial extent differ, so a scale that took
        // the whole shape would not go unnoticed.
        assert_eq!(shape_of(&desc, "gamma"), vec![8]);
        assert_eq!(shape_of(&desc, "beta"), vec![8]);

        // The synthesised bias is a constant, and a constant is not a graph
        // input; listing it makes the model invalid.
        let bias_index = conv.inputs[2];
        assert!(
            desc.tensors[bias_index as usize].data.is_some(),
            "the synthesised bias has no contents"
        );
        assert!(
            !desc.graph_inputs.contains(&bias_index),
            "the synthesised bias is listed as a graph input"
        );
    }

    #[test]
    fn a_fused_convolution_that_pads_writes_the_pad_and_narrows_its_output() {
        // Padding is the PAD operator's business here, as it is for the
        // standalone convolution, so the output has to be sized for the padded
        // extent rather than the input's: 8 + 2 -> 10, a 3-wide reach at
        // stride 2 leaves 4.
        let mut conv = make_conv2d();
        if let KernelPattern::Conv2D { shape, .. } = &mut conv {
            shape.pad_h = 1;
            shape.pad_w = 1;
            shape.stride_h = 2;
            shape.stride_w = 2;
        }
        let desc =
            collect_conv_batchnorm_graph(&conv, &make_normalization("conv_out", "bn_out"), 8)
                .unwrap();

        assert!(
            desc.ops.iter().any(|o| o.opcode == builtin_op::PAD),
            "the padding was never written"
        );
        assert_eq!(shape_of(&desc, "x_padded"), vec![8, 10, 10, 8]);
        assert_eq!(shape_of(&desc, "conv_out"), vec![8, 4, 4, 8]);
        assert_eq!(shape_of(&desc, "bn_out"), vec![8, 4, 4, 8]);
    }

    #[test]
    fn a_fused_convolutions_own_bias_survives_the_fusion() {
        // The convolution's bias is added before the normalization's scale
        // multiplies, so dropping it computes a different function. It used to
        // be dropped: the fused builder never looked at the field.
        let mut conv = make_conv2d();
        if let KernelPattern::Conv2D { bias, .. } = &mut conv {
            *bias = Some(make_tensor("conv_bias", TensorRole::Input));
        }
        let desc =
            collect_conv_batchnorm_graph(&conv, &make_normalization("conv_out", "bn_out"), 8)
                .unwrap();

        assert_eq!(shape_of(&desc, "conv_bias"), vec![8]);
        let conv_op = desc
            .ops
            .iter()
            .find(|o| o.opcode == builtin_op::CONV_2D)
            .expect("no convolution in the fused graph");
        let bias_index = conv_op.inputs[2];
        assert_eq!(desc.tensors[bias_index as usize].name, "conv_bias");
        assert!(
            desc.graph_inputs.contains(&bias_index),
            "a bias the kernel supplies is a graph input"
        );
    }

    /// Hand the fused model to a real interpreter.
    ///
    /// `TFL3` in the header is what this backend asserted for as long as it
    /// had a fused convolution, and it is true of a model no interpreter will
    /// load. Nothing in this workspace links TFLite, so the bytes go to a file
    /// and the interpreter runs outside it; set `NXPU_TFLITE_DUMP_DIR` and the
    /// models this test builds appear there.
    #[test]
    fn the_fused_models_can_be_handed_to_an_interpreter() {
        use nxpu_analysis::fusion::FusedPattern;

        let mut padded = make_conv2d();
        if let KernelPattern::Conv2D { shape, .. } = &mut padded {
            shape.pad_h = 1;
            shape.pad_w = 1;
            shape.stride_h = 2;
            shape.stride_w = 2;
        }
        let mut biased = make_conv2d();
        if let KernelPattern::Conv2D { bias, .. } = &mut biased {
            *bias = Some(make_tensor("conv_bias", TensorRole::Input));
        }

        let conv_bn = |conv| FusedPattern::ConvBatchNorm {
            conv,
            norm: Box::new(make_normalization("conv_out", "bn_out")),
        };
        let cases = [
            ("conv_batchnorm", conv_bn(make_conv2d())),
            ("conv_batchnorm_padded", conv_bn(padded)),
            ("conv_batchnorm_biased", conv_bn(biased)),
            // The activation is appended onto this graph rather than built
            // with it, so it is the one composition that can go wrong without
            // any of the assertions above noticing.
            (
                "conv_batchnorm_relu",
                FusedPattern::WithActivation {
                    base: Box::new(conv_bn(make_conv2d())),
                    activation: nxpu_analysis::fusion::FusedActivation::Relu,
                    activation_pattern: Box::new(make_activation(
                        ActivationOp::Relu,
                        "bn_out",
                        "relu_out",
                    )),
                },
            ),
        ];
        let dir = std::env::var_os("NXPU_TFLITE_DUMP_DIR");
        for (name, fused) in cases {
            let bytes = build_fused_model(&fused, 8).unwrap();
            if let Some(dir) = &dir {
                std::fs::write(
                    std::path::Path::new(dir).join(format!("{name}.tflite")),
                    &bytes,
                )
                .unwrap();
            }
        }
    }

    #[test]
    fn fused_model_matmul_bias() {
        use nxpu_analysis::fusion::FusedPattern;

        let fused = FusedPattern::MatMulBias {
            matmul: make_matmul(),
            bias_add: Box::new(make_bias_add("mm_out", "out")),
        };

        let bytes = build_fused_model(&fused, 1).unwrap();
        assert!(bytes.len() > 8);
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_model_single_matmul() {
        use nxpu_analysis::fusion::FusedPattern;

        let fused = FusedPattern::Single(make_matmul());
        let bytes = build_fused_model(&fused, 1).unwrap();
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

        let bytes = build_fused_model(&fused, 1).unwrap();
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

        let bytes = build_fused_model(&fused, 1).unwrap();
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

        let bytes = build_fused_model(&fused, 1).unwrap();
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

        let bytes = build_fused_model(&fused, 1).unwrap();
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

        let bytes = build_fused_model(&fused, 1).unwrap();
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

        let bytes = build_fused_model(&fused, 1).unwrap();
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
        let bytes = build_fused_model(&outer, 1).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    // ---- collect_single_graph tests for each pattern ----

    #[test]
    fn fused_single_conv2d() {
        use nxpu_analysis::fusion::FusedPattern;
        let fused = FusedPattern::Single(make_conv2d());
        let bytes = build_fused_model(&fused, 1).unwrap();
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
        let bytes = build_fused_model(&fused, 1).unwrap();
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
        let bytes = build_fused_model(&fused, 1).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_single_activation_relu() {
        use nxpu_analysis::fusion::FusedPattern;
        let fused = FusedPattern::Single(make_activation(ActivationOp::Relu, "x", "y"));
        let bytes = build_fused_model(&fused, 1).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_single_activation_sigmoid() {
        use nxpu_analysis::fusion::FusedPattern;
        let fused = FusedPattern::Single(make_activation(ActivationOp::Sigmoid, "x", "y"));
        let bytes = build_fused_model(&fused, 1).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_single_activation_tanh() {
        use nxpu_analysis::fusion::FusedPattern;
        let fused = FusedPattern::Single(make_activation(ActivationOp::Tanh, "x", "y"));
        let bytes = build_fused_model(&fused, 1).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_single_activation_softmax() {
        use nxpu_analysis::fusion::FusedPattern;
        let fused = FusedPattern::Single(make_activation(ActivationOp::Softmax, "x", "y"));
        let bytes = build_fused_model(&fused, 1).unwrap();
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
            let bytes = build_fused_model(&fused, 1).unwrap();
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
        let bytes = build_fused_model(&fused, 1).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_single_reshape() {
        use nxpu_analysis::fusion::FusedPattern;
        let fused = FusedPattern::Single(KernelPattern::Reshape {
            input: make_tensor("x", TensorRole::Input),
            output: make_tensor("y", TensorRole::Output),
        });
        let bytes = build_fused_model(&fused, 1).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn fused_single_normalization() {
        use nxpu_analysis::fusion::FusedPattern;
        let fused = FusedPattern::Single(make_normalization("x", "y"));
        let bytes = build_fused_model(&fused, 1).unwrap();
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
        let bytes = build_fused_model(&fused, 1).unwrap();
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
            let bytes = build_fused_model(&fused, 1).unwrap();
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
        let result = collect_conv_batchnorm_graph(
            &make_matmul(),
            &make_normalization("conv_out", "bn_out"),
            1,
        );
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
        let result = collect_conv_batchnorm_graph(&make_conv2d(), &make_matmul(), 1);
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
        assert!(build_fused_model(&fused, 1).is_err());
    }

    #[test]
    fn fused_matmul_bias_wrong_inner_errors() {
        use nxpu_analysis::fusion::FusedPattern;

        // MatMulBias with wrong matmul slot returns error
        let fused = FusedPattern::MatMulBias {
            matmul: make_conv2d(), // wrong: should be MatMul
            bias_add: Box::new(make_bias_add("x", "y")),
        };
        assert!(build_fused_model(&fused, 1).is_err());
    }

    // ---- collect_single_graph error cases ----

    #[test]
    fn collect_single_graph_unknown_errors() {
        let pattern = KernelPattern::Unknown {
            reason: "test".into(),
        };
        let result = collect_single_graph(&pattern, 8);
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
        let result = collect_single_graph(&pattern, 8);
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
        let result = collect_single_graph(&pattern, 8);
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
        let bytes = build_fused_model(&fused, 1).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    // ---- build_from_graph_desc tests ----

    #[test]
    fn build_from_graph_desc_simple() {
        let desc = GraphDesc {
            tensors: vec![
                TensorInfo::input("in", data_type::FLOAT, vec![-1]),
                TensorInfo::input("out", data_type::FLOAT, vec![-1]),
            ],
            ops: vec![OpDesc {
                opcode: builtin_op::RELU,
                inputs: vec![0],
                outputs: vec![1],
                options: OpOptions::None,
            }],
            graph_inputs: vec![0],
            graph_outputs: vec![1],
            graph_name: "test".into(),
        };
        let bytes = build_from_graph_desc(&desc, 1);
        assert!(bytes.len() > 8);
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn build_from_graph_desc_deduplicates_opcodes() {
        // Two ops with the same opcode should result in one entry in operator_codes
        let desc = GraphDesc {
            tensors: vec![
                TensorInfo::input("a", data_type::FLOAT, vec![-1]),
                TensorInfo::input("b", data_type::FLOAT, vec![-1]),
                TensorInfo::input("c", data_type::FLOAT, vec![-1]),
            ],
            ops: vec![
                OpDesc {
                    opcode: builtin_op::RELU,
                    inputs: vec![0],
                    outputs: vec![1],
                    options: OpOptions::None,
                },
                OpDesc {
                    opcode: builtin_op::RELU,
                    inputs: vec![1],
                    outputs: vec![2],
                    options: OpOptions::None,
                },
            ],
            graph_inputs: vec![0],
            graph_outputs: vec![2],
            graph_name: "dedup_test".into(),
        };
        let bytes = build_from_graph_desc(&desc, 1);
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    // ---- append_activation tests ----

    #[test]
    fn append_activation_adds_op_and_tensor() {
        use nxpu_analysis::fusion::FusedActivation;

        let mut desc = GraphDesc {
            tensors: vec![
                TensorInfo::input("in", data_type::FLOAT, vec![-1]),
                TensorInfo::input("out", data_type::FLOAT, vec![-1]),
            ],
            ops: vec![OpDesc {
                opcode: builtin_op::RELU,
                inputs: vec![0],
                outputs: vec![1],
                options: OpOptions::None,
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
        let bytes = build_fused_model(&fused, 1).unwrap();
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
        let bytes = build_fused_model(&fused, 1).unwrap();
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
        let bytes = build_fused_model(&fused, 1).unwrap();
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
        let bytes = build_fused_model(&fused, 1).unwrap();
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
        let bytes = build_fused_model(&fused, 1).unwrap();
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
        let bytes = build_fused_model(&fused, 1).unwrap();
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
        let bytes = build_fused_model(&fused, 1).unwrap();
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
        let bytes = build_fused_model(&fused, 1).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    // ---- build_model edge cases ----

    #[test]
    fn build_model_unknown_returns_error() {
        let pattern = KernelPattern::Unknown {
            reason: "test error".into(),
        };
        let result = build_model(&pattern, 1);
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
        let bytes = build_model(&pattern, 1).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn build_model_transpose() {
        let pattern = KernelPattern::Transpose {
            input: make_tensor("x", TensorRole::Input),
            output: make_tensor("y", TensorRole::Output),
            perm: vec![1, 0],
        };
        let bytes = build_model(&pattern, 1).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn build_model_reshape() {
        let pattern = KernelPattern::Reshape {
            input: make_tensor("x", TensorRole::Input),
            output: make_tensor("y", TensorRole::Output),
        };
        let bytes = build_model(&pattern, 1).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn build_model_normalization() {
        let pattern = make_normalization("x", "y");
        let bytes = build_model(&pattern, 1).unwrap();
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
        let bytes = build_model(&pattern, 1).unwrap();
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
        let bytes = build_model(&pattern, 1).unwrap();
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
        let bytes = build_model(&pattern, 1).unwrap();
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
            let bytes = build_model(&pattern, 1).unwrap();
            assert_eq!(&bytes[4..8], b"TFL3", "failed for {:?}", op);
        }
    }

    #[test]
    fn build_layout_transpose_nchw_to_nhwc() {
        let input = make_tensor("input_nchw", TensorRole::Input);
        let output = make_tensor("input_nhwc", TensorRole::Output);
        let perm = [0i64, 2, 3, 1]; // NCHW -> NHWC
        let bytes = build_layout_transpose(&input, &output, &perm, 1);
        assert!(bytes.is_some());
        let bytes = bytes.unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn build_layout_transpose_identity_returns_none() {
        let input = make_tensor("x", TensorRole::Input);
        let output = make_tensor("y", TensorRole::Output);
        let perm = [0i64, 1, 2, 3]; // identity
        assert!(build_layout_transpose(&input, &output, &perm, 1).is_none());
    }

    #[test]
    fn build_layout_transpose_empty_returns_none() {
        let input = make_tensor("x", TensorRole::Input);
        let output = make_tensor("y", TensorRole::Output);
        assert!(build_layout_transpose(&input, &output, &[], 1).is_none());
    }
}
