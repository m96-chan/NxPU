//! ONNX graph construction from classified kernel patterns.
//!
//! Converts a [`KernelPattern`] into an ONNX [`ModelProto`] with the
//! appropriate graph topology.

use crate::proto::*;
use nxpu_analysis::analyze::{
    ActivationOp, Conv2DShape, ElementWiseOp, KernelPattern, MatMulShape, PoolKind, PoolShape,
    ReduceOp, TensorBinding,
};
use nxpu_analysis::fusion::{FusedActivation, FusedPattern};
use nxpu_backend_core::BackendError;

/// Build an ONNX model from a classified kernel pattern.
pub fn build_model(pattern: &KernelPattern, ep_name: &str) -> Result<ModelProto, BackendError> {
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
        KernelPattern::Concat {
            inputs,
            output,
            axis,
        } => build_concat_graph(inputs, output, *axis, ep_name),
        KernelPattern::Split {
            input,
            outputs,
            axis,
        } => build_split_graph(input, outputs, *axis, ep_name),
        KernelPattern::Attention {
            query,
            key,
            value,
            output,
            d_k,
            seq_len,
        } => build_attention_graph(query, key, value, output, seq_len, d_k, ep_name),
        KernelPattern::Unknown { reason } => {
            return Err(BackendError::Unsupported(format!(
                "cannot lower Unknown pattern to ONNX: {reason}"
            )));
        }
    };

    Ok(ModelProto {
        ir_version: 7,
        producer_name: "nxpu".into(),
        producer_version: env!("CARGO_PKG_VERSION").into(),
        graph: Some(graph),
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
    })
}

/// Build an ONNX model from a fused pattern.
///
/// Handles single patterns, Conv+BatchNorm fusion, and activation fusion.
pub fn build_fused_model(fp: &FusedPattern, ep_name: &str) -> Result<ModelProto, BackendError> {
    match fp {
        FusedPattern::Single(p) => build_model(p, ep_name),
        FusedPattern::ConvBatchNorm { conv, norm } => {
            let (input, weight, shape) = match conv {
                KernelPattern::Conv2D {
                    input,
                    weight,
                    shape,
                    ..
                } => (input, weight, shape),
                _ => {
                    return Err(BackendError::Unsupported(
                        "ConvBatchNorm conv must be Conv2D".into(),
                    ));
                }
            };
            let (scale, bias, norm_output) = match norm.as_ref() {
                KernelPattern::Normalization {
                    scale,
                    bias,
                    output,
                    ..
                } => (scale, bias, output),
                _ => {
                    return Err(BackendError::Unsupported(
                        "ConvBatchNorm norm must be Normalization".into(),
                    ));
                }
            };

            let intermediate = "conv_out_intermediate";

            let conv_node = NodeProto::with_attrs(
                "Conv",
                "conv_0",
                vec![input.name.clone(), weight.name.clone()],
                vec![intermediate.into()],
                vec![
                    AttributeProto::ints(
                        "kernel_shape",
                        vec![shape.kernel_h_val.max(1), shape.kernel_w_val.max(1)],
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
            );

            let bn_node = NodeProto::with_attrs(
                "BatchNormalization",
                "batchnorm_0",
                vec![
                    intermediate.into(),
                    scale.name.clone(),
                    bias.name.clone(),
                    "running_mean".into(),
                    "running_var".into(),
                ],
                vec![norm_output.name.clone()],
                vec![AttributeProto::float("epsilon", 1e-5)],
            );

            let graph = GraphProto {
                name: ep_name.into(),
                initializer: vec![],
                node: vec![conv_node, bn_node],
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
                    ValueInfoProto::tensor(
                        &scale.name,
                        scale.elem_type,
                        vec![TensorShapeDimension::symbolic(&shape.channels_out)],
                    ),
                    ValueInfoProto::tensor(
                        &bias.name,
                        bias.elem_type,
                        vec![TensorShapeDimension::symbolic(&shape.channels_out)],
                    ),
                ],
                output: vec![ValueInfoProto::tensor(
                    &norm_output.name,
                    norm_output.elem_type,
                    vec![
                        TensorShapeDimension::symbolic(&shape.batch),
                        TensorShapeDimension::symbolic(&shape.channels_out),
                        TensorShapeDimension::symbolic("OH"),
                        TensorShapeDimension::symbolic("OW"),
                    ],
                )],
            };

            Ok(ModelProto {
                ir_version: 7,
                producer_name: "nxpu".into(),
                producer_version: env!("CARGO_PKG_VERSION").into(),
                graph: Some(graph),
                opset_import: vec![OperatorSetIdProto {
                    domain: String::new(),
                    version: 13,
                }],
            })
        }
        FusedPattern::WithActivation {
            base, activation, ..
        } => {
            let mut model = build_fused_model(base, ep_name)?;
            if let (Some(graph), FusedActivation::Relu) = (model.graph.as_mut(), activation) {
                // Rename the last output to an intermediate and add a Relu node.
                if let Some(last_output) = graph.output.last_mut() {
                    let original_name = last_output.name.clone();
                    let intermediate_name = format!("{original_name}_pre_relu");

                    // Rename graph output to intermediate.
                    last_output.name = intermediate_name.clone();

                    // Rename the last node's output to the intermediate.
                    if let Some(last_node) = graph.node.last_mut() {
                        for out in &mut last_node.output {
                            if *out == original_name {
                                *out = intermediate_name.clone();
                            }
                        }
                    }

                    // Add the Relu node.
                    graph.node.push(NodeProto::simple(
                        "Relu",
                        "relu_0",
                        vec![intermediate_name],
                        vec![original_name.clone()],
                    ));

                    // Restore the original output name.
                    if let Some(last_output) = graph.output.last_mut() {
                        last_output.name = original_name;
                    }
                }
            }
            Ok(model)
        }
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
        initializer: vec![],
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
        initializer: vec![],
        node: vec![NodeProto::simple(
            op.op_name(),
            format!("{}_0", op.op_name().to_lowercase()),
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
        initializer: vec![],
        node: vec![NodeProto::with_attrs(
            "Conv",
            "conv_0",
            vec![input.name.clone(), weight.name.clone()],
            vec![output.name.clone()],
            vec![
                AttributeProto::ints(
                    "kernel_shape",
                    vec![shape.kernel_h_val.max(1), shape.kernel_w_val.max(1)],
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
        initializer: vec![],
        node: vec![NodeProto::with_attrs(
            kind.op_name(),
            format!("{}_0", kind.op_name().to_lowercase()),
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
        initializer: vec![],
        node: vec![NodeProto::simple(
            op.op_name(),
            format!("{}_0", op.op_name().to_lowercase()),
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
        initializer: vec![],
        node: vec![NodeProto::with_attrs(
            op.op_name(),
            format!("{}_0", op.op_name().to_lowercase()),
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
        initializer: vec![],
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

/// Build a Reshape graph.
///
/// Currently emits a single-element shape initializer of `[-1]`, which tells
/// the ONNX runtime to flatten the input into a 1-D tensor.  This is correct
/// for the simple reshape patterns the analysis pass currently recognises, but
/// will need to be extended once multi-dimensional target shapes are supported.
fn build_reshape_graph(input: &TensorBinding, output: &TensorBinding, ep_name: &str) -> GraphProto {
    let shape_name = format!("{}_shape", output.name);
    GraphProto {
        name: ep_name.into(),
        initializer: vec![TensorProto {
            dims: vec![1],
            data_type: data_type::INT64,
            name: shape_name.clone(),
            float_data: vec![],
            int64_data: vec![-1],
        }],
        node: vec![NodeProto::simple(
            "Reshape",
            "reshape_0",
            vec![input.name.clone(), shape_name.clone()],
            vec![output.name.clone()],
        )],
        input: vec![
            ValueInfoProto::tensor(
                &input.name,
                input.elem_type,
                vec![TensorShapeDimension::symbolic("N")],
            ),
            ValueInfoProto::tensor(
                &shape_name,
                data_type::INT64,
                vec![TensorShapeDimension::fixed(1)],
            ),
        ],
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
        initializer: vec![],
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
            vec![AttributeProto::float("epsilon", 1e-5)], // IEEE float bit pattern for 1e-5 stored as int (convention)
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

fn build_concat_graph(
    inputs: &[TensorBinding],
    output: &TensorBinding,
    axis: i64,
    ep_name: &str,
) -> GraphProto {
    let input_names: Vec<String> = inputs.iter().map(|i| i.name.clone()).collect();
    GraphProto {
        name: ep_name.into(),
        initializer: vec![],
        node: vec![NodeProto::with_attrs(
            "Concat",
            "concat_0",
            input_names.clone(),
            vec![output.name.clone()],
            vec![AttributeProto::int("axis", axis)],
        )],
        input: inputs
            .iter()
            .map(|i| {
                ValueInfoProto::tensor(
                    &i.name,
                    i.elem_type,
                    vec![TensorShapeDimension::symbolic("N")],
                )
            })
            .collect(),
        output: vec![ValueInfoProto::tensor(
            &output.name,
            output.elem_type,
            vec![TensorShapeDimension::symbolic("N_out")],
        )],
    }
}

fn build_split_graph(
    input: &TensorBinding,
    outputs: &[TensorBinding],
    axis: i64,
    ep_name: &str,
) -> GraphProto {
    let output_names: Vec<String> = outputs.iter().map(|o| o.name.clone()).collect();
    GraphProto {
        name: ep_name.into(),
        initializer: vec![],
        node: vec![NodeProto::with_attrs(
            "Split",
            "split_0",
            vec![input.name.clone()],
            output_names,
            vec![AttributeProto::int("axis", axis)],
        )],
        input: vec![ValueInfoProto::tensor(
            &input.name,
            input.elem_type,
            vec![TensorShapeDimension::symbolic("N")],
        )],
        output: outputs
            .iter()
            .map(|o| {
                ValueInfoProto::tensor(
                    &o.name,
                    o.elem_type,
                    vec![TensorShapeDimension::symbolic("N_part")],
                )
            })
            .collect(),
    }
}

fn build_attention_graph(
    query: &TensorBinding,
    key: &TensorBinding,
    value: &TensorBinding,
    output: &TensorBinding,
    seq_len: &str,
    d_k: &str,
    ep_name: &str,
) -> GraphProto {
    // Emit a subgraph: Transpose(K) → MatMul(Q,K^T) → Div(sqrt_dk) → Softmax → MatMul(attn,V)
    let kt_name = "key_transposed";
    let scores_name = "scores";
    let scaled_name = "scaled_scores";
    let attn_name = "attn_weights";

    let q_shape = vec![
        TensorShapeDimension::symbolic(seq_len),
        TensorShapeDimension::symbolic(d_k),
    ];
    let k_shape = q_shape.clone();
    let v_shape = q_shape.clone();
    let out_shape = q_shape.clone();
    GraphProto {
        name: ep_name.into(),
        initializer: vec![TensorProto {
            dims: vec![1],
            data_type: data_type::FLOAT,
            name: "sqrt_dk".into(),
            float_data: vec![(d_k.parse::<f32>().unwrap_or(64.0)).sqrt()],
            int64_data: vec![],
        }],
        node: vec![
            // Transpose K
            NodeProto::with_attrs(
                "Transpose",
                "transpose_k",
                vec![key.name.clone()],
                vec![kt_name.into()],
                vec![AttributeProto::ints("perm", vec![1, 0])],
            ),
            // MatMul(Q, K^T) → scores
            NodeProto::simple(
                "MatMul",
                "matmul_qk",
                vec![query.name.clone(), kt_name.into()],
                vec![scores_name.into()],
            ),
            // Div by sqrt(d_k)
            NodeProto::simple(
                "Div",
                "scale_scores",
                vec![scores_name.into(), "sqrt_dk".into()],
                vec![scaled_name.into()],
            ),
            // Softmax
            NodeProto::simple(
                "Softmax",
                "softmax_0",
                vec![scaled_name.into()],
                vec![attn_name.into()],
            ),
            // MatMul(attn, V) → output
            NodeProto::simple(
                "MatMul",
                "matmul_av",
                vec![attn_name.into(), value.name.clone()],
                vec![output.name.clone()],
            ),
        ],
        input: vec![
            ValueInfoProto::tensor(&query.name, query.elem_type, q_shape),
            ValueInfoProto::tensor(&key.name, key.elem_type, k_shape),
            ValueInfoProto::tensor(&value.name, value.elem_type, v_shape),
            ValueInfoProto::tensor(
                "sqrt_dk",
                query.elem_type,
                vec![TensorShapeDimension::fixed(1)],
            ),
        ],
        output: vec![ValueInfoProto::tensor(
            &output.name,
            output.elem_type,
            out_shape,
        )],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::{data_type, tensor_shape_dimension, type_proto};
    use nxpu_analysis::analyze::TensorRole;
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

        let model = build_model(&pattern, "matmul_kernel").unwrap();

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
        let type_proto::Value::TensorType(a_tensor) = a_type.value.as_ref().unwrap();
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
        let type_proto::Value::TensorType(c_tensor) = c_type.value.as_ref().unwrap();
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

        let model = build_model(&pattern, "vecadd").unwrap();
        let graph = model.graph.as_ref().unwrap();

        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "Add");
        assert_eq!(graph.node[0].name, "add_0");
        assert_eq!(graph.node[0].input, vec!["x", "y"]);
        assert_eq!(graph.node[0].output, vec!["z"]);

        // All tensors are 1D with symbolic dim "N".
        for vi in graph.input.iter().chain(graph.output.iter()) {
            let type_proto::Value::TensorType(tensor) =
                vi.r#type.as_ref().unwrap().value.as_ref().unwrap();
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
            let model = build_model(&pattern, "test").unwrap();
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
                kernel_h_val: 3,
                kernel_w_val: 3,
                stride_h: 1,
                stride_w: 1,
                pad_h: 0,
                pad_w: 0,
            },
        };
        let model = build_model(&pattern, "conv2d").unwrap();
        let graph = model.graph.as_ref().unwrap();
        assert_eq!(graph.node[0].op_type, "Conv");

        // Verify kernel_shape uses kernel dimensions, not strides (#61).
        let attrs = &graph.node[0].attribute;
        let kernel_shape = attrs.iter().find(|a| a.name == "kernel_shape").unwrap();
        let strides = attrs.iter().find(|a| a.name == "strides").unwrap();
        assert_eq!(kernel_shape.ints, vec![3, 3]);
        assert_eq!(strides.ints, vec![1, 1]);
        assert_ne!(kernel_shape.ints, strides.ints);
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
        let model = build_model(&pattern, "maxpool").unwrap();
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
            let model = build_model(&pattern, "test").unwrap();
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
        let model = build_model(&pattern, "reduce").unwrap();
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
        let model = build_model(&pattern, "transpose").unwrap();
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
        let model = build_model(&pattern, "batchnorm").unwrap();
        let graph = model.graph.as_ref().unwrap();
        assert_eq!(graph.node[0].op_type, "BatchNormalization");
    }
}
