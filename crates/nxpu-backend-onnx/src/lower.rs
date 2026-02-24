//! ONNX graph construction from classified kernel patterns.
//!
//! Converts a [`KernelPattern`] into an ONNX [`ModelProto`] with the
//! appropriate graph topology.

use crate::proto::*;
use nxpu_analysis::analyze::{
    ActivationOp, Conv2DShape, ElementWiseOp, EmbeddedWeight, KernelPattern, MatMulShape, NormType,
    PoolKind, PoolShape, ReduceOp, TensorBinding,
};
use nxpu_analysis::fusion::{FusedActivation, FusedPattern};
use nxpu_backend_core::BackendError;

/// Build an ONNX model from a classified kernel pattern.
pub fn build_model(
    pattern: &KernelPattern,
    ep_name: &str,
    weights: &[EmbeddedWeight],
) -> Result<ModelProto, BackendError> {
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
            norm_type,
            ..
        } => build_normalization_graph(input, scale, bias, output, *norm_type, ep_name),
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
            num_heads,
            causal,
            ..
        } => build_attention_graph(
            query, key, value, output, seq_len, d_k, *num_heads, *causal, ep_name,
        ),
        KernelPattern::Gather {
            data,
            indices,
            output,
            axis,
        } => build_gather_graph(data, indices, output, *axis, ep_name),
        KernelPattern::Scatter {
            data,
            indices,
            updates,
            output,
            axis,
        } => build_scatter_graph(data, indices, updates, output, *axis, ep_name),
        KernelPattern::Unknown { reason } => {
            return Err(BackendError::Unsupported(format!(
                "cannot lower Unknown pattern to ONNX: {reason}"
            )));
        }
    };

    let mut graph = graph;
    inject_weights(&mut graph, weights);

    Ok(ModelProto {
        ir_version: 7,
        producer_name: "nxpu".into(),
        producer_version: env!("CARGO_PKG_VERSION").into(),
        graph: Some(graph),
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        metadata_props: vec![],
    })
}

/// Build an ONNX model from a fused pattern.
///
/// Handles single patterns, Conv+BatchNorm fusion, and activation fusion.
pub fn build_fused_model(
    fp: &FusedPattern,
    ep_name: &str,
    weights: &[EmbeddedWeight],
) -> Result<ModelProto, BackendError> {
    match fp {
        FusedPattern::Single(p) => build_model(p, ep_name, weights),
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

            let mut graph = graph;
            inject_weights(&mut graph, weights);

            Ok(ModelProto {
                ir_version: 7,
                producer_name: "nxpu".into(),
                producer_version: env!("CARGO_PKG_VERSION").into(),
                graph: Some(graph),
                opset_import: vec![OperatorSetIdProto {
                    domain: String::new(),
                    version: 13,
                }],
                metadata_props: vec![],
            })
        }
        FusedPattern::WithActivation {
            base, activation, ..
        } => {
            let mut model = build_fused_model(base, ep_name, weights)?;
            let onnx_op = match activation {
                FusedActivation::Relu => Some("Relu"),
                FusedActivation::Sigmoid => Some("Sigmoid"),
                FusedActivation::Tanh => Some("Tanh"),
                FusedActivation::None => None,
            };

            if let Some(op_name) = onnx_op {
                append_activation_node(&mut model, op_name);
            }
            Ok(model)
        }
        FusedPattern::MatMulBias { matmul, bias_add } => {
            let (inputs, output, shape) = match matmul {
                KernelPattern::MatMul {
                    inputs,
                    output,
                    shape,
                } => (inputs, output, shape),
                _ => {
                    return Err(BackendError::Unsupported(
                        "MatMulBias matmul must be MatMul".into(),
                    ));
                }
            };
            let bias_binding = match bias_add.as_ref() {
                KernelPattern::ElementWise {
                    op: ElementWiseOp::Add,
                    inputs: bias_inputs,
                    output: bias_output,
                    ..
                } => {
                    // The bias is the input that is NOT the matmul output.
                    let bias = if bias_inputs[0].name == output.name {
                        &bias_inputs[1]
                    } else {
                        &bias_inputs[0]
                    };
                    (bias, bias_output)
                }
                _ => {
                    return Err(BackendError::Unsupported(
                        "MatMulBias bias_add must be ElementWise Add".into(),
                    ));
                }
            };
            let (bias, gemm_output) = bias_binding;

            let a_name = &inputs[0].name;
            let b_name = &inputs[1].name;
            let c_name = &bias.name;
            let out_name = &gemm_output.name;

            let gemm_node = NodeProto::with_attrs(
                "Gemm",
                "gemm_0",
                vec![a_name.clone(), b_name.clone(), c_name.clone()],
                vec![out_name.clone()],
                vec![
                    AttributeProto::float("alpha", 1.0),
                    AttributeProto::float("beta", 1.0),
                ],
            );

            let graph = GraphProto {
                name: ep_name.into(),
                initializer: vec![],
                node: vec![gemm_node],
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
                    ValueInfoProto::tensor(
                        c_name,
                        bias.elem_type,
                        vec![TensorShapeDimension::symbolic(&shape.n)],
                    ),
                ],
                output: vec![ValueInfoProto::tensor(
                    out_name,
                    gemm_output.elem_type,
                    vec![
                        TensorShapeDimension::symbolic(&shape.m),
                        TensorShapeDimension::symbolic(&shape.n),
                    ],
                )],
            };

            let mut graph = graph;
            inject_weights(&mut graph, weights);

            Ok(ModelProto {
                ir_version: 7,
                producer_name: "nxpu".into(),
                producer_version: env!("CARGO_PKG_VERSION").into(),
                graph: Some(graph),
                opset_import: vec![OperatorSetIdProto {
                    domain: String::new(),
                    version: 13,
                }],
                metadata_props: vec![],
            })
        }
    }
}

/// Append an activation node (Relu, Sigmoid, Tanh) to the model's graph,
/// renaming the last output to an intermediate and adding the activation.
#[allow(clippy::collapsible_if)] // nested if-let for MSRV 1.87 compat (no let chains)
fn append_activation_node(model: &mut ModelProto, op_name: &str) {
    if let Some(graph) = model.graph.as_mut() {
        if let Some(last_output) = graph.output.last_mut() {
            let original_name = last_output.name.clone();
            let intermediate_name = format!("{original_name}_pre_{}", op_name.to_lowercase());

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

            // Add the activation node.
            graph.node.push(NodeProto::simple(
                op_name,
                format!("{}_0", op_name.to_lowercase()),
                vec![intermediate_name],
                vec![original_name.clone()],
            ));

            // Restore the original output name.
            if let Some(last_output) = graph.output.last_mut() {
                last_output.name = original_name;
            }
        }
    }
}

/// Inject embedded weight initializers into the graph.
///
/// For each weight whose name appears as an input to any graph node,
/// a `TensorProto` initializer is added to `graph.initializer`.
fn inject_weights(graph: &mut GraphProto, weights: &[EmbeddedWeight]) {
    for w in weights {
        let is_referenced = graph.node.iter().any(|n| n.input.contains(&w.name));
        if is_referenced {
            graph
                .initializer
                .push(make_weight_initializer(&w.name, &w.dims, &w.data));
        }
    }
}

/// Create a weight initializer from raw float data.
///
/// Used when constant weight values are available at compile time
/// (e.g., from `GlobalVariable.init`). The data is stored in `raw_data`
/// as little-endian bytes for efficiency.
pub fn make_weight_initializer(name: &str, dims: &[i64], data: &[f32]) -> TensorProto {
    let raw_data: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    TensorProto {
        dims: dims.to_vec(),
        data_type: data_type::FLOAT,
        name: name.into(),
        float_data: vec![],
        int32_data: vec![],
        int64_data: vec![],
        raw_data,
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
        node: vec![{
            let mut attrs = Vec::new();
            // Only emit kernel_shape when concrete values are known (> 0).
            // When unknown (0), ONNX infers kernel_shape from the weight tensor.
            if shape.kernel_h_val > 0 && shape.kernel_w_val > 0 {
                attrs.push(AttributeProto::ints(
                    "kernel_shape",
                    vec![shape.kernel_h_val, shape.kernel_w_val],
                ));
            }
            attrs.push(AttributeProto::ints(
                "strides",
                vec![shape.stride_h.max(1), shape.stride_w.max(1)],
            ));
            attrs.push(AttributeProto::ints(
                "pads",
                vec![shape.pad_h, shape.pad_w, shape.pad_h, shape.pad_w],
            ));
            if shape.groups != 1 {
                attrs.push(AttributeProto::int("group", shape.groups));
            }
            if shape.dilation_h != 1 || shape.dilation_w != 1 {
                attrs.push(AttributeProto::ints(
                    "dilations",
                    vec![shape.dilation_h, shape.dilation_w],
                ));
            }
            NodeProto::with_attrs(
                "Conv",
                "conv_0",
                vec![input.name.clone(), weight.name.clone()],
                vec![output.name.clone()],
                attrs,
            )
        }],
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
    match op {
        ActivationOp::Gelu => build_gelu_graph(input, output, dim_name, ep_name),
        ActivationOp::Silu => build_silu_graph(input, output, dim_name, ep_name),
        ActivationOp::Mish => build_mish_graph(input, output, dim_name, ep_name),
        _ => GraphProto {
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
        },
    }
}

/// Build GELU graph: x * 0.5 * (1 + Erf(x / sqrt(2)))
fn build_gelu_graph(
    input: &TensorBinding,
    output: &TensorBinding,
    dim_name: &str,
    ep_name: &str,
) -> GraphProto {
    // Decompose: x * 0.5 * (1 + Erf(x / sqrt(2)))
    let sqrt2_name = "gelu_sqrt2";
    let div_name = "gelu_div";
    let erf_name = "gelu_erf";
    let one_name = "gelu_one";
    let add_name = "gelu_add";
    let half_name = "gelu_half";
    let mul1_name = "gelu_mul1";

    GraphProto {
        name: ep_name.into(),
        initializer: vec![
            TensorProto {
                dims: vec![],
                data_type: data_type::FLOAT,
                name: sqrt2_name.into(),
                float_data: vec![std::f32::consts::SQRT_2],
                int32_data: vec![],
                int64_data: vec![],
                raw_data: vec![],
            },
            TensorProto {
                dims: vec![],
                data_type: data_type::FLOAT,
                name: one_name.into(),
                float_data: vec![1.0],
                int32_data: vec![],
                int64_data: vec![],
                raw_data: vec![],
            },
            TensorProto {
                dims: vec![],
                data_type: data_type::FLOAT,
                name: half_name.into(),
                float_data: vec![0.5],
                int32_data: vec![],
                int64_data: vec![],
                raw_data: vec![],
            },
        ],
        node: vec![
            // x / sqrt(2)
            NodeProto::simple(
                "Div",
                "gelu_div_0",
                vec![input.name.clone(), sqrt2_name.into()],
                vec![div_name.into()],
            ),
            // Erf(x / sqrt(2))
            NodeProto::simple(
                "Erf",
                "gelu_erf_0",
                vec![div_name.into()],
                vec![erf_name.into()],
            ),
            // 1 + Erf(...)
            NodeProto::simple(
                "Add",
                "gelu_add_0",
                vec![erf_name.into(), one_name.into()],
                vec![add_name.into()],
            ),
            // x * (1 + Erf(...))
            NodeProto::simple(
                "Mul",
                "gelu_mul_0",
                vec![input.name.clone(), add_name.into()],
                vec![mul1_name.into()],
            ),
            // result * 0.5
            NodeProto::simple(
                "Mul",
                "gelu_mul_1",
                vec![mul1_name.into(), half_name.into()],
                vec![output.name.clone()],
            ),
        ],
        input: vec![
            ValueInfoProto::tensor(
                &input.name,
                input.elem_type,
                vec![TensorShapeDimension::symbolic(dim_name)],
            ),
            ValueInfoProto::tensor(sqrt2_name, data_type::FLOAT, vec![]),
            ValueInfoProto::tensor(one_name, data_type::FLOAT, vec![]),
            ValueInfoProto::tensor(half_name, data_type::FLOAT, vec![]),
        ],
        output: vec![ValueInfoProto::tensor(
            &output.name,
            output.elem_type,
            vec![TensorShapeDimension::symbolic(dim_name)],
        )],
    }
}

/// Build SiLU graph: x * sigmoid(x) = x * (1 / (1 + exp(-x)))
fn build_silu_graph(
    input: &TensorBinding,
    output: &TensorBinding,
    dim_name: &str,
    ep_name: &str,
) -> GraphProto {
    let sigmoid_name = "silu_sigmoid";

    GraphProto {
        name: ep_name.into(),
        initializer: vec![],
        node: vec![
            NodeProto::simple(
                "Sigmoid",
                "silu_sigmoid_0",
                vec![input.name.clone()],
                vec![sigmoid_name.into()],
            ),
            NodeProto::simple(
                "Mul",
                "silu_mul_0",
                vec![input.name.clone(), sigmoid_name.into()],
                vec![output.name.clone()],
            ),
        ],
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

/// Build Mish graph: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
fn build_mish_graph(
    input: &TensorBinding,
    output: &TensorBinding,
    dim_name: &str,
    ep_name: &str,
) -> GraphProto {
    let softplus_name = "mish_softplus";
    let tanh_name = "mish_tanh";

    GraphProto {
        name: ep_name.into(),
        initializer: vec![],
        node: vec![
            NodeProto::simple(
                "Softplus",
                "mish_softplus_0",
                vec![input.name.clone()],
                vec![softplus_name.into()],
            ),
            NodeProto::simple(
                "Tanh",
                "mish_tanh_0",
                vec![softplus_name.into()],
                vec![tanh_name.into()],
            ),
            NodeProto::simple(
                "Mul",
                "mish_mul_0",
                vec![input.name.clone(), tanh_name.into()],
                vec![output.name.clone()],
            ),
        ],
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
            int32_data: vec![],
            int64_data: vec![-1],
            raw_data: vec![],
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
    norm_type: NormType,
    ep_name: &str,
) -> GraphProto {
    match norm_type {
        NormType::Batch => {
            // mean -> empty string (runtime computed)
            // var -> empty string (runtime computed)
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
                    vec![AttributeProto::float("epsilon", 1e-5)],
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
        NormType::Layer => {
            // ONNX LayerNormalization (opset 17+)
            GraphProto {
                name: ep_name.into(),
                initializer: vec![],
                node: vec![NodeProto::with_attrs(
                    "LayerNormalization",
                    "layernorm_0",
                    vec![input.name.clone(), scale.name.clone(), bias.name.clone()],
                    vec![output.name.clone()],
                    vec![
                        AttributeProto::float("epsilon", 1e-5),
                        AttributeProto::int("axis", -1),
                    ],
                )],
                input: vec![
                    ValueInfoProto::tensor(
                        &input.name,
                        input.elem_type,
                        vec![
                            TensorShapeDimension::symbolic("N"),
                            TensorShapeDimension::symbolic("C"),
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
                    ],
                )],
            }
        }
    }
}

fn build_gather_graph(
    data: &TensorBinding,
    indices: &TensorBinding,
    output: &TensorBinding,
    axis: i64,
    ep_name: &str,
) -> GraphProto {
    GraphProto {
        name: ep_name.into(),
        initializer: vec![],
        node: vec![NodeProto::with_attrs(
            "Gather",
            "gather_0",
            vec![data.name.clone(), indices.name.clone()],
            vec![output.name.clone()],
            vec![AttributeProto::int("axis", axis)],
        )],
        input: vec![
            ValueInfoProto::tensor(
                &data.name,
                data.elem_type,
                vec![TensorShapeDimension::symbolic("N")],
            ),
            ValueInfoProto::tensor(
                &indices.name,
                indices.elem_type,
                vec![TensorShapeDimension::symbolic("M")],
            ),
        ],
        output: vec![ValueInfoProto::tensor(
            &output.name,
            output.elem_type,
            vec![TensorShapeDimension::symbolic("M")],
        )],
    }
}

fn build_scatter_graph(
    data: &TensorBinding,
    indices: &TensorBinding,
    updates: &TensorBinding,
    output: &TensorBinding,
    _axis: i64,
    ep_name: &str,
) -> GraphProto {
    GraphProto {
        name: ep_name.into(),
        initializer: vec![],
        node: vec![NodeProto::simple(
            "ScatterND",
            "scatter_0",
            vec![
                data.name.clone(),
                indices.name.clone(),
                updates.name.clone(),
            ],
            vec![output.name.clone()],
        )],
        input: vec![
            ValueInfoProto::tensor(
                &data.name,
                data.elem_type,
                vec![TensorShapeDimension::symbolic("N")],
            ),
            ValueInfoProto::tensor(
                &indices.name,
                indices.elem_type,
                vec![TensorShapeDimension::symbolic("M")],
            ),
            ValueInfoProto::tensor(
                &updates.name,
                updates.elem_type,
                vec![TensorShapeDimension::symbolic("M")],
            ),
        ],
        output: vec![ValueInfoProto::tensor(
            &output.name,
            output.elem_type,
            vec![TensorShapeDimension::symbolic("N")],
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

#[allow(clippy::too_many_arguments)]
fn build_attention_graph(
    query: &TensorBinding,
    key: &TensorBinding,
    value: &TensorBinding,
    output: &TensorBinding,
    seq_len: &str,
    d_k: &str,
    num_heads: u32,
    causal: bool,
    ep_name: &str,
) -> GraphProto {
    // Emit a subgraph: Transpose(K) → MatMul(Q,K^T) → Div(sqrt_dk) → Softmax → MatMul(attn,V)
    // For multi-head: prepend Reshape+Transpose for Q/K/V, and append Reshape after.
    // For causal: insert a Where node with a triangular mask before Softmax.
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

    let mut nodes = Vec::new();
    let mut initializers = vec![TensorProto {
        dims: vec![],
        data_type: data_type::INT64,
        name: "dk_axis".into(),
        float_data: vec![],
        int32_data: vec![],
        int64_data: vec![1],
        raw_data: vec![],
    }];

    // Source names for Q/K/V (may be overwritten by multi-head reshape)
    let mut q_src = query.name.clone();
    let mut k_src = key.name.clone();
    let mut v_src = value.name.clone();
    let final_output_name;

    if num_heads > 1 {
        // Multi-head: Reshape Q/K/V from [batch, seq, d_model] → [batch, num_heads, seq, head_dim]
        // Then Transpose to [batch, num_heads, seq, head_dim] (perm [0,2,1,3])
        for (src_name, reshaped, transposed) in [
            (&query.name, "q_reshaped", "q_mh"),
            (&key.name, "k_reshaped", "k_mh"),
            (&value.name, "v_reshaped", "v_mh"),
        ] {
            nodes.push(NodeProto::with_attrs(
                "Reshape",
                format!("reshape_{reshaped}"),
                vec![src_name.clone(), "mh_shape".into()],
                vec![reshaped.into()],
                vec![],
            ));
            nodes.push(NodeProto::with_attrs(
                "Transpose",
                format!("transpose_{transposed}"),
                vec![reshaped.into()],
                vec![transposed.into()],
                vec![AttributeProto::ints("perm", vec![0, 2, 1, 3])],
            ));
        }
        q_src = "q_mh".into();
        k_src = "k_mh".into();
        v_src = "v_mh".into();
        final_output_name = "attn_mh_out".to_string();

        // mh_shape initializer: symbolic reshape target
        initializers.push(TensorProto {
            dims: vec![4],
            data_type: data_type::INT64,
            name: "mh_shape".into(),
            float_data: vec![],
            int32_data: vec![],
            int64_data: vec![0, -1, num_heads as i64, -1],
            raw_data: vec![],
        });
    } else {
        final_output_name = output.name.clone();
    }

    // Transpose K
    nodes.push(NodeProto::with_attrs(
        "Transpose",
        "transpose_k",
        vec![k_src.clone()],
        vec![kt_name.into()],
        vec![AttributeProto::ints("perm", vec![1, 0])],
    ));
    // MatMul(Q, K^T) → scores
    nodes.push(NodeProto::simple(
        "MatMul",
        "matmul_qk",
        vec![q_src, kt_name.into()],
        vec![scores_name.into()],
    ));
    // Shape(Q) → [seq_len, d_k]
    nodes.push(NodeProto::simple(
        "Shape",
        "shape_q",
        vec![query.name.clone()],
        vec!["query_shape".into()],
    ));
    // Gather(query_shape, 1) → d_k dimension (scalar)
    nodes.push(NodeProto::simple(
        "Gather",
        "gather_dk",
        vec!["query_shape".into(), "dk_axis".into()],
        vec!["dk_dim".into()],
    ));
    // Cast(dk_dim, to=FLOAT) → scalar f32
    nodes.push(NodeProto::with_attrs(
        "Cast",
        "cast_dk",
        vec!["dk_dim".into()],
        vec!["dk_float".into()],
        vec![AttributeProto::int("to", data_type::FLOAT as i64)],
    ));
    // Sqrt(dk_float) → sqrt(d_k)
    nodes.push(NodeProto::simple(
        "Sqrt",
        "sqrt_dk",
        vec!["dk_float".into()],
        vec!["sqrt_dk_val".into()],
    ));
    // Div(scores, sqrt_dk_val) → scaled_scores
    nodes.push(NodeProto::simple(
        "Div",
        "scale_scores",
        vec![scores_name.into(), "sqrt_dk_val".into()],
        vec![scaled_name.into()],
    ));

    // Causal mask: Where(tri_mask, scaled_scores, -1e9)
    let softmax_input = if causal {
        let neg_inf_bytes: Vec<u8> = (-1e9_f32).to_le_bytes().to_vec();
        initializers.push(TensorProto {
            dims: vec![],
            data_type: data_type::FLOAT,
            name: "neg_inf".into(),
            float_data: vec![],
            int32_data: vec![],
            int64_data: vec![],
            raw_data: neg_inf_bytes,
        });
        // tri_mask is a boolean constant that must be provided at runtime or as an initializer.
        // We add a placeholder input for it.
        nodes.push(NodeProto::simple(
            "Where",
            "causal_mask",
            vec!["tri_mask".into(), scaled_name.into(), "neg_inf".into()],
            vec!["masked_scores".into()],
        ));
        "masked_scores"
    } else {
        scaled_name
    };

    // Softmax
    nodes.push(NodeProto::simple(
        "Softmax",
        "softmax_0",
        vec![softmax_input.into()],
        vec![attn_name.into()],
    ));
    // MatMul(attn, V) → output
    nodes.push(NodeProto::simple(
        "MatMul",
        "matmul_av",
        vec![attn_name.into(), v_src],
        vec![final_output_name.clone()],
    ));

    if num_heads > 1 {
        // Reshape back: [batch, num_heads, seq, head_dim] → [batch, seq, d_model]
        nodes.push(NodeProto::with_attrs(
            "Transpose",
            "transpose_mh_out",
            vec![final_output_name],
            vec!["attn_transposed".into()],
            vec![AttributeProto::ints("perm", vec![0, 2, 1, 3])],
        ));
        nodes.push(NodeProto::with_attrs(
            "Reshape",
            "reshape_mh_out",
            vec!["attn_transposed".into(), "out_shape".into()],
            vec![output.name.clone()],
            vec![],
        ));
        initializers.push(TensorProto {
            dims: vec![3],
            data_type: data_type::INT64,
            name: "out_shape".into(),
            float_data: vec![],
            int32_data: vec![],
            int64_data: vec![0, -1, -1],
            raw_data: vec![],
        });
    }

    let mut inputs = vec![
        ValueInfoProto::tensor(&query.name, query.elem_type, q_shape),
        ValueInfoProto::tensor(&key.name, key.elem_type, k_shape),
        ValueInfoProto::tensor(&value.name, value.elem_type, v_shape),
        // dk_axis is an initializer with default value; listed as input for ONNX compliance.
        ValueInfoProto::tensor("dk_axis", data_type::INT64, vec![]),
    ];
    if causal {
        inputs.push(ValueInfoProto::tensor(
            "tri_mask",
            data_type::BOOL,
            vec![
                TensorShapeDimension::symbolic(seq_len),
                TensorShapeDimension::symbolic(seq_len),
            ],
        ));
    }

    GraphProto {
        name: ep_name.into(),
        initializer: initializers,
        node: nodes,
        input: inputs,
        output: vec![ValueInfoProto::tensor(
            &output.name,
            output.elem_type,
            out_shape,
        )],
    }
}

/// Inject QDQ (QuantizeLinear -> DequantizeLinear) nodes for per-channel
/// quantized weight tensors.
///
/// For each [`PerChannelParam`], this function:
/// 1. Adds scale and zero_point initializer tensors to the graph.
/// 2. Inserts QuantizeLinear and DequantizeLinear nodes at the beginning.
/// 3. Rewrites downstream node inputs to reference the dequantized output.
pub fn inject_per_channel_qdq(
    graph: &mut GraphProto,
    per_channel_params: &[nxpu_backend_core::PerChannelParam],
) {
    for pcp in per_channel_params {
        let original_name = &pcp.name;
        let quant_name = format!("{}_quantized", original_name);
        let dequant_name = format!("{}_dequantized", original_name);
        let scale_name = format!("{}_scale", original_name);
        let zp_name = format!("{}_zero_point", original_name);

        // Add scale as a FLOAT initializer tensor.
        graph.initializer.push(TensorProto {
            name: scale_name.clone(),
            data_type: data_type::FLOAT,
            dims: vec![pcp.scales.len() as i64],
            float_data: pcp.scales.clone(),
            int32_data: vec![],
            int64_data: vec![],
            raw_data: vec![],
        });

        // Add zero_point as an INT8 initializer tensor.
        graph.initializer.push(TensorProto {
            name: zp_name.clone(),
            data_type: data_type::INT8,
            dims: vec![pcp.zero_points.len() as i64],
            float_data: vec![],
            int32_data: pcp.zero_points.clone(),
            int64_data: vec![],
            raw_data: vec![],
        });

        // QuantizeLinear: original -> quantized
        let quant_node = NodeProto::with_attrs(
            "QuantizeLinear",
            format!("quantize_{}", original_name),
            vec![original_name.clone(), scale_name.clone(), zp_name.clone()],
            vec![quant_name.clone()],
            vec![AttributeProto::int("axis", pcp.channel_axis as i64)],
        );

        // DequantizeLinear: quantized -> dequantized
        let dequant_node = NodeProto::with_attrs(
            "DequantizeLinear",
            format!("dequantize_{}", original_name),
            vec![quant_name, scale_name, zp_name],
            vec![dequant_name.clone()],
            vec![AttributeProto::int("axis", pcp.channel_axis as i64)],
        );

        // Insert QDQ nodes at the beginning of the graph.
        graph.node.insert(0, dequant_node);
        graph.node.insert(0, quant_node);

        // Update all subsequent nodes that reference the original weight
        // to use the dequantized output instead.
        for node in &mut graph.node[2..] {
            for input in &mut node.input {
                if *input == *original_name {
                    *input = dequant_name.clone();
                }
            }
        }
    }
}

/// Inserts a Transpose node into an ONNX graph for layout conversion.
///
/// If `perm` is a non-identity permutation, this adds a Transpose node that
/// reads from `input_name` and writes to `output_name`.  The caller is
/// responsible for wiring the surrounding graph so that the right tensors
/// flow through the transpose.
///
/// This is used by the ONNX backend when the source IR has a different
/// memory layout (e.g. NHWC) than the ONNX-expected layout (NCHW).
#[allow(dead_code)] // Infrastructure for layout conversion; wired in by backends as needed.
pub fn maybe_add_layout_transpose(
    graph: &mut GraphProto,
    perm: &[i64],
    input_name: &str,
    output_name: &str,
) {
    // Identity check: if perm is [0, 1, 2, ...] there is nothing to do.
    let is_identity = perm.iter().enumerate().all(|(i, &p)| p as usize == i);
    if is_identity || perm.is_empty() {
        return;
    }

    graph.node.push(NodeProto::with_attrs(
        "Transpose",
        format!("layout_transpose_{output_name}"),
        vec![input_name.into()],
        vec![output_name.into()],
        vec![AttributeProto::ints("perm", perm.to_vec())],
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::{data_type, tensor_shape_dimension, type_proto};
    use nxpu_analysis::analyze::{NormType, TensorRole};
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

        let model = build_model(&pattern, "matmul_kernel", &[]).unwrap();

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

        let model = build_model(&pattern, "vecadd", &[]).unwrap();
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
            let model = build_model(&pattern, "test", &[]).unwrap();
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
                groups: 1,
                dilation_h: 1,
                dilation_w: 1,
            },
        };
        let model = build_model(&pattern, "conv2d", &[]).unwrap();
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
        let model = build_model(&pattern, "maxpool", &[]).unwrap();
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
            let model = build_model(&pattern, "test", &[]).unwrap();
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
        let model = build_model(&pattern, "reduce", &[]).unwrap();
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
        let model = build_model(&pattern, "transpose", &[]).unwrap();
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
            norm_type: NormType::Batch,
        };
        let model = build_model(&pattern, "batchnorm", &[]).unwrap();
        let graph = model.graph.as_ref().unwrap();
        assert_eq!(graph.node[0].op_type, "BatchNormalization");
    }

    #[test]
    fn weight_initializer_roundtrip() {
        let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let init = make_weight_initializer("weights", &[2, 3], &data);
        assert_eq!(init.dims, vec![2, 3]);
        assert_eq!(init.data_type, data_type::FLOAT);
        assert_eq!(init.name, "weights");
        assert!(init.float_data.is_empty());
        assert_eq!(init.raw_data.len(), 24); // 6 * 4 bytes

        // Verify raw_data decodes back to original floats
        let decoded: Vec<f32> = init
            .raw_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(decoded, data);
    }

    #[test]
    fn attention_model_has_dynamic_sqrt_dk() {
        let pattern = KernelPattern::Attention {
            query: make_tensor("Q", TensorRole::Input),
            key: make_tensor("K", TensorRole::Input),
            value: make_tensor("V", TensorRole::Input),
            output: make_tensor("out", TensorRole::Output),
            d_k: "d_k".into(),
            seq_len: "seq_len".into(),
            num_heads: 1,
            num_kv_heads: 1,
            causal: false,
        };
        let model = build_model(&pattern, "attention", &[]).unwrap();
        let graph = model.graph.as_ref().unwrap();

        let op_types: Vec<&str> = graph.node.iter().map(|n| n.op_type.as_str()).collect();
        assert!(op_types.contains(&"Shape"), "should have Shape node");
        assert!(op_types.contains(&"Gather"), "should have Gather node");
        assert!(op_types.contains(&"Cast"), "should have Cast node");
        assert!(op_types.contains(&"Sqrt"), "should have Sqrt node");

        // The initializer should be dk_axis (int64 scalar), not a hardcoded sqrt_dk float
        assert_eq!(graph.initializer.len(), 1);
        assert_eq!(graph.initializer[0].name, "dk_axis");
        assert_eq!(graph.initializer[0].data_type, data_type::INT64);
    }

    // ---------------------------------------------------------------
    // Fusion-related tests
    // ---------------------------------------------------------------

    fn make_conv2d_shape() -> Conv2DShape {
        Conv2DShape {
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
        }
    }

    fn make_matmul_shape() -> MatMulShape {
        MatMulShape {
            m: "M".into(),
            n: "N".into(),
            k: "K".into(),
        }
    }

    #[test]
    fn fused_single_delegates_to_build_model() {
        let pattern = KernelPattern::MatMul {
            inputs: [
                make_tensor("A", TensorRole::Input),
                make_tensor("B", TensorRole::Input),
            ],
            output: make_tensor("C", TensorRole::Output),
            shape: make_matmul_shape(),
        };
        let fused = FusedPattern::Single(pattern);
        let model = build_fused_model(&fused, "single_matmul", &[]).unwrap();

        let graph = model.graph.as_ref().unwrap();
        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "MatMul");
        assert_eq!(graph.name, "single_matmul");
    }

    #[test]
    fn fused_single_conv2d() {
        let pattern = KernelPattern::Conv2D {
            input: make_tensor("x", TensorRole::Input),
            weight: make_tensor("w", TensorRole::Input),
            output: make_tensor("y", TensorRole::Output),
            shape: make_conv2d_shape(),
        };
        let fused = FusedPattern::Single(pattern);
        let model = build_fused_model(&fused, "single_conv", &[]).unwrap();

        let graph = model.graph.as_ref().unwrap();
        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "Conv");
    }

    #[test]
    fn fused_conv_batchnorm() {
        let conv = KernelPattern::Conv2D {
            input: make_tensor("x", TensorRole::Input),
            weight: make_tensor("w", TensorRole::Input),
            output: make_tensor("conv_out", TensorRole::Output),
            shape: make_conv2d_shape(),
        };
        let norm = KernelPattern::Normalization {
            input: make_tensor("conv_out", TensorRole::Input),
            scale: make_tensor("gamma", TensorRole::Input),
            bias: make_tensor("beta", TensorRole::Input),
            output: make_tensor("bn_out", TensorRole::Output),
            epsilon: 1e-5,
            norm_type: NormType::Batch,
        };
        let fused = FusedPattern::ConvBatchNorm {
            conv,
            norm: Box::new(norm),
        };

        let model = build_fused_model(&fused, "conv_bn", &[]).unwrap();
        assert_eq!(model.ir_version, 7);
        assert_eq!(model.producer_name, "nxpu");

        let graph = model.graph.as_ref().unwrap();
        assert_eq!(graph.name, "conv_bn");
        assert_eq!(graph.node.len(), 2);
        assert_eq!(graph.node[0].op_type, "Conv");
        assert_eq!(graph.node[0].name, "conv_0");
        assert_eq!(graph.node[1].op_type, "BatchNormalization");
        assert_eq!(graph.node[1].name, "batchnorm_0");

        // Conv output feeds into BatchNorm via intermediate name.
        assert_eq!(graph.node[0].output, vec!["conv_out_intermediate"]);
        assert_eq!(graph.node[1].input[0], "conv_out_intermediate");

        // Graph inputs: x, w, gamma, beta.
        assert_eq!(graph.input.len(), 4);
        assert_eq!(graph.input[0].name, "x");
        assert_eq!(graph.input[1].name, "w");
        assert_eq!(graph.input[2].name, "gamma");
        assert_eq!(graph.input[3].name, "beta");

        // Graph output: bn_out.
        assert_eq!(graph.output.len(), 1);
        assert_eq!(graph.output[0].name, "bn_out");

        // Conv node attributes: kernel_shape, strides, pads.
        let conv_attrs = &graph.node[0].attribute;
        let ks = conv_attrs
            .iter()
            .find(|a| a.name == "kernel_shape")
            .unwrap();
        assert_eq!(ks.ints, vec![3, 3]);
        let strides = conv_attrs.iter().find(|a| a.name == "strides").unwrap();
        assert_eq!(strides.ints, vec![1, 1]);

        // BatchNorm node has epsilon attribute.
        let bn_attrs = &graph.node[1].attribute;
        let eps = bn_attrs.iter().find(|a| a.name == "epsilon").unwrap();
        assert!((eps.f - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn fused_matmul_bias_produces_gemm() {
        let matmul = KernelPattern::MatMul {
            inputs: [
                make_tensor("A", TensorRole::Input),
                make_tensor("B", TensorRole::Input),
            ],
            output: make_tensor("mm_out", TensorRole::Output),
            shape: make_matmul_shape(),
        };
        let bias_add = KernelPattern::ElementWise {
            op: ElementWiseOp::Add,
            inputs: [
                make_tensor("mm_out", TensorRole::Input),
                make_tensor("bias", TensorRole::Input),
            ],
            output: make_tensor("out", TensorRole::Output),
            dim_name: "N".into(),
        };
        let fused = FusedPattern::MatMulBias {
            matmul,
            bias_add: Box::new(bias_add),
        };

        let model = build_fused_model(&fused, "gemm_ep", &[]).unwrap();
        assert_eq!(model.ir_version, 7);
        assert_eq!(model.producer_name, "nxpu");

        let graph = model.graph.as_ref().unwrap();
        assert_eq!(graph.name, "gemm_ep");
        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "Gemm");
        assert_eq!(graph.node[0].name, "gemm_0");

        // Gemm inputs: A, B, bias.
        assert_eq!(graph.node[0].input, vec!["A", "B", "bias"]);
        // Gemm output: out.
        assert_eq!(graph.node[0].output, vec!["out"]);

        // Graph inputs: A [M,K], B [K,N], bias [N].
        assert_eq!(graph.input.len(), 3);
        assert_eq!(graph.input[0].name, "A");
        assert_eq!(graph.input[1].name, "B");
        assert_eq!(graph.input[2].name, "bias");

        // Graph output: out [M,N].
        assert_eq!(graph.output.len(), 1);
        assert_eq!(graph.output[0].name, "out");

        // Gemm attributes: alpha=1.0, beta=1.0.
        let attrs = &graph.node[0].attribute;
        let alpha = attrs.iter().find(|a| a.name == "alpha").unwrap();
        assert!((alpha.f - 1.0).abs() < 1e-6);
        let beta = attrs.iter().find(|a| a.name == "beta").unwrap();
        assert!((beta.f - 1.0).abs() < 1e-6);

        // Verify shapes via type protos.
        let a_type = graph.input[0].r#type.as_ref().unwrap();
        let type_proto::Value::TensorType(a_tensor) = a_type.value.as_ref().unwrap();
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

        // bias shape: [N].
        let bias_type = graph.input[2].r#type.as_ref().unwrap();
        let type_proto::Value::TensorType(bias_tensor) = bias_type.value.as_ref().unwrap();
        let bias_dims = &bias_tensor.shape.as_ref().unwrap().dim;
        assert_eq!(bias_dims.len(), 1);
        assert_eq!(
            bias_dims[0].value,
            Some(tensor_shape_dimension::Value::DimParam("N".into()))
        );
    }

    #[test]
    fn fused_matmul_bias_reversed_add_inputs() {
        // When bias is the first input of the Add (not mm_out), it should still work.
        let matmul = KernelPattern::MatMul {
            inputs: [
                make_tensor("A", TensorRole::Input),
                make_tensor("B", TensorRole::Input),
            ],
            output: make_tensor("mm_out", TensorRole::Output),
            shape: make_matmul_shape(),
        };
        let bias_add = KernelPattern::ElementWise {
            op: ElementWiseOp::Add,
            inputs: [
                make_tensor("bias", TensorRole::Input),
                make_tensor("mm_out", TensorRole::Input),
            ],
            output: make_tensor("out", TensorRole::Output),
            dim_name: "N".into(),
        };
        let fused = FusedPattern::MatMulBias {
            matmul,
            bias_add: Box::new(bias_add),
        };

        let model = build_fused_model(&fused, "gemm_rev", &[]).unwrap();
        let graph = model.graph.as_ref().unwrap();
        assert_eq!(graph.node[0].op_type, "Gemm");
        // bias should still be identified correctly.
        assert_eq!(graph.node[0].input, vec!["A", "B", "bias"]);
    }

    #[test]
    fn fused_with_activation_relu() {
        let base = FusedPattern::Single(KernelPattern::ElementWise {
            op: ElementWiseOp::Add,
            inputs: [
                make_tensor("a", TensorRole::Input),
                make_tensor("b", TensorRole::Input),
            ],
            output: make_tensor("c", TensorRole::Output),
            dim_name: "N".into(),
        });
        let act_pattern = KernelPattern::Activation {
            op: ActivationOp::Relu,
            input: make_tensor("c", TensorRole::Input),
            output: make_tensor("d", TensorRole::Output),
            dim_name: "N".into(),
        };
        let fused = FusedPattern::WithActivation {
            base: Box::new(base),
            activation: FusedActivation::Relu,
            activation_pattern: Box::new(act_pattern),
        };

        let model = build_fused_model(&fused, "add_relu", &[]).unwrap();
        let graph = model.graph.as_ref().unwrap();

        // Should have Add + Relu nodes.
        assert_eq!(graph.node.len(), 2);
        assert_eq!(graph.node[0].op_type, "Add");
        assert_eq!(graph.node[1].op_type, "Relu");

        // The Add output is renamed to an intermediate.
        assert_eq!(graph.node[0].output, vec!["c_pre_relu"]);
        // Relu takes the intermediate and produces the original output name.
        assert_eq!(graph.node[1].input, vec!["c_pre_relu"]);
        assert_eq!(graph.node[1].output, vec!["c"]);

        // Graph output should be the original name.
        assert_eq!(graph.output[0].name, "c");
    }

    #[test]
    fn fused_with_activation_sigmoid() {
        let base = FusedPattern::Single(KernelPattern::ElementWise {
            op: ElementWiseOp::Mul,
            inputs: [
                make_tensor("a", TensorRole::Input),
                make_tensor("b", TensorRole::Input),
            ],
            output: make_tensor("c", TensorRole::Output),
            dim_name: "N".into(),
        });
        let act_pattern = KernelPattern::Activation {
            op: ActivationOp::Sigmoid,
            input: make_tensor("c", TensorRole::Input),
            output: make_tensor("d", TensorRole::Output),
            dim_name: "N".into(),
        };
        let fused = FusedPattern::WithActivation {
            base: Box::new(base),
            activation: FusedActivation::Sigmoid,
            activation_pattern: Box::new(act_pattern),
        };

        let model = build_fused_model(&fused, "mul_sigmoid", &[]).unwrap();
        let graph = model.graph.as_ref().unwrap();

        assert_eq!(graph.node.len(), 2);
        assert_eq!(graph.node[0].op_type, "Mul");
        assert_eq!(graph.node[1].op_type, "Sigmoid");

        assert_eq!(graph.node[0].output, vec!["c_pre_sigmoid"]);
        assert_eq!(graph.node[1].input, vec!["c_pre_sigmoid"]);
        assert_eq!(graph.node[1].output, vec!["c"]);
    }

    #[test]
    fn fused_with_activation_tanh() {
        let base = FusedPattern::Single(KernelPattern::MatMul {
            inputs: [
                make_tensor("A", TensorRole::Input),
                make_tensor("B", TensorRole::Input),
            ],
            output: make_tensor("C", TensorRole::Output),
            shape: make_matmul_shape(),
        });
        let act_pattern = KernelPattern::Activation {
            op: ActivationOp::Tanh,
            input: make_tensor("C", TensorRole::Input),
            output: make_tensor("D", TensorRole::Output),
            dim_name: "N".into(),
        };
        let fused = FusedPattern::WithActivation {
            base: Box::new(base),
            activation: FusedActivation::Tanh,
            activation_pattern: Box::new(act_pattern),
        };

        let model = build_fused_model(&fused, "matmul_tanh", &[]).unwrap();
        let graph = model.graph.as_ref().unwrap();

        assert_eq!(graph.node.len(), 2);
        assert_eq!(graph.node[0].op_type, "MatMul");
        assert_eq!(graph.node[1].op_type, "Tanh");

        assert_eq!(graph.node[0].output, vec!["C_pre_tanh"]);
        assert_eq!(graph.node[1].input, vec!["C_pre_tanh"]);
        assert_eq!(graph.node[1].output, vec!["C"]);
    }

    #[test]
    fn fused_with_activation_none_is_noop() {
        let base = FusedPattern::Single(KernelPattern::ElementWise {
            op: ElementWiseOp::Add,
            inputs: [
                make_tensor("a", TensorRole::Input),
                make_tensor("b", TensorRole::Input),
            ],
            output: make_tensor("c", TensorRole::Output),
            dim_name: "N".into(),
        });
        let act_pattern = KernelPattern::Activation {
            op: ActivationOp::Relu,
            input: make_tensor("c", TensorRole::Input),
            output: make_tensor("d", TensorRole::Output),
            dim_name: "N".into(),
        };
        let fused = FusedPattern::WithActivation {
            base: Box::new(base),
            activation: FusedActivation::None,
            activation_pattern: Box::new(act_pattern),
        };

        let model = build_fused_model(&fused, "add_none", &[]).unwrap();
        let graph = model.graph.as_ref().unwrap();

        // FusedActivation::None should not append any activation node.
        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "Add");
    }

    #[test]
    fn fused_conv_batchnorm_with_relu() {
        let conv = KernelPattern::Conv2D {
            input: make_tensor("x", TensorRole::Input),
            weight: make_tensor("w", TensorRole::Input),
            output: make_tensor("conv_out", TensorRole::Output),
            shape: make_conv2d_shape(),
        };
        let norm = KernelPattern::Normalization {
            input: make_tensor("conv_out", TensorRole::Input),
            scale: make_tensor("gamma", TensorRole::Input),
            bias: make_tensor("beta", TensorRole::Input),
            output: make_tensor("bn_out", TensorRole::Output),
            epsilon: 1e-5,
            norm_type: NormType::Batch,
        };
        let conv_bn = FusedPattern::ConvBatchNorm {
            conv,
            norm: Box::new(norm),
        };
        let act_pattern = KernelPattern::Activation {
            op: ActivationOp::Relu,
            input: make_tensor("bn_out", TensorRole::Input),
            output: make_tensor("relu_out", TensorRole::Output),
            dim_name: "N".into(),
        };
        let fused = FusedPattern::WithActivation {
            base: Box::new(conv_bn),
            activation: FusedActivation::Relu,
            activation_pattern: Box::new(act_pattern),
        };

        let model = build_fused_model(&fused, "conv_bn_relu", &[]).unwrap();
        let graph = model.graph.as_ref().unwrap();

        // Conv + BatchNorm + Relu = 3 nodes.
        assert_eq!(graph.node.len(), 3);
        assert_eq!(graph.node[0].op_type, "Conv");
        assert_eq!(graph.node[1].op_type, "BatchNormalization");
        assert_eq!(graph.node[2].op_type, "Relu");

        // BN output is renamed to intermediate, Relu restores original.
        assert_eq!(graph.node[1].output, vec!["bn_out_pre_relu"]);
        assert_eq!(graph.node[2].input, vec!["bn_out_pre_relu"]);
        assert_eq!(graph.node[2].output, vec!["bn_out"]);
    }

    #[test]
    fn fused_gemm_with_relu() {
        let matmul = KernelPattern::MatMul {
            inputs: [
                make_tensor("A", TensorRole::Input),
                make_tensor("B", TensorRole::Input),
            ],
            output: make_tensor("mm_out", TensorRole::Output),
            shape: make_matmul_shape(),
        };
        let bias_add = KernelPattern::ElementWise {
            op: ElementWiseOp::Add,
            inputs: [
                make_tensor("mm_out", TensorRole::Input),
                make_tensor("bias", TensorRole::Input),
            ],
            output: make_tensor("out", TensorRole::Output),
            dim_name: "N".into(),
        };
        let gemm = FusedPattern::MatMulBias {
            matmul,
            bias_add: Box::new(bias_add),
        };
        let act_pattern = KernelPattern::Activation {
            op: ActivationOp::Relu,
            input: make_tensor("out", TensorRole::Input),
            output: make_tensor("relu_out", TensorRole::Output),
            dim_name: "N".into(),
        };
        let fused = FusedPattern::WithActivation {
            base: Box::new(gemm),
            activation: FusedActivation::Relu,
            activation_pattern: Box::new(act_pattern),
        };

        let model = build_fused_model(&fused, "gemm_relu", &[]).unwrap();
        let graph = model.graph.as_ref().unwrap();

        // Gemm + Relu = 2 nodes.
        assert_eq!(graph.node.len(), 2);
        assert_eq!(graph.node[0].op_type, "Gemm");
        assert_eq!(graph.node[1].op_type, "Relu");

        assert_eq!(graph.node[0].output, vec!["out_pre_relu"]);
        assert_eq!(graph.node[1].input, vec!["out_pre_relu"]);
        assert_eq!(graph.node[1].output, vec!["out"]);
    }

    #[test]
    fn append_activation_node_relu() {
        // Build a simple model then append Relu.
        let pattern = KernelPattern::ElementWise {
            op: ElementWiseOp::Add,
            inputs: [
                make_tensor("a", TensorRole::Input),
                make_tensor("b", TensorRole::Input),
            ],
            output: make_tensor("c", TensorRole::Output),
            dim_name: "N".into(),
        };
        let mut model = build_model(&pattern, "test", &[]).unwrap();
        append_activation_node(&mut model, "Relu");

        let graph = model.graph.as_ref().unwrap();
        assert_eq!(graph.node.len(), 2);
        assert_eq!(graph.node[1].op_type, "Relu");
        assert_eq!(graph.node[1].name, "relu_0");
        assert_eq!(graph.node[1].input, vec!["c_pre_relu"]);
        assert_eq!(graph.node[1].output, vec!["c"]);
        // Original last node output renamed.
        assert_eq!(graph.node[0].output, vec!["c_pre_relu"]);
        // Graph output retains original name.
        assert_eq!(graph.output[0].name, "c");
    }

    #[test]
    fn append_activation_node_sigmoid() {
        let pattern = KernelPattern::ElementWise {
            op: ElementWiseOp::Add,
            inputs: [
                make_tensor("a", TensorRole::Input),
                make_tensor("b", TensorRole::Input),
            ],
            output: make_tensor("c", TensorRole::Output),
            dim_name: "N".into(),
        };
        let mut model = build_model(&pattern, "test", &[]).unwrap();
        append_activation_node(&mut model, "Sigmoid");

        let graph = model.graph.as_ref().unwrap();
        assert_eq!(graph.node.len(), 2);
        assert_eq!(graph.node[1].op_type, "Sigmoid");
        assert_eq!(graph.node[1].name, "sigmoid_0");
        assert_eq!(graph.node[1].input, vec!["c_pre_sigmoid"]);
        assert_eq!(graph.node[1].output, vec!["c"]);
    }

    #[test]
    fn append_activation_node_tanh() {
        let pattern = KernelPattern::ElementWise {
            op: ElementWiseOp::Add,
            inputs: [
                make_tensor("a", TensorRole::Input),
                make_tensor("b", TensorRole::Input),
            ],
            output: make_tensor("c", TensorRole::Output),
            dim_name: "N".into(),
        };
        let mut model = build_model(&pattern, "test", &[]).unwrap();
        append_activation_node(&mut model, "Tanh");

        let graph = model.graph.as_ref().unwrap();
        assert_eq!(graph.node.len(), 2);
        assert_eq!(graph.node[1].op_type, "Tanh");
        assert_eq!(graph.node[1].name, "tanh_0");
        assert_eq!(graph.node[1].input, vec!["c_pre_tanh"]);
        assert_eq!(graph.node[1].output, vec!["c"]);
    }

    #[test]
    fn fused_conv_batchnorm_with_weights_injected() {
        let conv = KernelPattern::Conv2D {
            input: make_tensor("x", TensorRole::Input),
            weight: make_tensor("w", TensorRole::Input),
            output: make_tensor("conv_out", TensorRole::Output),
            shape: make_conv2d_shape(),
        };
        let norm = KernelPattern::Normalization {
            input: make_tensor("conv_out", TensorRole::Input),
            scale: make_tensor("gamma", TensorRole::Input),
            bias: make_tensor("beta", TensorRole::Input),
            output: make_tensor("bn_out", TensorRole::Output),
            epsilon: 1e-5,
            norm_type: NormType::Batch,
        };
        let fused = FusedPattern::ConvBatchNorm {
            conv,
            norm: Box::new(norm),
        };

        let weights = vec![
            EmbeddedWeight {
                name: "w".into(),
                dims: vec![8, 3, 3, 3],
                data: vec![0.1; 8 * 3 * 3 * 3],
            },
            EmbeddedWeight {
                name: "gamma".into(),
                dims: vec![8],
                data: vec![1.0; 8],
            },
        ];

        let model = build_fused_model(&fused, "conv_bn_w", &weights).unwrap();
        let graph = model.graph.as_ref().unwrap();

        // Both referenced weights should be injected.
        assert_eq!(graph.initializer.len(), 2);
        let init_names: Vec<&str> = graph.initializer.iter().map(|i| i.name.as_str()).collect();
        assert!(init_names.contains(&"w"));
        assert!(init_names.contains(&"gamma"));
    }

    #[test]
    fn fused_matmul_bias_with_weights_injected() {
        let matmul = KernelPattern::MatMul {
            inputs: [
                make_tensor("A", TensorRole::Input),
                make_tensor("B", TensorRole::Input),
            ],
            output: make_tensor("mm_out", TensorRole::Output),
            shape: make_matmul_shape(),
        };
        let bias_add = KernelPattern::ElementWise {
            op: ElementWiseOp::Add,
            inputs: [
                make_tensor("mm_out", TensorRole::Input),
                make_tensor("bias", TensorRole::Input),
            ],
            output: make_tensor("out", TensorRole::Output),
            dim_name: "N".into(),
        };
        let fused = FusedPattern::MatMulBias {
            matmul,
            bias_add: Box::new(bias_add),
        };

        let weights = vec![
            EmbeddedWeight {
                name: "B".into(),
                dims: vec![4, 8],
                data: vec![0.5; 32],
            },
            EmbeddedWeight {
                name: "bias".into(),
                dims: vec![8],
                data: vec![0.1; 8],
            },
            EmbeddedWeight {
                name: "unused".into(),
                dims: vec![2],
                data: vec![9.9, 9.9],
            },
        ];

        let model = build_fused_model(&fused, "gemm_w", &weights).unwrap();
        let graph = model.graph.as_ref().unwrap();

        // Only B and bias are referenced; unused should not appear.
        assert_eq!(graph.initializer.len(), 2);
        let init_names: Vec<&str> = graph.initializer.iter().map(|i| i.name.as_str()).collect();
        assert!(init_names.contains(&"B"));
        assert!(init_names.contains(&"bias"));
        assert!(!init_names.contains(&"unused"));
    }

    #[test]
    fn inject_weights_adds_initializer() {
        let mut graph = GraphProto {
            name: "test".into(),
            initializer: vec![],
            node: vec![NodeProto::simple(
                "Add",
                "add_0",
                vec!["input".into(), "bias".into()],
                vec!["output".into()],
            )],
            input: vec![
                ValueInfoProto::tensor(
                    "input",
                    data_type::FLOAT,
                    vec![TensorShapeDimension::symbolic("N")],
                ),
                ValueInfoProto::tensor(
                    "bias",
                    data_type::FLOAT,
                    vec![TensorShapeDimension::symbolic("N")],
                ),
            ],
            output: vec![ValueInfoProto::tensor(
                "output",
                data_type::FLOAT,
                vec![TensorShapeDimension::symbolic("N")],
            )],
        };

        let weights = vec![
            EmbeddedWeight {
                name: "bias".into(),
                dims: vec![4],
                data: vec![0.1, 0.2, 0.3, 0.4],
            },
            EmbeddedWeight {
                name: "unreferenced".into(),
                dims: vec![2],
                data: vec![1.0, 2.0],
            },
        ];

        inject_weights(&mut graph, &weights);

        // Only "bias" should be added (referenced by the Add node).
        assert_eq!(graph.initializer.len(), 1);
        assert_eq!(graph.initializer[0].name, "bias");
        assert_eq!(graph.initializer[0].dims, vec![4]);
        assert_eq!(graph.initializer[0].data_type, data_type::FLOAT);
        // raw_data = 4 floats * 4 bytes = 16 bytes
        assert_eq!(graph.initializer[0].raw_data.len(), 16);
    }

    #[test]
    fn inject_per_channel_qdq_adds_nodes_and_initializers() {
        // Build a minimal graph with a Conv node referencing "weight".
        let mut graph = GraphProto {
            name: "test".into(),
            initializer: vec![],
            node: vec![NodeProto::simple(
                "Conv",
                "conv_0",
                vec!["input".into(), "weight".into()],
                vec!["output".into()],
            )],
            input: vec![ValueInfoProto::tensor(
                "input",
                data_type::FLOAT,
                vec![
                    TensorShapeDimension::fixed(1),
                    TensorShapeDimension::fixed(3),
                    TensorShapeDimension::fixed(224),
                    TensorShapeDimension::fixed(224),
                ],
            )],
            output: vec![ValueInfoProto::tensor(
                "output",
                data_type::FLOAT,
                vec![
                    TensorShapeDimension::fixed(1),
                    TensorShapeDimension::fixed(16),
                    TensorShapeDimension::fixed(222),
                    TensorShapeDimension::fixed(222),
                ],
            )],
        };

        let params = vec![nxpu_backend_core::PerChannelParam {
            name: "weight".into(),
            scales: vec![0.1, 0.2, 0.3],
            zero_points: vec![0, 0, 0],
            channel_axis: 0,
        }];

        inject_per_channel_qdq(&mut graph, &params);

        // Should have 3 nodes: QuantizeLinear, DequantizeLinear, Conv
        assert_eq!(graph.node.len(), 3);
        assert_eq!(graph.node[0].op_type, "QuantizeLinear");
        assert_eq!(graph.node[0].name, "quantize_weight");
        assert_eq!(graph.node[1].op_type, "DequantizeLinear");
        assert_eq!(graph.node[1].name, "dequantize_weight");
        assert_eq!(graph.node[2].op_type, "Conv");

        // Conv node should now reference the dequantized weight.
        assert_eq!(graph.node[2].input[0], "input");
        assert_eq!(graph.node[2].input[1], "weight_dequantized");

        // Check initializers: scale and zero_point tensors.
        assert_eq!(graph.initializer.len(), 2);
        assert_eq!(graph.initializer[0].name, "weight_scale");
        assert_eq!(graph.initializer[0].data_type, data_type::FLOAT);
        assert_eq!(graph.initializer[0].float_data, vec![0.1, 0.2, 0.3]);
        assert_eq!(graph.initializer[1].name, "weight_zero_point");
        assert_eq!(graph.initializer[1].data_type, data_type::INT8);
        assert_eq!(graph.initializer[1].int32_data, vec![0, 0, 0]);

        // Check axis attribute on QuantizeLinear.
        assert_eq!(graph.node[0].attribute.len(), 1);
        assert_eq!(graph.node[0].attribute[0].name, "axis");
        assert_eq!(graph.node[0].attribute[0].i, 0);
    }

    #[test]
    fn inject_per_channel_qdq_empty_params_is_noop() {
        let mut graph = GraphProto {
            name: "test".into(),
            initializer: vec![],
            node: vec![NodeProto::simple(
                "Conv",
                "conv_0",
                vec!["input".into(), "weight".into()],
                vec!["output".into()],
            )],
            input: vec![],
            output: vec![],
        };

        inject_per_channel_qdq(&mut graph, &[]);

        // Graph should be unchanged.
        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "Conv");
        assert!(graph.initializer.is_empty());
    }

    #[test]
    fn layout_transpose_adds_node() {
        let mut graph = GraphProto {
            name: "test".into(),
            initializer: vec![],
            node: vec![],
            input: vec![],
            output: vec![],
        };
        maybe_add_layout_transpose(&mut graph, &[0, 3, 1, 2], "input_nhwc", "input_nchw");
        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "Transpose");
        assert_eq!(graph.node[0].input, vec!["input_nhwc"]);
        assert_eq!(graph.node[0].output, vec!["input_nchw"]);
        let perm_attr = graph.node[0]
            .attribute
            .iter()
            .find(|a| a.name == "perm")
            .expect("expected perm attribute");
        assert_eq!(perm_attr.ints, vec![0, 3, 1, 2]);
    }

    #[test]
    fn layout_transpose_identity_is_noop() {
        let mut graph = GraphProto {
            name: "test".into(),
            initializer: vec![],
            node: vec![],
            input: vec![],
            output: vec![],
        };
        maybe_add_layout_transpose(&mut graph, &[0, 1, 2, 3], "x", "y");
        assert!(graph.node.is_empty());
    }

    #[test]
    fn layout_transpose_empty_perm_is_noop() {
        let mut graph = GraphProto {
            name: "test".into(),
            initializer: vec![],
            node: vec![],
            input: vec![],
            output: vec![],
        };
        maybe_add_layout_transpose(&mut graph, &[], "x", "y");
        assert!(graph.node.is_empty());
    }
}
