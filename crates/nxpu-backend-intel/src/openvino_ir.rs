//! OpenVINO IR v11 XML emitter.
//!
//! Generates `model.xml` (network description) and `model.bin` (weights placeholder)
//! from classified kernel patterns.

use nxpu_analysis::{
    ActivationOp, ElementWiseOp, KernelPattern, PoolKind, ReduceOp, TensorBinding,
};

/// A layer in the OpenVINO IR graph.
struct IrLayer {
    id: usize,
    name: String,
    layer_type: String,
    /// Layer-specific attributes as `<data key="value" .../>`.
    data_attrs: Vec<(String, String)>,
    /// Input ports: (port_id, dims).
    input_ports: Vec<(usize, Vec<String>)>,
    /// Output ports: (port_id, dims).
    output_ports: Vec<(usize, Vec<String>)>,
}

/// An edge connecting two layers.
struct IrEdge {
    from_layer: usize,
    from_port: usize,
    to_layer: usize,
    to_port: usize,
}

/// Build OpenVINO IR v11 XML from a list of classified patterns.
pub fn build_ir_xml(patterns: &[KernelPattern], model_name: &str) -> String {
    let mut layers = Vec::new();
    let mut edges = Vec::new();
    let mut layer_id = 0;

    // Create Parameter (input) layers and Compute layers for each pattern.
    for pattern in patterns {
        let (input_layer_ids, compute_layer) =
            build_pattern_layers(pattern, &mut layer_id, &mut layers);

        let compute_id = compute_layer.id;
        layers.push(compute_layer);

        // Wire input layers to compute layer input ports.
        for (port_idx, &input_id) in input_layer_ids.iter().enumerate() {
            edges.push(IrEdge {
                from_layer: input_id,
                from_port: 0,
                to_layer: compute_id,
                to_port: port_idx,
            });
        }

        // Create Result (output) layer.
        let result_id = layer_id;
        layer_id += 1;
        layers.push(IrLayer {
            id: result_id,
            name: format!("result_{result_id}"),
            layer_type: "Result".into(),
            data_attrs: vec![],
            input_ports: vec![(0, vec!["?".into()])],
            output_ports: vec![],
        });
        edges.push(IrEdge {
            from_layer: compute_id,
            from_port: 0,
            to_layer: result_id,
            to_port: 0,
        });
    }

    format_ir_xml(&layers, &edges, model_name)
}

/// Build OpenVINO Parameter + compute layers for a single pattern.
/// Returns (input_layer_ids, compute_layer).
fn build_pattern_layers(
    pattern: &KernelPattern,
    layer_id: &mut usize,
    layers: &mut Vec<IrLayer>,
) -> (Vec<usize>, IrLayer) {
    match pattern {
        KernelPattern::MatMul { inputs, output, .. } => {
            let input_ids = create_parameter_layers(&[&inputs[0], &inputs[1]], layer_id, layers);
            let compute = IrLayer {
                id: next_id(layer_id),
                name: format!("matmul_{}", layer_id.wrapping_sub(1)),
                layer_type: "MatMul".into(),
                data_attrs: vec![],
                input_ports: vec![(0, tensor_dims(&inputs[0])), (1, tensor_dims(&inputs[1]))],
                output_ports: vec![(0, tensor_dims(output))],
            };
            (input_ids, compute)
        }
        KernelPattern::Conv2D {
            input,
            weight,
            output,
            shape,
        } => {
            let input_ids = create_parameter_layers(&[input, weight], layer_id, layers);
            let compute = IrLayer {
                id: next_id(layer_id),
                name: format!("conv_{}", layer_id.wrapping_sub(1)),
                layer_type: "Convolution".into(),
                data_attrs: vec![
                    (
                        "strides".into(),
                        format!("{},{}", shape.stride_h, shape.stride_w),
                    ),
                    (
                        "pads_begin".into(),
                        format!("{},{}", shape.pad_h, shape.pad_w),
                    ),
                    (
                        "pads_end".into(),
                        format!("{},{}", shape.pad_h, shape.pad_w),
                    ),
                    (
                        "kernel".into(),
                        format!("{},{}", shape.kernel_h_val, shape.kernel_w_val),
                    ),
                ],
                input_ports: vec![(0, tensor_dims(input)), (1, tensor_dims(weight))],
                output_ports: vec![(0, tensor_dims(output))],
            };
            (input_ids, compute)
        }
        KernelPattern::ElementWise {
            op, inputs, output, ..
        } => {
            let layer_type = match op {
                ElementWiseOp::Add => "Add",
                ElementWiseOp::Sub => "Subtract",
                ElementWiseOp::Mul => "Multiply",
                ElementWiseOp::Div => "Divide",
            };
            let input_ids = create_parameter_layers(&[&inputs[0], &inputs[1]], layer_id, layers);
            let compute = IrLayer {
                id: next_id(layer_id),
                name: format!("{}_{}", layer_type.to_lowercase(), layer_id.wrapping_sub(1)),
                layer_type: layer_type.into(),
                data_attrs: vec![],
                input_ports: vec![(0, tensor_dims(&inputs[0])), (1, tensor_dims(&inputs[1]))],
                output_ports: vec![(0, tensor_dims(output))],
            };
            (input_ids, compute)
        }
        KernelPattern::Activation {
            op, input, output, ..
        } => {
            let layer_type = match op {
                ActivationOp::Relu => "ReLU",
                ActivationOp::Sigmoid => "Sigmoid",
                ActivationOp::Tanh => "Tanh",
                ActivationOp::Softmax => "SoftMax",
            };
            let input_ids = create_parameter_layers(&[input], layer_id, layers);
            let compute = IrLayer {
                id: next_id(layer_id),
                name: format!("{}_{}", layer_type.to_lowercase(), layer_id.wrapping_sub(1)),
                layer_type: layer_type.into(),
                data_attrs: vec![],
                input_ports: vec![(0, tensor_dims(input))],
                output_ports: vec![(0, tensor_dims(output))],
            };
            (input_ids, compute)
        }
        KernelPattern::Pool {
            kind,
            input,
            output,
            shape,
        } => {
            let layer_type = match kind {
                PoolKind::Max => "MaxPool",
                PoolKind::Avg => "AvgPool",
            };
            let input_ids = create_parameter_layers(&[input], layer_id, layers);
            let compute = IrLayer {
                id: next_id(layer_id),
                name: format!("pool_{}", layer_id.wrapping_sub(1)),
                layer_type: layer_type.into(),
                data_attrs: vec![
                    (
                        "kernel".into(),
                        format!("{},{}", shape.kernel_h, shape.kernel_w),
                    ),
                    (
                        "strides".into(),
                        format!("{},{}", shape.stride_h, shape.stride_w),
                    ),
                    ("pads_begin".into(), "0,0".into()),
                    ("pads_end".into(), "0,0".into()),
                ],
                input_ports: vec![(0, tensor_dims(input))],
                output_ports: vec![(0, tensor_dims(output))],
            };
            (input_ids, compute)
        }
        KernelPattern::Reduce {
            op,
            input,
            output,
            axis,
            ..
        } => {
            let layer_type = match op {
                ReduceOp::Sum => "ReduceSum",
                ReduceOp::Mean => "ReduceMean",
                ReduceOp::Max => "ReduceMax",
                ReduceOp::Min => "ReduceMin",
            };
            let input_ids = create_parameter_layers(&[input], layer_id, layers);
            let compute = IrLayer {
                id: next_id(layer_id),
                name: format!("reduce_{}", layer_id.wrapping_sub(1)),
                layer_type: layer_type.into(),
                data_attrs: vec![("axis".into(), axis.to_string())],
                input_ports: vec![(0, tensor_dims(input))],
                output_ports: vec![(0, tensor_dims(output))],
            };
            (input_ids, compute)
        }
        KernelPattern::Normalization {
            input,
            scale,
            bias,
            output,
            epsilon,
        } => {
            let input_ids = create_parameter_layers(&[input, scale, bias], layer_id, layers);
            let compute = IrLayer {
                id: next_id(layer_id),
                name: format!("batchnorm_{}", layer_id.wrapping_sub(1)),
                layer_type: "BatchNormInference".into(),
                data_attrs: vec![("epsilon".into(), epsilon.to_string())],
                input_ports: vec![
                    (0, tensor_dims(input)),
                    (1, tensor_dims(scale)),
                    (2, tensor_dims(bias)),
                ],
                output_ports: vec![(0, tensor_dims(output))],
            };
            (input_ids, compute)
        }
        // For patterns not directly mapped, emit a generic placeholder
        _ => {
            let (inputs, output_name) = generic_io(pattern);
            let input_bindings: Vec<&TensorBinding> = inputs.iter().collect();
            let input_ids = create_parameter_layers(&input_bindings, layer_id, layers);
            let in_ports = input_ids
                .iter()
                .enumerate()
                .map(|(i, _)| (i, vec!["?".into()]))
                .collect();
            let compute = IrLayer {
                id: next_id(layer_id),
                name: format!("op_{}", layer_id.wrapping_sub(1)),
                layer_type: output_name,
                data_attrs: vec![],
                input_ports: in_ports,
                output_ports: vec![(0, vec!["?".into()])],
            };
            (input_ids, compute)
        }
    }
}

/// Create Parameter layers for each input binding.
fn create_parameter_layers(
    inputs: &[&TensorBinding],
    layer_id: &mut usize,
    layers: &mut Vec<IrLayer>,
) -> Vec<usize> {
    inputs
        .iter()
        .map(|tb| {
            let id = next_id(layer_id);
            layers.push(IrLayer {
                id,
                name: tb.name.clone(),
                layer_type: "Parameter".into(),
                data_attrs: vec![],
                input_ports: vec![],
                output_ports: vec![(0, tensor_dims(tb))],
            });
            id
        })
        .collect()
}

fn next_id(id: &mut usize) -> usize {
    let current = *id;
    *id += 1;
    current
}

fn tensor_dims(tb: &TensorBinding) -> Vec<String> {
    // Use the tensor name as a symbolic dimension placeholder.
    vec![tb.name.clone()]
}

/// Extract input/output info from generic patterns.
fn generic_io(pattern: &KernelPattern) -> (Vec<TensorBinding>, String) {
    match pattern {
        KernelPattern::Transpose { input, .. } => (vec![input.clone()], "Transpose".into()),
        KernelPattern::Reshape { input, .. } => (vec![input.clone()], "Reshape".into()),
        KernelPattern::Concat { inputs, .. } => (inputs.clone(), "Concat".into()),
        KernelPattern::Split { input, .. } => (vec![input.clone()], "Split".into()),
        KernelPattern::Attention {
            query, key, value, ..
        } => (
            vec![query.clone(), key.clone(), value.clone()],
            "ScaledDotProductAttention".into(),
        ),
        KernelPattern::Unknown { reason } => (vec![], format!("Unknown({reason})")),
        // Covered patterns should not reach here.
        _ => (vec![], "Unknown".into()),
    }
}

/// Format layers and edges into OpenVINO IR v11 XML.
fn format_ir_xml(layers: &[IrLayer], edges: &[IrEdge], model_name: &str) -> String {
    let mut xml = String::new();
    xml.push_str("<?xml version=\"1.0\"?>\n");
    xml.push_str(&format!("<net name=\"{model_name}\" version=\"11\">\n"));
    xml.push_str("  <layers>\n");

    for layer in layers {
        xml.push_str(&format!(
            "    <layer id=\"{}\" name=\"{}\" type=\"{}\">\n",
            layer.id, layer.name, layer.layer_type
        ));

        if !layer.data_attrs.is_empty() {
            xml.push_str("      <data");
            for (key, value) in &layer.data_attrs {
                xml.push_str(&format!(" {key}=\"{value}\""));
            }
            xml.push_str("/>\n");
        }

        if !layer.input_ports.is_empty() {
            xml.push_str("      <input>\n");
            for (port_id, _dims) in &layer.input_ports {
                xml.push_str(&format!("        <port id=\"{port_id}\"/>\n"));
            }
            xml.push_str("      </input>\n");
        }

        if !layer.output_ports.is_empty() {
            xml.push_str("      <output>\n");
            for (port_id, _dims) in &layer.output_ports {
                xml.push_str(&format!("        <port id=\"{port_id}\"/>\n"));
            }
            xml.push_str("      </output>\n");
        }

        xml.push_str("    </layer>\n");
    }

    xml.push_str("  </layers>\n");
    xml.push_str("  <edges>\n");

    for edge in edges {
        xml.push_str(&format!(
            "    <edge from-layer=\"{}\" from-port=\"{}\" to-layer=\"{}\" to-port=\"{}\"/>\n",
            edge.from_layer, edge.from_port, edge.to_layer, edge.to_port
        ));
    }

    xml.push_str("  </edges>\n");
    xml.push_str("</net>\n");
    xml
}

#[cfg(test)]
mod tests {
    use super::*;
    use nxpu_analysis::{MatMulShape, TensorRole};

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

    fn dummy_binding(name: &str, role: TensorRole) -> TensorBinding {
        TensorBinding {
            handle: dummy_handle(),
            name: name.into(),
            elem_type: 1,
            role,
        }
    }

    #[test]
    fn matmul_ir_xml() {
        let patterns = vec![KernelPattern::MatMul {
            inputs: [
                dummy_binding("A", TensorRole::Input),
                dummy_binding("B", TensorRole::Input),
            ],
            output: dummy_binding("C", TensorRole::Output),
            shape: MatMulShape {
                m: "M".into(),
                n: "N".into(),
                k: "K".into(),
            },
        }];

        let xml = build_ir_xml(&patterns, "test_matmul");
        assert!(xml.contains("<net name=\"test_matmul\" version=\"11\">"));
        assert!(xml.contains("<layer"));
        assert!(xml.contains("type=\"MatMul\""));
        assert!(xml.contains("type=\"Parameter\""));
        assert!(xml.contains("type=\"Result\""));
        assert!(xml.contains("<edge"));
    }

    #[test]
    fn elementwise_add_ir_xml() {
        let patterns = vec![KernelPattern::ElementWise {
            op: ElementWiseOp::Add,
            inputs: [
                dummy_binding("X", TensorRole::Input),
                dummy_binding("Y", TensorRole::Input),
            ],
            output: dummy_binding("Z", TensorRole::Output),
            dim_name: "N".into(),
        }];

        let xml = build_ir_xml(&patterns, "test_add");
        assert!(xml.contains("type=\"Add\""));
    }

    #[test]
    fn relu_ir_xml() {
        let patterns = vec![KernelPattern::Activation {
            op: ActivationOp::Relu,
            input: dummy_binding("X", TensorRole::Input),
            output: dummy_binding("Y", TensorRole::Output),
            dim_name: "N".into(),
        }];

        let xml = build_ir_xml(&patterns, "test_relu");
        assert!(xml.contains("type=\"ReLU\""));
    }

    use nxpu_analysis::{Conv2DShape, PoolKind, PoolShape, ReduceOp};

    #[test]
    fn conv2d_ir_xml() {
        let patterns = vec![KernelPattern::Conv2D {
            input: dummy_binding("input", TensorRole::Input),
            weight: dummy_binding("weight", TensorRole::Input),
            output: dummy_binding("output", TensorRole::Output),
            shape: Conv2DShape {
                batch: "N".into(),
                channels_in: "C".into(),
                channels_out: "K".into(),
                height: "H".into(),
                width: "W".into(),
                kernel_h: "KH".into(),
                kernel_w: "KW".into(),
                kernel_h_val: 3,
                kernel_w_val: 3,
                stride_h: 1,
                stride_w: 1,
                pad_h: 1,
                pad_w: 1,
            },
        }];

        let xml = build_ir_xml(&patterns, "test_conv2d");
        assert!(xml.contains("type=\"Convolution\""));
        assert!(xml.contains("strides=\"1,1\""));
        assert!(xml.contains("kernel=\"3,3\""));
    }

    #[test]
    fn elementwise_sub_ir_xml() {
        let patterns = vec![KernelPattern::ElementWise {
            op: ElementWiseOp::Sub,
            inputs: [
                dummy_binding("X", TensorRole::Input),
                dummy_binding("Y", TensorRole::Input),
            ],
            output: dummy_binding("Z", TensorRole::Output),
            dim_name: "N".into(),
        }];

        let xml = build_ir_xml(&patterns, "test_sub");
        assert!(xml.contains("type=\"Subtract\""));
    }

    #[test]
    fn elementwise_mul_ir_xml() {
        let patterns = vec![KernelPattern::ElementWise {
            op: ElementWiseOp::Mul,
            inputs: [
                dummy_binding("X", TensorRole::Input),
                dummy_binding("Y", TensorRole::Input),
            ],
            output: dummy_binding("Z", TensorRole::Output),
            dim_name: "N".into(),
        }];

        let xml = build_ir_xml(&patterns, "test_mul");
        assert!(xml.contains("type=\"Multiply\""));
    }

    #[test]
    fn elementwise_div_ir_xml() {
        let patterns = vec![KernelPattern::ElementWise {
            op: ElementWiseOp::Div,
            inputs: [
                dummy_binding("X", TensorRole::Input),
                dummy_binding("Y", TensorRole::Input),
            ],
            output: dummy_binding("Z", TensorRole::Output),
            dim_name: "N".into(),
        }];

        let xml = build_ir_xml(&patterns, "test_div");
        assert!(xml.contains("type=\"Divide\""));
    }

    #[test]
    fn sigmoid_ir_xml() {
        let patterns = vec![KernelPattern::Activation {
            op: ActivationOp::Sigmoid,
            input: dummy_binding("X", TensorRole::Input),
            output: dummy_binding("Y", TensorRole::Output),
            dim_name: "N".into(),
        }];

        let xml = build_ir_xml(&patterns, "test_sigmoid");
        assert!(xml.contains("type=\"Sigmoid\""));
    }

    #[test]
    fn tanh_ir_xml() {
        let patterns = vec![KernelPattern::Activation {
            op: ActivationOp::Tanh,
            input: dummy_binding("X", TensorRole::Input),
            output: dummy_binding("Y", TensorRole::Output),
            dim_name: "N".into(),
        }];

        let xml = build_ir_xml(&patterns, "test_tanh");
        assert!(xml.contains("type=\"Tanh\""));
    }

    #[test]
    fn softmax_ir_xml() {
        let patterns = vec![KernelPattern::Activation {
            op: ActivationOp::Softmax,
            input: dummy_binding("X", TensorRole::Input),
            output: dummy_binding("Y", TensorRole::Output),
            dim_name: "N".into(),
        }];

        let xml = build_ir_xml(&patterns, "test_softmax");
        assert!(xml.contains("type=\"SoftMax\""));
    }

    #[test]
    fn maxpool_ir_xml() {
        let patterns = vec![KernelPattern::Pool {
            kind: PoolKind::Max,
            input: dummy_binding("input", TensorRole::Input),
            output: dummy_binding("output", TensorRole::Output),
            shape: PoolShape {
                kernel_h: 2,
                kernel_w: 2,
                stride_h: 2,
                stride_w: 2,
            },
        }];

        let xml = build_ir_xml(&patterns, "test_maxpool");
        assert!(xml.contains("type=\"MaxPool\""));
    }

    #[test]
    fn avgpool_ir_xml() {
        let patterns = vec![KernelPattern::Pool {
            kind: PoolKind::Avg,
            input: dummy_binding("input", TensorRole::Input),
            output: dummy_binding("output", TensorRole::Output),
            shape: PoolShape {
                kernel_h: 2,
                kernel_w: 2,
                stride_h: 2,
                stride_w: 2,
            },
        }];

        let xml = build_ir_xml(&patterns, "test_avgpool");
        assert!(xml.contains("type=\"AvgPool\""));
    }

    #[test]
    fn reduce_sum_ir_xml() {
        let patterns = vec![KernelPattern::Reduce {
            op: ReduceOp::Sum,
            input: dummy_binding("input", TensorRole::Input),
            output: dummy_binding("output", TensorRole::Output),
            axis: 0,
        }];

        let xml = build_ir_xml(&patterns, "test_reduce_sum");
        assert!(xml.contains("type=\"ReduceSum\""));
        assert!(xml.contains("axis=\"0\""));
    }

    #[test]
    fn reduce_mean_ir_xml() {
        let patterns = vec![KernelPattern::Reduce {
            op: ReduceOp::Mean,
            input: dummy_binding("input", TensorRole::Input),
            output: dummy_binding("output", TensorRole::Output),
            axis: 1,
        }];

        let xml = build_ir_xml(&patterns, "test_reduce_mean");
        assert!(xml.contains("type=\"ReduceMean\""));
    }

    #[test]
    fn normalization_ir_xml() {
        let patterns = vec![KernelPattern::Normalization {
            input: dummy_binding("input", TensorRole::Input),
            scale: dummy_binding("scale", TensorRole::Input),
            bias: dummy_binding("bias", TensorRole::Input),
            output: dummy_binding("output", TensorRole::Output),
            epsilon: 1e-5,
        }];

        let xml = build_ir_xml(&patterns, "test_batchnorm");
        assert!(xml.contains("type=\"BatchNormInference\""));
        assert!(xml.contains("epsilon"));
    }

    #[test]
    fn transpose_ir_xml() {
        let patterns = vec![KernelPattern::Transpose {
            input: dummy_binding("input", TensorRole::Input),
            output: dummy_binding("output", TensorRole::Output),
            perm: vec![1, 0],
        }];

        let xml = build_ir_xml(&patterns, "test_transpose");
        assert!(xml.contains("type=\"Transpose\""));
    }

    #[test]
    fn attention_ir_xml() {
        let patterns = vec![KernelPattern::Attention {
            query: dummy_binding("Q", TensorRole::Input),
            key: dummy_binding("K", TensorRole::Input),
            value: dummy_binding("V", TensorRole::Input),
            output: dummy_binding("out", TensorRole::Output),
            d_k: "D".into(),
            seq_len: "S".into(),
        }];

        let xml = build_ir_xml(&patterns, "test_attention");
        assert!(xml.contains("type=\"ScaledDotProductAttention\""));
    }

    #[test]
    fn unknown_ir_xml() {
        let patterns = vec![KernelPattern::Unknown {
            reason: "test".into(),
        }];

        let xml = build_ir_xml(&patterns, "test_unknown");
        assert!(xml.contains("type=\"Unknown(test)\""));
    }

    #[test]
    fn concat_ir_xml() {
        let patterns = vec![KernelPattern::Concat {
            inputs: vec![
                dummy_binding("A", TensorRole::Input),
                dummy_binding("B", TensorRole::Input),
            ],
            output: dummy_binding("out", TensorRole::Output),
            axis: 0,
        }];

        let xml = build_ir_xml(&patterns, "test_concat");
        assert!(xml.contains("type=\"Concat\""));
    }
}
