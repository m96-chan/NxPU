//! Operator fusion pass.
//!
//! Fuses adjacent operations in the compute graph into single fused
//! operations to reduce memory traffic and kernel launch overhead.
//!
//! Supported fusion patterns:
//! - **Conv2D + Bias + Activation** → `FusedConv2d` with activation attribute
//! - **MatMul + Bias (Add)** → `Gemm` (with beta=1)
//! - **ElementWise + Activation** → `FusedElementWise` with activation

use std::collections::HashMap;

use nxpu_ir::graph::{ActivationFunction, ComputeGraph, EdgeId, GraphOp};

use crate::Pass;

/// Fuses adjacent operations in the compute graph into single fused ops.
#[derive(Debug)]
pub struct OperatorFusion;

impl Pass for OperatorFusion {
    fn name(&self) -> &str {
        "operator-fusion"
    }

    fn run(&self, module: &mut nxpu_ir::Module) -> bool {
        // This pass is a no-op on Module (expression-level IR).
        // It operates on ComputeGraph via `run_on_graph`.
        let _ = module;
        false
    }
}

impl OperatorFusion {
    /// Run the fusion pass on a compute graph.
    /// Returns `true` if any fusion was performed.
    pub fn run_on_graph(&self, graph: &mut ComputeGraph) -> bool {
        let mut changed = false;

        // Keep running until no more fusions are possible (fixed-point).
        loop {
            let fused_any = try_fuse_conv_bias_activation(graph)
                | try_fuse_matmul_bias(graph)
                | try_fuse_elementwise_activation(graph);

            if fused_any {
                changed = true;
            } else {
                break;
            }
        }

        changed
    }
}

/// Returns `true` if the given `GraphOp` is an activation function.
fn is_activation(op: &GraphOp) -> Option<ActivationFunction> {
    match op {
        GraphOp::Relu => Some(ActivationFunction::Relu),
        GraphOp::Sigmoid => Some(ActivationFunction::Sigmoid),
        _ => None,
    }
}

/// Returns `true` if the given `GraphOp` is an element-wise binary op.
fn is_elementwise_binary(op: &GraphOp) -> bool {
    matches!(
        op,
        GraphOp::Add | GraphOp::Sub | GraphOp::Mul | GraphOp::Div
    )
}

/// Build a map from edge → list of consumer node indices.
fn build_edge_consumer_map(graph: &ComputeGraph) -> HashMap<EdgeId, Vec<usize>> {
    let mut map: HashMap<EdgeId, Vec<usize>> = HashMap::new();
    for (i, node) in graph.nodes.iter().enumerate() {
        for &inp in &node.inputs {
            map.entry(inp).or_default().push(i);
        }
    }
    map
}

/// Check if an edge has exactly one consumer in the graph.
fn has_single_consumer(edge: EdgeId, consumer_map: &HashMap<EdgeId, Vec<usize>>) -> bool {
    match consumer_map.get(&edge) {
        Some(consumers) => consumers.len() == 1,
        None => false,
    }
}

/// Remove a node from the graph by index and rewire edges.
///
/// The consumer node is removed, and the graph output edges of the removed
/// node become the output edges of the surviving (fused) node.
fn remove_node_and_rewire(
    graph: &mut ComputeGraph,
    remove_idx: usize,
    fused_idx: usize,
    intermediate_edge: EdgeId,
) {
    // The fused node takes the output edges from the removed node.
    let new_outputs = graph.nodes[remove_idx].outputs.clone();
    graph.nodes[fused_idx].outputs = new_outputs;

    // Remove the intermediate edge from graph inputs/outputs if present.
    graph.inputs.retain(|e| *e != intermediate_edge);
    graph.outputs.retain(|e| *e != intermediate_edge);

    // Remove the consumed node.
    graph.nodes.remove(remove_idx);

    // Clean up the intermediate edge from the edges map.
    graph.edges.remove(&intermediate_edge);
}

/// Try to fuse Conv2D + (optional Add/bias) + (optional activation).
///
/// Pattern: Conv2d → Add(conv_out, bias) → Relu/Sigmoid
/// Result:  FusedConv2d { activation: Relu } with inputs [input, weight, bias]
fn try_fuse_conv_bias_activation(graph: &mut ComputeGraph) -> bool {
    let edge_consumer = build_edge_consumer_map(graph);

    // Find Conv2D nodes.
    for conv_idx in 0..graph.nodes.len() {
        if graph.nodes[conv_idx].op != GraphOp::Conv2d {
            continue;
        }

        // Conv2d must have exactly one output edge.
        if graph.nodes[conv_idx].outputs.len() != 1 {
            continue;
        }
        let conv_out_edge = graph.nodes[conv_idx].outputs[0];

        // The output must have a single consumer.
        if !has_single_consumer(conv_out_edge, &edge_consumer) {
            continue;
        }

        let consumer_indices = &edge_consumer[&conv_out_edge];
        let next_idx = consumer_indices[0];

        // Check if the consumer is an Add (bias) or an activation.
        match &graph.nodes[next_idx].op {
            GraphOp::Add => {
                // Conv2d + Add (bias pattern). Try to find a subsequent activation.
                let add_idx = next_idx;
                if graph.nodes[add_idx].outputs.len() != 1 {
                    continue;
                }
                let add_out_edge = graph.nodes[add_idx].outputs[0];

                // Get the bias input (the non-conv input to Add).
                let bias_edge = graph.nodes[add_idx]
                    .inputs
                    .iter()
                    .find(|&&e| e != conv_out_edge)
                    .copied();

                let Some(bias_edge) = bias_edge else {
                    continue;
                };

                // Add the bias to the conv inputs.
                let mut fused_inputs = graph.nodes[conv_idx].inputs.clone();
                fused_inputs.push(bias_edge);

                // Check if there's an activation after the Add.
                let activation = if has_single_consumer(add_out_edge, &edge_consumer) {
                    let act_consumers = &edge_consumer[&add_out_edge];
                    let act_idx = act_consumers[0];
                    is_activation(&graph.nodes[act_idx].op)
                } else {
                    None
                };

                let act = activation.unwrap_or(ActivationFunction::None);

                // Create the fused op.
                graph.nodes[conv_idx].op = GraphOp::FusedConv2d { activation: act };
                graph.nodes[conv_idx].inputs = fused_inputs;
                graph.nodes[conv_idx].name = format!("{}_fused", graph.nodes[conv_idx].name);

                remove_node_and_rewire(graph, add_idx, conv_idx, conv_out_edge);

                // If activation was found, remove it too.
                if activation.is_some() {
                    remove_activation_after_fused_conv(graph, add_out_edge, conv_idx);
                }

                return true;
            }
            op if is_activation(op).is_some() => {
                // Conv2d + Activation (no bias).
                let act = is_activation(op).unwrap();
                let act_idx = next_idx;

                graph.nodes[conv_idx].op = GraphOp::FusedConv2d { activation: act };
                graph.nodes[conv_idx].name = format!("{}_fused", graph.nodes[conv_idx].name);

                remove_node_and_rewire(graph, act_idx, conv_idx, conv_out_edge);

                return true;
            }
            _ => {}
        }
    }

    false
}

/// After fusing Conv+Bias, remove the activation node that follows.
fn remove_activation_after_fused_conv(
    graph: &mut ComputeGraph,
    add_out_edge: EdgeId,
    conv_idx: usize,
) {
    let edge_consumer_new = build_edge_consumer_map(graph);
    let act_consumers = match edge_consumer_new.get(&add_out_edge) {
        Some(c) if c.len() == 1 => c,
        _ => return,
    };

    let act_idx = act_consumers[0];
    let fused_idx = graph
        .nodes
        .iter()
        .position(|n| {
            matches!(n.op, GraphOp::FusedConv2d { .. }) && n.outputs.contains(&add_out_edge)
        })
        .unwrap_or(conv_idx.min(graph.nodes.len() - 1));

    remove_node_and_rewire(graph, act_idx, fused_idx, add_out_edge);
}

/// Try to fuse MatMul + Add → Gemm.
///
/// Pattern: MatMul(A, B) → Add(matmul_out, C) where C is the bias
/// Result:  Gemm { alpha: 1, beta: 1 } with inputs [A, B, C]
fn try_fuse_matmul_bias(graph: &mut ComputeGraph) -> bool {
    let edge_consumer = build_edge_consumer_map(graph);

    for mm_idx in 0..graph.nodes.len() {
        if graph.nodes[mm_idx].op != GraphOp::MatMul {
            continue;
        }

        if graph.nodes[mm_idx].outputs.len() != 1 {
            continue;
        }
        let mm_out_edge = graph.nodes[mm_idx].outputs[0];

        if !has_single_consumer(mm_out_edge, &edge_consumer) {
            continue;
        }

        let consumer_indices = &edge_consumer[&mm_out_edge];
        let add_idx = consumer_indices[0];

        if graph.nodes[add_idx].op != GraphOp::Add {
            continue;
        }

        // Get the bias input (the non-matmul input to Add).
        let bias_edge = graph.nodes[add_idx]
            .inputs
            .iter()
            .find(|&&e| e != mm_out_edge)
            .copied();

        let Some(bias_edge) = bias_edge else {
            continue;
        };

        // Fuse: MatMul + Add → Gemm.
        let mut gemm_inputs = graph.nodes[mm_idx].inputs.clone();
        gemm_inputs.push(bias_edge);

        graph.nodes[mm_idx].op = GraphOp::Gemm { alpha: 1, beta: 1 };
        graph.nodes[mm_idx].inputs = gemm_inputs;
        graph.nodes[mm_idx].name = format!("{}_fused", graph.nodes[mm_idx].name);

        remove_node_and_rewire(graph, add_idx, mm_idx, mm_out_edge);

        return true;
    }

    false
}

/// Try to fuse ElementWise + Activation.
///
/// Pattern: Add/Sub/Mul/Div → Relu/Sigmoid
/// Result:  FusedElementWise { base_op, activation }
fn try_fuse_elementwise_activation(graph: &mut ComputeGraph) -> bool {
    let edge_consumer = build_edge_consumer_map(graph);

    for ew_idx in 0..graph.nodes.len() {
        if !is_elementwise_binary(&graph.nodes[ew_idx].op) {
            continue;
        }

        if graph.nodes[ew_idx].outputs.len() != 1 {
            continue;
        }
        let ew_out_edge = graph.nodes[ew_idx].outputs[0];

        if !has_single_consumer(ew_out_edge, &edge_consumer) {
            continue;
        }

        let consumer_indices = &edge_consumer[&ew_out_edge];
        let act_idx = consumer_indices[0];

        let act = match is_activation(&graph.nodes[act_idx].op) {
            Some(a) => a,
            None => continue,
        };

        // Fuse: ElementWise + Activation → FusedElementWise.
        let base_op = graph.nodes[ew_idx].op.clone();
        graph.nodes[ew_idx].op = GraphOp::FusedElementWise {
            base_op: Box::new(base_op),
            activation: act,
        };
        graph.nodes[ew_idx].name = format!("{}_fused", graph.nodes[ew_idx].name);

        remove_node_and_rewire(graph, act_idx, ew_idx, ew_out_edge);

        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use nxpu_ir::graph::{ComputeGraph, GraphOp, TensorInfo};
    use nxpu_ir::{Dimension, Scalar, TensorShape};

    fn make_tensor(name: &str, shape: &[i64]) -> TensorInfo {
        TensorInfo {
            name: name.into(),
            scalar: Scalar::F32,
            shape: TensorShape {
                dims: shape
                    .iter()
                    .map(|&d| {
                        if d < 0 {
                            Dimension::Dynamic(None)
                        } else {
                            Dimension::Fixed(d as u32)
                        }
                    })
                    .collect(),
            },
        }
    }

    #[test]
    fn fuse_conv_relu() {
        let mut graph = ComputeGraph::new();

        let input = graph.add_edge(make_tensor("input", &[-1, 3, 224, 224]));
        let weight = graph.add_edge(make_tensor("weight", &[64, 3, 3, 3]));
        let conv_out = graph.add_edge(make_tensor("conv_out", &[-1, 64, 222, 222]));
        let relu_out = graph.add_edge(make_tensor("relu_out", &[-1, 64, 222, 222]));

        graph.inputs = vec![input, weight];
        graph.outputs = vec![relu_out];

        graph
            .add_node(GraphOp::Conv2d, vec![input, weight], vec![conv_out], "conv")
            .unwrap();
        graph
            .add_node(GraphOp::Relu, vec![conv_out], vec![relu_out], "relu")
            .unwrap();

        assert_eq!(graph.node_count(), 2);

        let fusion = OperatorFusion;
        let changed = fusion.run_on_graph(&mut graph);

        assert!(changed);
        assert_eq!(graph.node_count(), 1);
        assert_eq!(
            graph.nodes[0].op,
            GraphOp::FusedConv2d {
                activation: ActivationFunction::Relu
            }
        );
        // The fused node should output relu_out, not conv_out.
        assert_eq!(graph.nodes[0].outputs, vec![relu_out]);
    }

    #[test]
    fn fuse_conv_bias_relu() {
        let mut graph = ComputeGraph::new();

        let input = graph.add_edge(make_tensor("input", &[-1, 3, 224, 224]));
        let weight = graph.add_edge(make_tensor("weight", &[64, 3, 3, 3]));
        let bias = graph.add_edge(make_tensor("bias", &[64]));
        let conv_out = graph.add_edge(make_tensor("conv_out", &[-1, 64, 222, 222]));
        let add_out = graph.add_edge(make_tensor("add_out", &[-1, 64, 222, 222]));
        let relu_out = graph.add_edge(make_tensor("relu_out", &[-1, 64, 222, 222]));

        graph.inputs = vec![input, weight, bias];
        graph.outputs = vec![relu_out];

        graph
            .add_node(GraphOp::Conv2d, vec![input, weight], vec![conv_out], "conv")
            .unwrap();
        graph
            .add_node(
                GraphOp::Add,
                vec![conv_out, bias],
                vec![add_out],
                "bias_add",
            )
            .unwrap();
        graph
            .add_node(GraphOp::Relu, vec![add_out], vec![relu_out], "relu")
            .unwrap();

        assert_eq!(graph.node_count(), 3);

        let fusion = OperatorFusion;
        let changed = fusion.run_on_graph(&mut graph);

        assert!(changed);
        assert_eq!(graph.node_count(), 1);
        assert_eq!(
            graph.nodes[0].op,
            GraphOp::FusedConv2d {
                activation: ActivationFunction::Relu
            }
        );
        // Should include bias in inputs.
        assert_eq!(graph.nodes[0].inputs, vec![input, weight, bias]);
        assert_eq!(graph.nodes[0].outputs, vec![relu_out]);
    }

    #[test]
    fn fuse_matmul_add_to_gemm() {
        let mut graph = ComputeGraph::new();

        let a = graph.add_edge(make_tensor("A", &[-1, 768]));
        let b = graph.add_edge(make_tensor("B", &[768, 768]));
        let mm_out = graph.add_edge(make_tensor("mm_out", &[-1, 768]));
        let bias = graph.add_edge(make_tensor("bias", &[768]));
        let add_out = graph.add_edge(make_tensor("add_out", &[-1, 768]));

        graph.inputs = vec![a, b, bias];
        graph.outputs = vec![add_out];

        graph
            .add_node(GraphOp::MatMul, vec![a, b], vec![mm_out], "matmul")
            .unwrap();
        graph
            .add_node(GraphOp::Add, vec![mm_out, bias], vec![add_out], "add")
            .unwrap();

        assert_eq!(graph.node_count(), 2);

        let fusion = OperatorFusion;
        let changed = fusion.run_on_graph(&mut graph);

        assert!(changed);
        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.nodes[0].op, GraphOp::Gemm { alpha: 1, beta: 1 });
        assert_eq!(graph.nodes[0].inputs, vec![a, b, bias]);
        assert_eq!(graph.nodes[0].outputs, vec![add_out]);
    }

    #[test]
    fn fuse_add_relu() {
        let mut graph = ComputeGraph::new();

        let a = graph.add_edge(make_tensor("a", &[-1, 256]));
        let b = graph.add_edge(make_tensor("b", &[-1, 256]));
        let add_out = graph.add_edge(make_tensor("add_out", &[-1, 256]));
        let relu_out = graph.add_edge(make_tensor("relu_out", &[-1, 256]));

        graph.inputs = vec![a, b];
        graph.outputs = vec![relu_out];

        graph
            .add_node(GraphOp::Add, vec![a, b], vec![add_out], "add")
            .unwrap();
        graph
            .add_node(GraphOp::Relu, vec![add_out], vec![relu_out], "relu")
            .unwrap();

        assert_eq!(graph.node_count(), 2);

        let fusion = OperatorFusion;
        let changed = fusion.run_on_graph(&mut graph);

        assert!(changed);
        assert_eq!(graph.node_count(), 1);
        assert_eq!(
            graph.nodes[0].op,
            GraphOp::FusedElementWise {
                base_op: Box::new(GraphOp::Add),
                activation: ActivationFunction::Relu,
            }
        );
        assert_eq!(graph.nodes[0].inputs, vec![a, b]);
        assert_eq!(graph.nodes[0].outputs, vec![relu_out]);
    }

    #[test]
    fn no_fusion_unfusible_pattern() {
        // MatMul followed by Reshape — should NOT fuse.
        let mut graph = ComputeGraph::new();

        let a = graph.add_edge(make_tensor("A", &[-1, 768]));
        let b = graph.add_edge(make_tensor("B", &[768, 768]));
        let mm_out = graph.add_edge(make_tensor("mm_out", &[-1, 768]));
        let reshape_out = graph.add_edge(make_tensor("reshape_out", &[-1, 768]));

        graph.inputs = vec![a, b];
        graph.outputs = vec![reshape_out];

        graph
            .add_node(GraphOp::MatMul, vec![a, b], vec![mm_out], "matmul")
            .unwrap();
        graph
            .add_node(GraphOp::Reshape, vec![mm_out], vec![reshape_out], "reshape")
            .unwrap();

        let fusion = OperatorFusion;
        let changed = fusion.run_on_graph(&mut graph);

        assert!(!changed);
        assert_eq!(graph.node_count(), 2);
    }

    #[test]
    fn no_fusion_multi_consumer() {
        // If the intermediate edge has multiple consumers, do not fuse.
        let mut graph = ComputeGraph::new();

        let a = graph.add_edge(make_tensor("A", &[-1, 768]));
        let b = graph.add_edge(make_tensor("B", &[768, 768]));
        let mm_out = graph.add_edge(make_tensor("mm_out", &[-1, 768]));
        let bias = graph.add_edge(make_tensor("bias", &[768]));
        let add_out = graph.add_edge(make_tensor("add_out", &[-1, 768]));
        let relu_out = graph.add_edge(make_tensor("relu_out", &[-1, 768]));

        graph.inputs = vec![a, b, bias];
        graph.outputs = vec![add_out, relu_out];

        graph
            .add_node(GraphOp::MatMul, vec![a, b], vec![mm_out], "matmul")
            .unwrap();
        // mm_out is consumed by both Add and Relu (two consumers).
        graph
            .add_node(GraphOp::Add, vec![mm_out, bias], vec![add_out], "add")
            .unwrap();
        graph
            .add_node(GraphOp::Relu, vec![mm_out], vec![relu_out], "relu")
            .unwrap();

        let fusion = OperatorFusion;
        let changed = fusion.run_on_graph(&mut graph);

        // MatMul+Add should NOT fuse because mm_out has 2 consumers.
        assert!(!changed);
        assert_eq!(graph.node_count(), 3);
    }

    #[test]
    fn no_fusion_standalone_activation() {
        // A standalone Relu with no preceding fusible op.
        let mut graph = ComputeGraph::new();

        let input = graph.add_edge(make_tensor("input", &[-1, 256]));
        let output = graph.add_edge(make_tensor("output", &[-1, 256]));

        graph.inputs = vec![input];
        graph.outputs = vec![output];

        graph
            .add_node(GraphOp::Relu, vec![input], vec![output], "relu")
            .unwrap();

        let fusion = OperatorFusion;
        let changed = fusion.run_on_graph(&mut graph);

        assert!(!changed);
        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn fuse_mul_sigmoid() {
        let mut graph = ComputeGraph::new();

        let a = graph.add_edge(make_tensor("a", &[-1, 256]));
        let b = graph.add_edge(make_tensor("b", &[-1, 256]));
        let mul_out = graph.add_edge(make_tensor("mul_out", &[-1, 256]));
        let sig_out = graph.add_edge(make_tensor("sig_out", &[-1, 256]));

        graph.inputs = vec![a, b];
        graph.outputs = vec![sig_out];

        graph
            .add_node(GraphOp::Mul, vec![a, b], vec![mul_out], "mul")
            .unwrap();
        graph
            .add_node(GraphOp::Sigmoid, vec![mul_out], vec![sig_out], "sigmoid")
            .unwrap();

        let fusion = OperatorFusion;
        let changed = fusion.run_on_graph(&mut graph);

        assert!(changed);
        assert_eq!(graph.node_count(), 1);
        assert_eq!(
            graph.nodes[0].op,
            GraphOp::FusedElementWise {
                base_op: Box::new(GraphOp::Mul),
                activation: ActivationFunction::Sigmoid,
            }
        );
    }

    #[test]
    fn chained_fusion() {
        // MatMul → Add → Relu: MatMul+Add fuses to Gemm, but Relu stays
        // because Gemm is not an elementwise binary op.
        let mut graph = ComputeGraph::new();

        let a = graph.add_edge(make_tensor("A", &[-1, 768]));
        let b = graph.add_edge(make_tensor("B", &[768, 768]));
        let mm_out = graph.add_edge(make_tensor("mm_out", &[-1, 768]));
        let bias = graph.add_edge(make_tensor("bias", &[768]));
        let add_out = graph.add_edge(make_tensor("add_out", &[-1, 768]));
        let relu_out = graph.add_edge(make_tensor("relu_out", &[-1, 768]));

        graph.inputs = vec![a, b, bias];
        graph.outputs = vec![relu_out];

        graph
            .add_node(GraphOp::MatMul, vec![a, b], vec![mm_out], "matmul")
            .unwrap();
        graph
            .add_node(GraphOp::Add, vec![mm_out, bias], vec![add_out], "add")
            .unwrap();
        graph
            .add_node(GraphOp::Relu, vec![add_out], vec![relu_out], "relu")
            .unwrap();

        let fusion = OperatorFusion;
        let changed = fusion.run_on_graph(&mut graph);

        assert!(changed);
        // MatMul+Add fuses to Gemm; Relu remains separate.
        assert_eq!(graph.node_count(), 2);
        assert!(matches!(
            graph.nodes[0].op,
            GraphOp::Gemm { alpha: 1, beta: 1 }
        ));
        assert_eq!(graph.nodes[1].op, GraphOp::Relu);
    }

    #[test]
    fn conv_bias_no_activation() {
        // Conv2D + Add (bias) without activation.
        let mut graph = ComputeGraph::new();

        let input = graph.add_edge(make_tensor("input", &[-1, 3, 224, 224]));
        let weight = graph.add_edge(make_tensor("weight", &[64, 3, 3, 3]));
        let bias = graph.add_edge(make_tensor("bias", &[64]));
        let conv_out = graph.add_edge(make_tensor("conv_out", &[-1, 64, 222, 222]));
        let add_out = graph.add_edge(make_tensor("add_out", &[-1, 64, 222, 222]));

        graph.inputs = vec![input, weight, bias];
        graph.outputs = vec![add_out];

        graph
            .add_node(GraphOp::Conv2d, vec![input, weight], vec![conv_out], "conv")
            .unwrap();
        graph
            .add_node(
                GraphOp::Add,
                vec![conv_out, bias],
                vec![add_out],
                "bias_add",
            )
            .unwrap();

        let fusion = OperatorFusion;
        let changed = fusion.run_on_graph(&mut graph);

        assert!(changed);
        assert_eq!(graph.node_count(), 1);
        assert_eq!(
            graph.nodes[0].op,
            GraphOp::FusedConv2d {
                activation: ActivationFunction::None
            }
        );
        assert_eq!(graph.nodes[0].inputs, vec![input, weight, bias]);
    }

    #[test]
    fn pass_on_empty_graph() {
        let mut graph = ComputeGraph::new();
        let fusion = OperatorFusion;
        let changed = fusion.run_on_graph(&mut graph);
        assert!(!changed);
    }

    #[test]
    fn pass_trait_on_module_is_noop() {
        let fusion = OperatorFusion;
        let mut module = nxpu_ir::Module::default();
        let changed = fusion.run(&mut module);
        assert!(!changed);
    }
}
