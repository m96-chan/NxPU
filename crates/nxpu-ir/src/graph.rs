//! Graph-level intermediate representation for multi-operation models.
//!
//! Extends the single-kernel IR to support directed acyclic graphs (DAGs)
//! of operations, enabling transpilation of production ML models with
//! 50-500+ operations.

use std::collections::HashMap;

use crate::types::{Scalar, TensorShape};

/// A unique identifier for a node in the computation graph.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct NodeId(pub u32);

/// A unique identifier for an edge (tensor) in the computation graph.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct EdgeId(pub u32);

/// The operation type for a graph node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphOp {
    /// Matrix multiplication.
    MatMul,
    /// Convolution 2D.
    Conv2d,
    /// Element-wise addition.
    Add,
    /// Element-wise subtraction.
    Sub,
    /// Element-wise multiplication.
    Mul,
    /// Element-wise division.
    Div,
    /// Rectified Linear Unit activation.
    Relu,
    /// Sigmoid activation.
    Sigmoid,
    /// Softmax activation.
    Softmax,
    /// Batch normalization.
    BatchNorm,
    /// Layer normalization.
    LayerNorm,
    /// Max pooling 2D.
    MaxPool2d,
    /// Average pooling 2D.
    AvgPool2d,
    /// Reshape/view.
    Reshape,
    /// Transpose/permute dimensions.
    Transpose,
    /// Concatenation along an axis.
    Concat { axis: i32 },
    /// Custom/vendor-specific operation.
    Custom { op_type: String },
}

impl GraphOp {
    /// Returns the ONNX operator type string.
    pub fn onnx_op_type(&self) -> &str {
        match self {
            Self::MatMul => "MatMul",
            Self::Conv2d => "Conv",
            Self::Add => "Add",
            Self::Sub => "Sub",
            Self::Mul => "Mul",
            Self::Div => "Div",
            Self::Relu => "Relu",
            Self::Sigmoid => "Sigmoid",
            Self::Softmax => "Softmax",
            Self::BatchNorm => "BatchNormalization",
            Self::LayerNorm => "LayerNormalization",
            Self::MaxPool2d => "MaxPool",
            Self::AvgPool2d => "AveragePool",
            Self::Reshape => "Reshape",
            Self::Transpose => "Transpose",
            Self::Concat { .. } => "Concat",
            Self::Custom { op_type } => op_type,
        }
    }
}

/// Metadata about a tensor edge in the graph.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    /// Human-readable name.
    pub name: String,
    /// Element scalar type.
    pub scalar: Scalar,
    /// Shape (may contain dynamic dimensions).
    pub shape: TensorShape,
}

/// A node in the computation graph.
#[derive(Clone, Debug)]
pub struct GraphNode {
    /// Unique identifier for this node.
    pub id: NodeId,
    /// The operation this node performs.
    pub op: GraphOp,
    /// Input edge identifiers (ordered).
    pub inputs: Vec<EdgeId>,
    /// Output edge identifiers (ordered).
    pub outputs: Vec<EdgeId>,
    /// Human-readable name for this node.
    pub name: String,
}

/// A computation graph representing a multi-operation model.
///
/// This is a DAG where nodes are operations and edges are tensors
/// flowing between operations.
#[derive(Clone, Debug, Default)]
pub struct ComputeGraph {
    /// All nodes in the graph, keyed by NodeId.
    pub nodes: Vec<GraphNode>,
    /// All tensor edges, keyed by EdgeId.
    pub edges: HashMap<EdgeId, TensorInfo>,
    /// Graph-level input edge ids (model inputs).
    pub inputs: Vec<EdgeId>,
    /// Graph-level output edge ids (model outputs).
    pub outputs: Vec<EdgeId>,
    /// Next available node id.
    next_node_id: u32,
    /// Next available edge id.
    next_edge_id: u32,
}

impl ComputeGraph {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tensor edge to the graph and return its id.
    pub fn add_edge(&mut self, info: TensorInfo) -> EdgeId {
        let id = EdgeId(self.next_edge_id);
        self.next_edge_id += 1;
        self.edges.insert(id, info);
        id
    }

    /// Add a node to the graph and return its id.
    pub fn add_node(
        &mut self,
        op: GraphOp,
        inputs: Vec<EdgeId>,
        outputs: Vec<EdgeId>,
        name: impl Into<String>,
    ) -> NodeId {
        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        self.nodes.push(GraphNode {
            id,
            op,
            inputs,
            outputs,
            name: name.into(),
        });
        id
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges (tensors) in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Returns nodes in topological order.
    ///
    /// Panics if the graph contains cycles (which it shouldn't for a DAG).
    pub fn topological_order(&self) -> Vec<&GraphNode> {
        // Build adjacency: which node produces each edge?
        let mut edge_producer: HashMap<EdgeId, NodeId> = HashMap::new();
        for node in &self.nodes {
            for &out in &node.outputs {
                edge_producer.insert(out, node.id);
            }
        }

        // Build in-degree
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        for node in &self.nodes {
            in_degree.entry(node.id).or_insert(0);
            for &inp in &node.inputs {
                if edge_producer.contains_key(&inp) {
                    *in_degree.entry(node.id).or_insert(0) += 1;
                }
            }
        }

        // Kahn's algorithm
        let mut queue: Vec<NodeId> = in_degree
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(id, _)| *id)
            .collect();

        let mut result: Vec<NodeId> = Vec::new();

        while let Some(nid) = queue.pop() {
            result.push(nid);
            let node = self.nodes.iter().find(|n| n.id == nid).unwrap();
            for &out_edge in &node.outputs {
                // Find consumers of this edge
                for consumer in &self.nodes {
                    if consumer.inputs.contains(&out_edge) {
                        let deg = in_degree.get_mut(&consumer.id).unwrap();
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push(consumer.id);
                        }
                    }
                }
            }
        }

        result
            .iter()
            .map(|nid| self.nodes.iter().find(|n| n.id == *nid).unwrap())
            .collect()
    }

    /// Find all nodes that consume the given edge.
    pub fn edge_consumers(&self, edge: EdgeId) -> Vec<&GraphNode> {
        self.nodes
            .iter()
            .filter(|n| n.inputs.contains(&edge))
            .collect()
    }

    /// Find the node that produces the given edge, if any.
    pub fn edge_producer(&self, edge: EdgeId) -> Option<&GraphNode> {
        self.nodes.iter().find(|n| n.outputs.contains(&edge))
    }
}

/// Extends [`crate::Module`] with an optional computation graph.
///
/// When present, the graph describes how multiple entry points
/// compose into a single model.
#[derive(Clone, Debug, Default)]
pub struct GraphModule {
    /// The base IR module with types, globals, and entry points.
    pub module: crate::Module,
    /// Optional multi-operation graph overlay.
    pub graph: Option<ComputeGraph>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Dimension, TensorShape};

    fn make_tensor_info(name: &str, shape: &[i64]) -> TensorInfo {
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
    fn build_simple_graph() {
        let mut graph = ComputeGraph::new();

        // MatMul → Add → ReLU
        let a = graph.add_edge(make_tensor_info("A", &[-1, 768]));
        let b = graph.add_edge(make_tensor_info("B", &[768, 768]));
        let matmul_out = graph.add_edge(make_tensor_info("matmul_out", &[-1, 768]));
        let bias = graph.add_edge(make_tensor_info("bias", &[768]));
        let add_out = graph.add_edge(make_tensor_info("add_out", &[-1, 768]));
        let relu_out = graph.add_edge(make_tensor_info("relu_out", &[-1, 768]));

        graph.inputs = vec![a, b, bias];
        graph.outputs = vec![relu_out];

        graph.add_node(GraphOp::MatMul, vec![a, b], vec![matmul_out], "matmul_0");
        graph.add_node(GraphOp::Add, vec![matmul_out, bias], vec![add_out], "add_0");
        graph.add_node(GraphOp::Relu, vec![add_out], vec![relu_out], "relu_0");

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 6);
    }

    #[test]
    fn topological_order() {
        let mut graph = ComputeGraph::new();

        let a = graph.add_edge(make_tensor_info("A", &[-1, 10]));
        let b = graph.add_edge(make_tensor_info("B", &[10, 10]));
        let c = graph.add_edge(make_tensor_info("C", &[-1, 10]));
        let d = graph.add_edge(make_tensor_info("D", &[-1, 10]));

        graph.inputs = vec![a, b];
        graph.outputs = vec![d];

        graph.add_node(GraphOp::MatMul, vec![a, b], vec![c], "matmul");
        graph.add_node(GraphOp::Relu, vec![c], vec![d], "relu");

        let order = graph.topological_order();
        assert_eq!(order.len(), 2);
        assert_eq!(order[0].name, "matmul");
        assert_eq!(order[1].name, "relu");
    }

    #[test]
    fn edge_producer_consumer() {
        let mut graph = ComputeGraph::new();

        let a = graph.add_edge(make_tensor_info("A", &[10]));
        let b = graph.add_edge(make_tensor_info("B", &[10]));
        let c = graph.add_edge(make_tensor_info("C", &[10]));

        graph.add_node(GraphOp::Add, vec![a], vec![b], "add");
        graph.add_node(GraphOp::Relu, vec![b], vec![c], "relu");

        let producer = graph.edge_producer(b).unwrap();
        assert_eq!(producer.name, "add");

        let consumers = graph.edge_consumers(b);
        assert_eq!(consumers.len(), 1);
        assert_eq!(consumers[0].name, "relu");

        // Graph input has no producer
        assert!(graph.edge_producer(a).is_none());
    }

    #[test]
    fn graph_op_onnx_names() {
        assert_eq!(GraphOp::MatMul.onnx_op_type(), "MatMul");
        assert_eq!(GraphOp::Conv2d.onnx_op_type(), "Conv");
        assert_eq!(GraphOp::Relu.onnx_op_type(), "Relu");
        assert_eq!(GraphOp::Softmax.onnx_op_type(), "Softmax");
        assert_eq!(
            GraphOp::Custom {
                op_type: "MyOp".into()
            }
            .onnx_op_type(),
            "MyOp"
        );
    }

    #[test]
    fn graph_module() {
        let gm = GraphModule::default();
        assert!(gm.graph.is_none());
        assert!(gm.module.entry_points.is_empty());
    }
}
