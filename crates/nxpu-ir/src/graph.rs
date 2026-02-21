//! Graph-level intermediate representation for multi-operation models.
//!
//! Extends the single-kernel IR to support directed acyclic graphs (DAGs)
//! of operations, enabling transpilation of production ML models with
//! 50-500+ operations.

use std::collections::{BTreeSet, HashMap};

use crate::types::{Scalar, TensorShape};

/// A unique identifier for a node in the computation graph.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
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
    ///
    /// # Panics
    ///
    /// Panics if any input or output [`EdgeId`] has not been previously
    /// registered via [`add_edge`](Self::add_edge), or if an output edge
    /// already has a producer node (each edge may have at most one producer).
    pub fn add_node(
        &mut self,
        op: GraphOp,
        inputs: Vec<EdgeId>,
        outputs: Vec<EdgeId>,
        name: impl Into<String>,
    ) -> NodeId {
        let name = name.into();

        // Validate that all referenced edges exist.
        for &e in inputs.iter().chain(outputs.iter()) {
            assert!(
                self.edges.contains_key(&e),
                "add_node({name}): EdgeId({}) not registered in graph",
                e.0,
            );
        }

        // Enforce single-producer-per-edge invariant.
        for &out in &outputs {
            let existing = self.nodes.iter().find(|n| n.outputs.contains(&out));
            assert!(
                existing.is_none(),
                "add_node({name}): EdgeId({}) already produced by node {:?}",
                out.0,
                existing.unwrap().name,
            );
        }

        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        self.nodes.push(GraphNode {
            id,
            op,
            inputs,
            outputs,
            name,
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
    /// The ordering is deterministic: among nodes with the same in-degree,
    /// the one with the smaller [`NodeId`] is emitted first.
    ///
    /// # Panics
    ///
    /// Panics if the graph contains a cycle.
    pub fn topological_order(&self) -> Vec<&GraphNode> {
        // Build adjacency: which node produces each edge?
        let mut edge_producer: HashMap<EdgeId, usize> = HashMap::new();
        for (i, node) in self.nodes.iter().enumerate() {
            for &out in &node.outputs {
                edge_producer.insert(out, i);
            }
        }

        // Build per-node consumer lists and in-degree (O(V+E))
        let n = self.nodes.len();
        let mut in_degree = vec![0usize; n];
        let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];

        for (ci, node) in self.nodes.iter().enumerate() {
            for &inp in &node.inputs {
                if let Some(&pi) = edge_producer.get(&inp) {
                    in_degree[ci] += 1;
                    consumers[pi].push(ci);
                }
            }
        }

        // Kahn's algorithm with deterministic BTreeSet (ordered by NodeId)
        let mut ready: BTreeSet<(NodeId, usize)> = BTreeSet::new();
        for (i, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                ready.insert((self.nodes[i].id, i));
            }
        }

        let mut result: Vec<&GraphNode> = Vec::with_capacity(n);

        while let Some(&(_, idx)) = ready.iter().next() {
            ready.remove(&(self.nodes[idx].id, idx));
            result.push(&self.nodes[idx]);

            for &ci in &consumers[idx] {
                in_degree[ci] -= 1;
                if in_degree[ci] == 0 {
                    ready.insert((self.nodes[ci].id, ci));
                }
            }
        }

        assert!(
            result.len() == n,
            "topological_order: graph contains a cycle ({} of {} nodes visited)",
            result.len(),
            n,
        );

        result
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

    #[test]
    fn topological_order_empty_graph() {
        let graph = ComputeGraph::new();
        let order = graph.topological_order();
        assert!(order.is_empty());
    }

    #[test]
    fn topological_order_diamond_dag() {
        // A → B, A → C, B → D, C → D
        let mut graph = ComputeGraph::new();

        let e_in = graph.add_edge(make_tensor_info("in", &[10]));
        let e_ab = graph.add_edge(make_tensor_info("ab", &[10]));
        let e_ac = graph.add_edge(make_tensor_info("ac", &[10]));
        let e_bd = graph.add_edge(make_tensor_info("bd", &[10]));
        let e_cd = graph.add_edge(make_tensor_info("cd", &[10]));
        let e_out = graph.add_edge(make_tensor_info("out", &[10]));

        graph.add_node(GraphOp::Relu, vec![e_in], vec![e_ab, e_ac], "A");
        graph.add_node(GraphOp::Relu, vec![e_ab], vec![e_bd], "B");
        graph.add_node(GraphOp::Relu, vec![e_ac], vec![e_cd], "C");
        graph.add_node(GraphOp::Add, vec![e_bd, e_cd], vec![e_out], "D");

        let order = graph.topological_order();
        assert_eq!(order.len(), 4);
        assert_eq!(order[0].name, "A");
        // B before C (deterministic by NodeId)
        assert_eq!(order[1].name, "B");
        assert_eq!(order[2].name, "C");
        assert_eq!(order[3].name, "D");
    }

    #[test]
    #[should_panic(expected = "graph contains a cycle")]
    fn topological_order_detects_cycle() {
        let mut graph = ComputeGraph::new();

        let e0 = graph.add_edge(make_tensor_info("e0", &[10]));
        let e1 = graph.add_edge(make_tensor_info("e1", &[10]));

        // Manually build a cycle by pushing nodes directly
        // (bypassing add_node validation which checks single-producer)
        graph.nodes.push(GraphNode {
            id: NodeId(0),
            op: GraphOp::Relu,
            inputs: vec![e1],
            outputs: vec![e0],
            name: "A".into(),
        });
        graph.nodes.push(GraphNode {
            id: NodeId(1),
            op: GraphOp::Relu,
            inputs: vec![e0],
            outputs: vec![e1],
            name: "B".into(),
        });
        graph.next_node_id = 2;

        graph.topological_order(); // should panic
    }

    #[test]
    #[should_panic(expected = "not registered in graph")]
    fn add_node_rejects_unknown_edge() {
        let mut graph = ComputeGraph::new();
        let fake_edge = EdgeId(999);
        let out = graph.add_edge(make_tensor_info("out", &[10]));
        graph.add_node(GraphOp::Relu, vec![fake_edge], vec![out], "bad");
    }

    #[test]
    #[should_panic(expected = "already produced by node")]
    fn add_node_rejects_duplicate_producer() {
        let mut graph = ComputeGraph::new();
        let a = graph.add_edge(make_tensor_info("a", &[10]));
        let b = graph.add_edge(make_tensor_info("b", &[10]));
        let c = graph.add_edge(make_tensor_info("c", &[10]));

        graph.add_node(GraphOp::Relu, vec![a], vec![b], "first");
        // b already has a producer — should panic
        graph.add_node(GraphOp::Relu, vec![c], vec![b], "second");
    }
}
