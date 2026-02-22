//! Error types for the NxPU IR.

/// Errors that can occur when constructing or validating IR.
#[derive(Debug, thiserror::Error)]
pub enum IrError {
    /// A handle index is out of bounds for its arena.
    #[error("handle index {index} out of bounds (arena size: {size})")]
    BadHandle { index: usize, size: usize },

    /// A type mismatch was detected.
    #[error("type mismatch: expected {expected}, found {found}")]
    TypeMismatch { expected: String, found: String },

    /// An invalid workgroup size was specified.
    #[error("invalid workgroup size: [{}, {}, {}]", .0[0], .0[1], .0[2])]
    InvalidWorkgroupSize([u32; 3]),

    /// An edge referenced by a graph node was not registered.
    #[error("unknown edge {edge_id} referenced by node {node_name:?}")]
    UnknownEdge { edge_id: u32, node_name: String },

    /// An output edge already has a producer node.
    #[error(
        "edge {edge_id} already produced by {existing_producer:?}, cannot add producer {new_producer:?}"
    )]
    DuplicateEdgeProducer {
        edge_id: u32,
        existing_producer: String,
        new_producer: String,
    },

    /// A cycle was detected in the computation graph.
    #[error("graph contains a cycle ({visited} of {total} nodes visited)")]
    CycleDetected { visited: usize, total: usize },
}
