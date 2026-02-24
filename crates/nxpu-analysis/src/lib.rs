pub mod analyze;
pub mod dataflow;
pub mod fusion;

pub use analyze::{
    ActivationOp, AnalysisError, Conv2DShape, ElementWiseOp, EmbeddedWeight, KernelPattern,
    MatMulShape, PoolKind, PoolShape, ReduceOp, TensorBinding, TensorRole, classify_entry_point,
    extract_embedded_weights,
};
pub use dataflow::{
    CriticalPathResult, DataflowError, DataflowGraph, DependencyKind, DfgEdge, DfgNode, DfgNodeKind,
};
pub use fusion::{
    FusedActivation, FusedPattern, fuse_patterns, input_tensor_names, output_tensor_names,
    tensors_connect,
};
