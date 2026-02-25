pub mod analyze;
pub mod cost;
pub mod dataflow;
pub mod fusion;

pub use analyze::{
    ActivationOp, AnalysisError, Conv2DShape, ElementWiseOp, EmbeddedWeight, KernelPattern,
    MatMulShape, NormType, PoolKind, PoolShape, ReduceOp, TensorBinding, TensorRole,
    classify_entry_point, extract_embedded_weights, normalize_axis,
};
pub use cost::{
    Bottleneck, HardwareProfile, OpCost, default_profiles, estimate_activation_cost,
    estimate_conv2d_cost, estimate_elementwise_cost, estimate_kernel_cost, estimate_matmul_cost,
};
pub use dataflow::{
    CriticalPathResult, DataflowError, DataflowGraph, DependencyKind, DfgEdge, DfgNode, DfgNodeKind,
};
pub use fusion::{
    FusedActivation, FusedPattern, fuse_patterns, input_tensor_names, output_tensor_names,
    tensors_connect,
};
