pub mod analyze;
pub mod fusion;

pub use analyze::{
    ActivationOp, AnalysisError, Conv2DShape, ElementWiseOp, KernelPattern, MatMulShape, PoolKind,
    PoolShape, ReduceOp, TensorBinding, TensorRole, classify_entry_point,
};
pub use fusion::{
    FusedActivation, FusedPattern, fuse_patterns, input_tensor_names, output_tensor_names,
    tensors_connect,
};
