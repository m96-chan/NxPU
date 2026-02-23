//! Rockchip RKNN NPU operator support matrix.

use nxpu_backend_core::{OperatorSupport, PerformanceTier, Precision};

/// Operator support for Rockchip RKNN NPU (RK3588 / RK3576).
pub struct RknnNpuSupport;

const NATIVE_OPS: &[&str] = &[
    "Conv",
    "MatMul",
    "Add",
    "Mul",
    "Relu",
    "Sigmoid",
    "MaxPool",
    "AveragePool",
    "Concat",
    "Reshape",
    "Softmax",
    "BatchNormalization",
];

const EMULATED_OPS: &[&str] = &[
    "Sub",
    "Div",
    "Tanh",
    "Transpose",
    "ReduceSum",
    "ReduceMean",
    "ReduceMax",
    "ReduceMin",
    "Split",
    "Attention",
];

impl OperatorSupport for RknnNpuSupport {
    fn op_support(&self, op_name: &str, precision: Precision) -> PerformanceTier {
        match precision {
            Precision::Int8 | Precision::F16 => {
                if NATIVE_OPS.contains(&op_name) {
                    PerformanceTier::Native
                } else if EMULATED_OPS.contains(&op_name) {
                    PerformanceTier::Emulated
                } else {
                    PerformanceTier::Unsupported
                }
            }
            Precision::F32 | Precision::BF16 => {
                if NATIVE_OPS.contains(&op_name) || EMULATED_OPS.contains(&op_name) {
                    PerformanceTier::Emulated
                } else {
                    PerformanceTier::Unsupported
                }
            }
        }
    }

    fn hardware_name(&self) -> &str {
        "Rockchip RKNN NPU"
    }

    fn native_ops(&self) -> &[&str] {
        NATIVE_OPS
    }

    fn emulated_ops(&self) -> &[&str] {
        EMULATED_OPS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_at_int8() {
        let s = RknnNpuSupport;
        assert_eq!(
            s.op_support("Conv", Precision::Int8),
            PerformanceTier::Native
        );
        assert_eq!(
            s.op_support("MatMul", Precision::Int8),
            PerformanceTier::Native
        );
    }

    #[test]
    fn native_at_f16() {
        let s = RknnNpuSupport;
        assert_eq!(
            s.op_support("Relu", Precision::F16),
            PerformanceTier::Native
        );
    }

    #[test]
    fn emulated_at_int8() {
        let s = RknnNpuSupport;
        assert_eq!(
            s.op_support("Attention", Precision::Int8),
            PerformanceTier::Emulated
        );
    }

    #[test]
    fn unknown_unsupported() {
        let s = RknnNpuSupport;
        assert_eq!(
            s.op_support("FakeOp", Precision::Int8),
            PerformanceTier::Unsupported
        );
    }

    #[test]
    fn hardware_name() {
        assert_eq!(RknnNpuSupport.hardware_name(), "Rockchip RKNN NPU");
    }

    #[test]
    fn f32_emulated() {
        let s = RknnNpuSupport;
        assert_eq!(
            s.op_support("MatMul", Precision::F32),
            PerformanceTier::Emulated
        );
    }

    #[test]
    fn f32_unknown_unsupported() {
        let s = RknnNpuSupport;
        assert_eq!(
            s.op_support("FakeOp", Precision::F32),
            PerformanceTier::Unsupported
        );
    }

    #[test]
    fn native_and_emulated_lists() {
        let s = RknnNpuSupport;
        assert!(s.native_ops().contains(&"Conv"));
        assert!(s.emulated_ops().contains(&"Attention"));
    }
}
