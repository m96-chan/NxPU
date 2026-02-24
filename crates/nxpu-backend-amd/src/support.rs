//! AMD XDNA NPU operator support matrix.

use nxpu_backend_core::{OperatorSupport, PerformanceTier, Precision};

/// Operator support for AMD XDNA (Ryzen AI).
pub struct AmdXdnaSupport;

const NATIVE_OPS: &[&str] = &[
    "MatMul",
    "Conv",
    "Add",
    "Mul",
    "Relu",
    "Sigmoid",
    "MaxPool",
    "AveragePool",
    "BatchNormalization",
    "Concat",
    "Reshape",
    "Softmax",
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

impl OperatorSupport for AmdXdnaSupport {
    fn op_support(&self, op_name: &str, precision: Precision) -> PerformanceTier {
        match precision {
            Precision::Int8 => {
                if NATIVE_OPS.contains(&op_name) {
                    PerformanceTier::Native
                } else if EMULATED_OPS.contains(&op_name) {
                    PerformanceTier::Emulated
                } else {
                    PerformanceTier::Unsupported
                }
            }
            Precision::F16 => {
                // F16 supported for many ops on XDNA
                if NATIVE_OPS.contains(&op_name) {
                    PerformanceTier::Native
                } else if EMULATED_OPS.contains(&op_name) {
                    PerformanceTier::Emulated
                } else {
                    PerformanceTier::Unsupported
                }
            }
            Precision::F32 => {
                if NATIVE_OPS.contains(&op_name) || EMULATED_OPS.contains(&op_name) {
                    PerformanceTier::Emulated
                } else {
                    PerformanceTier::Unsupported
                }
            }
            Precision::BF16 => {
                if NATIVE_OPS.contains(&op_name) || EMULATED_OPS.contains(&op_name) {
                    PerformanceTier::Emulated
                } else {
                    PerformanceTier::Unsupported
                }
            }
        }
    }

    fn hardware_name(&self) -> &str {
        "AMD XDNA"
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
        let s = AmdXdnaSupport;
        assert_eq!(
            s.op_support("MatMul", Precision::Int8),
            PerformanceTier::Native
        );
        assert_eq!(
            s.op_support("Conv", Precision::Int8),
            PerformanceTier::Native
        );
    }

    #[test]
    fn emulated_at_int8() {
        let s = AmdXdnaSupport;
        assert_eq!(
            s.op_support("Attention", Precision::Int8),
            PerformanceTier::Emulated
        );
    }

    #[test]
    fn f32_emulated() {
        let s = AmdXdnaSupport;
        assert_eq!(
            s.op_support("MatMul", Precision::F32),
            PerformanceTier::Emulated
        );
    }

    #[test]
    fn unknown_unsupported() {
        let s = AmdXdnaSupport;
        assert_eq!(
            s.op_support("FakeOp", Precision::Int8),
            PerformanceTier::Unsupported
        );
    }

    #[test]
    fn hardware_name() {
        assert_eq!(AmdXdnaSupport.hardware_name(), "AMD XDNA");
    }

    #[test]
    fn native_and_emulated_lists() {
        let s = AmdXdnaSupport;
        assert!(s.native_ops().contains(&"MatMul"));
        assert!(s.emulated_ops().contains(&"Attention"));
    }

    #[test]
    fn native_at_f16() {
        let s = AmdXdnaSupport;
        assert_eq!(
            s.op_support("MatMul", Precision::F16),
            PerformanceTier::Native
        );
        assert_eq!(
            s.op_support("Conv", Precision::F16),
            PerformanceTier::Native
        );
    }

    #[test]
    fn emulated_at_f16() {
        let s = AmdXdnaSupport;
        assert_eq!(
            s.op_support("Attention", Precision::F16),
            PerformanceTier::Emulated
        );
    }

    #[test]
    fn bf16_emulated() {
        let s = AmdXdnaSupport;
        assert_eq!(
            s.op_support("MatMul", Precision::BF16),
            PerformanceTier::Emulated
        );
    }

    #[test]
    fn bf16_unknown_unsupported() {
        let s = AmdXdnaSupport;
        assert_eq!(
            s.op_support("FakeOp", Precision::BF16),
            PerformanceTier::Unsupported
        );
    }

    #[test]
    fn f32_unknown_unsupported() {
        let s = AmdXdnaSupport;
        assert_eq!(
            s.op_support("FakeOp", Precision::F32),
            PerformanceTier::Unsupported
        );
    }
}
