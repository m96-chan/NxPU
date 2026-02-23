//! Qualcomm Hexagon NPU operator support matrix.

use nxpu_backend_core::{OperatorSupport, PerformanceTier, Precision};

/// Operator support for Qualcomm Hexagon NPU (QNN).
pub struct HexagonNpuSupport;

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
    "Transpose",
];

const EMULATED_OPS: &[&str] = &[
    "Sub",
    "Div",
    "Tanh",
    "ReduceSum",
    "ReduceMean",
    "ReduceMax",
    "ReduceMin",
    "Split",
    "Attention",
];

impl OperatorSupport for HexagonNpuSupport {
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
        "Qualcomm Hexagon NPU"
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
        let s = HexagonNpuSupport;
        assert_eq!(
            s.op_support("Conv", Precision::Int8),
            PerformanceTier::Native
        );
        assert_eq!(
            s.op_support("Transpose", Precision::Int8),
            PerformanceTier::Native
        );
    }

    #[test]
    fn native_at_f16() {
        let s = HexagonNpuSupport;
        assert_eq!(
            s.op_support("MatMul", Precision::F16),
            PerformanceTier::Native
        );
    }

    #[test]
    fn emulated_at_int8() {
        let s = HexagonNpuSupport;
        assert_eq!(
            s.op_support("Attention", Precision::Int8),
            PerformanceTier::Emulated
        );
    }

    #[test]
    fn unknown_unsupported() {
        let s = HexagonNpuSupport;
        assert_eq!(
            s.op_support("FakeOp", Precision::Int8),
            PerformanceTier::Unsupported
        );
    }

    #[test]
    fn hardware_name() {
        assert_eq!(HexagonNpuSupport.hardware_name(), "Qualcomm Hexagon NPU");
    }

    #[test]
    fn f32_emulated() {
        let s = HexagonNpuSupport;
        assert_eq!(
            s.op_support("MatMul", Precision::F32),
            PerformanceTier::Emulated
        );
    }

    #[test]
    fn f32_unknown_unsupported() {
        let s = HexagonNpuSupport;
        assert_eq!(
            s.op_support("FakeOp", Precision::F32),
            PerformanceTier::Unsupported
        );
    }

    #[test]
    fn native_and_emulated_lists() {
        let s = HexagonNpuSupport;
        assert!(s.native_ops().contains(&"Conv"));
        assert!(s.emulated_ops().contains(&"Attention"));
    }
}
