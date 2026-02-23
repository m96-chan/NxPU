//! Samsung Exynos NPU operator support matrix.

use nxpu_backend_core::{OperatorSupport, PerformanceTier, Precision};

/// Operator support for Samsung Exynos NPU.
pub struct SamsungNpuSupport;

const NATIVE_OPS: &[&str] = &[
    "Conv",
    "MatMul",
    "Add",
    "Mul",
    "Relu",
    "MaxPool",
    "AveragePool",
    "Concat",
    "Reshape",
    "BatchNormalization",
];

const EMULATED_OPS: &[&str] = &[
    "Sub",
    "Div",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "Transpose",
    "ReduceSum",
    "ReduceMean",
    "ReduceMax",
    "ReduceMin",
    "Split",
    "Attention",
];

impl OperatorSupport for SamsungNpuSupport {
    fn op_support(&self, op_name: &str, precision: Precision) -> PerformanceTier {
        match precision {
            Precision::F16 | Precision::Int8 => {
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
        "Samsung Exynos NPU"
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
    fn native_at_f16() {
        let s = SamsungNpuSupport;
        assert_eq!(
            s.op_support("Conv", Precision::F16),
            PerformanceTier::Native
        );
        assert_eq!(
            s.op_support("MatMul", Precision::F16),
            PerformanceTier::Native
        );
    }

    #[test]
    fn emulated_at_f16() {
        let s = SamsungNpuSupport;
        assert_eq!(
            s.op_support("Attention", Precision::F16),
            PerformanceTier::Emulated
        );
    }

    #[test]
    fn unknown_unsupported() {
        let s = SamsungNpuSupport;
        assert_eq!(
            s.op_support("FakeOp", Precision::F16),
            PerformanceTier::Unsupported
        );
    }

    #[test]
    fn hardware_name() {
        assert_eq!(SamsungNpuSupport.hardware_name(), "Samsung Exynos NPU");
    }
}
