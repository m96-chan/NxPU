//! CEVA NeuPro NPU operator support matrix.

use nxpu_backend_core::{OperatorSupport, PerformanceTier, Precision};

/// Operator support for CEVA NeuPro NPU.
///
/// CEVA NeuPro has a limited operator set optimized for embedded vision
/// and audio workloads.
pub struct CevaNeuProSupport;

const NATIVE_OPS: &[&str] = &["Conv", "MaxPool", "AveragePool", "Relu"];

const EMULATED_OPS: &[&str] = &[
    "MatMul",
    "Add",
    "Mul",
    "Sigmoid",
    "Concat",
    "Reshape",
    "BatchNormalization",
];

impl OperatorSupport for CevaNeuProSupport {
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
                // Limited F16 support — all known ops are emulated
                if NATIVE_OPS.contains(&op_name) || EMULATED_OPS.contains(&op_name) {
                    PerformanceTier::Emulated
                } else {
                    PerformanceTier::Unsupported
                }
            }
            _ => {
                if NATIVE_OPS.contains(&op_name) || EMULATED_OPS.contains(&op_name) {
                    PerformanceTier::Emulated
                } else {
                    PerformanceTier::Unsupported
                }
            }
        }
    }

    fn hardware_name(&self) -> &str {
        "CEVA NeuPro"
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
        let s = CevaNeuProSupport;
        assert_eq!(
            s.op_support("Conv", Precision::Int8),
            PerformanceTier::Native
        );
        assert_eq!(
            s.op_support("Relu", Precision::Int8),
            PerformanceTier::Native
        );
        assert_eq!(
            s.op_support("MaxPool", Precision::Int8),
            PerformanceTier::Native
        );
    }

    #[test]
    fn emulated_at_int8() {
        let s = CevaNeuProSupport;
        assert_eq!(
            s.op_support("MatMul", Precision::Int8),
            PerformanceTier::Emulated
        );
    }

    #[test]
    fn unsupported_ops() {
        let s = CevaNeuProSupport;
        assert_eq!(
            s.op_support("Attention", Precision::Int8),
            PerformanceTier::Unsupported
        );
        assert_eq!(
            s.op_support("Softmax", Precision::Int8),
            PerformanceTier::Unsupported
        );
    }

    #[test]
    fn hardware_name() {
        assert_eq!(CevaNeuProSupport.hardware_name(), "CEVA NeuPro");
    }

    #[test]
    fn f16_emulated() {
        let s = CevaNeuProSupport;
        assert_eq!(
            s.op_support("Conv", Precision::F16),
            PerformanceTier::Emulated
        );
    }

    #[test]
    fn f16_unknown_unsupported() {
        let s = CevaNeuProSupport;
        assert_eq!(
            s.op_support("FakeOp", Precision::F16),
            PerformanceTier::Unsupported
        );
    }

    #[test]
    fn f32_emulated() {
        let s = CevaNeuProSupport;
        assert_eq!(
            s.op_support("Conv", Precision::F32),
            PerformanceTier::Emulated
        );
    }

    #[test]
    fn native_and_emulated_lists() {
        let s = CevaNeuProSupport;
        assert!(s.native_ops().contains(&"Conv"));
        assert!(s.emulated_ops().contains(&"MatMul"));
    }
}
