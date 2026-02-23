//! Intel NPU operator support matrix.

use nxpu_backend_core::{OperatorSupport, PerformanceTier, Precision};

/// Operator support for Intel NPU (Meteor Lake / Arrow Lake).
pub struct IntelNpuSupport;

const NATIVE_OPS: &[&str] = &[
    "MatMul",
    "Conv",
    "Add",
    "Sub",
    "Mul",
    "Relu",
    "Sigmoid",
    "Tanh",
    "MaxPool",
    "AveragePool",
    "BatchNormalization",
    "Concat",
    "Reshape",
    "Transpose",
    "Softmax",
];

const EMULATED_OPS: &[&str] = &[
    "Div",
    "ReduceSum",
    "ReduceMean",
    "ReduceMax",
    "ReduceMin",
    "Split",
    "Attention",
];

impl OperatorSupport for IntelNpuSupport {
    fn op_support(&self, op_name: &str, precision: Precision) -> PerformanceTier {
        // Intel NPU has best support at F16
        match precision {
            Precision::F16 => {
                if NATIVE_OPS.contains(&op_name) {
                    PerformanceTier::Native
                } else if EMULATED_OPS.contains(&op_name) {
                    PerformanceTier::Emulated
                } else {
                    PerformanceTier::Unsupported
                }
            }
            Precision::F32 => {
                // F32 ops mostly emulated (cast down to F16 internally)
                if NATIVE_OPS.contains(&op_name) || EMULATED_OPS.contains(&op_name) {
                    PerformanceTier::Emulated
                } else {
                    PerformanceTier::Unsupported
                }
            }
            Precision::Int8 => {
                // Int8 supported for core compute ops
                match op_name {
                    "MatMul" | "Conv" | "Add" | "Relu" | "MaxPool" | "AveragePool" => {
                        PerformanceTier::Native
                    }
                    _ if NATIVE_OPS.contains(&op_name) => PerformanceTier::Emulated,
                    _ => PerformanceTier::Unsupported,
                }
            }
            Precision::BF16 => {
                // BF16 not natively supported on Intel NPU
                if NATIVE_OPS.contains(&op_name) || EMULATED_OPS.contains(&op_name) {
                    PerformanceTier::Emulated
                } else {
                    PerformanceTier::Unsupported
                }
            }
        }
    }

    fn hardware_name(&self) -> &str {
        "Intel NPU"
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
        let s = IntelNpuSupport;
        assert_eq!(
            s.op_support("MatMul", Precision::F16),
            PerformanceTier::Native
        );
        assert_eq!(
            s.op_support("Conv", Precision::F16),
            PerformanceTier::Native
        );
        assert_eq!(
            s.op_support("Relu", Precision::F16),
            PerformanceTier::Native
        );
    }

    #[test]
    fn emulated_at_f16() {
        let s = IntelNpuSupport;
        assert_eq!(
            s.op_support("Attention", Precision::F16),
            PerformanceTier::Emulated
        );
        assert_eq!(
            s.op_support("ReduceSum", Precision::F16),
            PerformanceTier::Emulated
        );
    }

    #[test]
    fn f32_ops_emulated() {
        let s = IntelNpuSupport;
        assert_eq!(
            s.op_support("MatMul", Precision::F32),
            PerformanceTier::Emulated
        );
    }

    #[test]
    fn unknown_op_unsupported() {
        let s = IntelNpuSupport;
        assert_eq!(
            s.op_support("FakeOp", Precision::F16),
            PerformanceTier::Unsupported
        );
    }

    #[test]
    fn hardware_name() {
        assert_eq!(IntelNpuSupport.hardware_name(), "Intel NPU");
    }

    #[test]
    fn native_and_emulated_lists() {
        let s = IntelNpuSupport;
        assert!(s.native_ops().contains(&"MatMul"));
        assert!(s.emulated_ops().contains(&"Attention"));
    }
}
