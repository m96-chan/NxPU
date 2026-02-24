//! Arm Ethos-U NPU operator support matrices.

use nxpu_backend_core::{OperatorSupport, PerformanceTier, Precision};

/// Operator support for Ethos-U55 (128 MAC configuration).
pub struct EthosU55Support;

/// Operator support for Ethos-U65 (512 MAC configuration).
pub struct EthosU65Support;

const ETHOS_NATIVE_INT8: &[&str] = &[
    "Conv",
    "MatMul",
    "Add",
    "MaxPool",
    "AveragePool",
    "Relu",
    "Reshape",
    "Concat",
];

const ETHOS_EMULATED_INT8: &[&str] = &[
    "Mul",
    "Sub",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "BatchNormalization",
    "Transpose",
];

/// Ops that U65 additionally supports natively (Int16).
pub(crate) const U65_NATIVE_INT16: &[&str] = &["Conv", "MatMul", "Add", "Relu"];

impl OperatorSupport for EthosU55Support {
    fn op_support(&self, op_name: &str, precision: Precision) -> PerformanceTier {
        match precision {
            Precision::Int8 => {
                if ETHOS_NATIVE_INT8.contains(&op_name) {
                    PerformanceTier::Native
                } else if ETHOS_EMULATED_INT8.contains(&op_name) {
                    PerformanceTier::Emulated
                } else {
                    PerformanceTier::Unsupported
                }
            }
            // U55 only supports Int8 natively
            _ => {
                if ETHOS_NATIVE_INT8.contains(&op_name) || ETHOS_EMULATED_INT8.contains(&op_name) {
                    PerformanceTier::Emulated
                } else {
                    PerformanceTier::Unsupported
                }
            }
        }
    }

    fn hardware_name(&self) -> &str {
        "Arm Ethos-U55"
    }

    fn native_ops(&self) -> &[&str] {
        ETHOS_NATIVE_INT8
    }

    fn emulated_ops(&self) -> &[&str] {
        ETHOS_EMULATED_INT8
    }
}

impl OperatorSupport for EthosU65Support {
    fn op_support(&self, op_name: &str, precision: Precision) -> PerformanceTier {
        match precision {
            Precision::Int8 => {
                if ETHOS_NATIVE_INT8.contains(&op_name) {
                    PerformanceTier::Native
                } else if ETHOS_EMULATED_INT8.contains(&op_name) {
                    PerformanceTier::Emulated
                } else {
                    PerformanceTier::Unsupported
                }
            }
            // U65 has limited Int16 support (mapped from F16)
            Precision::F16 => {
                if U65_NATIVE_INT16.contains(&op_name) {
                    PerformanceTier::Native
                } else if ETHOS_NATIVE_INT8.contains(&op_name)
                    || ETHOS_EMULATED_INT8.contains(&op_name)
                {
                    PerformanceTier::Emulated
                } else {
                    PerformanceTier::Unsupported
                }
            }
            _ => {
                if ETHOS_NATIVE_INT8.contains(&op_name) || ETHOS_EMULATED_INT8.contains(&op_name) {
                    PerformanceTier::Emulated
                } else {
                    PerformanceTier::Unsupported
                }
            }
        }
    }

    fn hardware_name(&self) -> &str {
        "Arm Ethos-U65"
    }

    fn native_ops(&self) -> &[&str] {
        ETHOS_NATIVE_INT8
    }

    fn emulated_ops(&self) -> &[&str] {
        ETHOS_EMULATED_INT8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u55_native_int8() {
        let s = EthosU55Support;
        assert_eq!(
            s.op_support("Conv", Precision::Int8),
            PerformanceTier::Native
        );
        assert_eq!(
            s.op_support("MatMul", Precision::Int8),
            PerformanceTier::Native
        );
        assert_eq!(
            s.op_support("Relu", Precision::Int8),
            PerformanceTier::Native
        );
    }

    #[test]
    fn u55_emulated_int8() {
        let s = EthosU55Support;
        assert_eq!(
            s.op_support("Sigmoid", Precision::Int8),
            PerformanceTier::Emulated
        );
    }

    #[test]
    fn u55_f16_emulated() {
        let s = EthosU55Support;
        assert_eq!(
            s.op_support("Conv", Precision::F16),
            PerformanceTier::Emulated
        );
    }

    #[test]
    fn u65_f16_native_for_core_ops() {
        let s = EthosU65Support;
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
    fn u65_f16_emulated_for_others() {
        let s = EthosU65Support;
        assert_eq!(
            s.op_support("Sigmoid", Precision::F16),
            PerformanceTier::Emulated
        );
    }

    #[test]
    fn unknown_op_unsupported() {
        assert_eq!(
            EthosU55Support.op_support("FakeOp", Precision::Int8),
            PerformanceTier::Unsupported
        );
        assert_eq!(
            EthosU65Support.op_support("FakeOp", Precision::Int8),
            PerformanceTier::Unsupported
        );
    }

    #[test]
    fn hardware_names() {
        assert_eq!(EthosU55Support.hardware_name(), "Arm Ethos-U55");
        assert_eq!(EthosU65Support.hardware_name(), "Arm Ethos-U65");
    }

    #[test]
    fn u55_bf16_emulated() {
        let s = EthosU55Support;
        assert_eq!(
            s.op_support("Conv", Precision::BF16),
            PerformanceTier::Emulated
        );
    }

    #[test]
    fn u55_f32_unknown_unsupported() {
        let s = EthosU55Support;
        assert_eq!(
            s.op_support("FakeOp", Precision::F32),
            PerformanceTier::Unsupported
        );
    }

    #[test]
    fn u65_bf16_emulated() {
        let s = EthosU65Support;
        assert_eq!(
            s.op_support("Conv", Precision::BF16),
            PerformanceTier::Emulated
        );
    }

    #[test]
    fn u65_f32_emulated() {
        let s = EthosU65Support;
        assert_eq!(
            s.op_support("Conv", Precision::F32),
            PerformanceTier::Emulated
        );
    }

    #[test]
    fn u65_f16_unsupported() {
        let s = EthosU65Support;
        assert_eq!(
            s.op_support("FakeOp", Precision::F16),
            PerformanceTier::Unsupported
        );
    }

    #[test]
    fn u65_int8_emulated() {
        let s = EthosU65Support;
        assert_eq!(
            s.op_support("Sigmoid", Precision::Int8),
            PerformanceTier::Emulated
        );
    }
}
