//! Qualcomm Hexagon NPU backend for NxPU.
//!
//! Thin vendor wrapper that delegates compilation to the ONNX backend
//! and emits `.onnx` files suitable for the Qualcomm QNN toolchain.

use nxpu_backend_core::{
    Backend, BackendError, BackendOptions, BackendOutput, Diagnostic, DiagnosticLevel, Precision,
};
use nxpu_backend_onnx::OnnxBackend;
use nxpu_ir::Module;

/// Qualcomm Hexagon NPU backend (delegates to ONNX).
#[derive(Debug)]
pub struct QualcommBackend;

impl Backend for QualcommBackend {
    fn name(&self) -> &str {
        "Qualcomm Hexagon NPU"
    }

    fn targets(&self) -> &[&str] {
        &["qualcomm", "hexagon-npu"]
    }

    fn preferred_precision(&self) -> Precision {
        Precision::Int8
    }

    fn compile(
        &self,
        module: &Module,
        opts: &BackendOptions,
    ) -> Result<BackendOutput, BackendError> {
        let mut output = OnnxBackend.compile(module, opts)?;
        output.diagnostics.push(Diagnostic {
            level: DiagnosticLevel::Info,
            message: "To compile for Hexagon NPU: qnn-onnx-converter output.onnx".into(),
        });
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nxpu_backend_core::{BackendOptions, OutputContent};

    #[test]
    fn backend_metadata() {
        let backend = QualcommBackend;
        assert_eq!(backend.name(), "Qualcomm Hexagon NPU");
        assert!(backend.targets().contains(&"qualcomm"));
        assert!(backend.targets().contains(&"hexagon-npu"));
    }

    #[test]
    fn compile_matmul_delegates() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/matmul.wgsl"
        ))
        .unwrap();
        let module = nxpu_parser::parse(&source).unwrap();

        let output = QualcommBackend
            .compile(&module, &BackendOptions::default())
            .unwrap();
        assert_eq!(output.files.len(), 1);
        assert_eq!(output.files[0].name, "output.onnx");
        assert!(matches!(output.files[0].content, OutputContent::Binary(_)));
    }
}
