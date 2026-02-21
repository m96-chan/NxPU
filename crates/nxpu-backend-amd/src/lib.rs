//! AMD XDNA NPU backend for NxPU.
//!
//! Thin vendor wrapper that delegates compilation to the ONNX backend
//! and emits `.onnx` files suitable for AMD XDNA / Vitis AI.

use nxpu_backend_core::{
    Backend, BackendError, BackendOptions, BackendOutput, Diagnostic, DiagnosticLevel,
};
use nxpu_backend_onnx::OnnxBackend;
use nxpu_ir::Module;

/// AMD XDNA NPU backend (delegates to ONNX).
#[derive(Debug)]
pub struct AmdBackend;

impl Backend for AmdBackend {
    fn name(&self) -> &str {
        "AMD XDNA NPU"
    }

    fn targets(&self) -> &[&str] {
        &["amd-xdna", "amd-npu"]
    }

    fn compile(
        &self,
        module: &Module,
        opts: &BackendOptions,
    ) -> Result<BackendOutput, BackendError> {
        let mut output = OnnxBackend.compile(module, opts)?;
        output.diagnostics.push(Diagnostic {
            level: DiagnosticLevel::Info,
            message: "To compile for XDNA: use Vitis AI EP with ONNX Runtime".into(),
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
        let backend = AmdBackend;
        assert_eq!(backend.name(), "AMD XDNA NPU");
        assert!(backend.targets().contains(&"amd-xdna"));
        assert!(backend.targets().contains(&"amd-npu"));
    }

    #[test]
    fn compile_matmul_delegates() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/matmul.wgsl"
        ))
        .unwrap();
        let module = nxpu_parser::parse(&source).unwrap();

        let output = AmdBackend
            .compile(&module, &BackendOptions::default())
            .unwrap();
        assert_eq!(output.files.len(), 1);
        assert_eq!(output.files[0].name, "output.onnx");
        assert!(matches!(output.files[0].content, OutputContent::Binary(_)));
    }
}
