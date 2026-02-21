//! Samsung Exynos NPU backend for NxPU.
//!
//! Thin vendor wrapper that delegates compilation to the ONNX backend
//! and emits `.onnx` files suitable for the Samsung Exynos NPU toolchain.

use nxpu_backend_core::{
    Backend, BackendError, BackendOptions, BackendOutput, Diagnostic, DiagnosticLevel,
};
use nxpu_backend_onnx::OnnxBackend;
use nxpu_ir::Module;

/// Samsung Exynos NPU backend (delegates to ONNX).
#[derive(Debug)]
pub struct SamsungBackend;

impl Backend for SamsungBackend {
    fn name(&self) -> &str {
        "Samsung Exynos NPU"
    }

    fn targets(&self) -> &[&str] {
        &["samsung", "exynos"]
    }

    fn compile(
        &self,
        module: &Module,
        opts: &BackendOptions,
    ) -> Result<BackendOutput, BackendError> {
        let mut output = OnnxBackend.compile(module, opts)?;
        output.diagnostics.push(Diagnostic {
            level: DiagnosticLevel::Info,
            message: "To compile for Exynos NPU: one-import-onnx output.onnx".into(),
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
        let backend = SamsungBackend;
        assert_eq!(backend.name(), "Samsung Exynos NPU");
        assert!(backend.targets().contains(&"samsung"));
        assert!(backend.targets().contains(&"exynos"));
    }

    #[test]
    fn compile_matmul_delegates() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/matmul.wgsl"
        ))
        .unwrap();
        let module = nxpu_parser::parse(&source).unwrap();

        let output = SamsungBackend
            .compile(&module, &BackendOptions::default())
            .unwrap();
        assert_eq!(output.files.len(), 1);
        assert_eq!(output.files[0].name, "output.onnx");
        assert!(matches!(output.files[0].content, OutputContent::Binary(_)));
    }
}
