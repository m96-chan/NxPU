//! Rockchip RKNN NPU backend for NxPU.
//!
//! Thin vendor wrapper that delegates compilation to the ONNX backend
//! and emits `.onnx` files suitable for the Rockchip RKNN toolkit.

use nxpu_backend_core::{
    Backend, BackendError, BackendOptions, BackendOutput, Diagnostic, DiagnosticLevel,
};
use nxpu_backend_onnx::OnnxBackend;
use nxpu_ir::Module;

/// Rockchip RKNN NPU backend (delegates to ONNX).
#[derive(Debug)]
pub struct RockchipBackend;

impl Backend for RockchipBackend {
    fn name(&self) -> &str {
        "Rockchip RKNN NPU"
    }

    fn targets(&self) -> &[&str] {
        &["rockchip", "rknn"]
    }

    fn compile(
        &self,
        module: &Module,
        opts: &BackendOptions,
    ) -> Result<BackendOutput, BackendError> {
        let mut output = OnnxBackend.compile(module, opts)?;
        output.diagnostics.push(Diagnostic {
            level: DiagnosticLevel::Info,
            message: "To compile for RKNN: rknn.load_onnx(\"output.onnx\")".into(),
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
        let backend = RockchipBackend;
        assert_eq!(backend.name(), "Rockchip RKNN NPU");
        assert!(backend.targets().contains(&"rockchip"));
        assert!(backend.targets().contains(&"rknn"));
    }

    #[test]
    fn compile_matmul_delegates() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/matmul.wgsl"
        ))
        .unwrap();
        let module = nxpu_parser::parse(&source).unwrap();

        let output = RockchipBackend
            .compile(&module, &BackendOptions::default())
            .unwrap();
        assert_eq!(output.files.len(), 1);
        assert_eq!(output.files[0].name, "output.onnx");
        assert!(matches!(output.files[0].content, OutputContent::Binary(_)));
    }
}
