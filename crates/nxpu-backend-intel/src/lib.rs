//! Intel NPU backend for NxPU.
//!
//! Thin vendor wrapper that delegates compilation to the ONNX backend
//! and emits `.onnx` files suitable for OpenVINO / Intel NPU.

use nxpu_backend_core::{
    Backend, BackendError, BackendOptions, BackendOutput, Diagnostic, DiagnosticLevel,
};
use nxpu_backend_onnx::OnnxBackend;
use nxpu_ir::Module;

/// Intel NPU backend (delegates to ONNX).
#[derive(Debug)]
pub struct IntelBackend;

impl Backend for IntelBackend {
    fn name(&self) -> &str {
        "Intel NPU"
    }

    fn targets(&self) -> &[&str] {
        &["intel-npu", "openvino"]
    }

    fn compile(
        &self,
        module: &Module,
        opts: &BackendOptions,
    ) -> Result<BackendOutput, BackendError> {
        let mut output = OnnxBackend.compile(module, opts)?;
        output.diagnostics.push(Diagnostic {
            level: DiagnosticLevel::Info,
            message: "Load in OpenVINO: ov::Core::read_model(\"output.onnx\")".into(),
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
        let backend = IntelBackend;
        assert_eq!(backend.name(), "Intel NPU");
        assert!(backend.targets().contains(&"intel-npu"));
        assert!(backend.targets().contains(&"openvino"));
    }

    #[test]
    fn compile_matmul_delegates() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/matmul.wgsl"
        ))
        .unwrap();
        let module = nxpu_parser::parse(&source).unwrap();

        let output = IntelBackend
            .compile(&module, &BackendOptions::default())
            .unwrap();
        assert_eq!(output.files.len(), 1);
        assert_eq!(output.files[0].name, "output.onnx");
        assert!(matches!(output.files[0].content, OutputContent::Binary(_)));
    }
}
