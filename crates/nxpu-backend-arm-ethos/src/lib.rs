//! Arm Ethos NPU backend for NxPU.
//!
//! Thin vendor wrapper that delegates compilation to the TFLite backend
//! and emits `.tflite` files suitable for the Arm Ethos-U / Vela toolchain.

use nxpu_backend_core::{
    Backend, BackendError, BackendOptions, BackendOutput, Diagnostic, DiagnosticLevel,
};
use nxpu_backend_tflite::TfLiteBackend;
use nxpu_ir::Module;

/// Arm Ethos NPU backend (delegates to TFLite).
#[derive(Debug)]
pub struct ArmEthosBackend;

impl Backend for ArmEthosBackend {
    fn name(&self) -> &str {
        "Arm Ethos NPU"
    }

    fn targets(&self) -> &[&str] {
        &["arm-ethos", "ethos-u"]
    }

    fn compile(
        &self,
        module: &Module,
        opts: &BackendOptions,
    ) -> Result<BackendOutput, BackendError> {
        let mut output = TfLiteBackend.compile(module, opts)?;
        output.diagnostics.push(Diagnostic {
            level: DiagnosticLevel::Info,
            message: "To compile for Ethos NPU: vela output.tflite".into(),
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
        let backend = ArmEthosBackend;
        assert_eq!(backend.name(), "Arm Ethos NPU");
        assert!(backend.targets().contains(&"arm-ethos"));
        assert!(backend.targets().contains(&"ethos-u"));
    }

    #[test]
    fn compile_matmul_delegates() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/matmul.wgsl"
        ))
        .unwrap();
        let module = nxpu_parser::parse(&source).unwrap();

        let output = ArmEthosBackend
            .compile(&module, &BackendOptions::default())
            .unwrap();
        assert_eq!(output.files.len(), 1);
        assert_eq!(output.files[0].name, "output.tflite");
        assert!(matches!(output.files[0].content, OutputContent::Binary(_)));
    }
}
