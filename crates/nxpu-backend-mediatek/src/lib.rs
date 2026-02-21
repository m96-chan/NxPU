//! MediaTek APU backend for NxPU.
//!
//! Thin vendor wrapper that delegates compilation to the TFLite backend
//! and emits `.tflite` files suitable for the MediaTek NeuroPilot toolchain.

use nxpu_backend_core::{
    Backend, BackendError, BackendOptions, BackendOutput, Diagnostic, DiagnosticLevel,
};
use nxpu_backend_tflite::TfLiteBackend;
use nxpu_ir::Module;

/// MediaTek APU backend (delegates to TFLite).
#[derive(Debug)]
pub struct MediaTekBackend;

impl Backend for MediaTekBackend {
    fn name(&self) -> &str {
        "MediaTek APU"
    }

    fn targets(&self) -> &[&str] {
        &["mediatek", "neuropilot"]
    }

    fn compile(
        &self,
        module: &Module,
        opts: &BackendOptions,
    ) -> Result<BackendOutput, BackendError> {
        let mut output = TfLiteBackend.compile(module, opts)?;
        output.diagnostics.push(Diagnostic {
            level: DiagnosticLevel::Info,
            message: "To compile for MediaTek APU: ncc-tflite output.tflite".into(),
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
        let backend = MediaTekBackend;
        assert_eq!(backend.name(), "MediaTek APU");
        assert!(backend.targets().contains(&"mediatek"));
        assert!(backend.targets().contains(&"neuropilot"));
    }

    #[test]
    fn compile_matmul_delegates() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/matmul.wgsl"
        ))
        .unwrap();
        let module = nxpu_parser::parse(&source).unwrap();

        let output = MediaTekBackend
            .compile(&module, &BackendOptions::default())
            .unwrap();
        assert_eq!(output.files.len(), 1);
        assert_eq!(output.files[0].name, "output.tflite");
        assert!(matches!(output.files[0].content, OutputContent::Binary(_)));
    }
}
