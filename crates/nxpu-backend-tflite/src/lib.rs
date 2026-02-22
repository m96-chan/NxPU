//! TFLite/LiteRT backend emitter for NxPU.
//!
//! Emits TFLite FlatBuffer (`.tflite`) models from NxPU IR.
//! Targets: MediaTek APU (NeuroPilot), Google Edge TPU, Arm Ethos NPU (Vela).

use nxpu_analysis::analyze;
use nxpu_backend_core::{
    Backend, BackendError, BackendOptions, BackendOutput, Diagnostic, DiagnosticLevel,
    OutputContent, OutputFile,
};
use nxpu_ir::Module;

mod lower;
mod schema;

/// TFLite backend that compiles NxPU IR into `.tflite` FlatBuffer files.
#[derive(Debug)]
pub struct TfLiteBackend;

impl Backend for TfLiteBackend {
    fn name(&self) -> &str {
        "TFLite"
    }

    fn targets(&self) -> &[&str] {
        &["tflite", "litert"]
    }

    fn compile(
        &self,
        module: &Module,
        _opts: &BackendOptions,
    ) -> Result<BackendOutput, BackendError> {
        if module.entry_points.is_empty() {
            return Err(BackendError::Other("no entry points in module".into()));
        }

        let mut files = Vec::new();
        let mut diagnostics = Vec::new();

        for (i, ep) in module.entry_points.iter().enumerate() {
            let pattern = analyze::classify_entry_point(module, i).map_err(|e| {
                BackendError::Unsupported(format!("entry point '{}': {e}", ep.name))
            })?;
            if let analyze::KernelPattern::Unknown { reason } = &pattern {
                return Err(BackendError::Unsupported(format!(
                    "entry point '{}': unrecognized pattern: {reason}",
                    ep.name
                )));
            }

            let summary = match &pattern {
                analyze::KernelPattern::MatMul { .. } => "BATCH_MATMUL",
                analyze::KernelPattern::ElementWise { op, .. } => op.op_name(),
                analyze::KernelPattern::Conv2D { .. } => "CONV_2D",
                analyze::KernelPattern::Pool { kind, .. } => kind.op_name(),
                analyze::KernelPattern::Activation { op, .. } => op.op_name(),
                analyze::KernelPattern::Reduce { op, .. } => op.op_name(),
                analyze::KernelPattern::Transpose { .. } => "TRANSPOSE",
                analyze::KernelPattern::Reshape { .. } => "RESHAPE",
                analyze::KernelPattern::Normalization { .. } => "BatchNormalization",
                analyze::KernelPattern::Concat { .. } => "CONCATENATION",
                analyze::KernelPattern::Split { .. } => "SPLIT",
                analyze::KernelPattern::Attention { .. } => "Attention",
                analyze::KernelPattern::Unknown { .. } => "Unknown",
            };

            diagnostics.push(Diagnostic {
                level: DiagnosticLevel::Info,
                message: format!("entry point '{}': classified as {}", ep.name, summary),
            });

            let bytes = lower::build_model(&pattern);

            let filename = if module.entry_points.len() == 1 {
                "output.tflite".into()
            } else {
                format!("{}.tflite", ep.name)
            };

            files.push(OutputFile {
                name: filename,
                content: OutputContent::Binary(bytes),
            });
        }

        Ok(BackendOutput { files, diagnostics })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nxpu_backend_core::BackendOptions;

    #[test]
    fn backend_metadata() {
        let backend = TfLiteBackend;
        assert_eq!(backend.name(), "TFLite");
        assert!(backend.targets().contains(&"tflite"));
        assert!(backend.targets().contains(&"litert"));
    }

    #[test]
    fn compile_empty_module_fails() {
        let backend = TfLiteBackend;
        let module = Module::default();
        let result = backend.compile(&module, &BackendOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn compile_matmul_wgsl() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/matmul.wgsl"
        ))
        .unwrap();
        let module = nxpu_parser::parse(&source).unwrap();

        let backend = TfLiteBackend;
        let output = backend
            .compile(&module, &BackendOptions::default())
            .unwrap();

        assert_eq!(output.files.len(), 1);
        assert_eq!(output.files[0].name, "output.tflite");

        let bytes = match &output.files[0].content {
            OutputContent::Binary(b) => b,
            _ => panic!("expected binary output"),
        };

        // Verify TFLite file identifier
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn compile_vecadd_wgsl() {
        let source = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

struct Params { N: u32 }
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.N) { return; }
  c[idx] = a[idx] + b[idx];
}
"#;

        let module = nxpu_parser::parse(source).unwrap();

        let backend = TfLiteBackend;
        let output = backend
            .compile(&module, &BackendOptions::default())
            .unwrap();

        assert_eq!(output.files.len(), 1);
        let bytes = match &output.files[0].content {
            OutputContent::Binary(b) => b,
            _ => panic!("expected binary output"),
        };
        assert_eq!(&bytes[4..8], b"TFL3");
    }
}
