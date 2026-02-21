//! StableHLO/XLA backend emitter for NxPU (Google Cloud TPU).
//!
//! Emits StableHLO MLIR textual format (`.mlir`) from NxPU IR.
//! StableHLO is the native entry point for Google Cloud TPU via OpenXLA.

use nxpu_backend_core::{
    Backend, BackendError, BackendOptions, BackendOutput, Diagnostic, DiagnosticLevel,
    OutputContent, OutputFile,
};
use nxpu_backend_onnx::analyze;
use nxpu_ir::Module;

mod lower;

/// StableHLO backend targeting Google Cloud TPU via OpenXLA.
#[derive(Debug)]
pub struct StableHloBackend;

impl Backend for StableHloBackend {
    fn name(&self) -> &str {
        "StableHLO"
    }

    fn targets(&self) -> &[&str] {
        &["stablehlo", "xla"]
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

            let summary = match &pattern {
                analyze::KernelPattern::MatMul { .. } => "dot_general",
                analyze::KernelPattern::ElementWise { op, .. } => op.onnx_op_type(),
                analyze::KernelPattern::Conv2D { .. } => "convolution",
                analyze::KernelPattern::Pool { .. } => "reduce_window",
                analyze::KernelPattern::Activation { op, .. } => op.onnx_op_type(),
                analyze::KernelPattern::Reduce { .. } => "reduce",
                analyze::KernelPattern::Transpose { .. } => "transpose",
                analyze::KernelPattern::Reshape { .. } => "reshape",
                analyze::KernelPattern::Normalization { .. } => "batch_norm_inference",
            };

            diagnostics.push(Diagnostic {
                level: DiagnosticLevel::Info,
                message: format!("entry point '{}': StableHLO {}", ep.name, summary),
            });

            let mlir = lower::build_mlir(&pattern, &ep.name);

            let filename = if module.entry_points.len() == 1 {
                "output.mlir".into()
            } else {
                format!("{}.mlir", ep.name)
            };

            files.push(OutputFile {
                name: filename,
                content: OutputContent::Text(mlir),
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
        let backend = StableHloBackend;
        assert_eq!(backend.name(), "StableHLO");
        assert!(backend.targets().contains(&"stablehlo"));
        assert!(backend.targets().contains(&"xla"));
    }

    #[test]
    fn compile_matmul_wgsl() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/matmul.wgsl"
        ))
        .unwrap();
        let module = nxpu_parser::parse(&source).unwrap();

        let output = StableHloBackend
            .compile(&module, &BackendOptions::default())
            .unwrap();
        assert_eq!(output.files.len(), 1);
        assert_eq!(output.files[0].name, "output.mlir");

        let text = match &output.files[0].content {
            OutputContent::Text(t) => t,
            _ => panic!("expected text output"),
        };
        assert!(text.contains("stablehlo.dot_general"));
        assert!(text.contains("module @main"));
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
        let output = StableHloBackend
            .compile(&module, &BackendOptions::default())
            .unwrap();

        let text = match &output.files[0].content {
            OutputContent::Text(t) => t,
            _ => panic!("expected text"),
        };
        assert!(text.contains("stablehlo.add"));
    }
}
