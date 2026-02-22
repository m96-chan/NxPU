//! CoreML backend emitter for NxPU (Apple ANE).
//!
//! Emits CoreML ML Program (`.mlmodel`) protobuf from NxPU IR.
//! The Apple Neural Engine operates at FP16 precision.

use nxpu_analysis::analyze;
use nxpu_backend_core::{
    Backend, BackendError, BackendOptions, BackendOutput, Diagnostic, DiagnosticLevel,
    OutputContent, OutputFile, Precision,
};
use nxpu_ir::Module;
use prost::Message;

mod lower;
pub mod proto;

/// CoreML backend targeting Apple Neural Engine.
#[derive(Debug)]
pub struct CoreMlBackend;

impl Backend for CoreMlBackend {
    fn name(&self) -> &str {
        "CoreML"
    }

    fn targets(&self) -> &[&str] {
        &["coreml", "apple-ane"]
    }

    fn preferred_precision(&self) -> Precision {
        Precision::F16
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
                analyze::KernelPattern::MatMul { .. } => "matmul",
                analyze::KernelPattern::ElementWise { op, .. } => op.onnx_op_type(),
                analyze::KernelPattern::Conv2D { .. } => "conv",
                analyze::KernelPattern::Pool { kind, .. } => match kind {
                    analyze::PoolKind::Max => "max_pool",
                    analyze::PoolKind::Avg => "avg_pool",
                },
                analyze::KernelPattern::Activation { op, .. } => op.onnx_op_type(),
                analyze::KernelPattern::Reduce { op, .. } => op.onnx_op_type(),
                analyze::KernelPattern::Transpose { .. } => "transpose",
                analyze::KernelPattern::Reshape { .. } => "reshape",
                analyze::KernelPattern::Normalization { .. } => "batch_norm",
                analyze::KernelPattern::Concat { .. } => "concat",
                analyze::KernelPattern::Split { .. } => "split",
                analyze::KernelPattern::Attention { .. } => "scaled_dot_product_attention",
                analyze::KernelPattern::Unknown { .. } => "Unknown",
            };

            diagnostics.push(Diagnostic {
                level: DiagnosticLevel::Info,
                message: format!("entry point '{}': MIL op {}", ep.name, summary),
            });

            let model = lower::build_model(&pattern, &ep.name);
            let bytes = model.encode_to_vec();

            let filename = if module.entry_points.len() == 1 {
                "output.mlmodel".into()
            } else {
                format!("{}.mlmodel", ep.name)
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
        let backend = CoreMlBackend;
        assert_eq!(backend.name(), "CoreML");
        assert!(backend.targets().contains(&"coreml"));
        assert!(backend.targets().contains(&"apple-ane"));
    }

    #[test]
    fn compile_matmul_wgsl() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/matmul.wgsl"
        ))
        .unwrap();
        let module = nxpu_parser::parse(&source).unwrap();

        let output = CoreMlBackend
            .compile(&module, &BackendOptions::default())
            .unwrap();
        assert_eq!(output.files.len(), 1);
        assert_eq!(output.files[0].name, "output.mlmodel");

        let bytes = match &output.files[0].content {
            OutputContent::Binary(b) => b,
            _ => panic!("expected binary output"),
        };

        let model = proto::Model::decode(bytes.as_slice()).unwrap();
        assert_eq!(model.specification_version, proto::SPECIFICATION_VERSION);
        let proto::model::Type::MlProgram(prog) = model.r#type.as_ref().unwrap();
        assert_eq!(
            prog.functions[0].block.as_ref().unwrap().operations[0].r#type,
            "matmul"
        );
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
        let output = CoreMlBackend
            .compile(&module, &BackendOptions::default())
            .unwrap();

        let bytes = match &output.files[0].content {
            OutputContent::Binary(b) => b,
            _ => panic!("expected binary"),
        };
        let model = proto::Model::decode(bytes.as_slice()).unwrap();
        let proto::model::Type::MlProgram(prog) = model.r#type.as_ref().unwrap();
        assert_eq!(
            prog.functions[0].block.as_ref().unwrap().operations[0].r#type,
            "add"
        );
    }
}
