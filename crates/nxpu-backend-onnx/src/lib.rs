//! ONNX backend emitter for NxPU.
//!
//! Analyzes NxPU IR to recognize high-level tensor operations (MatMul,
//! element-wise ops) and emits ONNX (`.onnx`) model files using protobuf
//! serialization via prost.

use nxpu_analysis::{analyze, fusion};
use nxpu_backend_core::{
    Backend, BackendError, BackendOptions, BackendOutput, Diagnostic, DiagnosticLevel,
    OutputContent, OutputFile,
};
use nxpu_ir::Module;
use prost::Message;

mod lower;
#[doc(hidden)]
pub mod proto;

/// ONNX backend that compiles NxPU IR into `.onnx` model files.
#[derive(Debug)]
pub struct OnnxBackend;

impl Backend for OnnxBackend {
    fn name(&self) -> &str {
        "ONNX"
    }

    fn targets(&self) -> &[&str] {
        &["onnx"]
    }

    fn compile(
        &self,
        module: &Module,
        _opts: &BackendOptions,
    ) -> Result<BackendOutput, BackendError> {
        if module.entry_points.is_empty() {
            return Err(BackendError::Other("no entry points in module".into()));
        }

        // 1. Classify all entry points.
        let mut patterns = Vec::new();
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
            patterns.push(pattern);
        }

        // 2. Extract embedded weights from module globals.
        let weights = analyze::extract_embedded_weights(module);

        // 3. Fuse adjacent patterns.
        let fused = fusion::fuse_patterns(patterns);

        // 4. Lower each fused pattern.
        let mut files = Vec::new();
        let mut diagnostics = Vec::new();

        for (fp, ep_idx) in &fused {
            let ep_name = if *ep_idx < module.entry_points.len() {
                &module.entry_points[*ep_idx].name
            } else {
                "fused"
            };

            let summary = match fp {
                fusion::FusedPattern::Single(p) => pattern_summary(p).to_string(),
                fusion::FusedPattern::ConvBatchNorm { .. } => "Conv+BatchNorm (fused)".into(),
                fusion::FusedPattern::WithActivation {
                    base, activation, ..
                } => {
                    let base_name = match base.as_ref() {
                        fusion::FusedPattern::Single(p) => pattern_summary(p),
                        fusion::FusedPattern::ConvBatchNorm { .. } => "Conv+BatchNorm",
                        fusion::FusedPattern::MatMulBias { .. } => "Gemm",
                        _ => "fused",
                    };
                    format!("{base_name}+{activation:?}")
                }
                fusion::FusedPattern::MatMulBias { .. } => "Gemm (fused)".into(),
            };

            diagnostics.push(Diagnostic {
                level: DiagnosticLevel::Info,
                message: format!("entry point '{ep_name}': classified as {summary}"),
            });

            let model = lower::build_fused_model(fp, ep_name, &weights)?;
            let bytes = model.encode_to_vec();

            let filename = if fused.len() == 1 {
                "output.onnx".into()
            } else {
                format!("{ep_name}.onnx")
            };

            files.push(OutputFile {
                name: filename,
                content: OutputContent::Binary(bytes),
            });
        }

        Ok(BackendOutput { files, diagnostics })
    }
}

fn pattern_summary(pattern: &analyze::KernelPattern) -> &'static str {
    match pattern {
        analyze::KernelPattern::MatMul { .. } => "MatMul",
        analyze::KernelPattern::ElementWise { op, .. } => op.op_name(),
        analyze::KernelPattern::Conv2D { .. } => "Conv",
        analyze::KernelPattern::Pool { kind, .. } => kind.op_name(),
        analyze::KernelPattern::Activation { op, .. } => op.op_name(),
        analyze::KernelPattern::Reduce { op, .. } => op.op_name(),
        analyze::KernelPattern::Transpose { .. } => "Transpose",
        analyze::KernelPattern::Reshape { .. } => "Reshape",
        analyze::KernelPattern::Normalization { .. } => "BatchNormalization",
        analyze::KernelPattern::Concat { .. } => "Concat",
        analyze::KernelPattern::Split { .. } => "Split",
        analyze::KernelPattern::Attention { .. } => "Attention",
        analyze::KernelPattern::Unknown { .. } => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nxpu_backend_core::BackendOptions;

    #[test]
    fn backend_metadata() {
        let backend = OnnxBackend;
        assert_eq!(backend.name(), "ONNX");
        assert!(backend.targets().contains(&"onnx"));
    }

    #[test]
    fn compile_empty_module_fails() {
        let backend = OnnxBackend;
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

        let backend = OnnxBackend;
        let output = backend
            .compile(&module, &BackendOptions::default())
            .unwrap();

        assert_eq!(output.files.len(), 1);
        assert_eq!(output.files[0].name, "output.onnx");

        let bytes = match &output.files[0].content {
            OutputContent::Binary(b) => b,
            _ => panic!("expected binary output"),
        };

        // Decode and verify the ONNX model.
        let model = proto::ModelProto::decode(bytes.as_slice()).unwrap();
        assert_eq!(model.ir_version, 7);
        assert_eq!(model.producer_name, "nxpu");
        assert_eq!(model.opset_import[0].version, 13);

        let graph = model.graph.as_ref().unwrap();
        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "MatMul");
        assert_eq!(graph.input.len(), 2);
        assert_eq!(graph.output.len(), 1);
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

        let backend = OnnxBackend;
        let output = backend
            .compile(&module, &BackendOptions::default())
            .unwrap();

        assert_eq!(output.files.len(), 1);

        let bytes = match &output.files[0].content {
            OutputContent::Binary(b) => b,
            _ => panic!("expected binary output"),
        };

        let model = proto::ModelProto::decode(bytes.as_slice()).unwrap();
        let graph = model.graph.as_ref().unwrap();
        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "Add");
        assert_eq!(graph.input.len(), 2);
        assert_eq!(graph.output.len(), 1);
    }
}
