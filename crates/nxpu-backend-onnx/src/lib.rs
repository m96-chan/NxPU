//! ONNX backend emitter for NxPU.
//!
//! Analyzes NxPU IR to recognize high-level tensor operations (MatMul,
//! element-wise ops) and emits ONNX (`.onnx`) model files using protobuf
//! serialization via prost.

use nxpu_backend_core::{
    Backend, BackendError, BackendOptions, BackendOutput, Diagnostic, DiagnosticLevel,
    OutputContent, OutputFile,
};
use nxpu_ir::Module;
use prost::Message;

pub mod analyze;
pub mod fusion;
mod lower;
pub mod proto;

pub use analyze::{
    ActivationOp, AnalysisError, Conv2DShape, ElementWiseOp, KernelPattern, MatMulShape, PoolKind,
    PoolShape, ReduceOp, TensorBinding, classify_entry_point,
};

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
            patterns.push(pattern);
        }

        // 2. Fuse adjacent patterns.
        let fused = fusion::fuse_patterns(patterns);

        // 3. Lower each fused pattern.
        let mut files = Vec::new();
        let mut diagnostics = Vec::new();

        for (i, fp) in fused.iter().enumerate() {
            let ep_name = if i < module.entry_points.len() {
                &module.entry_points[i].name
            } else {
                "fused"
            };

            let summary = match fp {
                fusion::FusedPattern::Single(p) => pattern_summary(p).to_string(),
                fusion::FusedPattern::ConvBatchNorm { .. } => "Conv+BatchNorm (fused)".into(),
                fusion::FusedPattern::WithActivation { base, activation } => {
                    let base_name = match base.as_ref() {
                        fusion::FusedPattern::Single(p) => pattern_summary(p),
                        fusion::FusedPattern::ConvBatchNorm { .. } => "Conv+BatchNorm",
                        _ => "fused",
                    };
                    format!("{base_name}+{activation:?}")
                }
            };

            diagnostics.push(Diagnostic {
                level: DiagnosticLevel::Info,
                message: format!("entry point '{ep_name}': classified as {summary}"),
            });

            // ONNX keeps separate nodes — runtime handles fusion.
            // Lower the primary pattern.
            let model = lower::build_model(fp.primary_pattern(), ep_name);
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

fn pattern_summary(pattern: &KernelPattern) -> &'static str {
    match pattern {
        KernelPattern::MatMul { .. } => "MatMul",
        KernelPattern::ElementWise { op, .. } => op.onnx_op_type(),
        KernelPattern::Conv2D { .. } => "Conv",
        KernelPattern::Pool { kind, .. } => kind.onnx_op_type(),
        KernelPattern::Activation { op, .. } => op.onnx_op_type(),
        KernelPattern::Reduce { op, .. } => op.onnx_op_type(),
        KernelPattern::Transpose { .. } => "Transpose",
        KernelPattern::Reshape { .. } => "Reshape",
        KernelPattern::Normalization { .. } => "BatchNormalization",
        KernelPattern::Concat { .. } => "Concat",
        KernelPattern::Split { .. } => "Split",
        KernelPattern::Attention { .. } => "Attention",
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
