//! Intel NPU backend for NxPU.
//!
//! Emits OpenVINO IR v11 format (`model.xml` + `model.bin`) alongside the
//! standard ONNX fallback. Validates operator patterns against the Intel NPU
//! support matrix and classifies entry points for vendor-aware diagnostics.

use nxpu_analysis::analyze;
use nxpu_backend_core::{
    Backend, BackendError, BackendOptions, BackendOutput, Diagnostic, DiagnosticLevel,
    OutputContent, OutputFile, Precision, validate_patterns,
};
use nxpu_backend_onnx::OnnxBackend;
use nxpu_ir::Module;

mod openvino_ir;
mod support;

use support::IntelNpuSupport;

/// Intel NPU backend with OpenVINO IR emission.
#[derive(Debug)]
pub struct IntelBackend;

impl Backend for IntelBackend {
    fn name(&self) -> &str {
        "Intel NPU"
    }

    fn targets(&self) -> &[&str] {
        &["intel-npu", "openvino"]
    }

    fn preferred_precision(&self) -> Precision {
        Precision::F16
    }

    fn compile(
        &self,
        module: &Module,
        opts: &BackendOptions,
    ) -> Result<BackendOutput, BackendError> {
        // 1. Classify entry points and collect op names for validation.
        let mut patterns = Vec::new();
        let mut op_names = Vec::new();
        for (i, ep) in module.entry_points.iter().enumerate() {
            match analyze::classify_entry_point(module, i) {
                Ok(pattern) => {
                    op_names.push(pattern_op_name(&pattern));
                    patterns.push(pattern);
                }
                Err(e) => {
                    return Err(BackendError::Unsupported(format!(
                        "entry point '{}': {e}",
                        ep.name
                    )));
                }
            }
        }

        // 2. Validate against Intel NPU support matrix.
        let precision = resolve_precision(opts, self.preferred_precision());
        let op_refs: Vec<&str> = op_names.iter().map(|s| s.as_str()).collect();
        let mut diagnostics = validate_patterns(&IntelNpuSupport, &op_refs, precision);

        // 3. Emit OpenVINO IR XML.
        let ir_xml = openvino_ir::build_ir_xml(&patterns, "nxpu_model");
        let mut files = vec![
            OutputFile {
                name: "model.xml".into(),
                content: OutputContent::Text(ir_xml),
            },
            OutputFile {
                name: "model.bin".into(),
                content: OutputContent::Binary(vec![]),
            },
        ];

        // 4. Also emit ONNX as fallback.
        match OnnxBackend.compile(module, opts) {
            Ok(onnx_output) => {
                files.extend(onnx_output.files);
                diagnostics.extend(onnx_output.diagnostics);
            }
            Err(e) => {
                diagnostics.push(Diagnostic {
                    level: DiagnosticLevel::Warning,
                    message: format!("ONNX fallback emission failed: {e}"),
                });
            }
        }

        diagnostics.push(Diagnostic {
            level: DiagnosticLevel::Info,
            message: "Load in OpenVINO: ov::Core::read_model(\"model.xml\")".into(),
        });
        diagnostics.push(Diagnostic {
            level: DiagnosticLevel::Info,
            message: "Alternative: ov::Core::read_model(\"output.onnx\")".into(),
        });

        Ok(BackendOutput { files, diagnostics })
    }
}

/// Map a KernelPattern to its ONNX-compatible op name string.
fn pattern_op_name(pattern: &analyze::KernelPattern) -> String {
    match pattern {
        analyze::KernelPattern::MatMul { .. } => "MatMul".into(),
        analyze::KernelPattern::ElementWise { op, .. } => op.op_name().into(),
        analyze::KernelPattern::Conv2D { .. } => "Conv".into(),
        analyze::KernelPattern::Pool { kind, .. } => kind.op_name().into(),
        analyze::KernelPattern::Activation { op, .. } => op.op_name().into(),
        analyze::KernelPattern::Reduce { op, .. } => op.op_name().into(),
        analyze::KernelPattern::Transpose { .. } => "Transpose".into(),
        analyze::KernelPattern::Reshape { .. } => "Reshape".into(),
        analyze::KernelPattern::Normalization { .. } => "BatchNormalization".into(),
        analyze::KernelPattern::Concat { .. } => "Concat".into(),
        analyze::KernelPattern::Split { .. } => "Split".into(),
        analyze::KernelPattern::Attention { .. } => "Attention".into(),
        analyze::KernelPattern::Unknown { .. } => "Unknown".into(),
    }
}

fn resolve_precision(opts: &BackendOptions, preferred: Precision) -> Precision {
    match opts.precision {
        nxpu_backend_core::PrecisionPolicy::Explicit(p) => p,
        nxpu_backend_core::PrecisionPolicy::Auto => preferred,
        nxpu_backend_core::PrecisionPolicy::Keep => Precision::F32,
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
        assert_eq!(backend.preferred_precision(), Precision::F16);
    }

    #[test]
    fn compile_matmul_emits_xml_and_onnx() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/matmul.wgsl"
        ))
        .unwrap();
        let module = nxpu_parser::parse(&source).unwrap();

        let output = IntelBackend
            .compile(&module, &BackendOptions::default())
            .unwrap();

        // Should have model.xml, model.bin, and output.onnx
        let names: Vec<&str> = output.files.iter().map(|f| f.name.as_str()).collect();
        assert!(names.contains(&"model.xml"));
        assert!(names.contains(&"model.bin"));
        assert!(names.contains(&"output.onnx"));

        // Verify XML contains expected elements
        let xml_file = output.files.iter().find(|f| f.name == "model.xml").unwrap();
        let xml = match &xml_file.content {
            OutputContent::Text(t) => t,
            _ => panic!("expected text"),
        };
        assert!(xml.contains("<net"));
        assert!(xml.contains("version=\"11\""));
        assert!(xml.contains("type=\"MatMul\""));

        // model.bin should be empty binary placeholder
        let bin_file = output.files.iter().find(|f| f.name == "model.bin").unwrap();
        assert!(matches!(&bin_file.content, OutputContent::Binary(b) if b.is_empty()));
    }

    #[test]
    fn compile_vecadd_emits_ir() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/vecadd.wgsl"
        ))
        .unwrap();
        let module = nxpu_parser::parse(&source).unwrap();

        let output = IntelBackend
            .compile(&module, &BackendOptions::default())
            .unwrap();
        let xml_file = output.files.iter().find(|f| f.name == "model.xml").unwrap();
        let xml = match &xml_file.content {
            OutputContent::Text(t) => t,
            _ => panic!("expected text"),
        };
        assert!(xml.contains("type=\"Add\""));
    }

    #[test]
    fn diagnostics_include_openvino_hint() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/matmul.wgsl"
        ))
        .unwrap();
        let module = nxpu_parser::parse(&source).unwrap();

        let output = IntelBackend
            .compile(&module, &BackendOptions::default())
            .unwrap();
        let messages: Vec<&str> = output
            .diagnostics
            .iter()
            .map(|d| d.message.as_str())
            .collect();
        assert!(messages.iter().any(|m| m.contains("read_model")));
    }

    fn load_and_compile(example: &str, opts: &BackendOptions) -> BackendOutput {
        let source = std::fs::read_to_string(format!(
            "{}/../../examples/{example}.wgsl",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let module = nxpu_parser::parse(&source).unwrap();
        IntelBackend.compile(&module, opts).unwrap()
    }

    #[test]
    fn compile_conv2d_emits_convolution() {
        let output = load_and_compile("conv2d", &BackendOptions::default());
        let xml_file = output.files.iter().find(|f| f.name == "model.xml").unwrap();
        let xml = match &xml_file.content {
            OutputContent::Text(t) => t,
            _ => panic!("expected text"),
        };
        assert!(xml.contains("type=\"Convolution\""));
    }

    #[test]
    fn compile_relu_emits_relu_layer() {
        let output = load_and_compile("relu", &BackendOptions::default());
        let xml_file = output.files.iter().find(|f| f.name == "model.xml").unwrap();
        let xml = match &xml_file.content {
            OutputContent::Text(t) => t,
            _ => panic!("expected text"),
        };
        assert!(xml.contains("type=\"ReLU\""));
    }

    #[test]
    fn compile_attention() {
        let output = load_and_compile("attention", &BackendOptions::default());
        assert!(!output.files.is_empty());
    }

    #[test]
    fn compile_maxpool() {
        let output = load_and_compile("maxpool", &BackendOptions::default());
        assert!(!output.files.is_empty());
    }

    #[test]
    fn compile_reduce_sum() {
        let output = load_and_compile("reduce_sum", &BackendOptions::default());
        assert!(!output.files.is_empty());
    }

    #[test]
    fn compile_batchnorm() {
        let output = load_and_compile("batchnorm", &BackendOptions::default());
        assert!(!output.files.is_empty());
    }

    #[test]
    fn resolve_precision_explicit_and_keep() {
        use nxpu_backend_core::PrecisionPolicy;

        // Explicit(F16) with preferred Int8 => F16
        let opts_explicit = BackendOptions {
            precision: PrecisionPolicy::Explicit(Precision::F16),
            ..Default::default()
        };
        assert_eq!(
            resolve_precision(&opts_explicit, Precision::Int8),
            Precision::F16
        );

        // Keep with preferred Int8 => F32
        let opts_keep = BackendOptions {
            precision: PrecisionPolicy::Keep,
            ..Default::default()
        };
        assert_eq!(
            resolve_precision(&opts_keep, Precision::Int8),
            Precision::F32
        );
    }
}
