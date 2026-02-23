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
}
