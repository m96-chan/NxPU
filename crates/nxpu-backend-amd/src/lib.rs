//! AMD XDNA NPU backend for NxPU.
//!
//! Delegates compilation to the ONNX backend with AMD XDNA-specific metadata
//! properties (target device, execution provider, quantization scheme).
//! Validates operator patterns against the XDNA support matrix and emits
//! Vitis AI conversion hints.

use nxpu_analysis::analyze;
use nxpu_backend_core::{
    Backend, BackendError, BackendOptions, BackendOutput, Diagnostic, DiagnosticLevel,
    OutputContent, Precision, PrecisionPolicy, validate_patterns,
};
use nxpu_backend_onnx::OnnxBackend;
use nxpu_backend_onnx::proto::{ModelProto, StringStringEntryProto};
use nxpu_ir::Module;
use prost::Message;

mod support;

use support::AmdXdnaSupport;

/// AMD XDNA NPU backend with enhanced ONNX metadata.
#[derive(Debug)]
pub struct AmdBackend;

impl Backend for AmdBackend {
    fn name(&self) -> &str {
        "AMD XDNA NPU"
    }

    fn targets(&self) -> &[&str] {
        &["amd-xdna", "amd-npu"]
    }

    fn preferred_precision(&self) -> Precision {
        Precision::Int8
    }

    fn compile(
        &self,
        module: &Module,
        opts: &BackendOptions,
    ) -> Result<BackendOutput, BackendError> {
        // 1. Classify entry points for validation.
        let mut op_names = Vec::new();
        for (i, ep) in module.entry_points.iter().enumerate() {
            match analyze::classify_entry_point(module, i) {
                Ok(pattern) => {
                    op_names.push(pattern_op_name(&pattern));
                }
                Err(e) => {
                    return Err(BackendError::Unsupported(format!(
                        "entry point '{}': {e}",
                        ep.name
                    )));
                }
            }
        }

        // 2. Validate against AMD XDNA support matrix.
        let precision = resolve_precision(opts, self.preferred_precision());
        let op_refs: Vec<&str> = op_names.iter().map(|s| s.as_str()).collect();
        let mut diagnostics = validate_patterns(&AmdXdnaSupport, &op_refs, precision);

        // 3. Generate ONNX model via base backend.
        let mut output = OnnxBackend.compile(module, opts)?;

        // 4. Inject XDNA metadata_props into each ONNX model file.
        let metadata_props = vec![
            StringStringEntryProto {
                key: "xdna:target_device".into(),
                value: "AMD Ryzen AI".into(),
            },
            StringStringEntryProto {
                key: "xdna:execution_provider".into(),
                value: "VitisAIExecutionProvider".into(),
            },
            StringStringEntryProto {
                key: "xdna:quantization".into(),
                value: format!("{precision}"),
            },
        ];

        for file in &mut output.files {
            let is_onnx = std::path::Path::new(&file.name)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("onnx"));
            if !is_onnx {
                continue;
            }
            let OutputContent::Binary(bytes) = &file.content else {
                continue;
            };
            let Ok(mut model) = ModelProto::decode(bytes.as_slice()) else {
                continue;
            };
            model.metadata_props.extend(metadata_props.clone());
            file.content = OutputContent::Binary(model.encode_to_vec());
        }

        diagnostics.extend(output.diagnostics);
        diagnostics.push(Diagnostic {
            level: DiagnosticLevel::Info,
            message: "To compile for XDNA: use Vitis AI EP with ONNX Runtime".into(),
        });
        diagnostics.push(Diagnostic {
            level: DiagnosticLevel::Info,
            message:
                "Vitis AI quantization: vai_q_onnx quantize_static --model output.onnx --calibration_data_reader <reader>"
                    .into(),
        });

        output.diagnostics = diagnostics;
        Ok(output)
    }
}

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
        PrecisionPolicy::Explicit(p) => p,
        PrecisionPolicy::Auto => preferred,
        PrecisionPolicy::Keep => Precision::F32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nxpu_backend_core::{BackendOptions, OutputContent};

    #[test]
    fn backend_metadata() {
        let backend = AmdBackend;
        assert_eq!(backend.name(), "AMD XDNA NPU");
        assert!(backend.targets().contains(&"amd-xdna"));
        assert!(backend.targets().contains(&"amd-npu"));
        assert_eq!(backend.preferred_precision(), Precision::Int8);
    }

    #[test]
    fn compile_matmul_has_metadata_props() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/matmul.wgsl"
        ))
        .unwrap();
        let module = nxpu_parser::parse(&source).unwrap();

        let output = AmdBackend
            .compile(&module, &BackendOptions::default())
            .unwrap();
        assert_eq!(output.files.len(), 1);
        assert_eq!(output.files[0].name, "output.onnx");

        // Decode and verify metadata_props
        let bytes = match &output.files[0].content {
            OutputContent::Binary(b) => b,
            _ => panic!("expected binary"),
        };
        let model = ModelProto::decode(bytes.as_slice()).unwrap();
        assert!(!model.metadata_props.is_empty());

        let keys: Vec<&str> = model
            .metadata_props
            .iter()
            .map(|p| p.key.as_str())
            .collect();
        assert!(keys.contains(&"xdna:target_device"));
        assert!(keys.contains(&"xdna:execution_provider"));
        assert!(keys.contains(&"xdna:quantization"));
    }

    #[test]
    fn diagnostics_include_vitis_hint() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/matmul.wgsl"
        ))
        .unwrap();
        let module = nxpu_parser::parse(&source).unwrap();

        let output = AmdBackend
            .compile(&module, &BackendOptions::default())
            .unwrap();
        let messages: Vec<&str> = output
            .diagnostics
            .iter()
            .map(|d| d.message.as_str())
            .collect();
        assert!(messages.iter().any(|m| m.contains("Vitis AI")));
    }

    fn load_and_compile(example: &str, opts: &BackendOptions) -> BackendOutput {
        let source = std::fs::read_to_string(format!(
            "{}/../../examples/{example}.wgsl",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let module = nxpu_parser::parse(&source).unwrap();
        AmdBackend.compile(&module, opts).unwrap()
    }

    #[test]
    fn compile_conv2d() {
        let output = load_and_compile("conv2d", &BackendOptions::default());
        assert!(!output.files.is_empty());
        for file in &output.files {
            match &file.content {
                OutputContent::Binary(b) => assert!(!b.is_empty()),
                OutputContent::Text(t) => assert!(!t.is_empty()),
            }
        }
    }

    #[test]
    fn compile_relu() {
        let output = load_and_compile("relu", &BackendOptions::default());
        assert!(!output.files.is_empty());
        for file in &output.files {
            match &file.content {
                OutputContent::Binary(b) => assert!(!b.is_empty()),
                OutputContent::Text(t) => assert!(!t.is_empty()),
            }
        }
    }

    #[test]
    fn compile_attention() {
        let output = load_and_compile("attention", &BackendOptions::default());
        assert!(!output.files.is_empty());
        for file in &output.files {
            match &file.content {
                OutputContent::Binary(b) => assert!(!b.is_empty()),
                OutputContent::Text(t) => assert!(!t.is_empty()),
            }
        }
    }

    #[test]
    fn resolve_precision_explicit_and_keep() {
        let explicit_opts = BackendOptions {
            precision: PrecisionPolicy::Explicit(Precision::F16),
            ..BackendOptions::default()
        };
        assert_eq!(
            resolve_precision(&explicit_opts, Precision::Int8),
            Precision::F16
        );

        let keep_opts = BackendOptions {
            precision: PrecisionPolicy::Keep,
            ..BackendOptions::default()
        };
        assert_eq!(
            resolve_precision(&keep_opts, Precision::Int8),
            Precision::F32
        );
    }

    #[test]
    fn metadata_reflects_explicit_precision() {
        let opts = BackendOptions {
            precision: PrecisionPolicy::Explicit(Precision::F16),
            ..BackendOptions::default()
        };
        let output = load_and_compile("matmul", &opts);

        let bytes = match &output.files[0].content {
            OutputContent::Binary(b) => b,
            _ => panic!("expected binary"),
        };
        let model = ModelProto::decode(bytes.as_slice()).unwrap();

        let quant_prop = model
            .metadata_props
            .iter()
            .find(|p| p.key == "xdna:quantization")
            .expect("missing xdna:quantization metadata prop");
        assert!(
            quant_prop.value.to_lowercase().contains("f16"),
            "expected quantization value to contain 'f16', got: {}",
            quant_prop.value
        );
    }
}
