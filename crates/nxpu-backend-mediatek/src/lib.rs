//! MediaTek APU backend for NxPU.
//!
//! Delegates compilation to the TFLite backend with MediaTek APU-specific
//! validation and NeuroPilot compilation hints.

use nxpu_analysis::analyze;
use nxpu_backend_core::{
    Backend, BackendError, BackendOptions, BackendOutput, Diagnostic, DiagnosticLevel, Precision,
    PrecisionPolicy, validate_patterns,
};
use nxpu_backend_tflite::TfLiteBackend;
use nxpu_ir::Module;

mod support;

use support::MediaTekApuSupport;

/// MediaTek APU backend with NeuroPilot compilation hints.
#[derive(Debug)]
pub struct MediaTekBackend;

impl Backend for MediaTekBackend {
    fn name(&self) -> &str {
        "MediaTek APU"
    }

    fn targets(&self) -> &[&str] {
        &["mediatek", "neuropilot"]
    }

    fn preferred_precision(&self) -> Precision {
        Precision::Int8
    }

    fn compile(
        &self,
        module: &Module,
        opts: &BackendOptions,
    ) -> Result<BackendOutput, BackendError> {
        // 1. Classify entry points and validate.
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

        let precision = resolve_precision(opts, self.preferred_precision());
        let op_refs: Vec<&str> = op_names.iter().map(|s| s.as_str()).collect();
        let mut diagnostics = validate_patterns(&MediaTekApuSupport, &op_refs, precision);

        // 2. Generate TFLite model.
        let mut output = TfLiteBackend.compile(module, opts)?;

        diagnostics.extend(output.diagnostics);

        // 3. NeuroPilot compilation hints.
        for file in &output.files {
            diagnostics.push(Diagnostic {
                level: DiagnosticLevel::Info,
                message: format!("To compile for MediaTek APU: ncc-tflite {}", file.name),
            });
        }
        diagnostics.push(Diagnostic {
            level: DiagnosticLevel::Info,
            message: "NeuroPilot SDK: ncc-tflite --arch mdla3.0 output.tflite -o output.dla".into(),
        });
        diagnostics.push(Diagnostic {
            level: DiagnosticLevel::Info,
            message: "For quantization: use TFLite quantization-aware training or \
                      post-training quantization"
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
        let backend = MediaTekBackend;
        assert_eq!(backend.name(), "MediaTek APU");
        assert!(backend.targets().contains(&"mediatek"));
        assert!(backend.targets().contains(&"neuropilot"));
        assert_eq!(backend.preferred_precision(), Precision::Int8);
    }

    #[test]
    fn compile_matmul_with_neuropilot_hints() {
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

        let messages: Vec<&str> = output
            .diagnostics
            .iter()
            .map(|d| d.message.as_str())
            .collect();
        assert!(messages.iter().any(|m| m.contains("ncc-tflite")));
    }

    fn load_and_compile(example: &str, opts: &BackendOptions) -> BackendOutput {
        let source = std::fs::read_to_string(format!(
            "{}/../../examples/{example}.wgsl",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let module = nxpu_parser::parse(&source).unwrap();
        MediaTekBackend.compile(&module, opts).unwrap()
    }

    #[test]
    fn compile_conv2d() {
        let output = load_and_compile("conv2d", &BackendOptions::default());
        assert!(!output.files.is_empty());
        let has_tflite = output.files.iter().any(|f| f.name.ends_with(".tflite"));
        assert!(has_tflite);
    }

    #[test]
    fn compile_relu() {
        let output = load_and_compile("relu", &BackendOptions::default());
        assert!(!output.files.is_empty());
    }

    #[test]
    fn compile_attention() {
        let output = load_and_compile("attention", &BackendOptions::default());
        assert!(!output.files.is_empty());
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
    fn neuropilot_arch_diagnostic() {
        let output = load_and_compile("matmul", &BackendOptions::default());
        let messages: Vec<&str> = output
            .diagnostics
            .iter()
            .map(|d| d.message.as_str())
            .collect();
        assert!(messages.iter().any(|m| m.contains("mdla3.0")));
    }
}
