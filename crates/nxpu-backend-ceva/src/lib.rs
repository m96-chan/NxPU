//! CEVA NeuPro NPU backend for NxPU.
//!
//! Delegates compilation to the ONNX backend with CEVA NeuPro-specific
//! validation and CDNN compiler hints.

use nxpu_analysis::analyze;
use nxpu_backend_core::{
    Backend, BackendError, BackendOptions, BackendOutput, Diagnostic, DiagnosticLevel, Precision,
    PrecisionPolicy, validate_patterns,
};
use nxpu_backend_onnx::OnnxBackend;
use nxpu_ir::Module;

mod support;

use support::CevaNeuProSupport;

/// CEVA NeuPro NPU backend with CDNN compiler hints.
#[derive(Debug)]
pub struct CevaBackend;

impl Backend for CevaBackend {
    fn name(&self) -> &str {
        "CEVA NeuPro NPU"
    }

    fn targets(&self) -> &[&str] {
        &["ceva", "neupro"]
    }

    fn preferred_precision(&self) -> Precision {
        Precision::Int8
    }

    fn compile(
        &self,
        module: &Module,
        opts: &BackendOptions,
    ) -> Result<BackendOutput, BackendError> {
        let mut op_names = Vec::new();
        for (i, ep) in module.entry_points.iter().enumerate() {
            match analyze::classify_entry_point(module, i) {
                Ok(pattern) => op_names.push(pattern_op_name(&pattern)),
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
        let mut diagnostics = validate_patterns(&CevaNeuProSupport, &op_refs, precision);

        let mut output = OnnxBackend.compile(module, opts)?;
        diagnostics.extend(output.diagnostics);

        diagnostics.push(Diagnostic {
            level: DiagnosticLevel::Info,
            message: "To compile for NeuPro: import output.onnx in CDNN compiler".into(),
        });
        diagnostics.push(Diagnostic {
            level: DiagnosticLevel::Info,
            message: "CDNN command: cdnn_cli --model output.onnx --target neupro-s".into(),
        });
        diagnostics.push(Diagnostic {
            level: DiagnosticLevel::Info,
            message:
                "Note: CEVA NeuPro has a limited operator set. Unsupported ops will fall back to CPU."
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
        let backend = CevaBackend;
        assert_eq!(backend.name(), "CEVA NeuPro NPU");
        assert!(backend.targets().contains(&"ceva"));
        assert!(backend.targets().contains(&"neupro"));
        assert_eq!(backend.preferred_precision(), Precision::Int8);
    }

    #[test]
    fn compile_matmul_with_cdnn_hints() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/matmul.wgsl"
        ))
        .unwrap();
        let module = nxpu_parser::parse(&source).unwrap();

        let output = CevaBackend
            .compile(&module, &BackendOptions::default())
            .unwrap();
        assert_eq!(output.files.len(), 1);
        assert_eq!(output.files[0].name, "output.onnx");
        assert!(matches!(output.files[0].content, OutputContent::Binary(_)));

        let messages: Vec<&str> = output
            .diagnostics
            .iter()
            .map(|d| d.message.as_str())
            .collect();
        assert!(messages.iter().any(|m| m.contains("CDNN")));
    }

    #[test]
    fn compile_matmul_validates_support() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/matmul.wgsl"
        ))
        .unwrap();
        let module = nxpu_parser::parse(&source).unwrap();

        let output = CevaBackend
            .compile(&module, &BackendOptions::default())
            .unwrap();

        // MatMul is emulated on CEVA, so we should get a warning
        let has_emulated_warning = output
            .diagnostics
            .iter()
            .any(|d| d.message.contains("emulated"));
        assert!(has_emulated_warning);
    }
}
