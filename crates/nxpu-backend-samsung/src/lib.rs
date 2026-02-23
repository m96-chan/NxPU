//! Samsung Exynos NPU backend for NxPU.
//!
//! Delegates compilation to the ONNX backend with Samsung Exynos NPU-specific
//! validation and ONE toolchain compilation hints.

use nxpu_analysis::analyze;
use nxpu_backend_core::{
    Backend, BackendError, BackendOptions, BackendOutput, Diagnostic, DiagnosticLevel, Precision,
    PrecisionPolicy, validate_patterns,
};
use nxpu_backend_onnx::OnnxBackend;
use nxpu_ir::Module;

mod support;

use support::SamsungNpuSupport;

/// Samsung Exynos NPU backend with ONE toolchain hints.
#[derive(Debug)]
pub struct SamsungBackend;

impl Backend for SamsungBackend {
    fn name(&self) -> &str {
        "Samsung Exynos NPU"
    }

    fn targets(&self) -> &[&str] {
        &["samsung", "exynos"]
    }

    fn preferred_precision(&self) -> Precision {
        Precision::F16
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
        let mut diagnostics = validate_patterns(&SamsungNpuSupport, &op_refs, precision);

        let mut output = OnnxBackend.compile(module, opts)?;
        diagnostics.extend(output.diagnostics);

        diagnostics.push(Diagnostic {
            level: DiagnosticLevel::Info,
            message: "To import for Exynos NPU: one-import-onnx -i output.onnx -o model.circle"
                .into(),
        });
        diagnostics.push(Diagnostic {
            level: DiagnosticLevel::Info,
            message: "To compile: one-codegen -b npu model.circle -o model.tvn".into(),
        });
        diagnostics.push(Diagnostic {
            level: DiagnosticLevel::Info,
            message:
                "For quantization: one-quantize -i model.circle -o model_q.circle --input_dtype float32 --quantized_dtype uint8"
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
        let backend = SamsungBackend;
        assert_eq!(backend.name(), "Samsung Exynos NPU");
        assert!(backend.targets().contains(&"samsung"));
        assert!(backend.targets().contains(&"exynos"));
        assert_eq!(backend.preferred_precision(), Precision::F16);
    }

    #[test]
    fn compile_matmul_with_one_hints() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/matmul.wgsl"
        ))
        .unwrap();
        let module = nxpu_parser::parse(&source).unwrap();

        let output = SamsungBackend
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
        assert!(messages.iter().any(|m| m.contains("one-import-onnx")));
        assert!(messages.iter().any(|m| m.contains("one-codegen")));
    }
}
