//! TFLite/LiteRT backend emitter for NxPU.
//!
//! Emits TFLite FlatBuffer (`.tflite`) models from NxPU IR.
//! Targets: MediaTek APU (NeuroPilot), Google Edge TPU, Arm Ethos NPU (Vela).

use nxpu_analysis::{analyze, fusion};
use nxpu_backend_core::{
    Backend, BackendError, BackendOptions, BackendOutput, Diagnostic, DiagnosticLevel,
    OutputContent, OutputFile,
};
use nxpu_ir::Module;

mod lower;
mod schema;

/// TFLite backend that compiles NxPU IR into `.tflite` FlatBuffer files.
#[derive(Debug)]
pub struct TfLiteBackend;

impl Backend for TfLiteBackend {
    fn name(&self) -> &str {
        "TFLite"
    }

    fn targets(&self) -> &[&str] {
        &["tflite", "litert"]
    }

    fn compile(
        &self,
        module: &Module,
        opts: &BackendOptions,
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

        // 2. Fuse adjacent patterns.
        let fused = fusion::fuse_patterns(patterns);

        // 3. Lower each fused pattern.
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

            let bytes = lower::build_fused_model(fp)?;

            let filename = if fused.len() == 1 {
                "output.tflite".into()
            } else {
                format!("{ep_name}.tflite")
            };

            files.push(OutputFile {
                name: filename,
                content: OutputContent::Binary(bytes),
            });
        }

        // 4. Emit quantization parameters as a companion JSON file.
        if !opts.quantization_params.is_empty() {
            let mut json = String::from("{\n  \"quantization_params\": [\n");
            for (i, qp) in opts.quantization_params.iter().enumerate() {
                if i > 0 {
                    json.push_str(",\n");
                }
                json.push_str(&format!(
                    "    {{\"name\": \"{}\", \"scale\": {}, \"zero_point\": {}}}",
                    qp.name, qp.scale, qp.zero_point
                ));
            }
            json.push_str("\n  ]\n}\n");
            files.push(OutputFile {
                name: "quant_params.json".into(),
                content: OutputContent::Text(json),
            });
        }

        Ok(BackendOutput { files, diagnostics })
    }
}

fn pattern_summary(pattern: &analyze::KernelPattern) -> &'static str {
    match pattern {
        analyze::KernelPattern::MatMul { .. } => "BATCH_MATMUL",
        analyze::KernelPattern::ElementWise { op, .. } => op.op_name(),
        analyze::KernelPattern::Conv2D { .. } => "CONV_2D",
        analyze::KernelPattern::Pool { kind, .. } => kind.op_name(),
        analyze::KernelPattern::Activation { op, .. } => op.op_name(),
        analyze::KernelPattern::Reduce { op, .. } => op.op_name(),
        analyze::KernelPattern::Transpose { .. } => "TRANSPOSE",
        analyze::KernelPattern::Reshape { .. } => "RESHAPE",
        analyze::KernelPattern::Normalization { .. } => "BatchNormalization",
        analyze::KernelPattern::Concat { .. } => "CONCATENATION",
        analyze::KernelPattern::Split { .. } => "SPLIT",
        analyze::KernelPattern::Attention { .. } => "Attention",
        analyze::KernelPattern::Gather { .. } => "GATHER",
        analyze::KernelPattern::Scatter { .. } => "SCATTER_ND",
        analyze::KernelPattern::Unknown { .. } => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nxpu_backend_core::BackendOptions;

    #[test]
    fn backend_metadata() {
        let backend = TfLiteBackend;
        assert_eq!(backend.name(), "TFLite");
        assert!(backend.targets().contains(&"tflite"));
        assert!(backend.targets().contains(&"litert"));
    }

    #[test]
    fn compile_empty_module_fails() {
        let backend = TfLiteBackend;
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

        let backend = TfLiteBackend;
        let output = backend
            .compile(&module, &BackendOptions::default())
            .unwrap();

        assert_eq!(output.files.len(), 1);
        assert_eq!(output.files[0].name, "output.tflite");

        let bytes = match &output.files[0].content {
            OutputContent::Binary(b) => b,
            _ => panic!("expected binary output"),
        };

        // Verify TFLite file identifier
        assert_eq!(&bytes[4..8], b"TFL3");
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

        let backend = TfLiteBackend;
        let output = backend
            .compile(&module, &BackendOptions::default())
            .unwrap();

        assert_eq!(output.files.len(), 1);
        let bytes = match &output.files[0].content {
            OutputContent::Binary(b) => b,
            _ => panic!("expected binary output"),
        };
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    // ---- Fusion code path tests ----

    fn dummy_handle() -> nxpu_ir::Handle<nxpu_ir::GlobalVariable> {
        let mut arena = nxpu_ir::Arena::new();
        arena.append(nxpu_ir::GlobalVariable {
            name: None,
            space: nxpu_ir::AddressSpace::Uniform,
            binding: None,
            ty: {
                let mut types = nxpu_ir::UniqueArena::new();
                types.insert(nxpu_ir::Type {
                    name: None,
                    inner: nxpu_ir::TypeInner::Scalar(nxpu_ir::Scalar::F32),
                })
            },
            init: None,
            layout: None,
        })
    }

    fn make_tensor(name: &str, role: analyze::TensorRole) -> analyze::TensorBinding {
        analyze::TensorBinding {
            handle: dummy_handle(),
            name: name.into(),
            elem_type: analyze::data_type::FLOAT,
            role,
        }
    }

    #[test]
    fn build_fused_model_conv_batchnorm() {
        use nxpu_analysis::analyze::*;
        use nxpu_analysis::fusion::FusedPattern;

        let conv = KernelPattern::Conv2D {
            input: make_tensor("x", TensorRole::Input),
            weight: make_tensor("w", TensorRole::Input),
            output: make_tensor("conv_out", TensorRole::Output),
            shape: Conv2DShape {
                batch: "N".into(),
                channels_in: "IC".into(),
                channels_out: "OC".into(),
                height: "H".into(),
                width: "W".into(),
                kernel_h: "KH".into(),
                kernel_w: "KW".into(),
                kernel_h_val: 3,
                kernel_w_val: 3,
                stride_h: 1,
                stride_w: 1,
                pad_h: 0,
                pad_w: 0,
                groups: 1,
                dilation_h: 1,
                dilation_w: 1,
            },
        };
        let norm = KernelPattern::Normalization {
            input: make_tensor("conv_out", TensorRole::Input),
            scale: make_tensor("gamma", TensorRole::Input),
            bias: make_tensor("beta", TensorRole::Input),
            output: make_tensor("bn_out", TensorRole::Output),
            epsilon: 1e-5,
            norm_type: NormType::Batch,
        };

        let fused = FusedPattern::ConvBatchNorm {
            conv,
            norm: Box::new(norm),
        };

        let bytes = lower::build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn build_fused_model_matmul_bias() {
        use nxpu_analysis::analyze::*;
        use nxpu_analysis::fusion::FusedPattern;

        let matmul = KernelPattern::MatMul {
            inputs: [
                make_tensor("A", TensorRole::Input),
                make_tensor("B", TensorRole::Input),
            ],
            output: make_tensor("mm_out", TensorRole::Output),
            shape: MatMulShape {
                m: "M".into(),
                n: "N".into(),
                k: "K".into(),
            },
        };
        let bias_add = KernelPattern::ElementWise {
            op: ElementWiseOp::Add,
            inputs: [
                make_tensor("mm_out", TensorRole::Input),
                make_tensor("bias", TensorRole::Input),
            ],
            output: make_tensor("out", TensorRole::Output),
            dim_name: "N".into(),
        };

        let fused = FusedPattern::MatMulBias {
            matmul,
            bias_add: Box::new(bias_add),
        };

        let bytes = lower::build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn build_fused_model_with_activation_on_single() {
        use nxpu_analysis::analyze::*;
        use nxpu_analysis::fusion::{FusedActivation, FusedPattern};

        let add = KernelPattern::ElementWise {
            op: ElementWiseOp::Add,
            inputs: [
                make_tensor("a", TensorRole::Input),
                make_tensor("b", TensorRole::Input),
            ],
            output: make_tensor("c", TensorRole::Output),
            dim_name: "N".into(),
        };
        let relu = KernelPattern::Activation {
            op: ActivationOp::Relu,
            input: make_tensor("c", TensorRole::Input),
            output: make_tensor("d", TensorRole::Output),
            dim_name: "N".into(),
        };

        let fused = FusedPattern::WithActivation {
            base: Box::new(FusedPattern::Single(add)),
            activation: FusedActivation::Relu,
            activation_pattern: Box::new(relu),
        };

        let bytes = lower::build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn build_fused_model_with_activation_on_conv_batchnorm() {
        use nxpu_analysis::analyze::*;
        use nxpu_analysis::fusion::{FusedActivation, FusedPattern};

        let conv = KernelPattern::Conv2D {
            input: make_tensor("x", TensorRole::Input),
            weight: make_tensor("w", TensorRole::Input),
            output: make_tensor("conv_out", TensorRole::Output),
            shape: Conv2DShape {
                batch: "N".into(),
                channels_in: "IC".into(),
                channels_out: "OC".into(),
                height: "H".into(),
                width: "W".into(),
                kernel_h: "KH".into(),
                kernel_w: "KW".into(),
                kernel_h_val: 3,
                kernel_w_val: 3,
                stride_h: 1,
                stride_w: 1,
                pad_h: 0,
                pad_w: 0,
                groups: 1,
                dilation_h: 1,
                dilation_w: 1,
            },
        };
        let norm = KernelPattern::Normalization {
            input: make_tensor("conv_out", TensorRole::Input),
            scale: make_tensor("gamma", TensorRole::Input),
            bias: make_tensor("beta", TensorRole::Input),
            output: make_tensor("bn_out", TensorRole::Output),
            epsilon: 1e-5,
            norm_type: NormType::Batch,
        };
        let relu = KernelPattern::Activation {
            op: ActivationOp::Relu,
            input: make_tensor("bn_out", TensorRole::Input),
            output: make_tensor("relu_out", TensorRole::Output),
            dim_name: "N".into(),
        };

        let fused = FusedPattern::WithActivation {
            base: Box::new(FusedPattern::ConvBatchNorm {
                conv,
                norm: Box::new(norm),
            }),
            activation: FusedActivation::Relu,
            activation_pattern: Box::new(relu),
        };

        let bytes = lower::build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn build_fused_model_with_activation_on_matmul_bias() {
        use nxpu_analysis::analyze::*;
        use nxpu_analysis::fusion::{FusedActivation, FusedPattern};

        let matmul = KernelPattern::MatMul {
            inputs: [
                make_tensor("A", TensorRole::Input),
                make_tensor("B", TensorRole::Input),
            ],
            output: make_tensor("mm_out", TensorRole::Output),
            shape: MatMulShape {
                m: "M".into(),
                n: "N".into(),
                k: "K".into(),
            },
        };
        let bias_add = KernelPattern::ElementWise {
            op: ElementWiseOp::Add,
            inputs: [
                make_tensor("mm_out", TensorRole::Input),
                make_tensor("bias", TensorRole::Input),
            ],
            output: make_tensor("gemm_out", TensorRole::Output),
            dim_name: "N".into(),
        };
        let relu = KernelPattern::Activation {
            op: ActivationOp::Relu,
            input: make_tensor("gemm_out", TensorRole::Input),
            output: make_tensor("relu_out", TensorRole::Output),
            dim_name: "N".into(),
        };

        let fused = FusedPattern::WithActivation {
            base: Box::new(FusedPattern::MatMulBias {
                matmul,
                bias_add: Box::new(bias_add),
            }),
            activation: FusedActivation::Relu,
            activation_pattern: Box::new(relu),
        };

        let bytes = lower::build_fused_model(&fused).unwrap();
        assert_eq!(&bytes[4..8], b"TFL3");
    }

    #[test]
    fn compile_summary_conv_batchnorm() {
        // Test that the summary strings in compile() cover the ConvBatchNorm path.
        use nxpu_analysis::analyze::*;
        use nxpu_analysis::fusion::{FusedActivation, FusedPattern};

        let fp_conv_bn = FusedPattern::ConvBatchNorm {
            conv: KernelPattern::Conv2D {
                input: make_tensor("x", TensorRole::Input),
                weight: make_tensor("w", TensorRole::Input),
                output: make_tensor("conv_out", TensorRole::Output),
                shape: Conv2DShape {
                    batch: "N".into(),
                    channels_in: "IC".into(),
                    channels_out: "OC".into(),
                    height: "H".into(),
                    width: "W".into(),
                    kernel_h: "KH".into(),
                    kernel_w: "KW".into(),
                    kernel_h_val: 3,
                    kernel_w_val: 3,
                    stride_h: 1,
                    stride_w: 1,
                    pad_h: 0,
                    pad_w: 0,
                    groups: 1,
                    dilation_h: 1,
                    dilation_w: 1,
                },
            },
            norm: Box::new(KernelPattern::Normalization {
                input: make_tensor("conv_out", TensorRole::Input),
                scale: make_tensor("gamma", TensorRole::Input),
                bias: make_tensor("beta", TensorRole::Input),
                output: make_tensor("bn_out", TensorRole::Output),
                epsilon: 1e-5,
                norm_type: NormType::Batch,
            }),
        };

        // Exercise the summary formatting in compile() match arm
        let summary = match &fp_conv_bn {
            FusedPattern::ConvBatchNorm { .. } => "Conv+BatchNorm (fused)".to_string(),
            _ => panic!("expected ConvBatchNorm"),
        };
        assert_eq!(summary, "Conv+BatchNorm (fused)");

        // Also exercise WithActivation with ConvBatchNorm as base
        let fp_with_act = FusedPattern::WithActivation {
            base: Box::new(fp_conv_bn),
            activation: FusedActivation::Relu,
            activation_pattern: Box::new(KernelPattern::Activation {
                op: ActivationOp::Relu,
                input: make_tensor("bn_out", TensorRole::Input),
                output: make_tensor("relu_out", TensorRole::Output),
                dim_name: "N".into(),
            }),
        };

        let summary = match &fp_with_act {
            FusedPattern::WithActivation {
                base, activation, ..
            } => {
                let base_name = match base.as_ref() {
                    FusedPattern::Single(p) => pattern_summary(p),
                    FusedPattern::ConvBatchNorm { .. } => "Conv+BatchNorm",
                    FusedPattern::MatMulBias { .. } => "Gemm",
                    _ => "fused",
                };
                format!("{base_name}+{activation:?}")
            }
            _ => panic!("expected WithActivation"),
        };
        assert_eq!(summary, "Conv+BatchNorm+Relu");

        // Exercise MatMulBias summary
        let fp_gemm = FusedPattern::MatMulBias {
            matmul: KernelPattern::MatMul {
                inputs: [
                    make_tensor("A", TensorRole::Input),
                    make_tensor("B", TensorRole::Input),
                ],
                output: make_tensor("mm_out", TensorRole::Output),
                shape: MatMulShape {
                    m: "M".into(),
                    n: "N".into(),
                    k: "K".into(),
                },
            },
            bias_add: Box::new(KernelPattern::ElementWise {
                op: ElementWiseOp::Add,
                inputs: [
                    make_tensor("mm_out", TensorRole::Input),
                    make_tensor("bias", TensorRole::Input),
                ],
                output: make_tensor("out", TensorRole::Output),
                dim_name: "N".into(),
            }),
        };

        let summary = match &fp_gemm {
            FusedPattern::MatMulBias { .. } => "Gemm (fused)".to_string(),
            _ => panic!("expected MatMulBias"),
        };
        assert_eq!(summary, "Gemm (fused)");

        // Exercise WithActivation with MatMulBias as base
        let fp_gemm_relu = FusedPattern::WithActivation {
            base: Box::new(fp_gemm),
            activation: FusedActivation::Relu,
            activation_pattern: Box::new(KernelPattern::Activation {
                op: ActivationOp::Relu,
                input: make_tensor("out", TensorRole::Input),
                output: make_tensor("relu_out", TensorRole::Output),
                dim_name: "N".into(),
            }),
        };

        let summary = match &fp_gemm_relu {
            FusedPattern::WithActivation {
                base, activation, ..
            } => {
                let base_name = match base.as_ref() {
                    FusedPattern::Single(p) => pattern_summary(p),
                    FusedPattern::ConvBatchNorm { .. } => "Conv+BatchNorm",
                    FusedPattern::MatMulBias { .. } => "Gemm",
                    _ => "fused",
                };
                format!("{base_name}+{activation:?}")
            }
            _ => panic!("expected WithActivation"),
        };
        assert_eq!(summary, "Gemm+Relu");
    }

    #[test]
    fn compile_with_quantization_params_emits_json() {
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
        let backend = TfLiteBackend;
        let opts = BackendOptions {
            quantization_params: vec![
                nxpu_backend_core::QuantParam {
                    name: "x".into(),
                    scale: 0.5,
                    zero_point: 0,
                },
                nxpu_backend_core::QuantParam {
                    name: "y".into(),
                    scale: 0.25,
                    zero_point: 128,
                },
            ],
            ..Default::default()
        };
        let output = backend.compile(&module, &opts).unwrap();

        // Should have the .tflite file plus quant_params.json
        assert!(output.files.len() >= 2);

        let json_file = output
            .files
            .iter()
            .find(|f| f.name == "quant_params.json")
            .expect("expected quant_params.json file");

        let json_text = match &json_file.content {
            OutputContent::Text(t) => t,
            _ => panic!("expected text content for quant_params.json"),
        };

        assert!(json_text.contains("\"name\": \"x\""));
        assert!(json_text.contains("\"scale\": 0.5"));
        assert!(json_text.contains("\"zero_point\": 0"));
        assert!(json_text.contains("\"name\": \"y\""));
        assert!(json_text.contains("\"scale\": 0.25"));
        assert!(json_text.contains("\"zero_point\": 128"));
    }

    #[test]
    fn pattern_summary_all_variants() {
        use nxpu_analysis::analyze::*;

        // Test a few representative summaries to ensure pattern_summary works
        let matmul = KernelPattern::MatMul {
            inputs: [
                make_tensor("A", TensorRole::Input),
                make_tensor("B", TensorRole::Input),
            ],
            output: make_tensor("C", TensorRole::Output),
            shape: MatMulShape {
                m: "M".into(),
                n: "N".into(),
                k: "K".into(),
            },
        };
        assert_eq!(pattern_summary(&matmul), "BATCH_MATMUL");

        let add = KernelPattern::ElementWise {
            op: ElementWiseOp::Add,
            inputs: [
                make_tensor("a", TensorRole::Input),
                make_tensor("b", TensorRole::Input),
            ],
            output: make_tensor("c", TensorRole::Output),
            dim_name: "N".into(),
        };
        assert_eq!(pattern_summary(&add), "Add");

        let conv = KernelPattern::Conv2D {
            input: make_tensor("x", TensorRole::Input),
            weight: make_tensor("w", TensorRole::Input),
            output: make_tensor("y", TensorRole::Output),
            shape: Conv2DShape {
                batch: "N".into(),
                channels_in: "IC".into(),
                channels_out: "OC".into(),
                height: "H".into(),
                width: "W".into(),
                kernel_h: "KH".into(),
                kernel_w: "KW".into(),
                kernel_h_val: 3,
                kernel_w_val: 3,
                stride_h: 1,
                stride_w: 1,
                pad_h: 0,
                pad_w: 0,
                groups: 1,
                dilation_h: 1,
                dilation_w: 1,
            },
        };
        assert_eq!(pattern_summary(&conv), "CONV_2D");

        let norm = KernelPattern::Normalization {
            input: make_tensor("x", TensorRole::Input),
            scale: make_tensor("g", TensorRole::Input),
            bias: make_tensor("b", TensorRole::Input),
            output: make_tensor("y", TensorRole::Output),
            epsilon: 1e-5,
            norm_type: NormType::Batch,
        };
        assert_eq!(pattern_summary(&norm), "BatchNormalization");
    }
}
