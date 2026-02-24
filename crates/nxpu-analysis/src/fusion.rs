//! Kernel fusion: merges adjacent classified patterns into fused operations.
//!
//! After pattern classification, `fuse_patterns()` scans adjacent patterns
//! and merges compatible sequences (e.g., Conv+BatchNorm, Add+ReLU).

use crate::analyze::KernelPattern;

/// Returns the output tensor names of a pattern.
pub fn output_tensor_names(pattern: &KernelPattern) -> Vec<&str> {
    match pattern {
        KernelPattern::MatMul { output, .. }
        | KernelPattern::ElementWise { output, .. }
        | KernelPattern::Conv2D { output, .. }
        | KernelPattern::Pool { output, .. }
        | KernelPattern::Activation { output, .. }
        | KernelPattern::Reduce { output, .. }
        | KernelPattern::Transpose { output, .. }
        | KernelPattern::Reshape { output, .. }
        | KernelPattern::Normalization { output, .. }
        | KernelPattern::Concat { output, .. }
        | KernelPattern::Attention { output, .. } => vec![output.name.as_str()],
        KernelPattern::Split { outputs, .. } => outputs.iter().map(|t| t.name.as_str()).collect(),
        KernelPattern::Unknown { .. } => vec![],
    }
}

/// Returns the input tensor names of a pattern.
pub fn input_tensor_names(pattern: &KernelPattern) -> Vec<&str> {
    match pattern {
        KernelPattern::MatMul { inputs, .. } => inputs.iter().map(|t| t.name.as_str()).collect(),
        KernelPattern::ElementWise { inputs, .. } => {
            inputs.iter().map(|t| t.name.as_str()).collect()
        }
        KernelPattern::Conv2D { input, weight, .. } => {
            vec![input.name.as_str(), weight.name.as_str()]
        }
        KernelPattern::Pool { input, .. }
        | KernelPattern::Activation { input, .. }
        | KernelPattern::Reduce { input, .. }
        | KernelPattern::Transpose { input, .. }
        | KernelPattern::Reshape { input, .. }
        | KernelPattern::Split { input, .. } => vec![input.name.as_str()],
        KernelPattern::Normalization {
            input, scale, bias, ..
        } => vec![input.name.as_str(), scale.name.as_str(), bias.name.as_str()],
        KernelPattern::Concat { inputs, .. } => inputs.iter().map(|t| t.name.as_str()).collect(),
        KernelPattern::Attention {
            query, key, value, ..
        } => vec![query.name.as_str(), key.name.as_str(), value.name.as_str()],
        KernelPattern::Unknown { .. } => vec![],
    }
}

/// Returns `true` if any output tensor name of `producer` matches any input
/// tensor name of `consumer`, indicating data flows between them.
pub fn tensors_connect(producer: &KernelPattern, consumer: &KernelPattern) -> bool {
    let outputs = output_tensor_names(producer);
    let inputs = input_tensor_names(consumer);
    outputs.iter().any(|o| inputs.contains(o))
}

/// Fused activation function appended to a base operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedActivation {
    None,
    Relu,
    Sigmoid,
    Tanh,
}

impl std::fmt::Display for FusedActivation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::None => "None",
            Self::Relu => "Relu",
            Self::Sigmoid => "Sigmoid",
            Self::Tanh => "Tanh",
        })
    }
}

/// Try to map an `ActivationOp` to a `FusedActivation`.
/// Returns `None` for activations that cannot be fused (e.g. Softmax).
fn try_fuse_activation(op: &crate::analyze::ActivationOp) -> Option<FusedActivation> {
    match op {
        crate::analyze::ActivationOp::Relu => Some(FusedActivation::Relu),
        crate::analyze::ActivationOp::Sigmoid => Some(FusedActivation::Sigmoid),
        crate::analyze::ActivationOp::Tanh => Some(FusedActivation::Tanh),
        crate::analyze::ActivationOp::Softmax => None,
    }
}

/// A pattern that may be fused from one or more classified patterns.
#[derive(Debug, Clone)]
pub enum FusedPattern {
    /// A single unfused pattern.
    Single(KernelPattern),
    /// Conv2D followed by BatchNormalization.
    ConvBatchNorm {
        conv: KernelPattern,
        norm: Box<KernelPattern>,
    },
    /// A base pattern followed by an activation function.
    WithActivation {
        base: Box<FusedPattern>,
        activation: FusedActivation,
        /// The original activation pattern (preserved for tensor connectivity).
        activation_pattern: Box<KernelPattern>,
    },
    /// MatMul followed by Add (bias) — maps to ONNX Gemm.
    MatMulBias {
        matmul: KernelPattern,
        bias_add: Box<KernelPattern>,
    },
}

impl std::fmt::Display for FusedPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Single(p) => write!(f, "{p}"),
            Self::ConvBatchNorm { conv, .. } => write!(f, "{conv}+BatchNorm"),
            Self::WithActivation {
                base, activation, ..
            } => write!(f, "{base}+{activation}"),
            Self::MatMulBias { .. } => write!(f, "Gemm"),
        }
    }
}

impl FusedPattern {
    /// Return a reference to the primary pattern (for lowering).
    pub fn primary_pattern(&self) -> &KernelPattern {
        match self {
            FusedPattern::Single(p) => p,
            FusedPattern::ConvBatchNorm { conv, .. } => conv,
            FusedPattern::WithActivation { base, .. } => base.primary_pattern(),
            FusedPattern::MatMulBias { matmul, .. } => matmul,
        }
    }

    /// Return a reference to the last (output-producing) pattern in the fused
    /// chain. For connectivity checks, this determines which tensor name
    /// flows out of the fused operation.
    fn output_pattern(&self) -> &KernelPattern {
        match self {
            FusedPattern::Single(p) => p,
            FusedPattern::ConvBatchNorm { norm, .. } => norm,
            FusedPattern::WithActivation {
                activation_pattern, ..
            } => activation_pattern,
            FusedPattern::MatMulBias { bias_add, .. } => bias_add,
        }
    }

    /// Return the fused activation, if any.
    pub fn fused_activation(&self) -> FusedActivation {
        match self {
            FusedPattern::WithActivation { activation, .. } => *activation,
            _ => FusedActivation::None,
        }
    }
}

/// Greedy adjacent fusion of classified kernel patterns.
///
/// Scans the pattern list and merges compatible adjacent pairs:
/// - Conv2D + Normalization → ConvBatchNorm
/// - MatMul + ElementWise(Add) → MatMulBias (Gemm)
/// - Any + Activation(Relu/Sigmoid/Tanh) → WithActivation { base, activation }
pub fn fuse_patterns(patterns: Vec<KernelPattern>) -> Vec<(FusedPattern, usize)> {
    let mut result: Vec<(FusedPattern, usize)> = Vec::new();
    let mut iter = patterns.into_iter().enumerate().peekable();

    while let Some((idx, pattern)) = iter.next() {
        // Unknown patterns pass through as Single — skip fusion attempts.
        if matches!(&pattern, KernelPattern::Unknown { .. }) {
            result.push((FusedPattern::Single(pattern), idx));
            continue;
        }

        let fused = match &pattern {
            KernelPattern::Conv2D { .. } => {
                if let Some((_, next)) = iter.peek() {
                    if matches!(next, KernelPattern::Normalization { .. })
                        && tensors_connect(&pattern, next)
                    {
                        let (_, norm) = iter.next().unwrap();
                        FusedPattern::ConvBatchNorm {
                            conv: pattern,
                            norm: Box::new(norm),
                        }
                    } else {
                        FusedPattern::Single(pattern)
                    }
                } else {
                    FusedPattern::Single(pattern)
                }
            }
            KernelPattern::MatMul { .. } => {
                if let Some((_, next)) = iter.peek() {
                    let is_add = matches!(
                        next,
                        KernelPattern::ElementWise {
                            op: crate::analyze::ElementWiseOp::Add,
                            ..
                        }
                    );
                    if is_add && tensors_connect(&pattern, next) {
                        let (_, bias_add) = iter.next().unwrap();
                        FusedPattern::MatMulBias {
                            matmul: pattern,
                            bias_add: Box::new(bias_add),
                        }
                    } else {
                        FusedPattern::Single(pattern)
                    }
                } else {
                    FusedPattern::Single(pattern)
                }
            }
            _ => FusedPattern::Single(pattern),
        };

        // Try to fuse a trailing activation.
        let fused = if let Some((_, next)) = iter.peek() {
            if let KernelPattern::Activation { op, .. } = next {
                if let Some(fused_act) = try_fuse_activation(op) {
                    if tensors_connect(fused.output_pattern(), next) {
                        let (_, act_pattern) = iter.next().unwrap();
                        FusedPattern::WithActivation {
                            base: Box::new(fused),
                            activation: fused_act,
                            activation_pattern: Box::new(act_pattern),
                        }
                    } else {
                        fused
                    }
                } else {
                    fused
                }
            } else {
                fused
            }
        } else {
            fused
        };

        result.push((fused, idx));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze::data_type;
    use crate::analyze::*;

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

    fn make_tensor(name: &str, role: TensorRole) -> TensorBinding {
        TensorBinding {
            handle: dummy_handle(),
            name: name.into(),
            elem_type: data_type::FLOAT,
            role,
        }
    }

    #[test]
    fn single_pattern_no_fusion() {
        let patterns = vec![KernelPattern::ElementWise {
            op: ElementWiseOp::Add,
            inputs: [
                make_tensor("a", TensorRole::Input),
                make_tensor("b", TensorRole::Input),
            ],
            output: make_tensor("c", TensorRole::Output),
            dim_name: "N".into(),
        }];

        let fused = fuse_patterns(patterns);
        assert_eq!(fused.len(), 1);
        let (ref fp, idx) = fused[0];
        assert!(matches!(fp, FusedPattern::Single(_)));
        assert_eq!(idx, 0);
    }

    #[test]
    fn conv_batchnorm_fusion() {
        let patterns = vec![
            KernelPattern::Conv2D {
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
                },
            },
            KernelPattern::Normalization {
                input: make_tensor("conv_out", TensorRole::Input),
                scale: make_tensor("gamma", TensorRole::Input),
                bias: make_tensor("beta", TensorRole::Input),
                output: make_tensor("bn_out", TensorRole::Output),
                epsilon: 1e-5,
            },
        ];

        let fused = fuse_patterns(patterns);
        assert_eq!(fused.len(), 1);
        let (ref fp, idx) = fused[0];
        assert!(matches!(fp, FusedPattern::ConvBatchNorm { .. }));
        assert_eq!(idx, 0);
    }

    #[test]
    fn add_relu_fusion() {
        let patterns = vec![
            KernelPattern::ElementWise {
                op: ElementWiseOp::Add,
                inputs: [
                    make_tensor("a", TensorRole::Input),
                    make_tensor("b", TensorRole::Input),
                ],
                output: make_tensor("c", TensorRole::Output),
                dim_name: "N".into(),
            },
            KernelPattern::Activation {
                op: ActivationOp::Relu,
                input: make_tensor("c", TensorRole::Input),
                output: make_tensor("d", TensorRole::Output),
                dim_name: "N".into(),
            },
        ];

        let fused = fuse_patterns(patterns);
        assert_eq!(fused.len(), 1);
        let (ref fp, idx) = fused[0];
        assert!(matches!(
            fp,
            FusedPattern::WithActivation {
                activation: FusedActivation::Relu,
                ..
            }
        ));
        assert_eq!(idx, 0);
    }

    #[test]
    fn conv_bn_relu_fusion() {
        let patterns = vec![
            KernelPattern::Conv2D {
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
                },
            },
            KernelPattern::Normalization {
                input: make_tensor("conv_out", TensorRole::Input),
                scale: make_tensor("gamma", TensorRole::Input),
                bias: make_tensor("beta", TensorRole::Input),
                output: make_tensor("bn_out", TensorRole::Output),
                epsilon: 1e-5,
            },
            KernelPattern::Activation {
                op: ActivationOp::Relu,
                input: make_tensor("bn_out", TensorRole::Input),
                output: make_tensor("relu_out", TensorRole::Output),
                dim_name: "N".into(),
            },
        ];

        let fused = fuse_patterns(patterns);
        assert_eq!(fused.len(), 1);
        let (ref fp, idx) = fused[0];
        assert_eq!(idx, 0);
        match fp {
            FusedPattern::WithActivation {
                base,
                activation: FusedActivation::Relu,
                ..
            } => {
                assert!(matches!(**base, FusedPattern::ConvBatchNorm { .. }));
            }
            other => panic!("expected WithActivation(ConvBatchNorm, Relu), got {other:?}"),
        }
    }

    #[test]
    fn tanh_activation_now_fused() {
        let patterns = vec![
            KernelPattern::ElementWise {
                op: ElementWiseOp::Add,
                inputs: [
                    make_tensor("a", TensorRole::Input),
                    make_tensor("b", TensorRole::Input),
                ],
                output: make_tensor("c", TensorRole::Output),
                dim_name: "N".into(),
            },
            KernelPattern::Activation {
                op: ActivationOp::Tanh,
                input: make_tensor("c", TensorRole::Input),
                output: make_tensor("d", TensorRole::Output),
                dim_name: "N".into(),
            },
        ];

        let fused = fuse_patterns(patterns);
        // Tanh is now fused.
        assert_eq!(fused.len(), 1);
        assert!(matches!(
            &fused[0].0,
            FusedPattern::WithActivation {
                activation: FusedActivation::Tanh,
                ..
            }
        ));
    }

    #[test]
    fn no_fusion_for_softmax_activation() {
        let patterns = vec![
            KernelPattern::ElementWise {
                op: ElementWiseOp::Add,
                inputs: [
                    make_tensor("a", TensorRole::Input),
                    make_tensor("b", TensorRole::Input),
                ],
                output: make_tensor("c", TensorRole::Output),
                dim_name: "N".into(),
            },
            KernelPattern::Activation {
                op: ActivationOp::Softmax,
                input: make_tensor("c", TensorRole::Input),
                output: make_tensor("d", TensorRole::Output),
                dim_name: "N".into(),
            },
        ];

        let fused = fuse_patterns(patterns);
        // Softmax is not fusible — remains as 2 separate patterns.
        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].1, 0);
        assert_eq!(fused[1].1, 1);
    }

    #[test]
    fn matmul_add_fusion_to_gemm() {
        let patterns = vec![
            KernelPattern::MatMul {
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
            KernelPattern::ElementWise {
                op: ElementWiseOp::Add,
                inputs: [
                    make_tensor("mm_out", TensorRole::Input),
                    make_tensor("bias", TensorRole::Input),
                ],
                output: make_tensor("out", TensorRole::Output),
                dim_name: "N".into(),
            },
        ];

        let fused = fuse_patterns(patterns);
        assert_eq!(fused.len(), 1);
        assert!(matches!(&fused[0].0, FusedPattern::MatMulBias { .. }));
    }

    #[test]
    fn add_sigmoid_fusion() {
        let patterns = vec![
            KernelPattern::ElementWise {
                op: ElementWiseOp::Add,
                inputs: [
                    make_tensor("a", TensorRole::Input),
                    make_tensor("b", TensorRole::Input),
                ],
                output: make_tensor("c", TensorRole::Output),
                dim_name: "N".into(),
            },
            KernelPattern::Activation {
                op: ActivationOp::Sigmoid,
                input: make_tensor("c", TensorRole::Input),
                output: make_tensor("d", TensorRole::Output),
                dim_name: "N".into(),
            },
        ];

        let fused = fuse_patterns(patterns);
        assert_eq!(fused.len(), 1);
        assert!(matches!(
            &fused[0].0,
            FusedPattern::WithActivation {
                activation: FusedActivation::Sigmoid,
                ..
            }
        ));
    }

    #[test]
    fn tensors_connect_matching_names() {
        let producer = KernelPattern::ElementWise {
            op: ElementWiseOp::Add,
            inputs: [
                make_tensor("a", TensorRole::Input),
                make_tensor("b", TensorRole::Input),
            ],
            output: make_tensor("c", TensorRole::Output),
            dim_name: "N".into(),
        };
        let consumer = KernelPattern::Activation {
            op: ActivationOp::Relu,
            input: make_tensor("c", TensorRole::Input),
            output: make_tensor("d", TensorRole::Output),
            dim_name: "N".into(),
        };
        assert!(tensors_connect(&producer, &consumer));
    }

    #[test]
    fn tensors_connect_mismatched_names() {
        let producer = KernelPattern::ElementWise {
            op: ElementWiseOp::Add,
            inputs: [
                make_tensor("a", TensorRole::Input),
                make_tensor("b", TensorRole::Input),
            ],
            output: make_tensor("c", TensorRole::Output),
            dim_name: "N".into(),
        };
        let consumer = KernelPattern::Activation {
            op: ActivationOp::Relu,
            input: make_tensor("x", TensorRole::Input),
            output: make_tensor("y", TensorRole::Output),
            dim_name: "N".into(),
        };
        assert!(!tensors_connect(&producer, &consumer));
    }

    #[test]
    fn no_fusion_mismatched_tensor_names() {
        // Add outputs "c" but Relu consumes "x" — should NOT fuse.
        let patterns = vec![
            KernelPattern::ElementWise {
                op: ElementWiseOp::Add,
                inputs: [
                    make_tensor("a", TensorRole::Input),
                    make_tensor("b", TensorRole::Input),
                ],
                output: make_tensor("c", TensorRole::Output),
                dim_name: "N".into(),
            },
            KernelPattern::Activation {
                op: ActivationOp::Relu,
                input: make_tensor("x", TensorRole::Input),
                output: make_tensor("y", TensorRole::Output),
                dim_name: "N".into(),
            },
        ];

        let fused = fuse_patterns(patterns);
        assert_eq!(fused.len(), 2);
    }
}
