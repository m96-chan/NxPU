//! SIMD vectorization hints.
//!
//! Analyzes classified kernel patterns to identify vectorizable dimensions
//! and compute appropriate vector widths for different register sizes.

use std::fmt;

use nxpu_ir::{Module, Scalar, ScalarKind};

use crate::Pass;

/// Vector width specification for a SIMD operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorWidth {
    /// Scalar element type.
    pub scalar: Scalar,
    /// Number of SIMD lanes.
    pub lanes: u32,
    /// Register width in bits.
    pub register_bits: u32,
}

impl VectorWidth {
    /// Compute the number of lanes that fit in a register of the given bit width.
    pub fn for_register_width(scalar: Scalar, register_bits: u32) -> Self {
        let scalar_bits = scalar_width_bits(scalar);
        let lanes = (register_bits).checked_div(scalar_bits).unwrap_or(1);
        Self {
            scalar,
            lanes,
            register_bits,
        }
    }
}

impl fmt::Display for VectorWidth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?}x{} ({}b reg)",
            self.scalar, self.lanes, self.register_bits
        )
    }
}

/// A vectorization hint for a specific dimension of an operation.
#[derive(Debug, Clone)]
pub struct VectorizationHint {
    /// Name of the operation.
    pub op_name: String,
    /// Name of the dimension to vectorize.
    pub dim_name: String,
    /// Recommended vector width.
    pub vector_width: VectorWidth,
    /// Whether this dimension is a reduction (e.g., K in MatMul).
    pub is_reduction: bool,
    /// Whether memory access along this dimension is contiguous.
    pub is_contiguous: bool,
}

impl fmt::Display for VectorizationHint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let red = if self.is_reduction {
            " [reduction]"
        } else {
            ""
        };
        let contig = if self.is_contiguous {
            " [contiguous]"
        } else {
            ""
        };
        write!(
            f,
            "{}.{}: {}{red}{contig}",
            self.op_name, self.dim_name, self.vector_width
        )
    }
}

/// Analyze a classified kernel pattern and produce vectorization hints.
///
/// `register_bits` specifies the SIMD register width (e.g., 128 for NEON/SSE,
/// 256 for AVX2, 512 for AVX-512).
pub fn analyze_vectorization(
    pattern: &nxpu_analysis::KernelPattern,
    register_bits: u32,
) -> Vec<VectorizationHint> {
    let scalar = Scalar::F32; // Default; could be refined from pattern.
    let vw = VectorWidth::for_register_width(scalar, register_bits);

    match pattern {
        nxpu_analysis::KernelPattern::MatMul { shape, .. } => {
            vec![
                VectorizationHint {
                    op_name: "MatMul".into(),
                    dim_name: shape.n.clone(),
                    vector_width: vw.clone(),
                    is_reduction: false,
                    is_contiguous: true,
                },
                VectorizationHint {
                    op_name: "MatMul".into(),
                    dim_name: shape.k.clone(),
                    vector_width: vw,
                    is_reduction: true,
                    is_contiguous: false,
                },
            ]
        }
        nxpu_analysis::KernelPattern::ElementWise { dim_name, .. } => {
            vec![VectorizationHint {
                op_name: "ElementWise".into(),
                dim_name: dim_name.clone(),
                vector_width: vw,
                is_reduction: false,
                is_contiguous: true,
            }]
        }
        nxpu_analysis::KernelPattern::Activation { dim_name, .. } => {
            vec![VectorizationHint {
                op_name: "Activation".into(),
                dim_name: dim_name.clone(),
                vector_width: vw,
                is_reduction: false,
                is_contiguous: true,
            }]
        }
        nxpu_analysis::KernelPattern::Conv2D { shape, .. } => {
            vec![VectorizationHint {
                op_name: "Conv2D".into(),
                dim_name: shape.width.clone(),
                vector_width: vw,
                is_reduction: false,
                is_contiguous: true,
            }]
        }
        nxpu_analysis::KernelPattern::Reduce { op, axis, .. } => {
            vec![VectorizationHint {
                op_name: op.op_name().to_string(),
                dim_name: format!("axis_{axis}"),
                vector_width: vw,
                is_reduction: true,
                is_contiguous: true,
            }]
        }
        _ => Vec::new(),
    }
}

/// Vectorization hint pass.
///
/// Classifies entry points and produces vectorization hints for each.
#[derive(Debug)]
pub struct VectorizationPass {
    register_bits: u32,
}

impl VectorizationPass {
    /// Create a vectorization pass for the given register width.
    pub fn new(register_bits: u32) -> Self {
        Self { register_bits }
    }
}

impl Default for VectorizationPass {
    fn default() -> Self {
        Self::new(128)
    }
}

impl Pass for VectorizationPass {
    fn name(&self) -> &str {
        "vectorize"
    }

    fn run(&self, module: &mut Module) -> bool {
        let mut any = false;
        for i in 0..module.entry_points.len() {
            if let Ok(pattern) = nxpu_analysis::classify_entry_point(module, i) {
                let hints = analyze_vectorization(&pattern, self.register_bits);
                if !hints.is_empty() {
                    any = true;
                }
            }
        }
        any
    }
}

/// Return the width of a scalar type in bits.
fn scalar_width_bits(scalar: Scalar) -> u32 {
    match scalar.kind {
        ScalarKind::Float | ScalarKind::BFloat | ScalarKind::Sint | ScalarKind::Uint => {
            scalar.width as u32 * 8
        }
        ScalarKind::Bool => 8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nxpu_analysis::{
        ActivationOp, Conv2DShape, ElementWiseOp, KernelPattern, MatMulShape, ReduceOp,
        TensorBinding, TensorRole,
    };
    use nxpu_ir::{
        AddressSpace, Arena, GlobalVariable, Scalar, StorageAccess, Type, TypeInner, UniqueArena,
    };

    fn dummy_binding(name: &str, role: TensorRole) -> TensorBinding {
        let mut types = UniqueArena::new();
        let ty = types.insert(Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::F32),
        });
        let mut arena = Arena::new();
        let handle = arena.append(GlobalVariable {
            name: Some(name.into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD,
            },
            binding: None,
            ty,
            init: None,
            layout: None,
        });
        TensorBinding {
            handle,
            name: name.into(),
            elem_type: 1, // FLOAT
            role,
        }
    }

    #[test]
    fn vector_width_f32_128() {
        let vw = VectorWidth::for_register_width(Scalar::F32, 128);
        assert_eq!(vw.lanes, 4);
    }

    #[test]
    fn vector_width_f16_128() {
        let vw = VectorWidth::for_register_width(Scalar::F16, 128);
        assert_eq!(vw.lanes, 8);
    }

    #[test]
    fn vector_width_int8_128() {
        let scalar = Scalar {
            kind: ScalarKind::Sint,
            width: 1,
        };
        let vw = VectorWidth::for_register_width(scalar, 128);
        assert_eq!(vw.lanes, 16);
    }

    #[test]
    fn vector_width_f32_256() {
        let vw = VectorWidth::for_register_width(Scalar::F32, 256);
        assert_eq!(vw.lanes, 8);
    }

    #[test]
    fn vectorize_matmul() {
        let pattern = KernelPattern::MatMul {
            inputs: [
                dummy_binding("A", TensorRole::Input),
                dummy_binding("B", TensorRole::Input),
            ],
            output: dummy_binding("C", TensorRole::Output),
            shape: MatMulShape {
                m: "M".into(),
                n: "N".into(),
                k: "K".into(),
            },
        };
        let hints = analyze_vectorization(&pattern, 128);
        assert_eq!(hints.len(), 2);
        // First hint: N dimension (contiguous)
        assert_eq!(hints[0].dim_name, "N");
        assert!(hints[0].is_contiguous);
        assert!(!hints[0].is_reduction);
        // Second hint: K dimension (reduction)
        assert_eq!(hints[1].dim_name, "K");
        assert!(hints[1].is_reduction);
    }

    #[test]
    fn vectorize_elementwise() {
        let pattern = KernelPattern::ElementWise {
            op: ElementWiseOp::Add,
            inputs: [
                dummy_binding("A", TensorRole::Input),
                dummy_binding("B", TensorRole::Input),
            ],
            output: dummy_binding("C", TensorRole::Output),
            dim_name: "N".into(),
        };
        let hints = analyze_vectorization(&pattern, 128);
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].dim_name, "N");
        assert!(hints[0].is_contiguous);
    }

    #[test]
    fn vectorize_conv2d() {
        let pattern = KernelPattern::Conv2D {
            input: dummy_binding("input", TensorRole::Input),
            weight: dummy_binding("weight", TensorRole::Input),
            output: dummy_binding("output", TensorRole::Output),
            shape: Conv2DShape {
                batch: "1".into(),
                channels_in: "3".into(),
                channels_out: "16".into(),
                height: "32".into(),
                width: "W".into(),
                kernel_h: "3".into(),
                kernel_w: "3".into(),
                kernel_h_val: 3,
                kernel_w_val: 3,
                stride_h: 1,
                stride_w: 1,
                pad_h: 1,
                pad_w: 1,
                groups: 1,
                dilation_h: 1,
                dilation_w: 1,
            },
        };
        let hints = analyze_vectorization(&pattern, 128);
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].dim_name, "W");
    }

    #[test]
    fn vectorize_reduction() {
        let pattern = KernelPattern::Reduce {
            op: ReduceOp::Sum,
            input: dummy_binding("input", TensorRole::Input),
            output: dummy_binding("output", TensorRole::Output),
            axis: 1,
        };
        let hints = analyze_vectorization(&pattern, 128);
        assert_eq!(hints.len(), 1);
        assert!(hints[0].is_reduction);
    }

    #[test]
    fn vectorize_activation() {
        let pattern = KernelPattern::Activation {
            op: ActivationOp::Relu,
            input: dummy_binding("input", TensorRole::Input),
            output: dummy_binding("output", TensorRole::Output),
            dim_name: "N".into(),
        };
        let hints = analyze_vectorization(&pattern, 128);
        assert_eq!(hints.len(), 1);
        assert!(hints[0].is_contiguous);
        assert!(!hints[0].is_reduction);
    }

    #[test]
    fn pass_on_module() {
        // Empty module → no entry points → no hints.
        let mut module = Module::default();
        let pass = VectorizationPass::default();
        let changed = pass.run(&mut module);
        assert!(!changed);
    }
}
