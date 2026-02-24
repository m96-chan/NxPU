//! Shape inference pass.
//!
//! Propagates tensor shapes through the IR module. For each global variable,
//! infers the concrete or symbolic shape and stores it in a `ShapeMap`.

use std::collections::HashMap;

use nxpu_ir::{AddressSpace, Dimension, GlobalVariable, Handle, Module, StorageAccess, TypeInner};

use crate::Pass;

/// A dimension: either a concrete size, a symbolic name, or fully dynamic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dim {
    /// Known concrete size.
    Known(u64),
    /// Symbolic (named, constrained) dimension.
    Symbolic(String),
    /// Fully dynamic (unknown, unnamed) dimension.
    Dynamic,
}

/// Inferred shape of a tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    pub dims: Vec<Dim>,
}

impl Shape {
    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }
}

/// Unify two dimensions according to shape inference rules:
/// - `Known(n) + Known(n)` -> `Known(n)` (match)
/// - `Known(n) + Dynamic` -> `Known(n)` (concrete wins)
/// - `Dynamic + Known(n)` -> `Known(n)` (concrete wins)
/// - `Dynamic + Dynamic` -> `Dynamic`
/// - `Symbolic(s) + Symbolic(s)` -> `Symbolic(s)` (same name matches)
/// - `Symbolic(s) + Dynamic` -> `Symbolic(s)` (named wins)
/// - `Known(1) + any` -> broadcast (returns the other)
///
/// Returns `None` if the dimensions are incompatible (e.g. `Known(3)` vs `Known(5)`).
pub fn unify_dim(a: &Dim, b: &Dim) -> Option<Dim> {
    match (a, b) {
        // Identical known dimensions
        (Dim::Known(x), Dim::Known(y)) if x == y => Some(Dim::Known(*x)),
        // Broadcast: Known(1) with any -> the other
        (Dim::Known(1), other) | (other, Dim::Known(1)) => Some(other.clone()),
        // Mismatched known dimensions (non-broadcastable)
        (Dim::Known(_), Dim::Known(_)) => None,
        // Concrete wins over dynamic/symbolic
        (Dim::Known(n), Dim::Dynamic) | (Dim::Dynamic, Dim::Known(n)) => Some(Dim::Known(*n)),
        (Dim::Known(n), Dim::Symbolic(_)) | (Dim::Symbolic(_), Dim::Known(n)) => {
            Some(Dim::Known(*n))
        }
        // Same symbolic name
        (Dim::Symbolic(a_name), Dim::Symbolic(b_name)) if a_name == b_name => {
            Some(Dim::Symbolic(a_name.clone()))
        }
        // Different symbolic names — cannot unify
        (Dim::Symbolic(_), Dim::Symbolic(_)) => None,
        // Symbolic wins over dynamic
        (Dim::Symbolic(s), Dim::Dynamic) | (Dim::Dynamic, Dim::Symbolic(s)) => {
            Some(Dim::Symbolic(s.clone()))
        }
        // Both dynamic
        (Dim::Dynamic, Dim::Dynamic) => Some(Dim::Dynamic),
    }
}

/// Unify two shapes element-wise. Returns `None` if ranks differ or
/// any dimension pair is incompatible.
pub fn unify_shapes(a: &Shape, b: &Shape) -> Option<Shape> {
    if a.rank() != b.rank() {
        return None;
    }
    let dims: Option<Vec<Dim>> = a
        .dims
        .iter()
        .zip(b.dims.iter())
        .map(|(da, db)| unify_dim(da, db))
        .collect();
    dims.map(|d| Shape { dims: d })
}

/// Convert an IR `Dimension` to a shape inference `Dim`.
pub fn ir_dim_to_dim(d: &Dimension) -> Dim {
    match d {
        Dimension::Fixed(n) => Dim::Known(*n as u64),
        Dimension::Symbolic(name) => Dim::Symbolic(name.clone()),
        Dimension::Dynamic(Some(name)) => Dim::Symbolic(name.clone()),
        Dimension::Dynamic(None) => Dim::Dynamic,
    }
}

/// Map from global variable handles to their inferred shapes.
pub type ShapeMap = HashMap<Handle<GlobalVariable>, Shape>;

/// Shape inference pass.
///
/// Analyzes the module's global variables and uniform params to infer
/// tensor shapes. Shapes are inferred from:
/// - Array types (runtime-sized → dynamic dim)
/// - Uniform struct member names (convention: N, M, K, C, H, W, etc.)
/// - Workgroup size hints
///
/// This pass does not modify the module; it only produces a `ShapeMap`
/// stored in module metadata (currently as a side-channel).
#[derive(Debug)]
pub struct ShapeInference;

impl Pass for ShapeInference {
    fn name(&self) -> &str {
        "ShapeInference"
    }

    fn run(&self, module: &mut Module) -> bool {
        let _shape_map = infer_shapes(module);
        // Shape inference is a pure analysis pass — it does not modify the IR.
        // Consumers should call `infer_shapes()` directly to obtain the map.
        false
    }
}

/// Infer shapes for all storage global variables in the module.
pub fn infer_shapes(module: &Module) -> ShapeMap {
    let mut map = ShapeMap::new();

    // 1. Extract param names from Uniform structs.
    let param_names = extract_param_names(module);

    // 2. For each storage buffer, infer shape from type and params.
    for (handle, gv) in module.global_variables.iter() {
        if let AddressSpace::Storage { access } = &gv.space {
            let shape = infer_global_shape(module, gv, &param_names, *access);
            map.insert(handle, shape);
        }
    }

    map
}

/// Extract parameter names from uniform struct members.
#[allow(clippy::collapsible_if)] // nested if-let for MSRV 1.87 compat (no let chains)
fn extract_param_names(module: &Module) -> Vec<String> {
    for (_handle, gv) in module.global_variables.iter() {
        if gv.space == AddressSpace::Uniform {
            if let TypeInner::Struct { members, .. } = &module.types[gv.ty].inner {
                return members.iter().filter_map(|m| m.name.clone()).collect();
            }
        }
    }
    vec![]
}

/// Infer shape for a single storage global variable.
fn infer_global_shape(
    module: &Module,
    gv: &GlobalVariable,
    param_names: &[String],
    access: StorageAccess,
) -> Shape {
    match &module.types[gv.ty].inner {
        TypeInner::Array { size, .. } => {
            match size {
                nxpu_ir::ArraySize::Constant(n) => {
                    // Fixed-size array → single known dimension.
                    Shape {
                        dims: vec![Dim::Known(*n as u64)],
                    }
                }
                nxpu_ir::ArraySize::Dynamic => {
                    // Dynamic array → infer shape from params.
                    infer_dynamic_shape(param_names, access)
                }
            }
        }
        TypeInner::Tensor { shape, .. } => {
            // Tensor type already carries shape information;
            // convert IR Dimensions to inference Dims.
            Shape {
                dims: shape.dims.iter().map(ir_dim_to_dim).collect(),
            }
        }
        _ => {
            // Non-array type → scalar, rank 0.
            Shape { dims: vec![] }
        }
    }
}

/// Infer shape for a dynamically-sized storage buffer from param names.
fn infer_dynamic_shape(param_names: &[String], access: StorageAccess) -> Shape {
    let is_output = access.contains(StorageAccess::STORE);

    match param_names.len() {
        // MatMul convention: M, N, K
        3 => {
            if is_output {
                // Output: [M, N]
                Shape {
                    dims: vec![
                        Dim::Symbolic(param_names[0].clone()),
                        Dim::Symbolic(param_names[1].clone()),
                    ],
                }
            } else {
                // Inputs are [M,K] or [K,N] — we'd need binding info
                // to distinguish, so use symbolic dims.
                Shape {
                    dims: vec![Dim::Symbolic("?".into()), Dim::Symbolic("?".into())],
                }
            }
        }
        // ElementWise or Activation: N
        1 => Shape {
            dims: vec![Dim::Symbolic(param_names[0].clone())],
        },
        // Conv2D or complex: multi-dimensional
        n if n > 3 => {
            // Assume NCHW layout
            let dims = param_names
                .iter()
                .take(4.min(n))
                .map(|name| Dim::Symbolic(name.clone()))
                .collect();
            Shape { dims }
        }
        // 2 params (e.g., transpose: rows, cols)
        2 => Shape {
            dims: vec![
                Dim::Symbolic(param_names[0].clone()),
                Dim::Symbolic(param_names[1].clone()),
            ],
        },
        // Unknown
        _ => Shape {
            dims: vec![Dim::Symbolic("?".into())],
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nxpu_ir::*;

    fn make_simple_module() -> Module {
        let mut module = Module::default();

        let f32_ty = module.types.insert(Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::F32),
        });
        let u32_ty = module.types.insert(Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::U32),
        });
        let array_f32 = module.types.insert(Type {
            name: None,
            inner: TypeInner::Array {
                base: f32_ty,
                size: ArraySize::Dynamic,
                stride: 4,
            },
        });
        let params_ty = module.types.insert(Type {
            name: Some("Params".into()),
            inner: TypeInner::Struct {
                members: vec![StructMember {
                    name: Some("N".into()),
                    ty: u32_ty,
                    offset: 0,
                }],
                span: 4,
            },
        });

        module.global_variables.append(GlobalVariable {
            name: Some("a".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: array_f32,
            init: None,
            layout: None,
        });
        module.global_variables.append(GlobalVariable {
            name: Some("c".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 1,
            }),
            ty: array_f32,
            init: None,
            layout: None,
        });
        module.global_variables.append(GlobalVariable {
            name: Some("params".into()),
            space: AddressSpace::Uniform,
            binding: Some(ResourceBinding {
                group: 0,
                binding: 2,
            }),
            ty: params_ty,
            init: None,
            layout: None,
        });

        module
    }

    #[test]
    fn infer_elementwise_shapes() {
        let module = make_simple_module();
        let shapes = infer_shapes(&module);
        assert_eq!(shapes.len(), 2); // only storage buffers

        for shape in shapes.values() {
            assert_eq!(shape.rank(), 1);
            assert_eq!(shape.dims[0], Dim::Symbolic("N".into()));
        }
    }

    #[test]
    fn infer_matmul_shapes() {
        let mut module = Module::default();

        let f32_ty = module.types.insert(Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::F32),
        });
        let u32_ty = module.types.insert(Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::U32),
        });
        let array_f32 = module.types.insert(Type {
            name: None,
            inner: TypeInner::Array {
                base: f32_ty,
                size: ArraySize::Dynamic,
                stride: 4,
            },
        });
        let params_ty = module.types.insert(Type {
            name: Some("Params".into()),
            inner: TypeInner::Struct {
                members: vec![
                    StructMember {
                        name: Some("M".into()),
                        ty: u32_ty,
                        offset: 0,
                    },
                    StructMember {
                        name: Some("N".into()),
                        ty: u32_ty,
                        offset: 4,
                    },
                    StructMember {
                        name: Some("K".into()),
                        ty: u32_ty,
                        offset: 8,
                    },
                ],
                span: 12,
            },
        });

        module.global_variables.append(GlobalVariable {
            name: Some("a".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: array_f32,
            init: None,
            layout: None,
        });
        module.global_variables.append(GlobalVariable {
            name: Some("result".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 2,
            }),
            ty: array_f32,
            init: None,
            layout: None,
        });
        module.global_variables.append(GlobalVariable {
            name: Some("params".into()),
            space: AddressSpace::Uniform,
            binding: Some(ResourceBinding {
                group: 0,
                binding: 3,
            }),
            ty: params_ty,
            init: None,
            layout: None,
        });

        let shapes = infer_shapes(&module);
        // Output should be [M, N]
        let result_handle = module
            .global_variables
            .iter()
            .find(|(_, gv)| gv.name.as_deref() == Some("result"))
            .unwrap()
            .0;
        let result_shape = &shapes[&result_handle];
        assert_eq!(result_shape.rank(), 2);
        assert_eq!(result_shape.dims[0], Dim::Symbolic("M".into()));
        assert_eq!(result_shape.dims[1], Dim::Symbolic("N".into()));
    }

    #[test]
    fn shape_inference_pass_runs() {
        let mut module = make_simple_module();
        let pass = ShapeInference;
        let changed = pass.run(&mut module);
        // Analysis pass never reports changes.
        assert!(!changed);
    }

    #[test]
    fn empty_module_no_shapes() {
        let module = Module::default();
        let shapes = infer_shapes(&module);
        assert!(shapes.is_empty());
    }

    #[test]
    fn unify_known_known_match() {
        assert_eq!(
            unify_dim(&Dim::Known(10), &Dim::Known(10)),
            Some(Dim::Known(10))
        );
    }

    #[test]
    fn unify_known_known_mismatch() {
        assert_eq!(unify_dim(&Dim::Known(3), &Dim::Known(5)), None);
    }

    #[test]
    fn unify_known_dynamic() {
        assert_eq!(
            unify_dim(&Dim::Known(10), &Dim::Dynamic),
            Some(Dim::Known(10))
        );
        assert_eq!(
            unify_dim(&Dim::Dynamic, &Dim::Known(10)),
            Some(Dim::Known(10))
        );
    }

    #[test]
    fn unify_dynamic_dynamic() {
        assert_eq!(unify_dim(&Dim::Dynamic, &Dim::Dynamic), Some(Dim::Dynamic));
    }

    #[test]
    fn unify_symbolic_same_name() {
        assert_eq!(
            unify_dim(
                &Dim::Symbolic("batch".into()),
                &Dim::Symbolic("batch".into())
            ),
            Some(Dim::Symbolic("batch".into()))
        );
    }

    #[test]
    fn unify_symbolic_different_names() {
        assert_eq!(
            unify_dim(&Dim::Symbolic("batch".into()), &Dim::Symbolic("seq".into())),
            None
        );
    }

    #[test]
    fn unify_symbolic_dynamic() {
        assert_eq!(
            unify_dim(&Dim::Symbolic("batch".into()), &Dim::Dynamic),
            Some(Dim::Symbolic("batch".into()))
        );
        assert_eq!(
            unify_dim(&Dim::Dynamic, &Dim::Symbolic("batch".into())),
            Some(Dim::Symbolic("batch".into()))
        );
    }

    #[test]
    fn unify_broadcast_known_one() {
        assert_eq!(
            unify_dim(&Dim::Known(1), &Dim::Known(10)),
            Some(Dim::Known(10))
        );
        assert_eq!(
            unify_dim(&Dim::Known(10), &Dim::Known(1)),
            Some(Dim::Known(10))
        );
        assert_eq!(unify_dim(&Dim::Known(1), &Dim::Dynamic), Some(Dim::Dynamic));
        assert_eq!(
            unify_dim(&Dim::Known(1), &Dim::Symbolic("batch".into())),
            Some(Dim::Symbolic("batch".into()))
        );
    }

    #[test]
    fn unify_shapes_match() {
        let a = Shape {
            dims: vec![Dim::Known(10), Dim::Symbolic("K".into())],
        };
        let b = Shape {
            dims: vec![Dim::Known(10), Dim::Symbolic("K".into())],
        };
        let result = unify_shapes(&a, &b).unwrap();
        assert_eq!(result.dims, vec![Dim::Known(10), Dim::Symbolic("K".into())]);
    }

    #[test]
    fn unify_shapes_rank_mismatch() {
        let a = Shape {
            dims: vec![Dim::Known(10)],
        };
        let b = Shape {
            dims: vec![Dim::Known(10), Dim::Known(5)],
        };
        assert!(unify_shapes(&a, &b).is_none());
    }

    #[test]
    fn unify_shapes_concrete_wins_over_dynamic() {
        let a = Shape {
            dims: vec![Dim::Known(10), Dim::Dynamic],
        };
        let b = Shape {
            dims: vec![Dim::Dynamic, Dim::Known(20)],
        };
        let result = unify_shapes(&a, &b).unwrap();
        assert_eq!(result.dims, vec![Dim::Known(10), Dim::Known(20)]);
    }

    #[test]
    fn ir_dim_conversion() {
        use nxpu_ir::Dimension;
        assert_eq!(ir_dim_to_dim(&Dimension::Fixed(42)), Dim::Known(42));
        assert_eq!(ir_dim_to_dim(&Dimension::Dynamic(None)), Dim::Dynamic);
        assert_eq!(
            ir_dim_to_dim(&Dimension::Dynamic(Some("batch".into()))),
            Dim::Symbolic("batch".into())
        );
        assert_eq!(
            ir_dim_to_dim(&Dimension::Symbolic("seq_len".into())),
            Dim::Symbolic("seq_len".into())
        );
    }

    #[test]
    fn infer_tensor_type_shape() {
        let mut module = Module::default();

        let tensor_ty = module.types.insert(Type {
            name: None,
            inner: TypeInner::Tensor {
                scalar: Scalar::F32,
                shape: TensorShape {
                    dims: vec![
                        Dimension::Symbolic("batch".into()),
                        Dimension::Fixed(224),
                        Dimension::Fixed(224),
                        Dimension::Fixed(3),
                    ],
                },
            },
        });

        module.global_variables.append(GlobalVariable {
            name: Some("image".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD,
            },
            binding: None,
            ty: tensor_ty,
            init: None,
            layout: None,
        });

        let shapes = infer_shapes(&module);
        assert_eq!(shapes.len(), 1);
        let shape = shapes.values().next().unwrap();
        assert_eq!(shape.rank(), 4);
        assert_eq!(shape.dims[0], Dim::Symbolic("batch".into()));
        assert_eq!(shape.dims[1], Dim::Known(224));
        assert_eq!(shape.dims[2], Dim::Known(224));
        assert_eq!(shape.dims[3], Dim::Known(3));
    }

    #[test]
    fn fixed_size_array() {
        let mut module = Module::default();

        let f32_ty = module.types.insert(Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::F32),
        });
        let array_fixed = module.types.insert(Type {
            name: None,
            inner: TypeInner::Array {
                base: f32_ty,
                size: ArraySize::Constant(128),
                stride: 4,
            },
        });

        module.global_variables.append(GlobalVariable {
            name: Some("buf".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD,
            },
            binding: None,
            ty: array_fixed,
            init: None,
            layout: None,
        });

        let shapes = infer_shapes(&module);
        assert_eq!(shapes.len(), 1);
        let shape = shapes.values().next().unwrap();
        assert_eq!(shape.dims[0], Dim::Known(128));
    }
}
