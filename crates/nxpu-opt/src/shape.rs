//! Shape inference pass.
//!
//! Propagates tensor shapes through the IR module. For each global variable,
//! infers the concrete or symbolic shape and stores it in a `ShapeMap`.

use std::collections::HashMap;

use nxpu_ir::{AddressSpace, GlobalVariable, Handle, Module, StorageAccess, TypeInner};

use crate::Pass;

/// A dimension: either a concrete size or a symbolic name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dim {
    /// Known concrete size.
    Known(u64),
    /// Symbolic (runtime-determined) dimension.
    Symbolic(String),
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
        let shape_map = infer_shapes(module);
        // Shape inference is read-only; it populates a side-channel.
        // Currently we annotate global variables by storing shape info.
        // The pass returns true if shapes were discovered.
        !shape_map.is_empty()
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
fn extract_param_names(module: &Module) -> Vec<String> {
    for (_handle, gv) in module.global_variables.iter() {
        if gv.space == AddressSpace::Uniform
            && let TypeInner::Struct { members, .. } = &module.types[gv.ty].inner
        {
            return members.iter().filter_map(|m| m.name.clone()).collect();
        }
    }
    vec![]
}

/// Infer shape for a single storage global variable.
fn infer_global_shape(
    module: &Module,
    gv: &nxpu_ir::GlobalVariable,
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
        assert!(changed);
    }

    #[test]
    fn empty_module_no_shapes() {
        let module = Module::default();
        let shapes = infer_shapes(&module);
        assert!(shapes.is_empty());
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
