//! Memory layout optimization pass.
//!
//! Assigns optimal memory layouts to tensor global variables based on the
//! target backend's preferred format, and tracks where layout conversions
//! would be needed.

use nxpu_ir::{MemoryLayout, Module, TypeInner};

use crate::Pass;

/// Returns the permutation needed to convert from one layout to another.
///
/// NHWC -> NCHW = [0, 3, 1, 2]  (move channels from last to second)
/// NCHW -> NHWC = [0, 2, 3, 1]  (move channels from second to last)
///
/// Returns `None` if layouts are the same or conversion is not applicable
/// (e.g. between `RowMajor` and a spatial layout).
pub fn layout_permutation(from: MemoryLayout, to: MemoryLayout) -> Option<Vec<i64>> {
    match (from, to) {
        (MemoryLayout::Nhwc, MemoryLayout::Nchw) => Some(vec![0, 3, 1, 2]),
        (MemoryLayout::Nchw, MemoryLayout::Nhwc) => Some(vec![0, 2, 3, 1]),
        _ => None, // Same layout or not a spatial layout conversion
    }
}

/// Reorder 4-D shape dimensions from one layout to another.
///
/// Given dimensions in `from` layout order, returns them in `to` layout order.
/// For example, if `from` = NCHW and the dims are `[N, C, H, W]`, converting to
/// NHWC gives `[N, H, W, C]`.
///
/// Non-4-D shapes are returned unchanged, as are shapes where no permutation
/// applies (same layout, or non-spatial layout pair).
pub fn reorder_dims(dims: &[i64], from: MemoryLayout, to: MemoryLayout) -> Vec<i64> {
    if dims.len() != 4 {
        return dims.to_vec();
    }
    match layout_permutation(from, to) {
        Some(perm) => perm.iter().map(|&p| dims[p as usize]).collect(),
        None => dims.to_vec(),
    }
}

/// Information about a needed transpose at a graph boundary.
#[derive(Debug, Clone)]
pub struct TransposeRecord {
    /// Name of the global variable that needs transposing.
    pub var_name: String,
    /// The permutation to apply (e.g. `[0, 3, 1, 2]`).
    pub perm: Vec<i64>,
    /// Whether this is an input transpose (before computation) or output (after).
    pub is_input: bool,
}

/// Pass that detects layout mismatches at graph boundaries and records
/// the transposes needed to convert between layouts.
///
/// This is a two-phase approach:
/// 1. [`LayoutTransform`] assigns target layouts to uninitialized globals.
/// 2. `TransposeInsertion` detects globals whose current layout differs from
///    the target and updates them, so the backend can emit the appropriate
///    transpose operations.
///
/// The backend can call [`layout_permutation`] with the original and target
/// layouts to obtain the concrete permutation vector.
#[derive(Debug)]
pub struct TransposeInsertion {
    /// The target layout that the backend expects.
    pub target: MemoryLayout,
}

impl Pass for TransposeInsertion {
    fn name(&self) -> &str {
        "TransposeInsertion"
    }

    fn run(&self, module: &mut Module) -> bool {
        let mut changed = false;

        for (_handle, gv) in module.global_variables.iter_mut() {
            let current_layout = match gv.layout {
                Some(l) => l,
                None => continue,
            };

            if current_layout == self.target {
                continue;
            }

            // Only convert between spatial layouts (NHWC <-> NCHW).
            if layout_permutation(current_layout, self.target).is_some() {
                gv.layout = Some(self.target);
                changed = true;
            }
        }

        changed
    }
}

/// Assigns a target memory layout to all tensor-typed global variables
/// that do not already have a layout annotation.
///
/// This pass propagates the target layout uniformly. A more advanced version
/// could perform per-tensor analysis to minimize conversion overhead.
#[derive(Debug)]
pub struct LayoutTransform {
    /// The target layout to assign.
    pub target: MemoryLayout,
}

impl Pass for LayoutTransform {
    fn name(&self) -> &str {
        "LayoutTransform"
    }

    fn run(&self, module: &mut Module) -> bool {
        let mut changed = false;

        for (_handle, gv) in module.global_variables.iter_mut() {
            // Only apply to tensor-typed globals or array globals that
            // represent tensor data (storage buffers).
            if gv.layout.is_some() {
                continue;
            }

            let is_tensor_like = match &module.types[gv.ty].inner {
                TypeInner::Tensor { .. } => true,
                TypeInner::Array { .. } => {
                    // Storage arrays are treated as flat tensor buffers.
                    matches!(gv.space, nxpu_ir::AddressSpace::Storage { .. })
                }
                _ => false,
            };

            if is_tensor_like {
                gv.layout = Some(self.target);
                changed = true;
            }
        }

        changed
    }
}

/// Counts the number of layout mismatches between global variables.
///
/// A mismatch occurs when two globals that feed into the same operation
/// have different layouts. This is useful for diagnostics.
pub fn count_layout_mismatches(module: &Module) -> usize {
    let layouts: Vec<Option<MemoryLayout>> = module
        .global_variables
        .iter()
        .map(|(_, gv)| gv.layout)
        .collect();

    let assigned: Vec<MemoryLayout> = layouts.into_iter().flatten().collect();
    if assigned.is_empty() {
        return 0;
    }

    let first = assigned[0];
    assigned.iter().filter(|&&l| l != first).count()
}

/// Returns the preferred memory layout for a given backend target name.
pub fn preferred_layout_for_target(target: &str) -> MemoryLayout {
    match target {
        "tflite" | "litert" | "arm-ethos" | "ethos-u" | "mediatek" | "rockchip" => {
            MemoryLayout::Nhwc
        }
        "onnx" | "intel" | "amd" | "qualcomm" => MemoryLayout::Nchw,
        "coreml" | "apple-ane" => MemoryLayout::Nhwc,
        _ => MemoryLayout::RowMajor,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nxpu_ir::*;

    fn make_tensor_module() -> Module {
        let mut module = Module::default();

        let f32_ty = module.types.insert(Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::F32),
        });
        let array_f32 = module.types.insert(Type {
            name: None,
            inner: TypeInner::Array {
                base: f32_ty,
                size: ArraySize::Dynamic,
                stride: 4,
            },
        });
        let tensor_ty = module.types.insert(Type {
            name: None,
            inner: TypeInner::Tensor {
                scalar: Scalar::F32,
                shape: TensorShape {
                    dims: vec![
                        Dimension::Dynamic(Some("batch".into())),
                        Dimension::Fixed(224),
                        Dimension::Fixed(224),
                        Dimension::Fixed(3),
                    ],
                },
            },
        });

        // Storage array (treated as tensor buffer)
        module.global_variables.append(GlobalVariable {
            name: Some("weights".into()),
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

        // Explicit tensor type
        module.global_variables.append(GlobalVariable {
            name: Some("input".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 1,
            }),
            ty: tensor_ty,
            init: None,
            layout: None,
        });

        // Uniform (should NOT get layout)
        let u32_ty = module.types.insert(Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::U32),
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
    fn assigns_nhwc_layout() {
        let mut module = make_tensor_module();
        let pass = LayoutTransform {
            target: MemoryLayout::Nhwc,
        };
        let changed = pass.run(&mut module);
        assert!(changed);

        let layouts: Vec<_> = module
            .global_variables
            .iter()
            .map(|(_, gv)| gv.layout)
            .collect();

        // weights (storage array) and input (tensor) should have NHWC
        assert_eq!(layouts[0], Some(MemoryLayout::Nhwc));
        assert_eq!(layouts[1], Some(MemoryLayout::Nhwc));
        // params (uniform struct) should have no layout
        assert_eq!(layouts[2], None);
    }

    #[test]
    fn idempotent() {
        let mut module = make_tensor_module();
        let pass = LayoutTransform {
            target: MemoryLayout::Nchw,
        };
        pass.run(&mut module);
        let changed = pass.run(&mut module);
        assert!(!changed); // Already assigned
    }

    #[test]
    fn does_not_overwrite_existing() {
        let mut module = make_tensor_module();
        // Pre-assign NCHW to first variable
        for (_, gv) in module.global_variables.iter_mut() {
            if gv.name.as_deref() == Some("weights") {
                gv.layout = Some(MemoryLayout::Nchw);
            }
        }

        let pass = LayoutTransform {
            target: MemoryLayout::Nhwc,
        };
        let changed = pass.run(&mut module);
        assert!(changed); // Changed the tensor

        let layouts: Vec<_> = module
            .global_variables
            .iter()
            .map(|(_, gv)| gv.layout)
            .collect();

        // weights keeps NCHW (not overwritten)
        assert_eq!(layouts[0], Some(MemoryLayout::Nchw));
        // input gets NHWC
        assert_eq!(layouts[1], Some(MemoryLayout::Nhwc));
    }

    #[test]
    fn preferred_layout_tflite() {
        assert_eq!(preferred_layout_for_target("tflite"), MemoryLayout::Nhwc);
        assert_eq!(preferred_layout_for_target("arm-ethos"), MemoryLayout::Nhwc);
    }

    #[test]
    fn preferred_layout_onnx() {
        assert_eq!(preferred_layout_for_target("onnx"), MemoryLayout::Nchw);
        assert_eq!(preferred_layout_for_target("intel"), MemoryLayout::Nchw);
    }

    #[test]
    fn count_mismatches() {
        let mut module = make_tensor_module();

        // No layouts assigned yet
        assert_eq!(count_layout_mismatches(&module), 0);

        // Assign same layout to all
        let pass = LayoutTransform {
            target: MemoryLayout::Nhwc,
        };
        pass.run(&mut module);
        assert_eq!(count_layout_mismatches(&module), 0);

        // Override one to create a mismatch
        for (_, gv) in module.global_variables.iter_mut() {
            if gv.name.as_deref() == Some("weights") {
                gv.layout = Some(MemoryLayout::Nchw);
            }
        }
        assert_eq!(count_layout_mismatches(&module), 1);
    }

    // ---- layout_permutation tests ----

    #[test]
    fn layout_permutation_nhwc_to_nchw() {
        assert_eq!(
            layout_permutation(MemoryLayout::Nhwc, MemoryLayout::Nchw),
            Some(vec![0, 3, 1, 2])
        );
    }

    #[test]
    fn layout_permutation_nchw_to_nhwc() {
        assert_eq!(
            layout_permutation(MemoryLayout::Nchw, MemoryLayout::Nhwc),
            Some(vec![0, 2, 3, 1])
        );
    }

    #[test]
    fn layout_permutation_same_layout() {
        assert_eq!(
            layout_permutation(MemoryLayout::Nhwc, MemoryLayout::Nhwc),
            None
        );
        assert_eq!(
            layout_permutation(MemoryLayout::Nchw, MemoryLayout::Nchw),
            None
        );
    }

    #[test]
    fn layout_permutation_row_major() {
        assert_eq!(
            layout_permutation(MemoryLayout::RowMajor, MemoryLayout::Nhwc),
            None
        );
        assert_eq!(
            layout_permutation(MemoryLayout::Nhwc, MemoryLayout::RowMajor),
            None
        );
    }

    #[test]
    fn layout_permutation_col_major() {
        assert_eq!(
            layout_permutation(MemoryLayout::ColMajor, MemoryLayout::Nchw),
            None
        );
    }

    // ---- reorder_dims tests ----

    #[test]
    fn reorder_dims_nchw_to_nhwc() {
        // NCHW [1, 3, 224, 224] -> NHWC [1, 224, 224, 3]
        let nchw = vec![1, 3, 224, 224];
        let nhwc = reorder_dims(&nchw, MemoryLayout::Nchw, MemoryLayout::Nhwc);
        assert_eq!(nhwc, vec![1, 224, 224, 3]);
    }

    #[test]
    fn reorder_dims_nhwc_to_nchw() {
        // NHWC [1, 224, 224, 3] -> NCHW [1, 3, 224, 224]
        let nhwc = vec![1, 224, 224, 3];
        let nchw = reorder_dims(&nhwc, MemoryLayout::Nhwc, MemoryLayout::Nchw);
        assert_eq!(nchw, vec![1, 3, 224, 224]);
    }

    #[test]
    fn reorder_dims_non_4d_passthrough() {
        let dims = vec![1, 2, 3];
        assert_eq!(
            reorder_dims(&dims, MemoryLayout::Nhwc, MemoryLayout::Nchw),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn reorder_dims_same_layout_passthrough() {
        let dims = vec![1, 3, 224, 224];
        assert_eq!(
            reorder_dims(&dims, MemoryLayout::Nchw, MemoryLayout::Nchw),
            vec![1, 3, 224, 224]
        );
    }

    #[test]
    fn reorder_dims_5d_passthrough() {
        let dims = vec![1, 2, 3, 4, 5];
        assert_eq!(
            reorder_dims(&dims, MemoryLayout::Nhwc, MemoryLayout::Nchw),
            vec![1, 2, 3, 4, 5]
        );
    }

    #[test]
    fn reorder_dims_roundtrip() {
        let original = vec![1, 224, 224, 3]; // NHWC
        let nchw = reorder_dims(&original, MemoryLayout::Nhwc, MemoryLayout::Nchw);
        let back = reorder_dims(&nchw, MemoryLayout::Nchw, MemoryLayout::Nhwc);
        assert_eq!(back, original);
    }

    // ---- TransposeInsertion pass tests ----

    #[test]
    fn no_transpose_when_layout_matches() {
        let mut module = make_tensor_module();
        // Assign all to Nhwc
        let pass = LayoutTransform {
            target: MemoryLayout::Nhwc,
        };
        pass.run(&mut module);
        // All are Nhwc now, so TransposeInsertion targeting Nhwc should be a no-op
        let pass2 = TransposeInsertion {
            target: MemoryLayout::Nhwc,
        };
        assert!(!pass2.run(&mut module));
    }

    #[test]
    fn insert_transpose_nhwc_to_nchw() {
        let mut module = make_tensor_module();
        // Assign Nhwc layout
        let pass = LayoutTransform {
            target: MemoryLayout::Nhwc,
        };
        pass.run(&mut module);
        // Now convert to Nchw
        let pass2 = TransposeInsertion {
            target: MemoryLayout::Nchw,
        };
        assert!(pass2.run(&mut module));
        // All tensor globals should now have Nchw layout
        for (_, gv) in module.global_variables.iter() {
            if gv.layout.is_some() {
                assert_eq!(gv.layout, Some(MemoryLayout::Nchw));
            }
        }
    }

    #[test]
    fn insert_transpose_nchw_to_nhwc() {
        let mut module = make_tensor_module();
        let pass = LayoutTransform {
            target: MemoryLayout::Nchw,
        };
        pass.run(&mut module);
        let pass2 = TransposeInsertion {
            target: MemoryLayout::Nhwc,
        };
        assert!(pass2.run(&mut module));
        for (_, gv) in module.global_variables.iter() {
            if gv.layout.is_some() {
                assert_eq!(gv.layout, Some(MemoryLayout::Nhwc));
            }
        }
    }

    #[test]
    fn transpose_insertion_idempotent() {
        let mut module = make_tensor_module();
        let pass = LayoutTransform {
            target: MemoryLayout::Nhwc,
        };
        pass.run(&mut module);
        let pass2 = TransposeInsertion {
            target: MemoryLayout::Nchw,
        };
        pass2.run(&mut module);
        // Running again should be a no-op
        assert!(!pass2.run(&mut module));
    }

    #[test]
    fn transpose_insertion_skips_non_spatial() {
        let mut module = make_tensor_module();
        // Assign RowMajor layout to all
        for (_, gv) in module.global_variables.iter_mut() {
            let is_tensor_like = match &module.types[gv.ty].inner {
                TypeInner::Tensor { .. } => true,
                TypeInner::Array { .. } => {
                    matches!(gv.space, AddressSpace::Storage { .. })
                }
                _ => false,
            };
            if is_tensor_like {
                gv.layout = Some(MemoryLayout::RowMajor);
            }
        }
        // TransposeInsertion to Nchw should not change RowMajor layouts
        // (no permutation exists for RowMajor -> Nchw)
        let pass = TransposeInsertion {
            target: MemoryLayout::Nchw,
        };
        assert!(!pass.run(&mut module));
    }

    #[test]
    fn transpose_insertion_skips_unset_layout() {
        let mut module = make_tensor_module();
        // No layouts are assigned yet
        let pass = TransposeInsertion {
            target: MemoryLayout::Nchw,
        };
        assert!(!pass.run(&mut module));
    }

    #[test]
    fn transpose_insertion_name() {
        let pass = TransposeInsertion {
            target: MemoryLayout::Nchw,
        };
        assert_eq!(pass.name(), "TransposeInsertion");
    }

    #[test]
    fn transpose_record_clone() {
        let record = TransposeRecord {
            var_name: "input".into(),
            perm: vec![0, 3, 1, 2],
            is_input: true,
        };
        let cloned = record.clone();
        assert_eq!(cloned.var_name, "input");
        assert_eq!(cloned.perm, vec![0, 3, 1, 2]);
        assert!(cloned.is_input);
    }
}
