//! Memory layout optimization pass.
//!
//! Assigns optimal memory layouts to tensor global variables based on the
//! target backend's preferred format, and tracks where layout conversions
//! would be needed.

use nxpu_ir::{MemoryLayout, Module, TypeInner};

use crate::Pass;

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
}
