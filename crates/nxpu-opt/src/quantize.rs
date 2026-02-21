//! Precision conversion passes for NPU quantization.
//!
//! Rewrites `array<f32>` global variable types to lower-precision
//! element types suitable for specific NPU backends.

use nxpu_ir::{Handle, Module, Scalar, Type, TypeInner};

use crate::Pass;

/// Parameters describing how floating-point values were quantized to integers.
#[derive(Clone, Debug)]
pub struct QuantizationParams {
    pub scale: f32,
    pub zero_point: i32,
}

/// Rewrite array element precision from F32 to a target scalar type.
///
/// 1. Insert the target scalar type into the module's type arena.
/// 2. Find the existing F32 scalar handle.
/// 3. For each Array type whose `base == f32_handle`, insert a new Array type
///    with the target base and adjusted stride.
/// 4. Update `GlobalVariable.ty` handles via the remap.
fn rewrite_array_elem_precision(module: &mut Module, target_scalar: Scalar) -> bool {
    // Insert target scalar type.
    let target_scalar_handle = module.types.insert(Type {
        name: None,
        inner: TypeInner::Scalar(target_scalar),
    });

    // Find existing F32 scalar handle.
    let f32_handle = module.types.insert(Type {
        name: None,
        inner: TypeInner::Scalar(Scalar::F32),
    });

    // Collect array types that need rewriting: (old_array_handle -> new_array_handle).
    let mut remap: Vec<(Handle<Type>, Handle<Type>)> = Vec::new();

    // First pass: find all array types with f32 base.
    let array_types: Vec<_> = module
        .types
        .iter()
        .filter_map(|(handle, ty)| {
            if let TypeInner::Array { base, size, stride } = &ty.inner
                && *base == f32_handle
            {
                return Some((handle, *size, *stride));
            }
            None
        })
        .collect();

    // Second pass: insert new array types and build remap.
    for (old_handle, size, old_stride) in array_types {
        let new_stride = old_stride * (target_scalar.width as u32) / (Scalar::F32.width as u32);
        let new_handle = module.types.insert(Type {
            name: None,
            inner: TypeInner::Array {
                base: target_scalar_handle,
                size,
                stride: new_stride,
            },
        });
        if old_handle != new_handle {
            remap.push((old_handle, new_handle));
        }
    }

    if remap.is_empty() {
        return false;
    }

    // Apply remap to global variables.
    let mut changed = false;
    for (_handle, gv) in module.global_variables.iter_mut() {
        for (old, new) in &remap {
            if gv.ty == *old {
                gv.ty = *new;
                changed = true;
            }
        }
    }

    changed
}

/// Converts `array<f32>` global variables to `array<f16>`.
#[derive(Debug)]
pub struct F32ToF16;

impl Pass for F32ToF16 {
    fn name(&self) -> &str {
        "F32ToF16"
    }

    fn run(&self, module: &mut Module) -> bool {
        rewrite_array_elem_precision(module, Scalar::F16)
    }
}

/// Converts `array<f32>` global variables to `array<bf16>`.
#[derive(Debug)]
pub struct F32ToBf16;

impl Pass for F32ToBf16 {
    fn name(&self) -> &str {
        "F32ToBf16"
    }

    fn run(&self, module: &mut Module) -> bool {
        rewrite_array_elem_precision(module, Scalar::BF16)
    }
}

/// Converts `array<f32>` global variables to `array<i8>`.
///
/// Stores default quantization parameters (scale=1.0, zero_point=0)
/// for downstream consumers that need them.
#[derive(Debug)]
pub struct F32ToInt8 {
    pub params: QuantizationParams,
}

impl Default for F32ToInt8 {
    fn default() -> Self {
        Self {
            params: QuantizationParams {
                scale: 1.0,
                zero_point: 0,
            },
        }
    }
}

impl Pass for F32ToInt8 {
    fn name(&self) -> &str {
        "F32ToInt8"
    }

    fn run(&self, module: &mut Module) -> bool {
        rewrite_array_elem_precision(module, Scalar::I8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nxpu_ir::*;

    fn make_f32_array_module() -> (Module, Handle<GlobalVariable>, Handle<GlobalVariable>) {
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

        let h0 = module.global_variables.append(GlobalVariable {
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
        });
        let h1 = module.global_variables.append(GlobalVariable {
            name: Some("b".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 1,
            }),
            ty: array_f32,
            init: None,
        });

        (module, h0, h1)
    }

    fn get_array_elem_scalar(module: &Module, gv_handle: Handle<GlobalVariable>) -> Scalar {
        let ty = &module.types[module.global_variables[gv_handle].ty];
        match &ty.inner {
            TypeInner::Array { base, .. } => match &module.types[*base].inner {
                TypeInner::Scalar(s) => *s,
                other => panic!("expected Scalar, got {other:?}"),
            },
            other => panic!("expected Array, got {other:?}"),
        }
    }

    fn get_array_stride(module: &Module, gv_handle: Handle<GlobalVariable>) -> u32 {
        let ty = &module.types[module.global_variables[gv_handle].ty];
        match &ty.inner {
            TypeInner::Array { stride, .. } => *stride,
            other => panic!("expected Array, got {other:?}"),
        }
    }

    #[test]
    fn f32_to_f16() {
        let (mut module, h0, h1) = make_f32_array_module();
        let changed = F32ToF16.run(&mut module);
        assert!(changed);

        assert_eq!(get_array_elem_scalar(&module, h0), Scalar::F16);
        assert_eq!(get_array_elem_scalar(&module, h1), Scalar::F16);
        assert_eq!(get_array_stride(&module, h0), 2);
    }

    #[test]
    fn f32_to_bf16() {
        let (mut module, h0, _h1) = make_f32_array_module();
        let changed = F32ToBf16.run(&mut module);
        assert!(changed);

        assert_eq!(get_array_elem_scalar(&module, h0), Scalar::BF16);
        assert_eq!(get_array_stride(&module, h0), 2);
    }

    #[test]
    fn f32_to_int8() {
        let (mut module, h0, _h1) = make_f32_array_module();
        let changed = F32ToInt8::default().run(&mut module);
        assert!(changed);

        assert_eq!(get_array_elem_scalar(&module, h0), Scalar::I8);
        assert_eq!(get_array_stride(&module, h0), 1);
    }

    #[test]
    fn no_change_when_no_f32_arrays() {
        let mut module = Module::default();
        let changed = F32ToF16.run(&mut module);
        assert!(!changed);
    }

    #[test]
    fn idempotent() {
        let (mut module, _h0, _h1) = make_f32_array_module();
        F32ToF16.run(&mut module);
        let changed = F32ToF16.run(&mut module);
        // After first rewrite there are no more f32 arrays, so no change.
        assert!(!changed);
    }
}
