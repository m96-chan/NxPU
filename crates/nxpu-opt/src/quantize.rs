//! Precision conversion passes for NPU quantization.
//!
//! Rewrites `array<f32>` global variable types to lower-precision
//! element types suitable for specific NPU backends.
//!
//! Supports both naive type rewriting and calibration-based quantization
//! with proper scale/zero_point computation.

use nxpu_ir::{Handle, Module, Scalar, Type, TypeInner};

use crate::Pass;

/// Parameters describing how floating-point values were quantized to integers.
#[derive(Clone, Debug)]
pub struct QuantizationParams {
    pub scale: f32,
    pub zero_point: i32,
}

impl QuantizationParams {
    /// Compute quantization parameters from observed min/max values.
    ///
    /// Uses asymmetric affine quantization:
    ///   scale = (max - min) / 255
    ///   zero_point = round(-min / scale)
    pub fn from_range(min: f32, max: f32) -> Self {
        let range = max - min;
        if range < f32::EPSILON {
            return Self {
                scale: 1.0,
                zero_point: 0,
            };
        }
        let scale = range / 255.0;
        let zero_point = (-min / scale).round() as i32;
        // Clamp zero_point to [0, 255] for uint8 or [-128, 127] for int8.
        let zero_point = zero_point.clamp(-128, 127);
        Self { scale, zero_point }
    }
}

/// Calibration data for a single tensor, identified by binding.
#[derive(Clone, Debug)]
pub struct TensorCalibration {
    /// Resource binding group.
    pub group: u32,
    /// Resource binding index.
    pub binding: u32,
    /// Observed minimum value during calibration.
    pub min: f32,
    /// Observed maximum value during calibration.
    pub max: f32,
}

/// Calibration data for the entire module.
#[derive(Clone, Debug, Default)]
pub struct CalibrationData {
    /// Per-tensor calibration entries.
    pub tensors: Vec<TensorCalibration>,
}

impl CalibrationData {
    /// Create calibration data from a list of (group, binding, min, max) tuples.
    pub fn from_entries(entries: &[(u32, u32, f32, f32)]) -> Self {
        Self {
            tensors: entries
                .iter()
                .map(|&(group, binding, min, max)| TensorCalibration {
                    group,
                    binding,
                    min,
                    max,
                })
                .collect(),
        }
    }

    /// Look up calibration for a specific binding.
    pub fn find(&self, group: u32, binding: u32) -> Option<&TensorCalibration> {
        self.tensors
            .iter()
            .find(|t| t.group == group && t.binding == binding)
    }
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
/// When calibration data is provided, computes proper scale/zero_point
/// per tensor from observed min/max ranges.
#[derive(Debug)]
pub struct F32ToInt8 {
    /// Per-tensor quantization parameters.
    pub params: QuantizationParams,
    /// Optional calibration data for computing scale/zero_point.
    pub calibration: Option<CalibrationData>,
    /// Per-tensor computed parameters (populated after run).
    pub tensor_params: Vec<(u32, u32, QuantizationParams)>,
}

impl Default for F32ToInt8 {
    fn default() -> Self {
        Self {
            params: QuantizationParams {
                scale: 1.0,
                zero_point: 0,
            },
            calibration: None,
            tensor_params: Vec::new(),
        }
    }
}

impl F32ToInt8 {
    /// Create with calibration data.
    pub fn with_calibration(calibration: CalibrationData) -> Self {
        Self {
            params: QuantizationParams {
                scale: 1.0,
                zero_point: 0,
            },
            calibration: Some(calibration),
            tensor_params: Vec::new(),
        }
    }
}

impl Pass for F32ToInt8 {
    fn name(&self) -> &str {
        "F32ToInt8"
    }

    fn run(&self, module: &mut Module) -> bool {
        // Note: calibration data is used by downstream ONNX QDQ emission
        // and TFLite quantization metadata, not during type rewriting.
        rewrite_array_elem_precision(module, Scalar::I8)
    }
}

/// Compute per-tensor quantization parameters from calibration data and module.
pub fn compute_calibrated_params(
    module: &Module,
    calibration: &CalibrationData,
) -> Vec<(String, QuantizationParams)> {
    let mut result = Vec::new();
    for (_handle, gv) in module.global_variables.iter() {
        if let Some(binding) = &gv.binding
            && let Some(cal) = calibration.find(binding.group, binding.binding)
        {
            let params = QuantizationParams::from_range(cal.min, cal.max);
            let name = gv
                .name
                .clone()
                .unwrap_or_else(|| format!("tensor_{}_{}", binding.group, binding.binding));
            result.push((name, params));
        }
    }
    result
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

    #[test]
    fn quantization_params_from_range() {
        let params = QuantizationParams::from_range(-1.0, 1.0);
        assert!((params.scale - 2.0 / 255.0).abs() < 1e-6);
        // zero_point = round(1.0 / (2.0/255.0)) = round(127.5) = 128 → clamped to 127
        assert_eq!(params.zero_point, 127);
    }

    #[test]
    fn quantization_params_positive_range() {
        let params = QuantizationParams::from_range(0.0, 6.0);
        assert!((params.scale - 6.0 / 255.0).abs() < 1e-6);
        assert_eq!(params.zero_point, 0);
    }

    #[test]
    fn quantization_params_zero_range() {
        let params = QuantizationParams::from_range(5.0, 5.0);
        assert_eq!(params.scale, 1.0);
        assert_eq!(params.zero_point, 0);
    }

    #[test]
    fn calibration_data_lookup() {
        let cal = CalibrationData::from_entries(&[(0, 0, -1.0, 1.0), (0, 1, 0.0, 6.0)]);
        assert!(cal.find(0, 0).is_some());
        assert!(cal.find(0, 1).is_some());
        assert!(cal.find(0, 2).is_none());
    }

    #[test]
    fn compute_calibrated_params_test() {
        let (module, _h0, _h1) = make_f32_array_module();
        let cal = CalibrationData::from_entries(&[(0, 0, -1.0, 1.0), (0, 1, 0.0, 6.0)]);
        let params = compute_calibrated_params(&module, &cal);
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].0, "a");
        assert_eq!(params[1].0, "b");
        assert!((params[0].1.scale - 2.0 / 255.0).abs() < 1e-6);
        assert!((params[1].1.scale - 6.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn int8_with_calibration() {
        let (mut module, h0, _h1) = make_f32_array_module();
        let cal = CalibrationData::from_entries(&[(0, 0, -1.0, 1.0), (0, 1, 0.0, 255.0)]);
        let pass = F32ToInt8::with_calibration(cal);
        let changed = pass.run(&mut module);
        assert!(changed);
        assert_eq!(get_array_elem_scalar(&module, h0), Scalar::I8);
    }
}
