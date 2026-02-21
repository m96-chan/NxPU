//! IR pattern classification for ONNX lowering.
//!
//! Analyzes an entry point's global variables and function body to classify
//! the computation into a known ONNX-mappable pattern (MatMul, ElementWise).

use nxpu_ir::{
    AddressSpace, Arena, BinaryOp, Expression, GlobalVariable, Handle, Module, Scalar, ScalarKind,
    Statement, StorageAccess, TypeInner,
};

use crate::proto::data_type;

/// Errors during IR pattern analysis.
#[derive(Debug, thiserror::Error)]
pub enum AnalysisError {
    #[error("no entry points in module")]
    NoEntryPoints,
    #[error("entry point index {0} out of range")]
    EntryPointOutOfRange(usize),
    #[error("unsupported pattern: {0}")]
    UnsupportedPattern(String),
    #[error("missing uniform params struct")]
    MissingParams,
}

/// Role of a tensor in the computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorRole {
    Input,
    Output,
}

/// A storage buffer bound as a tensor.
#[derive(Debug, Clone)]
pub struct TensorBinding {
    pub handle: Handle<GlobalVariable>,
    pub name: String,
    pub elem_type: i32,
    pub role: TensorRole,
}

/// Symbolic dimension names for matrix multiplication.
#[derive(Debug, Clone)]
pub struct MatMulShape {
    pub m: String,
    pub n: String,
    pub k: String,
}

/// Element-wise binary operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementWiseOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl ElementWiseOp {
    /// Returns the ONNX operator type string.
    pub fn onnx_op_type(self) -> &'static str {
        match self {
            Self::Add => "Add",
            Self::Sub => "Sub",
            Self::Mul => "Mul",
            Self::Div => "Div",
        }
    }
}

/// A classified kernel pattern that maps to ONNX operators.
#[derive(Debug, Clone)]
pub enum KernelPattern {
    /// Loop + accumulation + 2 read arrays + 1 write array → ONNX `MatMul`.
    MatMul {
        inputs: [TensorBinding; 2],
        output: TensorBinding,
        shape: MatMulShape,
    },
    /// No loop + binary op on arrays → ONNX `Add`/`Sub`/`Mul`/`Div`.
    ElementWise {
        op: ElementWiseOp,
        inputs: [TensorBinding; 2],
        output: TensorBinding,
        dim_name: String,
    },
}

/// Classify an entry point into a known ONNX-mappable pattern.
pub fn classify_entry_point(
    module: &Module,
    ep_index: usize,
) -> Result<KernelPattern, AnalysisError> {
    if module.entry_points.is_empty() {
        return Err(AnalysisError::NoEntryPoints);
    }
    let ep = module
        .entry_points
        .get(ep_index)
        .ok_or(AnalysisError::EntryPointOutOfRange(ep_index))?;

    // 1. Classify globals by address space.
    let mut inputs: Vec<(Handle<GlobalVariable>, &nxpu_ir::GlobalVariable)> = Vec::new();
    let mut outputs: Vec<(Handle<GlobalVariable>, &nxpu_ir::GlobalVariable)> = Vec::new();
    let mut params_members: Option<&[nxpu_ir::StructMember]> = None;

    for (handle, gv) in module.global_variables.iter() {
        match &gv.space {
            AddressSpace::Storage { access } => {
                if access.contains(StorageAccess::STORE) {
                    outputs.push((handle, gv));
                } else {
                    inputs.push((handle, gv));
                }
            }
            AddressSpace::Uniform => {
                if let TypeInner::Struct { members, .. } = &module.types[gv.ty].inner {
                    params_members = Some(members);
                }
            }
            _ => {}
        }
    }

    // Sort inputs by resource binding order (binding 0 = A, binding 1 = B).
    inputs.sort_by_key(|(_, gv)| gv.binding.map(|b| b.binding).unwrap_or(u32::MAX));

    if inputs.len() < 2 || outputs.is_empty() {
        return Err(AnalysisError::UnsupportedPattern(
            "expected at least 2 input and 1 output storage buffers".into(),
        ));
    }

    // 2. Build tensor bindings.
    let input_a = make_binding(module, inputs[0].0, inputs[0].1, TensorRole::Input);
    let input_b = make_binding(module, inputs[1].0, inputs[1].1, TensorRole::Input);
    let output_c = make_binding(module, outputs[0].0, outputs[0].1, TensorRole::Output);

    // 3. Detect pattern from function body structure.
    if has_loop(&ep.function.body) {
        // MatMul: loop + accumulation pattern.
        let shape_names: Vec<String> = params_members
            .ok_or(AnalysisError::MissingParams)?
            .iter()
            .filter_map(|m| m.name.clone())
            .collect();

        let shape = if shape_names.len() >= 3 {
            MatMulShape {
                m: shape_names[0].clone(),
                n: shape_names[1].clone(),
                k: shape_names[2].clone(),
            }
        } else {
            MatMulShape {
                m: "M".into(),
                n: "N".into(),
                k: "K".into(),
            }
        };

        Ok(KernelPattern::MatMul {
            inputs: [input_a, input_b],
            output: output_c,
            shape,
        })
    } else {
        // ElementWise: store of binary operation.
        let op =
            find_store_binary_op(&ep.function.body, &ep.function.expressions).ok_or_else(|| {
                AnalysisError::UnsupportedPattern("no recognizable binary operation found".into())
            })?;

        let dim_name = params_members
            .and_then(|members| members.first())
            .and_then(|m| m.name.clone())
            .unwrap_or_else(|| "N".into());

        Ok(KernelPattern::ElementWise {
            op,
            inputs: [input_a, input_b],
            output: output_c,
            dim_name,
        })
    }
}

fn make_binding(
    module: &Module,
    handle: Handle<GlobalVariable>,
    gv: &nxpu_ir::GlobalVariable,
    role: TensorRole,
) -> TensorBinding {
    let elem_type = resolve_array_elem_type(module, gv.ty).unwrap_or(data_type::FLOAT);
    TensorBinding {
        handle,
        name: gv
            .name
            .clone()
            .unwrap_or_else(|| format!("tensor_{}", handle.index())),
        elem_type,
        role,
    }
}

/// Resolve an array or tensor type to its element's ONNX data type.
fn resolve_array_elem_type(module: &Module, ty: nxpu_ir::Handle<nxpu_ir::Type>) -> Option<i32> {
    match &module.types[ty].inner {
        TypeInner::Array { base, .. } => match &module.types[*base].inner {
            TypeInner::Scalar(s) => Some(scalar_to_onnx_data_type(s)),
            _ => None,
        },
        TypeInner::Tensor { scalar, .. } => Some(scalar_to_onnx_data_type(scalar)),
        _ => None,
    }
}

/// Map an IR scalar type to an ONNX data type constant.
fn scalar_to_onnx_data_type(scalar: &Scalar) -> i32 {
    match (scalar.kind, scalar.width) {
        (ScalarKind::Float, 4) => data_type::FLOAT,
        (ScalarKind::Float, 2) => data_type::FLOAT16,
        (ScalarKind::BFloat, 2) => data_type::BFLOAT16,
        (ScalarKind::Sint, 4) => data_type::INT32,
        (ScalarKind::Sint, 1) => data_type::INT8,
        (ScalarKind::Uint, 4) => data_type::UINT32,
        (ScalarKind::Uint, 1) => data_type::UINT8,
        (ScalarKind::Bool, _) => data_type::BOOL,
        _ => data_type::FLOAT,
    }
}

/// Check if a block (or any nested block) contains a Loop statement.
fn has_loop(body: &[Statement]) -> bool {
    body.iter().any(|stmt| match stmt {
        Statement::Loop { .. } => true,
        Statement::If { accept, reject, .. } => has_loop(accept) || has_loop(reject),
        _ => false,
    })
}

/// Search a block for a Store whose value is a Binary expression,
/// returning the corresponding element-wise op.
fn find_store_binary_op(body: &[Statement], exprs: &Arena<Expression>) -> Option<ElementWiseOp> {
    for stmt in body {
        match stmt {
            Statement::Store { value, .. } => {
                if let Some(Expression::Binary { op, .. }) = exprs.try_get(*value) {
                    let ew = match op {
                        BinaryOp::Add => ElementWiseOp::Add,
                        BinaryOp::Subtract => ElementWiseOp::Sub,
                        BinaryOp::Multiply => ElementWiseOp::Mul,
                        BinaryOp::Divide => ElementWiseOp::Div,
                        _ => continue,
                    };
                    return Some(ew);
                }
            }
            Statement::If { accept, reject, .. } => {
                if let Some(op) = find_store_binary_op(accept, exprs) {
                    return Some(op);
                }
                if let Some(op) = find_store_binary_op(reject, exprs) {
                    return Some(op);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use nxpu_ir::*;

    fn make_matmul_module() -> Module {
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

        // Entry point with a loop in the body (triggers MatMul detection).
        let mut func = Function::new("main");
        func.body.push(Statement::Loop {
            body: vec![Statement::Break],
            continuing: vec![],
            break_if: None,
        });

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [16, 16, 1],
            function: func,
        });

        module
    }

    fn make_elementwise_module(op: BinaryOp) -> Module {
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
            layout: None,
        });
        module.global_variables.append(GlobalVariable {
            name: Some("c".into()),
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

        // Entry point with Store of Binary (no loop → ElementWise).
        let mut func = Function::new("main");
        let left = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        let right = func
            .expressions
            .append(Expression::Literal(Literal::F32(2.0)));
        let binary = func
            .expressions
            .append(Expression::Binary { op, left, right });
        let ptr = func
            .expressions
            .append(Expression::Literal(Literal::F32(0.0)));
        func.body.push(Statement::Store {
            pointer: ptr,
            value: binary,
        });

        module.entry_points.push(EntryPoint {
            name: "vecadd".into(),
            workgroup_size: [256, 1, 1],
            function: func,
        });

        module
    }

    #[test]
    fn classify_matmul() {
        let module = make_matmul_module();
        let pattern = classify_entry_point(&module, 0).unwrap();
        match pattern {
            KernelPattern::MatMul {
                inputs,
                output,
                shape,
            } => {
                assert_eq!(inputs[0].name, "a");
                assert_eq!(inputs[1].name, "b");
                assert_eq!(output.name, "result");
                assert_eq!(inputs[0].elem_type, data_type::FLOAT);
                assert_eq!(shape.m, "M");
                assert_eq!(shape.n, "N");
                assert_eq!(shape.k, "K");
            }
            _ => panic!("expected MatMul pattern"),
        }
    }

    #[test]
    fn classify_elementwise_add() {
        let module = make_elementwise_module(BinaryOp::Add);
        let pattern = classify_entry_point(&module, 0).unwrap();
        match pattern {
            KernelPattern::ElementWise {
                op,
                inputs,
                output,
                dim_name,
            } => {
                assert_eq!(op, ElementWiseOp::Add);
                assert_eq!(inputs[0].name, "a");
                assert_eq!(inputs[1].name, "b");
                assert_eq!(output.name, "c");
                assert_eq!(dim_name, "N");
            }
            _ => panic!("expected ElementWise pattern"),
        }
    }

    #[test]
    fn classify_elementwise_div() {
        let module = make_elementwise_module(BinaryOp::Divide);
        let pattern = classify_entry_point(&module, 0).unwrap();
        match &pattern {
            KernelPattern::ElementWise { op, .. } => {
                assert_eq!(*op, ElementWiseOp::Div);
            }
            _ => panic!("expected ElementWise pattern"),
        }
    }

    #[test]
    fn classify_out_of_range() {
        let module = make_matmul_module();
        let err = classify_entry_point(&module, 99).unwrap_err();
        assert!(matches!(err, AnalysisError::EntryPointOutOfRange(99)));
    }

    #[test]
    fn classify_empty_module() {
        let module = Module::default();
        let err = classify_entry_point(&module, 0).unwrap_err();
        assert!(matches!(err, AnalysisError::NoEntryPoints));
    }

    #[test]
    fn input_sorted_by_binding() {
        // Build a module where 'b' (binding 1) is appended before 'a' (binding 0)
        // to verify that classify sorts inputs by binding order.
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

        // Append b (binding 1) BEFORE a (binding 0).
        module.global_variables.append(GlobalVariable {
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
            layout: None,
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

        let mut func = Function::new("main");
        func.body.push(Statement::Loop {
            body: vec![Statement::Break],
            continuing: vec![],
            break_if: None,
        });
        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [16, 16, 1],
            function: func,
        });

        let pattern = classify_entry_point(&module, 0).unwrap();
        match pattern {
            KernelPattern::MatMul { inputs, .. } => {
                assert_eq!(inputs[0].name, "a"); // binding 0 sorted first
                assert_eq!(inputs[1].name, "b"); // binding 1 sorted second
            }
            _ => panic!("expected MatMul pattern"),
        }
    }
}
