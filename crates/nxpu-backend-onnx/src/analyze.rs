//! IR pattern classification for ONNX lowering.
//!
//! Analyzes an entry point's global variables and function body to classify
//! the computation into a known ONNX-mappable pattern.

use nxpu_ir::{
    AddressSpace, Arena, BinaryOp, Expression, GlobalVariable, Handle, MathFunction, Module,
    Scalar, ScalarKind, Statement, StorageAccess, TypeInner,
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

/// Activation function kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationOp {
    Relu,
    Sigmoid,
    Tanh,
    Softmax,
}

impl ActivationOp {
    /// Returns the ONNX operator type string.
    pub fn onnx_op_type(self) -> &'static str {
        match self {
            Self::Relu => "Relu",
            Self::Sigmoid => "Sigmoid",
            Self::Tanh => "Tanh",
            Self::Softmax => "Softmax",
        }
    }
}

/// Reduction operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
    Min,
}

impl ReduceOp {
    /// Returns the ONNX operator type string.
    pub fn onnx_op_type(self) -> &'static str {
        match self {
            Self::Sum => "ReduceSum",
            Self::Mean => "ReduceMean",
            Self::Max => "ReduceMax",
            Self::Min => "ReduceMin",
        }
    }
}

/// Pooling operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolKind {
    Max,
    Avg,
}

impl PoolKind {
    /// Returns the ONNX operator type string.
    pub fn onnx_op_type(self) -> &'static str {
        match self {
            Self::Max => "MaxPool",
            Self::Avg => "AveragePool",
        }
    }
}

/// Conv2D shape parameters extracted from uniform params.
#[derive(Debug, Clone)]
pub struct Conv2DShape {
    pub batch: String,
    pub channels_in: String,
    pub channels_out: String,
    pub height: String,
    pub width: String,
    pub kernel_h: String,
    pub kernel_w: String,
    pub stride_h: i64,
    pub stride_w: i64,
    pub pad_h: i64,
    pub pad_w: i64,
}

/// Pooling shape parameters.
#[derive(Debug, Clone)]
pub struct PoolShape {
    pub kernel_h: i64,
    pub kernel_w: i64,
    pub stride_h: i64,
    pub stride_w: i64,
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
    /// 2D convolution: nested loops + kernel window + accumulation.
    Conv2D {
        input: TensorBinding,
        weight: TensorBinding,
        output: TensorBinding,
        shape: Conv2DShape,
    },
    /// Pooling: nested loops + reduction over spatial window.
    Pool {
        kind: PoolKind,
        input: TensorBinding,
        output: TensorBinding,
        shape: PoolShape,
    },
    /// Activation function: no loop, single input, unary math op.
    Activation {
        op: ActivationOp,
        input: TensorBinding,
        output: TensorBinding,
        dim_name: String,
    },
    /// Reduction over an axis: loop + accumulation, single input.
    Reduce {
        op: ReduceOp,
        input: TensorBinding,
        output: TensorBinding,
        axis: i64,
    },
    /// Transpose: permute tensor axes.
    Transpose {
        input: TensorBinding,
        output: TensorBinding,
        perm: Vec<i64>,
    },
    /// Reshape: change tensor shape without data copy.
    Reshape {
        input: TensorBinding,
        output: TensorBinding,
    },
    /// Normalization: mean + variance + scale + bias.
    Normalization {
        input: TensorBinding,
        scale: TensorBinding,
        bias: TensorBinding,
        output: TensorBinding,
        epsilon: f32,
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

    if outputs.is_empty() {
        return Err(AnalysisError::UnsupportedPattern(
            "expected at least 1 output storage buffer".into(),
        ));
    }

    let shape_names: Vec<String> = params_members
        .map(|members| members.iter().filter_map(|m| m.name.clone()).collect())
        .unwrap_or_default();

    let has_loop = has_loop(&ep.function.body);
    let num_inputs = inputs.len();

    // Single input patterns
    if num_inputs == 1 {
        let input = make_binding(module, inputs[0].0, inputs[0].1, TensorRole::Input);
        let output = make_binding(module, outputs[0].0, outputs[0].1, TensorRole::Output);

        if !has_loop {
            // Try activation (Math expression)
            if let Some(act_op) = find_store_activation(&ep.function.body, &ep.function.expressions)
            {
                let dim_name = shape_names.first().cloned().unwrap_or_else(|| "N".into());
                return Ok(KernelPattern::Activation {
                    op: act_op,
                    input,
                    output,
                    dim_name,
                });
            }

            // Try transpose (2+ params like rows, cols suggest reindexing)
            if shape_names.len() >= 2 {
                let perm: Vec<i64> = (0..shape_names.len() as i64).rev().collect();
                return Ok(KernelPattern::Transpose {
                    input,
                    output,
                    perm,
                });
            }

            // Reshape (single input, no loop, 1 param)
            return Ok(KernelPattern::Reshape { input, output });
        }

        // Has loop + single input → Pool or Reduce
        if shape_names.len() >= 4 {
            // Pool pattern: many spatial params
            let pool_kind = if find_store_math_fun(
                &ep.function.body,
                &ep.function.expressions,
                MathFunction::Max,
            ) {
                PoolKind::Max
            } else {
                PoolKind::Avg
            };
            let pool_shape = PoolShape {
                kernel_h: 2,
                kernel_w: 2,
                stride_h: 2,
                stride_w: 2,
            };
            return Ok(KernelPattern::Pool {
                kind: pool_kind,
                input,
                output,
                shape: pool_shape,
            });
        }

        // Single input + loop → Reduce
        let reduce_op = detect_reduce_op(&ep.function.body, &ep.function.expressions);
        let axis = if shape_names.len() >= 2 { 1 } else { 0 };
        return Ok(KernelPattern::Reduce {
            op: reduce_op,
            input,
            output,
            axis: axis as i64,
        });
    }

    // 2-input patterns require at least 2 inputs
    if num_inputs < 2 {
        return Err(AnalysisError::UnsupportedPattern(
            "expected at least 1 input storage buffer".into(),
        ));
    }

    // 3+ inputs → Normalization (input + scale + bias)
    if num_inputs >= 3 {
        let input = make_binding(module, inputs[0].0, inputs[0].1, TensorRole::Input);
        let scale = make_binding(module, inputs[1].0, inputs[1].1, TensorRole::Input);
        let bias = make_binding(module, inputs[2].0, inputs[2].1, TensorRole::Input);
        let output = make_binding(module, outputs[0].0, outputs[0].1, TensorRole::Output);
        return Ok(KernelPattern::Normalization {
            input,
            scale,
            bias,
            output,
            epsilon: 1e-5,
        });
    }

    // 2 inputs
    let input_a = make_binding(module, inputs[0].0, inputs[0].1, TensorRole::Input);
    let input_b = make_binding(module, inputs[1].0, inputs[1].1, TensorRole::Input);
    let output_c = make_binding(module, outputs[0].0, outputs[0].1, TensorRole::Output);

    if !has_loop {
        // ElementWise: store of binary operation.
        let op =
            find_store_binary_op(&ep.function.body, &ep.function.expressions).ok_or_else(|| {
                AnalysisError::UnsupportedPattern("no recognizable binary operation found".into())
            })?;

        let dim_name = shape_names.first().cloned().unwrap_or_else(|| "N".into());

        return Ok(KernelPattern::ElementWise {
            op,
            inputs: [input_a, input_b],
            output: output_c,
            dim_name,
        });
    }

    // 2 inputs + loop: Conv2D (many params) vs MatMul (3 params)
    if shape_names.len() > 3 {
        let conv_shape = extract_conv2d_shape(&shape_names);
        return Ok(KernelPattern::Conv2D {
            input: input_a,
            weight: input_b,
            output: output_c,
            shape: conv_shape,
        });
    }

    // MatMul: loop + accumulation pattern.
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
}

/// Extract Conv2D shape from param names.
fn extract_conv2d_shape(shape_names: &[String]) -> Conv2DShape {
    // Convention: params struct has N, IC, IH, IW, OC, OH, OW, KH, KW, ...
    let get = |i: usize| {
        shape_names
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("d{i}"))
    };
    Conv2DShape {
        batch: get(0),
        channels_in: get(1),
        height: get(2),
        width: get(3),
        channels_out: get(4),
        kernel_h: get(5),
        kernel_w: get(6),
        stride_h: 1,
        stride_w: 1,
        pad_h: 0,
        pad_w: 0,
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

/// Search for a Store whose value is a Math expression, detecting activation type.
fn find_store_activation(body: &[Statement], exprs: &Arena<Expression>) -> Option<ActivationOp> {
    for stmt in body {
        match stmt {
            Statement::Store { value, .. } => {
                if let Some(act) = classify_activation_expr(exprs, *value) {
                    return Some(act);
                }
            }
            Statement::If { accept, reject, .. } => {
                if let Some(op) = find_store_activation(accept, exprs) {
                    return Some(op);
                }
                if let Some(op) = find_store_activation(reject, exprs) {
                    return Some(op);
                }
            }
            _ => {}
        }
    }
    None
}

/// Classify an expression as an activation function.
fn classify_activation_expr(
    exprs: &Arena<Expression>,
    handle: Handle<Expression>,
) -> Option<ActivationOp> {
    match exprs.try_get(handle)? {
        // max(x, 0) → ReLU
        Expression::Math {
            fun: MathFunction::Max,
            ..
        } => Some(ActivationOp::Relu),
        // tanh(x) → Tanh
        Expression::Math {
            fun: MathFunction::Tanh,
            ..
        } => Some(ActivationOp::Tanh),
        // 1/(1+exp(-x)) → Sigmoid: detected as Divide whose right is Add(1, Exp(Negate(x)))
        Expression::Binary {
            op: BinaryOp::Divide,
            right,
            ..
        } => {
            // Check if right side contains an Exp
            if contains_math_fun(exprs, *right, MathFunction::Exp) {
                Some(ActivationOp::Sigmoid)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Check if an expression (recursively) contains a specific math function.
fn contains_math_fun(
    exprs: &Arena<Expression>,
    handle: Handle<Expression>,
    target: MathFunction,
) -> bool {
    match exprs.try_get(handle) {
        Some(Expression::Math { fun, arg, .. }) => {
            *fun == target || contains_math_fun(exprs, *arg, target)
        }
        Some(Expression::Binary { left, right, .. }) => {
            contains_math_fun(exprs, *left, target) || contains_math_fun(exprs, *right, target)
        }
        Some(Expression::Unary { expr, .. }) => contains_math_fun(exprs, *expr, target),
        _ => false,
    }
}

/// Check if a block contains a Store whose value (or sub-expr) uses a specific Math function.
fn find_store_math_fun(
    body: &[Statement],
    exprs: &Arena<Expression>,
    target: MathFunction,
) -> bool {
    for stmt in body {
        match stmt {
            Statement::Store { value, .. } => {
                if contains_math_fun(exprs, *value, target) {
                    return true;
                }
            }
            Statement::If { accept, reject, .. } => {
                if find_store_math_fun(accept, exprs, target)
                    || find_store_math_fun(reject, exprs, target)
                {
                    return true;
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                if find_store_math_fun(body, exprs, target)
                    || find_store_math_fun(continuing, exprs, target)
                {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// Detect the reduce operation type from the loop body.
fn detect_reduce_op(body: &[Statement], exprs: &Arena<Expression>) -> ReduceOp {
    if find_store_math_fun(body, exprs, MathFunction::Max) {
        ReduceOp::Max
    } else if find_store_math_fun(body, exprs, MathFunction::Min) {
        ReduceOp::Min
    } else {
        // Default: sum accumulation (binary add in loop)
        ReduceOp::Sum
    }
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

    fn make_activation_module(math_fun: MathFunction) -> Module {
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

        // Single input
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
        // Single output
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

        let mut func = Function::new("main");
        let arg = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        let arg1 = func
            .expressions
            .append(Expression::Literal(Literal::F32(0.0)));
        let math = func.expressions.append(Expression::Math {
            fun: math_fun,
            arg,
            arg1: Some(arg1),
            arg2: None,
            arg3: None,
        });
        let ptr = func
            .expressions
            .append(Expression::Literal(Literal::F32(0.0)));
        func.body.push(Statement::Store {
            pointer: ptr,
            value: math,
        });

        module.entry_points.push(EntryPoint {
            name: "activation".into(),
            workgroup_size: [256, 1, 1],
            function: func,
        });

        module
    }

    fn make_reduce_module() -> Module {
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

        let mut func = Function::new("main");
        // A loop with a binary add → reduce sum
        let left = func
            .expressions
            .append(Expression::Literal(Literal::F32(0.0)));
        let right = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        let add = func.expressions.append(Expression::Binary {
            op: BinaryOp::Add,
            left,
            right,
        });
        let ptr = func
            .expressions
            .append(Expression::Literal(Literal::F32(0.0)));
        func.body.push(Statement::Loop {
            body: vec![
                Statement::Store {
                    pointer: ptr,
                    value: add,
                },
                Statement::Break,
            ],
            continuing: vec![],
            break_if: None,
        });

        module.entry_points.push(EntryPoint {
            name: "reduce".into(),
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
    fn classify_activation_relu() {
        let module = make_activation_module(MathFunction::Max);
        let pattern = classify_entry_point(&module, 0).unwrap();
        match &pattern {
            KernelPattern::Activation { op, .. } => {
                assert_eq!(*op, ActivationOp::Relu);
            }
            _ => panic!("expected Activation pattern, got {pattern:?}"),
        }
    }

    #[test]
    fn classify_activation_tanh() {
        let module = make_activation_module(MathFunction::Tanh);
        let pattern = classify_entry_point(&module, 0).unwrap();
        match &pattern {
            KernelPattern::Activation { op, .. } => {
                assert_eq!(*op, ActivationOp::Tanh);
            }
            _ => panic!("expected Activation pattern, got {pattern:?}"),
        }
    }

    #[test]
    fn classify_reduce_sum() {
        let module = make_reduce_module();
        let pattern = classify_entry_point(&module, 0).unwrap();
        match &pattern {
            KernelPattern::Reduce { op, .. } => {
                assert_eq!(*op, ReduceOp::Sum);
            }
            _ => panic!("expected Reduce pattern, got {pattern:?}"),
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
