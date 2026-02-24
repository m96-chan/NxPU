//! IR pattern classification for ONNX lowering.
//!
//! Analyzes an entry point's global variables and function body to classify
//! the computation into a known ONNX-mappable pattern.

use std::fmt;

use nxpu_ir::{
    AddressSpace, Arena, ArraySize, BinaryOp, Expression, GlobalVariable, Handle, Literal,
    MathFunction, Module, Scalar, ScalarKind, Statement, StorageAccess, Type, TypeInner,
    UniqueArena,
};

/// ONNX-compatible data type constants (matches TensorProto.DataType).
pub mod data_type {
    pub const FLOAT: i32 = 1;
    pub const UINT8: i32 = 2;
    pub const INT8: i32 = 3;
    pub const INT32: i32 = 6;
    pub const INT64: i32 = 7;
    pub const BOOL: i32 = 9;
    pub const FLOAT16: i32 = 10;
    pub const UINT32: i32 = 12;
    pub const BFLOAT16: i32 = 16;
}

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

impl fmt::Display for TensorRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Input => "Input",
            Self::Output => "Output",
        })
    }
}

/// A storage buffer bound as a tensor.
#[derive(Debug, Clone)]
pub struct TensorBinding {
    /// Handle to the underlying global variable.
    pub handle: Handle<GlobalVariable>,
    /// Human-readable tensor name.
    pub name: String,
    /// ONNX element data type (see [`data_type`]).
    pub elem_type: i32,
    /// Whether this tensor is an input or output.
    pub role: TensorRole,
}

impl fmt::Display for TensorBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.name, self.role)
    }
}

/// Symbolic dimension names for matrix multiplication.
#[derive(Debug, Clone)]
pub struct MatMulShape {
    /// Number of rows in the left matrix.
    pub m: String,
    /// Number of columns in the right matrix.
    pub n: String,
    /// Shared inner dimension.
    pub k: String,
}

impl fmt::Display for MatMulShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MatMul({}x{} * {}x{})", self.m, self.k, self.k, self.n)
    }
}

/// Element-wise binary operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementWiseOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl fmt::Display for ElementWiseOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.op_name())
    }
}

impl ElementWiseOp {
    /// Returns the canonical operator name (e.g. "Add", "Relu", "ReduceSum").
    pub fn op_name(self) -> &'static str {
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
    Gelu,
    Silu,
    Mish,
}

impl fmt::Display for ActivationOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.op_name())
    }
}

impl ActivationOp {
    /// Returns the canonical operator name (e.g. "Relu", "Sigmoid").
    pub fn op_name(self) -> &'static str {
        match self {
            Self::Relu => "Relu",
            Self::Sigmoid => "Sigmoid",
            Self::Tanh => "Tanh",
            Self::Softmax => "Softmax",
            Self::Gelu => "Gelu",
            Self::Silu => "Silu",
            Self::Mish => "Mish",
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

impl fmt::Display for ReduceOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.op_name())
    }
}

impl ReduceOp {
    /// Returns the canonical operator name (e.g. "ReduceSum", "ReduceMean").
    pub fn op_name(self) -> &'static str {
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

impl fmt::Display for PoolKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.op_name())
    }
}

impl PoolKind {
    /// Returns the canonical operator name (e.g. "MaxPool", "AveragePool").
    pub fn op_name(self) -> &'static str {
        match self {
            Self::Max => "MaxPool",
            Self::Avg => "AveragePool",
        }
    }
}

/// Normalization type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    Batch,
    Layer,
}

impl fmt::Display for NormType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Batch => "Batch",
            Self::Layer => "Layer",
        })
    }
}

/// Conv2D shape parameters extracted from uniform params.
#[derive(Debug, Clone)]
pub struct Conv2DShape {
    /// Batch dimension name.
    pub batch: String,
    /// Input channel dimension name.
    pub channels_in: String,
    /// Output channel dimension name.
    pub channels_out: String,
    /// Spatial height dimension name.
    pub height: String,
    /// Spatial width dimension name.
    pub width: String,
    /// Kernel height dimension name.
    pub kernel_h: String,
    /// Kernel width dimension name.
    pub kernel_w: String,
    /// Kernel height as a concrete value.
    pub kernel_h_val: i64,
    /// Kernel width as a concrete value.
    pub kernel_w_val: i64,
    /// Vertical stride.
    pub stride_h: i64,
    /// Horizontal stride.
    pub stride_w: i64,
    /// Vertical padding.
    pub pad_h: i64,
    /// Horizontal padding.
    pub pad_w: i64,
    /// Number of groups (1 = standard conv, channels_in = depthwise).
    pub groups: i64,
    /// Vertical dilation.
    pub dilation_h: i64,
    /// Horizontal dilation.
    pub dilation_w: i64,
}

impl fmt::Display for Conv2DShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Conv2D({}x{}x{} k{}x{} g{} d{}x{})",
            self.channels_in,
            self.height,
            self.width,
            self.kernel_h,
            self.kernel_w,
            self.groups,
            self.dilation_h,
            self.dilation_w
        )
    }
}

/// Pooling shape parameters.
#[derive(Debug, Clone)]
pub struct PoolShape {
    /// Kernel height.
    pub kernel_h: i64,
    /// Kernel width.
    pub kernel_w: i64,
    /// Vertical stride.
    pub stride_h: i64,
    /// Horizontal stride.
    pub stride_w: i64,
}

impl fmt::Display for PoolShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Pool(k{}x{} s{}x{})",
            self.kernel_h, self.kernel_w, self.stride_h, self.stride_w
        )
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
        norm_type: NormType,
    },
    /// Concatenation of multiple inputs along an axis.
    Concat {
        inputs: Vec<TensorBinding>,
        output: TensorBinding,
        axis: i64,
    },
    /// Split a single input into multiple outputs along an axis.
    Split {
        input: TensorBinding,
        outputs: Vec<TensorBinding>,
        axis: i64,
    },
    /// Scaled dot-product attention.
    Attention {
        query: TensorBinding,
        key: TensorBinding,
        value: TensorBinding,
        output: TensorBinding,
        d_k: String,
        seq_len: String,
        /// Number of attention heads (1 = single-head).
        num_heads: u32,
        /// Number of K/V heads (for GQA; equals num_heads for MHA).
        num_kv_heads: u32,
        /// Whether a causal mask is applied.
        causal: bool,
    },
    /// Gather: index into data tensor using indices.
    Gather {
        data: TensorBinding,
        indices: TensorBinding,
        output: TensorBinding,
        axis: i64,
    },
    /// Scatter: write updates into output tensor at given indices.
    Scatter {
        data: TensorBinding,
        indices: TensorBinding,
        updates: TensorBinding,
        output: TensorBinding,
        axis: i64,
    },
    /// Unrecognized pattern — classification could not determine a known op.
    Unknown { reason: String },
}

impl fmt::Display for KernelPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MatMul { shape, .. } => write!(f, "{shape}"),
            Self::ElementWise { op, .. } => write!(f, "{op}"),
            Self::Conv2D { shape, .. } => write!(f, "{shape}"),
            Self::Pool { kind, shape, .. } => {
                write!(
                    f,
                    "{}(k{}x{} s{}x{})",
                    kind, shape.kernel_h, shape.kernel_w, shape.stride_h, shape.stride_w
                )
            }
            Self::Activation { op, .. } => write!(f, "{op}"),
            Self::Reduce { op, axis, .. } => write!(f, "{op}(axis={axis})"),
            Self::Transpose { perm, .. } => write!(f, "Transpose({perm:?})"),
            Self::Reshape { .. } => f.write_str("Reshape"),
            Self::Normalization {
                epsilon, norm_type, ..
            } => write!(f, "{norm_type}Normalization(eps={epsilon})"),
            Self::Concat { axis, .. } => write!(f, "Concat(axis={axis})"),
            Self::Split { axis, .. } => write!(f, "Split(axis={axis})"),
            Self::Attention {
                d_k,
                seq_len,
                num_heads,
                causal,
                ..
            } => {
                let mask = if *causal { ", causal" } else { "" };
                write!(
                    f,
                    "Attention(d_k={d_k}, seq_len={seq_len}, heads={num_heads}{mask})"
                )
            }
            Self::Gather { axis, .. } => write!(f, "Gather(axis={axis})"),
            Self::Scatter { axis, .. } => write!(f, "Scatter(axis={axis})"),
            Self::Unknown { reason } => write!(f, "Unknown({reason})"),
        }
    }
}

/// Embedded constant weight data extracted from GlobalVariable initializers.
#[derive(Debug, Clone)]
pub struct EmbeddedWeight {
    /// Tensor name (from the global variable name).
    pub name: String,
    /// Tensor dimensions (e.g. `[4]` for `array<f32, 4>`).
    pub dims: Vec<i64>,
    /// Flattened f32 data.
    pub data: Vec<f32>,
}

/// Extract constant weight data from module globals with initializers.
///
/// Scans all global variables with `init: Some(...)` and evaluates
/// Compose/Literal/ZeroValue expressions into flat f32 arrays.
pub fn extract_embedded_weights(module: &Module) -> Vec<EmbeddedWeight> {
    let mut weights = Vec::new();
    for (handle, gv) in module.global_variables.iter() {
        let Some(init_handle) = gv.init else {
            continue;
        };
        let name = gv
            .name
            .clone()
            .unwrap_or_else(|| format!("weight_{}", handle.index()));
        let dims = type_dims(gv.ty, &module.types);
        let Some(data) =
            eval_const_expr_f32(init_handle, &module.global_expressions, &module.types)
        else {
            continue;
        };
        weights.push(EmbeddedWeight { name, dims, data });
    }
    weights
}

/// Recursively evaluate a constant expression to flat f32 values.
fn eval_const_expr_f32(
    handle: Handle<Expression>,
    exprs: &Arena<Expression>,
    types: &UniqueArena<Type>,
) -> Option<Vec<f32>> {
    match exprs.try_get(handle)? {
        Expression::Literal(Literal::F32(v)) => Some(vec![*v]),
        Expression::Literal(Literal::I32(v)) => Some(vec![*v as f32]),
        Expression::Literal(Literal::U32(v)) => Some(vec![*v as f32]),
        Expression::Compose { components, .. } => {
            let mut result = Vec::new();
            for &comp in components {
                result.extend(eval_const_expr_f32(comp, exprs, types)?);
            }
            Some(result)
        }
        Expression::ZeroValue(ty) => {
            let dims = type_dims(*ty, types);
            let count = dims.iter().product::<i64>().max(1) as usize;
            Some(vec![0.0; count])
        }
        _ => None,
    }
}

/// Compute flattened dimensions from a Type (Array, Scalar, Vector).
fn type_dims(ty_handle: Handle<Type>, types: &UniqueArena<Type>) -> Vec<i64> {
    match &types[ty_handle].inner {
        TypeInner::Scalar(_) => vec![],
        TypeInner::Vector { size, .. } => vec![*size as i64],
        TypeInner::Array {
            size: ArraySize::Constant(n),
            ..
        } => vec![*n as i64],
        _ => vec![],
    }
}

/// Detect number of attention heads from shape_names or literal divisions.
fn detect_num_heads(shape_names: &[String], exprs: &Arena<Expression>) -> u32 {
    // Look for param named "num_heads", "n_head", "H", "n_heads", "nhead"
    for name in shape_names.iter() {
        let lower = name.to_lowercase();
        if lower == "num_heads" || lower == "n_head" || lower == "n_heads" || lower == "nhead" {
            // Can't get runtime value, but if there's a matching literal, use it
            // Otherwise return 1 as we know it's multi-head but can't determine count
            return find_division_literal(exprs).unwrap_or(1);
        }
    }
    // Also check for "H" (common in transformer code)
    if shape_names.iter().any(|n| n == "H") {
        return find_division_literal(exprs).unwrap_or(1);
    }
    1
}

/// Look for a division by a literal (d_model / num_heads -> head_dim).
#[allow(clippy::collapsible_if)] // nested if-let for MSRV 1.87 compat (no let chains)
fn find_division_literal(exprs: &Arena<Expression>) -> Option<u32> {
    for (_, expr) in exprs.iter() {
        if let Expression::Binary {
            op: BinaryOp::Divide,
            right,
            ..
        } = expr
        {
            if let Some(Expression::Literal(Literal::U32(n))) = exprs.try_get(*right) {
                if *n > 1 {
                    return Some(*n);
                }
            }
        }
    }
    None
}

/// Detect causal attention mask: look for an If guarding assignment of a large
/// negative literal inside a loop. The pattern is:
///   `if (j > i) { score = -1e30; }` or similar.
///
/// We specifically look for an `If` whose accept or reject block contains (or
/// leads to) a Store of a large negative float literal. This avoids false
/// positives from for-loop break conditions and numerically-stable softmax
/// `max_score` initializations that also use large negative values but are
/// not inside an If-guarded store.
fn detect_causal_mask(body: &[Statement], exprs: &Arena<Expression>) -> bool {
    has_causal_if_in_loop(body, exprs)
}

/// Recursively search loops for an `If` whose condition is a Greater/Less
/// comparison and whose accept or reject branch stores a large negative
/// float literal (like -1e30, -1e9, or -inf).
#[allow(clippy::collapsible_if)] // nested if-let for MSRV 1.87 compat (no let chains)
fn has_causal_if_in_loop(body: &[Statement], exprs: &Arena<Expression>) -> bool {
    for stmt in body {
        if let Statement::Loop {
            body, continuing, ..
        } = stmt
        {
            // Search loop body; skip the first non-Emit If that acts as a
            // for-loop break guard.
            let mut saw_non_emit = false;
            for s in body.iter() {
                match s {
                    Statement::Emit(_) => {}
                    Statement::If {
                        condition,
                        accept,
                        reject,
                    } => {
                        // The first non-emit If in a naga for-loop is the break
                        // guard (e.g. `if (j < N) {} else { break; }`). Skip it.
                        if !saw_non_emit
                            && (matches!(accept.as_slice(), [Statement::Break])
                                || matches!(reject.as_slice(), [Statement::Break]))
                        {
                            saw_non_emit = true;
                            continue;
                        }
                        saw_non_emit = true;
                        if is_causal_guard(*condition, accept, reject, exprs) {
                            return true;
                        }
                    }
                    _ => {
                        saw_non_emit = true;
                    }
                }
            }
            // Also search the continuing block
            for s in continuing.iter() {
                if let Statement::If {
                    condition,
                    accept,
                    reject,
                } = s
                {
                    if is_causal_guard(*condition, accept, reject, exprs) {
                        return true;
                    }
                }
            }
            // Recurse into nested loops
            if has_causal_if_in_loop(body, exprs) || has_causal_if_in_loop(continuing, exprs) {
                return true;
            }
        }
    }
    false
}

/// Check if a conditional is a comparison (Greater/Less/GE/LE) and one of
/// its branches (accept or reject) contains a Store of a large negative float.
fn is_causal_guard(
    condition: Handle<Expression>,
    accept: &[Statement],
    reject: &[Statement],
    exprs: &Arena<Expression>,
) -> bool {
    let is_comparison = match exprs.try_get(condition) {
        Some(Expression::Binary { op, .. }) => matches!(
            op,
            BinaryOp::Greater | BinaryOp::Less | BinaryOp::GreaterEqual | BinaryOp::LessEqual
        ),
        _ => false,
    };
    if !is_comparison {
        return false;
    }
    block_stores_large_negative(accept, exprs) || block_stores_large_negative(reject, exprs)
}

/// Check if a block contains a Store whose value is a large negative float literal.
fn block_stores_large_negative(body: &[Statement], exprs: &Arena<Expression>) -> bool {
    for stmt in body {
        match stmt {
            Statement::Store { value, .. } => {
                if expr_is_large_negative(*value, exprs) {
                    return true;
                }
            }
            Statement::If { accept, reject, .. } => {
                if block_stores_large_negative(accept, exprs)
                    || block_stores_large_negative(reject, exprs)
                {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// Check if an expression is (or contains) a large negative float literal (< -1e6).
fn expr_is_large_negative(handle: Handle<Expression>, exprs: &Arena<Expression>) -> bool {
    match exprs.try_get(handle) {
        Some(Expression::Literal(Literal::F32(v))) => *v < -1e6,
        Some(Expression::Unary {
            op: nxpu_ir::UnaryOp::Negate,
            expr,
        }) => {
            // -literal
            matches!(exprs.try_get(*expr), Some(Expression::Literal(Literal::F32(v))) if *v > 1e6)
        }
        _ => false,
    }
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
    let mut inputs: Vec<(Handle<GlobalVariable>, &GlobalVariable)> = Vec::new();
    let mut outputs: Vec<(Handle<GlobalVariable>, &GlobalVariable)> = Vec::new();
    let mut init_globals: Vec<(Handle<GlobalVariable>, &GlobalVariable)> = Vec::new();
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
            _ => {
                if gv.init.is_some() {
                    init_globals.push((handle, gv));
                }
            }
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

        // Split: 1 input + 2+ outputs + has If
        if outputs.len() >= 2 && has_if_statement(&ep.function.body) {
            let out_bindings: Vec<TensorBinding> = outputs
                .iter()
                .map(|(h, gv)| make_binding(module, *h, gv, TensorRole::Output))
                .collect();
            let axis = infer_split_axis(&ep.function.body, &ep.function.expressions, &shape_names);
            return Ok(KernelPattern::Split {
                input,
                outputs: out_bindings,
                axis,
            });
        }

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

            // ElementWise with embedded weight: 1 storage input + binary op + private init global.
            let ew_op = find_store_binary_op(&ep.function.body, &ep.function.expressions);
            if let (Some(ew_op), Some(&(wh, wgv))) = (ew_op, init_globals.first()) {
                let weight = make_binding(module, wh, wgv, TensorRole::Input);
                let dim_name = shape_names.first().cloned().unwrap_or_else(|| "N".into());
                return Ok(KernelPattern::ElementWise {
                    op: ew_op,
                    inputs: [input, weight],
                    output,
                    dim_name,
                });
            }

            // No recognized activation — unknown pattern.
            return Ok(KernelPattern::Unknown {
                reason: "single input, no loop, no recognized activation function".into(),
            });
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
            let bounds = extract_loop_bound_literals(&ep.function.body, &ep.function.expressions);
            let strides = extract_multiply_literals(&ep.function.expressions);
            let kh_u = bounds.first().copied().unwrap_or(2);
            let kw_u = bounds.get(1).copied().unwrap_or(kh_u);
            let sh_u = strides.first().copied().unwrap_or(kh_u);
            let sw_u = strides.get(1).copied().unwrap_or(sh_u);
            let kh = kh_u as i64;
            let kw = kw_u as i64;
            let sh = sh_u as i64;
            let sw = sw_u as i64;
            let pool_shape = PoolShape {
                kernel_h: kh,
                kernel_w: kw,
                stride_h: sh,
                stride_w: sw,
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

    // 3+ inputs: check for Attention, Normalization, Scatter
    if num_inputs >= 3 {
        // Attention heuristic: 3 inputs + has loop + contains Exp + Sqrt.
        // NOTE: This is fragile — any 3-input kernel with loop + exp() + sqrt()
        // will match. A false positive is possible for custom kernels that
        // happen to use both functions. Consider adding more structural checks
        // (e.g., nested loop depth, softmax pattern) if this becomes an issue.
        if has_loop
            && has_math_function_in_expressions(&ep.function.expressions, MathFunction::Exp)
            && has_math_function_in_expressions(&ep.function.expressions, MathFunction::Sqrt)
        {
            let query = make_binding(module, inputs[0].0, inputs[0].1, TensorRole::Input);
            let key = make_binding(module, inputs[1].0, inputs[1].1, TensorRole::Input);
            let value = make_binding(module, inputs[2].0, inputs[2].1, TensorRole::Input);
            let output = make_binding(module, outputs[0].0, outputs[0].1, TensorRole::Output);
            let seq_len = shape_names
                .first()
                .cloned()
                .unwrap_or_else(|| "seq_len".into());
            let d_k = shape_names.get(1).cloned().unwrap_or_else(|| "d_k".into());
            let num_heads = detect_num_heads(&shape_names, &ep.function.expressions);
            let causal = detect_causal_mask(&ep.function.body, &ep.function.expressions);
            return Ok(KernelPattern::Attention {
                query,
                key,
                value,
                output,
                d_k,
                seq_len,
                num_heads,
                num_kv_heads: num_heads, // Default: MHA (same as num_heads)
                causal,
            });
        }

        // Scatter: 3 inputs + 1 output + no loop (simple scatter write)
        if num_inputs == 3 && outputs.len() == 1 && !has_loop {
            let data = make_binding(module, inputs[0].0, inputs[0].1, TensorRole::Input);
            let indices = make_binding(module, inputs[1].0, inputs[1].1, TensorRole::Input);
            let updates = make_binding(module, inputs[2].0, inputs[2].1, TensorRole::Input);
            let output = make_binding(module, outputs[0].0, outputs[0].1, TensorRole::Output);
            return Ok(KernelPattern::Scatter {
                data,
                indices,
                updates,
                output,
                axis: 0,
            });
        }

        // 3+ inputs but no recognized attention/scatter pattern — unknown.
        return Ok(KernelPattern::Unknown {
            reason: "3+ inputs but no recognized pattern (expected Attention or Scatter)".into(),
        });
    }

    // 2 inputs
    let input_a = make_binding(module, inputs[0].0, inputs[0].1, TensorRole::Input);
    let input_b = make_binding(module, inputs[1].0, inputs[1].1, TensorRole::Input);
    let output_c = make_binding(module, outputs[0].0, outputs[0].1, TensorRole::Output);

    if !has_loop {
        let has_if = has_if_statement(&ep.function.body);

        // Gather: 2 inputs + no loop + one input is u32 array (indices)
        // Check if second input is a u32 array (integer index type).
        let second_is_int = input_b.elem_type == data_type::UINT32
            || input_b.elem_type == data_type::INT32
            || input_b.elem_type == data_type::INT64;
        if second_is_int
            && find_store_binary_op(&ep.function.body, &ep.function.expressions).is_none()
            && !has_if
        {
            return Ok(KernelPattern::Gather {
                data: input_a,
                indices: input_b,
                output: output_c,
                axis: 0,
            });
        }

        // Concat: 2 inputs + no loop + has If + no binary store op
        if has_if && find_store_binary_op(&ep.function.body, &ep.function.expressions).is_none() {
            let axis = infer_concat_axis(&ep.function.body, &ep.function.expressions, &shape_names);
            return Ok(KernelPattern::Concat {
                inputs: vec![input_a, input_b],
                output: output_c,
                axis,
            });
        }

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
        let conv_shape = extract_conv2d_shape(
            &shape_names,
            Some(&ep.function.body),
            Some(&ep.function.expressions),
        );
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

/// Extract literal U32 values from loop break conditions.
///
/// For `for (kh = 0; kh < N; ...)`, naga generates `break_if: kh >= N`.
/// When N is `Literal(U32(n))`, we collect n as a loop bound.
fn extract_loop_bound_literals(body: &[Statement], exprs: &Arena<Expression>) -> Vec<u32> {
    let mut bounds = Vec::new();
    for stmt in body {
        match stmt {
            Statement::Loop {
                body,
                continuing,
                break_if,
            } => {
                // Check this loop's break condition for a literal bound.
                // Pattern: break_if is `Binary { GreaterEqual|Greater, _, Literal(U32(n)) }`.
                if let Some(n) = break_if.and_then(|bi| {
                    let Expression::Binary {
                        op: BinaryOp::GreaterEqual | BinaryOp::Greater,
                        right,
                        ..
                    } = exprs.try_get(bi)?
                    else {
                        return None;
                    };
                    match exprs.try_get(*right)? {
                        Expression::Literal(Literal::U32(n)) => Some(*n),
                        _ => None,
                    }
                }) {
                    bounds.push(n);
                }
                // naga 28 pattern: for-loop condition lowered as
                //   Emit; ...; If(cond) { } else { Break }  at the start of the loop body,
                // where cond is `Binary { Less|LessEqual, _, Literal(U32(n)) }`.
                // Skip leading Emit statements to find the If.
                if let Some(Statement::If {
                    condition,
                    accept,
                    reject,
                }) = body.iter().find(|s| !matches!(s, Statement::Emit(_)))
                {
                    let rhs = match exprs.try_get(*condition) {
                        Some(&Expression::Binary {
                            op: BinaryOp::Less | BinaryOp::LessEqual,
                            right,
                            ..
                        }) if accept.is_empty()
                            && matches!(reject.as_slice(), [Statement::Break]) =>
                        {
                            Some(right)
                        }
                        Some(&Expression::Binary {
                            op: BinaryOp::GreaterEqual | BinaryOp::Greater,
                            right,
                            ..
                        }) if reject.is_empty()
                            && matches!(accept.as_slice(), [Statement::Break]) =>
                        {
                            Some(right)
                        }
                        _ => None,
                    };
                    let bound = rhs.and_then(|h| exprs.try_get(h)).and_then(|e| match *e {
                        Expression::Literal(Literal::U32(n)) => Some(n),
                        _ => None,
                    });
                    if let Some(n) = bound {
                        bounds.push(n);
                    }
                }
                // Recurse into nested loops.
                bounds.extend(extract_loop_bound_literals(body, exprs));
                bounds.extend(extract_loop_bound_literals(continuing, exprs));
            }
            Statement::If { accept, reject, .. } => {
                bounds.extend(extract_loop_bound_literals(accept, exprs));
                bounds.extend(extract_loop_bound_literals(reject, exprs));
            }
            _ => {}
        }
    }
    bounds
}

/// Scan the expression arena for `Binary(Multiply, _, Literal(U32(n)))` where n > 1.
/// These represent stride factors in index computations like `oh * 2u + kh`.
fn extract_multiply_literals(exprs: &Arena<Expression>) -> Vec<u32> {
    let mut strides = Vec::new();
    for (_, expr) in exprs.iter() {
        let Expression::Binary {
            op: BinaryOp::Multiply,
            left,
            right,
        } = expr
        else {
            continue;
        };
        if let Some(&Expression::Literal(Literal::U32(n @ 2..))) = exprs.try_get(*right) {
            strides.push(n);
        } else if let Some(&Expression::Literal(Literal::U32(n @ 2..))) = exprs.try_get(*left) {
            strides.push(n);
        }
    }
    strides.sort_unstable();
    strides.dedup();
    strides
}

/// Scan the expression arena for `Binary(Subtract, _, Literal(U32(n)))` where n > 0.
/// These represent padding offsets in index computations like `oh * 2u + kh - 1u`.
fn extract_subtract_literals(exprs: &Arena<Expression>) -> Vec<u32> {
    let mut pads = Vec::new();
    for (_, expr) in exprs.iter() {
        let Expression::Binary {
            op: BinaryOp::Subtract,
            right,
            ..
        } = expr
        else {
            continue;
        };
        if let Some(&Expression::Literal(Literal::U32(n @ 1..))) = exprs.try_get(*right) {
            pads.push(n);
        }
    }
    pads.sort_unstable();
    pads.dedup();
    pads
}

/// Extract Conv2D shape from param names and (optionally) the function body/expressions.
///
/// When body and expressions are provided, attempts to extract kernel size from
/// loop bound literals, stride from multiplication factors, and padding from
/// subtraction offsets. Falls back to 0 (unknown) for values that cannot be
/// determined at compile time.
fn extract_conv2d_shape(
    shape_names: &[String],
    body: Option<&[Statement]>,
    exprs: Option<&Arena<Expression>>,
) -> Conv2DShape {
    // Convention: params struct has N, IC, IH, IW, OC, KH, KW, ...
    let get = |i: usize| {
        shape_names
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("d{i}"))
    };

    let (kernel_h_val, kernel_w_val, stride_h, stride_w, pad_h, pad_w) =
        if let (Some(body), Some(exprs)) = (body, exprs) {
            let bounds = extract_loop_bound_literals(body, exprs);
            let strides = extract_multiply_literals(exprs);
            let pads = extract_subtract_literals(exprs);

            // Innermost loop bounds are kernel sizes (e.g., kh < 5u → kernel_h = 5).
            let kh = bounds.first().copied().unwrap_or(0) as i64;
            let kw = bounds.get(1).copied().unwrap_or(0) as i64;
            // If only one bound found, assume square kernel.
            let kw = if kw == 0 && kh > 0 { kh } else { kw };

            // Stride from multiplication factors (e.g., oh * 2u → stride = 2).
            let sh_u = strides.first().copied().unwrap_or(1);
            let sw_u = strides.get(1).copied().unwrap_or(sh_u);
            let sh = sh_u as i64;
            let sw = sw_u as i64;

            // Padding from subtraction offsets (e.g., ih - 1u → pad = 1).
            let ph_u = pads.first().copied().unwrap_or(0);
            let pw_u = pads.get(1).copied().unwrap_or(ph_u);
            let ph = ph_u as i64;
            let pw = pw_u as i64;

            (kh, kw, sh, sw, ph, pw)
        } else {
            (0, 0, 1, 1, 0, 0)
        };

    // Detect groups parameter: if shape_names contains "groups", use it.
    let groups = if shape_names.iter().any(|n| n.eq_ignore_ascii_case("groups")) {
        // Depthwise convention: groups == channels_in
        -1 // Sentinel: resolved at runtime or marked as depthwise
    } else {
        1
    };

    Conv2DShape {
        batch: get(0),
        channels_in: get(1),
        height: get(2),
        width: get(3),
        channels_out: get(4),
        kernel_h: get(5),
        kernel_w: get(6),
        kernel_h_val,
        kernel_w_val,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        groups,
        dilation_h: 1,
        dilation_w: 1,
    }
}

fn make_binding(
    module: &Module,
    handle: Handle<GlobalVariable>,
    gv: &GlobalVariable,
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
fn resolve_array_elem_type(module: &Module, ty: Handle<Type>) -> Option<i32> {
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

/// Normalize a potentially negative axis to a positive axis.
///
/// For example, `normalize_axis(-1, 4)` returns `3`.
pub fn normalize_axis(axis: i64, ndim: usize) -> i64 {
    if axis < 0 {
        let ndim = ndim as i64;
        ((axis % ndim) + ndim) % ndim
    } else {
        axis
    }
}

/// Infer the concat axis from the If condition in the function body.
///
/// In WGSL concat patterns, the If condition checks whether the linear index
/// falls before or after a boundary (e.g., `if idx < params.C1`). The boundary
/// param name maps to a position in the params struct, which gives us the axis.
fn infer_concat_axis(body: &[Statement], exprs: &Arena<Expression>, shape_names: &[String]) -> i64 {
    find_if_comparison_axis(body, exprs, shape_names).unwrap_or(0)
}

/// Infer the split axis from the If condition in the function body.
///
/// Works identically to [`infer_concat_axis`]: the If condition compares
/// against a params struct member whose index indicates the split axis.
fn infer_split_axis(body: &[Statement], exprs: &Arena<Expression>, shape_names: &[String]) -> i64 {
    find_if_comparison_axis(body, exprs, shape_names).unwrap_or(0)
}

/// Walk the statement tree to find an If whose condition is a comparison
/// (Less, LessEqual, Greater, GreaterEqual) where one operand is an
/// `AccessIndex` into the uniform params struct. The `index` field of
/// `AccessIndex` gives the struct member position, which corresponds to
/// the concat/split axis.
fn find_if_comparison_axis(
    body: &[Statement],
    exprs: &Arena<Expression>,
    _shape_names: &[String],
) -> Option<i64> {
    for stmt in body {
        match stmt {
            Statement::If { condition, .. } => {
                if let Some(Expression::Binary {
                    op, left, right, ..
                }) = exprs.try_get(*condition)
                    && matches!(
                        op,
                        BinaryOp::Less
                            | BinaryOp::LessEqual
                            | BinaryOp::Greater
                            | BinaryOp::GreaterEqual
                    )
                {
                    // Check right operand first (most common: `idx < params.C1`)
                    if let Some(idx) = extract_uniform_member_index(exprs, *right) {
                        return Some(idx as i64);
                    }
                    if let Some(idx) = extract_uniform_member_index(exprs, *left) {
                        return Some(idx as i64);
                    }
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                if let Some(axis) = find_if_comparison_axis(body, exprs, _shape_names) {
                    return Some(axis);
                }
                if let Some(axis) = find_if_comparison_axis(continuing, exprs, _shape_names) {
                    return Some(axis);
                }
            }
            _ => {}
        }
    }
    None
}

/// Extract the struct member index from an expression that accesses a
/// uniform params member.
///
/// Handles both direct `AccessIndex { base, index }` and
/// `Load { pointer: AccessIndex { base, index } }` patterns, since
/// different naga versions may emit either form.
fn extract_uniform_member_index(
    exprs: &Arena<Expression>,
    handle: Handle<Expression>,
) -> Option<u32> {
    match exprs.try_get(handle)? {
        Expression::AccessIndex { index, .. } => Some(*index),
        Expression::Load { pointer } => match exprs.try_get(*pointer)? {
            Expression::AccessIndex { index, .. } => Some(*index),
            _ => None,
        },
        _ => None,
    }
}

/// Check if any expression in the arena uses a specific math function.
fn has_math_function_in_expressions(exprs: &Arena<Expression>, target: MathFunction) -> bool {
    exprs
        .iter()
        .any(|(_, expr)| matches!(expr, Expression::Math { fun, .. } if *fun == target))
}

/// Check if a block (or any nested block) contains an If statement.
fn has_if_statement(body: &[Statement]) -> bool {
    body.iter().any(|stmt| match stmt {
        Statement::If { .. } => true,
        Statement::Loop {
            body, continuing, ..
        } => has_if_statement(body) || has_if_statement(continuing),
        _ => false,
    })
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
        // tanh(x) → Tanh (standalone, not part of a Multiply)
        Expression::Math {
            fun: MathFunction::Tanh,
            ..
        } => Some(ActivationOp::Tanh),
        // Multiply patterns: GELU, SiLU, Mish
        Expression::Binary {
            op: BinaryOp::Multiply,
            left,
            right,
            ..
        } => {
            let has_tanh_left = contains_math_fun(exprs, *left, MathFunction::Tanh);
            let has_tanh_right = contains_math_fun(exprs, *right, MathFunction::Tanh);
            let has_exp_left = contains_math_fun(exprs, *left, MathFunction::Exp);
            let has_exp_right = contains_math_fun(exprs, *right, MathFunction::Exp);

            // GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
            // Detected as Multiply where one sub-expression contains Tanh
            // but NOT Exp (to distinguish from Mish which has tanh+exp)
            if (has_tanh_left || has_tanh_right) && !has_exp_left && !has_exp_right {
                return Some(ActivationOp::Gelu);
            }

            // Mish: x * tanh(ln(1 + exp(x))) = x * tanh(softplus(x))
            // Detected as Multiply containing both Tanh and Exp
            if (has_tanh_left || has_tanh_right) && (has_exp_left || has_exp_right) {
                return Some(ActivationOp::Mish);
            }

            // SiLU: x * sigmoid(x) = x * (1 / (1 + exp(-x)))
            // Detected as Multiply where one sub-expression contains Divide+Exp (sigmoid pattern)
            if has_exp_left || has_exp_right {
                let has_div_left = contains_binary_op(exprs, *left, BinaryOp::Divide);
                let has_div_right = contains_binary_op(exprs, *right, BinaryOp::Divide);
                if has_div_left || has_div_right {
                    return Some(ActivationOp::Silu);
                }
            }

            // Not a recognized activation multiply, try recursing
            classify_activation_expr(exprs, *left)
                .or_else(|| classify_activation_expr(exprs, *right))
        }
        // 1/(1+exp(-x)) → Sigmoid: detected as Divide whose right is Add(1, Exp(Negate(x)))
        // exp(x)/sum(exp(x)) → Softmax: detected as Divide with Exp on left
        Expression::Binary {
            op: BinaryOp::Divide,
            left,
            right,
            ..
        } => {
            if contains_math_fun(exprs, *left, MathFunction::Exp) {
                // Softmax: exp(x) / sum(exp(x))
                Some(ActivationOp::Softmax)
            } else if contains_math_fun(exprs, *right, MathFunction::Exp) {
                // Sigmoid: 1 / (1 + exp(-x))
                Some(ActivationOp::Sigmoid)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Check if an expression (recursively) contains a specific binary operation.
fn contains_binary_op(
    exprs: &Arena<Expression>,
    handle: Handle<Expression>,
    target: BinaryOp,
) -> bool {
    match exprs.try_get(handle) {
        Some(Expression::Binary {
            op, left, right, ..
        }) => {
            *op == target
                || contains_binary_op(exprs, *left, target)
                || contains_binary_op(exprs, *right, target)
        }
        Some(Expression::Math { arg, .. }) => contains_binary_op(exprs, *arg, target),
        Some(Expression::Unary { expr, .. }) => contains_binary_op(exprs, *expr, target),
        _ => false,
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
    } else if find_store_binary_divide(body, exprs) {
        // Sum followed by divide → Mean
        ReduceOp::Mean
    } else {
        // Default: sum accumulation (binary add in loop)
        ReduceOp::Sum
    }
}

/// Check if a block contains a Store whose value is a Divide expression.
fn find_store_binary_divide(body: &[Statement], exprs: &Arena<Expression>) -> bool {
    for stmt in body {
        match stmt {
            Statement::Store { value, .. } => {
                if let Some(Expression::Binary {
                    op: BinaryOp::Divide,
                    ..
                }) = exprs.try_get(*value)
                {
                    return true;
                }
            }
            Statement::If { accept, reject, .. } => {
                if find_store_binary_divide(accept, exprs)
                    || find_store_binary_divide(reject, exprs)
                {
                    return true;
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                if find_store_binary_divide(body, exprs)
                    || find_store_binary_divide(continuing, exprs)
                {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
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
    fn classify_single_input_no_activation_unknown() {
        // 1 input, no loop, no recognized activation, 2+ params → Unknown.
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
                        name: Some("rows".into()),
                        ty: u32_ty,
                        offset: 0,
                    },
                    StructMember {
                        name: Some("cols".into()),
                        ty: u32_ty,
                        offset: 4,
                    },
                ],
                span: 8,
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

        // Body: just a store of a literal (no activation, no loop)
        let mut func = Function::new("main");
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(42.0)));
        let ptr = func
            .expressions
            .append(Expression::Literal(Literal::F32(0.0)));
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });

        module.entry_points.push(EntryPoint {
            name: "unknown_kernel".into(),
            workgroup_size: [256, 1, 1],
            function: func,
        });

        let pattern = classify_entry_point(&module, 0).unwrap();
        assert!(
            matches!(&pattern, KernelPattern::Unknown { .. }),
            "expected Unknown pattern, got {pattern:?}"
        );
    }

    #[test]
    fn classify_elementwise_with_embedded_weight() {
        // 1 storage input + 1 output + private global with init + binary Add → ElementWise.
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
        let array_f32_4 = module.types.insert(Type {
            name: None,
            inner: TypeInner::Array {
                base: f32_ty,
                size: ArraySize::Constant(4),
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

        // Storage input (binding 0)
        module.global_variables.append(GlobalVariable {
            name: Some("input".into()),
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
        // Storage output (binding 1)
        module.global_variables.append(GlobalVariable {
            name: Some("output".into()),
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

        // Private global with init (embedded weight)
        let lit0 = module
            .global_expressions
            .append(Expression::Literal(Literal::F32(0.1)));
        let lit1 = module
            .global_expressions
            .append(Expression::Literal(Literal::F32(0.2)));
        let lit2 = module
            .global_expressions
            .append(Expression::Literal(Literal::F32(0.3)));
        let lit3 = module
            .global_expressions
            .append(Expression::Literal(Literal::F32(0.4)));
        let compose = module.global_expressions.append(Expression::Compose {
            ty: array_f32_4,
            components: vec![lit0, lit1, lit2, lit3],
        });
        module.global_variables.append(GlobalVariable {
            name: Some("bias".into()),
            space: AddressSpace::Private,
            binding: None,
            ty: array_f32_4,
            init: Some(compose),
            layout: None,
        });

        // Uniform params
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

        // Entry point: Store of Binary Add (no loop)
        let mut func = Function::new("main");
        let left = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        let right = func
            .expressions
            .append(Expression::Literal(Literal::F32(2.0)));
        let binary = func.expressions.append(Expression::Binary {
            op: BinaryOp::Add,
            left,
            right,
        });
        let ptr = func
            .expressions
            .append(Expression::Literal(Literal::F32(0.0)));
        func.body.push(Statement::Store {
            pointer: ptr,
            value: binary,
        });

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [64, 1, 1],
            function: func,
        });

        let pattern = classify_entry_point(&module, 0).unwrap();
        match &pattern {
            KernelPattern::ElementWise {
                op,
                inputs,
                output,
                dim_name,
            } => {
                assert_eq!(*op, ElementWiseOp::Add);
                assert_eq!(inputs[0].name, "input");
                assert_eq!(inputs[1].name, "bias");
                assert_eq!(output.name, "output");
                assert_eq!(dim_name, "N");
            }
            _ => panic!("expected ElementWise pattern, got {pattern:?}"),
        }
    }

    #[test]
    fn detect_reduce_mean() {
        // Loop body with a divide → ReduceOp::Mean
        let mut func = Function::new("test");
        let left = func
            .expressions
            .append(Expression::Literal(Literal::F32(0.0)));
        let right = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        let div = func.expressions.append(Expression::Binary {
            op: BinaryOp::Divide,
            left,
            right,
        });
        let ptr = func
            .expressions
            .append(Expression::Literal(Literal::F32(0.0)));
        let body = vec![
            Statement::Store {
                pointer: ptr,
                value: div,
            },
            Statement::Break,
        ];
        let result = detect_reduce_op(&body, &func.expressions);
        assert_eq!(result, ReduceOp::Mean);
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

    #[test]
    fn extract_loop_bounds_literal_u32() {
        let mut func = Function::new("test");
        let local = func.local_variables.append(LocalVariable {
            name: Some("kh".into()),
            ty: {
                let mut types = UniqueArena::new();
                types.insert(Type {
                    name: None,
                    inner: TypeInner::Scalar(Scalar::U32),
                })
            },
            init: None,
        });
        let load = func.expressions.append(Expression::LocalVariable(local));
        let lit = func
            .expressions
            .append(Expression::Literal(Literal::U32(5)));
        let cmp = func.expressions.append(Expression::Binary {
            op: BinaryOp::GreaterEqual,
            left: load,
            right: lit,
        });
        let body = vec![Statement::Loop {
            body: vec![Statement::Break],
            continuing: vec![],
            break_if: Some(cmp),
        }];
        let bounds = extract_loop_bound_literals(&body, &func.expressions);
        assert_eq!(bounds, vec![5]);
    }

    #[test]
    fn extract_loop_bounds_nested() {
        let mut func = Function::new("test");
        // Inner loop: kh < 3
        let local_kh = func
            .expressions
            .append(Expression::Literal(Literal::U32(0)));
        let lit3 = func
            .expressions
            .append(Expression::Literal(Literal::U32(3)));
        let cmp_inner = func.expressions.append(Expression::Binary {
            op: BinaryOp::GreaterEqual,
            left: local_kh,
            right: lit3,
        });
        // Outer loop: kw < 5
        let local_kw = func
            .expressions
            .append(Expression::Literal(Literal::U32(0)));
        let lit5 = func
            .expressions
            .append(Expression::Literal(Literal::U32(5)));
        let cmp_outer = func.expressions.append(Expression::Binary {
            op: BinaryOp::GreaterEqual,
            left: local_kw,
            right: lit5,
        });
        let body = vec![Statement::Loop {
            body: vec![Statement::Loop {
                body: vec![Statement::Break],
                continuing: vec![],
                break_if: Some(cmp_inner),
            }],
            continuing: vec![],
            break_if: Some(cmp_outer),
        }];
        let bounds = extract_loop_bound_literals(&body, &func.expressions);
        assert_eq!(bounds, vec![5, 3]);
    }

    #[test]
    fn extract_loop_bounds_no_literal() {
        let func = Function::new("test");
        let body = vec![Statement::Loop {
            body: vec![Statement::Break],
            continuing: vec![],
            break_if: None,
        }];
        let bounds = extract_loop_bound_literals(&body, &func.expressions);
        assert!(bounds.is_empty());
    }

    /// naga 28 lowers `for (var kh = 0u; kh < 5u; ...)` as:
    ///   Loop { body: [Emit, Emit, If(kh<5) {} else {Break}, ...], break_if: None }
    #[test]
    fn extract_loop_bounds_if_else_break_pattern() {
        let mut func = Function::new("test");
        let local = func.local_variables.append(LocalVariable {
            name: Some("kh".into()),
            ty: {
                let mut types = UniqueArena::new();
                types.insert(Type {
                    name: None,
                    inner: TypeInner::Scalar(Scalar::U32),
                })
            },
            init: None,
        });
        let load = func.expressions.append(Expression::LocalVariable(local));
        let lit5 = func
            .expressions
            .append(Expression::Literal(Literal::U32(5)));
        let cmp = func.expressions.append(Expression::Binary {
            op: BinaryOp::Less,
            left: load,
            right: lit5,
        });
        // Simulate If(cond){} else {Break} inside loop body (skipping Emit)
        let body = vec![Statement::Loop {
            body: vec![Statement::If {
                condition: cmp,
                accept: vec![],
                reject: vec![Statement::Break],
            }],
            continuing: vec![],
            break_if: None,
        }];
        let bounds = extract_loop_bound_literals(&body, &func.expressions);
        assert_eq!(bounds, vec![5]);
    }

    #[test]
    fn extract_multiply_literals_stride() {
        let mut func = Function::new("test");
        let oh = func
            .expressions
            .append(Expression::Literal(Literal::U32(0)));
        let two = func
            .expressions
            .append(Expression::Literal(Literal::U32(2)));
        func.expressions.append(Expression::Binary {
            op: BinaryOp::Multiply,
            left: oh,
            right: two,
        });
        let strides = extract_multiply_literals(&func.expressions);
        assert_eq!(strides, vec![2]);
    }

    #[test]
    fn extract_multiply_literals_ignores_one() {
        let mut func = Function::new("test");
        let oh = func
            .expressions
            .append(Expression::Literal(Literal::U32(0)));
        let one = func
            .expressions
            .append(Expression::Literal(Literal::U32(1)));
        func.expressions.append(Expression::Binary {
            op: BinaryOp::Multiply,
            left: oh,
            right: one,
        });
        let strides = extract_multiply_literals(&func.expressions);
        assert!(strides.is_empty());
    }

    #[test]
    fn extract_subtract_literals_padding() {
        let mut func = Function::new("test");
        let idx = func
            .expressions
            .append(Expression::Literal(Literal::U32(10)));
        let pad = func
            .expressions
            .append(Expression::Literal(Literal::U32(1)));
        func.expressions.append(Expression::Binary {
            op: BinaryOp::Subtract,
            left: idx,
            right: pad,
        });
        let pads = extract_subtract_literals(&func.expressions);
        assert_eq!(pads, vec![1]);
    }

    #[test]
    fn conv2d_shape_no_body_defaults_to_unknown() {
        let names: Vec<String> = ["N", "IC", "IH", "IW", "OC", "KH", "KW"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let shape = extract_conv2d_shape(&names, None, None);
        assert_eq!(shape.batch, "N");
        assert_eq!(shape.kernel_h, "KH");
        assert_eq!(shape.kernel_w, "KW");
        assert_eq!(shape.kernel_h_val, 0);
        assert_eq!(shape.kernel_w_val, 0);
        assert_eq!(shape.stride_h, 1);
        assert_eq!(shape.stride_w, 1);
        assert_eq!(shape.pad_h, 0);
        assert_eq!(shape.pad_w, 0);
    }

    #[test]
    fn conv2d_shape_with_literal_kernel() {
        let names: Vec<String> = ["N", "IC", "IH", "IW", "OC", "KH", "KW"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        // Build a function body with loops: kh < 5, kw < 5
        let mut func = Function::new("test");
        let kh_var = func
            .expressions
            .append(Expression::Literal(Literal::U32(0)));
        let lit5a = func
            .expressions
            .append(Expression::Literal(Literal::U32(5)));
        let cmp_h = func.expressions.append(Expression::Binary {
            op: BinaryOp::GreaterEqual,
            left: kh_var,
            right: lit5a,
        });
        let kw_var = func
            .expressions
            .append(Expression::Literal(Literal::U32(0)));
        let lit5b = func
            .expressions
            .append(Expression::Literal(Literal::U32(5)));
        let cmp_w = func.expressions.append(Expression::Binary {
            op: BinaryOp::GreaterEqual,
            left: kw_var,
            right: lit5b,
        });
        // Also add stride: oh * 2u
        let oh = func
            .expressions
            .append(Expression::Literal(Literal::U32(0)));
        let two = func
            .expressions
            .append(Expression::Literal(Literal::U32(2)));
        func.expressions.append(Expression::Binary {
            op: BinaryOp::Multiply,
            left: oh,
            right: two,
        });
        // Also add padding: ih - 1u
        let ih = func
            .expressions
            .append(Expression::Literal(Literal::U32(10)));
        let one = func
            .expressions
            .append(Expression::Literal(Literal::U32(1)));
        func.expressions.append(Expression::Binary {
            op: BinaryOp::Subtract,
            left: ih,
            right: one,
        });

        let body = vec![Statement::Loop {
            body: vec![Statement::Loop {
                body: vec![Statement::Break],
                continuing: vec![],
                break_if: Some(cmp_w),
            }],
            continuing: vec![],
            break_if: Some(cmp_h),
        }];

        let shape = extract_conv2d_shape(&names, Some(&body), Some(&func.expressions));
        assert_eq!(shape.kernel_h_val, 5);
        assert_eq!(shape.kernel_w_val, 5);
        assert_eq!(shape.stride_h, 2);
        assert_eq!(shape.stride_w, 2);
        assert_eq!(shape.pad_h, 1);
        assert_eq!(shape.pad_w, 1);
    }

    /// Build a pool-like module with literal kernel and stride.
    fn make_pool_module(kernel: u32, stride: u32) -> Module {
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
                        name: Some("C".into()),
                        ty: u32_ty,
                        offset: 0,
                    },
                    StructMember {
                        name: Some("IH".into()),
                        ty: u32_ty,
                        offset: 4,
                    },
                    StructMember {
                        name: Some("IW".into()),
                        ty: u32_ty,
                        offset: 8,
                    },
                    StructMember {
                        name: Some("OH".into()),
                        ty: u32_ty,
                        offset: 12,
                    },
                ],
                span: 16,
            },
        });

        module.global_variables.append(GlobalVariable {
            name: Some("input".into()),
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
            name: Some("output".into()),
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

        // Loop bounds: kh < kernel, kw < kernel
        let kh_var = func
            .expressions
            .append(Expression::Literal(Literal::U32(0)));
        let lit_k1 = func
            .expressions
            .append(Expression::Literal(Literal::U32(kernel)));
        let cmp_h = func.expressions.append(Expression::Binary {
            op: BinaryOp::GreaterEqual,
            left: kh_var,
            right: lit_k1,
        });
        let kw_var = func
            .expressions
            .append(Expression::Literal(Literal::U32(0)));
        let lit_k2 = func
            .expressions
            .append(Expression::Literal(Literal::U32(kernel)));
        let cmp_w = func.expressions.append(Expression::Binary {
            op: BinaryOp::GreaterEqual,
            left: kw_var,
            right: lit_k2,
        });

        // Stride: oh * stride
        if stride > 1 {
            let oh = func
                .expressions
                .append(Expression::Literal(Literal::U32(0)));
            let lit_s = func
                .expressions
                .append(Expression::Literal(Literal::U32(stride)));
            func.expressions.append(Expression::Binary {
                op: BinaryOp::Multiply,
                left: oh,
                right: lit_s,
            });
        }

        // Max function for MaxPool detection
        let arg = func
            .expressions
            .append(Expression::Literal(Literal::F32(0.0)));
        let arg1 = func
            .expressions
            .append(Expression::Literal(Literal::F32(0.0)));
        let max_expr = func.expressions.append(Expression::Math {
            fun: MathFunction::Max,
            arg,
            arg1: Some(arg1),
            arg2: None,
            arg3: None,
        });
        let ptr = func
            .expressions
            .append(Expression::Literal(Literal::F32(0.0)));
        func.body.push(Statement::Loop {
            body: vec![
                Statement::Loop {
                    body: vec![
                        Statement::Store {
                            pointer: ptr,
                            value: max_expr,
                        },
                        Statement::Break,
                    ],
                    continuing: vec![],
                    break_if: Some(cmp_w),
                },
                Statement::Break,
            ],
            continuing: vec![],
            break_if: Some(cmp_h),
        });

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [16, 16, 1],
            function: func,
        });

        module
    }

    #[test]
    fn classify_pool_extracts_2x2_stride2() {
        let module = make_pool_module(2, 2);
        let pattern = classify_entry_point(&module, 0).unwrap();
        match &pattern {
            KernelPattern::Pool { kind, shape, .. } => {
                assert_eq!(*kind, PoolKind::Max);
                assert_eq!(shape.kernel_h, 2);
                assert_eq!(shape.kernel_w, 2);
                assert_eq!(shape.stride_h, 2);
                assert_eq!(shape.stride_w, 2);
            }
            _ => panic!("expected Pool pattern, got {pattern:?}"),
        }
    }

    #[test]
    fn classify_pool_extracts_3x3_stride1() {
        let module = make_pool_module(3, 1);
        let pattern = classify_entry_point(&module, 0).unwrap();
        match &pattern {
            KernelPattern::Pool { kind, shape, .. } => {
                assert_eq!(*kind, PoolKind::Max);
                assert_eq!(shape.kernel_h, 3);
                assert_eq!(shape.kernel_w, 3);
                // stride=1: no multiply literals, default to kernel size
                assert_eq!(shape.stride_h, 3);
                assert_eq!(shape.stride_w, 3);
            }
            _ => panic!("expected Pool pattern, got {pattern:?}"),
        }
    }

    // --- EmbeddedWeight extraction tests ---

    #[test]
    fn eval_const_expr_f32_literal() {
        let mut exprs = Arena::new();
        let types = UniqueArena::new();
        let h = exprs.append(Expression::Literal(Literal::F32(2.78)));
        assert_eq!(eval_const_expr_f32(h, &exprs, &types), Some(vec![2.78]));
    }

    #[test]
    fn eval_const_expr_f32_compose() {
        let mut exprs = Arena::new();
        let types = UniqueArena::new();
        let a = exprs.append(Expression::Literal(Literal::F32(0.1)));
        let b = exprs.append(Expression::Literal(Literal::F32(0.2)));
        let c = exprs.append(Expression::Literal(Literal::F32(0.3)));
        let f32_ty = {
            let mut t = UniqueArena::new();
            t.insert(Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::F32),
            })
        };
        let compose = exprs.append(Expression::Compose {
            ty: f32_ty,
            components: vec![a, b, c],
        });
        let result = eval_const_expr_f32(compose, &exprs, &types).unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.1).abs() < 1e-6);
        assert!((result[1] - 0.2).abs() < 1e-6);
        assert!((result[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn eval_const_expr_f32_zero_value() {
        let mut exprs = Arena::new();
        let mut types = UniqueArena::new();
        let f32_ty = types.insert(Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::F32),
        });
        let arr_ty = types.insert(Type {
            name: None,
            inner: TypeInner::Array {
                base: f32_ty,
                size: ArraySize::Constant(3),
                stride: 4,
            },
        });
        let h = exprs.append(Expression::ZeroValue(arr_ty));
        let result = eval_const_expr_f32(h, &exprs, &types).unwrap();
        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn extract_embedded_weights_private_global() {
        let mut module = Module::default();

        let f32_ty = module.types.insert(Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::F32),
        });
        let array_f32_4 = module.types.insert(Type {
            name: None,
            inner: TypeInner::Array {
                base: f32_ty,
                size: ArraySize::Constant(4),
                stride: 4,
            },
        });

        // Build init expression: Compose([0.1, 0.2, 0.3, 0.4])
        let lit0 = module
            .global_expressions
            .append(Expression::Literal(Literal::F32(0.1)));
        let lit1 = module
            .global_expressions
            .append(Expression::Literal(Literal::F32(0.2)));
        let lit2 = module
            .global_expressions
            .append(Expression::Literal(Literal::F32(0.3)));
        let lit3 = module
            .global_expressions
            .append(Expression::Literal(Literal::F32(0.4)));
        let compose = module.global_expressions.append(Expression::Compose {
            ty: array_f32_4,
            components: vec![lit0, lit1, lit2, lit3],
        });

        module.global_variables.append(GlobalVariable {
            name: Some("bias".into()),
            space: AddressSpace::Private,
            binding: None,
            ty: array_f32_4,
            init: Some(compose),
            layout: None,
        });

        let weights = extract_embedded_weights(&module);
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0].name, "bias");
        assert_eq!(weights[0].dims, vec![4]);
        assert_eq!(weights[0].data.len(), 4);
        assert!((weights[0].data[0] - 0.1).abs() < 1e-6);
        assert!((weights[0].data[3] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn extract_embedded_weights_no_init() {
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

        let weights = extract_embedded_weights(&module);
        assert!(weights.is_empty());
    }

    // --- Concat/Split axis inference tests ---

    /// Build a concat-like module with 2 inputs, 1 output, and an If condition
    /// that compares against `AccessIndex { base: params, index: boundary_index }`.
    /// This simulates the IR that naga produces for `if c < params.C1`.
    fn make_concat_module(boundary_index: u32, shape_names: &[&str]) -> Module {
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
                members: shape_names
                    .iter()
                    .enumerate()
                    .map(|(i, name)| StructMember {
                        name: Some((*name).into()),
                        ty: u32_ty,
                        offset: (i * 4) as u32,
                    })
                    .collect(),
                span: (shape_names.len() * 4) as u32,
            },
        });

        // Input a (binding 0)
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
        // Input b (binding 1)
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
        // Output (binding 2)
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
        // Uniform params (binding 3)
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

        // Build function with If(idx < AccessIndex(params, boundary_index))
        let mut func = Function::new("main");
        let idx = func
            .expressions
            .append(Expression::Literal(Literal::U32(0)));
        // params GlobalVariable (handle index 3)
        let params_gv = {
            let mut iter = module.global_variables.iter();
            iter.nth(3).unwrap().0
        };
        let params_expr = func
            .expressions
            .append(Expression::GlobalVariable(params_gv));
        let access = func.expressions.append(Expression::AccessIndex {
            base: params_expr,
            index: boundary_index,
        });
        let load = func
            .expressions
            .append(Expression::Load { pointer: access });
        let cmp = func.expressions.append(Expression::Binary {
            op: BinaryOp::Less,
            left: idx,
            right: load,
        });

        // If body: Store (accept), Store (reject)
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(0.0)));
        let ptr = func
            .expressions
            .append(Expression::Literal(Literal::F32(0.0)));
        func.body.push(Statement::If {
            condition: cmp,
            accept: vec![Statement::Store {
                pointer: ptr,
                value: val,
            }],
            reject: vec![Statement::Store {
                pointer: ptr,
                value: val,
            }],
        });

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [256, 1, 1],
            function: func,
        });

        module
    }

    /// Build a split-like module with 1 input, 2 outputs, and an If condition
    /// that compares against `AccessIndex { base: params, index: boundary_index }`.
    fn make_split_module(boundary_index: u32, shape_names: &[&str]) -> Module {
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
                members: shape_names
                    .iter()
                    .enumerate()
                    .map(|(i, name)| StructMember {
                        name: Some((*name).into()),
                        ty: u32_ty,
                        offset: (i * 4) as u32,
                    })
                    .collect(),
                span: (shape_names.len() * 4) as u32,
            },
        });

        // Input (binding 0) - read-only storage
        module.global_variables.append(GlobalVariable {
            name: Some("input".into()),
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
        // Output a (binding 1)
        module.global_variables.append(GlobalVariable {
            name: Some("out_a".into()),
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
        // Output b (binding 2)
        module.global_variables.append(GlobalVariable {
            name: Some("out_b".into()),
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
        // Uniform params (binding 3)
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

        // Build function with If(idx < AccessIndex(params, boundary_index))
        let mut func = Function::new("main");
        let idx = func
            .expressions
            .append(Expression::Literal(Literal::U32(0)));
        let params_gv = {
            let mut iter = module.global_variables.iter();
            iter.nth(3).unwrap().0
        };
        let params_expr = func
            .expressions
            .append(Expression::GlobalVariable(params_gv));
        let access = func.expressions.append(Expression::AccessIndex {
            base: params_expr,
            index: boundary_index,
        });
        let load = func
            .expressions
            .append(Expression::Load { pointer: access });
        let cmp = func.expressions.append(Expression::Binary {
            op: BinaryOp::Less,
            left: idx,
            right: load,
        });

        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(0.0)));
        let ptr = func
            .expressions
            .append(Expression::Literal(Literal::F32(0.0)));
        func.body.push(Statement::If {
            condition: cmp,
            accept: vec![Statement::Store {
                pointer: ptr,
                value: val,
            }],
            reject: vec![Statement::Store {
                pointer: ptr,
                value: val,
            }],
        });

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [256, 1, 1],
            function: func,
        });

        module
    }

    #[test]
    fn classify_concat_axis_0_default() {
        // Concat with boundary at index 0 → axis = 0 (same as before).
        let module = make_concat_module(0, &["N_a", "N_b"]);
        let pattern = classify_entry_point(&module, 0).unwrap();
        match &pattern {
            KernelPattern::Concat { axis, .. } => {
                assert_eq!(*axis, 0, "expected axis 0 for boundary at index 0");
            }
            _ => panic!("expected Concat pattern, got {pattern:?}"),
        }
    }

    #[test]
    fn classify_concat_axis_1_inferred() {
        // Concat with boundary at index 1 (params.C1) → axis = 1.
        let module = make_concat_module(1, &["N", "C1", "C2"]);
        let pattern = classify_entry_point(&module, 0).unwrap();
        match &pattern {
            KernelPattern::Concat {
                axis,
                inputs,
                output,
                ..
            } => {
                assert_eq!(*axis, 1, "expected axis 1 for boundary at index 1");
                assert_eq!(inputs[0].name, "a");
                assert_eq!(inputs[1].name, "b");
                assert_eq!(output.name, "result");
            }
            _ => panic!("expected Concat pattern, got {pattern:?}"),
        }
    }

    #[test]
    fn classify_concat_axis_2_inferred() {
        // Concat with boundary at index 2 → axis = 2.
        let module = make_concat_module(2, &["N", "C", "W1", "W2"]);
        let pattern = classify_entry_point(&module, 0).unwrap();
        match &pattern {
            KernelPattern::Concat { axis, .. } => {
                assert_eq!(*axis, 2, "expected axis 2 for boundary at index 2");
            }
            _ => panic!("expected Concat pattern, got {pattern:?}"),
        }
    }

    #[test]
    fn classify_split_axis_0_default() {
        // Split with boundary at index 0 → axis = 0.
        let module = make_split_module(0, &["N_a", "N_b"]);
        let pattern = classify_entry_point(&module, 0).unwrap();
        match &pattern {
            KernelPattern::Split { axis, .. } => {
                assert_eq!(*axis, 0, "expected axis 0 for boundary at index 0");
            }
            _ => panic!("expected Split pattern, got {pattern:?}"),
        }
    }

    #[test]
    fn classify_split_axis_1_inferred() {
        // Split with boundary at index 1 (params.C1) → axis = 1.
        let module = make_split_module(1, &["N", "C1", "C2"]);
        let pattern = classify_entry_point(&module, 0).unwrap();
        match &pattern {
            KernelPattern::Split {
                axis,
                input,
                outputs,
                ..
            } => {
                assert_eq!(*axis, 1, "expected axis 1 for boundary at index 1");
                assert_eq!(input.name, "input");
                assert_eq!(outputs.len(), 2);
                assert_eq!(outputs[0].name, "out_a");
                assert_eq!(outputs[1].name, "out_b");
            }
            _ => panic!("expected Split pattern, got {pattern:?}"),
        }
    }

    #[test]
    fn classify_split_axis_2_inferred() {
        // Split with boundary at index 2 → axis = 2.
        let module = make_split_module(2, &["N", "C", "W1", "W2"]);
        let pattern = classify_entry_point(&module, 0).unwrap();
        match &pattern {
            KernelPattern::Split { axis, .. } => {
                assert_eq!(*axis, 2, "expected axis 2 for boundary at index 2");
            }
            _ => panic!("expected Split pattern, got {pattern:?}"),
        }
    }

    #[test]
    fn normalize_axis_positive() {
        assert_eq!(normalize_axis(0, 3), 0);
        assert_eq!(normalize_axis(1, 4), 1);
        assert_eq!(normalize_axis(2, 4), 2);
    }

    #[test]
    fn normalize_axis_negative() {
        assert_eq!(normalize_axis(-1, 4), 3);
        assert_eq!(normalize_axis(-2, 4), 2);
        assert_eq!(normalize_axis(-3, 4), 1);
        assert_eq!(normalize_axis(-4, 4), 0);
    }

    #[test]
    fn normalize_axis_negative_wraps() {
        // -1 with ndim 3 → 2
        assert_eq!(normalize_axis(-1, 3), 2);
        // -2 with ndim 3 → 1
        assert_eq!(normalize_axis(-2, 3), 1);
    }

    #[test]
    fn find_if_comparison_axis_direct_access_index() {
        // Test that find_if_comparison_axis works with direct AccessIndex
        // (without Load wrapper).
        let mut func = Function::new("test");
        let idx = func
            .expressions
            .append(Expression::Literal(Literal::U32(0)));
        let base = func
            .expressions
            .append(Expression::Literal(Literal::U32(0)));
        let access = func
            .expressions
            .append(Expression::AccessIndex { base, index: 2 });
        let cmp = func.expressions.append(Expression::Binary {
            op: BinaryOp::Less,
            left: idx,
            right: access,
        });

        let body = vec![Statement::If {
            condition: cmp,
            accept: vec![],
            reject: vec![],
        }];
        let names: Vec<String> = vec!["N".into(), "C".into(), "W".into()];
        let result = find_if_comparison_axis(&body, &func.expressions, &names);
        assert_eq!(result, Some(2));
    }

    #[test]
    fn find_if_comparison_axis_in_loop() {
        // Test that the function searches into Loop bodies.
        let mut func = Function::new("test");
        let idx = func
            .expressions
            .append(Expression::Literal(Literal::U32(0)));
        let base = func
            .expressions
            .append(Expression::Literal(Literal::U32(0)));
        let access = func
            .expressions
            .append(Expression::AccessIndex { base, index: 1 });
        let load = func
            .expressions
            .append(Expression::Load { pointer: access });
        let cmp = func.expressions.append(Expression::Binary {
            op: BinaryOp::LessEqual,
            left: idx,
            right: load,
        });

        let body = vec![Statement::Loop {
            body: vec![Statement::If {
                condition: cmp,
                accept: vec![],
                reject: vec![],
            }],
            continuing: vec![],
            break_if: None,
        }];
        let names: Vec<String> = vec!["N".into(), "C".into()];
        let result = find_if_comparison_axis(&body, &func.expressions, &names);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn find_if_comparison_axis_no_if_returns_none() {
        let func = Function::new("test");
        let body = vec![Statement::Break];
        let names: Vec<String> = vec!["N".into()];
        let result = find_if_comparison_axis(&body, &func.expressions, &names);
        assert_eq!(result, None);
    }

    // --- Multi-head & causal attention classification tests ---

    /// Build a module with 3 inputs (query, key, value), 1 output, a uniform
    /// params struct with the given member names, Loop + Exp + Sqrt in the
    /// function expressions, and optionally a division-by-literal and/or a
    /// causal If-guard inside the loop body.
    fn make_attention_module(param_names: &[&str], divide_by: Option<u32>, causal: bool) -> Module {
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

        // Build params struct from provided member names
        let members: Vec<StructMember> = param_names
            .iter()
            .enumerate()
            .map(|(i, name)| StructMember {
                name: Some((*name).into()),
                ty: u32_ty,
                offset: (i * 4) as u32,
            })
            .collect();
        let params_ty = module.types.insert(Type {
            name: Some("Params".into()),
            inner: TypeInner::Struct {
                members,
                span: (param_names.len() * 4) as u32,
            },
        });

        // 3 storage-read inputs (query, key, value)
        for (i, name) in ["query", "key", "value"].iter().enumerate() {
            module.global_variables.append(GlobalVariable {
                name: Some((*name).into()),
                space: AddressSpace::Storage {
                    access: StorageAccess::LOAD,
                },
                binding: Some(ResourceBinding {
                    group: 0,
                    binding: i as u32,
                }),
                ty: array_f32,
                init: None,
                layout: None,
            });
        }

        // 1 storage-read_write output
        module.global_variables.append(GlobalVariable {
            name: Some("output".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 3,
            }),
            ty: array_f32,
            init: None,
            layout: None,
        });

        // Uniform params
        module.global_variables.append(GlobalVariable {
            name: Some("params".into()),
            space: AddressSpace::Uniform,
            binding: Some(ResourceBinding {
                group: 0,
                binding: 4,
            }),
            ty: params_ty,
            init: None,
            layout: None,
        });

        // Build function with Exp, Sqrt, and optionally Division + causal If
        let mut func = Function::new("main");

        // Exp expression
        let arg_exp = func
            .expressions
            .append(Expression::Literal(Literal::F32(0.0)));
        func.expressions.append(Expression::Math {
            fun: MathFunction::Exp,
            arg: arg_exp,
            arg1: None,
            arg2: None,
            arg3: None,
        });

        // Sqrt expression
        let arg_sqrt = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        func.expressions.append(Expression::Math {
            fun: MathFunction::Sqrt,
            arg: arg_sqrt,
            arg1: None,
            arg2: None,
            arg3: None,
        });

        // Optional: division by literal (e.g. d_model / num_heads)
        if let Some(n) = divide_by {
            let dividend = func
                .expressions
                .append(Expression::Literal(Literal::U32(64)));
            let divisor = func
                .expressions
                .append(Expression::Literal(Literal::U32(n)));
            func.expressions.append(Expression::Binary {
                op: BinaryOp::Divide,
                left: dividend,
                right: divisor,
            });
        }

        // Build loop body contents
        let mut loop_inner: Vec<Statement> = Vec::new();

        if causal {
            // Causal mask pattern: If(j > i) { score = -1e30; }
            // First, add a comparison expression: Greater(j, i)
            let j_expr = func
                .expressions
                .append(Expression::Literal(Literal::U32(0)));
            let i_expr = func
                .expressions
                .append(Expression::Literal(Literal::U32(0)));
            let cmp = func.expressions.append(Expression::Binary {
                op: BinaryOp::Greater,
                left: j_expr,
                right: i_expr,
            });

            // Store of large negative literal
            let neg_val = func
                .expressions
                .append(Expression::Literal(Literal::F32(-1e30)));
            let ptr = func
                .expressions
                .append(Expression::Literal(Literal::F32(0.0)));

            // The for-loop break guard (first If in naga loop)
            let break_cond = func
                .expressions
                .append(Expression::Literal(Literal::Bool(true)));
            loop_inner.push(Statement::If {
                condition: break_cond,
                accept: vec![],
                reject: vec![Statement::Break],
            });

            // The causal If guard (second If in the loop)
            loop_inner.push(Statement::If {
                condition: cmp,
                accept: vec![Statement::Store {
                    pointer: ptr,
                    value: neg_val,
                }],
                reject: vec![],
            });
        }

        loop_inner.push(Statement::Break);

        func.body.push(Statement::Loop {
            body: loop_inner,
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

    #[test]
    fn classify_multihead_4_heads() {
        let module = make_attention_module(&["seq_len", "d_model", "num_heads"], Some(4), false);
        let pattern = classify_entry_point(&module, 0).unwrap();
        match &pattern {
            KernelPattern::Attention {
                num_heads,
                num_kv_heads,
                causal,
                seq_len,
                ..
            } => {
                assert_eq!(*num_heads, 4, "expected 4 heads from division literal");
                assert_eq!(
                    *num_kv_heads, 4,
                    "num_kv_heads should equal num_heads for MHA"
                );
                assert!(!causal, "should not detect causal mask");
                assert_eq!(seq_len, "seq_len");
            }
            _ => panic!("expected Attention pattern, got {pattern:?}"),
        }
        // Display should include heads=4 and no causal marker
        let display = format!("{pattern}");
        assert!(
            display.contains("heads=4"),
            "display should show heads=4: {display}"
        );
        assert!(
            !display.contains("causal"),
            "display should not mention causal: {display}"
        );
    }

    #[test]
    fn classify_causal_attention() {
        let module = make_attention_module(&["seq_len", "d_k"], None, true);
        let pattern = classify_entry_point(&module, 0).unwrap();
        match &pattern {
            KernelPattern::Attention {
                num_heads,
                causal,
                d_k,
                ..
            } => {
                assert_eq!(*num_heads, 1, "no num_heads param → default 1");
                assert!(*causal, "should detect causal mask from If + Store(-1e30)");
                assert_eq!(d_k, "d_k");
            }
            _ => panic!("expected Attention pattern, got {pattern:?}"),
        }
        // Display should include causal marker
        let display = format!("{pattern}");
        assert!(
            display.contains("causal"),
            "display should mention causal: {display}"
        );
        assert!(
            display.contains("heads=1"),
            "display should show heads=1: {display}"
        );
    }
}
