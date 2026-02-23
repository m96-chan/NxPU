//! Display implementations and text dump for debugging.

use std::fmt;

use crate::Module;
use crate::arena::{Handle, UniqueArena};
use crate::expr::{
    AtomicFunction, BinaryOp, Expression, Literal, MathFunction, SwizzleComponent, UnaryOp,
};
use crate::global::{AddressSpace, Binding, BuiltIn, ResourceBinding, StorageAccess};
use crate::stmt::{Barrier, Statement};
use crate::types::{ArraySize, Scalar, ScalarKind, Type, TypeInner, VectorSize};

impl fmt::Display for ScalarKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool => write!(f, "bool"),
            Self::Sint => write!(f, "sint"),
            Self::Uint => write!(f, "uint"),
            Self::Float => write!(f, "float"),
            Self::BFloat => write!(f, "bfloat"),
        }
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            ScalarKind::Bool => write!(f, "bool"),
            ScalarKind::Sint => write!(f, "i{}", self.width * 8),
            ScalarKind::Uint => write!(f, "u{}", self.width * 8),
            ScalarKind::Float => write!(f, "f{}", self.width * 8),
            ScalarKind::BFloat => write!(f, "bf{}", self.width * 8),
        }
    }
}

impl fmt::Display for VectorSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", *self as u32)
    }
}

impl fmt::Display for StorageAccess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let has_load = self.contains(StorageAccess::LOAD);
        let has_store = self.contains(StorageAccess::STORE);
        match (has_load, has_store) {
            (true, true) => write!(f, "read_write"),
            (true, false) => write!(f, "read"),
            (false, true) => write!(f, "write"),
            (false, false) => write!(f, "none"),
        }
    }
}

impl fmt::Display for AddressSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Function => write!(f, "function"),
            Self::Private => write!(f, "private"),
            Self::Workgroup => write!(f, "workgroup"),
            Self::Uniform => write!(f, "uniform"),
            Self::Storage { access } => write!(f, "storage, {access}"),
        }
    }
}

impl fmt::Display for BuiltIn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GlobalInvocationId => write!(f, "global_invocation_id"),
            Self::LocalInvocationId => write!(f, "local_invocation_id"),
            Self::LocalInvocationIndex => write!(f, "local_invocation_index"),
            Self::WorkgroupId => write!(f, "workgroup_id"),
            Self::NumWorkgroups => write!(f, "num_workgroups"),
        }
    }
}

impl fmt::Display for Binding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BuiltIn(b) => write!(f, "@builtin({b})"),
            Self::Location { location } => write!(f, "@location({location})"),
        }
    }
}

impl fmt::Display for ResourceBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "@group({}) @binding({})", self.group, self.binding)
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool(v) => write!(f, "{v}"),
            Self::I32(v) => write!(f, "{v}i"),
            Self::U32(v) => write!(f, "{v}u"),
            Self::F32(v) => write!(f, "{v}f"),
            Self::F64(v) => write!(f, "{v}lf"),
            Self::AbstractInt(v) => write!(f, "{v}"),
            Self::AbstractFloat(v) => write!(f, "{v}"),
        }
    }
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Negate => write!(f, "-"),
            Self::LogicalNot => write!(f, "!"),
            Self::BitwiseNot => write!(f, "~"),
        }
    }
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Add => write!(f, "+"),
            Self::Subtract => write!(f, "-"),
            Self::Multiply => write!(f, "*"),
            Self::Divide => write!(f, "/"),
            Self::Modulo => write!(f, "%"),
            Self::Equal => write!(f, "=="),
            Self::NotEqual => write!(f, "!="),
            Self::Less => write!(f, "<"),
            Self::LessEqual => write!(f, "<="),
            Self::Greater => write!(f, ">"),
            Self::GreaterEqual => write!(f, ">="),
            Self::LogicalAnd => write!(f, "&&"),
            Self::LogicalOr => write!(f, "||"),
            Self::BitwiseAnd => write!(f, "&"),
            Self::BitwiseOr => write!(f, "|"),
            Self::BitwiseXor => write!(f, "^"),
            Self::ShiftLeft => write!(f, "<<"),
            Self::ShiftRight => write!(f, ">>"),
        }
    }
}

impl fmt::Display for MathFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Abs => "abs",
            Self::Min => "min",
            Self::Max => "max",
            Self::Clamp => "clamp",
            Self::Saturate => "saturate",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Round => "round",
            Self::Fract => "fract",
            Self::Trunc => "trunc",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Asin => "asin",
            Self::Acos => "acos",
            Self::Atan => "atan",
            Self::Atan2 => "atan2",
            Self::Sinh => "sinh",
            Self::Cosh => "cosh",
            Self::Tanh => "tanh",
            Self::Sqrt => "sqrt",
            Self::InverseSqrt => "inverseSqrt",
            Self::Log => "log",
            Self::Log2 => "log2",
            Self::Exp => "exp",
            Self::Exp2 => "exp2",
            Self::Pow => "pow",
            Self::Dot => "dot",
            Self::Cross => "cross",
            Self::Normalize => "normalize",
            Self::Length => "length",
            Self::Distance => "distance",
            Self::Mix => "mix",
            Self::Step => "step",
            Self::SmoothStep => "smoothStep",
            Self::Fma => "fma",
        };
        write!(f, "{name}")
    }
}

impl fmt::Display for Barrier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let storage = self.contains(Barrier::STORAGE);
        let workgroup = self.contains(Barrier::WORKGROUP);
        match (storage, workgroup) {
            (true, true) => write!(f, "storageBarrier | workgroupBarrier"),
            (true, false) => write!(f, "storageBarrier"),
            (false, true) => write!(f, "workgroupBarrier"),
            (false, false) => write!(f, "<no barrier>"),
        }
    }
}

impl fmt::Display for SwizzleComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::X => write!(f, "x"),
            Self::Y => write!(f, "y"),
            Self::Z => write!(f, "z"),
            Self::W => write!(f, "w"),
        }
    }
}

impl fmt::Display for AtomicFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Add => write!(f, "atomicAdd"),
            Self::Subtract => write!(f, "atomicSub"),
            Self::And => write!(f, "atomicAnd"),
            Self::ExclusiveOr => write!(f, "atomicXor"),
            Self::InclusiveOr => write!(f, "atomicOr"),
            Self::Min => write!(f, "atomicMin"),
            Self::Max => write!(f, "atomicMax"),
            Self::Exchange { compare: None } => write!(f, "atomicExchange"),
            Self::Exchange { compare: Some(c) } => write!(f, "atomicCompareExchange({c:?})"),
        }
    }
}

/// Formats a type using the type arena for resolving inner references.
pub fn format_type(ty: &Type, types: &UniqueArena<Type>) -> String {
    if let Some(ref name) = ty.name {
        return name.clone();
    }
    format_type_inner(&ty.inner, types)
}

/// Formats a [`TypeInner`] using the type arena for resolving references.
pub fn format_type_inner(inner: &TypeInner, types: &UniqueArena<Type>) -> String {
    match inner {
        TypeInner::Scalar(s) => format!("{s}"),
        TypeInner::Vector { size, scalar } => format!("vec{size}<{scalar}>"),
        TypeInner::Matrix {
            columns,
            rows,
            scalar,
        } => format!("mat{columns}x{rows}<{scalar}>"),
        TypeInner::Atomic(s) => format!("atomic<{s}>"),
        TypeInner::Pointer { base, space } => {
            let base_str = format_type(&types[*base], types);
            format!("ptr<{space}, {base_str}>")
        }
        TypeInner::Array { base, size, stride } => {
            let base_str = format_type(&types[*base], types);
            match size {
                ArraySize::Constant(n) => format!("array<{base_str}, {n}> /*stride {stride}*/"),
                ArraySize::Dynamic => format!("array<{base_str}> /*stride {stride}*/"),
            }
        }
        TypeInner::Struct { members, span } => {
            format!("struct({} members, span {span})", members.len())
        }
        TypeInner::Tensor { scalar, shape } => {
            let dims: Vec<String> = shape
                .dims
                .iter()
                .map(|d| match d {
                    crate::Dimension::Fixed(n) => n.to_string(),
                    crate::Dimension::Dynamic(Some(name)) => name.clone(),
                    crate::Dimension::Dynamic(None) => "?".into(),
                })
                .collect();
            format!("tensor<{scalar}>[{}]", dims.join(", "))
        }
    }
}

fn format_expr(handle: Handle<Expression>, exprs: &crate::Arena<Expression>) -> String {
    match &exprs[handle] {
        Expression::Literal(lit) => format!("{lit}"),
        Expression::Compose { ty, components } => {
            let args: Vec<_> = components.iter().map(|h| format!("{h:?}")).collect();
            format!("Compose({ty:?}, [{}])", args.join(", "))
        }
        Expression::FunctionArgument(i) => format!("FunctionArgument({i})"),
        Expression::GlobalVariable(h) => format!("GlobalVariable({h:?})"),
        Expression::LocalVariable(h) => format!("LocalVariable({h:?})"),
        Expression::Load { pointer } => format!("Load({pointer:?})"),
        Expression::Access { base, index } => format!("Access({base:?}, {index:?})"),
        Expression::AccessIndex { base, index } => format!("AccessIndex({base:?}, {index})"),
        Expression::Swizzle {
            size,
            vector,
            pattern,
        } => {
            let n = *size as usize;
            let comps: Vec<_> = pattern[..n].iter().map(|c| format!("{c}")).collect();
            format!("Swizzle({vector:?}).{}", comps.join(""))
        }
        Expression::Splat { size, value } => format!("Splat({value:?}, vec{size})"),
        Expression::Unary { op, expr } => format!("{op}{expr:?}"),
        Expression::Binary { op, left, right } => format!("{left:?} {op} {right:?}"),
        Expression::Select {
            condition,
            accept,
            reject,
        } => format!("Select({condition:?}, {accept:?}, {reject:?})"),
        Expression::Math {
            fun,
            arg,
            arg1,
            arg2,
            arg3,
        } => {
            let mut args = format!("{arg:?}");
            if let Some(a1) = arg1 {
                args += &format!(", {a1:?}");
            }
            if let Some(a2) = arg2 {
                args += &format!(", {a2:?}");
            }
            if let Some(a3) = arg3 {
                args += &format!(", {a3:?}");
            }
            format!("{fun}({args})")
        }
        Expression::As {
            expr,
            kind,
            convert,
        } => match convert {
            Some(w) => format!("As({expr:?} -> {kind}/{w})"),
            None => format!("Bitcast({expr:?} -> {kind})"),
        },
        Expression::ArrayLength(expr) => format!("ArrayLength({expr:?})"),
        Expression::CallResult(f) => format!("CallResult({f:?})"),
        Expression::AtomicResult { ty, comparison } => {
            format!("AtomicResult({ty:?}, cmp={comparison})")
        }
        Expression::ZeroValue(ty) => format!("ZeroValue({ty:?})"),
    }
}

fn write_stmt(out: &mut String, stmt: &Statement, indent: usize) {
    let pad = " ".repeat(indent);
    match stmt {
        Statement::Emit(range) => {
            out.push_str(&format!("{pad}Emit({range:?})\n"));
        }
        Statement::Store { pointer, value } => {
            out.push_str(&format!("{pad}Store {pointer:?} = {value:?}\n"));
        }
        Statement::If {
            condition,
            accept,
            reject,
        } => {
            out.push_str(&format!("{pad}If ({condition:?}) {{\n"));
            for s in accept {
                write_stmt(out, s, indent + 4);
            }
            if !reject.is_empty() {
                out.push_str(&format!("{pad}}} else {{\n"));
                for s in reject {
                    write_stmt(out, s, indent + 4);
                }
            }
            out.push_str(&format!("{pad}}}\n"));
        }
        Statement::Loop {
            body,
            continuing,
            break_if,
        } => {
            out.push_str(&format!("{pad}Loop {{\n"));
            for s in body {
                write_stmt(out, s, indent + 4);
            }
            if !continuing.is_empty() {
                out.push_str(&format!("{pad}  Continuing {{\n"));
                for s in continuing {
                    write_stmt(out, s, indent + 8);
                }
                if let Some(brk) = break_if {
                    out.push_str(&format!("{pad}    BreakIf({brk:?})\n"));
                }
                out.push_str(&format!("{pad}  }}\n"));
            }
            out.push_str(&format!("{pad}}}\n"));
        }
        Statement::Call {
            function,
            arguments,
            result,
        } => {
            let args: Vec<_> = arguments.iter().map(|h| format!("{h:?}")).collect();
            let res = match result {
                Some(r) => format!(" -> {r:?}"),
                None => String::new(),
            };
            out.push_str(&format!(
                "{pad}Call {function:?}({}){res}\n",
                args.join(", ")
            ));
        }
        Statement::Atomic {
            pointer,
            fun,
            value,
            result,
        } => {
            let res = match result {
                Some(r) => format!(" -> {r:?}"),
                None => String::new(),
            };
            out.push_str(&format!("{pad}{fun}({pointer:?}, {value:?}){res}\n"));
        }
        Statement::Break => {
            out.push_str(&format!("{pad}Break\n"));
        }
        Statement::Continue => {
            out.push_str(&format!("{pad}Continue\n"));
        }
        Statement::Return { value } => match value {
            Some(v) => out.push_str(&format!("{pad}Return {v:?}\n")),
            None => out.push_str(&format!("{pad}Return\n")),
        },
        Statement::Barrier(b) => {
            out.push_str(&format!("{pad}Barrier({b})\n"));
        }
    }
}

/// Produces a human-readable text dump of a [`Module`] for debugging.
pub fn dump_module(module: &Module) -> String {
    let mut out = String::new();

    // Types
    out.push_str("Types:\n");
    for (handle, ty) in module.types.iter() {
        let formatted = format_type(ty, &module.types);
        out.push_str(&format!("  {handle:?} {formatted}\n"));
    }

    // Global variables
    if !module.global_variables.is_empty() {
        out.push_str("\nGlobal Variables:\n");
        for (handle, var) in module.global_variables.iter() {
            let name = var.name.as_deref().unwrap_or("_");
            let ty_str = format_type(&module.types[var.ty], &module.types);
            let binding_str = match &var.binding {
                Some(b) => format!("{b} "),
                None => String::new(),
            };
            out.push_str(&format!(
                "  {handle:?} {binding_str}var<{}>  {name}: {ty_str}\n",
                var.space
            ));
        }
    }

    // Global expressions
    if !module.global_expressions.is_empty() {
        out.push_str("\nGlobal Expressions:\n");
        for (handle, _) in module.global_expressions.iter() {
            let formatted = format_expr(handle, &module.global_expressions);
            out.push_str(&format!("  {handle:?} {formatted}\n"));
        }
    }

    // Helper functions
    if !module.functions.is_empty() {
        out.push_str("\nFunctions:\n");
        for (handle, func) in module.functions.iter() {
            dump_function(&mut out, &format!("{handle:?}"), func, &module.types);
        }
    }

    // Entry points
    if !module.entry_points.is_empty() {
        out.push_str("\nEntry Points:\n");
        for ep in &module.entry_points {
            let [x, y, z] = ep.workgroup_size;
            out.push_str(&format!("  @compute @workgroup_size({x}, {y}, {z})\n"));
            dump_function(&mut out, &ep.name, &ep.function, &module.types);
        }
    }

    out
}

fn dump_function(out: &mut String, label: &str, func: &crate::Function, types: &UniqueArena<Type>) {
    let name = func.name.as_deref().unwrap_or("_");

    // Signature
    let args: Vec<_> = func
        .arguments
        .iter()
        .map(|arg| {
            let arg_name = arg.name.as_deref().unwrap_or("_");
            let ty_str = format_type(&types[arg.ty], types);
            let binding = match &arg.binding {
                Some(b) => format!("{b} "),
                None => String::new(),
            };
            format!("{binding}{arg_name}: {ty_str}")
        })
        .collect();
    let ret = match &func.result {
        Some(r) => format!(" -> {}", format_type(&types[r.ty], types)),
        None => String::new(),
    };
    out.push_str(&format!(
        "  fn {name}({})  [{label}]{ret} {{\n",
        args.join(", ")
    ));

    // Local variables
    for (handle, var) in func.local_variables.iter() {
        let var_name = var.name.as_deref().unwrap_or("_");
        let ty_str = format_type(&types[var.ty], types);
        let init = match var.init {
            Some(h) => format!(" = {}", format_expr(h, &func.expressions)),
            None => String::new(),
        };
        out.push_str(&format!("    var {handle:?} {var_name}: {ty_str}{init}\n"));
    }

    // Expressions
    if !func.expressions.is_empty() {
        out.push_str("    Expressions:\n");
        for (handle, _) in func.expressions.iter() {
            let formatted = format_expr(handle, &func.expressions);
            out.push_str(&format!("      {handle:?} {formatted}\n"));
        }
    }

    // Body
    if !func.body.is_empty() {
        out.push_str("    Body:\n");
        for stmt in &func.body {
            write_stmt(out, stmt, 6);
        }
    }

    out.push_str("  }\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_scalar() {
        assert_eq!(format!("{}", Scalar::F32), "f32");
        assert_eq!(format!("{}", Scalar::I32), "i32");
        assert_eq!(format!("{}", Scalar::U32), "u32");
        assert_eq!(format!("{}", Scalar::F16), "f16");
        assert_eq!(format!("{}", Scalar::BOOL), "bool");
    }

    #[test]
    fn display_address_space() {
        assert_eq!(format!("{}", AddressSpace::Uniform), "uniform");
        assert_eq!(format!("{}", AddressSpace::Workgroup), "workgroup");
        assert_eq!(
            format!(
                "{}",
                AddressSpace::Storage {
                    access: StorageAccess::LOAD | StorageAccess::STORE
                }
            ),
            "storage, read_write"
        );
    }

    #[test]
    fn display_literal() {
        assert_eq!(format!("{}", Literal::F32(3.125)), "3.125f");
        assert_eq!(format!("{}", Literal::U32(42)), "42u");
        assert_eq!(format!("{}", Literal::Bool(true)), "true");
    }

    #[test]
    fn display_binary_op() {
        assert_eq!(format!("{}", BinaryOp::Add), "+");
        assert_eq!(format!("{}", BinaryOp::Equal), "==");
        assert_eq!(format!("{}", BinaryOp::ShiftLeft), "<<");
    }

    #[test]
    fn display_math_function() {
        assert_eq!(format!("{}", MathFunction::Dot), "dot");
        assert_eq!(format!("{}", MathFunction::Normalize), "normalize");
    }

    #[test]
    fn display_binding() {
        let b = Binding::BuiltIn(BuiltIn::GlobalInvocationId);
        assert_eq!(format!("{b}"), "@builtin(global_invocation_id)");
    }

    #[test]
    fn display_resource_binding() {
        let rb = ResourceBinding {
            group: 0,
            binding: 2,
        };
        assert_eq!(format!("{rb}"), "@group(0) @binding(2)");
    }

    #[test]
    fn dump_empty_module() {
        let module = Module::default();
        let dump = dump_module(&module);
        assert!(dump.contains("Types:"));
    }

    #[test]
    fn display_scalar_kind_all_variants() {
        assert_eq!(format!("{}", ScalarKind::Bool), "bool");
        assert_eq!(format!("{}", ScalarKind::Sint), "sint");
        assert_eq!(format!("{}", ScalarKind::Uint), "uint");
        assert_eq!(format!("{}", ScalarKind::Float), "float");
        assert_eq!(format!("{}", ScalarKind::BFloat), "bfloat");
    }

    #[test]
    fn display_scalar_bfloat() {
        assert_eq!(format!("{}", Scalar::BF16), "bf16");
    }

    #[test]
    fn display_storage_access_all_variants() {
        assert_eq!(format!("{}", StorageAccess::LOAD), "read");
        assert_eq!(format!("{}", StorageAccess::STORE), "write");
        assert_eq!(
            format!("{}", StorageAccess::LOAD | StorageAccess::STORE),
            "read_write"
        );
        assert_eq!(format!("{}", StorageAccess::EMPTY), "none");
    }

    #[test]
    fn display_address_space_all_variants() {
        assert_eq!(format!("{}", AddressSpace::Function), "function");
        assert_eq!(format!("{}", AddressSpace::Private), "private");
        assert_eq!(format!("{}", AddressSpace::Workgroup), "workgroup");
        assert_eq!(format!("{}", AddressSpace::Uniform), "uniform");
        assert_eq!(
            format!(
                "{}",
                AddressSpace::Storage {
                    access: StorageAccess::LOAD
                }
            ),
            "storage, read"
        );
    }

    #[test]
    fn display_builtin_all_variants() {
        assert_eq!(
            format!("{}", BuiltIn::GlobalInvocationId),
            "global_invocation_id"
        );
        assert_eq!(
            format!("{}", BuiltIn::LocalInvocationId),
            "local_invocation_id"
        );
        assert_eq!(
            format!("{}", BuiltIn::LocalInvocationIndex),
            "local_invocation_index"
        );
        assert_eq!(format!("{}", BuiltIn::WorkgroupId), "workgroup_id");
        assert_eq!(format!("{}", BuiltIn::NumWorkgroups), "num_workgroups");
    }

    #[test]
    fn display_binding_location() {
        let b = Binding::Location { location: 3 };
        assert_eq!(format!("{b}"), "@location(3)");
    }

    #[test]
    fn display_literal_all_variants() {
        assert_eq!(format!("{}", Literal::Bool(false)), "false");
        assert_eq!(format!("{}", Literal::I32(-7)), "-7i");
        assert_eq!(format!("{}", Literal::U32(42)), "42u");
        assert_eq!(format!("{}", Literal::F32(1.5)), "1.5f");
        assert_eq!(format!("{}", Literal::F64(2.5)), "2.5lf");
        assert_eq!(format!("{}", Literal::AbstractInt(99)), "99");
        assert_eq!(format!("{}", Literal::AbstractFloat(1.23)), "1.23");
    }

    #[test]
    fn display_unary_op_all_variants() {
        assert_eq!(format!("{}", UnaryOp::Negate), "-");
        assert_eq!(format!("{}", UnaryOp::LogicalNot), "!");
        assert_eq!(format!("{}", UnaryOp::BitwiseNot), "~");
    }

    #[test]
    fn display_binary_op_all_variants() {
        assert_eq!(format!("{}", BinaryOp::Add), "+");
        assert_eq!(format!("{}", BinaryOp::Subtract), "-");
        assert_eq!(format!("{}", BinaryOp::Multiply), "*");
        assert_eq!(format!("{}", BinaryOp::Divide), "/");
        assert_eq!(format!("{}", BinaryOp::Modulo), "%");
        assert_eq!(format!("{}", BinaryOp::Equal), "==");
        assert_eq!(format!("{}", BinaryOp::NotEqual), "!=");
        assert_eq!(format!("{}", BinaryOp::Less), "<");
        assert_eq!(format!("{}", BinaryOp::LessEqual), "<=");
        assert_eq!(format!("{}", BinaryOp::Greater), ">");
        assert_eq!(format!("{}", BinaryOp::GreaterEqual), ">=");
        assert_eq!(format!("{}", BinaryOp::LogicalAnd), "&&");
        assert_eq!(format!("{}", BinaryOp::LogicalOr), "||");
        assert_eq!(format!("{}", BinaryOp::BitwiseAnd), "&");
        assert_eq!(format!("{}", BinaryOp::BitwiseOr), "|");
        assert_eq!(format!("{}", BinaryOp::BitwiseXor), "^");
        assert_eq!(format!("{}", BinaryOp::ShiftLeft), "<<");
        assert_eq!(format!("{}", BinaryOp::ShiftRight), ">>");
    }

    #[test]
    fn display_swizzle_component_all_variants() {
        assert_eq!(format!("{}", SwizzleComponent::X), "x");
        assert_eq!(format!("{}", SwizzleComponent::Y), "y");
        assert_eq!(format!("{}", SwizzleComponent::Z), "z");
        assert_eq!(format!("{}", SwizzleComponent::W), "w");
    }

    #[test]
    fn display_atomic_function_all_variants() {
        assert_eq!(format!("{}", AtomicFunction::Add), "atomicAdd");
        assert_eq!(format!("{}", AtomicFunction::Subtract), "atomicSub");
        assert_eq!(format!("{}", AtomicFunction::And), "atomicAnd");
        assert_eq!(format!("{}", AtomicFunction::ExclusiveOr), "atomicXor");
        assert_eq!(format!("{}", AtomicFunction::InclusiveOr), "atomicOr");
        assert_eq!(format!("{}", AtomicFunction::Min), "atomicMin");
        assert_eq!(format!("{}", AtomicFunction::Max), "atomicMax");
        assert_eq!(
            format!("{}", AtomicFunction::Exchange { compare: None }),
            "atomicExchange"
        );
        let cmp_handle = {
            let mut arena = crate::Arena::new();
            arena.append(Expression::Literal(Literal::U32(0)))
        };
        assert!(
            format!(
                "{}",
                AtomicFunction::Exchange {
                    compare: Some(cmp_handle)
                }
            )
            .starts_with("atomicCompareExchange(")
        );
    }

    #[test]
    fn display_barrier_all_variants() {
        assert_eq!(format!("{}", Barrier::STORAGE), "storageBarrier");
        assert_eq!(format!("{}", Barrier::WORKGROUP), "workgroupBarrier");
        assert_eq!(
            format!("{}", Barrier::STORAGE | Barrier::WORKGROUP),
            "storageBarrier | workgroupBarrier"
        );
        assert_eq!(format!("{}", Barrier::EMPTY), "<no barrier>");
    }

    #[test]
    fn format_type_named() {
        let mut types = UniqueArena::new();
        let h = types.insert(Type {
            name: Some("MyStruct".into()),
            inner: TypeInner::Scalar(Scalar::F32),
        });
        assert_eq!(format_type(&types[h], &types), "MyStruct");
    }

    #[test]
    fn format_type_inner_all_variants() {
        let mut types = UniqueArena::new();
        let f32_ty = types.insert(Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::F32),
        });

        // Scalar
        assert_eq!(
            format_type_inner(&TypeInner::Scalar(Scalar::F32), &types),
            "f32"
        );

        // Vector
        assert_eq!(
            format_type_inner(
                &TypeInner::Vector {
                    size: VectorSize::Tri,
                    scalar: Scalar::F32
                },
                &types
            ),
            "vec3<f32>"
        );

        // Matrix
        assert_eq!(
            format_type_inner(
                &TypeInner::Matrix {
                    columns: VectorSize::Quad,
                    rows: VectorSize::Quad,
                    scalar: Scalar::F32
                },
                &types
            ),
            "mat4x4<f32>"
        );

        // Atomic
        assert_eq!(
            format_type_inner(&TypeInner::Atomic(Scalar::U32), &types),
            "atomic<u32>"
        );

        // Pointer
        assert_eq!(
            format_type_inner(
                &TypeInner::Pointer {
                    base: f32_ty,
                    space: AddressSpace::Function
                },
                &types
            ),
            "ptr<function, f32>"
        );

        // Array (constant)
        assert_eq!(
            format_type_inner(
                &TypeInner::Array {
                    base: f32_ty,
                    size: ArraySize::Constant(16),
                    stride: 4
                },
                &types
            ),
            "array<f32, 16> /*stride 4*/"
        );

        // Array (dynamic)
        assert_eq!(
            format_type_inner(
                &TypeInner::Array {
                    base: f32_ty,
                    size: ArraySize::Dynamic,
                    stride: 4
                },
                &types
            ),
            "array<f32> /*stride 4*/"
        );

        // Struct
        assert_eq!(
            format_type_inner(
                &TypeInner::Struct {
                    members: vec![],
                    span: 16
                },
                &types
            ),
            "struct(0 members, span 16)"
        );

        // Tensor (mixed dims)
        assert_eq!(
            format_type_inner(
                &TypeInner::Tensor {
                    scalar: Scalar::F32,
                    shape: crate::TensorShape {
                        dims: vec![
                            crate::Dimension::Fixed(224),
                            crate::Dimension::Dynamic(Some("batch".into())),
                            crate::Dimension::Dynamic(None),
                        ],
                    }
                },
                &types
            ),
            "tensor<f32>[224, batch, ?]"
        );
    }

    #[test]
    fn dump_module_with_globals_and_entry_point() {
        use crate::{EntryPoint, Expression, Function, GlobalVariable, ResourceBinding, Statement};

        let mut module = Module::default();

        let f32_ty = module.types.insert(Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::F32),
        });
        let arr_ty = module.types.insert(Type {
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
            ty: arr_ty,
            init: None,
            layout: None,
        });

        module.global_variables.append(GlobalVariable {
            name: None,
            space: AddressSpace::Private,
            binding: None,
            ty: f32_ty,
            init: None,
            layout: None,
        });

        let mut func = Function::new("main");
        let lit = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        let gv = func.expressions.append(Expression::GlobalVariable(
            module.global_variables.next_handle(),
        ));
        func.body.push(Statement::Store {
            pointer: gv,
            value: lit,
        });

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [256, 1, 1],
            function: func,
        });

        let dump = dump_module(&module);
        assert!(dump.contains("Global Variables:"));
        assert!(dump.contains("@group(0) @binding(0)"));
        assert!(dump.contains("a:"));
        assert!(dump.contains("storage, read"));
        assert!(dump.contains("Entry Points:"));
        assert!(dump.contains("@compute @workgroup_size(256, 1, 1)"));
        assert!(dump.contains("Expressions:"));
        assert!(dump.contains("Body:"));
        assert!(dump.contains("Store"));
    }

    #[test]
    fn dump_module_with_if_and_loop() {
        use crate::{EntryPoint, Expression, Function, Statement};

        let mut module = Module::default();
        let mut func = Function::new("main");

        let cond = func
            .expressions
            .append(Expression::Literal(Literal::Bool(true)));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));

        func.body.push(Statement::If {
            condition: cond,
            accept: vec![Statement::Return { value: Some(val) }],
            reject: vec![Statement::Return { value: None }],
        });
        func.body.push(Statement::Loop {
            body: vec![Statement::Break],
            continuing: vec![Statement::Continue],
            break_if: Some(cond),
        });
        func.body.push(Statement::Barrier(Barrier::STORAGE));

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });

        let dump = dump_module(&module);
        assert!(dump.contains("If ("));
        assert!(dump.contains("} else {"));
        assert!(dump.contains("Return"));
        assert!(dump.contains("Loop {"));
        assert!(dump.contains("Continuing {"));
        assert!(dump.contains("BreakIf("));
        assert!(dump.contains("Break"));
        assert!(dump.contains("Continue"));
        assert!(dump.contains("Barrier(storageBarrier)"));
    }

    #[test]
    fn dump_module_with_helper_function() {
        use crate::{Function, FunctionArgument, FunctionResult};

        let mut module = Module::default();
        let f32_ty = module.types.insert(Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::F32),
        });

        let mut func = Function::new("helper");
        func.arguments.push(FunctionArgument {
            name: Some("x".into()),
            ty: f32_ty,
            binding: None,
        });
        func.result = Some(FunctionResult {
            ty: f32_ty,
            binding: None,
        });
        module.functions.append(func);

        let dump = dump_module(&module);
        assert!(dump.contains("Functions:"));
        assert!(dump.contains("fn helper(x: f32)"));
        assert!(dump.contains("-> f32"));
    }

    #[test]
    fn display_vector_size() {
        assert_eq!(format!("{}", VectorSize::Bi), "2");
        assert_eq!(format!("{}", VectorSize::Tri), "3");
        assert_eq!(format!("{}", VectorSize::Quad), "4");
    }

    #[test]
    fn display_math_function_all_variants() {
        // Spot-check a representative sample beyond what existing tests cover
        assert_eq!(format!("{}", MathFunction::Abs), "abs");
        assert_eq!(format!("{}", MathFunction::Clamp), "clamp");
        assert_eq!(format!("{}", MathFunction::Fma), "fma");
        assert_eq!(format!("{}", MathFunction::Sin), "sin");
        assert_eq!(format!("{}", MathFunction::Tanh), "tanh");
        assert_eq!(format!("{}", MathFunction::Sqrt), "sqrt");
        assert_eq!(format!("{}", MathFunction::Pow), "pow");
        assert_eq!(format!("{}", MathFunction::Mix), "mix");
        assert_eq!(format!("{}", MathFunction::SmoothStep), "smoothStep");
        assert_eq!(format!("{}", MathFunction::InverseSqrt), "inverseSqrt");
    }
}
