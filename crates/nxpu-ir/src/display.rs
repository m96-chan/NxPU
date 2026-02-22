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
}
