//! Functions, entry points, and local variables.

use std::collections::HashMap;

use crate::arena::{Arena, Handle};
use crate::expr::Expression;
use crate::global::Binding;
use crate::stmt::Block;
use crate::types::Type;

/// A function argument declaration.
#[derive(Clone, Debug)]
pub struct FunctionArgument {
    /// Optional argument name.
    pub name: Option<String>,
    /// The type of this argument.
    pub ty: Handle<Type>,
    /// Optional binding (e.g. built-in or location).
    pub binding: Option<Binding>,
}

/// The return type and optional binding of a function.
#[derive(Clone, Debug)]
pub struct FunctionResult {
    /// The return type.
    pub ty: Handle<Type>,
    /// Optional binding for the return value.
    pub binding: Option<Binding>,
}

/// A function-local variable.
#[derive(Clone, Debug)]
pub struct LocalVariable {
    /// Optional variable name.
    pub name: Option<String>,
    /// The type of this variable.
    pub ty: Handle<Type>,
    /// Optional initializer expression.
    pub init: Option<Handle<Expression>>,
}

/// An IR function.
#[derive(Clone, Debug)]
pub struct Function {
    /// Optional function name.
    pub name: Option<String>,
    /// Formal parameters.
    pub arguments: Vec<FunctionArgument>,
    /// Return type and optional binding.
    pub result: Option<FunctionResult>,
    /// Function-local variable declarations.
    pub local_variables: Arena<LocalVariable>,
    /// Expression arena for this function.
    pub expressions: Arena<Expression>,
    /// Map from expression handles to user-defined names.
    pub named_expressions: HashMap<Handle<Expression>, String>,
    /// The function body.
    pub body: Block,
}

impl Function {
    /// Creates an empty function with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            arguments: Vec::new(),
            result: None,
            local_variables: Arena::new(),
            expressions: Arena::new(),
            named_expressions: HashMap::new(),
            body: Vec::new(),
        }
    }
}

/// A compute shader entry point.
#[derive(Clone, Debug)]
pub struct EntryPoint {
    /// Entry point name (matches the WGSL function name).
    pub name: String,
    /// Workgroup dimensions `[x, y, z]`.
    pub workgroup_size: [u32; 3],
    /// The entry point function body.
    pub function: Function,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Literal;
    use crate::types::{Scalar, TypeInner};

    #[test]
    fn function_new() {
        let f = Function::new("test");
        assert_eq!(f.name.as_deref(), Some("test"));
        assert!(f.arguments.is_empty());
        assert!(f.result.is_none());
        assert!(f.body.is_empty());
        assert!(f.expressions.is_empty());
    }

    #[test]
    fn function_with_local_vars() {
        let mut types = crate::arena::UniqueArena::new();
        let f32_ty = types.insert(Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::F32),
        });

        let mut f = Function::new("test");
        let init = f.expressions.append(Expression::Literal(Literal::F32(0.0)));
        let _var = f.local_variables.append(LocalVariable {
            name: Some("sum".into()),
            ty: f32_ty,
            init: Some(init),
        });
        assert_eq!(f.local_variables.len(), 1);
    }

    #[test]
    fn entry_point() {
        let ep = EntryPoint {
            name: "main".into(),
            workgroup_size: [256, 1, 1],
            function: Function::new("main"),
        };
        assert_eq!(ep.workgroup_size, [256, 1, 1]);
    }
}
