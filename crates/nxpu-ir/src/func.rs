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
    pub name: Option<String>,
    pub ty: Handle<Type>,
    pub binding: Option<Binding>,
}

/// The return type and optional binding of a function.
#[derive(Clone, Debug)]
pub struct FunctionResult {
    pub ty: Handle<Type>,
    pub binding: Option<Binding>,
}

/// A function-local variable.
#[derive(Clone, Debug)]
pub struct LocalVariable {
    pub name: Option<String>,
    pub ty: Handle<Type>,
    pub init: Option<Handle<Expression>>,
}

/// An IR function.
#[derive(Clone, Debug)]
pub struct Function {
    pub name: Option<String>,
    pub arguments: Vec<FunctionArgument>,
    pub result: Option<FunctionResult>,
    pub local_variables: Arena<LocalVariable>,
    pub expressions: Arena<Expression>,
    pub named_expressions: HashMap<Handle<Expression>, String>,
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
    pub name: String,
    pub workgroup_size: [u32; 3],
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
        let f32_ty = types.insert(crate::types::Type {
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
