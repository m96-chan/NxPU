//! Statements — operations with side effects and control flow.

use crate::arena::{Handle, Range};
use crate::expr::{AtomicFunction, Expression};

/// A block of statements.
pub type Block = Vec<Statement>;

/// Bitflags for synchronization barriers.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct Barrier(u32);

impl Barrier {
    /// Empty barrier (no flags set).
    pub const EMPTY: Self = Self(0);
    /// Storage buffer barrier.
    pub const STORAGE: Self = Self(1);
    /// Workgroup memory barrier.
    pub const WORKGROUP: Self = Self(2);

    /// Returns `true` if `self` contains all flags in `other`.
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Returns `true` if no flags are set.
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl std::ops::BitOr for Barrier {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for Barrier {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// A statement in the IR.
///
/// Statements have side effects and/or control flow.
/// They operate on expressions referenced by handles.
#[derive(Clone, Debug)]
pub enum Statement {
    /// Mark a range of expressions as producing live values.
    Emit(Range<Expression>),
    /// Write a value through a pointer.
    Store {
        pointer: Handle<Expression>,
        value: Handle<Expression>,
    },
    /// Conditional branch.
    If {
        condition: Handle<Expression>,
        accept: Block,
        reject: Block,
    },
    /// Unified loop construct (handles for/while/loop).
    Loop {
        body: Block,
        continuing: Block,
        break_if: Option<Handle<Expression>>,
    },
    /// Call a function.
    Call {
        function: Handle<crate::Function>,
        arguments: Vec<Handle<Expression>>,
        result: Option<Handle<Expression>>,
    },
    /// Perform an atomic operation.
    Atomic {
        pointer: Handle<Expression>,
        fun: AtomicFunction,
        value: Handle<Expression>,
        result: Option<Handle<Expression>>,
    },
    /// Break out of the innermost loop.
    Break,
    /// Continue to the next iteration of the innermost loop.
    Continue,
    /// Return from the function.
    Return { value: Option<Handle<Expression>> },
    /// Synchronization barrier.
    Barrier(Barrier),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;
    use crate::expr::Literal;

    #[test]
    fn barrier_flags() {
        let storage = Barrier::STORAGE;
        let workgroup = Barrier::WORKGROUP;
        let both = storage | workgroup;
        assert!(both.contains(storage));
        assert!(both.contains(workgroup));
        assert!(!storage.contains(workgroup));
    }

    #[test]
    fn build_if_statement() {
        let mut exprs = Arena::new();
        let cond = exprs.append(Expression::Literal(Literal::Bool(true)));
        let stmt = Statement::If {
            condition: cond,
            accept: vec![Statement::Break],
            reject: vec![],
        };
        if let Statement::If { accept, reject, .. } = &stmt {
            assert_eq!(accept.len(), 1);
            assert!(reject.is_empty());
        } else {
            panic!("expected If");
        }
    }

    #[test]
    fn build_loop_statement() {
        let stmt = Statement::Loop {
            body: vec![Statement::Continue],
            continuing: vec![],
            break_if: None,
        };
        if let Statement::Loop {
            body, continuing, ..
        } = &stmt
        {
            assert_eq!(body.len(), 1);
            assert!(continuing.is_empty());
        } else {
            panic!("expected Loop");
        }
    }
}
