//! Expressions — pure SSA values with no side effects.

use crate::arena::Handle;
use crate::types::{Bytes, Scalar, ScalarKind, Type, VectorSize};

/// A vector swizzle component.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum SwizzleComponent {
    X = 0,
    Y = 1,
    Z = 2,
    W = 3,
}

/// A literal constant value.
#[derive(Clone, Copy, Debug)]
pub enum Literal {
    Bool(bool),
    I32(i32),
    U32(u32),
    F32(f32),
    F64(f64),
    AbstractInt(i64),
    AbstractFloat(f64),
}

impl Literal {
    /// Returns the scalar type of this literal.
    pub fn scalar(&self) -> Scalar {
        match *self {
            Self::Bool(_) => Scalar::BOOL,
            Self::I32(_) => Scalar::I32,
            Self::U32(_) => Scalar::U32,
            Self::F32(_) => Scalar::F32,
            Self::F64(_) => Scalar {
                kind: ScalarKind::Float,
                width: 8,
            },
            Self::AbstractInt(_) => Scalar {
                kind: ScalarKind::Sint,
                width: 8,
            },
            Self::AbstractFloat(_) => Scalar {
                kind: ScalarKind::Float,
                width: 8,
            },
        }
    }
}

/// A unary operator.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum UnaryOp {
    Negate,
    LogicalNot,
    BitwiseNot,
}

/// A binary operator.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    LogicalAnd,
    LogicalOr,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    ShiftLeft,
    ShiftRight,
}

/// A built-in math function.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum MathFunction {
    // Component-wise
    Abs,
    Min,
    Max,
    Clamp,
    Saturate,
    // Rounding
    Floor,
    Ceil,
    Round,
    Fract,
    Trunc,
    // Trigonometric
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Atan2,
    Sinh,
    Cosh,
    Tanh,
    // Exponential
    Sqrt,
    InverseSqrt,
    Log,
    Log2,
    Exp,
    Exp2,
    Pow,
    // Linear algebra
    Dot,
    Cross,
    Normalize,
    Length,
    Distance,
    // Interpolation
    Mix,
    Step,
    SmoothStep,
    // Fused multiply-add
    Fma,
}

/// An atomic operation.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum AtomicFunction {
    Add,
    Subtract,
    And,
    ExclusiveOr,
    InclusiveOr,
    Min,
    Max,
    Exchange { compare: Option<Handle<Expression>> },
}

/// An expression in the IR — a pure SSA value with no side effects.
///
/// Expressions are stored in per-function or module-level arenas.
/// They are referenced by [`Handle<Expression>`].
#[derive(Clone, Debug)]
pub enum Expression {
    /// A literal constant.
    Literal(Literal),
    /// Construct a composite type from components.
    Compose {
        ty: Handle<Type>,
        components: Vec<Handle<Expression>>,
    },
    /// Reference to a function argument by index.
    FunctionArgument(u32),
    /// Reference to a global variable (produces a pointer).
    GlobalVariable(Handle<crate::GlobalVariable>),
    /// Reference to a local variable (produces a pointer).
    LocalVariable(Handle<crate::LocalVariable>),
    /// Load a value through a pointer.
    Load { pointer: Handle<Expression> },
    /// Dynamic index into a composite (array, vector, matrix).
    Access {
        base: Handle<Expression>,
        index: Handle<Expression>,
    },
    /// Static index into a composite.
    AccessIndex {
        base: Handle<Expression>,
        index: u32,
    },
    /// Swizzle vector components.
    Swizzle {
        size: VectorSize,
        vector: Handle<Expression>,
        pattern: [SwizzleComponent; 4],
    },
    /// Broadcast a scalar to a vector.
    Splat {
        size: VectorSize,
        value: Handle<Expression>,
    },
    /// Apply a unary operator.
    Unary {
        op: UnaryOp,
        expr: Handle<Expression>,
    },
    /// Apply a binary operator.
    Binary {
        op: BinaryOp,
        left: Handle<Expression>,
        right: Handle<Expression>,
    },
    /// Select between two values based on a condition.
    Select {
        condition: Handle<Expression>,
        accept: Handle<Expression>,
        reject: Handle<Expression>,
    },
    /// Call a built-in math function.
    Math {
        fun: MathFunction,
        arg: Handle<Expression>,
        arg1: Option<Handle<Expression>>,
        arg2: Option<Handle<Expression>>,
        arg3: Option<Handle<Expression>>,
    },
    /// Type cast / bitcast.
    As {
        expr: Handle<Expression>,
        kind: ScalarKind,
        convert: Option<Bytes>,
    },
    /// Get the length of a runtime-sized array.
    ArrayLength(Handle<Expression>),
    /// The result of a function call (paired with a `Call` statement).
    CallResult(Handle<crate::Function>),
    /// The result of an atomic operation (paired with an `Atomic` statement).
    AtomicResult { ty: Handle<Type>, comparison: bool },
    /// A zero-initialized value of the given type.
    ///
    /// Used for vector, matrix, struct, and array zero-initialization where a
    /// simple scalar literal would be type-incorrect.
    ZeroValue(Handle<Type>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;

    #[test]
    fn literal_scalars() {
        assert_eq!(Literal::F32(1.0).scalar(), Scalar::F32);
        assert_eq!(Literal::I32(-1).scalar(), Scalar::I32);
        assert_eq!(Literal::U32(42).scalar(), Scalar::U32);
        assert_eq!(Literal::Bool(true).scalar(), Scalar::BOOL);
    }

    #[test]
    fn expression_arena() {
        let mut exprs = Arena::new();
        let lit = exprs.append(Expression::Literal(Literal::F32(3.125)));
        let neg = exprs.append(Expression::Unary {
            op: UnaryOp::Negate,
            expr: lit,
        });
        assert_eq!(lit.index(), 0);
        assert_eq!(neg.index(), 1);
        assert_eq!(exprs.len(), 2);
    }

    #[test]
    fn binary_expression() {
        let mut exprs = Arena::new();
        let left = exprs.append(Expression::Literal(Literal::F32(1.0)));
        let right = exprs.append(Expression::Literal(Literal::F32(2.0)));
        let add = exprs.append(Expression::Binary {
            op: BinaryOp::Add,
            left,
            right,
        });
        if let Expression::Binary { op, .. } = &exprs[add] {
            assert_eq!(*op, BinaryOp::Add);
        } else {
            panic!("expected Binary");
        }
    }
}
