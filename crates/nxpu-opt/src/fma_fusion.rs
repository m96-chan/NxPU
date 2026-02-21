//! FMA (fused multiply-add) fusion pass.
//!
//! Detects `a * b + c` patterns and replaces them with `fma(a, b, c)`.
//! This is a common NPU/GPU optimization that reduces two operations to one.

use nxpu_ir::{BinaryOp, Expression, Function, Handle, MathFunction, Module};

use crate::Pass;

/// Fuses multiply-add patterns into `fma` math operations.
#[derive(Debug)]
pub struct FmaFusion;

impl Pass for FmaFusion {
    fn name(&self) -> &str {
        "fma-fusion"
    }

    fn run(&self, module: &mut Module) -> bool {
        let mut changed = false;
        for (_, func) in module.functions.iter_mut() {
            changed |= run_on_function(func);
        }
        for ep in &mut module.entry_points {
            changed |= run_on_function(&mut ep.function);
        }
        changed
    }
}

fn run_on_function(func: &mut Function) -> bool {
    let mut changed = false;

    let handles: Vec<Handle<Expression>> = func.expressions.iter().map(|(h, _)| h).collect();

    for handle in handles {
        let replacement = match &func.expressions[handle] {
            Expression::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => try_fuse_fma(&func.expressions, *left, *right),
            _ => None,
        };

        if let Some(new_expr) = replacement {
            func.expressions[handle] = new_expr;
            changed = true;
        }
    }

    changed
}

/// Checks if `left + right` can be fused into `fma(a, b, c)`.
///
/// Matches:
/// - `(a * b) + c`  →  `fma(a, b, c)`
/// - `c + (a * b)`  →  `fma(a, b, c)`
fn try_fuse_fma(
    exprs: &nxpu_ir::Arena<Expression>,
    left: Handle<Expression>,
    right: Handle<Expression>,
) -> Option<Expression> {
    // Pattern 1: left = a * b, addend = right
    if let Expression::Binary {
        op: BinaryOp::Multiply,
        left: a,
        right: b,
    } = &exprs[left]
    {
        return Some(Expression::Math {
            fun: MathFunction::Fma,
            arg: *a,
            arg1: Some(*b),
            arg2: Some(right),
            arg3: None,
        });
    }

    // Pattern 2: right = a * b, addend = left
    if let Expression::Binary {
        op: BinaryOp::Multiply,
        left: a,
        right: b,
    } = &exprs[right]
    {
        return Some(Expression::Math {
            fun: MathFunction::Fma,
            arg: *a,
            arg1: Some(*b),
            arg2: Some(left),
            arg3: None,
        });
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use nxpu_ir::Literal;

    #[test]
    fn fuse_mul_add() {
        let mut func = Function::new("test");
        let a = func
            .expressions
            .append(Expression::Literal(Literal::F32(2.0)));
        let b = func
            .expressions
            .append(Expression::Literal(Literal::F32(3.0)));
        let c = func
            .expressions
            .append(Expression::Literal(Literal::F32(4.0)));
        let mul = func.expressions.append(Expression::Binary {
            op: BinaryOp::Multiply,
            left: a,
            right: b,
        });
        let add = func.expressions.append(Expression::Binary {
            op: BinaryOp::Add,
            left: mul,
            right: c,
        });

        let changed = run_on_function(&mut func);
        assert!(changed);
        match &func.expressions[add] {
            Expression::Math {
                fun: MathFunction::Fma,
                arg,
                arg1,
                arg2,
                arg3,
            } => {
                assert_eq!(*arg, a);
                assert_eq!(*arg1, Some(b));
                assert_eq!(*arg2, Some(c));
                assert!(arg3.is_none());
            }
            other => panic!("expected Fma, got {other:?}"),
        }
    }

    #[test]
    fn fuse_commuted_add_mul() {
        // c + (a * b)
        let mut func = Function::new("test");
        let a = func
            .expressions
            .append(Expression::Literal(Literal::F32(2.0)));
        let b = func
            .expressions
            .append(Expression::Literal(Literal::F32(3.0)));
        let c = func
            .expressions
            .append(Expression::Literal(Literal::F32(4.0)));
        let mul = func.expressions.append(Expression::Binary {
            op: BinaryOp::Multiply,
            left: a,
            right: b,
        });
        let add = func.expressions.append(Expression::Binary {
            op: BinaryOp::Add,
            left: c,
            right: mul,
        });

        let changed = run_on_function(&mut func);
        assert!(changed);
        match &func.expressions[add] {
            Expression::Math {
                fun: MathFunction::Fma,
                arg,
                arg1,
                arg2,
                arg3,
            } => {
                assert_eq!(*arg, a);
                assert_eq!(*arg1, Some(b));
                assert_eq!(*arg2, Some(c));
                assert!(arg3.is_none());
            }
            other => panic!("expected Fma, got {other:?}"),
        }
    }

    #[test]
    fn no_fusion_without_multiply() {
        let mut func = Function::new("test");
        let a = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        let b = func
            .expressions
            .append(Expression::Literal(Literal::F32(2.0)));
        let _add = func.expressions.append(Expression::Binary {
            op: BinaryOp::Add,
            left: a,
            right: b,
        });

        let changed = run_on_function(&mut func);
        assert!(!changed);
    }

    #[test]
    fn no_fusion_on_subtract() {
        let mut func = Function::new("test");
        let a = func
            .expressions
            .append(Expression::Literal(Literal::F32(2.0)));
        let b = func
            .expressions
            .append(Expression::Literal(Literal::F32(3.0)));
        let c = func
            .expressions
            .append(Expression::Literal(Literal::F32(4.0)));
        let mul = func.expressions.append(Expression::Binary {
            op: BinaryOp::Multiply,
            left: a,
            right: b,
        });
        let _sub = func.expressions.append(Expression::Binary {
            op: BinaryOp::Subtract,
            left: mul,
            right: c,
        });

        let changed = run_on_function(&mut func);
        assert!(!changed);
    }
}
