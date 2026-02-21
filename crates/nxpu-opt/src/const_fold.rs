//! Constant folding pass.
//!
//! Evaluates binary and unary operations on literal operands at compile time,
//! replacing them with the resulting literal in the expression arena.

use nxpu_ir::{BinaryOp, Expression, Function, Handle, Literal, Module, UnaryOp};

use crate::Pass;

/// Folds constant expressions (literal operands) at compile time.
#[derive(Debug)]
pub struct ConstantFolding;

impl Pass for ConstantFolding {
    fn name(&self) -> &str {
        "const-fold"
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

    // Collect handles first to avoid borrowing the arena while mutating.
    let handles: Vec<Handle<Expression>> = func.expressions.iter().map(|(h, _)| h).collect();

    for handle in handles {
        let replacement = match &func.expressions[handle] {
            Expression::Binary { op, left, right } => {
                let left_val = &func.expressions[*left];
                let right_val = &func.expressions[*right];
                if let (Expression::Literal(l), Expression::Literal(r)) = (left_val, right_val) {
                    fold_binary(*op, *l, *r).map(Expression::Literal)
                } else {
                    None
                }
            }
            Expression::Unary { op, expr } => {
                if let Expression::Literal(lit) = &func.expressions[*expr] {
                    fold_unary(*op, *lit).map(Expression::Literal)
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(new_expr) = replacement {
            func.expressions[handle] = new_expr;
            changed = true;
        }
    }

    changed
}

fn fold_binary(op: BinaryOp, left: Literal, right: Literal) -> Option<Literal> {
    match (left, right) {
        (Literal::F32(l), Literal::F32(r)) => fold_f32(op, l, r),
        (Literal::I32(l), Literal::I32(r)) => fold_i32(op, l, r),
        (Literal::U32(l), Literal::U32(r)) => fold_u32(op, l, r),
        (Literal::Bool(l), Literal::Bool(r)) => fold_bool(op, l, r),
        _ => None,
    }
}

fn fold_f32(op: BinaryOp, l: f32, r: f32) -> Option<Literal> {
    match op {
        BinaryOp::Add => Some(Literal::F32(l + r)),
        BinaryOp::Subtract => Some(Literal::F32(l - r)),
        BinaryOp::Multiply => Some(Literal::F32(l * r)),
        BinaryOp::Divide => Some(Literal::F32(l / r)),
        BinaryOp::Modulo => Some(Literal::F32(l % r)),
        BinaryOp::Equal => Some(Literal::Bool(l == r)),
        BinaryOp::NotEqual => Some(Literal::Bool(l != r)),
        BinaryOp::Less => Some(Literal::Bool(l < r)),
        BinaryOp::LessEqual => Some(Literal::Bool(l <= r)),
        BinaryOp::Greater => Some(Literal::Bool(l > r)),
        BinaryOp::GreaterEqual => Some(Literal::Bool(l >= r)),
        _ => None,
    }
}

fn fold_i32(op: BinaryOp, l: i32, r: i32) -> Option<Literal> {
    match op {
        BinaryOp::Add => Some(Literal::I32(l.wrapping_add(r))),
        BinaryOp::Subtract => Some(Literal::I32(l.wrapping_sub(r))),
        BinaryOp::Multiply => Some(Literal::I32(l.wrapping_mul(r))),
        BinaryOp::Divide if r != 0 => Some(Literal::I32(l.wrapping_div(r))),
        BinaryOp::Modulo if r != 0 => Some(Literal::I32(l.wrapping_rem(r))),
        BinaryOp::Equal => Some(Literal::Bool(l == r)),
        BinaryOp::NotEqual => Some(Literal::Bool(l != r)),
        BinaryOp::Less => Some(Literal::Bool(l < r)),
        BinaryOp::LessEqual => Some(Literal::Bool(l <= r)),
        BinaryOp::Greater => Some(Literal::Bool(l > r)),
        BinaryOp::GreaterEqual => Some(Literal::Bool(l >= r)),
        BinaryOp::BitwiseAnd => Some(Literal::I32(l & r)),
        BinaryOp::BitwiseOr => Some(Literal::I32(l | r)),
        BinaryOp::BitwiseXor => Some(Literal::I32(l ^ r)),
        BinaryOp::ShiftLeft => Some(Literal::I32(l.wrapping_shl(r as u32))),
        BinaryOp::ShiftRight => Some(Literal::I32(l.wrapping_shr(r as u32))),
        _ => None,
    }
}

fn fold_u32(op: BinaryOp, l: u32, r: u32) -> Option<Literal> {
    match op {
        BinaryOp::Add => Some(Literal::U32(l.wrapping_add(r))),
        BinaryOp::Subtract => Some(Literal::U32(l.wrapping_sub(r))),
        BinaryOp::Multiply => Some(Literal::U32(l.wrapping_mul(r))),
        BinaryOp::Divide if r != 0 => Some(Literal::U32(l / r)),
        BinaryOp::Modulo if r != 0 => Some(Literal::U32(l % r)),
        BinaryOp::Equal => Some(Literal::Bool(l == r)),
        BinaryOp::NotEqual => Some(Literal::Bool(l != r)),
        BinaryOp::Less => Some(Literal::Bool(l < r)),
        BinaryOp::LessEqual => Some(Literal::Bool(l <= r)),
        BinaryOp::Greater => Some(Literal::Bool(l > r)),
        BinaryOp::GreaterEqual => Some(Literal::Bool(l >= r)),
        BinaryOp::BitwiseAnd => Some(Literal::U32(l & r)),
        BinaryOp::BitwiseOr => Some(Literal::U32(l | r)),
        BinaryOp::BitwiseXor => Some(Literal::U32(l ^ r)),
        BinaryOp::ShiftLeft => Some(Literal::U32(l.wrapping_shl(r))),
        BinaryOp::ShiftRight => Some(Literal::U32(l.wrapping_shr(r))),
        _ => None,
    }
}

fn fold_bool(op: BinaryOp, l: bool, r: bool) -> Option<Literal> {
    match op {
        BinaryOp::Equal => Some(Literal::Bool(l == r)),
        BinaryOp::NotEqual => Some(Literal::Bool(l != r)),
        BinaryOp::LogicalAnd => Some(Literal::Bool(l && r)),
        BinaryOp::LogicalOr => Some(Literal::Bool(l || r)),
        _ => None,
    }
}

fn fold_unary(op: UnaryOp, lit: Literal) -> Option<Literal> {
    match (op, lit) {
        (UnaryOp::Negate, Literal::F32(v)) => Some(Literal::F32(-v)),
        (UnaryOp::Negate, Literal::I32(v)) => Some(Literal::I32(v.wrapping_neg())),
        (UnaryOp::LogicalNot, Literal::Bool(v)) => Some(Literal::Bool(!v)),
        (UnaryOp::BitwiseNot, Literal::I32(v)) => Some(Literal::I32(!v)),
        (UnaryOp::BitwiseNot, Literal::U32(v)) => Some(Literal::U32(!v)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nxpu_ir::Literal;

    #[test]
    fn fold_f32_add() {
        let mut func = Function::new("test");
        let one = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        let two = func
            .expressions
            .append(Expression::Literal(Literal::F32(2.0)));
        let add = func.expressions.append(Expression::Binary {
            op: BinaryOp::Add,
            left: one,
            right: two,
        });

        let changed = run_on_function(&mut func);
        assert!(changed);
        match &func.expressions[add] {
            Expression::Literal(Literal::F32(v)) => assert_eq!(*v, 3.0),
            other => panic!("expected Literal(F32(3.0)), got {other:?}"),
        }
    }

    #[test]
    fn fold_i32_multiply() {
        let mut func = Function::new("test");
        let three = func
            .expressions
            .append(Expression::Literal(Literal::I32(3)));
        let four = func
            .expressions
            .append(Expression::Literal(Literal::I32(4)));
        let mul = func.expressions.append(Expression::Binary {
            op: BinaryOp::Multiply,
            left: three,
            right: four,
        });

        let changed = run_on_function(&mut func);
        assert!(changed);
        match &func.expressions[mul] {
            Expression::Literal(Literal::I32(v)) => assert_eq!(*v, 12),
            other => panic!("expected Literal(I32(12)), got {other:?}"),
        }
    }

    #[test]
    fn fold_unary_negate() {
        let mut func = Function::new("test");
        let five = func
            .expressions
            .append(Expression::Literal(Literal::F32(5.0)));
        let neg = func.expressions.append(Expression::Unary {
            op: UnaryOp::Negate,
            expr: five,
        });

        let changed = run_on_function(&mut func);
        assert!(changed);
        match &func.expressions[neg] {
            Expression::Literal(Literal::F32(v)) => assert_eq!(*v, -5.0),
            other => panic!("expected Literal(F32(-5.0)), got {other:?}"),
        }
    }

    #[test]
    fn no_fold_non_literal_operands() {
        let mut func = Function::new("test");
        let arg = func.expressions.append(Expression::FunctionArgument(0));
        let lit = func
            .expressions
            .append(Expression::Literal(Literal::F32(2.0)));
        let _add = func.expressions.append(Expression::Binary {
            op: BinaryOp::Add,
            left: arg,
            right: lit,
        });

        let changed = run_on_function(&mut func);
        assert!(!changed);
    }

    #[test]
    fn fold_comparison() {
        let mut func = Function::new("test");
        let a = func
            .expressions
            .append(Expression::Literal(Literal::U32(5)));
        let b = func
            .expressions
            .append(Expression::Literal(Literal::U32(3)));
        let cmp = func.expressions.append(Expression::Binary {
            op: BinaryOp::Greater,
            left: a,
            right: b,
        });

        let changed = run_on_function(&mut func);
        assert!(changed);
        match &func.expressions[cmp] {
            Expression::Literal(Literal::Bool(v)) => assert!(*v),
            other => panic!("expected Literal(Bool(true)), got {other:?}"),
        }
    }

    #[test]
    fn cascade_folding() {
        // Tests that a*b -> literal, then (a*b)+c -> literal in the same pass.
        let mut func = Function::new("test");
        let a = func
            .expressions
            .append(Expression::Literal(Literal::F32(2.0)));
        let b = func
            .expressions
            .append(Expression::Literal(Literal::F32(3.0)));
        let mul = func.expressions.append(Expression::Binary {
            op: BinaryOp::Multiply,
            left: a,
            right: b,
        });
        let c = func
            .expressions
            .append(Expression::Literal(Literal::F32(4.0)));
        let add = func.expressions.append(Expression::Binary {
            op: BinaryOp::Add,
            left: mul,
            right: c,
        });

        let changed = run_on_function(&mut func);
        assert!(changed);
        // mul should be folded to 6.0, then add should be folded to 10.0.
        match &func.expressions[add] {
            Expression::Literal(Literal::F32(v)) => assert_eq!(*v, 10.0),
            other => panic!("expected Literal(F32(10.0)), got {other:?}"),
        }
    }
}
