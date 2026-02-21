//! Constant folding pass.
//!
//! Evaluates binary and unary operations on literal operands at compile time,
//! replacing them with the resulting literal in the expression arena.

use nxpu_ir::{BinaryOp, Expression, Function, Handle, Literal, MathFunction, Module, UnaryOp};

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
            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3: _,
            } => fold_math(
                *fun,
                &func.expressions[*arg],
                arg1.map(|h| &func.expressions[h]),
                arg2.map(|h| &func.expressions[h]),
            )
            .map(Expression::Literal),
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

fn fold_math(
    fun: MathFunction,
    arg: &Expression,
    arg1: Option<&Expression>,
    arg2: Option<&Expression>,
) -> Option<Literal> {
    // Only fold f32 scalar literals; skip vector ops.
    let a = match arg {
        Expression::Literal(Literal::F32(v)) => *v,
        _ => return None,
    };
    let b = arg1.and_then(|e| match e {
        Expression::Literal(Literal::F32(v)) => Some(*v),
        _ => None,
    });
    let c = arg2.and_then(|e| match e {
        Expression::Literal(Literal::F32(v)) => Some(*v),
        _ => None,
    });

    let result = match fun {
        // 1-arg
        MathFunction::Abs => Some(a.abs()),
        MathFunction::Floor => Some(a.floor()),
        MathFunction::Ceil => Some(a.ceil()),
        MathFunction::Round => Some(a.round()),
        MathFunction::Fract => Some(a.fract()),
        MathFunction::Trunc => Some(a.trunc()),
        MathFunction::Sqrt => Some(a.sqrt()),
        MathFunction::InverseSqrt => Some(1.0 / a.sqrt()),
        MathFunction::Log => Some(a.ln()),
        MathFunction::Log2 => Some(a.log2()),
        MathFunction::Exp => Some(a.exp()),
        MathFunction::Exp2 => Some(a.exp2()),
        MathFunction::Sin => Some(a.sin()),
        MathFunction::Cos => Some(a.cos()),
        MathFunction::Tan => Some(a.tan()),
        MathFunction::Asin => Some(a.asin()),
        MathFunction::Acos => Some(a.acos()),
        MathFunction::Atan => Some(a.atan()),
        MathFunction::Sinh => Some(a.sinh()),
        MathFunction::Cosh => Some(a.cosh()),
        MathFunction::Tanh => Some(a.tanh()),
        MathFunction::Saturate => Some(a.clamp(0.0, 1.0)),
        // 2-arg
        MathFunction::Min => Some(a.min(b?)),
        MathFunction::Max => Some(a.max(b?)),
        MathFunction::Pow => Some(a.powf(b?)),
        MathFunction::Atan2 => Some(a.atan2(b?)),
        MathFunction::Step => {
            let edge = b?;
            Some(if a < edge { 0.0 } else { 1.0 })
        }
        // 3-arg
        MathFunction::Clamp => {
            let lo = b?;
            let hi = c?;
            Some(a.clamp(lo, hi))
        }
        MathFunction::Mix => {
            let y = b?;
            let t = c?;
            Some(a * (1.0 - t) + y * t)
        }
        MathFunction::Fma => {
            let mb = b?;
            let mc = c?;
            Some(a.mul_add(mb, mc))
        }
        // Vector ops — skip
        MathFunction::Dot
        | MathFunction::Cross
        | MathFunction::Normalize
        | MathFunction::Length
        | MathFunction::Distance
        | MathFunction::SmoothStep => None,
    };

    result.map(Literal::F32)
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

    // --- Math expression folding tests ---

    fn make_math1(func: &mut Function, fun: MathFunction, val: f32) -> Handle<Expression> {
        let arg = func
            .expressions
            .append(Expression::Literal(Literal::F32(val)));
        func.expressions.append(Expression::Math {
            fun,
            arg,
            arg1: None,
            arg2: None,
            arg3: None,
        })
    }

    fn make_math2(func: &mut Function, fun: MathFunction, a: f32, b: f32) -> Handle<Expression> {
        let arg = func
            .expressions
            .append(Expression::Literal(Literal::F32(a)));
        let arg1 = func
            .expressions
            .append(Expression::Literal(Literal::F32(b)));
        func.expressions.append(Expression::Math {
            fun,
            arg,
            arg1: Some(arg1),
            arg2: None,
            arg3: None,
        })
    }

    fn make_math3(
        func: &mut Function,
        fun: MathFunction,
        a: f32,
        b: f32,
        c: f32,
    ) -> Handle<Expression> {
        let arg = func
            .expressions
            .append(Expression::Literal(Literal::F32(a)));
        let arg1 = func
            .expressions
            .append(Expression::Literal(Literal::F32(b)));
        let arg2 = func
            .expressions
            .append(Expression::Literal(Literal::F32(c)));
        func.expressions.append(Expression::Math {
            fun,
            arg,
            arg1: Some(arg1),
            arg2: Some(arg2),
            arg3: None,
        })
    }

    #[test]
    fn fold_math_abs() {
        let mut func = Function::new("test");
        let h = make_math1(&mut func, MathFunction::Abs, -3.0);
        let changed = run_on_function(&mut func);
        assert!(changed);
        match &func.expressions[h] {
            Expression::Literal(Literal::F32(v)) => assert_eq!(*v, 3.0),
            other => panic!("expected 3.0, got {other:?}"),
        }
    }

    #[test]
    fn fold_math_sqrt() {
        let mut func = Function::new("test");
        let h = make_math1(&mut func, MathFunction::Sqrt, 9.0);
        let changed = run_on_function(&mut func);
        assert!(changed);
        match &func.expressions[h] {
            Expression::Literal(Literal::F32(v)) => assert_eq!(*v, 3.0),
            other => panic!("expected 3.0, got {other:?}"),
        }
    }

    #[test]
    fn fold_math_min_max() {
        let mut func = Function::new("test");
        let min_h = make_math2(&mut func, MathFunction::Min, 2.0, 5.0);
        let max_h = make_math2(&mut func, MathFunction::Max, 2.0, 5.0);
        let changed = run_on_function(&mut func);
        assert!(changed);
        match &func.expressions[min_h] {
            Expression::Literal(Literal::F32(v)) => assert_eq!(*v, 2.0),
            other => panic!("expected 2.0, got {other:?}"),
        }
        match &func.expressions[max_h] {
            Expression::Literal(Literal::F32(v)) => assert_eq!(*v, 5.0),
            other => panic!("expected 5.0, got {other:?}"),
        }
    }

    #[test]
    fn fold_math_clamp() {
        let mut func = Function::new("test");
        let h = make_math3(&mut func, MathFunction::Clamp, 10.0, 0.0, 5.0);
        let changed = run_on_function(&mut func);
        assert!(changed);
        match &func.expressions[h] {
            Expression::Literal(Literal::F32(v)) => assert_eq!(*v, 5.0),
            other => panic!("expected 5.0, got {other:?}"),
        }
    }

    #[test]
    fn fold_math_trig() {
        let mut func = Function::new("test");
        let h = make_math1(&mut func, MathFunction::Sin, 0.0);
        let changed = run_on_function(&mut func);
        assert!(changed);
        match &func.expressions[h] {
            Expression::Literal(Literal::F32(v)) => assert!(v.abs() < 1e-6),
            other => panic!("expected ~0.0, got {other:?}"),
        }
    }

    #[test]
    fn fold_math_fma() {
        let mut func = Function::new("test");
        // fma(2.0, 3.0, 4.0) = 2*3 + 4 = 10
        let h = make_math3(&mut func, MathFunction::Fma, 2.0, 3.0, 4.0);
        let changed = run_on_function(&mut func);
        assert!(changed);
        match &func.expressions[h] {
            Expression::Literal(Literal::F32(v)) => assert_eq!(*v, 10.0),
            other => panic!("expected 10.0, got {other:?}"),
        }
    }

    #[test]
    fn no_fold_math_non_literal() {
        let mut func = Function::new("test");
        let arg = func.expressions.append(Expression::FunctionArgument(0));
        let _h = func.expressions.append(Expression::Math {
            fun: MathFunction::Abs,
            arg,
            arg1: None,
            arg2: None,
            arg3: None,
        });
        let changed = run_on_function(&mut func);
        assert!(!changed);
    }
}
