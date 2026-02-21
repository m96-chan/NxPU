//! Dead code elimination pass.
//!
//! Removes `Emit` statements whose expression ranges contain no
//! expressions referenced (directly or transitively) by any other statement.

use std::collections::HashSet;

use nxpu_ir::{Expression, Function, Handle, Module, Statement};

use crate::Pass;

/// Removes unused `Emit` statements from function bodies.
#[derive(Debug)]
pub struct DeadCodeElimination;

impl Pass for DeadCodeElimination {
    fn name(&self) -> &str {
        "dce"
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
    // 1. Collect root expression handles from non-Emit statements.
    let mut used: HashSet<Handle<Expression>> = HashSet::new();
    collect_used_from_block(&func.body, &mut used);

    // Also mark local variable init expressions as used.
    for (_, local) in func.local_variables.iter() {
        if let Some(init) = local.init {
            used.insert(init);
        }
    }

    // 2. Transitively mark operands of used expressions.
    let mut worklist: Vec<Handle<Expression>> = used.iter().copied().collect();
    while let Some(handle) = worklist.pop() {
        if let Some(expr) = func.expressions.try_get(handle) {
            for operand in expression_operands(expr) {
                if used.insert(operand) {
                    worklist.push(operand);
                }
            }
        }
    }

    // 3. Filter out Emit statements with no used expressions.
    filter_emits_in_block(&mut func.body, &used)
}

fn collect_used_from_block(block: &[Statement], used: &mut HashSet<Handle<Expression>>) {
    for stmt in block {
        match stmt {
            Statement::Emit(_) => {}
            Statement::Store { pointer, value } => {
                used.insert(*pointer);
                used.insert(*value);
            }
            Statement::If {
                condition,
                accept,
                reject,
            } => {
                used.insert(*condition);
                collect_used_from_block(accept, used);
                collect_used_from_block(reject, used);
            }
            Statement::Loop {
                body,
                continuing,
                break_if,
            } => {
                collect_used_from_block(body, used);
                collect_used_from_block(continuing, used);
                if let Some(brk) = break_if {
                    used.insert(*brk);
                }
            }
            Statement::Call {
                arguments, result, ..
            } => {
                for arg in arguments {
                    used.insert(*arg);
                }
                if let Some(r) = result {
                    used.insert(*r);
                }
            }
            Statement::Atomic {
                pointer,
                fun,
                value,
                result,
            } => {
                used.insert(*pointer);
                used.insert(*value);
                if let Some(r) = result {
                    used.insert(*r);
                }
                if let nxpu_ir::AtomicFunction::Exchange {
                    compare: Some(cmp), ..
                } = fun
                {
                    used.insert(*cmp);
                }
            }
            Statement::Return { value } => {
                if let Some(v) = value {
                    used.insert(*v);
                }
            }
            Statement::Barrier(_) | Statement::Break | Statement::Continue => {}
        }
    }
}

/// Returns all expression handles directly referenced by an expression.
fn expression_operands(expr: &Expression) -> Vec<Handle<Expression>> {
    match expr {
        Expression::Literal(_)
        | Expression::FunctionArgument(_)
        | Expression::GlobalVariable(_)
        | Expression::LocalVariable(_)
        | Expression::CallResult(_)
        | Expression::AtomicResult { .. } => vec![],

        Expression::Load { pointer } => vec![*pointer],
        Expression::Unary { expr, .. } => vec![*expr],
        Expression::ArrayLength(e) => vec![*e],
        Expression::Splat { value, .. } => vec![*value],
        Expression::As { expr, .. } => vec![*expr],

        Expression::Binary { left, right, .. } => vec![*left, *right],
        Expression::Access { base, index } => vec![*base, *index],
        Expression::AccessIndex { base, .. } => vec![*base],
        Expression::Select {
            condition,
            accept,
            reject,
        } => vec![*condition, *accept, *reject],
        Expression::Swizzle { vector, .. } => vec![*vector],

        Expression::Compose { components, .. } => components.clone(),
        Expression::Math {
            arg,
            arg1,
            arg2,
            arg3,
            ..
        } => {
            let mut ops = vec![*arg];
            if let Some(a) = arg1 {
                ops.push(*a);
            }
            if let Some(a) = arg2 {
                ops.push(*a);
            }
            if let Some(a) = arg3 {
                ops.push(*a);
            }
            ops
        }
    }
}

fn filter_emits_in_block(block: &mut Vec<Statement>, used: &HashSet<Handle<Expression>>) -> bool {
    let mut changed = false;
    block.retain_mut(|stmt| match stmt {
        Statement::Emit(range) => {
            let has_used = used
                .iter()
                .any(|h| range.index_range().contains(&(h.index() as u32)));
            if !has_used {
                changed = true;
                return false;
            }
            true
        }
        Statement::If { accept, reject, .. } => {
            changed |= filter_emits_in_block(accept, used);
            changed |= filter_emits_in_block(reject, used);
            true
        }
        Statement::Loop {
            body, continuing, ..
        } => {
            changed |= filter_emits_in_block(body, used);
            changed |= filter_emits_in_block(continuing, used);
            true
        }
        _ => true,
    });
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use nxpu_ir::{BinaryOp, Literal, Range, Statement};

    #[test]
    fn removes_unused_emit() {
        let mut func = Function::new("test");
        let lit_a = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        let _lit_b = func
            .expressions
            .append(Expression::Literal(Literal::F32(2.0)));
        let _lit_c = func
            .expressions
            .append(Expression::Literal(Literal::F32(3.0)));

        // Emit all three.
        func.body
            .push(Statement::Emit(Range::from_index_range(0..3)));
        // Only lit_a is used.
        func.body.push(Statement::Return { value: Some(lit_a) });
        // Emit a fourth unused expression.
        let _lit_d = func
            .expressions
            .append(Expression::Literal(Literal::F32(99.0)));
        func.body
            .push(Statement::Emit(Range::from_index_range(3..4)));

        let changed = run_on_function(&mut func);
        assert!(changed);
        // The second Emit (range 3..4) should be removed; the first stays because lit_a is used.
        assert_eq!(func.body.len(), 2);
    }

    #[test]
    fn keeps_transitively_used() {
        let mut func = Function::new("test");
        let a = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        let b = func
            .expressions
            .append(Expression::Literal(Literal::F32(2.0)));
        let add = func.expressions.append(Expression::Binary {
            op: BinaryOp::Add,
            left: a,
            right: b,
        });

        func.body
            .push(Statement::Emit(Range::from_index_range(0..3)));
        func.body.push(Statement::Return { value: Some(add) });

        let changed = run_on_function(&mut func);
        // No change — all expressions are transitively used via `add`.
        assert!(!changed);
        assert_eq!(func.body.len(), 2);
    }

    #[test]
    fn no_change_on_empty_function() {
        let mut func = Function::new("test");
        let changed = run_on_function(&mut func);
        assert!(!changed);
    }
}
