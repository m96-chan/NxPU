//! Dead code elimination pass.
//!
//! Removes `Emit` statements whose expression ranges contain no
//! expressions referenced (directly or transitively) by any other statement.

use std::collections::{HashMap, HashSet};

use nxpu_ir::{Arena, Expression, Function, Handle, LocalVariable, Module, Statement};

use crate::Pass;

/// Removes unused `Emit` statements and unreferenced local variables from
/// function bodies.
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
            changed |= remove_dead_locals(func);
        }
        for ep in &mut module.entry_points {
            changed |= run_on_function(&mut ep.function);
            changed |= remove_dead_locals(&mut ep.function);
        }
        changed
    }
}

fn run_on_function(func: &mut Function) -> bool {
    // Collect locals that are loaded (read) somewhere.
    let loaded_locals = collect_loaded_locals(func);

    // 1. Collect root expression handles from non-Emit statements.
    let mut used: HashSet<Handle<Expression>> = HashSet::new();
    collect_used_from_block(&func.body, &mut used, &loaded_locals, func);

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

    // 3. Pre-compute which pointers target dead locals.
    let dead_store_ptrs: HashSet<Handle<Expression>> = func
        .expressions
        .iter()
        .filter(|(_, expr)| {
            matches!(expr, Expression::LocalVariable(lv) if !loaded_locals.contains(lv))
        })
        .map(|(h, _)| h)
        .collect();

    // 4. Filter out dead Emit/Store/Call statements.
    filter_dead_in_block(&mut func.body, &used, &dead_store_ptrs)
}

/// Collect all local variables that are loaded (read) anywhere in the function.
#[allow(clippy::collapsible_if)] // nested if-let for MSRV 1.87 compat (no let chains)
fn collect_loaded_locals(func: &Function) -> HashSet<Handle<LocalVariable>> {
    let mut loaded = HashSet::new();
    for (_, expr) in func.expressions.iter() {
        if let Expression::Load { pointer } = expr {
            if let Some(Expression::LocalVariable(lv)) = func.expressions.try_get(*pointer) {
                loaded.insert(*lv);
            }
        }
    }
    loaded
}

fn collect_used_from_block(
    block: &[Statement],
    used: &mut HashSet<Handle<Expression>>,
    loaded_locals: &HashSet<Handle<LocalVariable>>,
    func: &Function,
) {
    for stmt in block {
        match stmt {
            Statement::Emit(_) => {}
            Statement::Store { pointer, value } => {
                // Check if this store targets an unread local variable.
                let is_dead_store = if let Some(Expression::LocalVariable(lv)) =
                    func.expressions.try_get(*pointer)
                {
                    !loaded_locals.contains(lv)
                } else {
                    false
                };
                if !is_dead_store {
                    used.insert(*pointer);
                    used.insert(*value);
                }
            }
            Statement::If {
                condition,
                accept,
                reject,
            } => {
                used.insert(*condition);
                collect_used_from_block(accept, used, loaded_locals, func);
                collect_used_from_block(reject, used, loaded_locals, func);
            }
            Statement::Loop {
                body,
                continuing,
                break_if,
            } => {
                collect_used_from_block(body, used, loaded_locals, func);
                collect_used_from_block(continuing, used, loaded_locals, func);
                if let Some(brk) = break_if {
                    used.insert(*brk);
                }
            }
            Statement::Call {
                arguments, result, ..
            } => {
                // Always mark call arguments and results as used (conservative).
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
pub(crate) fn expression_operands(expr: &Expression) -> Vec<Handle<Expression>> {
    match expr {
        Expression::Literal(_)
        | Expression::FunctionArgument(_)
        | Expression::GlobalVariable(_)
        | Expression::LocalVariable(_)
        | Expression::CallResult(_)
        | Expression::AtomicResult { .. }
        | Expression::ZeroValue(_) => vec![],

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

fn filter_dead_in_block(
    block: &mut Vec<Statement>,
    used: &HashSet<Handle<Expression>>,
    dead_store_ptrs: &HashSet<Handle<Expression>>,
) -> bool {
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
        Statement::Store { pointer, .. } => {
            // Remove stores to unread local variables.
            if dead_store_ptrs.contains(pointer) {
                changed = true;
                return false;
            }
            true
        }
        Statement::If { accept, reject, .. } => {
            changed |= filter_dead_in_block(accept, used, dead_store_ptrs);
            changed |= filter_dead_in_block(reject, used, dead_store_ptrs);
            true
        }
        Statement::Loop {
            body, continuing, ..
        } => {
            changed |= filter_dead_in_block(body, used, dead_store_ptrs);
            changed |= filter_dead_in_block(continuing, used, dead_store_ptrs);
            true
        }
        _ => true,
    });
    changed
}

/// Removes local variables that are never referenced in any expression,
/// and remaps `Expression::LocalVariable` handles accordingly.
#[allow(clippy::collapsible_if)] // nested if-let for MSRV 1.87 compat (no let chains)
fn remove_dead_locals(func: &mut Function) -> bool {
    // Collect referenced local variable handles.
    let mut referenced: HashSet<Handle<LocalVariable>> = HashSet::new();
    for (_, expr) in func.expressions.iter() {
        if let Expression::LocalVariable(lv) = expr {
            referenced.insert(*lv);
        }
    }

    let old_len = func.local_variables.len();
    // Check if every local variable in the arena is referenced.
    let all_referenced = func
        .local_variables
        .iter()
        .all(|(h, _)| referenced.contains(&h));
    if all_referenced {
        return false;
    }

    // Build a new arena and a handle remap table.
    let mut new_arena: Arena<LocalVariable> = Arena::new();
    let mut remap: HashMap<Handle<LocalVariable>, Handle<LocalVariable>> = HashMap::new();

    for (old_handle, local) in func.local_variables.iter() {
        if referenced.contains(&old_handle) {
            let new_handle = new_arena.append(local.clone());
            remap.insert(old_handle, new_handle);
        }
    }

    // Remap all Expression::LocalVariable handles.
    let expr_handles: Vec<Handle<Expression>> = func.expressions.iter().map(|(h, _)| h).collect();
    for h in expr_handles {
        if let Expression::LocalVariable(lv) = &func.expressions[h] {
            if let Some(&new_lv) = remap.get(lv) {
                if new_lv != *lv {
                    func.expressions[h] = Expression::LocalVariable(new_lv);
                }
            }
        }
    }

    func.local_variables = new_arena;
    func.local_variables.len() < old_len
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

    fn dummy_type_handle() -> Handle<nxpu_ir::Type> {
        let mut types = nxpu_ir::UniqueArena::new();
        types.insert(nxpu_ir::Type {
            name: None,
            inner: nxpu_ir::TypeInner::Scalar(nxpu_ir::Scalar::F32),
        })
    }

    fn dummy_gv_handle() -> Handle<nxpu_ir::GlobalVariable> {
        let mut arena = Arena::new();
        arena.append(nxpu_ir::GlobalVariable {
            name: None,
            space: nxpu_ir::AddressSpace::Storage {
                access: nxpu_ir::StorageAccess::LOAD | nxpu_ir::StorageAccess::STORE,
            },
            binding: None,
            ty: dummy_type_handle(),
            init: None,
            layout: None,
        })
    }

    #[test]
    fn removes_dead_store_to_unread_local() {
        let mut func = Function::new("test");
        let lv = func.local_variables.append(LocalVariable {
            name: Some("temp".into()),
            ty: dummy_type_handle(),
            init: None,
        });
        let ptr = func.expressions.append(Expression::LocalVariable(lv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(42.0)));
        // Store to the local (never read).
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });
        func.body.push(Statement::Return { value: None });

        let changed = run_on_function(&mut func);
        assert!(changed);
        // The store should have been removed.
        assert_eq!(func.body.len(), 1); // only Return remains
    }

    #[test]
    fn keeps_store_to_read_local() {
        let mut func = Function::new("test");
        let lv = func.local_variables.append(LocalVariable {
            name: Some("temp".into()),
            ty: dummy_type_handle(),
            init: None,
        });
        let ptr = func.expressions.append(Expression::LocalVariable(lv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(42.0)));
        // Load the local (read).
        let loaded = func.expressions.append(Expression::Load { pointer: ptr });
        func.body.push(Statement::Emit(Range::from_index_range(
            ptr.index() as u32..val.index() as u32 + 1,
        )));
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });
        func.body.push(Statement::Return {
            value: Some(loaded),
        });

        let changed = run_on_function(&mut func);
        let store_count = func
            .body
            .iter()
            .filter(|s| matches!(s, Statement::Store { .. }))
            .count();
        assert_eq!(store_count, 1);
        let _ = changed;
    }

    #[test]
    fn keeps_store_to_global() {
        let mut func = Function::new("test");
        let gv_handle = dummy_gv_handle();
        let ptr = func
            .expressions
            .append(Expression::GlobalVariable(gv_handle));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(42.0)));
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });
        func.body.push(Statement::Return { value: None });

        let changed = run_on_function(&mut func);
        let store_count = func
            .body
            .iter()
            .filter(|s| matches!(s, Statement::Store { .. }))
            .count();
        assert_eq!(store_count, 1);
        let _ = changed;
    }

    #[test]
    fn removes_unreferenced_local() {
        let mut func = Function::new("test");
        // Add two locals; only reference one in expressions.
        let _lv_unused = func.local_variables.append(LocalVariable {
            name: Some("unused".into()),
            ty: dummy_type_handle(),
            init: None,
        });
        let lv_used = func.local_variables.append(LocalVariable {
            name: Some("used".into()),
            ty: dummy_type_handle(),
            init: None,
        });
        // Reference only lv_used.
        let _ptr = func.expressions.append(Expression::LocalVariable(lv_used));

        assert_eq!(func.local_variables.len(), 2);
        let changed = remove_dead_locals(&mut func);
        assert!(changed);
        assert_eq!(func.local_variables.len(), 1);
    }

    #[test]
    fn keeps_all_referenced_locals() {
        let mut func = Function::new("test");
        let lv_a = func.local_variables.append(LocalVariable {
            name: Some("a".into()),
            ty: dummy_type_handle(),
            init: None,
        });
        let lv_b = func.local_variables.append(LocalVariable {
            name: Some("b".into()),
            ty: dummy_type_handle(),
            init: None,
        });
        let _ptr_a = func.expressions.append(Expression::LocalVariable(lv_a));
        let _ptr_b = func.expressions.append(Expression::LocalVariable(lv_b));

        assert_eq!(func.local_variables.len(), 2);
        let changed = remove_dead_locals(&mut func);
        assert!(!changed);
        assert_eq!(func.local_variables.len(), 2);
    }
}
