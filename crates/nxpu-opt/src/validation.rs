//! IR validation pass.
//!
//! Checks structural invariants of the IR module and logs warnings for
//! problems found. This pass never modifies the module.

use nxpu_ir::{Expression, Module};

use crate::Pass;

/// Validates IR structural invariants. Returns `false` (never modifies the module).
#[derive(Debug)]
pub struct IrValidation;

impl Pass for IrValidation {
    fn name(&self) -> &str {
        "ir-validation"
    }

    fn run(&self, module: &mut Module) -> bool {
        validate_module(module);
        false
    }
}

fn validate_module(module: &Module) {
    // Validate expression operand bounds in global expressions.
    validate_expression_arena(&module.global_expressions, "global_expressions");

    // Validate global variable type handles.
    let type_count = module.types.len();
    for (handle, gv) in module.global_variables.iter() {
        if gv.ty.index() >= type_count {
            log::warn!(
                "global variable {:?} (handle {:?}) references out-of-bounds type handle {:?}",
                gv.name,
                handle,
                gv.ty
            );
        }
    }

    // Validate entry points.
    for ep in &module.entry_points {
        // Workgroup sizes must be > 0.
        for (i, &size) in ep.workgroup_size.iter().enumerate() {
            if size == 0 {
                log::warn!("entry point '{}' has workgroup_size[{}] = 0", ep.name, i);
            }
        }

        validate_expression_arena(&ep.function.expressions, &format!("ep '{}'", ep.name));
    }

    // Validate helper functions.
    for (handle, func) in module.functions.iter() {
        validate_expression_arena(
            &func.expressions,
            &format!(
                "function '{}' ({:?})",
                func.name.as_deref().unwrap_or("<unnamed>"),
                handle
            ),
        );
    }
}

fn validate_expression_arena(arena: &nxpu_ir::Arena<Expression>, context: &str) {
    let arena_len = arena.len();

    for (handle, expr) in arena.iter() {
        let operands = crate::dce::expression_operands(expr);
        for operand in operands {
            if operand.index() >= arena_len {
                log::warn!(
                    "{}: expression {:?} references out-of-bounds operand {:?} (arena size {})",
                    context,
                    handle,
                    operand,
                    arena_len,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nxpu_ir::{EntryPoint, Function, Literal};

    #[test]
    fn valid_module_passes() {
        let mut module = Module::default();
        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [64, 1, 1],
            function: Function::new("main"),
        });

        let pass = IrValidation;
        let changed = pass.run(&mut module);
        assert!(!changed);
    }

    #[test]
    fn zero_workgroup_size_warns() {
        // This test verifies the pass runs without panicking on a zero
        // workgroup size. The warning is emitted via log::warn! which
        // is a no-op in tests unless a logger is configured.
        let mut module = Module::default();
        module.entry_points.push(EntryPoint {
            name: "bad_ep".into(),
            workgroup_size: [0, 1, 1],
            function: Function::new("bad_ep"),
        });

        let pass = IrValidation;
        let changed = pass.run(&mut module);
        assert!(!changed);
    }

    #[test]
    fn valid_expressions_ok() {
        let mut module = Module::default();
        let mut func = Function::new("main");
        let _lit = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });

        let pass = IrValidation;
        let changed = pass.run(&mut module);
        assert!(!changed);
    }
}
