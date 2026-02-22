//! IR validation pass.
//!
//! Checks structural invariants of the IR module and collects warnings.
//! This pass never modifies the module.

use std::fmt;

use nxpu_ir::{Expression, Module};

use crate::Pass;

/// A validation warning describing a structural issue in the IR.
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    pub message: String,
}

impl fmt::Display for ValidationWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

/// Validates IR structural invariants. Returns `false` (never modifies the module).
#[derive(Debug)]
pub struct IrValidation;

impl Pass for IrValidation {
    fn name(&self) -> &str {
        "ir-validation"
    }

    fn run(&self, module: &mut Module) -> bool {
        for w in collect_warnings(module) {
            log::warn!("{}", w.message);
        }
        false
    }
}

/// Collect all validation warnings for a module without logging.
///
/// This is the primary validation API — usable in tests and debug builds
/// without needing a logger configured.
pub fn collect_warnings(module: &Module) -> Vec<ValidationWarning> {
    let mut warnings = Vec::new();
    let type_count = module.types.len();

    // Validate expression operand bounds in global expressions.
    validate_expression_arena(
        &module.global_expressions,
        "global_expressions",
        &mut warnings,
    );

    // Validate global variable type handles.
    for (handle, gv) in module.global_variables.iter() {
        if gv.ty.index() >= type_count {
            warnings.push(ValidationWarning {
                message: format!(
                    "global variable {:?} (handle {:?}) references out-of-bounds type handle {:?}",
                    gv.name, handle, gv.ty
                ),
            });
        }
    }

    // Validate entry points.
    for ep in &module.entry_points {
        // Workgroup sizes must be > 0.
        for (i, &size) in ep.workgroup_size.iter().enumerate() {
            if size == 0 {
                warnings.push(ValidationWarning {
                    message: format!("entry point '{}' has workgroup_size[{}] = 0", ep.name, i),
                });
            }
        }

        validate_expression_arena(
            &ep.function.expressions,
            &format!("ep '{}'", ep.name),
            &mut warnings,
        );

        // Validate local variable type handles.
        validate_local_and_arg_types(
            &ep.function,
            type_count,
            &format!("ep '{}'", ep.name),
            &mut warnings,
        );
    }

    // Validate helper functions.
    for (handle, func) in module.functions.iter() {
        let ctx = format!(
            "function '{}' ({:?})",
            func.name.as_deref().unwrap_or("<unnamed>"),
            handle
        );
        validate_expression_arena(&func.expressions, &ctx, &mut warnings);
        validate_local_and_arg_types(func, type_count, &ctx, &mut warnings);
    }

    warnings
}

fn validate_expression_arena(
    arena: &nxpu_ir::Arena<Expression>,
    context: &str,
    warnings: &mut Vec<ValidationWarning>,
) {
    let arena_len = arena.len();

    for (handle, expr) in arena.iter() {
        let operands = crate::dce::expression_operands(expr);
        for operand in operands {
            if operand.index() >= arena_len {
                warnings.push(ValidationWarning {
                    message: format!(
                        "{}: expression {:?} references out-of-bounds operand {:?} (arena size {})",
                        context, handle, operand, arena_len,
                    ),
                });
            }
        }
    }
}

fn validate_local_and_arg_types(
    func: &nxpu_ir::Function,
    type_count: usize,
    context: &str,
    warnings: &mut Vec<ValidationWarning>,
) {
    for (lv_handle, local) in func.local_variables.iter() {
        if local.ty.index() >= type_count {
            warnings.push(ValidationWarning {
                message: format!(
                    "{}: local variable {:?} ({:?}) references out-of-bounds type {:?}",
                    context, local.name, lv_handle, local.ty
                ),
            });
        }
    }
    for (i, arg) in func.arguments.iter().enumerate() {
        if arg.ty.index() >= type_count {
            warnings.push(ValidationWarning {
                message: format!(
                    "{}: argument {} ({:?}) references out-of-bounds type {:?}",
                    context, i, arg.name, arg.ty
                ),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nxpu_ir::{EntryPoint, Function, Literal};

    #[test]
    fn valid_module_no_warnings() {
        let mut module = Module::default();
        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [64, 1, 1],
            function: Function::new("main"),
        });

        let warnings = collect_warnings(&module);
        assert!(warnings.is_empty());
    }

    #[test]
    fn zero_workgroup_size_detected() {
        let mut module = Module::default();
        module.entry_points.push(EntryPoint {
            name: "bad_ep".into(),
            workgroup_size: [0, 1, 1],
            function: Function::new("bad_ep"),
        });

        let warnings = collect_warnings(&module);
        assert!(!warnings.is_empty());
        assert!(
            warnings[0].message.contains("workgroup_size[0] = 0"),
            "unexpected message: {}",
            warnings[0].message
        );
    }

    #[test]
    fn valid_expressions_no_warnings() {
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

        let warnings = collect_warnings(&module);
        assert!(warnings.is_empty());
    }

    #[test]
    fn pass_returns_false() {
        let mut module = Module::default();
        let pass = IrValidation;
        assert!(!pass.run(&mut module));
    }
}
