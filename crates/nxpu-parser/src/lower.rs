//! Lowering pass: `naga::Module` → `nxpu_ir::Module`.

use std::collections::HashMap;

use nxpu_ir::{Arena, Handle};

use crate::ParseError;

// ---------------------------------------------------------------------------
// Contexts
// ---------------------------------------------------------------------------

/// Module-level lowering context — maintains handle mappings between naga and
/// NxPU-IR arenas.
struct LowerCtx<'a> {
    naga: &'a naga::Module,
    module: nxpu_ir::Module,
    type_map: HashMap<naga::Handle<naga::Type>, Handle<nxpu_ir::Type>>,
    global_var_map: HashMap<naga::Handle<naga::GlobalVariable>, Handle<nxpu_ir::GlobalVariable>>,
    const_expr_map: HashMap<naga::Handle<naga::Expression>, Handle<nxpu_ir::Expression>>,
    func_map: HashMap<naga::Handle<naga::Function>, Handle<nxpu_ir::Function>>,
}

/// Per-function lowering context for expressions and locals.
struct FuncCtx {
    function: nxpu_ir::Function,
    expr_map: HashMap<naga::Handle<naga::Expression>, Handle<nxpu_ir::Expression>>,
    local_var_map: HashMap<naga::Handle<naga::LocalVariable>, Handle<nxpu_ir::LocalVariable>>,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn lower_module(naga: &naga::Module) -> Result<nxpu_ir::Module, ParseError> {
    let mut ctx = LowerCtx {
        naga,
        module: nxpu_ir::Module::default(),
        type_map: HashMap::new(),
        global_var_map: HashMap::new(),
        const_expr_map: HashMap::new(),
        func_map: HashMap::new(),
    };

    ctx.lower_types()?;
    ctx.lower_global_variables()?;
    ctx.lower_global_expressions()?;
    ctx.lower_functions()?;
    ctx.lower_entry_points()?;

    Ok(ctx.module)
}

// ---------------------------------------------------------------------------
// Type lowering
// ---------------------------------------------------------------------------

impl LowerCtx<'_> {
    fn lower_types(&mut self) -> Result<(), ParseError> {
        // Collect to break the borrow on self.naga during iteration.
        let naga_types: Vec<_> = self
            .naga
            .types
            .iter()
            .map(|(h, t)| (h, t.name.clone(), t.inner.clone()))
            .collect();

        for (naga_handle, name, inner) in naga_types {
            let ir_inner = self.lower_type_inner(&inner)?;
            let ir_handle = self.module.types.insert(nxpu_ir::Type {
                name,
                inner: ir_inner,
            });
            self.type_map.insert(naga_handle, ir_handle);
        }
        Ok(())
    }

    fn lower_type_inner(
        &mut self,
        inner: &naga::TypeInner,
    ) -> Result<nxpu_ir::TypeInner, ParseError> {
        match *inner {
            naga::TypeInner::Scalar(s) => Ok(nxpu_ir::TypeInner::Scalar(lower_scalar(s))),
            naga::TypeInner::Vector { size, scalar } => Ok(nxpu_ir::TypeInner::Vector {
                size: lower_vector_size(size),
                scalar: lower_scalar(scalar),
            }),
            naga::TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } => Ok(nxpu_ir::TypeInner::Matrix {
                columns: lower_vector_size(columns),
                rows: lower_vector_size(rows),
                scalar: lower_scalar(scalar),
            }),
            naga::TypeInner::Atomic(s) => Ok(nxpu_ir::TypeInner::Atomic(lower_scalar(s))),
            naga::TypeInner::Pointer { base, space } => Ok(nxpu_ir::TypeInner::Pointer {
                base: self.map_type(base)?,
                space: lower_address_space(space)?,
            }),
            naga::TypeInner::ValuePointer {
                size,
                scalar,
                space,
            } => {
                // ValuePointer is equivalent to Pointer whose base is a
                // Scalar or Vector type. Materialize the base type in our arena.
                let base_inner = match size {
                    Some(sz) => nxpu_ir::TypeInner::Vector {
                        size: lower_vector_size(sz),
                        scalar: lower_scalar(scalar),
                    },
                    None => nxpu_ir::TypeInner::Scalar(lower_scalar(scalar)),
                };
                let base_handle = self.module.types.insert(nxpu_ir::Type {
                    name: None,
                    inner: base_inner,
                });
                Ok(nxpu_ir::TypeInner::Pointer {
                    base: base_handle,
                    space: lower_address_space(space)?,
                })
            }
            naga::TypeInner::Array { base, size, stride } => Ok(nxpu_ir::TypeInner::Array {
                base: self.map_type(base)?,
                size: lower_array_size(size)?,
                stride,
            }),
            naga::TypeInner::Struct { ref members, span } => {
                let ir_members = members
                    .iter()
                    .map(|m| {
                        Ok(nxpu_ir::StructMember {
                            name: m.name.clone(),
                            ty: self.map_type(m.ty)?,
                            offset: m.offset,
                        })
                    })
                    .collect::<Result<Vec<_>, ParseError>>()?;
                Ok(nxpu_ir::TypeInner::Struct {
                    members: ir_members,
                    span,
                })
            }
            naga::TypeInner::Image { .. } => Err(unsupported("Image type")),
            naga::TypeInner::Sampler { .. } => Err(unsupported("Sampler type")),
            naga::TypeInner::AccelerationStructure { .. } => {
                Err(unsupported("AccelerationStructure type"))
            }
            naga::TypeInner::RayQuery { .. } => Err(unsupported("RayQuery type")),
            naga::TypeInner::BindingArray { .. } => Err(unsupported("BindingArray type")),
        }
    }

    fn map_type(&self, h: naga::Handle<naga::Type>) -> Result<Handle<nxpu_ir::Type>, ParseError> {
        self.type_map
            .get(&h)
            .copied()
            .ok_or_else(|| ParseError::Lowering(format!("unmapped type {h:?}")))
    }
}

// ---------------------------------------------------------------------------
// Global variables
// ---------------------------------------------------------------------------

impl LowerCtx<'_> {
    fn lower_global_variables(&mut self) -> Result<(), ParseError> {
        for (naga_handle, var) in self.naga.global_variables.iter() {
            // Skip non-compute address spaces.
            if matches!(var.space, naga::AddressSpace::Handle) {
                continue;
            }

            let space = lower_address_space(var.space)?;
            let ty = self.map_type(var.ty)?;
            let binding = var.binding.as_ref().map(|b| nxpu_ir::ResourceBinding {
                group: b.group,
                binding: b.binding,
            });
            let init = match var.init {
                Some(h) => Some(self.map_const_expr(h)?),
                None => None,
            };

            let ir_handle = self
                .module
                .global_variables
                .append(nxpu_ir::GlobalVariable {
                    name: var.name.clone(),
                    space,
                    binding,
                    ty,
                    init,
                    layout: None,
                });
            self.global_var_map.insert(naga_handle, ir_handle);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Global (constant) expressions
// ---------------------------------------------------------------------------

impl LowerCtx<'_> {
    fn lower_global_expressions(&mut self) -> Result<(), ParseError> {
        for (naga_handle, expr) in self.naga.global_expressions.iter() {
            let ir_expr = self.lower_const_expr(expr)?;
            let ir_handle = self.module.global_expressions.append(ir_expr);
            self.const_expr_map.insert(naga_handle, ir_handle);
        }
        Ok(())
    }

    fn lower_const_expr(&self, expr: &naga::Expression) -> Result<nxpu_ir::Expression, ParseError> {
        match *expr {
            naga::Expression::Literal(lit) => Ok(nxpu_ir::Expression::Literal(lower_literal(lit)?)),
            naga::Expression::ZeroValue(ty) => Ok(synthesize_zero(&self.naga.types[ty].inner)),
            naga::Expression::Constant(h) => {
                // Inline: copy the constant's init expression.
                let constant = &self.naga.constants[h];
                self.lower_const_expr(&self.naga.global_expressions[constant.init])
            }
            naga::Expression::Compose { ty, ref components } => {
                let ir_ty = self.map_type(ty)?;
                let ir_components = components
                    .iter()
                    .map(|c| self.map_const_expr(*c))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(nxpu_ir::Expression::Compose {
                    ty: ir_ty,
                    components: ir_components,
                })
            }
            _ => Err(ParseError::Lowering(format!(
                "unsupported global expression: {expr:?}"
            ))),
        }
    }

    fn map_const_expr(
        &self,
        h: naga::Handle<naga::Expression>,
    ) -> Result<Handle<nxpu_ir::Expression>, ParseError> {
        self.const_expr_map
            .get(&h)
            .copied()
            .ok_or_else(|| ParseError::Lowering(format!("unmapped const expression {h:?}")))
    }
}

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

impl LowerCtx<'_> {
    fn lower_functions(&mut self) -> Result<(), ParseError> {
        for (naga_handle, naga_func) in self.naga.functions.iter() {
            let ir_func = self.lower_function(naga_func)?;
            let ir_handle = self.module.functions.append(ir_func);
            self.func_map.insert(naga_handle, ir_handle);
        }
        Ok(())
    }

    fn lower_entry_points(&mut self) -> Result<(), ParseError> {
        for ep in &self.naga.entry_points {
            if ep.stage != naga::ShaderStage::Compute {
                continue;
            }
            let ir_func = self.lower_function(&ep.function)?;
            self.module.entry_points.push(nxpu_ir::EntryPoint {
                name: ep.name.clone(),
                workgroup_size: ep.workgroup_size,
                function: ir_func,
            });
        }
        Ok(())
    }

    fn lower_function(&self, naga_func: &naga::Function) -> Result<nxpu_ir::Function, ParseError> {
        let mut fctx = FuncCtx {
            function: nxpu_ir::Function {
                name: naga_func.name.clone(),
                arguments: Vec::new(),
                result: None,
                local_variables: Arena::new(),
                expressions: Arena::new(),
                named_expressions: HashMap::new(),
                body: Vec::new(),
            },
            expr_map: HashMap::new(),
            local_var_map: HashMap::new(),
        };

        // Arguments
        for arg in &naga_func.arguments {
            let ty = self.map_type(arg.ty)?;
            let binding = match &arg.binding {
                Some(b) => Some(lower_binding(b)?),
                None => None,
            };
            fctx.function.arguments.push(nxpu_ir::FunctionArgument {
                name: arg.name.clone(),
                ty,
                binding,
            });
        }

        // Result
        if let Some(ref res) = naga_func.result {
            let ty = self.map_type(res.ty)?;
            let binding = match &res.binding {
                Some(b) => Some(lower_binding(b)?),
                None => None,
            };
            fctx.function.result = Some(nxpu_ir::FunctionResult { ty, binding });
        }

        // Local variables — first pass: create with init=None to populate
        // the handle map (expressions may reference local variables).
        let mut local_inits: Vec<(
            naga::Handle<naga::LocalVariable>,
            naga::Handle<naga::Expression>,
        )> = Vec::new();
        for (naga_handle, var) in naga_func.local_variables.iter() {
            let ty = self.map_type(var.ty)?;
            if let Some(init_h) = var.init {
                local_inits.push((naga_handle, init_h));
            }
            let ir_handle = fctx
                .function
                .local_variables
                .append(nxpu_ir::LocalVariable {
                    name: var.name.clone(),
                    ty,
                    init: None,
                });
            fctx.local_var_map.insert(naga_handle, ir_handle);
        }

        // Expressions — lower all upfront since naga expressions reference
        // each other by handle.
        for (naga_handle, expr) in naga_func.expressions.iter() {
            let ir_expr = self.lower_expression(expr, &fctx, naga_func)?;
            let ir_handle = fctx.function.expressions.append(ir_expr);
            fctx.expr_map.insert(naga_handle, ir_handle);
        }

        // Local variables — second pass: fill in init expressions now that
        // both function and const expression maps are populated.
        for (naga_handle, init_h) in local_inits {
            let ir_init = self.map_func_or_const_expr(&fctx, init_h)?;
            let ir_handle = *fctx.local_var_map.get(&naga_handle).unwrap();
            fctx.function.local_variables[ir_handle].init = Some(ir_init);
        }

        // Named expressions
        for (naga_handle, name) in &naga_func.named_expressions {
            if let Some(&ir_handle) = fctx.expr_map.get(naga_handle) {
                fctx.function
                    .named_expressions
                    .insert(ir_handle, name.clone());
            }
        }

        // Body
        fctx.function.body = self.lower_block(&naga_func.body, &fctx)?;

        Ok(fctx.function)
    }
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

impl LowerCtx<'_> {
    fn lower_expression(
        &self,
        expr: &naga::Expression,
        fctx: &FuncCtx,
        _naga_func: &naga::Function,
    ) -> Result<nxpu_ir::Expression, ParseError> {
        match *expr {
            naga::Expression::Literal(lit) => Ok(nxpu_ir::Expression::Literal(lower_literal(lit)?)),
            naga::Expression::Constant(h) => {
                // Inline the constant's init expression value.
                let constant = &self.naga.constants[h];
                self.lower_const_expr_into_func(&self.naga.global_expressions[constant.init])
            }
            naga::Expression::Override(_) => Err(unsupported("Override expression")),
            naga::Expression::ZeroValue(ty) => Ok(synthesize_zero(&self.naga.types[ty].inner)),
            naga::Expression::Compose { ty, ref components } => {
                let ir_ty = self.map_type(ty)?;
                let ir_components = components
                    .iter()
                    .map(|c| self.map_func_expr(fctx, *c))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(nxpu_ir::Expression::Compose {
                    ty: ir_ty,
                    components: ir_components,
                })
            }
            naga::Expression::Access { base, index } => Ok(nxpu_ir::Expression::Access {
                base: self.map_func_expr(fctx, base)?,
                index: self.map_func_expr(fctx, index)?,
            }),
            naga::Expression::AccessIndex { base, index } => Ok(nxpu_ir::Expression::AccessIndex {
                base: self.map_func_expr(fctx, base)?,
                index,
            }),
            naga::Expression::Splat { size, value } => Ok(nxpu_ir::Expression::Splat {
                size: lower_vector_size(size),
                value: self.map_func_expr(fctx, value)?,
            }),
            naga::Expression::Swizzle {
                size,
                vector,
                pattern,
            } => Ok(nxpu_ir::Expression::Swizzle {
                size: lower_vector_size(size),
                vector: self.map_func_expr(fctx, vector)?,
                pattern: lower_swizzle_pattern(pattern),
            }),
            naga::Expression::FunctionArgument(idx) => {
                Ok(nxpu_ir::Expression::FunctionArgument(idx))
            }
            naga::Expression::GlobalVariable(h) => {
                let ir_h =
                    self.global_var_map.get(&h).copied().ok_or_else(|| {
                        ParseError::Lowering(format!("unmapped global var {h:?}"))
                    })?;
                Ok(nxpu_ir::Expression::GlobalVariable(ir_h))
            }
            naga::Expression::LocalVariable(h) => {
                let ir_h = fctx
                    .local_var_map
                    .get(&h)
                    .copied()
                    .ok_or_else(|| ParseError::Lowering(format!("unmapped local var {h:?}")))?;
                Ok(nxpu_ir::Expression::LocalVariable(ir_h))
            }
            naga::Expression::Load { pointer } => Ok(nxpu_ir::Expression::Load {
                pointer: self.map_func_expr(fctx, pointer)?,
            }),
            naga::Expression::Unary { op, expr } => Ok(nxpu_ir::Expression::Unary {
                op: lower_unary_op(op),
                expr: self.map_func_expr(fctx, expr)?,
            }),
            naga::Expression::Binary { op, left, right } => Ok(nxpu_ir::Expression::Binary {
                op: lower_binary_op(op),
                left: self.map_func_expr(fctx, left)?,
                right: self.map_func_expr(fctx, right)?,
            }),
            naga::Expression::Select {
                condition,
                accept,
                reject,
            } => Ok(nxpu_ir::Expression::Select {
                condition: self.map_func_expr(fctx, condition)?,
                accept: self.map_func_expr(fctx, accept)?,
                reject: self.map_func_expr(fctx, reject)?,
            }),
            naga::Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => {
                let ir_fun = lower_math_function(fun)?;
                Ok(nxpu_ir::Expression::Math {
                    fun: ir_fun,
                    arg: self.map_func_expr(fctx, arg)?,
                    arg1: self.map_func_expr_opt(fctx, arg1)?,
                    arg2: self.map_func_expr_opt(fctx, arg2)?,
                    arg3: self.map_func_expr_opt(fctx, arg3)?,
                })
            }
            naga::Expression::As {
                expr,
                kind,
                convert,
            } => Ok(nxpu_ir::Expression::As {
                expr: self.map_func_expr(fctx, expr)?,
                kind: lower_scalar_kind(kind)?,
                convert,
            }),
            naga::Expression::CallResult(h) => {
                let ir_h = self
                    .func_map
                    .get(&h)
                    .copied()
                    .ok_or_else(|| ParseError::Lowering(format!("unmapped function {h:?}")))?;
                Ok(nxpu_ir::Expression::CallResult(ir_h))
            }
            naga::Expression::AtomicResult { ty, comparison } => {
                Ok(nxpu_ir::Expression::AtomicResult {
                    ty: self.map_type(ty)?,
                    comparison,
                })
            }
            naga::Expression::ArrayLength(expr) => Ok(nxpu_ir::Expression::ArrayLength(
                self.map_func_expr(fctx, expr)?,
            )),
            // Unsupported expression kinds
            naga::Expression::Derivative { .. } => Err(unsupported("Derivative expression")),
            naga::Expression::Relational { .. } => Err(unsupported("Relational expression")),
            naga::Expression::ImageSample { .. } => Err(unsupported("ImageSample expression")),
            naga::Expression::ImageLoad { .. } => Err(unsupported("ImageLoad expression")),
            naga::Expression::ImageQuery { .. } => Err(unsupported("ImageQuery expression")),
            naga::Expression::RayQueryProceedResult => {
                Err(unsupported("RayQueryProceedResult expression"))
            }
            naga::Expression::RayQueryGetIntersection { .. } => {
                Err(unsupported("RayQueryGetIntersection expression"))
            }
            naga::Expression::RayQueryVertexPositions { .. } => {
                Err(unsupported("RayQueryVertexPositions expression"))
            }
            naga::Expression::SubgroupBallotResult => {
                Err(unsupported("SubgroupBallotResult expression"))
            }
            naga::Expression::SubgroupOperationResult { .. } => {
                Err(unsupported("SubgroupOperationResult expression"))
            }
            naga::Expression::WorkGroupUniformLoadResult { .. } => {
                Err(unsupported("WorkGroupUniformLoadResult expression"))
            }
        }
    }

    /// Lower a global/const expression into a form suitable for a function
    /// arena (recursively inlines constants).
    fn lower_const_expr_into_func(
        &self,
        expr: &naga::Expression,
    ) -> Result<nxpu_ir::Expression, ParseError> {
        self.lower_const_expr(expr)
    }

    fn map_func_expr(
        &self,
        fctx: &FuncCtx,
        h: naga::Handle<naga::Expression>,
    ) -> Result<Handle<nxpu_ir::Expression>, ParseError> {
        fctx.expr_map
            .get(&h)
            .copied()
            .ok_or_else(|| ParseError::Lowering(format!("unmapped expression {h:?}")))
    }

    fn map_func_expr_opt(
        &self,
        fctx: &FuncCtx,
        h: Option<naga::Handle<naga::Expression>>,
    ) -> Result<Option<Handle<nxpu_ir::Expression>>, ParseError> {
        match h {
            Some(handle) => Ok(Some(self.map_func_expr(fctx, handle)?)),
            None => Ok(None),
        }
    }

    /// Resolve an expression handle that might live in either the function
    /// arena or the global const-expression arena (for local variable inits).
    fn map_func_or_const_expr(
        &self,
        fctx: &FuncCtx,
        h: naga::Handle<naga::Expression>,
    ) -> Result<Handle<nxpu_ir::Expression>, ParseError> {
        if let Some(&ir) = fctx.expr_map.get(&h) {
            return Ok(ir);
        }
        if let Some(&ir) = self.const_expr_map.get(&h) {
            return Ok(ir);
        }
        Err(ParseError::Lowering(format!(
            "unmapped expression (func or const) {h:?}"
        )))
    }
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

impl LowerCtx<'_> {
    fn lower_block(
        &self,
        block: &naga::Block,
        fctx: &FuncCtx,
    ) -> Result<nxpu_ir::Block, ParseError> {
        let mut out = Vec::new();
        for stmt in block.iter() {
            self.lower_statement(stmt, fctx, &mut out)?;
        }
        Ok(out)
    }

    fn lower_statement(
        &self,
        stmt: &naga::Statement,
        fctx: &FuncCtx,
        out: &mut nxpu_ir::Block,
    ) -> Result<(), ParseError> {
        match *stmt {
            naga::Statement::Emit(ref range) => {
                if let Some((first_naga, last_naga)) = range.clone().first_and_last() {
                    let first = self.map_func_expr(fctx, first_naga)?;
                    let last = self.map_func_expr(fctx, last_naga)?;
                    // nxpu-ir Range is half-open [first, last); naga's last is inclusive.
                    out.push(nxpu_ir::Statement::Emit(nxpu_ir::Range::from_index_range(
                        first.index() as u32..last.index() as u32 + 1,
                    )));
                }
            }
            naga::Statement::Block(ref block) => {
                // Flatten nested blocks into parent.
                let stmts = self.lower_block(block, fctx)?;
                out.extend(stmts);
            }
            naga::Statement::Store { pointer, value } => {
                out.push(nxpu_ir::Statement::Store {
                    pointer: self.map_func_expr(fctx, pointer)?,
                    value: self.map_func_expr(fctx, value)?,
                });
            }
            naga::Statement::If {
                condition,
                ref accept,
                ref reject,
            } => {
                out.push(nxpu_ir::Statement::If {
                    condition: self.map_func_expr(fctx, condition)?,
                    accept: self.lower_block(accept, fctx)?,
                    reject: self.lower_block(reject, fctx)?,
                });
            }
            naga::Statement::Loop {
                ref body,
                ref continuing,
                break_if,
            } => {
                out.push(nxpu_ir::Statement::Loop {
                    body: self.lower_block(body, fctx)?,
                    continuing: self.lower_block(continuing, fctx)?,
                    break_if: self.map_func_expr_opt(fctx, break_if)?,
                });
            }
            naga::Statement::Break => out.push(nxpu_ir::Statement::Break),
            naga::Statement::Continue => out.push(nxpu_ir::Statement::Continue),
            naga::Statement::Return { value } => {
                out.push(nxpu_ir::Statement::Return {
                    value: self.map_func_expr_opt(fctx, value)?,
                });
            }
            naga::Statement::Call {
                function,
                ref arguments,
                result,
            } => {
                let ir_func = self.func_map.get(&function).copied().ok_or_else(|| {
                    ParseError::Lowering(format!("unmapped called function {function:?}"))
                })?;
                let ir_args = arguments
                    .iter()
                    .map(|a| self.map_func_expr(fctx, *a))
                    .collect::<Result<Vec<_>, _>>()?;
                out.push(nxpu_ir::Statement::Call {
                    function: ir_func,
                    arguments: ir_args,
                    result: self.map_func_expr_opt(fctx, result)?,
                });
            }
            naga::Statement::Atomic {
                pointer,
                ref fun,
                value,
                result,
            } => {
                out.push(nxpu_ir::Statement::Atomic {
                    pointer: self.map_func_expr(fctx, pointer)?,
                    fun: lower_atomic_function(fun, fctx)?,
                    value: self.map_func_expr(fctx, value)?,
                    result: self.map_func_expr_opt(fctx, result)?,
                });
            }
            naga::Statement::ControlBarrier(barrier) | naga::Statement::MemoryBarrier(barrier) => {
                let ir_barrier = lower_barrier(barrier);
                if !ir_barrier.is_empty() {
                    out.push(nxpu_ir::Statement::Barrier(ir_barrier));
                }
            }
            // Unsupported statements
            naga::Statement::Switch { .. } => return Err(unsupported("Switch statement")),
            naga::Statement::Kill => return Err(unsupported("Kill statement")),
            naga::Statement::ImageStore { .. } => return Err(unsupported("ImageStore statement")),
            naga::Statement::ImageAtomic { .. } => {
                return Err(unsupported("ImageAtomic statement"));
            }
            naga::Statement::RayQuery { .. } => return Err(unsupported("RayQuery statement")),
            naga::Statement::SubgroupBallot { .. } => {
                return Err(unsupported("SubgroupBallot statement"));
            }
            naga::Statement::SubgroupGather { .. } => {
                return Err(unsupported("SubgroupGather statement"));
            }
            naga::Statement::SubgroupCollectiveOperation { .. } => {
                return Err(unsupported("SubgroupCollectiveOperation statement"));
            }
            naga::Statement::WorkGroupUniformLoad { .. } => {
                return Err(unsupported("WorkGroupUniformLoad statement"));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Enum mapping helpers
// ---------------------------------------------------------------------------

fn lower_scalar(s: naga::Scalar) -> nxpu_ir::Scalar {
    nxpu_ir::Scalar {
        kind: lower_scalar_kind_infallible(s.kind),
        width: s.width,
    }
}

fn lower_scalar_kind(kind: naga::ScalarKind) -> Result<nxpu_ir::ScalarKind, ParseError> {
    match kind {
        naga::ScalarKind::Bool => Ok(nxpu_ir::ScalarKind::Bool),
        naga::ScalarKind::Sint | naga::ScalarKind::AbstractInt => Ok(nxpu_ir::ScalarKind::Sint),
        naga::ScalarKind::Uint => Ok(nxpu_ir::ScalarKind::Uint),
        naga::ScalarKind::Float | naga::ScalarKind::AbstractFloat => Ok(nxpu_ir::ScalarKind::Float),
    }
}

fn lower_scalar_kind_infallible(kind: naga::ScalarKind) -> nxpu_ir::ScalarKind {
    match kind {
        naga::ScalarKind::Bool => nxpu_ir::ScalarKind::Bool,
        naga::ScalarKind::Sint | naga::ScalarKind::AbstractInt => nxpu_ir::ScalarKind::Sint,
        naga::ScalarKind::Uint => nxpu_ir::ScalarKind::Uint,
        naga::ScalarKind::Float | naga::ScalarKind::AbstractFloat => nxpu_ir::ScalarKind::Float,
    }
}

fn lower_vector_size(size: naga::VectorSize) -> nxpu_ir::VectorSize {
    match size {
        naga::VectorSize::Bi => nxpu_ir::VectorSize::Bi,
        naga::VectorSize::Tri => nxpu_ir::VectorSize::Tri,
        naga::VectorSize::Quad => nxpu_ir::VectorSize::Quad,
    }
}

fn lower_array_size(size: naga::ArraySize) -> Result<nxpu_ir::ArraySize, ParseError> {
    match size {
        naga::ArraySize::Constant(n) => Ok(nxpu_ir::ArraySize::Constant(n.into())),
        naga::ArraySize::Dynamic => Ok(nxpu_ir::ArraySize::Dynamic),
        naga::ArraySize::Pending(_) => Err(unsupported("override-sized array")),
    }
}

fn lower_address_space(space: naga::AddressSpace) -> Result<nxpu_ir::AddressSpace, ParseError> {
    match space {
        naga::AddressSpace::Function => Ok(nxpu_ir::AddressSpace::Function),
        naga::AddressSpace::Private => Ok(nxpu_ir::AddressSpace::Private),
        naga::AddressSpace::WorkGroup => Ok(nxpu_ir::AddressSpace::Workgroup),
        naga::AddressSpace::Uniform => Ok(nxpu_ir::AddressSpace::Uniform),
        naga::AddressSpace::Storage { access } => {
            let mut ir_access = nxpu_ir::StorageAccess::EMPTY;
            if access.contains(naga::StorageAccess::LOAD) {
                ir_access |= nxpu_ir::StorageAccess::LOAD;
            }
            if access.contains(naga::StorageAccess::STORE) {
                ir_access |= nxpu_ir::StorageAccess::STORE;
            }
            Ok(nxpu_ir::AddressSpace::Storage { access: ir_access })
        }
        naga::AddressSpace::Handle => Err(unsupported("Handle address space")),
        _ => Err(unsupported(&format!("{space:?} address space"))),
    }
}

fn lower_builtin(builtin: naga::BuiltIn) -> Result<nxpu_ir::BuiltIn, ParseError> {
    match builtin {
        naga::BuiltIn::GlobalInvocationId => Ok(nxpu_ir::BuiltIn::GlobalInvocationId),
        naga::BuiltIn::LocalInvocationId => Ok(nxpu_ir::BuiltIn::LocalInvocationId),
        naga::BuiltIn::LocalInvocationIndex => Ok(nxpu_ir::BuiltIn::LocalInvocationIndex),
        naga::BuiltIn::WorkGroupId => Ok(nxpu_ir::BuiltIn::WorkgroupId),
        naga::BuiltIn::NumWorkGroups => Ok(nxpu_ir::BuiltIn::NumWorkgroups),
        other => Err(unsupported(&format!("{other:?} builtin"))),
    }
}

fn lower_binding(binding: &naga::Binding) -> Result<nxpu_ir::Binding, ParseError> {
    match *binding {
        naga::Binding::BuiltIn(b) => Ok(nxpu_ir::Binding::BuiltIn(lower_builtin(b)?)),
        naga::Binding::Location { location, .. } => Ok(nxpu_ir::Binding::Location { location }),
    }
}

fn lower_unary_op(op: naga::UnaryOperator) -> nxpu_ir::UnaryOp {
    match op {
        naga::UnaryOperator::Negate => nxpu_ir::UnaryOp::Negate,
        naga::UnaryOperator::LogicalNot => nxpu_ir::UnaryOp::LogicalNot,
        naga::UnaryOperator::BitwiseNot => nxpu_ir::UnaryOp::BitwiseNot,
    }
}

fn lower_binary_op(op: naga::BinaryOperator) -> nxpu_ir::BinaryOp {
    match op {
        naga::BinaryOperator::Add => nxpu_ir::BinaryOp::Add,
        naga::BinaryOperator::Subtract => nxpu_ir::BinaryOp::Subtract,
        naga::BinaryOperator::Multiply => nxpu_ir::BinaryOp::Multiply,
        naga::BinaryOperator::Divide => nxpu_ir::BinaryOp::Divide,
        naga::BinaryOperator::Modulo => nxpu_ir::BinaryOp::Modulo,
        naga::BinaryOperator::Equal => nxpu_ir::BinaryOp::Equal,
        naga::BinaryOperator::NotEqual => nxpu_ir::BinaryOp::NotEqual,
        naga::BinaryOperator::Less => nxpu_ir::BinaryOp::Less,
        naga::BinaryOperator::LessEqual => nxpu_ir::BinaryOp::LessEqual,
        naga::BinaryOperator::Greater => nxpu_ir::BinaryOp::Greater,
        naga::BinaryOperator::GreaterEqual => nxpu_ir::BinaryOp::GreaterEqual,
        naga::BinaryOperator::And => nxpu_ir::BinaryOp::BitwiseAnd,
        naga::BinaryOperator::ExclusiveOr => nxpu_ir::BinaryOp::BitwiseXor,
        naga::BinaryOperator::InclusiveOr => nxpu_ir::BinaryOp::BitwiseOr,
        naga::BinaryOperator::LogicalAnd => nxpu_ir::BinaryOp::LogicalAnd,
        naga::BinaryOperator::LogicalOr => nxpu_ir::BinaryOp::LogicalOr,
        naga::BinaryOperator::ShiftLeft => nxpu_ir::BinaryOp::ShiftLeft,
        naga::BinaryOperator::ShiftRight => nxpu_ir::BinaryOp::ShiftRight,
    }
}

fn lower_math_function(fun: naga::MathFunction) -> Result<nxpu_ir::MathFunction, ParseError> {
    match fun {
        naga::MathFunction::Abs => Ok(nxpu_ir::MathFunction::Abs),
        naga::MathFunction::Min => Ok(nxpu_ir::MathFunction::Min),
        naga::MathFunction::Max => Ok(nxpu_ir::MathFunction::Max),
        naga::MathFunction::Clamp => Ok(nxpu_ir::MathFunction::Clamp),
        naga::MathFunction::Saturate => Ok(nxpu_ir::MathFunction::Saturate),
        naga::MathFunction::Floor => Ok(nxpu_ir::MathFunction::Floor),
        naga::MathFunction::Ceil => Ok(nxpu_ir::MathFunction::Ceil),
        naga::MathFunction::Round => Ok(nxpu_ir::MathFunction::Round),
        naga::MathFunction::Fract => Ok(nxpu_ir::MathFunction::Fract),
        naga::MathFunction::Trunc => Ok(nxpu_ir::MathFunction::Trunc),
        naga::MathFunction::Sin => Ok(nxpu_ir::MathFunction::Sin),
        naga::MathFunction::Cos => Ok(nxpu_ir::MathFunction::Cos),
        naga::MathFunction::Tan => Ok(nxpu_ir::MathFunction::Tan),
        naga::MathFunction::Asin => Ok(nxpu_ir::MathFunction::Asin),
        naga::MathFunction::Acos => Ok(nxpu_ir::MathFunction::Acos),
        naga::MathFunction::Atan => Ok(nxpu_ir::MathFunction::Atan),
        naga::MathFunction::Atan2 => Ok(nxpu_ir::MathFunction::Atan2),
        naga::MathFunction::Sinh => Ok(nxpu_ir::MathFunction::Sinh),
        naga::MathFunction::Cosh => Ok(nxpu_ir::MathFunction::Cosh),
        naga::MathFunction::Tanh => Ok(nxpu_ir::MathFunction::Tanh),
        naga::MathFunction::Sqrt => Ok(nxpu_ir::MathFunction::Sqrt),
        naga::MathFunction::InverseSqrt => Ok(nxpu_ir::MathFunction::InverseSqrt),
        naga::MathFunction::Log => Ok(nxpu_ir::MathFunction::Log),
        naga::MathFunction::Log2 => Ok(nxpu_ir::MathFunction::Log2),
        naga::MathFunction::Exp => Ok(nxpu_ir::MathFunction::Exp),
        naga::MathFunction::Exp2 => Ok(nxpu_ir::MathFunction::Exp2),
        naga::MathFunction::Pow => Ok(nxpu_ir::MathFunction::Pow),
        naga::MathFunction::Dot => Ok(nxpu_ir::MathFunction::Dot),
        naga::MathFunction::Cross => Ok(nxpu_ir::MathFunction::Cross),
        naga::MathFunction::Normalize => Ok(nxpu_ir::MathFunction::Normalize),
        naga::MathFunction::Length => Ok(nxpu_ir::MathFunction::Length),
        naga::MathFunction::Distance => Ok(nxpu_ir::MathFunction::Distance),
        naga::MathFunction::Mix => Ok(nxpu_ir::MathFunction::Mix),
        naga::MathFunction::Step => Ok(nxpu_ir::MathFunction::Step),
        naga::MathFunction::SmoothStep => Ok(nxpu_ir::MathFunction::SmoothStep),
        naga::MathFunction::Fma => Ok(nxpu_ir::MathFunction::Fma),
        other => Err(unsupported(&format!("{other:?} math function"))),
    }
}

fn lower_atomic_function(
    fun: &naga::AtomicFunction,
    fctx: &FuncCtx,
) -> Result<nxpu_ir::AtomicFunction, ParseError> {
    match *fun {
        naga::AtomicFunction::Add => Ok(nxpu_ir::AtomicFunction::Add),
        naga::AtomicFunction::Subtract => Ok(nxpu_ir::AtomicFunction::Subtract),
        naga::AtomicFunction::And => Ok(nxpu_ir::AtomicFunction::And),
        naga::AtomicFunction::ExclusiveOr => Ok(nxpu_ir::AtomicFunction::ExclusiveOr),
        naga::AtomicFunction::InclusiveOr => Ok(nxpu_ir::AtomicFunction::InclusiveOr),
        naga::AtomicFunction::Min => Ok(nxpu_ir::AtomicFunction::Min),
        naga::AtomicFunction::Max => Ok(nxpu_ir::AtomicFunction::Max),
        naga::AtomicFunction::Exchange { compare } => {
            let ir_compare = match compare {
                Some(h) => Some(
                    fctx.expr_map
                        .get(&h)
                        .copied()
                        .ok_or_else(|| ParseError::Lowering(format!("unmapped expr {h:?}")))?,
                ),
                None => None,
            };
            Ok(nxpu_ir::AtomicFunction::Exchange {
                compare: ir_compare,
            })
        }
    }
}

fn lower_literal(lit: naga::Literal) -> Result<nxpu_ir::Literal, ParseError> {
    match lit {
        naga::Literal::Bool(v) => Ok(nxpu_ir::Literal::Bool(v)),
        naga::Literal::I32(v) => Ok(nxpu_ir::Literal::I32(v)),
        naga::Literal::U32(v) => Ok(nxpu_ir::Literal::U32(v)),
        naga::Literal::F32(v) => Ok(nxpu_ir::Literal::F32(v)),
        naga::Literal::F64(v) => Ok(nxpu_ir::Literal::F64(v)),
        naga::Literal::AbstractInt(v) => Ok(nxpu_ir::Literal::AbstractInt(v)),
        naga::Literal::AbstractFloat(v) => Ok(nxpu_ir::Literal::AbstractFloat(v)),
        _ => Err(unsupported(&format!("{lit:?} literal"))),
    }
}

fn lower_swizzle_pattern(pattern: [naga::SwizzleComponent; 4]) -> [nxpu_ir::SwizzleComponent; 4] {
    pattern.map(|c| match c {
        naga::SwizzleComponent::X => nxpu_ir::SwizzleComponent::X,
        naga::SwizzleComponent::Y => nxpu_ir::SwizzleComponent::Y,
        naga::SwizzleComponent::Z => nxpu_ir::SwizzleComponent::Z,
        naga::SwizzleComponent::W => nxpu_ir::SwizzleComponent::W,
    })
}

fn lower_barrier(barrier: naga::Barrier) -> nxpu_ir::Barrier {
    let has_storage = barrier.contains(naga::Barrier::STORAGE);
    let has_workgroup = barrier.contains(naga::Barrier::WORK_GROUP);
    match (has_storage, has_workgroup) {
        (true, true) => nxpu_ir::Barrier::STORAGE | nxpu_ir::Barrier::WORKGROUP,
        (true, false) => nxpu_ir::Barrier::STORAGE,
        (false, true) => nxpu_ir::Barrier::WORKGROUP,
        // Caller checks is_empty() and skips.
        (false, false) => nxpu_ir::Barrier::STORAGE,
    }
}

/// Synthesize a zero-value expression for the given naga type.
fn synthesize_zero(inner: &naga::TypeInner) -> nxpu_ir::Expression {
    match *inner {
        naga::TypeInner::Scalar(s) => nxpu_ir::Expression::Literal(scalar_zero(s)),
        naga::TypeInner::Vector { scalar, .. } => nxpu_ir::Expression::Literal(scalar_zero(scalar)),
        naga::TypeInner::Matrix { scalar, .. } => nxpu_ir::Expression::Literal(scalar_zero(scalar)),
        naga::TypeInner::Atomic(s) => nxpu_ir::Expression::Literal(scalar_zero(s)),
        _ => nxpu_ir::Expression::Literal(nxpu_ir::Literal::U32(0)),
    }
}

fn scalar_zero(s: naga::Scalar) -> nxpu_ir::Literal {
    match s.kind {
        naga::ScalarKind::Bool => nxpu_ir::Literal::Bool(false),
        naga::ScalarKind::Sint | naga::ScalarKind::AbstractInt => nxpu_ir::Literal::I32(0),
        naga::ScalarKind::Uint => nxpu_ir::Literal::U32(0),
        naga::ScalarKind::Float | naga::ScalarKind::AbstractFloat => nxpu_ir::Literal::F32(0.0),
    }
}

fn unsupported(what: &str) -> ParseError {
    ParseError::Unsupported(what.to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lower_scalar_kind() {
        assert_eq!(
            lower_scalar_kind(naga::ScalarKind::Bool).unwrap(),
            nxpu_ir::ScalarKind::Bool
        );
        assert_eq!(
            lower_scalar_kind(naga::ScalarKind::Sint).unwrap(),
            nxpu_ir::ScalarKind::Sint
        );
        assert_eq!(
            lower_scalar_kind(naga::ScalarKind::Uint).unwrap(),
            nxpu_ir::ScalarKind::Uint
        );
        assert_eq!(
            lower_scalar_kind(naga::ScalarKind::Float).unwrap(),
            nxpu_ir::ScalarKind::Float
        );
    }

    #[test]
    fn test_lower_vector_size() {
        assert_eq!(
            lower_vector_size(naga::VectorSize::Bi),
            nxpu_ir::VectorSize::Bi
        );
        assert_eq!(
            lower_vector_size(naga::VectorSize::Tri),
            nxpu_ir::VectorSize::Tri
        );
        assert_eq!(
            lower_vector_size(naga::VectorSize::Quad),
            nxpu_ir::VectorSize::Quad
        );
    }

    #[test]
    fn test_lower_binary_op() {
        assert_eq!(
            lower_binary_op(naga::BinaryOperator::Add),
            nxpu_ir::BinaryOp::Add
        );
        assert_eq!(
            lower_binary_op(naga::BinaryOperator::And),
            nxpu_ir::BinaryOp::BitwiseAnd
        );
        assert_eq!(
            lower_binary_op(naga::BinaryOperator::LogicalAnd),
            nxpu_ir::BinaryOp::LogicalAnd
        );
    }

    #[test]
    fn test_lower_unary_op() {
        assert_eq!(
            lower_unary_op(naga::UnaryOperator::Negate),
            nxpu_ir::UnaryOp::Negate
        );
        assert_eq!(
            lower_unary_op(naga::UnaryOperator::BitwiseNot),
            nxpu_ir::UnaryOp::BitwiseNot
        );
    }

    #[test]
    fn test_lower_address_space() {
        assert_eq!(
            lower_address_space(naga::AddressSpace::Function).unwrap(),
            nxpu_ir::AddressSpace::Function
        );
        assert_eq!(
            lower_address_space(naga::AddressSpace::WorkGroup).unwrap(),
            nxpu_ir::AddressSpace::Workgroup
        );
        assert!(lower_address_space(naga::AddressSpace::Handle).is_err());
    }

    #[test]
    fn test_lower_math_function() {
        assert_eq!(
            lower_math_function(naga::MathFunction::Dot).unwrap(),
            nxpu_ir::MathFunction::Dot
        );
        assert_eq!(
            lower_math_function(naga::MathFunction::Fma).unwrap(),
            nxpu_ir::MathFunction::Fma
        );
        // Unsupported math functions should error.
        assert!(lower_math_function(naga::MathFunction::Determinant).is_err());
    }

    #[test]
    fn test_lower_literal() {
        match lower_literal(naga::Literal::F32(2.75)).unwrap() {
            nxpu_ir::Literal::F32(v) => assert_eq!(v, 2.75),
            _ => panic!("expected F32"),
        }
        match lower_literal(naga::Literal::Bool(true)).unwrap() {
            nxpu_ir::Literal::Bool(v) => assert!(v),
            _ => panic!("expected Bool"),
        }
    }

    #[test]
    fn test_lower_simple_module() {
        let source = "@group(0) @binding(0) var<storage, read_write> buf: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    buf[i] = buf[i] + 1.0;
}";
        let naga_module = naga::front::wgsl::parse_str(source).expect("WGSL parse failed");
        let module = lower_module(&naga_module).expect("lowering failed");

        // Should have one entry point.
        assert_eq!(module.entry_points.len(), 1);
        assert_eq!(module.entry_points[0].name, "main");
        assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);

        // Should have at least one global variable (buf).
        assert!(!module.global_variables.is_empty());
    }
}
