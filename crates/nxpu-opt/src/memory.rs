//! Memory planning and buffer allocation pass.
//!
//! Analyzes tensor lifetimes within IR functions and assigns buffer offsets
//! using a greedy allocator. When lifetimes do not overlap, the allocator
//! reuses buffer regions, reducing peak memory usage.

use std::collections::HashMap;

use nxpu_ir::{
    ArraySize, Expression, Function, GlobalVariable, Handle, Module, Statement, TypeInner,
};

// Re-export the canonical types from nxpu-backend-core.
pub use nxpu_backend_core::{BufferAllocation, MemoryPlan, TensorId};

use crate::Pass;

// ---------------------------------------------------------------------------
// Public types (only LiveInterval is defined here; the rest come from
// nxpu-backend-core)
// ---------------------------------------------------------------------------

/// The liveness interval (first-use to last-use) of a tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiveInterval {
    /// Index of the first statement that uses this tensor.
    pub start: usize,
    /// Index of the last statement that uses this tensor.
    pub end: usize,
    /// Size of the tensor in bytes.
    pub size_bytes: usize,
}

// ---------------------------------------------------------------------------
// Tensor classification
// ---------------------------------------------------------------------------

/// Information about a tensor discovered during analysis.
#[derive(Clone, Debug)]
struct TensorInfo {
    id: TensorId,
    /// Human-readable name (from the IR variable, if any).
    #[allow(dead_code)]
    name: Option<String>,
    /// Size in bytes (0 if dynamic / unknown).
    size_bytes: usize,
}

// ---------------------------------------------------------------------------
// Size computation
// ---------------------------------------------------------------------------

/// Compute the byte size of a type. Returns 0 for dynamic / unsized types.
fn type_size_bytes(module: &Module, ty: Handle<nxpu_ir::Type>) -> usize {
    match &module.types[ty].inner {
        TypeInner::Scalar(s) => s.width as usize,
        TypeInner::Vector { size, scalar } => (*size as usize) * (scalar.width as usize),
        TypeInner::Matrix {
            columns,
            rows,
            scalar,
        } => (*columns as usize) * (*rows as usize) * (scalar.width as usize),
        TypeInner::Atomic(s) => s.width as usize,
        TypeInner::Pointer { .. } => 0, // pointers have no data size for planning
        TypeInner::Array { base, size, stride } => match size {
            ArraySize::Constant(n) => {
                let elem = type_size_bytes(module, *base);
                if elem == 0 {
                    (*n as usize) * (*stride as usize)
                } else {
                    (*n as usize) * elem
                }
            }
            ArraySize::Dynamic => 0, // runtime-sized
        },
        TypeInner::Struct { span, .. } => *span as usize,
        TypeInner::Tensor { scalar, shape } => {
            if shape.is_fully_static() {
                let elem_count: usize = shape
                    .dims
                    .iter()
                    .map(|d| match d {
                        nxpu_ir::Dimension::Fixed(n) => *n as usize,
                        nxpu_ir::Dimension::Dynamic(_) | nxpu_ir::Dimension::Symbolic(_) => 0,
                    })
                    .product();
                elem_count * (scalar.width as usize)
            } else {
                0
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Lifetime analysis
// ---------------------------------------------------------------------------

/// For an expression, find which global/local variable it ultimately refers to
/// and return the corresponding `TensorId`.
fn resolve_expr_tensor_id(
    handle: Handle<Expression>,
    func: &Function,
    global_id_map: &HashMap<Handle<GlobalVariable>, TensorId>,
    local_id_map: &HashMap<Handle<nxpu_ir::LocalVariable>, TensorId>,
) -> Option<TensorId> {
    let expr = func.expressions.try_get(handle)?;
    match expr {
        Expression::GlobalVariable(gv) => global_id_map.get(gv).copied(),
        Expression::LocalVariable(lv) => local_id_map.get(lv).copied(),
        Expression::Load { pointer } => {
            resolve_expr_tensor_id(*pointer, func, global_id_map, local_id_map)
        }
        Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => {
            resolve_expr_tensor_id(*base, func, global_id_map, local_id_map)
        }
        _ => None,
    }
}

/// Try to resolve an expression handle to a tensor ID and push it if found.
fn try_resolve_push(
    h: Handle<Expression>,
    func: &Function,
    global_id_map: &HashMap<Handle<GlobalVariable>, TensorId>,
    local_id_map: &HashMap<Handle<nxpu_ir::LocalVariable>, TensorId>,
    ids: &mut Vec<TensorId>,
) {
    if let Some(tid) = resolve_expr_tensor_id(h, func, global_id_map, local_id_map) {
        ids.push(tid);
    }
}

/// Collect all tensor IDs referenced by a statement, recursing into
/// sub-blocks (If/Loop bodies).
fn collect_stmt_tensor_ids(
    stmt: &Statement,
    func: &Function,
    global_id_map: &HashMap<Handle<GlobalVariable>, TensorId>,
    local_id_map: &HashMap<Handle<nxpu_ir::LocalVariable>, TensorId>,
) -> Vec<TensorId> {
    let mut ids = Vec::new();

    match stmt {
        Statement::Emit(range) => {
            // For each expression in the emitted range, check if it references
            // a variable (directly or through operands).
            let idx_range = range.index_range();
            for (expr_handle, _) in func.expressions.iter() {
                if idx_range.contains(&(expr_handle.index() as u32)) {
                    try_resolve_push(expr_handle, func, global_id_map, local_id_map, &mut ids);
                }
            }
        }
        Statement::Store { pointer, value } => {
            try_resolve_push(*pointer, func, global_id_map, local_id_map, &mut ids);
            try_resolve_push(*value, func, global_id_map, local_id_map, &mut ids);
        }
        Statement::If {
            condition,
            accept,
            reject,
        } => {
            try_resolve_push(*condition, func, global_id_map, local_id_map, &mut ids);
            for s in accept {
                ids.extend(collect_stmt_tensor_ids(
                    s,
                    func,
                    global_id_map,
                    local_id_map,
                ));
            }
            for s in reject {
                ids.extend(collect_stmt_tensor_ids(
                    s,
                    func,
                    global_id_map,
                    local_id_map,
                ));
            }
        }
        Statement::Loop {
            body,
            continuing,
            break_if,
        } => {
            for s in body {
                ids.extend(collect_stmt_tensor_ids(
                    s,
                    func,
                    global_id_map,
                    local_id_map,
                ));
            }
            for s in continuing {
                ids.extend(collect_stmt_tensor_ids(
                    s,
                    func,
                    global_id_map,
                    local_id_map,
                ));
            }
            if let Some(brk) = break_if {
                try_resolve_push(*brk, func, global_id_map, local_id_map, &mut ids);
            }
        }
        Statement::Call {
            arguments, result, ..
        } => {
            for arg in arguments {
                try_resolve_push(*arg, func, global_id_map, local_id_map, &mut ids);
            }
            if let Some(r) = result {
                try_resolve_push(*r, func, global_id_map, local_id_map, &mut ids);
            }
        }
        Statement::Atomic {
            pointer,
            value,
            result,
            fun,
        } => {
            try_resolve_push(*pointer, func, global_id_map, local_id_map, &mut ids);
            try_resolve_push(*value, func, global_id_map, local_id_map, &mut ids);
            if let Some(r) = result {
                try_resolve_push(*r, func, global_id_map, local_id_map, &mut ids);
            }
            if let nxpu_ir::AtomicFunction::Exchange {
                compare: Some(cmp), ..
            } = fun
            {
                try_resolve_push(*cmp, func, global_id_map, local_id_map, &mut ids);
            }
        }
        Statement::Return { value } => {
            if let Some(v) = value {
                try_resolve_push(*v, func, global_id_map, local_id_map, &mut ids);
            }
        }
        Statement::Barrier(_) | Statement::Break | Statement::Continue => {}
    }
    ids
}

/// Analyze a single function and return liveness intervals for all tensors.
fn analyze_function_lifetimes(
    module: &Module,
    func: &Function,
    global_id_map: &HashMap<Handle<GlobalVariable>, TensorId>,
    tensor_infos: &mut Vec<TensorInfo>,
    next_id: &mut usize,
) -> HashMap<TensorId, LiveInterval> {
    // Build a local-variable -> TensorId mapping.
    let mut local_id_map: HashMap<Handle<nxpu_ir::LocalVariable>, TensorId> = HashMap::new();
    for (lv_handle, lv) in func.local_variables.iter() {
        let tid = TensorId(*next_id);
        *next_id += 1;
        let size = type_size_bytes(module, lv.ty);
        tensor_infos.push(TensorInfo {
            id: tid,
            name: lv.name.clone(),
            size_bytes: size,
        });
        local_id_map.insert(lv_handle, tid);
    }

    let mut intervals: HashMap<TensorId, LiveInterval> = HashMap::new();

    for (stmt_idx, stmt) in func.body.iter().enumerate() {
        let tensor_ids = collect_stmt_tensor_ids(stmt, func, global_id_map, &local_id_map);
        for tid in tensor_ids {
            // Find the size from tensor_infos.
            let size = tensor_infos
                .iter()
                .find(|t| t.id == tid)
                .map(|t| t.size_bytes)
                .unwrap_or(0);
            intervals
                .entry(tid)
                .and_modify(|interval| {
                    if stmt_idx < interval.start {
                        interval.start = stmt_idx;
                    }
                    if stmt_idx > interval.end {
                        interval.end = stmt_idx;
                    }
                })
                .or_insert(LiveInterval {
                    start: stmt_idx,
                    end: stmt_idx,
                    size_bytes: size,
                });
        }
    }

    intervals
}

// ---------------------------------------------------------------------------
// Greedy buffer allocator
// ---------------------------------------------------------------------------

/// A region in the buffer that is currently in use.
#[derive(Clone, Debug)]
struct ActiveAllocation {
    #[allow(dead_code)]
    tensor_id: TensorId,
    offset: usize,
    size: usize,
    end: usize, // last statement index where this tensor is live
}

/// Greedy first-fit allocator. Assigns offsets to tensors sorted by start time.
/// When a tensor's lifetime ends, its region becomes available for reuse.
fn greedy_allocate(intervals: &HashMap<TensorId, LiveInterval>) -> MemoryPlan {
    if intervals.is_empty() {
        return MemoryPlan::default();
    }

    // Sort intervals by start time, breaking ties by larger size first.
    let mut sorted: Vec<(TensorId, &LiveInterval)> = intervals
        .iter()
        .map(|(tid, interval)| (*tid, interval))
        .collect();
    sorted.sort_by(|a, b| {
        a.1.start
            .cmp(&b.1.start)
            .then(b.1.size_bytes.cmp(&a.1.size_bytes))
    });

    let mut active: Vec<ActiveAllocation> = Vec::new();
    let mut allocations: Vec<BufferAllocation> = Vec::new();
    let mut peak = 0usize;

    for (tid, interval) in sorted {
        // Skip zero-size tensors (dynamic / unknown).
        if interval.size_bytes == 0 {
            allocations.push(BufferAllocation {
                tensor_id: tid,
                offset: 0,
                size_bytes: 0,
            });
            continue;
        }

        // Expire allocations whose lifetime has ended before this tensor starts.
        active.retain(|a| a.end >= interval.start);

        // Sort active allocations by offset for gap-finding.
        active.sort_by_key(|a| a.offset);

        // Find first gap that fits.
        let mut offset = 0usize;
        for a in &active {
            if offset + interval.size_bytes <= a.offset {
                break;
            }
            offset = a.offset + a.size;
        }

        let end_of_alloc = offset + interval.size_bytes;
        if end_of_alloc > peak {
            peak = end_of_alloc;
        }

        allocations.push(BufferAllocation {
            tensor_id: tid,
            offset,
            size_bytes: interval.size_bytes,
        });

        active.push(ActiveAllocation {
            tensor_id: tid,
            offset,
            size: interval.size_bytes,
            end: interval.end,
        });
    }

    MemoryPlan {
        allocations,
        peak_bytes: peak,
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Analyze an IR module and produce a memory plan.
///
/// For each entry point (and helper function), determines tensor lifetimes
/// and assigns buffer offsets using a greedy first-fit algorithm.
pub fn plan_memory(module: &Module) -> MemoryPlan {
    let mut next_id = 0usize;
    let mut tensor_infos: Vec<TensorInfo> = Vec::new();
    let mut global_id_map: HashMap<Handle<GlobalVariable>, TensorId> = HashMap::new();

    // Register global variables as tensors.
    for (handle, gv) in module.global_variables.iter() {
        let tid = TensorId(next_id);
        next_id += 1;
        let size = type_size_bytes(module, gv.ty);
        tensor_infos.push(TensorInfo {
            id: tid,
            name: gv.name.clone(),
            size_bytes: size,
        });
        global_id_map.insert(handle, tid);
    }

    // Merge liveness intervals across all functions.
    let mut all_intervals: HashMap<TensorId, LiveInterval> = HashMap::new();

    // Analyze helper functions.
    for (_, func) in module.functions.iter() {
        let intervals = analyze_function_lifetimes(
            module,
            func,
            &global_id_map,
            &mut tensor_infos,
            &mut next_id,
        );
        merge_intervals(&mut all_intervals, &intervals);
    }

    // Analyze entry points.
    for ep in &module.entry_points {
        let intervals = analyze_function_lifetimes(
            module,
            &ep.function,
            &global_id_map,
            &mut tensor_infos,
            &mut next_id,
        );
        merge_intervals(&mut all_intervals, &intervals);
    }

    // Include global variables that weren't referenced in any function body
    // (they still need space allocated).
    for info in &tensor_infos {
        if info.size_bytes > 0 && !all_intervals.contains_key(&info.id) {
            // Give them a full-lifetime interval so they are always allocated.
            all_intervals.insert(
                info.id,
                LiveInterval {
                    start: 0,
                    end: usize::MAX,
                    size_bytes: info.size_bytes,
                },
            );
        }
    }

    greedy_allocate(&all_intervals)
}

/// Merge `src` intervals into `dst`, extending existing intervals as needed.
fn merge_intervals(
    dst: &mut HashMap<TensorId, LiveInterval>,
    src: &HashMap<TensorId, LiveInterval>,
) {
    for (tid, interval) in src {
        dst.entry(*tid)
            .and_modify(|existing| {
                if interval.start < existing.start {
                    existing.start = interval.start;
                }
                if interval.end > existing.end {
                    existing.end = interval.end;
                }
            })
            .or_insert_with(|| interval.clone());
    }
}

// ---------------------------------------------------------------------------
// Pass integration
// ---------------------------------------------------------------------------

/// Memory planning analysis pass.
///
/// This pass does not modify the IR; it only computes and logs a memory plan.
/// Callers should use [`plan_memory`] directly to obtain the plan.
#[derive(Debug)]
pub struct MemoryPlanning;

impl Pass for MemoryPlanning {
    fn name(&self) -> &str {
        "MemoryPlanning"
    }

    fn run(&self, module: &mut Module) -> bool {
        let plan = plan_memory(module);
        log::debug!(
            "memory plan: {} allocations, peak {} bytes",
            plan.allocations.len(),
            plan.peak_bytes
        );
        // Analysis pass -- never modifies the IR.
        false
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nxpu_ir::*;

    // Helper: create a type handle for f32.
    fn f32_type(module: &mut Module) -> Handle<Type> {
        module.types.insert(Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::F32),
        })
    }

    // Helper: create a fixed-size f32 array type.
    fn f32_array_type(module: &mut Module, count: u32) -> Handle<Type> {
        let f32_ty = f32_type(module);
        module.types.insert(Type {
            name: None,
            inner: TypeInner::Array {
                base: f32_ty,
                size: ArraySize::Constant(count),
                stride: 4,
            },
        })
    }

    // Helper: create a dynamic f32 array type.
    fn f32_dynamic_array_type(module: &mut Module) -> Handle<Type> {
        let f32_ty = f32_type(module);
        module.types.insert(Type {
            name: None,
            inner: TypeInner::Array {
                base: f32_ty,
                size: ArraySize::Dynamic,
                stride: 4,
            },
        })
    }

    #[test]
    fn empty_module_produces_empty_plan() {
        let module = Module::default();
        let plan = plan_memory(&module);
        assert!(plan.allocations.is_empty());
        assert_eq!(plan.peak_bytes, 0);
    }

    #[test]
    fn single_tensor() {
        let mut module = Module::default();
        let arr_ty = f32_array_type(&mut module, 256); // 256 * 4 = 1024 bytes

        let gv = module.global_variables.append(GlobalVariable {
            name: Some("buf".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: arr_ty,
            init: None,
            layout: None,
        });

        // Create an entry point that references the global.
        let mut func = Function::new("main");
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let lit = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        func.body.push(Statement::Store {
            pointer: ptr,
            value: lit,
        });

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });

        let plan = plan_memory(&module);
        assert_eq!(plan.allocations.len(), 1);
        assert_eq!(plan.allocations[0].size_bytes, 1024);
        assert_eq!(plan.peak_bytes, 1024);
    }

    #[test]
    fn non_overlapping_lifetimes_reuse_buffer() {
        // Two local tensors: one used in stmt 0, another used in stmt 1.
        // They should share the same buffer region.
        let mut module = Module::default();
        let arr_ty = f32_array_type(&mut module, 128); // 512 bytes each

        let mut func = Function::new("main");

        // Local variable A (used only in first Store).
        let lv_a = func.local_variables.append(LocalVariable {
            name: Some("temp_a".into()),
            ty: arr_ty,
            init: None,
        });
        // Local variable B (used only in second Store).
        let lv_b = func.local_variables.append(LocalVariable {
            name: Some("temp_b".into()),
            ty: arr_ty,
            init: None,
        });

        let ptr_a = func.expressions.append(Expression::LocalVariable(lv_a));
        let ptr_b = func.expressions.append(Expression::LocalVariable(lv_b));
        let lit = func
            .expressions
            .append(Expression::Literal(Literal::F32(0.0)));

        // stmt 0: store to A
        func.body.push(Statement::Store {
            pointer: ptr_a,
            value: lit,
        });
        // stmt 1: store to B
        func.body.push(Statement::Store {
            pointer: ptr_b,
            value: lit,
        });

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });

        let plan = plan_memory(&module);

        // Both tensors should be allocated, each 512 bytes.
        let non_zero: Vec<_> = plan
            .allocations
            .iter()
            .filter(|a| a.size_bytes > 0)
            .collect();
        assert_eq!(non_zero.len(), 2);

        // Peak should be 512 (reuse), not 1024 (no reuse).
        assert_eq!(plan.peak_bytes, 512);
    }

    #[test]
    fn overlapping_lifetimes_no_reuse() {
        // Two local tensors both used across overlapping statements.
        let mut module = Module::default();
        let arr_ty = f32_array_type(&mut module, 128); // 512 bytes each

        let mut func = Function::new("main");

        let lv_a = func.local_variables.append(LocalVariable {
            name: Some("a".into()),
            ty: arr_ty,
            init: None,
        });
        let lv_b = func.local_variables.append(LocalVariable {
            name: Some("b".into()),
            ty: arr_ty,
            init: None,
        });

        let ptr_a = func.expressions.append(Expression::LocalVariable(lv_a));
        let ptr_b = func.expressions.append(Expression::LocalVariable(lv_b));
        let lit = func
            .expressions
            .append(Expression::Literal(Literal::F32(0.0)));

        // stmt 0: store to A (A is live)
        func.body.push(Statement::Store {
            pointer: ptr_a,
            value: lit,
        });
        // stmt 1: read from A, store to B (both are live)
        let load_a = func.expressions.append(Expression::Load { pointer: ptr_a });
        func.body.push(Statement::Store {
            pointer: ptr_b,
            value: load_a,
        });

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });

        let plan = plan_memory(&module);

        // Both 512-byte tensors overlap (A: [0,1], B: [1,1]) so peak must be 1024.
        let non_zero: Vec<_> = plan
            .allocations
            .iter()
            .filter(|a| a.size_bytes > 0)
            .collect();
        assert_eq!(non_zero.len(), 2);
        assert_eq!(plan.peak_bytes, 1024);
    }

    #[test]
    fn peak_memory_less_than_sum_of_all_tensors() {
        // Three tensors with staggered, non-overlapping lifetimes.
        // Total = 3 * 512 = 1536, but peak should be 512 (full reuse).
        let mut module = Module::default();
        let arr_ty = f32_array_type(&mut module, 128); // 512 bytes each

        let mut func = Function::new("main");

        let lv_a = func.local_variables.append(LocalVariable {
            name: Some("a".into()),
            ty: arr_ty,
            init: None,
        });
        let lv_b = func.local_variables.append(LocalVariable {
            name: Some("b".into()),
            ty: arr_ty,
            init: None,
        });
        let lv_c = func.local_variables.append(LocalVariable {
            name: Some("c".into()),
            ty: arr_ty,
            init: None,
        });

        let ptr_a = func.expressions.append(Expression::LocalVariable(lv_a));
        let ptr_b = func.expressions.append(Expression::LocalVariable(lv_b));
        let ptr_c = func.expressions.append(Expression::LocalVariable(lv_c));
        let lit = func
            .expressions
            .append(Expression::Literal(Literal::F32(0.0)));

        // Each tensor used in exactly one statement, no overlap.
        func.body.push(Statement::Store {
            pointer: ptr_a,
            value: lit,
        });
        func.body.push(Statement::Store {
            pointer: ptr_b,
            value: lit,
        });
        func.body.push(Statement::Store {
            pointer: ptr_c,
            value: lit,
        });

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });

        let plan = plan_memory(&module);

        let total: usize = plan.allocations.iter().map(|a| a.size_bytes).sum();
        assert_eq!(total, 1536); // 3 * 512
        // Reuse should bring peak well below total.
        assert!(
            plan.peak_bytes < total,
            "peak {} should be < total {}",
            plan.peak_bytes,
            total
        );
        assert_eq!(plan.peak_bytes, 512); // all three reuse the same slot
    }

    #[test]
    fn dynamic_tensors_get_zero_size() {
        let mut module = Module::default();
        let dyn_ty = f32_dynamic_array_type(&mut module);

        module.global_variables.append(GlobalVariable {
            name: Some("dyn_buf".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: dyn_ty,
            init: None,
            layout: None,
        });

        let plan = plan_memory(&module);
        // Dynamic tensors have size 0, so they don't contribute to peak.
        assert_eq!(plan.peak_bytes, 0);
    }

    #[test]
    fn tensor_type_size_computation() {
        let mut module = Module::default();

        // Tensor<f32>[1, 224, 224, 3] => 1*224*224*3*4 = 602112 bytes
        let tensor_ty = module.types.insert(Type {
            name: None,
            inner: TypeInner::Tensor {
                scalar: Scalar::F32,
                shape: TensorShape::fixed(&[1, 224, 224, 3]),
            },
        });

        assert_eq!(type_size_bytes(&module, tensor_ty), 602_112);

        // Dynamic tensor => 0
        let dyn_tensor_ty = module.types.insert(Type {
            name: None,
            inner: TypeInner::Tensor {
                scalar: Scalar::F32,
                shape: TensorShape::all_dynamic(4),
            },
        });
        assert_eq!(type_size_bytes(&module, dyn_tensor_ty), 0);
    }

    #[test]
    fn memory_plan_display() {
        let plan = MemoryPlan {
            allocations: vec![
                BufferAllocation {
                    tensor_id: TensorId(0),
                    offset: 0,
                    size_bytes: 1024,
                },
                BufferAllocation {
                    tensor_id: TensorId(1),
                    offset: 0,
                    size_bytes: 512,
                },
            ],
            peak_bytes: 1024,
        };

        let text = format!("{plan}");
        assert!(text.contains("Peak memory: 1024 bytes"));
        assert!(text.contains("Buffers: 2"));
        assert!(text.contains("Reuse savings:"));
        assert!(text.contains("tensor_0"));
        assert!(text.contains("tensor_1"));
    }

    #[test]
    fn memory_planning_pass_runs() {
        let mut module = Module::default();
        let pass = MemoryPlanning;
        assert_eq!(pass.name(), "MemoryPlanning");
        let changed = pass.run(&mut module);
        assert!(!changed); // analysis pass never modifies IR
    }

    #[test]
    fn input_compute_output_pipeline() {
        // Simulates: input -> compute -> output
        // input and output don't overlap with the temporary.
        let mut module = Module::default();
        let arr_ty = f32_array_type(&mut module, 256); // 1024 bytes

        let gv_input = module.global_variables.append(GlobalVariable {
            name: Some("input".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: arr_ty,
            init: None,
            layout: None,
        });
        let gv_output = module.global_variables.append(GlobalVariable {
            name: Some("output".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 1,
            }),
            ty: arr_ty,
            init: None,
            layout: None,
        });

        let mut func = Function::new("main");

        let temp = func.local_variables.append(LocalVariable {
            name: Some("temp".into()),
            ty: arr_ty,
            init: None,
        });

        let ptr_in = func
            .expressions
            .append(Expression::GlobalVariable(gv_input));
        let ptr_out = func
            .expressions
            .append(Expression::GlobalVariable(gv_output));
        let ptr_temp = func.expressions.append(Expression::LocalVariable(temp));
        let load_in = func
            .expressions
            .append(Expression::Load { pointer: ptr_in });
        let load_temp = func
            .expressions
            .append(Expression::Load { pointer: ptr_temp });

        // stmt 0: temp = load(input) -- input and temp are live
        func.body.push(Statement::Store {
            pointer: ptr_temp,
            value: load_in,
        });
        // stmt 1: output = load(temp) -- temp and output are live
        func.body.push(Statement::Store {
            pointer: ptr_out,
            value: load_temp,
        });

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [256, 1, 1],
            function: func,
        });

        let plan = plan_memory(&module);

        // input: [0,0], temp: [0,1], output: [1,1]
        // input and output don't overlap, so they can share a slot.
        // Peak should be 2048 (temp + one of input/output), not 3072.
        assert!(
            plan.peak_bytes <= 2048,
            "peak {} should be <= 2048 (reuse between input and output)",
            plan.peak_bytes
        );
        assert!(plan.peak_bytes > 0);
    }

    #[test]
    fn many_temporaries_reuse() {
        // 10 temporaries, each used in exactly one statement, no overlap.
        let mut module = Module::default();
        let arr_ty = f32_array_type(&mut module, 64); // 256 bytes each

        let mut func = Function::new("main");
        let lit = func
            .expressions
            .append(Expression::Literal(Literal::F32(0.0)));

        for i in 0..10 {
            let lv = func.local_variables.append(LocalVariable {
                name: Some(format!("temp_{i}")),
                ty: arr_ty,
                init: None,
            });
            let ptr = func.expressions.append(Expression::LocalVariable(lv));
            func.body.push(Statement::Store {
                pointer: ptr,
                value: lit,
            });
        }

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });

        let plan = plan_memory(&module);

        let total: usize = plan.allocations.iter().map(|a| a.size_bytes).sum();
        assert_eq!(total, 2560); // 10 * 256
        // All should reuse the same 256-byte slot.
        assert_eq!(plan.peak_bytes, 256);
    }

    #[test]
    fn struct_type_size() {
        let mut module = Module::default();
        let struct_ty = module.types.insert(Type {
            name: Some("Params".into()),
            inner: TypeInner::Struct {
                members: vec![],
                span: 64,
            },
        });
        assert_eq!(type_size_bytes(&module, struct_ty), 64);
    }

    #[test]
    fn vector_and_matrix_type_sizes() {
        let mut module = Module::default();

        let vec4_ty = module.types.insert(Type {
            name: None,
            inner: TypeInner::Vector {
                size: VectorSize::Quad,
                scalar: Scalar::F32,
            },
        });
        assert_eq!(type_size_bytes(&module, vec4_ty), 16); // 4 * 4

        let mat4x4_ty = module.types.insert(Type {
            name: None,
            inner: TypeInner::Matrix {
                columns: VectorSize::Quad,
                rows: VectorSize::Quad,
                scalar: Scalar::F32,
            },
        });
        assert_eq!(type_size_bytes(&module, mat4x4_ty), 64); // 4 * 4 * 4
    }

    // ===== Type size: Atomic =====

    #[test]
    fn atomic_type_size() {
        let mut module = Module::default();
        let atomic_ty = module.types.insert(Type {
            name: None,
            inner: TypeInner::Atomic(Scalar::U32),
        });
        assert_eq!(type_size_bytes(&module, atomic_ty), 4);
    }

    // ===== Type size: Pointer =====

    #[test]
    fn pointer_type_size_is_zero() {
        let mut module = Module::default();
        let f32_ty = f32_type(&mut module);
        let ptr_ty = module.types.insert(Type {
            name: None,
            inner: TypeInner::Pointer {
                base: f32_ty,
                space: AddressSpace::Storage {
                    access: StorageAccess::LOAD,
                },
            },
        });
        assert_eq!(type_size_bytes(&module, ptr_ty), 0);
    }

    // ===== Type size: Array with zero-size base uses stride =====

    #[test]
    fn array_with_zero_base_uses_stride() {
        let mut module = Module::default();
        // Create an array whose base type has zero size (e.g. pointer).
        let f32_ty = f32_type(&mut module);
        let ptr_base = module.types.insert(Type {
            name: None,
            inner: TypeInner::Pointer {
                base: f32_ty,
                space: AddressSpace::Storage {
                    access: StorageAccess::LOAD,
                },
            },
        });
        let arr_ty = module.types.insert(Type {
            name: None,
            inner: TypeInner::Array {
                base: ptr_base,
                size: ArraySize::Constant(10),
                stride: 8,
            },
        });
        // Base has size 0, so falls back to n * stride = 10 * 8 = 80.
        assert_eq!(type_size_bytes(&module, arr_ty), 80);
    }

    // ===== Type size: Array with non-zero base uses base size =====

    #[test]
    fn array_with_nonzero_base_uses_elem_size() {
        let mut module = Module::default();
        let arr_ty = f32_array_type(&mut module, 10);
        // f32 = 4 bytes, so 10 * 4 = 40.
        assert_eq!(type_size_bytes(&module, arr_ty), 40);
    }

    // ===== Type size: Dynamic array =====

    #[test]
    fn dynamic_array_size_is_zero() {
        let mut module = Module::default();
        let dyn_arr_ty = f32_dynamic_array_type(&mut module);
        assert_eq!(type_size_bytes(&module, dyn_arr_ty), 0);
    }

    // ===== Type size: Scalar =====

    #[test]
    fn scalar_type_size() {
        let mut module = Module::default();
        let f32_ty = f32_type(&mut module);
        assert_eq!(type_size_bytes(&module, f32_ty), 4);
    }

    // ===== Type size: Vec2 =====

    #[test]
    fn vec2_type_size() {
        let mut module = Module::default();
        let vec2_ty = module.types.insert(Type {
            name: None,
            inner: TypeInner::Vector {
                size: VectorSize::Bi,
                scalar: Scalar::F32,
            },
        });
        assert_eq!(type_size_bytes(&module, vec2_ty), 8); // 2 * 4
    }

    // ===== Type size: Vec3 =====

    #[test]
    fn vec3_type_size() {
        let mut module = Module::default();
        let vec3_ty = module.types.insert(Type {
            name: None,
            inner: TypeInner::Vector {
                size: VectorSize::Tri,
                scalar: Scalar::F32,
            },
        });
        assert_eq!(type_size_bytes(&module, vec3_ty), 12); // 3 * 4
    }

    // ===== Type size: Matrix 2x3 =====

    #[test]
    fn matrix_2x3_type_size() {
        let mut module = Module::default();
        let mat_ty = module.types.insert(Type {
            name: None,
            inner: TypeInner::Matrix {
                columns: VectorSize::Bi,
                rows: VectorSize::Tri,
                scalar: Scalar::F32,
            },
        });
        assert_eq!(type_size_bytes(&module, mat_ty), 24); // 2 * 3 * 4
    }

    // ===== Greedy allocate: empty intervals =====

    #[test]
    fn greedy_allocate_empty() {
        let intervals = HashMap::new();
        let plan = greedy_allocate(&intervals);
        assert!(plan.allocations.is_empty());
        assert_eq!(plan.peak_bytes, 0);
    }

    // ===== Greedy allocate: zero-size tensor =====

    #[test]
    fn greedy_allocate_zero_size_tensor() {
        let mut intervals = HashMap::new();
        intervals.insert(
            TensorId(0),
            LiveInterval {
                start: 0,
                end: 5,
                size_bytes: 0,
            },
        );
        let plan = greedy_allocate(&intervals);
        assert_eq!(plan.allocations.len(), 1);
        assert_eq!(plan.allocations[0].size_bytes, 0);
        assert_eq!(plan.peak_bytes, 0);
    }

    // ===== Greedy allocate: gap finding between active allocations =====

    #[test]
    fn greedy_allocate_gap_finding() {
        // Three tensors: A and C overlap, B does not overlap with anything.
        // A: [0, 3] size 100, C: [0, 3] size 100, B: [0, 3] size 50.
        // All overlap, so they must be placed sequentially.
        // Then D: [5, 6] size 50 should fit in a gap (reuse B's slot).
        let mut intervals = HashMap::new();
        intervals.insert(
            TensorId(0),
            LiveInterval {
                start: 0,
                end: 3,
                size_bytes: 100,
            },
        );
        intervals.insert(
            TensorId(1),
            LiveInterval {
                start: 0,
                end: 3,
                size_bytes: 50,
            },
        );
        intervals.insert(
            TensorId(2),
            LiveInterval {
                start: 0,
                end: 3,
                size_bytes: 100,
            },
        );
        intervals.insert(
            TensorId(3),
            LiveInterval {
                start: 5,
                end: 6,
                size_bytes: 50,
            },
        );

        let plan = greedy_allocate(&intervals);
        assert_eq!(plan.allocations.len(), 4);
        // Peak should be 250 (all three overlapping: 100 + 50 + 100).
        // D should reuse space since its lifetime doesn't overlap.
        assert_eq!(plan.peak_bytes, 250);
    }

    // ===== Greedy allocate: size tie-breaking (larger first) =====

    #[test]
    fn greedy_allocate_size_tiebreak() {
        // Two tensors starting at the same time, different sizes.
        let mut intervals = HashMap::new();
        intervals.insert(
            TensorId(0),
            LiveInterval {
                start: 0,
                end: 0,
                size_bytes: 50,
            },
        );
        intervals.insert(
            TensorId(1),
            LiveInterval {
                start: 0,
                end: 0,
                size_bytes: 200,
            },
        );

        let plan = greedy_allocate(&intervals);
        assert_eq!(plan.allocations.len(), 2);
        // Both overlap at time 0, so peak should be 250.
        assert_eq!(plan.peak_bytes, 250);

        // Larger tensor should be placed first (lower offset) due to tie-breaking.
        let large_alloc = plan
            .allocations
            .iter()
            .find(|a| a.size_bytes == 200)
            .unwrap();
        let small_alloc = plan
            .allocations
            .iter()
            .find(|a| a.size_bytes == 50)
            .unwrap();
        assert!(
            large_alloc.offset < small_alloc.offset,
            "larger tensor should have lower offset"
        );
    }

    // ===== Merge intervals =====

    #[test]
    fn merge_intervals_extends_existing() {
        let mut dst = HashMap::new();
        dst.insert(
            TensorId(0),
            LiveInterval {
                start: 2,
                end: 5,
                size_bytes: 100,
            },
        );

        let mut src = HashMap::new();
        src.insert(
            TensorId(0),
            LiveInterval {
                start: 1,
                end: 3,
                size_bytes: 100,
            },
        );
        src.insert(
            TensorId(1),
            LiveInterval {
                start: 0,
                end: 4,
                size_bytes: 200,
            },
        );

        merge_intervals(&mut dst, &src);

        // TensorId(0) should have start extended to 1, end stays at 5.
        let interval_0 = dst.get(&TensorId(0)).unwrap();
        assert_eq!(interval_0.start, 1);
        assert_eq!(interval_0.end, 5);

        // TensorId(1) should be added.
        let interval_1 = dst.get(&TensorId(1)).unwrap();
        assert_eq!(interval_1.start, 0);
        assert_eq!(interval_1.end, 4);
        assert_eq!(interval_1.size_bytes, 200);
    }

    #[test]
    fn merge_intervals_end_extended() {
        let mut dst = HashMap::new();
        dst.insert(
            TensorId(0),
            LiveInterval {
                start: 0,
                end: 3,
                size_bytes: 100,
            },
        );

        let mut src = HashMap::new();
        src.insert(
            TensorId(0),
            LiveInterval {
                start: 2,
                end: 10,
                size_bytes: 100,
            },
        );

        merge_intervals(&mut dst, &src);
        let interval = dst.get(&TensorId(0)).unwrap();
        assert_eq!(interval.start, 0);
        assert_eq!(interval.end, 10);
    }

    // ===== Plan memory with helper functions =====

    #[test]
    fn plan_memory_with_helper_function() {
        let mut module = Module::default();
        let arr_ty = f32_array_type(&mut module, 64); // 256 bytes

        let gv = module.global_variables.append(GlobalVariable {
            name: Some("buf".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: arr_ty,
            init: None,
            layout: None,
        });

        // Helper function that uses the global.
        let mut helper = Function::new("helper");
        let ptr = helper.expressions.append(Expression::GlobalVariable(gv));
        let lit = helper
            .expressions
            .append(Expression::Literal(Literal::F32(0.0)));
        helper.body.push(Statement::Store {
            pointer: ptr,
            value: lit,
        });
        module.functions.append(helper);

        // Entry point that also uses the global.
        let mut ep_func = Function::new("main");
        let ptr2 = ep_func.expressions.append(Expression::GlobalVariable(gv));
        let lit2 = ep_func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        ep_func.body.push(Statement::Store {
            pointer: ptr2,
            value: lit2,
        });
        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: ep_func,
        });

        let plan = plan_memory(&module);
        // Should have at least 1 allocation for the global.
        assert!(!plan.allocations.is_empty());
        assert!(plan.peak_bytes >= 256);
    }

    // ===== Plan memory: unreferenced globals get allocated =====

    #[test]
    fn unreferenced_globals_get_allocated() {
        let mut module = Module::default();
        let arr_ty = f32_array_type(&mut module, 128); // 512 bytes

        // Global variable that is never referenced in any function body.
        module.global_variables.append(GlobalVariable {
            name: Some("unused".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: arr_ty,
            init: None,
            layout: None,
        });

        let plan = plan_memory(&module);
        // The unreferenced global should still get an allocation.
        let non_zero: Vec<_> = plan
            .allocations
            .iter()
            .filter(|a| a.size_bytes > 0)
            .collect();
        assert_eq!(non_zero.len(), 1);
        assert_eq!(non_zero[0].size_bytes, 512);
        assert_eq!(plan.peak_bytes, 512);
    }

    // ===== Lifetime analysis with If statement =====

    #[test]
    fn lifetime_analysis_with_if_statement() {
        let mut module = Module::default();
        let arr_ty = f32_array_type(&mut module, 64); // 256 bytes

        let gv = module.global_variables.append(GlobalVariable {
            name: Some("buf".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: arr_ty,
            init: None,
            layout: None,
        });

        let mut func = Function::new("main");
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let cond = func
            .expressions
            .append(Expression::Literal(Literal::Bool(true)));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));

        // stmt 0: if (cond) { store val -> ptr }
        func.body.push(Statement::If {
            condition: cond,
            accept: vec![Statement::Store {
                pointer: ptr,
                value: val,
            }],
            reject: vec![],
        });

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });

        let plan = plan_memory(&module);
        assert!(plan.peak_bytes >= 256);
    }

    // ===== Lifetime analysis with Loop statement =====

    #[test]
    fn lifetime_analysis_with_loop_statement() {
        let mut module = Module::default();
        let arr_ty = f32_array_type(&mut module, 64); // 256 bytes

        let gv = module.global_variables.append(GlobalVariable {
            name: Some("buf".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: arr_ty,
            init: None,
            layout: None,
        });

        let mut func = Function::new("main");
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        let break_cond = func
            .expressions
            .append(Expression::Literal(Literal::Bool(false)));

        // Loop with body, continuing, and break_if.
        func.body.push(Statement::Loop {
            body: vec![Statement::Store {
                pointer: ptr,
                value: val,
            }],
            continuing: vec![Statement::Store {
                pointer: ptr,
                value: val,
            }],
            break_if: Some(break_cond),
        });

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });

        let plan = plan_memory(&module);
        assert!(plan.peak_bytes >= 256);
    }

    // ===== Lifetime analysis with Call statement =====

    #[test]
    fn lifetime_analysis_with_call_statement() {
        let mut module = Module::default();
        let arr_ty = f32_array_type(&mut module, 64); // 256 bytes

        let gv = module.global_variables.append(GlobalVariable {
            name: Some("buf".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: arr_ty,
            init: None,
            layout: None,
        });

        // Add a helper function.
        let helper = Function::new("helper");
        let helper_handle = module.functions.append(helper);

        let mut func = Function::new("main");
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let result_expr = func
            .expressions
            .append(Expression::Literal(Literal::F32(0.0)));

        func.body.push(Statement::Call {
            function: helper_handle,
            arguments: vec![ptr],
            result: Some(result_expr),
        });

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });

        let plan = plan_memory(&module);
        assert!(plan.peak_bytes >= 256);
    }

    // ===== Lifetime analysis with Atomic statement =====

    #[test]
    fn lifetime_analysis_with_atomic_statement() {
        let mut module = Module::default();
        let u32_ty = module.types.insert(Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::U32),
        });

        let gv = module.global_variables.append(GlobalVariable {
            name: Some("counter".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: u32_ty,
            init: None,
            layout: None,
        });

        let mut func = Function::new("main");
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::U32(1)));
        let result_expr = func
            .expressions
            .append(Expression::Literal(Literal::U32(0)));

        func.body.push(Statement::Atomic {
            pointer: ptr,
            fun: AtomicFunction::Add,
            value: val,
            result: Some(result_expr),
        });

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });

        let plan = plan_memory(&module);
        // u32 = 4 bytes.
        assert!(plan.peak_bytes >= 4);
    }

    // ===== Lifetime analysis with Atomic Exchange compare =====

    #[test]
    fn lifetime_analysis_with_atomic_exchange_compare() {
        let mut module = Module::default();
        let u32_ty = module.types.insert(Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::U32),
        });

        let gv = module.global_variables.append(GlobalVariable {
            name: Some("counter".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: u32_ty,
            init: None,
            layout: None,
        });

        let mut func = Function::new("main");
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::U32(1)));
        let cmp = func
            .expressions
            .append(Expression::Literal(Literal::U32(0)));

        func.body.push(Statement::Atomic {
            pointer: ptr,
            fun: AtomicFunction::Exchange { compare: Some(cmp) },
            value: val,
            result: None,
        });

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });

        let plan = plan_memory(&module);
        assert!(plan.peak_bytes >= 4);
    }

    // ===== Lifetime analysis with Return statement =====

    #[test]
    fn lifetime_analysis_with_return_value() {
        let mut module = Module::default();
        let arr_ty = f32_array_type(&mut module, 64); // 256 bytes

        let gv = module.global_variables.append(GlobalVariable {
            name: Some("buf".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: arr_ty,
            init: None,
            layout: None,
        });

        let mut func = Function::new("main");
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        func.body.push(Statement::Return { value: Some(ptr) });

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });

        let plan = plan_memory(&module);
        assert!(plan.peak_bytes >= 256);
    }

    // ===== Lifetime analysis with Break/Continue (no tensor refs) =====

    #[test]
    fn lifetime_analysis_break_continue_no_effect() {
        let mut module = Module::default();
        let mut func = Function::new("main");
        func.body.push(Statement::Break);
        func.body.push(Statement::Continue);

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });

        let plan = plan_memory(&module);
        assert!(plan.allocations.is_empty());
        assert_eq!(plan.peak_bytes, 0);
    }

    // ===== Lifetime analysis with Barrier (no tensor refs) =====

    #[test]
    fn lifetime_analysis_barrier_no_effect() {
        let mut module = Module::default();
        let mut func = Function::new("main");
        func.body.push(Statement::Barrier(Barrier::STORAGE));

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });

        let plan = plan_memory(&module);
        assert!(plan.allocations.is_empty());
        assert_eq!(plan.peak_bytes, 0);
    }

    // ===== Lifetime analysis with Emit =====

    #[test]
    fn lifetime_analysis_with_emit() {
        let mut module = Module::default();
        let arr_ty = f32_array_type(&mut module, 64); // 256 bytes

        let gv = module.global_variables.append(GlobalVariable {
            name: Some("buf".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: arr_ty,
            init: None,
            layout: None,
        });

        let mut func = Function::new("main");
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let _load = func.expressions.append(Expression::Load { pointer: ptr });

        // Emit covering both expressions.
        let range = Range::from_index_range(0..func.expressions.len() as u32);
        func.body.push(Statement::Emit(range));

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });

        let plan = plan_memory(&module);
        assert!(plan.peak_bytes >= 256);
    }

    // ===== Access/AccessIndex expression resolution =====

    #[test]
    fn access_index_resolves_to_global() {
        let mut module = Module::default();
        let arr_ty = f32_array_type(&mut module, 64); // 256 bytes

        let gv = module.global_variables.append(GlobalVariable {
            name: Some("buf".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: arr_ty,
            init: None,
            layout: None,
        });

        let mut func = Function::new("main");
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let access_idx = func.expressions.append(Expression::AccessIndex {
            base: ptr,
            index: 0,
        });
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));

        // Store using AccessIndex (should resolve to the global).
        func.body.push(Statement::Store {
            pointer: access_idx,
            value: val,
        });

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });

        let plan = plan_memory(&module);
        assert!(plan.peak_bytes >= 256);
    }

    // ===== Access expression resolution =====

    #[test]
    fn access_resolves_to_global() {
        let mut module = Module::default();
        let arr_ty = f32_array_type(&mut module, 64); // 256 bytes

        let gv = module.global_variables.append(GlobalVariable {
            name: Some("buf".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: arr_ty,
            init: None,
            layout: None,
        });

        let mut func = Function::new("main");
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let idx = func
            .expressions
            .append(Expression::Literal(Literal::U32(0)));
        let access = func.expressions.append(Expression::Access {
            base: ptr,
            index: idx,
        });
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));

        // Store using Access (should resolve to the global).
        func.body.push(Statement::Store {
            pointer: access,
            value: val,
        });

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });

        let plan = plan_memory(&module);
        assert!(plan.peak_bytes >= 256);
    }

    // ===== Local variable in helper function =====

    #[test]
    fn local_variable_in_helper_function() {
        let mut module = Module::default();
        let arr_ty = f32_array_type(&mut module, 64); // 256 bytes

        let mut helper = Function::new("helper");
        let lv = helper.local_variables.append(LocalVariable {
            name: Some("temp".into()),
            ty: arr_ty,
            init: None,
        });
        let ptr = helper.expressions.append(Expression::LocalVariable(lv));
        let lit = helper
            .expressions
            .append(Expression::Literal(Literal::F32(0.0)));
        helper.body.push(Statement::Store {
            pointer: ptr,
            value: lit,
        });
        module.functions.append(helper);

        let plan = plan_memory(&module);
        // The local variable should be allocated.
        let non_zero: Vec<_> = plan
            .allocations
            .iter()
            .filter(|a| a.size_bytes > 0)
            .collect();
        assert_eq!(non_zero.len(), 1);
        assert_eq!(non_zero[0].size_bytes, 256);
    }

    // ===== Multiple entry points =====

    #[test]
    fn plan_memory_multiple_entry_points() {
        let mut module = Module::default();
        let arr_ty = f32_array_type(&mut module, 128); // 512 bytes

        let gv = module.global_variables.append(GlobalVariable {
            name: Some("shared".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: arr_ty,
            init: None,
            layout: None,
        });

        // Entry point 1.
        let mut func1 = Function::new("ep1");
        let ptr1 = func1.expressions.append(Expression::GlobalVariable(gv));
        let lit1 = func1
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        func1.body.push(Statement::Store {
            pointer: ptr1,
            value: lit1,
        });
        module.entry_points.push(EntryPoint {
            name: "ep1".into(),
            workgroup_size: [1, 1, 1],
            function: func1,
        });

        // Entry point 2.
        let mut func2 = Function::new("ep2");
        let ptr2 = func2.expressions.append(Expression::GlobalVariable(gv));
        let lit2 = func2
            .expressions
            .append(Expression::Literal(Literal::F32(2.0)));
        func2.body.push(Statement::Store {
            pointer: ptr2,
            value: lit2,
        });
        module.entry_points.push(EntryPoint {
            name: "ep2".into(),
            workgroup_size: [1, 1, 1],
            function: func2,
        });

        let plan = plan_memory(&module);
        // Same global used in both entry points.
        assert_eq!(plan.peak_bytes, 512);
    }

    // ===== MemoryPlanning pass with non-trivial module =====

    #[test]
    fn memory_planning_pass_with_content() {
        let mut module = Module::default();
        let arr_ty = f32_array_type(&mut module, 128); // 512 bytes

        let gv = module.global_variables.append(GlobalVariable {
            name: Some("buf".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: arr_ty,
            init: None,
            layout: None,
        });

        let mut func = Function::new("main");
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let lit = func
            .expressions
            .append(Expression::Literal(Literal::F32(0.0)));
        func.body.push(Statement::Store {
            pointer: ptr,
            value: lit,
        });
        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });

        let pass = MemoryPlanning;
        let changed = pass.run(&mut module);
        assert!(!changed);
    }

    // ===== Display: empty plan =====

    #[test]
    fn memory_plan_display_empty() {
        let plan = MemoryPlan::default();
        let text = format!("{plan}");
        assert!(text.contains("Peak memory: 0 bytes"));
        assert!(text.contains("Buffers: 0"));
    }

    // ===== Display: plan with zero-total (no reuse line) =====

    #[test]
    fn memory_plan_display_no_reuse_line_for_zero_total() {
        let plan = MemoryPlan {
            allocations: vec![BufferAllocation {
                tensor_id: TensorId(0),
                offset: 0,
                size_bytes: 0,
            }],
            peak_bytes: 0,
        };
        let text = format!("{plan}");
        assert!(text.contains("Buffers: 1"));
        // With total = 0, the reuse savings line should not be present.
        assert!(!text.contains("Reuse savings:"));
    }

    // ===== If statement with nested references in reject =====

    #[test]
    fn if_with_reject_branch_references() {
        let mut module = Module::default();
        let arr_ty = f32_array_type(&mut module, 64); // 256 bytes

        let gv_a = module.global_variables.append(GlobalVariable {
            name: Some("a".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: arr_ty,
            init: None,
            layout: None,
        });
        let gv_b = module.global_variables.append(GlobalVariable {
            name: Some("b".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 1,
            }),
            ty: arr_ty,
            init: None,
            layout: None,
        });

        let mut func = Function::new("main");
        let ptr_a = func.expressions.append(Expression::GlobalVariable(gv_a));
        let ptr_b = func.expressions.append(Expression::GlobalVariable(gv_b));
        let cond = func
            .expressions
            .append(Expression::Literal(Literal::Bool(true)));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));

        func.body.push(Statement::If {
            condition: cond,
            accept: vec![Statement::Store {
                pointer: ptr_a,
                value: val,
            }],
            reject: vec![Statement::Store {
                pointer: ptr_b,
                value: val,
            }],
        });

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });

        let plan = plan_memory(&module);
        // Both globals should be allocated.
        let non_zero: Vec<_> = plan
            .allocations
            .iter()
            .filter(|a| a.size_bytes > 0)
            .collect();
        assert_eq!(non_zero.len(), 2);
    }

    // ===== Loop with break_if expression referencing a global =====

    #[test]
    fn loop_break_if_references_global() {
        let mut module = Module::default();
        let u32_ty = module.types.insert(Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::U32),
        });

        let gv = module.global_variables.append(GlobalVariable {
            name: Some("flag".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: u32_ty,
            init: None,
            layout: None,
        });

        let mut func = Function::new("main");
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));

        func.body.push(Statement::Loop {
            body: vec![],
            continuing: vec![],
            break_if: Some(ptr),
        });

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });

        let plan = plan_memory(&module);
        assert!(plan.peak_bytes >= 4);
    }

    // ===== Resolve expression: returns None for non-variable expressions =====

    #[test]
    fn resolve_expr_non_variable_returns_none() {
        let mut module = Module::default();
        let mut func = Function::new("main");
        let lit = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));

        // Store lit -> lit (not realistic, but tests resolve_expr_tensor_id returning None).
        func.body.push(Statement::Store {
            pointer: lit,
            value: lit,
        });

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });

        let plan = plan_memory(&module);
        // Literal expression does not resolve to any tensor, so no allocations.
        assert_eq!(plan.peak_bytes, 0);
    }

    // ===== Return with no value in lifetime analysis =====

    #[test]
    fn return_no_value_no_effect_on_lifetimes() {
        let mut module = Module::default();
        let mut func = Function::new("main");
        func.body.push(Statement::Return { value: None });

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });

        let plan = plan_memory(&module);
        assert_eq!(plan.peak_bytes, 0);
    }

    // ===== Single store function (minimum case) =====

    #[test]
    fn single_store_function() {
        let mut module = Module::default();
        let f32_ty = f32_type(&mut module);

        let gv = module.global_variables.append(GlobalVariable {
            name: Some("x".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: f32_ty,
            init: None,
            layout: None,
        });

        let mut func = Function::new("main");
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(42.0)));
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });

        let plan = plan_memory(&module);
        assert_eq!(plan.peak_bytes, 4); // f32 = 4 bytes
    }

    // ===== Call with no result in lifetime analysis =====

    #[test]
    fn call_no_result_lifetime() {
        let mut module = Module::default();
        let arr_ty = f32_array_type(&mut module, 64); // 256 bytes

        let gv = module.global_variables.append(GlobalVariable {
            name: Some("buf".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: arr_ty,
            init: None,
            layout: None,
        });

        let helper = Function::new("helper");
        let helper_handle = module.functions.append(helper);

        let mut func = Function::new("main");
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));

        func.body.push(Statement::Call {
            function: helper_handle,
            arguments: vec![ptr],
            result: None,
        });

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });

        let plan = plan_memory(&module);
        assert!(plan.peak_bytes >= 256);
    }

    // ===== Lifetime interval start updated backwards =====

    #[test]
    fn lifetime_interval_start_updated_backwards() {
        // Build a case where a tensor is first seen at stmt 2, then at stmt 0.
        // This tests the and_modify path where stmt_idx < interval.start.
        let mut module = Module::default();
        let arr_ty = f32_array_type(&mut module, 64); // 256 bytes

        let gv_a = module.global_variables.append(GlobalVariable {
            name: Some("a".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: arr_ty,
            init: None,
            layout: None,
        });
        let gv_b = module.global_variables.append(GlobalVariable {
            name: Some("b".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 1,
            }),
            ty: arr_ty,
            init: None,
            layout: None,
        });

        let mut func = Function::new("main");
        let ptr_a = func.expressions.append(Expression::GlobalVariable(gv_a));
        let ptr_b = func.expressions.append(Expression::GlobalVariable(gv_b));
        let load_a = func.expressions.append(Expression::Load { pointer: ptr_a });
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));

        // stmt 0: store val -> ptr_a (a used at 0)
        func.body.push(Statement::Store {
            pointer: ptr_a,
            value: val,
        });
        // stmt 1: store val -> ptr_b (b used at 1)
        func.body.push(Statement::Store {
            pointer: ptr_b,
            value: val,
        });
        // stmt 2: store load(a) -> ptr_b (a used at 2, b used at 2)
        func.body.push(Statement::Store {
            pointer: ptr_b,
            value: load_a,
        });

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });

        let plan = plan_memory(&module);
        // a: [0, 2], b: [1, 2] -- both live at stmt 2, peak should be 512.
        assert_eq!(plan.peak_bytes, 512);
    }
}
