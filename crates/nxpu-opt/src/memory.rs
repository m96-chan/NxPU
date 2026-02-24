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
                        nxpu_ir::Dimension::Dynamic(_) => 0,
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
}
