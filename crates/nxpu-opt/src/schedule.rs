//! Operation scheduling pass.
//!
//! Implements list scheduling with critical-path priority to determine
//! an efficient execution order for operations in a function body.
//! Identifies parallel execution opportunities by grouping independent
//! operations into time slots.

use std::fmt;

use nxpu_analysis::DataflowGraph;
use nxpu_ir::Module;

use crate::Pass;

/// A time slot in a schedule, containing one or more operations
/// that can execute concurrently.
#[derive(Clone, Debug)]
pub struct ScheduleSlot {
    /// The time step (0-indexed).
    pub time: usize,
    /// Node IDs of operations scheduled in this slot.
    pub ops: Vec<usize>,
}

/// A complete schedule for a function, mapping operations to time slots.
#[derive(Clone, Debug)]
pub struct Schedule {
    /// Ordered time slots.
    pub slots: Vec<ScheduleSlot>,
}

impl Schedule {
    /// Returns the total number of time steps in the schedule.
    pub fn total_time(&self) -> usize {
        self.slots.len()
    }

    /// Returns the total number of scheduled operations.
    pub fn total_ops(&self) -> usize {
        self.slots.iter().map(|s| s.ops.len()).sum()
    }

    /// Returns the maximum parallelism (most ops in a single slot).
    pub fn max_parallelism(&self) -> usize {
        self.slots.iter().map(|s| s.ops.len()).max().unwrap_or(0)
    }

    /// Returns `true` if the schedule respects all dependency edges in the DFG.
    ///
    /// For every edge `from -> to`, the time of `from` must be strictly less
    /// than the time of `to`.
    pub fn is_valid(&self, dfg: &DataflowGraph) -> bool {
        // Build a map from node_id to time slot.
        let mut node_time: Vec<Option<usize>> = vec![None; dfg.node_count()];
        for slot in &self.slots {
            for &op in &slot.ops {
                if op < node_time.len() {
                    node_time[op] = Some(slot.time);
                }
            }
        }

        // Every edge must have from.time < to.time.
        for edge in dfg.edges() {
            let from_time = match node_time.get(edge.from) {
                Some(Some(t)) => *t,
                _ => return false,
            };
            let to_time = match node_time.get(edge.to) {
                Some(Some(t)) => *t,
                _ => return false,
            };
            if from_time >= to_time {
                return false;
            }
        }

        true
    }
}

impl fmt::Display for Schedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Schedule ({} slots, {} ops):",
            self.total_time(),
            self.total_ops()
        )?;
        for slot in &self.slots {
            writeln!(f, "  t={}: {:?}", slot.time, slot.ops)?;
        }
        Ok(())
    }
}

/// Perform list scheduling on a dataflow graph.
///
/// Uses critical-path-based priority: nodes on or near the critical path
/// are scheduled first. This is a greedy heuristic that works well for
/// many practical cases.
///
/// # Algorithm
///
/// 1. Compute critical path distances for all nodes.
/// 2. Initialize ready set with nodes that have no predecessors.
/// 3. At each time step, select the highest-priority ready node(s).
/// 4. After scheduling a node, add newly ready successors.
/// 5. Continue until all nodes are scheduled.
pub fn list_schedule(dfg: &DataflowGraph) -> Schedule {
    let n = dfg.node_count();
    if n == 0 {
        return Schedule { slots: Vec::new() };
    }

    // Compute critical path priorities (higher distance = higher priority).
    let cp_result = dfg.critical_path();
    let priority = &cp_result.node_distances;

    // Build successor and predecessor lists.
    let mut preds_count = vec![0usize; n];
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];

    for edge in dfg.edges() {
        preds_count[edge.to] += 1;
        successors[edge.from].push(edge.to);
    }

    // Track remaining predecessor count for each node.
    let mut remaining_preds = preds_count.clone();

    // Initialize ready set (nodes with no predecessors).
    let mut ready: Vec<usize> = Vec::new();
    for (i, &pred_count) in remaining_preds.iter().enumerate() {
        if pred_count == 0 {
            ready.push(i);
        }
    }

    let mut slots: Vec<ScheduleSlot> = Vec::new();
    let mut scheduled = vec![false; n];
    let mut time = 0;

    while !ready.is_empty() {
        // Sort ready nodes by priority (descending: higher priority first).
        ready.sort_by(|&a, &b| {
            priority
                .get(b)
                .unwrap_or(&0)
                .cmp(priority.get(a).unwrap_or(&0))
                .then_with(|| a.cmp(&b)) // tie-break by id for determinism
        });

        // All ready nodes can execute in parallel in this time slot.
        let slot_ops: Vec<usize> = std::mem::take(&mut ready);

        for &op in &slot_ops {
            scheduled[op] = true;
        }

        // Find newly ready successors.
        let mut next_ready: Vec<usize> = Vec::new();
        for &op in &slot_ops {
            for &succ in &successors[op] {
                remaining_preds[succ] -= 1;
                if remaining_preds[succ] == 0 && !scheduled[succ] && !next_ready.contains(&succ) {
                    next_ready.push(succ);
                }
            }
        }

        slots.push(ScheduleSlot {
            time,
            ops: slot_ops,
        });

        ready = next_ready;
        time += 1;
    }

    Schedule { slots }
}

/// Convenience function: build DFG and schedule for an IR function.
pub fn schedule_function(func: &nxpu_ir::Function) -> (DataflowGraph, Schedule) {
    let dfg = DataflowGraph::build(func);
    let schedule = list_schedule(&dfg);
    (dfg, schedule)
}

/// An optimization pass that computes and logs the schedule for debugging.
///
/// This pass does NOT modify the module. It only computes and logs schedules
/// for all entry points when the `RUST_LOG` level includes debug messages.
/// It can also be used to attach schedule metadata.
#[derive(Debug)]
pub struct SchedulePass;

impl Pass for SchedulePass {
    fn name(&self) -> &str {
        "schedule"
    }

    fn run(&self, module: &mut Module) -> bool {
        for ep in &module.entry_points {
            let (dfg, schedule) = schedule_function(&ep.function);
            let cp = dfg.critical_path();
            log::debug!(
                "entry point '{}': {} nodes, {} edges, critical path length {}, schedule: {} slots",
                ep.name,
                dfg.node_count(),
                dfg.edge_count(),
                cp.critical_path_length,
                schedule.total_time(),
            );
        }
        // This pass is analysis-only; it does not modify the module.
        false
    }
}

/// Compute schedules for all entry points in a module.
///
/// Returns a list of `(entry_point_name, DataflowGraph, Schedule)` triples.
pub fn compute_schedules(module: &Module) -> Vec<(String, DataflowGraph, Schedule)> {
    module
        .entry_points
        .iter()
        .map(|ep| {
            let (dfg, schedule) = schedule_function(&ep.function);
            (ep.name.clone(), dfg, schedule)
        })
        .collect()
}

/// Format a schedule for human-readable output (used by --emit-schedule).
pub fn format_schedule(name: &str, dfg: &DataflowGraph, schedule: &Schedule) -> String {
    let cp = dfg.critical_path();
    let mut out = String::new();

    out.push_str(&format!("=== Schedule for '{}' ===\n", name));
    out.push_str(&format!(
        "Nodes: {}, Edges: {}, Critical path: {}\n",
        dfg.node_count(),
        dfg.edge_count(),
        cp.critical_path_length,
    ));
    out.push_str(&format!(
        "Time slots: {}, Max parallelism: {}\n\n",
        schedule.total_time(),
        schedule.max_parallelism(),
    ));

    for slot in &schedule.slots {
        out.push_str(&format!("  t={}: ", slot.time));
        let descs: Vec<String> = slot
            .ops
            .iter()
            .map(|&id| {
                let node = &dfg.nodes()[id];
                format!("node[{}] {}", id, node.kind)
            })
            .collect();
        out.push_str(&descs.join(", "));
        out.push('\n');
    }

    if !cp.critical_path.is_empty() {
        out.push_str("\n  Critical path: ");
        let path_strs: Vec<String> = cp
            .critical_path
            .iter()
            .map(|&id| format!("{}", id))
            .collect();
        out.push_str(&path_strs.join(" -> "));
        out.push('\n');
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use nxpu_ir::{Expression, Function, Literal, Statement};

    fn dummy_gv_handle() -> nxpu_ir::Handle<nxpu_ir::GlobalVariable> {
        let mut arena = nxpu_ir::Arena::new();
        arena.append(nxpu_ir::GlobalVariable {
            name: None,
            space: nxpu_ir::AddressSpace::Storage {
                access: nxpu_ir::StorageAccess::LOAD | nxpu_ir::StorageAccess::STORE,
            },
            binding: None,
            ty: {
                let mut types = nxpu_ir::UniqueArena::new();
                types.insert(nxpu_ir::Type {
                    name: None,
                    inner: nxpu_ir::TypeInner::Scalar(nxpu_ir::Scalar::F32),
                })
            },
            init: None,
            layout: None,
        })
    }

    #[test]
    fn schedule_empty_function() {
        let func = Function::new("test");
        let (dfg, schedule) = schedule_function(&func);
        assert_eq!(dfg.node_count(), 0);
        assert_eq!(schedule.total_time(), 0);
        assert_eq!(schedule.total_ops(), 0);
        assert_eq!(schedule.max_parallelism(), 0);
    }

    #[test]
    fn schedule_single_statement() {
        let mut func = Function::new("test");
        let gv = dummy_gv_handle();
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });

        let (dfg, schedule) = schedule_function(&func);
        assert_eq!(dfg.node_count(), 1);
        assert_eq!(schedule.total_time(), 1);
        assert_eq!(schedule.total_ops(), 1);
    }

    #[test]
    fn schedule_dependent_chain() {
        let mut func = Function::new("test");
        let gv = dummy_gv_handle();
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));

        // Three stores to the same pointer (WAW dependency chain).
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });

        let (dfg, schedule) = schedule_function(&func);
        assert_eq!(dfg.node_count(), 3);
        // Must be serialized due to WAW dependencies.
        assert_eq!(schedule.total_time(), 3);
        assert!(
            schedule.is_valid(&dfg),
            "schedule must respect dependencies"
        );
    }

    #[test]
    fn schedule_independent_ops_parallel() {
        let mut func = Function::new("test");

        // Two stores to different pointers with different values.
        let gv0 = dummy_gv_handle();
        let gv1 = {
            let mut arena = nxpu_ir::Arena::new();
            let _ = arena.append(nxpu_ir::GlobalVariable {
                name: None,
                space: nxpu_ir::AddressSpace::Storage {
                    access: nxpu_ir::StorageAccess::STORE,
                },
                binding: None,
                ty: {
                    let mut types = nxpu_ir::UniqueArena::new();
                    types.insert(nxpu_ir::Type {
                        name: None,
                        inner: nxpu_ir::TypeInner::Scalar(nxpu_ir::Scalar::F32),
                    })
                },
                init: None,
                layout: None,
            });
            arena.append(nxpu_ir::GlobalVariable {
                name: None,
                space: nxpu_ir::AddressSpace::Storage {
                    access: nxpu_ir::StorageAccess::STORE,
                },
                binding: None,
                ty: {
                    let mut types = nxpu_ir::UniqueArena::new();
                    types.insert(nxpu_ir::Type {
                        name: None,
                        inner: nxpu_ir::TypeInner::Scalar(nxpu_ir::Scalar::F32),
                    })
                },
                init: None,
                layout: None,
            })
        };

        let ptr_a = func.expressions.append(Expression::GlobalVariable(gv0));
        let val_a = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        let ptr_b = func.expressions.append(Expression::GlobalVariable(gv1));
        let val_b = func
            .expressions
            .append(Expression::Literal(Literal::F32(2.0)));

        func.body.push(Statement::Store {
            pointer: ptr_a,
            value: val_a,
        });
        func.body.push(Statement::Store {
            pointer: ptr_b,
            value: val_b,
        });

        let (dfg, schedule) = schedule_function(&func);
        assert_eq!(dfg.node_count(), 2);
        // Independent ops should be in the same time slot.
        assert!(
            schedule.max_parallelism() >= 2,
            "expected parallel execution of independent stores"
        );
        assert!(
            schedule.is_valid(&dfg),
            "schedule must respect dependencies"
        );
    }

    #[test]
    fn schedule_respects_barrier() {
        let mut func = Function::new("test");
        let gv = dummy_gv_handle();
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));

        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });
        func.body
            .push(Statement::Barrier(nxpu_ir::Barrier::STORAGE));
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });

        let (dfg, schedule) = schedule_function(&func);
        assert!(
            schedule.is_valid(&dfg),
            "schedule must respect barrier deps"
        );
        // At minimum 2 time slots: store+barrier can't overlap with post-barrier store.
        assert!(
            schedule.total_time() >= 2,
            "barrier should enforce ordering"
        );
    }

    #[test]
    fn schedule_display() {
        let func = Function::new("test");
        let (_, schedule) = schedule_function(&func);
        let s = format!("{schedule}");
        assert!(s.contains("Schedule"));
    }

    #[test]
    fn format_schedule_output() {
        let mut func = Function::new("test");
        let gv = dummy_gv_handle();
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });

        let (dfg, schedule) = schedule_function(&func);
        let output = format_schedule("test_ep", &dfg, &schedule);
        assert!(output.contains("test_ep"));
        assert!(output.contains("Nodes:"));
        assert!(output.contains("t=0"));
    }

    #[test]
    fn schedule_pass_runs_on_module() {
        let pass = SchedulePass;
        let mut module = Module::default();
        // No entry points -- should not crash.
        let changed = pass.run(&mut module);
        assert!(!changed, "schedule pass should not modify the module");
    }

    #[test]
    fn compute_schedules_empty_module() {
        let module = Module::default();
        let schedules = compute_schedules(&module);
        assert!(schedules.is_empty());
    }

    #[test]
    fn schedule_validity_check() {
        let mut func = Function::new("test");
        let gv = dummy_gv_handle();
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });

        let (dfg, schedule) = schedule_function(&func);
        assert!(schedule.is_valid(&dfg));

        // Create an invalid schedule (both ops at same time with dependency).
        let invalid = Schedule {
            slots: vec![ScheduleSlot {
                time: 0,
                ops: vec![0, 1],
            }],
        };
        assert!(
            !invalid.is_valid(&dfg),
            "schedule with dependent ops at same time should be invalid"
        );
    }

    #[test]
    fn schedule_pass_with_entry_points() {
        let mut module = Module::default();
        let mut func = Function::new("ep_test");
        let gv = dummy_gv_handle();
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(42.0)));
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });

        module.entry_points.push(nxpu_ir::EntryPoint {
            name: "ep_test".into(),
            workgroup_size: [1, 1, 1],
            function: func,
        });

        let pass = SchedulePass;
        let changed = pass.run(&mut module);
        assert!(!changed, "schedule pass should not modify the module");
    }

    #[test]
    fn compute_schedules_with_entry_points() {
        let mut module = Module::default();

        let mut func1 = Function::new("ep1");
        let gv = dummy_gv_handle();
        let ptr = func1.expressions.append(Expression::GlobalVariable(gv));
        let val = func1
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        func1.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });

        let mut func2 = Function::new("ep2");
        let gv2 = dummy_gv_handle();
        let ptr2 = func2.expressions.append(Expression::GlobalVariable(gv2));
        let val2 = func2
            .expressions
            .append(Expression::Literal(Literal::F32(2.0)));
        func2.body.push(Statement::Store {
            pointer: ptr2,
            value: val2,
        });
        func2.body.push(Statement::Store {
            pointer: ptr2,
            value: val2,
        });

        module.entry_points.push(nxpu_ir::EntryPoint {
            name: "ep1".into(),
            workgroup_size: [1, 1, 1],
            function: func1,
        });
        module.entry_points.push(nxpu_ir::EntryPoint {
            name: "ep2".into(),
            workgroup_size: [1, 1, 1],
            function: func2,
        });

        let schedules = compute_schedules(&module);
        assert_eq!(schedules.len(), 2);
        assert_eq!(schedules[0].0, "ep1");
        assert_eq!(schedules[1].0, "ep2");
        assert_eq!(schedules[0].2.total_ops(), 1);
        assert_eq!(schedules[1].2.total_ops(), 2);
    }

    #[test]
    fn format_schedule_with_critical_path() {
        let mut func = Function::new("test");
        let gv = dummy_gv_handle();
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));

        // Create a chain to produce a non-empty critical path
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });

        let (dfg, schedule) = schedule_function(&func);
        let output = format_schedule("test_chain", &dfg, &schedule);
        assert!(output.contains("test_chain"));
        assert!(output.contains("Nodes: 3"));
        assert!(output.contains("Critical path:"));
    }

    #[test]
    fn schedule_display_with_ops() {
        let mut func = Function::new("test");
        let gv = dummy_gv_handle();
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });

        let (_, schedule) = schedule_function(&func);
        let s = format!("{schedule}");
        assert!(s.contains("Schedule (2 slots, 2 ops):"));
        assert!(s.contains("t=0"));
        assert!(s.contains("t=1"));
    }

    #[test]
    fn schedule_is_valid_missing_node() {
        // Build a DFG with 2 nodes, but schedule only contains 1 node.
        // The missing node should cause is_valid to return false.
        let mut func = Function::new("test");
        let gv = dummy_gv_handle();
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });

        let dfg = DataflowGraph::build(&func);
        // Only schedule node 0, missing node 1
        let partial = Schedule {
            slots: vec![ScheduleSlot {
                time: 0,
                ops: vec![0],
            }],
        };
        // The edge from 0->1 exists, but node 1 has no time slot -> invalid
        assert!(!partial.is_valid(&dfg));
    }
}
