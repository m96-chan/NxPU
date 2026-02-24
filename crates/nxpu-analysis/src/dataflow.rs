//! Dataflow graph construction and analysis.
//!
//! Builds a dataflow graph (DFG) from an IR function by analyzing
//! read/write sets of each statement and computing data dependencies
//! (RAW, WAR, WAW) and control dependencies (barriers).
//!
//! Provides topological sorting and critical path analysis.

use std::collections::{BTreeSet, HashSet, VecDeque};
use std::fmt;

use nxpu_ir::{Expression, Function, Handle, Statement};

/// Errors during dataflow analysis.
#[derive(Debug, thiserror::Error)]
pub enum DataflowError {
    /// The DFG contains a cycle, which should not occur in well-formed SSA IR.
    #[error("cycle detected in dataflow graph ({visited} of {total} nodes visited)")]
    CycleDetected { visited: usize, total: usize },
}

/// The kind of a DFG node, indicating what type of statement it represents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DfgNodeKind {
    /// A `Store` statement (write through a pointer).
    Store,
    /// A function `Call` statement.
    Call,
    /// An `Atomic` operation statement.
    Atomic,
    /// A synchronization `Barrier` statement.
    Barrier,
    /// An `Emit` statement (makes expression results available).
    Emit,
    /// An `If` control flow statement.
    If,
    /// A `Loop` control flow statement.
    Loop,
    /// A `Return` statement.
    Return,
    /// A `Break` statement.
    Break,
    /// A `Continue` statement.
    Continue,
}

impl fmt::Display for DfgNodeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Store => "Store",
            Self::Call => "Call",
            Self::Atomic => "Atomic",
            Self::Barrier => "Barrier",
            Self::Emit => "Emit",
            Self::If => "If",
            Self::Loop => "Loop",
            Self::Return => "Return",
            Self::Break => "Break",
            Self::Continue => "Continue",
        })
    }
}

/// The kind of dependency between two DFG nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DependencyKind {
    /// Read-after-write: consumer reads what producer wrote.
    DataFlow,
    /// Write-after-read: producer reads, consumer writes to same location.
    AntiDependency,
    /// Write-after-write: both nodes write to the same location.
    OutputDependency,
    /// Control dependency (barriers, control flow).
    Control,
}

impl fmt::Display for DependencyKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::DataFlow => "RAW",
            Self::AntiDependency => "WAR",
            Self::OutputDependency => "WAW",
            Self::Control => "Control",
        })
    }
}

/// A node in the dataflow graph representing a statement.
#[derive(Clone, Debug)]
pub struct DfgNode {
    /// Unique node identifier (index into the DFG node array).
    pub id: usize,
    /// What kind of statement this node represents.
    pub kind: DfgNodeKind,
    /// Expression handles read by this statement.
    pub reads: Vec<Handle<Expression>>,
    /// Expression handles written by this statement.
    pub writes: Vec<Handle<Expression>>,
}

/// An edge in the dataflow graph representing a dependency.
#[derive(Clone, Debug)]
pub struct DfgEdge {
    /// Index of the producer (source) node.
    pub from: usize,
    /// Index of the consumer (destination) node.
    pub to: usize,
    /// Kind of dependency.
    pub kind: DependencyKind,
}

/// A dataflow graph built from an IR function's statements.
///
/// Nodes correspond to statements and edges represent data/control
/// dependencies between them.
#[derive(Clone, Debug)]
pub struct DataflowGraph {
    /// Nodes in the graph, indexed by their `id`.
    nodes: Vec<DfgNode>,
    /// Edges representing dependencies.
    edges: Vec<DfgEdge>,
}

impl DataflowGraph {
    /// Build a dataflow graph from an IR function.
    ///
    /// Walks statements in order, computes read/write sets, and creates
    /// dependency edges:
    /// - RAW (DataFlow): a write followed by a read of the same expression
    /// - WAR (AntiDependency): a read followed by a write of the same expression
    /// - WAW (OutputDependency): two writes to the same expression
    /// - Control: barriers create control edges to all subsequent memory operations
    pub fn build(func: &Function) -> Self {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // Build nodes from top-level statements.
        for stmt in &func.body {
            let id = nodes.len();
            let (kind, reads, writes) = classify_statement(stmt, &func.expressions);
            nodes.push(DfgNode {
                id,
                kind,
                reads,
                writes,
            });
        }

        // Build dependency edges by comparing read/write sets.
        for (i, node_i) in nodes.iter().enumerate() {
            for (j, node_j) in nodes.iter().enumerate().skip(i + 1) {
                // RAW: node i writes, node j reads the same handle.
                for w in &node_i.writes {
                    if node_j.reads.contains(w) {
                        edges.push(DfgEdge {
                            from: i,
                            to: j,
                            kind: DependencyKind::DataFlow,
                        });
                    }
                }

                // WAR: node i reads, node j writes the same handle.
                for r in &node_i.reads {
                    if node_j.writes.contains(r) {
                        edges.push(DfgEdge {
                            from: i,
                            to: j,
                            kind: DependencyKind::AntiDependency,
                        });
                    }
                }

                // WAW: node i writes, node j writes the same handle.
                for w in &node_i.writes {
                    if node_j.writes.contains(w) {
                        edges.push(DfgEdge {
                            from: i,
                            to: j,
                            kind: DependencyKind::OutputDependency,
                        });
                    }
                }
            }
        }

        // Barriers create control edges to all subsequent memory operations.
        for (i, node_i) in nodes.iter().enumerate() {
            if node_i.kind == DfgNodeKind::Barrier {
                for (j, node_j) in nodes.iter().enumerate().skip(i + 1) {
                    let has_memory_effect = !node_j.reads.is_empty()
                        || !node_j.writes.is_empty()
                        || node_j.kind == DfgNodeKind::Barrier;
                    if has_memory_effect {
                        edges.push(DfgEdge {
                            from: i,
                            to: j,
                            kind: DependencyKind::Control,
                        });
                    }
                }
            }
        }

        Self { nodes, edges }
    }

    /// Returns the nodes in this graph.
    pub fn nodes(&self) -> &[DfgNode] {
        &self.nodes
    }

    /// Returns the edges in this graph.
    pub fn edges(&self) -> &[DfgEdge] {
        &self.edges
    }

    /// Returns the number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Returns all edges originating from the given node.
    pub fn successors(&self, node_id: usize) -> Vec<&DfgEdge> {
        self.edges.iter().filter(|e| e.from == node_id).collect()
    }

    /// Returns all edges pointing to the given node.
    pub fn predecessors(&self, node_id: usize) -> Vec<&DfgEdge> {
        self.edges.iter().filter(|e| e.to == node_id).collect()
    }

    /// Perform topological sort using Kahn's algorithm.
    ///
    /// Returns nodes in a valid execution order respecting all dependencies.
    ///
    /// # Errors
    ///
    /// Returns [`DataflowError::CycleDetected`] if the graph contains a cycle.
    pub fn topological_sort(&self) -> Result<Vec<usize>, DataflowError> {
        let n = self.nodes.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // Build in-degree and adjacency.
        let mut in_degree = vec![0usize; n];
        let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];

        for edge in &self.edges {
            in_degree[edge.to] += 1;
            successors[edge.from].push(edge.to);
        }

        // Use BTreeSet for deterministic ordering (lower id first).
        let mut ready: BTreeSet<usize> = BTreeSet::new();
        for (i, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                ready.insert(i);
            }
        }

        let mut result = Vec::with_capacity(n);

        while let Some(&node_id) = ready.iter().next() {
            ready.remove(&node_id);
            result.push(node_id);

            for &succ in &successors[node_id] {
                in_degree[succ] -= 1;
                if in_degree[succ] == 0 {
                    ready.insert(succ);
                }
            }
        }

        if result.len() != n {
            return Err(DataflowError::CycleDetected {
                visited: result.len(),
                total: n,
            });
        }

        Ok(result)
    }

    /// Compute the critical path through the DFG.
    ///
    /// Assigns a unit cost of 1 to each node and computes the longest path
    /// from any source node (no predecessors) to any sink node (no successors).
    ///
    /// Returns the cost for each node (longest path from any source to that node)
    /// and the critical path length.
    pub fn critical_path(&self) -> CriticalPathResult {
        self.critical_path_with_costs(&vec![1usize; self.nodes.len()])
    }

    /// Compute the critical path with custom per-node costs.
    ///
    /// `costs[i]` is the execution cost of node `i`.
    ///
    /// Returns the cost for each node and the critical path length.
    pub fn critical_path_with_costs(&self, costs: &[usize]) -> CriticalPathResult {
        let n = self.nodes.len();
        if n == 0 {
            return CriticalPathResult {
                node_distances: Vec::new(),
                critical_path_length: 0,
                critical_path: Vec::new(),
            };
        }

        // Compute topological order (handle cycles gracefully).
        let topo = match self.topological_sort() {
            Ok(order) => order,
            Err(_) => {
                return CriticalPathResult {
                    node_distances: vec![0; n],
                    critical_path_length: 0,
                    critical_path: Vec::new(),
                };
            }
        };

        // Build predecessor lists.
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
        for edge in &self.edges {
            preds[edge.to].push(edge.from);
        }

        // Compute longest distance from any source to each node.
        let mut dist = vec![0usize; n];
        // Track the predecessor on the critical path.
        let mut prev: Vec<Option<usize>> = vec![None; n];

        for &node_id in &topo {
            let node_cost = costs.get(node_id).copied().unwrap_or(1);
            let mut best_dist = 0;
            let mut best_pred = None;

            for &pred_id in &preds[node_id] {
                if dist[pred_id] > best_dist {
                    best_dist = dist[pred_id];
                    best_pred = Some(pred_id);
                }
            }

            dist[node_id] = best_dist + node_cost;
            prev[node_id] = best_pred;
        }

        // Find the sink with the maximum distance (the critical path endpoint).
        let mut max_dist = 0;
        let mut max_node = 0;
        for (i, &d) in dist.iter().enumerate() {
            if d > max_dist {
                max_dist = d;
                max_node = i;
            }
        }

        // Reconstruct the critical path by following prev pointers.
        let mut path = VecDeque::new();
        let mut current = Some(max_node);
        while let Some(node_id) = current {
            path.push_front(node_id);
            current = prev[node_id];
        }

        CriticalPathResult {
            node_distances: dist,
            critical_path_length: max_dist,
            critical_path: path.into(),
        }
    }

    /// Identify groups of independent nodes that can execute concurrently.
    ///
    /// Returns a list of groups, where each group contains node IDs that have
    /// no dependencies between them and can execute in parallel.
    pub fn parallel_groups(&self) -> Vec<Vec<usize>> {
        let n = self.nodes.len();
        if n == 0 {
            return Vec::new();
        }

        let topo = match self.topological_sort() {
            Ok(order) => order,
            Err(_) => return Vec::new(),
        };

        // Build predecessor/successor sets for dependency checks.
        let mut has_pred: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        for edge in &self.edges {
            has_pred[edge.to].insert(edge.from);
        }

        // Compute ASAP (As Soon As Possible) time for each node.
        let mut asap = vec![0usize; n];
        for &node_id in &topo {
            let mut max_pred_time = 0;
            for &pred_id in &has_pred[node_id] {
                let pred_finish = asap[pred_id] + 1;
                if pred_finish > max_pred_time {
                    max_pred_time = pred_finish;
                }
            }
            asap[node_id] = max_pred_time;
        }

        // Group nodes by their ASAP time.
        let max_time = asap.iter().copied().max().unwrap_or(0);
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); max_time + 1];
        for &node_id in &topo {
            groups[asap[node_id]].push(node_id);
        }

        // Filter out empty groups.
        groups.retain(|g| !g.is_empty());
        groups
    }
}

/// Result of critical path analysis.
#[derive(Clone, Debug)]
pub struct CriticalPathResult {
    /// The longest-path distance from any source to each node.
    pub node_distances: Vec<usize>,
    /// The length of the critical path (maximum distance).
    pub critical_path_length: usize,
    /// Node IDs on the critical path, from source to sink.
    pub critical_path: Vec<usize>,
}

/// Classify a statement into its DFG node kind and compute read/write sets.
fn classify_statement(
    stmt: &Statement,
    expressions: &nxpu_ir::Arena<Expression>,
) -> (
    DfgNodeKind,
    Vec<Handle<Expression>>,
    Vec<Handle<Expression>>,
) {
    match stmt {
        Statement::Store { pointer, value } => {
            let mut reads = Vec::new();
            reads.push(*value);
            collect_expr_reads(*value, expressions, &mut reads);
            collect_expr_reads(*pointer, expressions, &mut reads);
            let writes = vec![*pointer];
            (DfgNodeKind::Store, reads, writes)
        }
        Statement::Call {
            arguments, result, ..
        } => {
            let mut reads: Vec<Handle<Expression>> = arguments.clone();
            for &arg in arguments {
                collect_expr_reads(arg, expressions, &mut reads);
            }
            let writes = result.iter().copied().collect();
            (DfgNodeKind::Call, reads, writes)
        }
        Statement::Atomic {
            pointer,
            value,
            result,
            fun,
        } => {
            let mut reads = vec![*pointer, *value];
            collect_expr_reads(*pointer, expressions, &mut reads);
            collect_expr_reads(*value, expressions, &mut reads);
            if let nxpu_ir::AtomicFunction::Exchange {
                compare: Some(cmp), ..
            } = fun
            {
                reads.push(*cmp);
                collect_expr_reads(*cmp, expressions, &mut reads);
            }
            let mut writes = vec![*pointer];
            if let Some(r) = result {
                writes.push(*r);
            }
            (DfgNodeKind::Atomic, reads, writes)
        }
        Statement::Barrier(_) => (DfgNodeKind::Barrier, Vec::new(), Vec::new()),
        Statement::Emit(range) => {
            let mut reads = Vec::new();
            let mut writes = Vec::new();
            let idx_range = range.index_range();
            // Iterate expression arena and pick handles within the emit range.
            for (handle, expr) in expressions.iter() {
                let idx = handle.index() as u32;
                if idx >= idx_range.start && idx < idx_range.end {
                    writes.push(handle);
                    for operand in expression_operands(expr) {
                        if !reads.contains(&operand) {
                            reads.push(operand);
                        }
                    }
                }
            }
            (DfgNodeKind::Emit, reads, writes)
        }
        Statement::If {
            condition,
            accept,
            reject,
        } => {
            let mut reads = vec![*condition];
            collect_expr_reads(*condition, expressions, &mut reads);
            // Conservatively include reads/writes from both branches.
            for child in accept.iter().chain(reject.iter()) {
                let (_, child_reads, _) = classify_statement(child, expressions);
                for r in child_reads {
                    if !reads.contains(&r) {
                        reads.push(r);
                    }
                }
            }
            let mut writes = Vec::new();
            for child in accept.iter().chain(reject.iter()) {
                let (_, _, child_writes) = classify_statement(child, expressions);
                for w in child_writes {
                    if !writes.contains(&w) {
                        writes.push(w);
                    }
                }
            }
            (DfgNodeKind::If, reads, writes)
        }
        Statement::Loop {
            body,
            continuing,
            break_if,
        } => {
            let mut reads = Vec::new();
            if let Some(brk) = break_if {
                reads.push(*brk);
                collect_expr_reads(*brk, expressions, &mut reads);
            }
            for child in body.iter().chain(continuing.iter()) {
                let (_, child_reads, _) = classify_statement(child, expressions);
                for r in child_reads {
                    if !reads.contains(&r) {
                        reads.push(r);
                    }
                }
            }
            let mut writes = Vec::new();
            for child in body.iter().chain(continuing.iter()) {
                let (_, _, child_writes) = classify_statement(child, expressions);
                for w in child_writes {
                    if !writes.contains(&w) {
                        writes.push(w);
                    }
                }
            }
            (DfgNodeKind::Loop, reads, writes)
        }
        Statement::Return { value } => {
            let mut reads = Vec::new();
            if let Some(v) = value {
                reads.push(*v);
                collect_expr_reads(*v, expressions, &mut reads);
            }
            (DfgNodeKind::Return, reads, Vec::new())
        }
        Statement::Break => (DfgNodeKind::Break, Vec::new(), Vec::new()),
        Statement::Continue => (DfgNodeKind::Continue, Vec::new(), Vec::new()),
    }
}

/// Recursively collect expression handles that are read by an expression.
fn collect_expr_reads(
    handle: Handle<Expression>,
    expressions: &nxpu_ir::Arena<Expression>,
    reads: &mut Vec<Handle<Expression>>,
) {
    if let Some(expr) = expressions.try_get(handle) {
        for operand in expression_operands(expr) {
            if !reads.contains(&operand) {
                reads.push(operand);
                collect_expr_reads(operand, expressions, reads);
            }
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

impl fmt::Display for DataflowGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "DataflowGraph ({} nodes, {} edges):",
            self.nodes.len(),
            self.edges.len()
        )?;
        for node in &self.nodes {
            writeln!(
                f,
                "  node[{}]: {} (reads: {}, writes: {})",
                node.id,
                node.kind,
                node.reads.len(),
                node.writes.len()
            )?;
        }
        for edge in &self.edges {
            writeln!(f, "  edge: {} -> {} ({})", edge.from, edge.to, edge.kind)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nxpu_ir::{Arena, BinaryOp, Expression, Function, Handle, Literal, Range, Statement};

    // Helper: build a simple function with two stores to different locations
    // that share a common value (creating a RAW dependency).
    fn make_store_chain() -> Function {
        let mut func = Function::new("test");

        // Expressions: gv0_ptr, gv1_ptr, literal, load(gv0_ptr), binary(load, lit)
        let gv0 = {
            let mut arena = nxpu_ir::Arena::new();
            arena.append(nxpu_ir::GlobalVariable {
                name: Some("a".into()),
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
        };
        let gv1 = {
            let mut arena = nxpu_ir::Arena::new();
            let _ = arena.append(nxpu_ir::GlobalVariable {
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
            });
            arena.append(nxpu_ir::GlobalVariable {
                name: Some("b".into()),
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
        };

        let ptr_a = func.expressions.append(Expression::GlobalVariable(gv0));
        let ptr_b = func.expressions.append(Expression::GlobalVariable(gv1));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(42.0)));

        // Store val to ptr_a
        func.body.push(Statement::Store {
            pointer: ptr_a,
            value: val,
        });

        // Store val to ptr_b (no dependency on first store since different pointers)
        func.body.push(Statement::Store {
            pointer: ptr_b,
            value: val,
        });

        func
    }

    #[test]
    fn build_dfg_empty_function() {
        let func = Function::new("empty");
        let dfg = DataflowGraph::build(&func);
        assert_eq!(dfg.node_count(), 0);
        assert_eq!(dfg.edge_count(), 0);
    }

    #[test]
    fn build_dfg_simple_stores() {
        let func = make_store_chain();
        let dfg = DataflowGraph::build(&func);
        assert_eq!(dfg.node_count(), 2);
        // Both stores read `val`, but write to different pointers.
        // The shared read of `val` means no WAR/WAW between them
        // unless the expressions overlap. Since val is read by both,
        // and neither writes val, there is no dependency beyond shared reads.
        assert!(dfg.nodes()[0].kind == DfgNodeKind::Store);
        assert!(dfg.nodes()[1].kind == DfgNodeKind::Store);
    }

    #[test]
    fn build_dfg_raw_dependency() {
        // Create: emit expr -> store -> load -> store
        // This creates a RAW dependency.
        let mut func = Function::new("test");

        let gv = {
            let mut arena = nxpu_ir::Arena::new();
            arena.append(nxpu_ir::GlobalVariable {
                name: Some("x".into()),
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
        };

        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));

        // Statement 0: store val -> ptr
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });
        // Statement 1: store val -> ptr (WAW on ptr)
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });

        let dfg = DataflowGraph::build(&func);
        assert_eq!(dfg.node_count(), 2);

        // Should have WAW edge (both write to ptr).
        let waw_edges: Vec<_> = dfg
            .edges()
            .iter()
            .filter(|e| e.kind == DependencyKind::OutputDependency)
            .collect();
        assert!(
            !waw_edges.is_empty(),
            "expected WAW dependency for two stores to same pointer"
        );
    }

    #[test]
    fn build_dfg_war_dependency() {
        // Create a read-then-write pattern.
        let mut func = Function::new("test");

        let gv = {
            let mut arena = nxpu_ir::Arena::new();
            arena.append(nxpu_ir::GlobalVariable {
                name: Some("x".into()),
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
        };

        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));

        // Statement 0: return (reads ptr via load expression contained within)
        func.body.push(Statement::Return { value: Some(ptr) });
        // Statement 1: store to ptr (writes ptr)
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });

        let dfg = DataflowGraph::build(&func);
        assert_eq!(dfg.node_count(), 2);

        // Should have WAR edge (node 0 reads ptr, node 1 writes ptr).
        let war_edges: Vec<_> = dfg
            .edges()
            .iter()
            .filter(|e| e.kind == DependencyKind::AntiDependency)
            .collect();
        assert!(!war_edges.is_empty(), "expected WAR dependency");
    }

    #[test]
    fn topological_sort_linear_chain() {
        // Three nodes: 0 -> 1 -> 2
        let mut func = Function::new("test");

        let gv = {
            let mut arena = nxpu_ir::Arena::new();
            arena.append(nxpu_ir::GlobalVariable {
                name: Some("x".into()),
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
        };

        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));

        // Three stores to the same pointer (creating WAW chain).
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

        let dfg = DataflowGraph::build(&func);
        let order = dfg.topological_sort().unwrap();
        assert_eq!(order.len(), 3);
        // Must be in order due to WAW dependencies.
        assert_eq!(order[0], 0);
        assert_eq!(order[1], 1);
        assert_eq!(order[2], 2);
    }

    #[test]
    fn topological_sort_empty() {
        let func = Function::new("empty");
        let dfg = DataflowGraph::build(&func);
        let order = dfg.topological_sort().unwrap();
        assert!(order.is_empty());
    }

    #[test]
    fn topological_sort_independent_nodes() {
        let mut func = Function::new("test");

        // Two independent stores to different pointers with different values.
        let gv0 = {
            let mut arena = nxpu_ir::Arena::new();
            arena.append(nxpu_ir::GlobalVariable {
                name: Some("a".into()),
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
                name: Some("b".into()),
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

        let dfg = DataflowGraph::build(&func);
        let order = dfg.topological_sort().unwrap();
        assert_eq!(order.len(), 2);
        // Both are valid orderings since they are independent.
    }

    #[test]
    fn critical_path_linear_chain() {
        let mut func = Function::new("test");

        let gv = {
            let mut arena = nxpu_ir::Arena::new();
            arena.append(nxpu_ir::GlobalVariable {
                name: Some("x".into()),
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
        };

        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));

        // Three stores to the same pointer.
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

        let dfg = DataflowGraph::build(&func);
        let cp = dfg.critical_path();
        // With unit cost and 3 chained nodes, the critical path length is 3.
        assert_eq!(cp.critical_path_length, 3);
        assert_eq!(cp.critical_path.len(), 3);
    }

    #[test]
    fn critical_path_empty() {
        let func = Function::new("empty");
        let dfg = DataflowGraph::build(&func);
        let cp = dfg.critical_path();
        assert_eq!(cp.critical_path_length, 0);
        assert!(cp.critical_path.is_empty());
    }

    #[test]
    fn barrier_creates_control_edges() {
        let mut func = Function::new("test");

        let gv = {
            let mut arena = nxpu_ir::Arena::new();
            arena.append(nxpu_ir::GlobalVariable {
                name: Some("x".into()),
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
        };

        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));

        // Statement 0: store
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });
        // Statement 1: barrier
        func.body
            .push(Statement::Barrier(nxpu_ir::Barrier::STORAGE));
        // Statement 2: store
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });

        let dfg = DataflowGraph::build(&func);
        assert_eq!(dfg.node_count(), 3);

        // Barrier should have control edge to subsequent memory op.
        let control_edges: Vec<_> = dfg
            .edges()
            .iter()
            .filter(|e| e.kind == DependencyKind::Control)
            .collect();
        assert!(
            !control_edges.is_empty(),
            "expected control dependency from barrier"
        );
        // Barrier (node 1) -> store (node 2).
        assert!(control_edges.iter().any(|e| e.from == 1 && e.to == 2));
    }

    #[test]
    fn parallel_groups_independent_ops() {
        let mut func = Function::new("test");

        let gv0 = {
            let mut arena = nxpu_ir::Arena::new();
            arena.append(nxpu_ir::GlobalVariable {
                name: Some("a".into()),
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
                name: Some("b".into()),
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

        let dfg = DataflowGraph::build(&func);
        let groups = dfg.parallel_groups();

        // If they are independent, they should be in the same group.
        // Check that at least one group has 2 elements (parallel).
        let max_group_size = groups.iter().map(|g| g.len()).max().unwrap_or(0);
        assert!(
            max_group_size >= 2,
            "expected independent ops to be grouped together, got groups: {groups:?}"
        );
    }

    #[test]
    fn display_dfg() {
        let func = Function::new("empty");
        let dfg = DataflowGraph::build(&func);
        let s = format!("{dfg}");
        assert!(s.contains("DataflowGraph"));
    }

    #[test]
    fn critical_path_with_custom_costs() {
        let mut func = Function::new("test");

        let gv = {
            let mut arena = nxpu_ir::Arena::new();
            arena.append(nxpu_ir::GlobalVariable {
                name: Some("x".into()),
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
        };

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
        let cp = dfg.critical_path_with_costs(&[5, 3]);
        // With costs [5, 3] and chain 0->1, critical path = 5 + 3 = 8.
        assert_eq!(cp.critical_path_length, 8);
    }
}
