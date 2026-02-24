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
    use nxpu_ir::{Expression, Function, Literal, Statement};

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

    // --- Helper to make a dummy global variable handle ---
    fn make_gv(index: usize) -> Handle<nxpu_ir::GlobalVariable> {
        let mut arena = nxpu_ir::Arena::new();
        // Append `index + 1` dummy variables so we get the handle at the desired index.
        let mut handle = None;
        for i in 0..=index {
            let h = arena.append(nxpu_ir::GlobalVariable {
                name: Some(format!("gv_{i}")),
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
            handle = Some(h);
        }
        handle.unwrap()
    }

    // ===== Display tests =====

    #[test]
    fn display_dfg_with_nodes_and_edges() {
        let mut func = Function::new("test");
        let gv = make_gv(0);
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));

        // Two stores to the same pointer -> WAW edge.
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });

        let dfg = DataflowGraph::build(&func);
        let s = format!("{dfg}");
        assert!(s.contains("DataflowGraph (2 nodes, "));
        assert!(s.contains("node[0]: Store"));
        assert!(s.contains("node[1]: Store"));
        assert!(s.contains("edge:"));
        assert!(s.contains("WAW"));
    }

    #[test]
    fn dfg_node_kind_display_all_variants() {
        assert_eq!(format!("{}", DfgNodeKind::Store), "Store");
        assert_eq!(format!("{}", DfgNodeKind::Call), "Call");
        assert_eq!(format!("{}", DfgNodeKind::Atomic), "Atomic");
        assert_eq!(format!("{}", DfgNodeKind::Barrier), "Barrier");
        assert_eq!(format!("{}", DfgNodeKind::Emit), "Emit");
        assert_eq!(format!("{}", DfgNodeKind::If), "If");
        assert_eq!(format!("{}", DfgNodeKind::Loop), "Loop");
        assert_eq!(format!("{}", DfgNodeKind::Return), "Return");
        assert_eq!(format!("{}", DfgNodeKind::Break), "Break");
        assert_eq!(format!("{}", DfgNodeKind::Continue), "Continue");
    }

    #[test]
    fn dependency_kind_display_all_variants() {
        assert_eq!(format!("{}", DependencyKind::DataFlow), "RAW");
        assert_eq!(format!("{}", DependencyKind::AntiDependency), "WAR");
        assert_eq!(format!("{}", DependencyKind::OutputDependency), "WAW");
        assert_eq!(format!("{}", DependencyKind::Control), "Control");
    }

    #[test]
    fn dataflow_error_display() {
        let err = DataflowError::CycleDetected {
            visited: 3,
            total: 5,
        };
        let msg = format!("{err}");
        assert!(msg.contains("cycle detected"));
        assert!(msg.contains("3 of 5"));
    }

    // ===== Successors / Predecessors =====

    #[test]
    fn successors_and_predecessors() {
        let mut func = Function::new("test");
        let gv = make_gv(0);
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));

        // Three stores to same pointer: 0->1->2 (WAW chain).
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

        // Node 0 should have successors to 1 and 2.
        let succs_0 = dfg.successors(0);
        assert!(!succs_0.is_empty());

        // Node 2 should have predecessors from 0 and/or 1.
        let preds_2 = dfg.predecessors(2);
        assert!(!preds_2.is_empty());

        // Node 0 should have no predecessors.
        let preds_0 = dfg.predecessors(0);
        assert!(preds_0.is_empty());
    }

    // ===== Break / Continue statement classification =====

    #[test]
    fn break_and_continue_nodes() {
        let mut func = Function::new("test");
        func.body.push(Statement::Break);
        func.body.push(Statement::Continue);

        let dfg = DataflowGraph::build(&func);
        assert_eq!(dfg.node_count(), 2);
        assert_eq!(dfg.nodes()[0].kind, DfgNodeKind::Break);
        assert_eq!(dfg.nodes()[1].kind, DfgNodeKind::Continue);
        // Break and Continue have no reads/writes so no edges.
        assert_eq!(dfg.edge_count(), 0);
    }

    // ===== Return with no value =====

    #[test]
    fn return_no_value() {
        let mut func = Function::new("test");
        func.body.push(Statement::Return { value: None });

        let dfg = DataflowGraph::build(&func);
        assert_eq!(dfg.node_count(), 1);
        assert_eq!(dfg.nodes()[0].kind, DfgNodeKind::Return);
        assert!(dfg.nodes()[0].reads.is_empty());
        assert!(dfg.nodes()[0].writes.is_empty());
    }

    // ===== Return with value =====

    #[test]
    fn return_with_value() {
        let mut func = Function::new("test");
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(42.0)));
        func.body.push(Statement::Return { value: Some(val) });

        let dfg = DataflowGraph::build(&func);
        assert_eq!(dfg.node_count(), 1);
        assert_eq!(dfg.nodes()[0].kind, DfgNodeKind::Return);
        assert!(dfg.nodes()[0].reads.contains(&val));
        assert!(dfg.nodes()[0].writes.is_empty());
    }

    // ===== If statement classification =====

    #[test]
    fn if_statement_classification() {
        let mut func = Function::new("test");
        let gv = make_gv(0);
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let cond = func
            .expressions
            .append(Expression::Literal(Literal::Bool(true)));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));

        // If statement with a store in the accept branch.
        func.body.push(Statement::If {
            condition: cond,
            accept: vec![Statement::Store {
                pointer: ptr,
                value: val,
            }],
            reject: vec![],
        });

        let dfg = DataflowGraph::build(&func);
        assert_eq!(dfg.node_count(), 1);
        assert_eq!(dfg.nodes()[0].kind, DfgNodeKind::If);
        // The If node should read the condition and the value from the child store.
        assert!(dfg.nodes()[0].reads.contains(&cond));
        // The If node should write the pointer from the child store.
        assert!(dfg.nodes()[0].writes.contains(&ptr));
    }

    // ===== If statement with reject branch =====

    #[test]
    fn if_statement_with_reject_branch() {
        let mut func = Function::new("test");
        let gv0 = make_gv(0);
        let gv1 = make_gv(1);
        let ptr_a = func.expressions.append(Expression::GlobalVariable(gv0));
        let ptr_b = func.expressions.append(Expression::GlobalVariable(gv1));
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

        let dfg = DataflowGraph::build(&func);
        assert_eq!(dfg.node_count(), 1);
        assert_eq!(dfg.nodes()[0].kind, DfgNodeKind::If);
        // Should include writes from both branches.
        assert!(dfg.nodes()[0].writes.contains(&ptr_a));
        assert!(dfg.nodes()[0].writes.contains(&ptr_b));
    }

    // ===== Loop statement classification =====

    #[test]
    fn loop_statement_classification() {
        let mut func = Function::new("test");
        let gv = make_gv(0);
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        let break_cond = func
            .expressions
            .append(Expression::Literal(Literal::Bool(false)));

        func.body.push(Statement::Loop {
            body: vec![Statement::Store {
                pointer: ptr,
                value: val,
            }],
            continuing: vec![],
            break_if: Some(break_cond),
        });

        let dfg = DataflowGraph::build(&func);
        assert_eq!(dfg.node_count(), 1);
        assert_eq!(dfg.nodes()[0].kind, DfgNodeKind::Loop);
        // Should read break_if condition.
        assert!(dfg.nodes()[0].reads.contains(&break_cond));
        // Should write via the child store.
        assert!(dfg.nodes()[0].writes.contains(&ptr));
    }

    // ===== Loop with continuing block =====

    #[test]
    fn loop_with_continuing_block() {
        let mut func = Function::new("test");
        let gv0 = make_gv(0);
        let gv1 = make_gv(1);
        let ptr_body = func.expressions.append(Expression::GlobalVariable(gv0));
        let ptr_cont = func.expressions.append(Expression::GlobalVariable(gv1));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));

        func.body.push(Statement::Loop {
            body: vec![Statement::Store {
                pointer: ptr_body,
                value: val,
            }],
            continuing: vec![Statement::Store {
                pointer: ptr_cont,
                value: val,
            }],
            break_if: None,
        });

        let dfg = DataflowGraph::build(&func);
        assert_eq!(dfg.node_count(), 1);
        assert_eq!(dfg.nodes()[0].kind, DfgNodeKind::Loop);
        // Should include writes from both body and continuing blocks.
        assert!(dfg.nodes()[0].writes.contains(&ptr_body));
        assert!(dfg.nodes()[0].writes.contains(&ptr_cont));
    }

    // ===== Call statement classification =====

    #[test]
    fn call_statement_classification() {
        let mut func = Function::new("test");
        let arg_val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        let result_val = func
            .expressions
            .append(Expression::Literal(Literal::F32(0.0)));

        // Build a dummy function handle.
        let callee_handle = {
            let mut funcs = nxpu_ir::Arena::new();
            funcs.append(Function::new("callee"))
        };

        func.body.push(Statement::Call {
            function: callee_handle,
            arguments: vec![arg_val],
            result: Some(result_val),
        });

        let dfg = DataflowGraph::build(&func);
        assert_eq!(dfg.node_count(), 1);
        assert_eq!(dfg.nodes()[0].kind, DfgNodeKind::Call);
        assert!(dfg.nodes()[0].reads.contains(&arg_val));
        assert!(dfg.nodes()[0].writes.contains(&result_val));
    }

    // ===== Call with no result =====

    #[test]
    fn call_no_result() {
        let mut func = Function::new("test");
        let arg_val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));

        let callee_handle = {
            let mut funcs = nxpu_ir::Arena::new();
            funcs.append(Function::new("callee"))
        };

        func.body.push(Statement::Call {
            function: callee_handle,
            arguments: vec![arg_val],
            result: None,
        });

        let dfg = DataflowGraph::build(&func);
        assert_eq!(dfg.node_count(), 1);
        assert_eq!(dfg.nodes()[0].kind, DfgNodeKind::Call);
        assert!(dfg.nodes()[0].writes.is_empty());
    }

    // ===== Atomic statement classification =====

    #[test]
    fn atomic_statement_classification() {
        let mut func = Function::new("test");
        let gv = make_gv(0);
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::U32(1)));
        let result_expr = func
            .expressions
            .append(Expression::Literal(Literal::U32(0)));

        func.body.push(Statement::Atomic {
            pointer: ptr,
            fun: nxpu_ir::AtomicFunction::Add,
            value: val,
            result: Some(result_expr),
        });

        let dfg = DataflowGraph::build(&func);
        assert_eq!(dfg.node_count(), 1);
        assert_eq!(dfg.nodes()[0].kind, DfgNodeKind::Atomic);
        assert!(dfg.nodes()[0].reads.contains(&ptr));
        assert!(dfg.nodes()[0].reads.contains(&val));
        assert!(dfg.nodes()[0].writes.contains(&ptr));
        assert!(dfg.nodes()[0].writes.contains(&result_expr));
    }

    // ===== Atomic with Exchange + compare =====

    #[test]
    fn atomic_exchange_with_compare() {
        let mut func = Function::new("test");
        let gv = make_gv(0);
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::U32(1)));
        let cmp = func
            .expressions
            .append(Expression::Literal(Literal::U32(0)));

        func.body.push(Statement::Atomic {
            pointer: ptr,
            fun: nxpu_ir::AtomicFunction::Exchange { compare: Some(cmp) },
            value: val,
            result: None,
        });

        let dfg = DataflowGraph::build(&func);
        assert_eq!(dfg.node_count(), 1);
        assert_eq!(dfg.nodes()[0].kind, DfgNodeKind::Atomic);
        // Compare should appear in reads.
        assert!(dfg.nodes()[0].reads.contains(&cmp));
    }

    // ===== Atomic with no result =====

    #[test]
    fn atomic_no_result() {
        let mut func = Function::new("test");
        let gv = make_gv(0);
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::U32(1)));

        func.body.push(Statement::Atomic {
            pointer: ptr,
            fun: nxpu_ir::AtomicFunction::Add,
            value: val,
            result: None,
        });

        let dfg = DataflowGraph::build(&func);
        assert_eq!(dfg.nodes()[0].writes.len(), 1);
        assert!(dfg.nodes()[0].writes.contains(&ptr));
    }

    // ===== Emit statement classification =====

    #[test]
    fn emit_statement_classification() {
        let mut func = Function::new("test");

        // Create some expressions to emit.
        let lit_a = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        let lit_b = func
            .expressions
            .append(Expression::Literal(Literal::F32(2.0)));
        let add = func.expressions.append(Expression::Binary {
            op: nxpu_ir::BinaryOp::Add,
            left: lit_a,
            right: lit_b,
        });

        // Emit range covering all three expressions.
        let range = nxpu_ir::Range::from_index_range(0..3);
        func.body.push(Statement::Emit(range));

        let dfg = DataflowGraph::build(&func);
        assert_eq!(dfg.node_count(), 1);
        assert_eq!(dfg.nodes()[0].kind, DfgNodeKind::Emit);
        // The emit should write all 3 expressions.
        assert!(dfg.nodes()[0].writes.contains(&lit_a));
        assert!(dfg.nodes()[0].writes.contains(&lit_b));
        assert!(dfg.nodes()[0].writes.contains(&add));
    }

    // ===== Barrier statement classification =====

    #[test]
    fn barrier_statement_no_reads_writes() {
        let mut func = Function::new("test");
        func.body
            .push(Statement::Barrier(nxpu_ir::Barrier::STORAGE));

        let dfg = DataflowGraph::build(&func);
        assert_eq!(dfg.node_count(), 1);
        assert_eq!(dfg.nodes()[0].kind, DfgNodeKind::Barrier);
        assert!(dfg.nodes()[0].reads.is_empty());
        assert!(dfg.nodes()[0].writes.is_empty());
    }

    // ===== Barrier to barrier control edge =====

    #[test]
    fn barrier_to_barrier_control_edge() {
        let mut func = Function::new("test");
        func.body
            .push(Statement::Barrier(nxpu_ir::Barrier::STORAGE));
        func.body
            .push(Statement::Barrier(nxpu_ir::Barrier::WORKGROUP));

        let dfg = DataflowGraph::build(&func);
        assert_eq!(dfg.node_count(), 2);
        // First barrier should have a control edge to the second barrier.
        let control_edges: Vec<_> = dfg
            .edges()
            .iter()
            .filter(|e| e.kind == DependencyKind::Control)
            .collect();
        assert!(
            control_edges.iter().any(|e| e.from == 0 && e.to == 1),
            "expected control edge from barrier[0] to barrier[1]"
        );
    }

    // ===== Barrier does not create control edge to non-memory op =====

    #[test]
    fn barrier_no_control_edge_to_break() {
        let mut func = Function::new("test");
        func.body
            .push(Statement::Barrier(nxpu_ir::Barrier::STORAGE));
        func.body.push(Statement::Break);

        let dfg = DataflowGraph::build(&func);
        assert_eq!(dfg.node_count(), 2);
        // Break has no reads/writes and is not a barrier, so no control edge.
        let control_edges: Vec<_> = dfg
            .edges()
            .iter()
            .filter(|e| e.kind == DependencyKind::Control)
            .collect();
        assert!(
            control_edges.is_empty(),
            "should not have control edge from barrier to break"
        );
    }

    // ===== RAW dependency (explicit read-after-write) =====

    #[test]
    fn explicit_raw_dependency() {
        let mut func = Function::new("test");
        let gv = make_gv(0);
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        let load = func.expressions.append(Expression::Load { pointer: ptr });

        // Statement 0: store val -> ptr (writes ptr)
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });
        // Statement 1: return load(ptr) (reads ptr via load)
        func.body.push(Statement::Return { value: Some(load) });

        let dfg = DataflowGraph::build(&func);
        assert_eq!(dfg.node_count(), 2);

        // Should have RAW: node 0 writes ptr, node 1 reads ptr (via load's pointer).
        let raw_edges: Vec<_> = dfg
            .edges()
            .iter()
            .filter(|e| e.kind == DependencyKind::DataFlow)
            .collect();
        assert!(
            !raw_edges.is_empty(),
            "expected RAW dependency: store then read"
        );
    }

    // ===== Combined WAR + WAW + RAW =====

    #[test]
    fn combined_war_waw_raw_dependencies() {
        let mut func = Function::new("test");
        let gv = make_gv(0);
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        let load = func.expressions.append(Expression::Load { pointer: ptr });

        // Statement 0: store val -> ptr (writes ptr, reads val)
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });
        // Statement 1: store load(ptr) -> ptr (reads ptr via load, writes ptr)
        //   RAW: node 0 writes ptr, node 1 reads ptr (via load)
        //   WAW: both write ptr
        //   WAR: node 0 reads val (no WAR here since node 1 doesn't write val)
        func.body.push(Statement::Store {
            pointer: ptr,
            value: load,
        });

        let dfg = DataflowGraph::build(&func);
        assert_eq!(dfg.node_count(), 2);

        let has_raw = dfg
            .edges()
            .iter()
            .any(|e| e.kind == DependencyKind::DataFlow);
        let has_waw = dfg
            .edges()
            .iter()
            .any(|e| e.kind == DependencyKind::OutputDependency);
        assert!(has_raw, "expected RAW dependency");
        assert!(has_waw, "expected WAW dependency");
    }

    // ===== Parallel groups with a dependency chain =====

    #[test]
    fn parallel_groups_with_chain() {
        let mut func = Function::new("test");
        let gv = make_gv(0);
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));

        // Three stores to same pointer: all dependent.
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
        let groups = dfg.parallel_groups();

        // All dependent: each should be in its own group.
        assert_eq!(groups.len(), 3, "expected 3 sequential groups");
        for g in &groups {
            assert_eq!(g.len(), 1, "each group should have exactly 1 node");
        }
    }

    // ===== Parallel groups empty graph =====

    #[test]
    fn parallel_groups_empty() {
        let func = Function::new("empty");
        let dfg = DataflowGraph::build(&func);
        let groups = dfg.parallel_groups();
        assert!(groups.is_empty());
    }

    // ===== Critical path multi-path diamond graph =====

    #[test]
    fn critical_path_diamond_graph() {
        // Build a diamond: node 0 -> node 1 (independent), node 0 -> node 2,
        // node 1 -> node 3, node 2 -> node 3.
        // We simulate this with stores that create the right dependency pattern.
        let mut func = Function::new("test");
        let gv0 = make_gv(0);
        let gv1 = make_gv(1);

        let ptr_a = func.expressions.append(Expression::GlobalVariable(gv0));
        let ptr_b = func.expressions.append(Expression::GlobalVariable(gv1));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        let load_a = func.expressions.append(Expression::Load { pointer: ptr_a });
        let load_b = func.expressions.append(Expression::Load { pointer: ptr_b });

        // stmt 0: store val -> ptr_a (writes a)
        func.body.push(Statement::Store {
            pointer: ptr_a,
            value: val,
        });
        // stmt 1: store val -> ptr_b (writes b, independent of stmt 0)
        func.body.push(Statement::Store {
            pointer: ptr_b,
            value: val,
        });
        // stmt 2: store load_a -> ptr_a (reads a -> RAW from 0, writes a -> WAW with 0)
        func.body.push(Statement::Store {
            pointer: ptr_a,
            value: load_a,
        });
        // stmt 3: store load_b -> ptr_b (reads b -> RAW from 1, writes b -> WAW with 1)
        func.body.push(Statement::Store {
            pointer: ptr_b,
            value: load_b,
        });

        let dfg = DataflowGraph::build(&func);
        let cp = dfg.critical_path();

        // Two parallel chains of length 2. Critical path = 2.
        assert_eq!(cp.critical_path_length, 2);
        assert_eq!(cp.critical_path.len(), 2);
        assert_eq!(cp.node_distances.len(), 4);
    }

    // ===== Critical path result node distances =====

    #[test]
    fn critical_path_node_distances() {
        let mut func = Function::new("test");
        let gv = make_gv(0);
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));

        // Three stores: 0 -> 1 -> 2 (WAW chain).
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
        assert_eq!(cp.node_distances.len(), 3);
        assert_eq!(cp.node_distances[0], 1);
        assert_eq!(cp.node_distances[1], 2);
        assert_eq!(cp.node_distances[2], 3);
        assert_eq!(cp.critical_path, vec![0, 1, 2]);
    }

    // ===== Expression operand coverage: Access, AccessIndex, Select, Swizzle =====

    #[test]
    fn expression_operands_access_and_select() {
        let mut func = Function::new("test");
        let gv = make_gv(0);
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let idx = func
            .expressions
            .append(Expression::Literal(Literal::U32(0)));
        let access = func.expressions.append(Expression::Access {
            base: ptr,
            index: idx,
        });
        let val_a = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        let val_b = func
            .expressions
            .append(Expression::Literal(Literal::F32(2.0)));
        let cond = func
            .expressions
            .append(Expression::Literal(Literal::Bool(true)));
        let select = func.expressions.append(Expression::Select {
            condition: cond,
            accept: val_a,
            reject: val_b,
        });

        // Emit a range covering all expressions.
        let range = nxpu_ir::Range::from_index_range(0..func.expressions.len() as u32);
        func.body.push(Statement::Emit(range));

        let dfg = DataflowGraph::build(&func);
        assert_eq!(dfg.node_count(), 1);
        assert_eq!(dfg.nodes()[0].kind, DfgNodeKind::Emit);
        // All expressions should appear in writes.
        assert!(dfg.nodes()[0].writes.contains(&access));
        assert!(dfg.nodes()[0].writes.contains(&select));
    }

    // ===== Expression operands: Splat, As, ArrayLength, Compose, Math =====

    #[test]
    fn expression_operands_splat_as_arraylength_compose_math() {
        let mut func = Function::new("test");
        let lit = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        let splat = func.expressions.append(Expression::Splat {
            size: nxpu_ir::VectorSize::Quad,
            value: lit,
        });
        let cast = func.expressions.append(Expression::As {
            expr: lit,
            kind: nxpu_ir::ScalarKind::Uint,
            convert: Some(4),
        });
        let arr_len = func.expressions.append(Expression::ArrayLength(lit));

        let ty_handle = {
            let mut types = nxpu_ir::UniqueArena::new();
            types.insert(nxpu_ir::Type {
                name: None,
                inner: nxpu_ir::TypeInner::Vector {
                    size: nxpu_ir::VectorSize::Quad,
                    scalar: nxpu_ir::Scalar::F32,
                },
            })
        };
        let compose = func.expressions.append(Expression::Compose {
            ty: ty_handle,
            components: vec![lit],
        });

        let lit2 = func
            .expressions
            .append(Expression::Literal(Literal::F32(2.0)));
        let lit3 = func
            .expressions
            .append(Expression::Literal(Literal::F32(3.0)));
        let math = func.expressions.append(Expression::Math {
            fun: nxpu_ir::MathFunction::Clamp,
            arg: lit,
            arg1: Some(lit2),
            arg2: Some(lit3),
            arg3: None,
        });

        let range = nxpu_ir::Range::from_index_range(0..func.expressions.len() as u32);
        func.body.push(Statement::Emit(range));

        let dfg = DataflowGraph::build(&func);
        assert_eq!(dfg.nodes()[0].kind, DfgNodeKind::Emit);
        assert!(dfg.nodes()[0].writes.contains(&splat));
        assert!(dfg.nodes()[0].writes.contains(&cast));
        assert!(dfg.nodes()[0].writes.contains(&arr_len));
        assert!(dfg.nodes()[0].writes.contains(&compose));
        assert!(dfg.nodes()[0].writes.contains(&math));
    }

    // ===== Expression operands: Swizzle, Unary, AccessIndex =====

    #[test]
    fn expression_operands_swizzle_unary_accessindex() {
        let mut func = Function::new("test");
        let lit = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        let neg = func.expressions.append(Expression::Unary {
            op: nxpu_ir::UnaryOp::Negate,
            expr: lit,
        });
        let gv = make_gv(0);
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let access_idx = func.expressions.append(Expression::AccessIndex {
            base: ptr,
            index: 0,
        });

        let vec_lit = func
            .expressions
            .append(Expression::Literal(Literal::F32(0.0)));
        let swizzle = func.expressions.append(Expression::Swizzle {
            size: nxpu_ir::VectorSize::Bi,
            vector: vec_lit,
            pattern: [
                nxpu_ir::SwizzleComponent::X,
                nxpu_ir::SwizzleComponent::Y,
                nxpu_ir::SwizzleComponent::Z,
                nxpu_ir::SwizzleComponent::W,
            ],
        });

        let range = nxpu_ir::Range::from_index_range(0..func.expressions.len() as u32);
        func.body.push(Statement::Emit(range));

        let dfg = DataflowGraph::build(&func);
        assert!(dfg.nodes()[0].writes.contains(&neg));
        assert!(dfg.nodes()[0].writes.contains(&access_idx));
        assert!(dfg.nodes()[0].writes.contains(&swizzle));
    }

    // ===== Expression operands: leaf expressions (no operands) =====

    #[test]
    fn expression_operands_leaf_nodes() {
        let mut func = Function::new("test");
        let _func_arg = func.expressions.append(Expression::FunctionArgument(0));
        let gv = make_gv(0);
        let _gv_expr = func.expressions.append(Expression::GlobalVariable(gv));

        let callee_handle = {
            let mut funcs = nxpu_ir::Arena::new();
            funcs.append(Function::new("callee"))
        };
        let _call_result = func
            .expressions
            .append(Expression::CallResult(callee_handle));

        let ty_handle = {
            let mut types = nxpu_ir::UniqueArena::new();
            types.insert(nxpu_ir::Type {
                name: None,
                inner: nxpu_ir::TypeInner::Scalar(nxpu_ir::Scalar::U32),
            })
        };
        let _atomic_result = func.expressions.append(Expression::AtomicResult {
            ty: ty_handle,
            comparison: false,
        });
        let _zero_val = func.expressions.append(Expression::ZeroValue(ty_handle));

        let range = nxpu_ir::Range::from_index_range(0..func.expressions.len() as u32);
        func.body.push(Statement::Emit(range));

        let dfg = DataflowGraph::build(&func);
        assert_eq!(dfg.node_count(), 1);
        // Leaf expressions produce no reads (they have no operands).
        // FunctionArgument, GlobalVariable, CallResult, AtomicResult, ZeroValue.
    }

    // ===== Topological sort with diamond shape =====

    #[test]
    fn topological_sort_diamond_shape() {
        let mut func = Function::new("test");
        let gv0 = make_gv(0);
        let gv1 = make_gv(1);
        let ptr_a = func.expressions.append(Expression::GlobalVariable(gv0));
        let ptr_b = func.expressions.append(Expression::GlobalVariable(gv1));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        let load_a = func.expressions.append(Expression::Load { pointer: ptr_a });

        // stmt 0: store val -> ptr_a (writes a)
        func.body.push(Statement::Store {
            pointer: ptr_a,
            value: val,
        });
        // stmt 1: store val -> ptr_b (writes b, independent)
        func.body.push(Statement::Store {
            pointer: ptr_b,
            value: val,
        });
        // stmt 2: store load(a) -> ptr_b (reads a -> RAW from 0, writes b -> WAW with 1)
        func.body.push(Statement::Store {
            pointer: ptr_b,
            value: load_a,
        });

        let dfg = DataflowGraph::build(&func);
        let order = dfg.topological_sort().unwrap();
        assert_eq!(order.len(), 3);

        // Node 0 must come before node 2 (RAW dependency on ptr_a).
        // Node 1 must come before node 2 (WAW dependency on ptr_b).
        let pos_0 = order.iter().position(|&n| n == 0).unwrap();
        let pos_1 = order.iter().position(|&n| n == 1).unwrap();
        let pos_2 = order.iter().position(|&n| n == 2).unwrap();
        assert!(pos_0 < pos_2);
        assert!(pos_1 < pos_2);
    }

    // ===== Parallel groups with mixed independent and dependent ops =====

    #[test]
    fn parallel_groups_mixed() {
        let mut func = Function::new("test");
        let gv0 = make_gv(0);
        let gv1 = make_gv(1);
        let ptr_a = func.expressions.append(Expression::GlobalVariable(gv0));
        let ptr_b = func.expressions.append(Expression::GlobalVariable(gv1));
        let val_a = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        let val_b = func
            .expressions
            .append(Expression::Literal(Literal::F32(2.0)));

        // Two independent stores (no shared pointers or values).
        func.body.push(Statement::Store {
            pointer: ptr_a,
            value: val_a,
        });
        func.body.push(Statement::Store {
            pointer: ptr_b,
            value: val_b,
        });
        // Third store depends on ptr_a (WAW with stmt 0).
        func.body.push(Statement::Store {
            pointer: ptr_a,
            value: val_b,
        });

        let dfg = DataflowGraph::build(&func);
        let groups = dfg.parallel_groups();

        // First group should have at least 2 independent ops.
        assert!(!groups.is_empty());
        let first_group = &groups[0];
        assert!(
            first_group.len() >= 2,
            "expected first parallel group to contain independent ops"
        );
    }

    // ===== Single node graph =====

    #[test]
    fn single_node_graph() {
        let mut func = Function::new("test");
        func.body.push(Statement::Break);

        let dfg = DataflowGraph::build(&func);
        assert_eq!(dfg.node_count(), 1);
        assert_eq!(dfg.edge_count(), 0);

        let order = dfg.topological_sort().unwrap();
        assert_eq!(order, vec![0]);

        let cp = dfg.critical_path();
        assert_eq!(cp.critical_path_length, 1);
        assert_eq!(cp.critical_path, vec![0]);

        let groups = dfg.parallel_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0], vec![0]);
    }

    // ===== Display with all edge kinds =====

    #[test]
    fn display_all_edge_kinds() {
        let mut func = Function::new("test");
        let gv = make_gv(0);
        let ptr = func.expressions.append(Expression::GlobalVariable(gv));
        let val = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        let load = func.expressions.append(Expression::Load { pointer: ptr });

        // stmt 0: store val -> ptr (writes ptr)
        func.body.push(Statement::Store {
            pointer: ptr,
            value: val,
        });
        // stmt 1: store load(ptr) -> ptr (reads ptr, writes ptr -> RAW + WAW)
        func.body.push(Statement::Store {
            pointer: ptr,
            value: load,
        });

        let dfg = DataflowGraph::build(&func);
        let s = format!("{dfg}");
        assert!(s.contains("RAW") || s.contains("WAW") || s.contains("WAR"));
    }

    // ===== Math expression with all 4 arguments =====

    #[test]
    fn math_expression_four_args() {
        let mut func = Function::new("test");
        let a = func
            .expressions
            .append(Expression::Literal(Literal::F32(1.0)));
        let b = func
            .expressions
            .append(Expression::Literal(Literal::F32(2.0)));
        let c = func
            .expressions
            .append(Expression::Literal(Literal::F32(3.0)));
        let d = func
            .expressions
            .append(Expression::Literal(Literal::F32(4.0)));
        let _math = func.expressions.append(Expression::Math {
            fun: nxpu_ir::MathFunction::Fma,
            arg: a,
            arg1: Some(b),
            arg2: Some(c),
            arg3: Some(d),
        });

        let range = nxpu_ir::Range::from_index_range(0..func.expressions.len() as u32);
        func.body.push(Statement::Emit(range));

        let dfg = DataflowGraph::build(&func);
        // The math expression should read all four argument literals.
        let reads = &dfg.nodes()[0].reads;
        assert!(reads.contains(&a), "math should read arg");
        assert!(reads.contains(&b), "math should read arg1");
        assert!(reads.contains(&c), "math should read arg2");
        assert!(reads.contains(&d), "math should read arg3");
    }
}
