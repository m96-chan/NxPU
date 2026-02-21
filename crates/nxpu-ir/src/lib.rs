//! NxPU intermediate representation.
//!
//! An arena-based SSA IR for representing compute shader programs.
//! Modeled after naga's architecture but tailored for NPU lowering.

pub mod arena;
mod display;
mod error;
mod expr;
mod func;
mod global;
pub mod graph;
mod stmt;
mod types;

pub use arena::{Arena, Handle, Range, UniqueArena};
pub use display::{dump_module, format_type, format_type_inner};
pub use error::IrError;
pub use expr::{
    AtomicFunction, BinaryOp, Expression, Literal, MathFunction, SwizzleComponent, UnaryOp,
};
pub use func::{EntryPoint, Function, FunctionArgument, FunctionResult, LocalVariable};
pub use global::{AddressSpace, Binding, BuiltIn, GlobalVariable, ResourceBinding, StorageAccess};
pub use stmt::{Barrier, Block, Statement};
pub use types::{
    ArraySize, Bytes, Dimension, MemoryLayout, Scalar, ScalarKind, StructMember, TensorShape, Type,
    TypeInner, VectorSize,
};

/// A compiled NxPU IR module.
#[derive(Clone, Debug, Default)]
pub struct Module {
    /// Deduplicated type arena.
    pub types: UniqueArena<Type>,
    /// Module-scope variables.
    pub global_variables: Arena<GlobalVariable>,
    /// Module-scope constant expressions.
    pub global_expressions: Arena<Expression>,
    /// Helper (non-entry-point) functions.
    pub functions: Arena<Function>,
    /// Compute entry points.
    pub entry_points: Vec<EntryPoint>,
}
