//! Error types for the NxPU IR.

/// Errors that can occur when constructing or validating IR.
#[derive(Debug, thiserror::Error)]
pub enum IrError {
    /// A handle index is out of bounds for its arena.
    #[error("handle index {index} out of bounds (arena size: {size})")]
    BadHandle { index: usize, size: usize },

    /// A type mismatch was detected.
    #[error("type mismatch: expected {expected}, found {found}")]
    TypeMismatch { expected: String, found: String },

    /// An invalid workgroup size was specified.
    #[error("invalid workgroup size: [{}, {}, {}]", .0[0], .0[1], .0[2])]
    InvalidWorkgroupSize([u32; 3]),
}
