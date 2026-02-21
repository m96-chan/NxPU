//! WGSL parser for NxPU.
//!
//! Parses WGSL source text into an [`nxpu_ir::Module`] by using
//! [naga](https://crates.io/crates/naga)'s WGSL frontend and then
//! lowering the resulting `naga::Module` to NxPU-IR.

mod lower;

/// Parse WGSL source into an NxPU IR module.
///
/// Only `@compute` entry points are retained; vertex/fragment shaders
/// and GPU-only features (images, samplers, derivatives) are rejected.
pub fn parse(source: &str) -> Result<nxpu_ir::Module, ParseError> {
    let naga_module = naga::front::wgsl::parse_str(source)?;
    lower::lower_module(&naga_module)
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error(transparent)]
    Wgsl(#[from] naga::front::wgsl::ParseError),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("lowering: {0}")]
    Lowering(String),
}
