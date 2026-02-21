//! WGSL parser for NxPU.

pub fn parse(_source: &str) -> Result<(), ParseError> {
    todo!()
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("unexpected token at {line}:{col}")]
    UnexpectedToken { line: usize, col: usize },
}
