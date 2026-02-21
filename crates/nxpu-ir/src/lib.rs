//! NxPU intermediate representation.

/// A compiled NxPU IR module.
#[derive(Debug, Clone)]
pub struct Module {
    pub functions: Vec<Function>,
}

/// An IR function.
#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub body: Vec<Instruction>,
}

/// An IR instruction.
#[derive(Debug, Clone)]
pub enum Instruction {
    // TODO: define IR instruction set
}
