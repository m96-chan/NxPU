//! TFLite schema constants.
//!
//! Defines tensor types, builtin operator codes, and FlatBuffer vtable
//! field offsets corresponding to the TFLite FlatBuffer schema.

/// TFLite `TensorType` enum values.
#[allow(dead_code)]
pub mod tensor_type {
    pub const FLOAT32: i8 = 0;
    pub const FLOAT16: i8 = 1;
    pub const INT32: i8 = 2;
    pub const BOOL: i8 = 6;
    pub const INT8: i8 = 9;
    pub const UINT32: i8 = 15;
}

/// TFLite `BuiltinOperator` enum values.
pub mod builtin_op {
    pub const ADD: i32 = 0;
    pub const MUL: i32 = 18;
    pub const SUB: i32 = 41;
    pub const DIV: i32 = 42;
    pub const BATCH_MATMUL: i32 = 126;
}

/// VTable field slot offsets for each TFLite FlatBuffer table.
/// Slot = 4 + 2 * field_index.
pub mod vt {
    pub mod model {
        pub const VERSION: u16 = 4;
        pub const OPERATOR_CODES: u16 = 6;
        pub const SUBGRAPHS: u16 = 8;
        pub const DESCRIPTION: u16 = 10;
        pub const BUFFERS: u16 = 12;
    }
    pub mod sub_graph {
        pub const TENSORS: u16 = 4;
        pub const INPUTS: u16 = 6;
        pub const OUTPUTS: u16 = 8;
        pub const OPERATORS: u16 = 10;
        pub const NAME: u16 = 12;
    }
    pub mod tensor {
        pub const SHAPE: u16 = 4;
        pub const TYPE: u16 = 6;
        pub const BUFFER: u16 = 8;
        pub const NAME: u16 = 10;
    }
    pub mod operator {
        pub const OPCODE_INDEX: u16 = 4;
        pub const INPUTS: u16 = 6;
        pub const OUTPUTS: u16 = 8;
    }
    pub mod operator_code {
        pub const DEPRECATED_BUILTIN_CODE: u16 = 4;
        pub const VERSION: u16 = 8;
        pub const BUILTIN_CODE: u16 = 10;
    }
    pub mod buffer {
        // Empty — no data field needed for dynamic tensors.
    }
}
