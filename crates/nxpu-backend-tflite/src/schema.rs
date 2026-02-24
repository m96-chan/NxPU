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
#[allow(dead_code)]
pub mod builtin_op {
    pub const ADD: i32 = 0;
    pub const AVERAGE_POOL_2D: i32 = 1;
    pub const CONV_2D: i32 = 3;
    pub const MAX_POOL_2D: i32 = 17;
    pub const MUL: i32 = 18;
    pub const RELU: i32 = 19;
    pub const RESHAPE: i32 = 22;
    pub const SOFTMAX: i32 = 25;
    pub const LOGISTIC: i32 = 14;
    pub const TANH: i32 = 28;
    pub const TRANSPOSE: i32 = 39;
    pub const MEAN: i32 = 40;
    pub const SUB: i32 = 41;
    pub const DIV: i32 = 42;
    pub const REDUCE_MAX: i32 = 82;
    pub const REDUCE_MIN: i32 = 83;
    pub const SUM: i32 = 74;
    pub const CONCATENATION: i32 = 2;
    pub const SPLIT: i32 = 49;
    pub const BATCH_MATMUL: i32 = 126;
    pub const GATHER: i32 = 36;
    pub const SCATTER_ND: i32 = 122;
    pub const DEPTHWISE_CONV_2D: i32 = 4;
    pub const CUSTOM: i32 = 32;
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
        pub const BUILTIN_OPTIONS_TYPE: u16 = 10;
        pub const BUILTIN_OPTIONS: u16 = 12;
    }
    pub mod operator_code {
        pub const DEPRECATED_BUILTIN_CODE: u16 = 4;
        pub const VERSION: u16 = 8;
        pub const BUILTIN_CODE: u16 = 10;
    }
    pub mod buffer {
        pub const DATA: u16 = 4;
    }
}

/// TFLite `BuiltinOptionsType` enum values (union discriminant stored in operator.builtin_options_type).
#[allow(dead_code)]
pub mod builtin_options_type {
    pub const NONE: u8 = 0;
    pub const CONV_2D: u8 = 1;
    pub const SOFTMAX: u8 = 9;
    pub const POOL_2D: u8 = 22;
}

/// VTable field offsets for `SoftmaxOptions`.
#[allow(dead_code)]
pub mod softmax_options {
    pub const BETA: u16 = 4;
}

/// VTable field offsets for `Conv2DOptions`.
#[allow(dead_code)]
pub mod conv2d_options {
    pub const PADDING: u16 = 4;
    pub const STRIDE_W: u16 = 6;
    pub const STRIDE_H: u16 = 8;
    pub const ACTIVATION: u16 = 10;
    pub const DILATION_W: u16 = 12;
    pub const DILATION_H: u16 = 14;
}

/// VTable field offsets for `Pool2DOptions`.
#[allow(dead_code)]
pub mod pool2d_options {
    pub const PADDING: u16 = 4;
    pub const STRIDE_W: u16 = 6;
    pub const STRIDE_H: u16 = 8;
    pub const FILTER_W: u16 = 10;
    pub const FILTER_H: u16 = 12;
    pub const ACTIVATION: u16 = 14;
}
