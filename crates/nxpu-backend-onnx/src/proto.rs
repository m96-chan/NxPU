//! ONNX protobuf types via prost derive.
//!
//! Hand-defined message types matching the ONNX IR specification (onnx.proto).
//! Field tags correspond to the official ONNX protobuf field numbers.

use prost::Message;

/// ONNX data type constants from `TensorProto.DataType`.
pub mod data_type {
    pub const FLOAT: i32 = 1;
    pub const UINT8: i32 = 2;
    pub const INT8: i32 = 3;
    pub const FLOAT16: i32 = 10;
    pub const INT32: i32 = 6;
    pub const UINT32: i32 = 12;
    pub const BOOL: i32 = 9;
    pub const BFLOAT16: i32 = 16;
}

/// Top-level ONNX model container.
#[derive(Clone, PartialEq, Message)]
pub struct ModelProto {
    #[prost(int64, tag = "1")]
    pub ir_version: i64,
    #[prost(string, tag = "2")]
    pub producer_name: String,
    #[prost(string, tag = "3")]
    pub producer_version: String,
    #[prost(message, optional, tag = "7")]
    pub graph: Option<GraphProto>,
    #[prost(message, repeated, tag = "8")]
    pub opset_import: Vec<OperatorSetIdProto>,
}

/// Operator set version declaration.
#[derive(Clone, PartialEq, Message)]
pub struct OperatorSetIdProto {
    #[prost(string, tag = "1")]
    pub domain: String,
    #[prost(int64, tag = "2")]
    pub version: i64,
}

/// A computation graph.
#[derive(Clone, PartialEq, Message)]
pub struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    pub node: Vec<NodeProto>,
    #[prost(string, tag = "2")]
    pub name: String,
    #[prost(message, repeated, tag = "11")]
    pub input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    pub output: Vec<ValueInfoProto>,
}

/// A single operator invocation.
#[derive(Clone, PartialEq, Message)]
pub struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    pub input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    pub output: Vec<String>,
    #[prost(string, tag = "3")]
    pub name: String,
    #[prost(string, tag = "4")]
    pub op_type: String,
    #[prost(string, tag = "7")]
    pub domain: String,
}

/// Typed tensor name declaration.
#[derive(Clone, PartialEq, Message)]
pub struct ValueInfoProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(message, optional, tag = "2")]
    pub r#type: Option<TypeProto>,
}

impl ValueInfoProto {
    /// Create a tensor value info with symbolic/fixed dimensions.
    pub fn tensor(
        name: impl Into<String>,
        elem_type: i32,
        dims: Vec<TensorShapeDimension>,
    ) -> Self {
        Self {
            name: name.into(),
            r#type: Some(TypeProto {
                value: Some(type_proto::Value::TensorType(TensorTypeProto {
                    elem_type,
                    shape: Some(TensorShapeProto { dim: dims }),
                })),
            }),
        }
    }
}

/// Type of a value (currently only tensor types).
#[derive(Clone, PartialEq, Message)]
pub struct TypeProto {
    #[prost(oneof = "type_proto::Value", tags = "1")]
    pub value: Option<type_proto::Value>,
}

pub mod type_proto {
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Value {
        #[prost(message, tag = "1")]
        TensorType(super::TensorTypeProto),
    }
}

/// Tensor type: element data type + shape.
#[derive(Clone, PartialEq, Message)]
pub struct TensorTypeProto {
    #[prost(int32, tag = "1")]
    pub elem_type: i32,
    #[prost(message, optional, tag = "2")]
    pub shape: Option<TensorShapeProto>,
}

/// Tensor shape: a list of dimensions.
#[derive(Clone, PartialEq, Message)]
pub struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    pub dim: Vec<TensorShapeDimension>,
}

/// A single dimension (either a fixed value or a symbolic parameter).
#[derive(Clone, PartialEq, Message)]
pub struct TensorShapeDimension {
    #[prost(oneof = "tensor_shape_dimension::Value", tags = "1, 2")]
    pub value: Option<tensor_shape_dimension::Value>,
}

pub mod tensor_shape_dimension {
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Value {
        #[prost(int64, tag = "1")]
        DimValue(i64),
        #[prost(string, tag = "2")]
        DimParam(String),
    }
}

impl TensorShapeDimension {
    /// Create a symbolic (named) dimension.
    pub fn symbolic(name: impl Into<String>) -> Self {
        Self {
            value: Some(tensor_shape_dimension::Value::DimParam(name.into())),
        }
    }

    /// Create a fixed-size dimension.
    pub fn fixed(size: i64) -> Self {
        Self {
            value: Some(tensor_shape_dimension::Value::DimValue(size)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    #[test]
    fn model_roundtrip() {
        let model = ModelProto {
            ir_version: 7,
            producer_name: "nxpu".into(),
            producer_version: "0.1.0".into(),
            graph: Some(GraphProto {
                name: "test".into(),
                node: vec![NodeProto {
                    input: vec!["A".into(), "B".into()],
                    output: vec!["C".into()],
                    name: "matmul_0".into(),
                    op_type: "MatMul".into(),
                    domain: String::new(),
                }],
                input: vec![ValueInfoProto::tensor(
                    "A",
                    data_type::FLOAT,
                    vec![
                        TensorShapeDimension::symbolic("M"),
                        TensorShapeDimension::symbolic("K"),
                    ],
                )],
                output: vec![ValueInfoProto::tensor(
                    "C",
                    data_type::FLOAT,
                    vec![
                        TensorShapeDimension::symbolic("M"),
                        TensorShapeDimension::symbolic("N"),
                    ],
                )],
            }),
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 13,
            }],
        };

        let bytes = model.encode_to_vec();
        let decoded = ModelProto::decode(bytes.as_slice()).unwrap();
        assert_eq!(model, decoded);
    }

    #[test]
    fn tensor_shape_symbolic() {
        let dim = TensorShapeDimension::symbolic("N");
        assert_eq!(
            dim.value,
            Some(tensor_shape_dimension::Value::DimParam("N".into()))
        );
    }

    #[test]
    fn tensor_shape_fixed() {
        let dim = TensorShapeDimension::fixed(128);
        assert_eq!(
            dim.value,
            Some(tensor_shape_dimension::Value::DimValue(128))
        );
    }

    #[test]
    fn value_info_tensor_helper() {
        let vi = ValueInfoProto::tensor(
            "X",
            data_type::FLOAT,
            vec![TensorShapeDimension::symbolic("batch")],
        );
        assert_eq!(vi.name, "X");
        let ty = vi.r#type.unwrap();
        let tensor = match ty.value.unwrap() {
            type_proto::Value::TensorType(t) => t,
        };
        assert_eq!(tensor.elem_type, data_type::FLOAT);
        assert_eq!(tensor.shape.unwrap().dim.len(), 1);
    }
}
