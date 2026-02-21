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
    #[prost(message, repeated, tag = "5")]
    pub attribute: Vec<AttributeProto>,
    #[prost(string, tag = "7")]
    pub domain: String,
}

/// ONNX attribute types.
pub mod attribute_type {
    pub const FLOAT: i32 = 1;
    pub const INT: i32 = 2;
    pub const INTS: i32 = 7;
}

/// An attribute of an operator node.
#[derive(Clone, PartialEq, Message)]
pub struct AttributeProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(int32, tag = "20")]
    pub r#type: i32,
    #[prost(int64, tag = "3")]
    pub i: i64,
    #[prost(float, tag = "4")]
    pub f: f32,
    #[prost(int64, repeated, tag = "8")]
    pub ints: Vec<i64>,
}

impl AttributeProto {
    /// Create an integer attribute.
    pub fn int(name: impl Into<String>, value: i64) -> Self {
        Self {
            name: name.into(),
            r#type: attribute_type::INT,
            i: value,
            f: 0.0,
            ints: vec![],
        }
    }

    /// Create a float attribute.
    pub fn float(name: impl Into<String>, value: f32) -> Self {
        Self {
            name: name.into(),
            r#type: attribute_type::FLOAT,
            i: 0,
            f: value,
            ints: vec![],
        }
    }

    /// Create a list-of-integers attribute.
    pub fn ints(name: impl Into<String>, values: Vec<i64>) -> Self {
        Self {
            name: name.into(),
            r#type: attribute_type::INTS,
            i: 0,
            f: 0.0,
            ints: values,
        }
    }
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

impl NodeProto {
    /// Create a node with no attributes.
    pub fn simple(
        op_type: impl Into<String>,
        name: impl Into<String>,
        input: Vec<String>,
        output: Vec<String>,
    ) -> Self {
        Self {
            input,
            output,
            name: name.into(),
            op_type: op_type.into(),
            attribute: vec![],
            domain: String::new(),
        }
    }

    /// Create a node with attributes.
    pub fn with_attrs(
        op_type: impl Into<String>,
        name: impl Into<String>,
        input: Vec<String>,
        output: Vec<String>,
        attribute: Vec<AttributeProto>,
    ) -> Self {
        Self {
            input,
            output,
            name: name.into(),
            op_type: op_type.into(),
            attribute,
            domain: String::new(),
        }
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
                node: vec![NodeProto::simple(
                    "MatMul",
                    "matmul_0",
                    vec!["A".into(), "B".into()],
                    vec!["C".into()],
                )],
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
    fn float_attribute_roundtrip() {
        let attr = AttributeProto::float("epsilon", 1e-5);
        assert_eq!(attr.r#type, attribute_type::FLOAT);
        assert_eq!(attr.f, 1e-5);
        assert_eq!(attr.i, 0);

        // Roundtrip through protobuf encoding
        let node = NodeProto::with_attrs(
            "BatchNormalization",
            "bn",
            vec!["x".into()],
            vec!["y".into()],
            vec![attr],
        );
        let bytes = node.encode_to_vec();
        let decoded = NodeProto::decode(bytes.as_slice()).unwrap();
        let eps = &decoded.attribute[0];
        assert_eq!(eps.name, "epsilon");
        assert_eq!(eps.r#type, attribute_type::FLOAT);
        assert!((eps.f - 1e-5).abs() < 1e-10);
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
        let type_proto::Value::TensorType(tensor) = ty.value.unwrap();
        assert_eq!(tensor.elem_type, data_type::FLOAT);
        assert_eq!(tensor.shape.unwrap().dim.len(), 1);
    }
}
