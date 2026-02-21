//! CoreML protobuf types via prost derive.
//!
//! Minimal hand-defined message types matching CoreML's Model.proto spec.
//! Targets the ML Program (MIL) representation for Apple Neural Engine.

use prost::Message;

/// CoreML specification version for ML Programs.
pub const SPECIFICATION_VERSION: i32 = 7;

/// Top-level CoreML model.
#[derive(Clone, PartialEq, Message)]
pub struct Model {
    #[prost(int32, tag = "1")]
    pub specification_version: i32,
    #[prost(message, optional, tag = "2")]
    pub description: Option<ModelDescription>,
    #[prost(oneof = "model::Type", tags = "10")]
    pub r#type: Option<model::Type>,
}

pub mod model {
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Type {
        #[prost(message, tag = "10")]
        MlProgram(super::MlProgram),
    }
}

/// Model metadata and I/O descriptions.
#[derive(Clone, PartialEq, Message)]
pub struct ModelDescription {
    #[prost(message, repeated, tag = "1")]
    pub input: Vec<FeatureDescription>,
    #[prost(message, repeated, tag = "2")]
    pub output: Vec<FeatureDescription>,
    #[prost(message, optional, tag = "5")]
    pub metadata: Option<Metadata>,
}

/// Description of a single input/output feature.
#[derive(Clone, PartialEq, Message)]
pub struct FeatureDescription {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(message, optional, tag = "3")]
    pub r#type: Option<FeatureType>,
}

/// Type of a feature (array / multi-array).
#[derive(Clone, PartialEq, Message)]
pub struct FeatureType {
    #[prost(oneof = "feature_type::Type", tags = "5")]
    pub r#type: Option<feature_type::Type>,
}

pub mod feature_type {
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Type {
        #[prost(message, tag = "5")]
        MultiArrayType(super::ArrayFeatureType),
    }
}

/// Multi-dimensional array type.
#[derive(Clone, PartialEq, Message)]
pub struct ArrayFeatureType {
    #[prost(int64, repeated, tag = "1")]
    pub shape: Vec<i64>,
    #[prost(enumeration = "ArrayDataType", tag = "2")]
    pub data_type: i32,
}

/// Array element data types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, prost::Enumeration)]
#[repr(i32)]
pub enum ArrayDataType {
    Float16 = 65552,
    Float32 = 65568,
    Int32 = 131104,
}

/// Model metadata (author, description, etc.).
#[derive(Clone, PartialEq, Message)]
pub struct Metadata {
    #[prost(string, tag = "2")]
    pub author: String,
    #[prost(string, tag = "3")]
    pub short_description: String,
}

/// ML Program container.
#[derive(Clone, PartialEq, Message)]
pub struct MlProgram {
    #[prost(message, repeated, tag = "1")]
    pub functions: Vec<MlFunction>,
}

/// A function in the ML Program.
#[derive(Clone, PartialEq, Message)]
pub struct MlFunction {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(message, repeated, tag = "2")]
    pub inputs: Vec<NamedValueType>,
    #[prost(message, optional, tag = "3")]
    pub block: Option<MlBlock>,
}

/// Named value with type info.
#[derive(Clone, PartialEq, Message)]
pub struct NamedValueType {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(string, tag = "2")]
    pub r#type: String,
}

/// A block of MIL operations.
#[derive(Clone, PartialEq, Message)]
pub struct MlBlock {
    #[prost(message, repeated, tag = "1")]
    pub operations: Vec<MlOperation>,
    #[prost(string, repeated, tag = "2")]
    pub outputs: Vec<String>,
}

/// A single MIL operation (matmul, add, etc.).
#[derive(Clone, PartialEq, Message)]
pub struct MlOperation {
    #[prost(string, tag = "1")]
    pub r#type: String,
    #[prost(string, tag = "2")]
    pub name: String,
    #[prost(message, repeated, tag = "3")]
    pub inputs: Vec<MlOperand>,
    #[prost(message, repeated, tag = "4")]
    pub outputs: Vec<MlOperand>,
}

/// An operand reference.
#[derive(Clone, PartialEq, Message)]
pub struct MlOperand {
    #[prost(string, tag = "1")]
    pub name: String,
}

impl FeatureDescription {
    /// Create a multi-array feature description.
    pub fn multi_array(name: impl Into<String>, data_type: ArrayDataType, shape: &[i64]) -> Self {
        Self {
            name: name.into(),
            r#type: Some(FeatureType {
                r#type: Some(feature_type::Type::MultiArrayType(ArrayFeatureType {
                    shape: shape.to_vec(),
                    data_type: data_type as i32,
                })),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    #[test]
    fn model_roundtrip() {
        let model = Model {
            specification_version: SPECIFICATION_VERSION,
            description: Some(ModelDescription {
                input: vec![FeatureDescription::multi_array(
                    "input",
                    ArrayDataType::Float16,
                    &[-1, -1],
                )],
                output: vec![],
                metadata: Some(Metadata {
                    author: "nxpu".into(),
                    short_description: "test".into(),
                }),
            }),
            r#type: Some(model::Type::MlProgram(MlProgram { functions: vec![] })),
        };

        let bytes = model.encode_to_vec();
        let decoded = Model::decode(bytes.as_slice()).unwrap();
        assert_eq!(model, decoded);
    }
}
