//! Type system for the NxPU IR.

use crate::arena::Handle;

/// Width of a scalar type in bytes.
pub type Bytes = u8;

/// The kind of a scalar type.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum ScalarKind {
    /// Boolean.
    Bool,
    /// Signed integer.
    Sint,
    /// Unsigned integer.
    Uint,
    /// Floating point.
    Float,
    /// Brain floating point.
    BFloat,
}

/// A scalar type: kind + byte width.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct Scalar {
    /// The kind of scalar (bool, int, float, etc.).
    pub kind: ScalarKind,
    /// Width of the scalar in bytes.
    pub width: Bytes,
}

impl Scalar {
    pub const BOOL: Self = Self {
        kind: ScalarKind::Bool,
        width: 1,
    };
    pub const I32: Self = Self {
        kind: ScalarKind::Sint,
        width: 4,
    };
    pub const U32: Self = Self {
        kind: ScalarKind::Uint,
        width: 4,
    };
    pub const F16: Self = Self {
        kind: ScalarKind::Float,
        width: 2,
    };
    pub const F32: Self = Self {
        kind: ScalarKind::Float,
        width: 4,
    };
    pub const I8: Self = Self {
        kind: ScalarKind::Sint,
        width: 1,
    };
    pub const U8: Self = Self {
        kind: ScalarKind::Uint,
        width: 1,
    };
    pub const BF16: Self = Self {
        kind: ScalarKind::BFloat,
        width: 2,
    };
}

/// Number of components in a vector.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum VectorSize {
    /// 2 components.
    Bi = 2,
    /// 3 components.
    Tri = 3,
    /// 4 components.
    Quad = 4,
}

/// Size of an array.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum ArraySize {
    /// Fixed-size array.
    Constant(u32),
    /// Runtime-sized array.
    Dynamic,
}

/// A single tensor dimension: either a fixed size or a named symbolic parameter.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum Dimension {
    /// Fixed (static) size known at compile time.
    Fixed(u32),
    /// Dynamic (symbolic) dimension with an optional name (e.g. "batch").
    Dynamic(Option<String>),
}

impl Dimension {
    /// Returns `true` if this dimension is statically known.
    pub fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed(_))
    }

    /// Returns `true` if this dimension is dynamic.
    pub fn is_dynamic(&self) -> bool {
        matches!(self, Self::Dynamic(_))
    }

    /// Returns the fixed size, if known.
    pub fn fixed_size(&self) -> Option<u32> {
        match self {
            Self::Fixed(n) => Some(*n),
            Self::Dynamic(_) => None,
        }
    }
}

/// A multi-dimensional tensor shape supporting mixed static/dynamic dimensions.
///
/// For example, `[batch, 224, 224, 3]` where `batch` is dynamic and spatial
/// dimensions are fixed.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct TensorShape {
    /// The dimensions of the tensor.
    pub dims: Vec<Dimension>,
}

impl TensorShape {
    /// Create a shape where all dimensions are fixed.
    pub fn fixed(sizes: &[u32]) -> Self {
        Self {
            dims: sizes.iter().map(|&s| Dimension::Fixed(s)).collect(),
        }
    }

    /// Create a shape where all dimensions are dynamic (unnamed).
    pub fn all_dynamic(rank: usize) -> Self {
        Self {
            dims: (0..rank).map(|_| Dimension::Dynamic(None)).collect(),
        }
    }

    /// Returns the number of dimensions (rank).
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Returns `true` if all dimensions are statically known.
    pub fn is_fully_static(&self) -> bool {
        self.dims.iter().all(|d| d.is_fixed())
    }

    /// Returns `true` if all dimensions are dynamic.
    pub fn is_fully_dynamic(&self) -> bool {
        self.dims.iter().all(|d| d.is_dynamic())
    }

    /// Returns `true` if the shape contains a mix of static and dynamic dims.
    pub fn is_mixed(&self) -> bool {
        !self.is_fully_static() && !self.is_fully_dynamic()
    }
}

/// Memory layout for tensor data.
///
/// Different NPU hardware expects tensors in specific memory formats.
/// This annotation allows the compiler to insert layout conversions
/// only when necessary.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum MemoryLayout {
    /// Row-major (C-style) — last dimension varies fastest.
    RowMajor,
    /// Column-major (Fortran-style) — first dimension varies fastest.
    ColMajor,
    /// Channels-last: (N, H, W, C) — used by TFLite, Arm Ethos.
    Nhwc,
    /// Channels-first: (N, C, H, W) — used by ONNX, Intel NPU.
    Nchw,
}

impl MemoryLayout {
    /// Returns a human-readable name for the layout.
    pub fn name(self) -> &'static str {
        match self {
            Self::RowMajor => "row_major",
            Self::ColMajor => "col_major",
            Self::Nhwc => "nhwc",
            Self::Nchw => "nchw",
        }
    }
}

/// A member of a struct type.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct StructMember {
    /// Optional member name.
    pub name: Option<String>,
    /// The type of this member.
    pub ty: Handle<Type>,
    /// Byte offset within the struct.
    pub offset: u32,
}

/// A named type.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct Type {
    /// Optional human-readable name.
    pub name: Option<String>,
    /// The concrete type shape.
    pub inner: TypeInner,
}

/// The concrete shape of a type.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum TypeInner {
    /// A single scalar value.
    Scalar(Scalar),
    /// A vector of scalars.
    Vector { size: VectorSize, scalar: Scalar },
    /// A matrix of column vectors.
    Matrix {
        columns: VectorSize,
        rows: VectorSize,
        scalar: Scalar,
    },
    /// An atomic scalar.
    Atomic(Scalar),
    /// A pointer to a value in a given address space.
    Pointer {
        base: Handle<Type>,
        space: crate::AddressSpace,
    },
    /// A fixed-size or runtime-sized array.
    Array {
        base: Handle<Type>,
        size: ArraySize,
        stride: u32,
    },
    /// A composite struct type.
    Struct {
        members: Vec<StructMember>,
        span: u32,
    },
    /// A multi-dimensional tensor with element type and shape.
    ///
    /// Supports mixed static/dynamic dimensions for production ML models
    /// (e.g. dynamic batch with fixed spatial dimensions).
    Tensor {
        /// Element scalar type (e.g. F32, F16, I8).
        scalar: Scalar,
        /// Shape with mixed static/dynamic dimensions.
        shape: TensorShape,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::UniqueArena;

    #[test]
    fn scalar_constants() {
        assert_eq!(Scalar::F32.kind, ScalarKind::Float);
        assert_eq!(Scalar::F32.width, 4);
        assert_eq!(Scalar::U32.kind, ScalarKind::Uint);
        assert_eq!(Scalar::U32.width, 4);
        assert_eq!(Scalar::BOOL.width, 1);
        assert_eq!(Scalar::F16.width, 2);
    }

    #[test]
    fn type_dedup() {
        let mut types = UniqueArena::new();
        let t0 = types.insert(Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::F32),
        });
        let t1 = types.insert(Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::F32),
        });
        assert_eq!(t0, t1);
        assert_eq!(types.len(), 1);
    }

    #[test]
    fn different_types_not_deduped() {
        let mut types = UniqueArena::new();
        let t0 = types.insert(Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::F32),
        });
        let t1 = types.insert(Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::I32),
        });
        assert_ne!(t0, t1);
        assert_eq!(types.len(), 2);
    }

    #[test]
    fn vector_type() {
        let ty = TypeInner::Vector {
            size: VectorSize::Tri,
            scalar: Scalar::F32,
        };
        if let TypeInner::Vector { size, scalar } = ty {
            assert_eq!(size, VectorSize::Tri);
            assert_eq!(scalar, Scalar::F32);
        } else {
            panic!("expected Vector");
        }
    }

    #[test]
    fn vector_size_values() {
        assert_eq!(VectorSize::Bi as u32, 2);
        assert_eq!(VectorSize::Tri as u32, 3);
        assert_eq!(VectorSize::Quad as u32, 4);
    }

    #[test]
    fn dimension_fixed() {
        let d = Dimension::Fixed(224);
        assert!(d.is_fixed());
        assert!(!d.is_dynamic());
        assert_eq!(d.fixed_size(), Some(224));
    }

    #[test]
    fn dimension_dynamic() {
        let d = Dimension::Dynamic(Some("batch".into()));
        assert!(!d.is_fixed());
        assert!(d.is_dynamic());
        assert_eq!(d.fixed_size(), None);
    }

    #[test]
    fn tensor_shape_fixed() {
        let shape = TensorShape::fixed(&[1, 224, 224, 3]);
        assert_eq!(shape.rank(), 4);
        assert!(shape.is_fully_static());
        assert!(!shape.is_fully_dynamic());
        assert!(!shape.is_mixed());
    }

    #[test]
    fn tensor_shape_all_dynamic() {
        let shape = TensorShape::all_dynamic(3);
        assert_eq!(shape.rank(), 3);
        assert!(shape.is_fully_dynamic());
        assert!(!shape.is_fully_static());
        assert!(!shape.is_mixed());
    }

    #[test]
    fn tensor_shape_mixed() {
        let shape = TensorShape {
            dims: vec![
                Dimension::Dynamic(Some("batch".into())),
                Dimension::Fixed(224),
                Dimension::Fixed(224),
                Dimension::Fixed(3),
            ],
        };
        assert_eq!(shape.rank(), 4);
        assert!(shape.is_mixed());
        assert!(!shape.is_fully_static());
        assert!(!shape.is_fully_dynamic());
    }

    #[test]
    fn tensor_type_inner() {
        let mut types = UniqueArena::new();
        let t = types.insert(Type {
            name: Some("image".into()),
            inner: TypeInner::Tensor {
                scalar: Scalar::F32,
                shape: TensorShape {
                    dims: vec![
                        Dimension::Dynamic(Some("batch".into())),
                        Dimension::Fixed(224),
                        Dimension::Fixed(224),
                        Dimension::Fixed(3),
                    ],
                },
            },
        });
        let ty = &types[t];
        match &ty.inner {
            TypeInner::Tensor { scalar, shape } => {
                assert_eq!(*scalar, Scalar::F32);
                assert_eq!(shape.rank(), 4);
                assert!(shape.is_mixed());
            }
            _ => panic!("expected Tensor"),
        }
    }
}
