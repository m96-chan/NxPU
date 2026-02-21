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
    pub kind: ScalarKind,
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

/// A member of a struct type.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct StructMember {
    pub name: Option<String>,
    pub ty: Handle<Type>,
    pub offset: u32,
}

/// A named type.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct Type {
    pub name: Option<String>,
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
}
