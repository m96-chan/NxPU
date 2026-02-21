//! Global variables, address spaces, and resource bindings.

use crate::arena::Handle;
use crate::types::Type;

/// Bitflags for storage buffer access modes.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct StorageAccess(u32);

impl StorageAccess {
    /// No access.
    pub const EMPTY: Self = Self(0);
    /// Read access.
    pub const LOAD: Self = Self(1);
    /// Write access.
    pub const STORE: Self = Self(2);

    /// Returns `true` if `self` contains all flags in `other`.
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Returns `true` if no flags are set.
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl std::ops::BitOr for StorageAccess {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for StorageAccess {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// Memory address space for variables.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum AddressSpace {
    /// Function-local storage.
    Function,
    /// Module-scope private storage.
    Private,
    /// Workgroup shared storage.
    Workgroup,
    /// Uniform buffer (read-only).
    Uniform,
    /// Storage buffer with specified access.
    Storage { access: StorageAccess },
}

/// `@group(N) @binding(N)` resource binding.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct ResourceBinding {
    pub group: u32,
    pub binding: u32,
}

/// Built-in shader inputs/outputs.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum BuiltIn {
    /// `@builtin(global_invocation_id)` — vec3<u32>
    GlobalInvocationId,
    /// `@builtin(local_invocation_id)` — vec3<u32>
    LocalInvocationId,
    /// `@builtin(local_invocation_index)` — u32
    LocalInvocationIndex,
    /// `@builtin(workgroup_id)` — vec3<u32>
    WorkgroupId,
    /// `@builtin(num_workgroups)` — vec3<u32>
    NumWorkgroups,
}

/// A binding for a function argument or result.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum Binding {
    /// A built-in shader variable.
    BuiltIn(BuiltIn),
    /// A user-defined location.
    Location { location: u32 },
}

/// A module-scope variable.
#[derive(Clone, Debug)]
pub struct GlobalVariable {
    pub name: Option<String>,
    pub space: AddressSpace,
    pub binding: Option<ResourceBinding>,
    pub ty: Handle<Type>,
    pub init: Option<Handle<crate::Expression>>,
    /// Optional memory layout annotation for tensor data.
    pub layout: Option<crate::types::MemoryLayout>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_access_flags() {
        let read = StorageAccess::LOAD;
        let write = StorageAccess::STORE;
        let rw = read | write;
        assert!(rw.contains(read));
        assert!(rw.contains(write));
        assert!(!read.contains(write));
        assert!(!StorageAccess::EMPTY.contains(read));
        assert!(StorageAccess::EMPTY.is_empty());
    }

    #[test]
    fn storage_access_bitor_assign() {
        let mut access = StorageAccess::LOAD;
        access |= StorageAccess::STORE;
        assert!(access.contains(StorageAccess::LOAD));
        assert!(access.contains(StorageAccess::STORE));
    }

    #[test]
    fn address_space_storage() {
        let space = AddressSpace::Storage {
            access: StorageAccess::LOAD | StorageAccess::STORE,
        };
        if let AddressSpace::Storage { access } = space {
            assert!(access.contains(StorageAccess::LOAD));
            assert!(access.contains(StorageAccess::STORE));
        } else {
            panic!("expected Storage");
        }
    }

    #[test]
    fn resource_binding() {
        let binding = ResourceBinding {
            group: 0,
            binding: 3,
        };
        assert_eq!(binding.group, 0);
        assert_eq!(binding.binding, 3);
    }
}
