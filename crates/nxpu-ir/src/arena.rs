//! Arena-based storage with typed handles.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

/// A typed handle into an [`Arena`] or [`UniqueArena`].
///
/// Handles are lightweight identifiers (u32 index) that provide
/// type-safe access to arena-allocated values.
pub struct Handle<T> {
    index: u32,
    _phantom: PhantomData<T>,
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Handle<T> {}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T> Eq for Handle<T> {}

impl<T> PartialOrd for Handle<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Handle<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.index.cmp(&other.index)
    }
}

impl<T> Hash for Handle<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
    }
}

impl<T> fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", self.index)
    }
}

impl<T> Handle<T> {
    /// Creates a new handle from a zero-based index.
    pub(crate) fn new(index: u32) -> Self {
        Self {
            index,
            _phantom: PhantomData,
        }
    }

    /// Returns the zero-based index of this handle.
    pub fn index(self) -> usize {
        self.index as usize
    }
}

/// A half-open range of [`Handle`]s: `[first, last)`.
pub struct Range<T> {
    first: u32,
    last: u32,
    _phantom: PhantomData<T>,
}

impl<T> Clone for Range<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Range<T> {}

impl<T> PartialEq for Range<T> {
    fn eq(&self, other: &Self) -> bool {
        self.first == other.first && self.last == other.last
    }
}

impl<T> Eq for Range<T> {}

impl<T> Hash for Range<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.first.hash(state);
        self.last.hash(state);
    }
}

impl<T> fmt::Debug for Range<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}..{})", self.first, self.last)
    }
}

impl<T> Range<T> {
    /// Creates a new range from two handles.
    pub fn new(first: Handle<T>, last: Handle<T>) -> Self {
        Self {
            first: first.index,
            last: last.index,
            _phantom: PhantomData,
        }
    }

    /// Creates a range from raw u32 indices.
    pub fn from_index_range(range: std::ops::Range<u32>) -> Self {
        Self {
            first: range.start,
            last: range.end,
            _phantom: PhantomData,
        }
    }

    /// Returns the first handle in the range.
    pub fn first(&self) -> Handle<T> {
        Handle::new(self.first)
    }

    /// Returns the handle one past the last element.
    pub fn end(&self) -> Handle<T> {
        Handle::new(self.last)
    }

    /// Returns this range as a `std::ops::Range<u32>`.
    pub fn index_range(&self) -> std::ops::Range<u32> {
        self.first..self.last
    }

    /// Returns `true` if the range contains no elements.
    pub fn is_empty(&self) -> bool {
        self.first >= self.last
    }
}

/// An append-only arena with typed [`Handle`]-based access.
#[derive(Clone, Debug)]
pub struct Arena<T> {
    data: Vec<T>,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Arena<T> {
    /// Creates an empty arena.
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Returns the number of elements in the arena.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the arena contains no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the handle that will be assigned to the next appended value.
    pub fn next_handle(&self) -> Handle<T> {
        let index = u32::try_from(self.data.len()).unwrap_or_else(|_| {
            panic!("arena overflow: {} items exceeds u32::MAX", self.data.len())
        });
        Handle::new(index)
    }

    /// Appends a value and returns its handle.
    pub fn append(&mut self, value: T) -> Handle<T> {
        let index = u32::try_from(self.data.len()).unwrap_or_else(|_| {
            panic!("arena overflow: {} items exceeds u32::MAX", self.data.len())
        });
        self.data.push(value);
        Handle::new(index)
    }

    /// Returns a reference to the value if the handle is valid.
    pub fn try_get(&self, handle: Handle<T>) -> Option<&T> {
        self.data.get(handle.index())
    }

    /// Iterates over `(handle, &value)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (Handle<T>, &T)> {
        // Safety: arena size bounded by u32::MAX (enforced in append)
        self.data
            .iter()
            .enumerate()
            .map(|(i, v)| (Handle::new(i as u32), v))
    }

    /// Iterates over `(handle, &mut value)` pairs.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (Handle<T>, &mut T)> {
        // Safety: arena size bounded by u32::MAX (enforced in append)
        self.data
            .iter_mut()
            .enumerate()
            .map(|(i, v)| (Handle::new(i as u32), v))
    }
}

impl<T> Index<Handle<T>> for Arena<T> {
    type Output = T;

    fn index(&self, handle: Handle<T>) -> &T {
        &self.data[handle.index()]
    }
}

impl<T> IndexMut<Handle<T>> for Arena<T> {
    fn index_mut(&mut self, handle: Handle<T>) -> &mut T {
        &mut self.data[handle.index()]
    }
}

/// A deduplicating arena that returns the same [`Handle`] for equal values.
#[derive(Clone, Debug)]
pub struct UniqueArena<T> {
    data: Vec<T>,
    map: HashMap<T, u32>,
}

impl<T: Hash + Eq> Default for UniqueArena<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Hash + Eq> UniqueArena<T> {
    /// Creates an empty deduplicating arena.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            map: HashMap::new(),
        }
    }

    /// Returns the number of unique elements in the arena.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the arena contains no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Inserts a value, returning an existing handle if the value is already present.
    pub fn insert(&mut self, value: T) -> Handle<T>
    where
        T: Clone,
    {
        if let Some(&index) = self.map.get(&value) {
            return Handle::new(index);
        }
        let index = u32::try_from(self.data.len()).unwrap_or_else(|_| {
            panic!("arena overflow: {} items exceeds u32::MAX", self.data.len())
        });
        self.map.insert(value.clone(), index);
        self.data.push(value);
        Handle::new(index)
    }

    /// Returns a reference to the value if the handle is valid.
    pub fn try_get(&self, handle: Handle<T>) -> Option<&T> {
        self.data.get(handle.index())
    }

    /// Iterates over `(handle, &value)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (Handle<T>, &T)> {
        // Safety: arena size bounded by u32::MAX (enforced in insert)
        self.data
            .iter()
            .enumerate()
            .map(|(i, v)| (Handle::new(i as u32), v))
    }
}

impl<T> Index<Handle<T>> for UniqueArena<T> {
    type Output = T;

    fn index(&self, handle: Handle<T>) -> &T {
        &self.data[handle.index()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_append_and_access() {
        let mut arena = Arena::new();
        let h0 = arena.append("hello");
        let h1 = arena.append("world");
        assert_eq!(arena[h0], "hello");
        assert_eq!(arena[h1], "world");
        assert_eq!(arena.len(), 2);
    }

    #[test]
    fn arena_iter() {
        let mut arena = Arena::new();
        arena.append(10);
        arena.append(20);
        arena.append(30);
        let items: Vec<_> = arena.iter().map(|(h, &v)| (h.index(), v)).collect();
        assert_eq!(items, vec![(0, 10), (1, 20), (2, 30)]);
    }

    #[test]
    fn arena_next_handle() {
        let mut arena = Arena::<i32>::new();
        let h0 = arena.next_handle();
        assert_eq!(h0.index(), 0);
        arena.append(42);
        let h1 = arena.next_handle();
        assert_eq!(h1.index(), 1);
    }

    #[test]
    fn unique_arena_dedup() {
        let mut arena = UniqueArena::new();
        let h0 = arena.insert(42);
        let h1 = arena.insert(99);
        let h2 = arena.insert(42); // duplicate
        assert_eq!(h0, h2);
        assert_ne!(h0, h1);
        assert_eq!(arena.len(), 2);
    }

    #[test]
    fn handle_ordering() {
        let h0: Handle<u32> = Handle::new(0);
        let h1: Handle<u32> = Handle::new(1);
        assert!(h0 < h1);
        assert_eq!(h0, h0);
    }

    #[test]
    fn range_operations() {
        let range = Range::<u32>::from_index_range(2..5);
        assert!(!range.is_empty());
        assert_eq!(range.first().index(), 2);
        assert_eq!(range.end().index(), 5);
        assert_eq!(range.index_range(), 2..5);

        let empty = Range::<u32>::from_index_range(3..3);
        assert!(empty.is_empty());
    }

    #[test]
    fn arena_try_get() {
        let mut arena = Arena::new();
        let h0 = arena.append(42);
        assert_eq!(arena.try_get(h0), Some(&42));
        assert_eq!(arena.try_get(Handle::new(99)), None);
    }
}
