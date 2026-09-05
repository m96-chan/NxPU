//! Recognising a reindexing kernel, and getting the permutation right.
//!
//! The label is the easy half. A `Transpose` carrying the wrong `perm` is a
//! graph that runs and moves the data somewhere else, which is the failure
//! this whole exercise exists to stop, so these assert the permutation rather
//! than the pattern's name.

use nxpu_analysis::analyze::{self, KernelPattern};

fn classify(source: &str) -> KernelPattern {
    let mut module = nxpu_parser::parse(source).expect("WGSL parse failed");
    nxpu_opt::PassManager::for_level(nxpu_opt::OptLevel::O1).run(&mut module);
    analyze::classify_entry_point(&module, 0).expect("classification failed")
}

/// `[dim0, dim1, D]` written back as `[dim1, dim0, D]` — web-xpu-ops' permute,
/// which swaps the two outer axes and leaves the innermost alone.
const SWAP_OUTER: &str = "
struct Params { dim0: u32, dim1: u32, D: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.dim0 * params.dim1 * params.D) { return; }
  let d = idx % params.D;
  let rest = idx / params.D;
  let i1 = rest % params.dim1;
  let i0 = rest / params.dim1;
  output[(i1 * params.dim0 + i0) * params.D + d] = input[idx];
}";

#[test]
fn outer_axes_swapped() {
    match classify(SWAP_OUTER) {
        KernelPattern::Transpose { perm, .. } => assert_eq!(perm, vec![1, 0, 2]),
        other => panic!("expected Transpose, got {other:?}"),
    }
}

/// The same kernel written to reassemble the index in the order it took it
/// apart. That is a copy, and calling it a transpose would be a permutation
/// that does nothing — worth refusing rather than emitting.
#[test]
fn the_identity_is_not_a_transpose() {
    let source = SWAP_OUTER.replace(
        "output[(i1 * params.dim0 + i0) * params.D + d] = input[idx];",
        "output[(i0 * params.dim1 + i1) * params.D + d] = input[idx];",
    );
    assert!(
        !matches!(classify(&source), KernelPattern::Transpose { .. }),
        "an identity permutation was reported as a transpose"
    );
}

/// A read that is not the plain flat id is a gather, and the permutation
/// derivation says nothing about where its values come from.
#[test]
fn an_indirect_read_is_not_a_transpose() {
    let source = "
struct Params { dim0: u32, dim1: u32, D: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.dim0 * params.dim1 * params.D) { return; }
  let d = idx % params.D;
  let rest = idx / params.D;
  let i1 = rest % params.dim1;
  let i0 = rest / params.dim1;
  output[(i1 * params.dim0 + i0) * params.D + d] = input[idx / 2u];
}";
    assert!(
        !matches!(classify(source), KernelPattern::Transpose { .. }),
        "a gather was reported as a transpose"
    );
}
