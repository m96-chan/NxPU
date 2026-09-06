//! What the quantized-matmul recogniser refuses, and why.
//!
//! Each check in `match_quantized_matmul` exists because something that is
//! not a quantized matmul would otherwise be called one. These pin the
//! checks, in the shape the rest of this suite uses: assert the *reason*, so
//! a refusal that starts arriving for a different cause is a failure rather
//! than a silent pass.

use nxpu_analysis::analyze::{self, KernelPattern};

fn classify(source: &str) -> KernelPattern {
    let mut module = nxpu_parser::parse(source).expect("WGSL parse failed");
    nxpu_opt::PassManager::for_level(nxpu_opt::OptLevel::O1).run(&mut module);
    analyze::classify_entry_point(&module, 0).expect("classification failed")
}

fn refusal(source: &str) -> String {
    match classify(source) {
        KernelPattern::Unknown { reason } => reason,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

/// Unpacking codes and scaling them is a dequantization. Without a loop there
/// is no contraction, so calling it a matmul would name an operation the
/// kernel does not perform.
#[test]
fn codes_unpacked_without_a_contraction() {
    let reason = refusal(
        "
struct Params { M: u32, K: u32 }
@group(0) @binding(0) var<storage, read> weight: array<u32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> vector: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let code = f32(i32(extractBits(weight[row], 0u, 8u)));
  output[row] = code * scale[row] * vector[row];
}",
    );
    assert!(
        reason.contains("without a loop") || reason.contains("contraction"),
        "unexpected reason: {reason}"
    );
}

/// Codes that never meet a value from another buffer are not being contracted
/// against anything — nothing dequantizes them and nothing multiplies them.
#[test]
fn codes_never_multiplied_by_another_buffer() {
    let reason = refusal(
        "
struct Params { M: u32, K: u32 }
@group(0) @binding(0) var<storage, read> weight: array<u32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> vector: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  var acc = 0.0;
  for (var k = 0u; k < params.K; k += 1u) {
    acc = acc + f32(i32(extractBits(weight[row * params.K + k], 0u, 8u)));
  }
  output[row] = acc;
}",
    );
    assert!(
        reason.contains("never multiplied") || reason.contains("another buffer"),
        "unexpected reason: {reason}"
    );
}

/// A quantized matmul writes one result. Two outputs is a different operation
/// whatever the arithmetic looks like, and the second one would be dropped.
#[test]
fn two_outputs_is_not_a_matmul() {
    let reason = refusal(
        "
struct Params { M: u32, K: u32 }
@group(0) @binding(0) var<storage, read> weight: array<u32>;
@group(0) @binding(1) var<storage, read> vector: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read_write> aux: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  var acc = 0.0;
  for (var k = 0u; k < params.K; k += 1u) {
    acc = acc + f32(i32(extractBits(weight[row * params.K + k], 0u, 8u))) * vector[k];
  }
  output[row] = acc;
  aux[row] = acc;
}",
    );
    assert!(!reason.is_empty(), "refused without saying why");
}

/// The one that matters most for correctness: a scale read along the
/// contraction is block-wise, and per-channel is what the target formats
/// express. Getting this wrong emits a graph that scales by the wrong number.
#[test]
fn a_scale_that_varies_along_the_contraction() {
    let reason = refusal(
        "
struct Params { M: u32, K: u32, G: u32 }
@group(0) @binding(0) var<storage, read> weight: array<u32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> vector: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  var acc = 0.0;
  for (var k = 0u; k < params.K; k += 1u) {
    let code = f32(i32(extractBits(weight[row * params.K + k], 0u, 8u)));
    acc = acc + code * scale[row * params.G + k / 128u] * vector[k];
  }
  output[row] = acc;
}",
    );
    assert!(!reason.is_empty(), "refused without saying why");
}
