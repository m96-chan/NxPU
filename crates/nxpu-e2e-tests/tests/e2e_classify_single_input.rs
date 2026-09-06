//! One input, and what the classifier makes of it.
//!
//! The single-input arm decided by counting: two or more outputs and an `If`
//! somewhere meant `Split`, and a loop meant a reduction whose kind was
//! whichever of `max`/`min`/`/` turned up first. Both answers are confident
//! and neither looks at what the kernel computes, so a softmax came back as
//! `ReduceMax` and three kernels that produce two unrelated tensors came back
//! as `Split`.
//!
//! These assert the outcome rather than the code path: for the softmax, the
//! activation and the dimension it names; for the refusals, that the reason
//! names the outputs that disqualified the kernel, so it is checkable against
//! the source in front of the reader.
//!
//! The sources are reduced from `vendor/web-xpu-ops` — comments stripped, the
//! workgroup reductions kept, since those are the part the walks have to see
//! through — so the test does not need the submodule checked out.

use nxpu_analysis::analyze::{self, ActivationOp, KernelPattern};
use nxpu_opt::{OptLevel, PassManager};

fn classify_at(source: &str, level: OptLevel) -> KernelPattern {
    let mut module = nxpu_parser::parse(source).expect("WGSL parse failed");
    PassManager::for_level(level).run(&mut module);
    analyze::classify_entry_point(&module, 0).expect("classification failed")
}

fn classify(source: &str) -> KernelPattern {
    classify_at(source, OptLevel::O1)
}

/// The reason string of an `Unknown`, or a panic naming what was returned
/// instead. Every refusal below has to be a refusal at every level.
fn refusal(source: &str) -> String {
    let mut reasons = Vec::new();
    for level in [OptLevel::O0, OptLevel::O1, OptLevel::O2] {
        match classify_at(source, level) {
            KernelPattern::Unknown { reason } => reasons.push(reason),
            other => panic!("expected Unknown at {level:?}, got {other:?}"),
        }
    }
    assert_eq!(reasons[0], reasons[1], "reason changed between -O0 and -O1");
    assert_eq!(reasons[0], reasons[2], "reason changed between -O0 and -O2");
    reasons.remove(0)
}

// ---------------------------------------------------------------------------
// softmax
// ---------------------------------------------------------------------------

/// `ops/softmax/wgsl/kernel.wgsl`: the numerically stable three-pass softmax,
/// max and sum each collapsed through a workgroup tree.
const SOFTMAX: &str = r#"
struct Params { N: u32, D: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
const WORKGROUP_SIZE: u32 = 256u;
var<workgroup> shared_val: array<f32, 256>;
@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let row = wg_id.x;
  if (row >= params.N) { return; }
  let tid = local_id.x;
  let row_offset = row * params.D;

  var local_max: f32 = -3.402823e+38;
  for (var col = tid; col < params.D; col += WORKGROUP_SIZE) {
    local_max = max(local_max, input[row_offset + col]);
  }
  shared_val[tid] = local_max;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { shared_val[tid] = max(shared_val[tid], shared_val[tid + stride]); }
    workgroupBarrier();
  }
  let row_max = shared_val[0];
  workgroupBarrier();

  var local_sum: f32 = 0.0;
  for (var col = tid; col < params.D; col += WORKGROUP_SIZE) {
    local_sum += exp(input[row_offset + col] - row_max);
  }
  shared_val[tid] = local_sum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { shared_val[tid] += shared_val[tid + stride]; }
    workgroupBarrier();
  }
  let inv_sum = 1.0 / shared_val[0];
  workgroupBarrier();

  for (var col = tid; col < params.D; col += WORKGROUP_SIZE) {
    output[row_offset + col] = exp(input[row_offset + col] - row_max) * inv_sum;
  }
}
"#;

/// The op, and the dimension it is taken over.
///
/// `ReduceMax` was the old answer, on the strength of the first pass. It keeps
/// one number per row and throws the distribution away, which is not a softmax
/// by any reading.
///
/// The dimension matters as much as the name here. The emitted graph carries a
/// single symbolic dimension, and the reduction runs along the innermost axis —
/// `D` — so labelling it `N`, the batch count, would put the wrong length on
/// the axis being normalised.
#[test]
fn softmax_is_a_softmax_over_the_inner_dimension() {
    match classify(SOFTMAX) {
        KernelPattern::Activation {
            op,
            input,
            output,
            dim_name,
        } => {
            assert_eq!(op, ActivationOp::Softmax);
            assert_eq!(input.name, "input");
            assert_eq!(output.name, "output");
            assert_eq!(dim_name, "D", "softmax runs along D, not N");
        }
        other => panic!("expected Activation(Softmax), got {other:?}"),
    }
}

#[test]
fn softmax_survives_every_opt_level() {
    for level in [OptLevel::O0, OptLevel::O1, OptLevel::O2] {
        match classify_at(SOFTMAX, level) {
            KernelPattern::Activation { op, .. } => assert_eq!(op, ActivationOp::Softmax),
            other => panic!("expected Activation(Softmax) at {level:?}, got {other:?}"),
        }
    }
}

/// A loop that sums exponentials and then writes the sum. Same accumulation as
/// the softmax's second pass, and it is not a softmax: nothing is divided by
/// the total. Recognition has to want both halves.
const SUM_OF_EXPONENTIALS: &str = r#"
struct Params { N: u32, D: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.N) { return; }
  var total = 0.0;
  for (var c = 0u; c < params.D; c += 1u) {
    total += exp(input[row * params.D + c]);
  }
  output[row] = total;
}
"#;

#[test]
fn a_sum_of_exponentials_is_not_a_softmax() {
    assert!(
        !matches!(
            classify(SUM_OF_EXPONENTIALS),
            KernelPattern::Activation {
                op: ActivationOp::Softmax,
                ..
            }
        ),
        "a reduction that never divides by its total was called a softmax"
    );
}

/// `exp(x) / c` with `c` a uniform: an exponential scaled by a constant, in a
/// loop. It stores an `exp` over a divide, which is the softmax's *other*
/// half, and there is no running total of exponentials anywhere. This is the
/// shape a hand-written sigmoid, `1 / (1 + exp(-x))`, also has.
const SCALED_EXPONENTIAL: &str = r#"
struct Params { N: u32, D: u32, scale: f32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.N) { return; }
  for (var c = 0u; c < params.D; c += 1u) {
    output[row * params.D + c] = exp(input[row * params.D + c]) / params.scale;
  }
}
"#;

#[test]
fn a_scaled_exponential_is_not_a_softmax() {
    assert!(
        !matches!(
            classify(SCALED_EXPONENTIAL),
            KernelPattern::Activation {
                op: ActivationOp::Softmax,
                ..
            }
        ),
        "an exp divided by a uniform was called a softmax"
    );
}

// ---------------------------------------------------------------------------
// One input, several outputs: a Split, or a refusal
// ---------------------------------------------------------------------------

/// `examples/split.wgsl`. A Split slices: every element it writes is an
/// element it read. Refusing the three kernels below must not cost this one.
const SPLIT: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_a: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_b: array<f32>;
struct Params { N: u32, split_at: u32 }
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.N) { return; }
  if (idx < params.split_at) {
    out_a[idx] = input[idx];
  } else {
    out_b[idx - params.split_at] = input[idx];
  }
}
"#;

#[test]
fn a_real_split_is_still_a_split() {
    for level in [OptLevel::O0, OptLevel::O1, OptLevel::O2] {
        match classify_at(SPLIT, level) {
            KernelPattern::Split { input, outputs, .. } => {
                assert_eq!(input.name, "input");
                assert_eq!(
                    outputs.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
                    vec!["out_a", "out_b"],
                );
            }
            other => panic!("expected Split at {level:?}, got {other:?}"),
        }
    }
}

/// `ops/quantize/wgsl/kernel.wgsl`: per-token absmax quantization. It emits
/// int8 codes and the per-row scale that decodes them — two tensors of
/// different shapes and different element types, neither of them a piece of
/// the input.
const QUANTIZE: &str = r#"
struct Params { N: u32, D: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;
@group(0) @binding(2) var<storage, read_write> scales: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
const WORKGROUP_SIZE: u32 = 256u;
var<workgroup> shared_max: array<f32, 256>;
@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let row = wg_id.x;
  if (row >= params.N) { return; }
  let tid = local_id.x;
  let row_offset = row * params.D;
  var local_max: f32 = 0.0;
  for (var col = tid; col < params.D; col += WORKGROUP_SIZE) {
    local_max = max(local_max, abs(input[row_offset + col]));
  }
  shared_max[tid] = local_max;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]); }
    workgroupBarrier();
  }
  let absmax = shared_max[0];
  let scale = select(absmax / 127.0, 1.0, absmax == 0.0);
  if (tid == 0u) { scales[row] = scale; }
  workgroupBarrier();
  let inv_scale = select(127.0 / absmax, 0.0, absmax == 0.0);
  for (var col = tid; col < params.D; col += WORKGROUP_SIZE) {
    let val = input[row_offset + col];
    output[row_offset + col] = clamp(i32(round(val * inv_scale)), -127, 127);
  }
}
"#;

#[test]
fn quantization_is_refused_and_names_both_outputs() {
    let reason = refusal(QUANTIZE);
    assert!(
        reason.contains("'output'") && reason.contains("'scales'"),
        "the reason should name the outputs it refused on, got: {reason}"
    );
    assert!(
        reason.contains("computed value rather than a copy"),
        "the reason should say why they are not slices, got: {reason}"
    );
    assert!(
        reason.contains("element type"),
        "int8 codes out of an f32 tensor is the other half of it, got: {reason}"
    );
}

/// `ops/ctc_decode/wgsl/kernel.wgsl`: greedy CTC decode. An argmax per frame,
/// repeats collapsed, blanks dropped — labels and how many of them there are.
/// The write position depends on the data, which no slice's does.
const CTC_DECODE: &str = r#"
struct Params { B: u32, T: u32, C: u32, blank: u32 }
@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read_write> tokens: array<i32>;
@group(0) @binding(2) var<storage, read_write> lengths: array<i32>;
@group(0) @binding(3) var<uniform> params: Params;
const WORKGROUP_SIZE: u32 = 256u;
const PAD: i32 = -1;
const LOWEST: f32 = -3.4028234e+38;
var<workgroup> top_score: array<f32, 256>;
var<workgroup> top_class: array<u32, 256>;
@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let row = wg_id.x;
  let tid = local_id.x;
  let base = row * params.T * params.C;
  var previous: u32 = params.blank;
  var count: u32 = 0u;
  for (var t: u32 = 0u; t < params.T; t += 1u) {
    let frame = base + t * params.C;
    var best: f32 = LOWEST;
    var best_c: u32 = 0u;
    for (var c = tid; c < params.C; c += WORKGROUP_SIZE) {
      let value = scores[frame + c];
      if (value > best) { best = value; best_c = c; }
    }
    top_score[tid] = best;
    top_class[tid] = best_c;
    workgroupBarrier();
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
      if (tid < stride) {
        let other = top_score[tid + stride];
        let other_c = top_class[tid + stride];
        if (other > top_score[tid] || (other == top_score[tid] && other_c < top_class[tid])) {
          top_score[tid] = other;
          top_class[tid] = other_c;
        }
      }
      workgroupBarrier();
    }
    let label = top_class[0];
    if (label != previous && label != params.blank) {
      if (tid == 0u) { tokens[row * params.T + count] = i32(label); }
      count += 1u;
    }
    previous = label;
    workgroupBarrier();
  }
  if (tid == 0u) { lengths[row] = i32(count); }
  for (var i = count + tid; i < params.T; i += WORKGROUP_SIZE) {
    tokens[row * params.T + i] = PAD;
  }
}
"#;

#[test]
fn ctc_decode_is_refused_and_names_both_outputs() {
    let reason = refusal(CTC_DECODE);
    assert!(
        reason.contains("'tokens'") && reason.contains("'lengths'"),
        "the reason should name the outputs it refused on, got: {reason}"
    );
    assert!(
        reason.contains("computed value rather than a copy"),
        "labels and counts are computed, not sliced, got: {reason}"
    );
}

/// `ops/moe/wgsl/router.wgsl`: top-k over expert logits, and the gate weights
/// that go with them. A data-dependent selection: which experts come back
/// depends on the values, and `Split` has no way to say that.
const MOE_ROUTER: &str = r#"
struct Params { T: u32, E: u32, k: u32, normalize: u32 }
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> expert: array<i32>;
@group(0) @binding(2) var<storage, read_write> gate: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= params.T) { return; }
  let row = t * params.E;
  let out_row = t * params.k;
  var row_max = logits[row];
  for (var e = 1u; e < params.E; e += 1u) { row_max = max(row_max, logits[row + e]); }
  var denominator = 0.0;
  for (var e = 0u; e < params.E; e += 1u) { denominator += exp(logits[row + e] - row_max); }
  var have_ceiling = false;
  var ceiling_value = 0.0;
  var ceiling_index = -1;
  var gate_sum = 0.0;
  for (var r = 0u; r < params.k; r += 1u) {
    var best_index = -1;
    var best_value = 0.0;
    for (var e = 0u; e < params.E; e += 1u) {
      let value = logits[row + e];
      let index = i32(e);
      if (have_ceiling && !(value < ceiling_value || (value == ceiling_value && index > ceiling_index))) {
        continue;
      }
      if (best_index < 0 || value > best_value) { best_index = index; best_value = value; }
    }
    if (best_index < 0) {
      expert[out_row + r] = -1;
      gate[out_row + r] = 0.0;
      continue;
    }
    let weight = exp(best_value - row_max) / denominator;
    expert[out_row + r] = best_index;
    gate[out_row + r] = weight;
    gate_sum += weight;
    have_ceiling = true;
    ceiling_value = best_value;
    ceiling_index = best_index;
  }
  if (params.normalize != 0u) {
    for (var r = 0u; r < params.k; r += 1u) {
      gate[out_row + r] = gate[out_row + r] / gate_sum;
    }
  }
}
"#;

#[test]
fn moe_routing_is_refused_and_names_both_outputs() {
    let reason = refusal(MOE_ROUTER);
    assert!(
        reason.contains("'expert'") && reason.contains("'gate'"),
        "the reason should name the outputs it refused on, got: {reason}"
    );
    assert!(
        reason.contains("computed value rather than a copy"),
        "a top-k selection is computed, not sliced, got: {reason}"
    );
}

/// The router divides an `exp` by a running total of `exp`s, which is the
/// softmax recogniser's evidence exactly — it computes a softmax on its way to
/// the top-k. It must still be refused: the outputs it writes are the experts
/// and their gates, not the distribution. The multi-output arm has to settle
/// this before the softmax walk ever runs.
#[test]
fn a_router_that_computes_a_softmax_is_not_reported_as_one() {
    assert!(
        !matches!(
            classify(MOE_ROUTER),
            KernelPattern::Activation {
                op: ActivationOp::Softmax,
                ..
            }
        ),
        "the router's internal softmax was mistaken for the op it implements"
    );
}
