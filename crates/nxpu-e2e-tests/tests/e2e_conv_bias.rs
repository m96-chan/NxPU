//! A convolution's bias has to reach the emitted model, not just the pattern.
//!
//! Recognising the bias and then dropping it on the way out would be the same
//! failure as mislabelling the operator: a graph that runs and computes
//! something else. These assert the bias by name in the bytes.

use nxpu_backend_core::{Backend, BackendOptions, OutputContent};
use nxpu_backend_onnx::OnnxBackend;
use nxpu_backend_tflite::TfLiteBackend;

/// A direct convolution with a per-output-channel bias, in the shape
/// web-xpu-ops writes one: the kernel window named in the params, and the
/// bias added once per output element.
const CONV_WITH_BIAS: &str = "
struct Params { Cin: u32, Cout: u32, H: u32, W: u32, KH: u32, KW: u32, Hout: u32, Wout: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.Cout * params.Hout * params.Wout) { return; }
  let oc = idx / (params.Hout * params.Wout);
  var acc = 0.0;
  for (var c = 0u; c < params.Cin; c += 1u) {
    for (var kh = 0u; kh < params.KH; kh += 1u) {
      for (var kw = 0u; kw < params.KW; kw += 1u) {
        acc = acc + input[c * params.H * params.W + kh * params.W + kw]
                  * weight[oc * params.Cin * params.KH * params.KW + c * params.KH * params.KW + kh * params.KW + kw];
      }
    }
  }
  output[idx] = acc + bias[oc];
}";

fn compile(backend: &dyn Backend, source: &str) -> Vec<u8> {
    let mut module = nxpu_parser::parse(source).expect("WGSL parse failed");
    nxpu_opt::PassManager::for_level(nxpu_opt::OptLevel::O1).run(&mut module);
    let output = backend
        .compile(&module, &BackendOptions::default())
        .expect("backend compilation failed");
    match &output.files[0].content {
        OutputContent::Binary(b) => b.clone(),
        OutputContent::Text(t) => t.clone().into_bytes(),
    }
}

fn contains(haystack: &[u8], needle: &str) -> bool {
    haystack
        .windows(needle.len())
        .any(|w| w == needle.as_bytes())
}

#[test]
fn tflite_carries_the_bias() {
    let bytes = compile(&TfLiteBackend, CONV_WITH_BIAS);
    assert_eq!(&bytes[4..8], b"TFL3");
    assert!(
        contains(&bytes, "bias"),
        "the bias tensor is not in the model"
    );
    assert!(contains(&bytes, "weight"));
}

#[test]
fn onnx_carries_the_bias() {
    let bytes = compile(&OnnxBackend, CONV_WITH_BIAS);
    assert!(contains(&bytes, "Conv"), "not a Conv node");
    // ONNX Conv takes B as its third input, and a node input the graph does
    // not declare makes the model invalid — so the name has to appear twice.
    let occurrences = bytes.windows(4).filter(|w| *w == b"bias").count();
    assert!(
        occurrences >= 2,
        "expected the bias on the node and in the graph inputs, found {occurrences} mention(s)"
    );
}

#[test]
fn a_convolution_without_a_bias_gets_one() {
    // TFLite's CONV_2D kernel requires three inputs — `has_bias was not true`
    // is a hard failure — so a convolution whose source has no bias is given
    // one here, as a constant of zeros. The model is otherwise unloadable.
    let source = CONV_WITH_BIAS
        .replace(
            "@group(0) @binding(2) var<storage, read> bias: array<f32>;\n",
            "",
        )
        .replace("acc + bias[oc]", "acc")
        .replace(
            "@binding(3) var<storage, read_write>",
            "@binding(2) var<storage, read_write>",
        )
        .replace("@binding(4) var<uniform>", "@binding(3) var<uniform>");
    let bytes = compile(&TfLiteBackend, &source);
    assert_eq!(&bytes[4..8], b"TFL3");
    assert!(
        contains(&bytes, "bias"),
        "a synthesised bias tensor is required for the model to load"
    );
}
