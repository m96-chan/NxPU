mod common;

use nxpu_backend_tflite::TfLiteBackend;

#[test]
fn matmul_tflite_magic() {
    let source = common::load_example("matmul");
    let output = common::compile_wgsl(&source, &TfLiteBackend, 1);
    let bytes = common::first_binary(&output);
    assert!(bytes.len() > 8);
    assert_eq!(&bytes[4..8], b"TFL3");
}

#[test]
fn vecadd_tflite_magic() {
    let source = common::load_example("vecadd");
    let output = common::compile_wgsl(&source, &TfLiteBackend, 1);
    let bytes = common::first_binary(&output);
    assert!(bytes.len() > 8);
    assert_eq!(&bytes[4..8], b"TFL3");
}

#[test]
fn vecsub_tflite_magic() {
    let source = common::load_example("vecsub");
    let output = common::compile_wgsl(&source, &TfLiteBackend, 1);
    let bytes = common::first_binary(&output);
    assert_eq!(&bytes[4..8], b"TFL3");
}

#[test]
fn vecmul_tflite_magic() {
    let source = common::load_example("vecmul");
    let output = common::compile_wgsl(&source, &TfLiteBackend, 1);
    let bytes = common::first_binary(&output);
    assert_eq!(&bytes[4..8], b"TFL3");
}

#[test]
fn matmul_all_opt_levels() {
    let source = common::load_example("matmul");
    for level in [0, 1, 2] {
        let output = common::compile_wgsl(&source, &TfLiteBackend, level);
        let bytes = common::first_binary(&output);
        assert_eq!(&bytes[4..8], b"TFL3", "failed at opt level {level}");
    }
}

#[test]
fn conv2d_tflite_magic() {
    let source = common::load_example("conv2d");
    let output = common::compile_wgsl(&source, &TfLiteBackend, 1);
    let bytes = common::first_binary(&output);
    assert_eq!(&bytes[4..8], b"TFL3");
}

#[test]
fn relu_tflite_magic() {
    let source = common::load_example("relu");
    let output = common::compile_wgsl(&source, &TfLiteBackend, 1);
    let bytes = common::first_binary(&output);
    assert_eq!(&bytes[4..8], b"TFL3");
}

#[test]
fn tanh_tflite_magic() {
    let source = common::load_example("tanh_act");
    let output = common::compile_wgsl(&source, &TfLiteBackend, 1);
    let bytes = common::first_binary(&output);
    assert_eq!(&bytes[4..8], b"TFL3");
}

#[test]
fn reduce_sum_tflite_magic() {
    let source = common::load_example("reduce_sum");
    let output = common::compile_wgsl(&source, &TfLiteBackend, 1);
    let bytes = common::first_binary(&output);
    assert_eq!(&bytes[4..8], b"TFL3");
}

#[test]
fn transpose_tflite_unknown() {
    // Transpose is now classified as Unknown (no silent fallback — #64).
    let source = common::load_example("transpose");
    let result = common::try_compile_wgsl(&source, &TfLiteBackend, 1);
    assert!(result.is_err(), "expected Unsupported error for transpose");
}

#[test]
fn batchnorm_tflite_compiles() {
    let source = common::load_example("batchnorm");
    let output = common::compile_wgsl(&source, &TfLiteBackend, 1);
    let bytes = common::first_binary(&output);
    assert_eq!(&bytes[4..8], b"TFL3");
}

#[test]
fn maxpool_tflite_magic() {
    let source = common::load_example("maxpool");
    let output = common::compile_wgsl(&source, &TfLiteBackend, 1);
    let bytes = common::first_binary(&output);
    assert_eq!(&bytes[4..8], b"TFL3");
}

#[test]
fn concat_tflite_magic() {
    let source = common::load_example("concat");
    let output = common::compile_wgsl(&source, &TfLiteBackend, 1);
    let bytes = common::first_binary(&output);
    assert_eq!(&bytes[4..8], b"TFL3");
}

#[test]
fn split_tflite_magic() {
    let source = common::load_example("split");
    let output = common::compile_wgsl(&source, &TfLiteBackend, 1);
    let bytes = common::first_binary(&output);
    assert_eq!(&bytes[4..8], b"TFL3");
}

#[test]
fn attention_tflite_magic() {
    let source = common::load_example("attention");
    let output = common::compile_wgsl(&source, &TfLiteBackend, 1);
    let bytes = common::first_binary(&output);
    assert_eq!(&bytes[4..8], b"TFL3");
}

// --- GELU ---

#[test]
fn gelu_tflite_magic() {
    let source = common::load_example("gelu");
    let output = common::compile_wgsl(&source, &TfLiteBackend, 1);
    let bytes = common::first_binary(&output);
    assert!(bytes.len() > 8);
    assert_eq!(&bytes[4..8], b"TFL3");
}

// --- LayerNorm ---

#[test]
fn layernorm_tflite_magic() {
    let source = common::load_example("layernorm");
    let output = common::compile_wgsl(&source, &TfLiteBackend, 1);
    let bytes = common::first_binary(&output);
    assert!(bytes.len() > 8);
    assert_eq!(&bytes[4..8], b"TFL3");
}

// --- Gather ---

#[test]
fn gather_tflite_magic() {
    let source = common::load_example("gather");
    let output = common::compile_wgsl(&source, &TfLiteBackend, 1);
    let bytes = common::first_binary(&output);
    assert!(bytes.len() > 8);
    assert_eq!(&bytes[4..8], b"TFL3");
}

// --- Scatter ---

#[test]
fn scatter_tflite_magic() {
    let source = common::load_example("scatter");
    let output = common::compile_wgsl(&source, &TfLiteBackend, 1);
    let bytes = common::first_binary(&output);
    assert!(bytes.len() > 8);
    assert_eq!(&bytes[4..8], b"TFL3");
}

// --- Depthwise Conv ---

#[test]
fn depthwise_conv_tflite_magic() {
    let source = common::load_example("depthwise_conv");
    let output = common::compile_wgsl(&source, &TfLiteBackend, 1);
    let bytes = common::first_binary(&output);
    assert!(bytes.len() > 8);
    assert_eq!(&bytes[4..8], b"TFL3");
}

// --- Multi-head Attention ---
// TFLite backend does not yet support multi-head splitting (num_heads is ignored),
// but compilation should succeed producing single-head SDPA output.

#[test]
fn multihead_attention_tflite_compiles() {
    let source = common::load_example("multihead_attention");
    let output = common::compile_wgsl(&source, &TfLiteBackend, 1);
    let bytes = common::first_binary(&output);
    assert!(bytes.len() > 8);
    assert_eq!(&bytes[4..8], b"TFL3");
}

// --- Causal Attention ---
// TFLite backend does not yet support causal masking (causal flag is ignored),
// but compilation should succeed producing unmasked SDPA output.

#[test]
fn causal_attention_tflite_compiles() {
    let source = common::load_example("causal_attention");
    let output = common::compile_wgsl(&source, &TfLiteBackend, 1);
    let bytes = common::first_binary(&output);
    assert!(bytes.len() > 8);
    assert_eq!(&bytes[4..8], b"TFL3");
}

// --- Loadability ---
//
// Every test above asserts the file begins with `TFL3` and stops there, which
// is why nothing noticed that no model this backend produced could be loaded
// at all: `Tensor.shape` carried `-1` for symbolic dimensions, and TFLite
// overflows `BytesRequired` on a negative extent rather than treating it as
// unknown. A phone reported it, after the models had been shipped for months:
//
//   Cannot create interpreter: BytesRequired number of bytes overflowed.
//   Tensor 0 is invalidly specified in schema.
//
// Scanning the buffer for the word is crude next to parsing the FlatBuffer,
// but it fails for exactly the reason the device failed, and it covers every
// pattern rather than the handful anyone would write a reader for.

fn has_negative_dim_word(bytes: &[u8]) -> bool {
    bytes.windows(4).any(|w| w == [0xff, 0xff, 0xff, 0xff])
}

#[test]
fn no_example_carries_a_negative_extent() {
    // The same set the tests above compile, so this asserts loadability of
    // exactly what is already asserted to be well-formed.
    let examples = [
        "attention",
        "batchnorm",
        "causal_attention",
        "concat",
        "conv2d",
        "depthwise_conv",
        "gather",
        "gelu",
        "layernorm",
        "matmul",
        "maxpool",
        "multihead_attention",
        "reduce_sum",
        "relu",
        "scatter",
        "split",
        "tanh_act",
        "vecadd",
        "vecmul",
        "vecsub",
    ];
    // transpose is deliberately absent: it classifies as Unknown and is
    // expected not to compile at all (see transpose_tflite_unknown).
    let offenders: Vec<&str> = examples
        .into_iter()
        .filter(|name| {
            let source = common::load_example(name);
            let output = common::compile_wgsl(&source, &TfLiteBackend, 1);
            has_negative_dim_word(common::first_binary(&output))
        })
        .collect();
    assert!(
        offenders.is_empty(),
        "these models cannot be loaded by any interpreter: {offenders:?}"
    );
}
