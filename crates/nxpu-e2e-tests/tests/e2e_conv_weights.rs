//! A convolution's filter has to be a constant, or no NPU driver will take it.
//!
//! Measured on a MediaTek MT6899: `mtk-neuron_shim` accelerates a convolution
//! whose filter is a compile-time constant and refuses the identical
//! convolution, at the same shapes, whose filter is a graph input. TFLite's own
//! GPU delegate takes it either way. A WGSL kernel has no filter contents to
//! give — the host binds them per dispatch — so the caller supplies them.
//!
//! What must never happen is filling the filter with zeros to win the
//! acceleration. The model would run, be attributed to the accelerator, and
//! compute the wrong thing.

use nxpu_backend_core::{Backend, BackendOptions, ConstantTensor, DiagnosticLevel, OutputContent};
use nxpu_backend_tflite::TfLiteBackend;
use nxpu_opt::{OptLevel, PassManager};

const CONV: &str = include_str!("../../../examples/conv2d_3x3.wgsl");

/// `[16, 3, 3, 16]` of f32 — the filter this kernel has at `--symbolic-dim 16`.
fn filter_bytes(count: usize) -> Vec<u8> {
    (0..count)
        .flat_map(|i| (i as f32 * 0.001).to_le_bytes())
        .collect()
}

fn compile(weights: Vec<ConstantTensor>) -> Result<nxpu_backend_core::BackendOutput, String> {
    let mut module = nxpu_parser::parse(CONV).expect("WGSL parse failed");
    PassManager::for_level(OptLevel::O1).run(&mut module);
    let opts = BackendOptions {
        symbolic_extent: Some(16),
        constant_tensors: weights,
        ..BackendOptions::default()
    };
    TfLiteBackend
        .compile(&module, &opts)
        .map_err(|e| format!("{e}"))
}

fn bytes_of(output: &nxpu_backend_core::BackendOutput) -> Vec<u8> {
    match &output.files[0].content {
        OutputContent::Binary(b) => b.clone(),
        OutputContent::Text(t) => t.clone().into_bytes(),
    }
}

#[test]
fn a_supplied_filter_reaches_the_model() {
    let data = filter_bytes(16 * 3 * 3 * 16);
    let output = compile(vec![ConstantTensor {
        name: "weight".into(),
        data: data.clone(),
    }])
    .expect("compilation failed");
    let bytes = bytes_of(&output);
    // The contents are in the file, so the filter is a constant and not an
    // empty buffer the runtime is expected to fill.
    assert!(
        bytes.windows(data.len()).any(|w| w == data.as_slice()),
        "the supplied filter is not in the emitted model"
    );
}

#[test]
fn a_filter_of_the_wrong_size_is_an_error_and_not_a_truncation() {
    let error = compile(vec![ConstantTensor {
        name: "weight".into(),
        data: filter_bytes(10),
    }])
    .expect_err("a short filter was accepted");
    // Both counts, so the reader can see which of the two is wrong rather than
    // being told only that they differ.
    assert!(error.contains("40 bytes supplied"), "got: {error}");
    assert!(error.contains("9216 wanted"), "got: {error}");
}

#[test]
fn a_runtime_filter_says_what_it_costs() {
    // Silent non-acceleration is what let this sit undetected: a driver reports
    // a refusal as `unsupported`, which reads as a statement about CONV_2D and
    // is a statement about this model.
    let output = compile(Vec::new()).expect("compilation failed");
    let warning = output
        .diagnostics
        .iter()
        .find(|d| matches!(d.level, DiagnosticLevel::Warning))
        .expect("no warning about the runtime filter");
    assert!(warning.message.contains("weight"), "{}", warning.message);
    assert!(warning.message.contains("NNAPI"), "{}", warning.message);
    assert!(warning.message.contains("--weights"), "{}", warning.message);
}

#[test]
fn supplying_it_silences_the_warning() {
    let output = compile(vec![ConstantTensor {
        name: "weight".into(),
        data: filter_bytes(16 * 3 * 3 * 16),
    }])
    .expect("compilation failed");
    assert!(
        !output
            .diagnostics
            .iter()
            .any(|d| d.message.contains("--weights")),
        "the warning survived the fix it asks for"
    );
}
