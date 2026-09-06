//! A convolution that ends in an activation is both, and must emit as both.
//!
//! The classifier returned `Conv2D` for a kernel storing `max(sum, 0.0)` and
//! dropped the `max`. The emitted model then held a CONV_2D and nothing else:
//! it loaded, a MediaTek driver accelerated it, and it returned unclipped
//! values. Nothing in the suite could see it, because the assertions were that
//! the file begins with `TFL3` and that the operator matrix cell said
//! `accelerated` — and both were true of the wrong graph.

use nxpu_analysis::analyze::{self, ActivationOp, KernelPattern};
use nxpu_backend_core::{Backend, BackendOptions, OutputContent};
use nxpu_backend_onnx::OnnxBackend;
use nxpu_backend_tflite::TfLiteBackend;
use nxpu_opt::{OptLevel, PassManager};

const CONV_RELU: &str = include_str!("../../../examples/conv2d_relu.wgsl");
const CONV_PLAIN: &str = include_str!("../../../examples/conv2d_3x3.wgsl");

fn classify(source: &str, level: OptLevel) -> KernelPattern {
    let mut module = nxpu_parser::parse(source).expect("WGSL parse failed");
    PassManager::for_level(level).run(&mut module);
    analyze::classify_entry_point(&module, 0).expect("classification failed")
}

#[test]
fn a_convolution_keeps_the_activation_it_ends_with() {
    // At every level: an optimizer that rewrites the accumulation must not be
    // able to make the activation disappear, which is how the opt-level
    // invariance bugs in this classifier have gone before.
    for level in [OptLevel::O0, OptLevel::O1, OptLevel::O2] {
        match classify(CONV_RELU, level) {
            KernelPattern::Conv2D { activation, .. } => assert_eq!(
                activation,
                Some(ActivationOp::Relu),
                "the max(sum, 0.0) was dropped at {level:?}"
            ),
            other => panic!("expected Conv2D at {level:?}, got {other:?}"),
        }
    }
}

#[test]
fn a_convolution_that_ends_with_nothing_reports_nothing() {
    // The other half of the claim: this must not start seeing activations that
    // are not there. A convolution's own accumulation is a multiply, and the
    // multiply-shaped activations are exactly why only `max` and `tanh` are
    // recognised.
    match classify(CONV_PLAIN, OptLevel::O1) {
        KernelPattern::Conv2D { activation, .. } => assert_eq!(activation, None),
        other => panic!("expected Conv2D, got {other:?}"),
    }
}

#[test]
fn tflite_folds_it_into_the_convolution_rather_than_appending_one() {
    // One operator, not two. A second would be excluded from the operator
    // matrix — which cannot attribute a refusal across two operators — and
    // would pay a partition boundary on a device that routes them apart.
    let mut module = nxpu_parser::parse(CONV_RELU).expect("WGSL parse failed");
    PassManager::for_level(OptLevel::O1).run(&mut module);
    let output = TfLiteBackend
        .compile(&module, &BackendOptions::default())
        .expect("TFLite compilation failed");
    let bytes = match &output.files[0].content {
        OutputContent::Binary(b) => b.clone(),
        OutputContent::Text(t) => t.clone().into_bytes(),
    };
    assert_eq!(&bytes[4..8], b"TFL3");
    // RELU is 1 in `ActivationFunctionType`, and `push_slot` writes the field
    // only because it differs from NONE. A model that dropped the activation
    // would not carry the byte at all.
    let plain = {
        let mut m = nxpu_parser::parse(CONV_PLAIN).expect("WGSL parse failed");
        PassManager::for_level(OptLevel::O1).run(&mut m);
        let o = TfLiteBackend
            .compile(&m, &BackendOptions::default())
            .expect("TFLite compilation failed");
        match &o.files[0].content {
            OutputContent::Binary(b) => b.clone(),
            OutputContent::Text(t) => t.clone().into_bytes(),
        }
    };
    assert_ne!(
        bytes, plain,
        "the model with a ReLU is byte-identical to the one without it"
    );
}

#[test]
fn onnx_refuses_rather_than_emitting_the_convolution_alone() {
    // ONNX has a Relu node and this lowering does not append one. Refusing by
    // name is worth more than a graph that runs and computes something else.
    let mut module = nxpu_parser::parse(CONV_RELU).expect("WGSL parse failed");
    PassManager::for_level(OptLevel::O1).run(&mut module);
    let error = OnnxBackend
        .compile(&module, &BackendOptions::default())
        .expect_err("ONNX emitted a convolution whose activation it cannot express");
    let message = format!("{error}");
    assert!(
        message.contains("Relu"),
        "the refusal does not name the activation: {message}"
    );
}
