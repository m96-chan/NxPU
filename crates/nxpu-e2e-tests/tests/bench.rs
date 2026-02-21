mod common;

use std::time::Instant;

use nxpu_backend_coreml::CoreMlBackend;
use nxpu_backend_onnx::OnnxBackend;
use nxpu_backend_stablehlo::StableHloBackend;
use nxpu_backend_tflite::TfLiteBackend;

fn bench_backend(name: &str, source: &str, backend: &dyn nxpu_backend_core::Backend) {
    let start = Instant::now();
    let output = common::compile_wgsl(source, backend, 1);
    let elapsed = start.elapsed();

    let size: usize = output
        .files
        .iter()
        .map(|f| match &f.content {
            nxpu_backend_core::OutputContent::Binary(b) => b.len(),
            nxpu_backend_core::OutputContent::Text(t) => t.len(),
        })
        .sum();

    eprintln!("{name}: {elapsed:?}, output size: {size} bytes");
    assert!(elapsed.as_secs() < 1, "{name} took too long: {elapsed:?}");
}

#[test]
fn bench_onnx_matmul() {
    let source = common::load_example("matmul");
    bench_backend("ONNX/matmul", &source, &OnnxBackend);
}

#[test]
fn bench_onnx_vecadd() {
    let source = common::load_example("vecadd");
    bench_backend("ONNX/vecadd", &source, &OnnxBackend);
}

#[test]
fn bench_tflite_matmul() {
    let source = common::load_example("matmul");
    bench_backend("TFLite/matmul", &source, &TfLiteBackend);
}

#[test]
fn bench_tflite_vecadd() {
    let source = common::load_example("vecadd");
    bench_backend("TFLite/vecadd", &source, &TfLiteBackend);
}

#[test]
fn bench_coreml_matmul() {
    let source = common::load_example("matmul");
    bench_backend("CoreML/matmul", &source, &CoreMlBackend);
}

#[test]
fn bench_coreml_vecadd() {
    let source = common::load_example("vecadd");
    bench_backend("CoreML/vecadd", &source, &CoreMlBackend);
}

#[test]
fn bench_stablehlo_matmul() {
    let source = common::load_example("matmul");
    bench_backend("StableHLO/matmul", &source, &StableHloBackend);
}

#[test]
fn bench_stablehlo_vecadd() {
    let source = common::load_example("vecadd");
    bench_backend("StableHLO/vecadd", &source, &StableHloBackend);
}
