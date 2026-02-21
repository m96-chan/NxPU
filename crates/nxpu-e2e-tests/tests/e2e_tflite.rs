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
