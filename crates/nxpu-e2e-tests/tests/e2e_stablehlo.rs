mod common;

use nxpu_backend_stablehlo::StableHloBackend;

#[test]
fn matmul_stablehlo_dot_general() {
    let source = common::load_example("matmul");
    let output = common::compile_wgsl(&source, &StableHloBackend, 1);
    let mlir = common::first_text(&output);
    assert!(mlir.contains("stablehlo.dot_general"));
    assert!(mlir.contains("module @"));
}

#[test]
fn vecadd_stablehlo_add() {
    let source = common::load_example("vecadd");
    let output = common::compile_wgsl(&source, &StableHloBackend, 1);
    let mlir = common::first_text(&output);
    assert!(mlir.contains("stablehlo.add"));
}

#[test]
fn vecsub_stablehlo_subtract() {
    let source = common::load_example("vecsub");
    let output = common::compile_wgsl(&source, &StableHloBackend, 1);
    let mlir = common::first_text(&output);
    assert!(mlir.contains("stablehlo.subtract"));
}

#[test]
fn vecmul_stablehlo_multiply() {
    let source = common::load_example("vecmul");
    let output = common::compile_wgsl(&source, &StableHloBackend, 1);
    let mlir = common::first_text(&output);
    assert!(mlir.contains("stablehlo.multiply"));
}

#[test]
fn matmul_all_opt_levels() {
    let source = common::load_example("matmul");
    for level in [0, 1, 2] {
        let output = common::compile_wgsl(&source, &StableHloBackend, level);
        let mlir = common::first_text(&output);
        assert!(
            mlir.contains("stablehlo.dot_general"),
            "failed at opt level {level}"
        );
    }
}
