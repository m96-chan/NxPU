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

#[test]
fn conv2d_stablehlo_convolution() {
    let source = common::load_example("conv2d");
    let output = common::compile_wgsl(&source, &StableHloBackend, 1);
    let mlir = common::first_text(&output);
    assert!(mlir.contains("stablehlo.convolution"));
}

#[test]
fn relu_stablehlo_maximum() {
    let source = common::load_example("relu");
    let output = common::compile_wgsl(&source, &StableHloBackend, 1);
    let mlir = common::first_text(&output);
    assert!(mlir.contains("stablehlo.maximum"));
}

#[test]
fn tanh_stablehlo_tanh() {
    let source = common::load_example("tanh_act");
    let output = common::compile_wgsl(&source, &StableHloBackend, 1);
    let mlir = common::first_text(&output);
    assert!(mlir.contains("stablehlo.tanh"));
}

#[test]
fn reduce_sum_stablehlo_reduce() {
    let source = common::load_example("reduce_sum");
    let output = common::compile_wgsl(&source, &StableHloBackend, 1);
    let mlir = common::first_text(&output);
    assert!(mlir.contains("stablehlo.reduce"));
}

#[test]
fn transpose_stablehlo_unknown() {
    // Transpose is now classified as Unknown (no silent fallback — #64).
    let source = common::load_example("transpose");
    let result = common::try_compile_wgsl(&source, &StableHloBackend, 1);
    assert!(result.is_err(), "expected Unsupported error for transpose");
}

#[test]
fn batchnorm_stablehlo_compiles() {
    let source = common::load_example("batchnorm");
    let output = common::compile_wgsl(&source, &StableHloBackend, 1);
    let mlir = common::first_text(&output);
    assert!(mlir.contains("stablehlo"));
}

#[test]
fn maxpool_stablehlo_reduce_window() {
    let source = common::load_example("maxpool");
    let output = common::compile_wgsl(&source, &StableHloBackend, 1);
    let mlir = common::first_text(&output);
    assert!(mlir.contains("stablehlo.reduce_window"));
}

#[test]
fn concat_stablehlo_concatenate() {
    let source = common::load_example("concat");
    let output = common::compile_wgsl(&source, &StableHloBackend, 1);
    let mlir = common::first_text(&output);
    assert!(mlir.contains("stablehlo.concatenate"));
}

#[test]
fn split_stablehlo_slice() {
    let source = common::load_example("split");
    let output = common::compile_wgsl(&source, &StableHloBackend, 1);
    let mlir = common::first_text(&output);
    assert!(mlir.contains("stablehlo.slice"));
}

#[test]
fn attention_stablehlo_dot_general() {
    let source = common::load_example("attention");
    let output = common::compile_wgsl(&source, &StableHloBackend, 1);
    let mlir = common::first_text(&output);
    assert!(mlir.contains("stablehlo.dot_general"));
}

// --- GELU ---

#[test]
fn gelu_stablehlo_compiles() {
    let source = common::load_example("gelu");
    let output = common::compile_wgsl(&source, &StableHloBackend, 1);
    let mlir = common::first_text(&output);
    assert!(
        mlir.contains("stablehlo"),
        "expected stablehlo in MLIR output"
    );
}

// --- LayerNorm ---

#[test]
fn layernorm_stablehlo_compiles() {
    let source = common::load_example("layernorm");
    let output = common::compile_wgsl(&source, &StableHloBackend, 1);
    let mlir = common::first_text(&output);
    assert!(
        mlir.contains("stablehlo"),
        "expected stablehlo in MLIR output"
    );
}

// --- Gather ---

#[test]
fn gather_stablehlo_compiles() {
    let source = common::load_example("gather");
    let output = common::compile_wgsl(&source, &StableHloBackend, 1);
    let mlir = common::first_text(&output);
    assert!(
        mlir.contains("stablehlo"),
        "expected stablehlo in MLIR output"
    );
}

// --- Scatter ---

#[test]
fn scatter_stablehlo_compiles() {
    let source = common::load_example("scatter");
    let output = common::compile_wgsl(&source, &StableHloBackend, 1);
    let mlir = common::first_text(&output);
    assert!(
        mlir.contains("stablehlo"),
        "expected stablehlo in MLIR output"
    );
}

// --- Depthwise Conv ---

#[test]
fn depthwise_conv_stablehlo_compiles() {
    let source = common::load_example("depthwise_conv");
    let output = common::compile_wgsl(&source, &StableHloBackend, 1);
    let mlir = common::first_text(&output);
    assert!(
        mlir.contains("stablehlo"),
        "expected stablehlo in MLIR output"
    );
}

// --- Multi-head Attention ---

#[test]
fn multihead_attention_stablehlo_compiles() {
    let source = common::load_example("multihead_attention");
    let output = common::compile_wgsl(&source, &StableHloBackend, 1);
    let mlir = common::first_text(&output);
    assert!(
        mlir.contains("stablehlo"),
        "expected stablehlo in MLIR output for multihead attention"
    );
}

// --- Causal Attention ---

#[test]
fn causal_attention_stablehlo_compiles() {
    let source = common::load_example("causal_attention");
    let output = common::compile_wgsl(&source, &StableHloBackend, 1);
    let mlir = common::first_text(&output);
    assert!(
        mlir.contains("stablehlo"),
        "expected stablehlo in MLIR output for causal attention"
    );
}
