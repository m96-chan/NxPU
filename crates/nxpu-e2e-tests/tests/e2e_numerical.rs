//! Numerical correctness tests for ONNX output.
//!
//! Compiles WGSL → ONNX, loads into tract, runs inference, and checks results.

mod common;

use ndarray::Array2;
use nxpu_backend_core::OutputContent;
use nxpu_backend_onnx::OnnxBackend;
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::internal::DimLike;

/// Compile WGSL to ONNX bytes, load in tract, run with given inputs, return output tensor.
fn run_onnx_1d(wgsl_source: &str, inputs: Vec<Vec<f32>>) -> Vec<f32> {
    let output = common::compile_wgsl(wgsl_source, &OnnxBackend, 1);
    let bytes = match &output.files[0].content {
        OutputContent::Binary(b) => b.clone(),
        _ => panic!("expected binary ONNX output"),
    };

    let model = onnx()
        .model_for_read(&mut std::io::Cursor::new(bytes))
        .expect("failed to load ONNX model in tract");

    // Fix all symbolic dimensions to the actual input sizes.
    let n = inputs[0].len();
    let mut model = model.into_typed().expect("failed to type model");
    for i in 0..model.inputs.len() {
        let input_id = model.inputs[i];
        let fact = model.outlet_fact(input_id).unwrap().clone();
        let shape: Vec<usize> = fact
            .shape
            .iter()
            .map(|d| d.to_usize().unwrap_or(n))
            .collect();
        model
            .set_input_fact(i, f32::fact(shape))
            .expect("failed to set input fact");
    }

    let model = model.into_optimized().expect("failed to optimize model");
    let model = model
        .into_runnable()
        .expect("failed to make runnable model");

    let tract_inputs: Vec<TValue> = inputs
        .into_iter()
        .map(|v| {
            let len = v.len();
            tract_ndarray::Array1::from_vec(v)
                .into_shape_with_order(vec![len])
                .unwrap()
                .into_tensor()
                .into()
        })
        .collect();

    let result = model.run(tract_inputs.into()).expect("inference failed");
    let output_tensor = result[0]
        .to_array_view::<f32>()
        .expect("failed to get f32 output");
    output_tensor.iter().copied().collect()
}

/// Compile WGSL to ONNX bytes, load in tract, run 2D matmul.
fn run_onnx_matmul(wgsl_source: &str, a: Array2<f32>, b: Array2<f32>) -> Array2<f32> {
    let output = common::compile_wgsl(wgsl_source, &OnnxBackend, 1);
    let bytes = match &output.files[0].content {
        OutputContent::Binary(b) => b.clone(),
        _ => panic!("expected binary ONNX output"),
    };

    let model = onnx()
        .model_for_read(&mut std::io::Cursor::new(bytes))
        .expect("failed to load ONNX model in tract");

    let (m, k) = (a.nrows(), a.ncols());
    let (_, n) = (b.nrows(), b.ncols());

    let mut model = model.into_typed().expect("failed to type model");
    model
        .set_input_fact(0, f32::fact([m, k]))
        .expect("set A shape");
    model
        .set_input_fact(1, f32::fact([k, n]))
        .expect("set B shape");

    let model = model.into_optimized().expect("failed to optimize");
    let model = model.into_runnable().expect("failed to make runnable");

    let a_tensor: TValue = a.into_tensor().into();
    let b_tensor: TValue = b.into_tensor().into();

    let result = model
        .run(tvec![a_tensor, b_tensor])
        .expect("matmul inference failed");
    let out = result[0]
        .to_array_view::<f32>()
        .expect("failed to get f32 output");
    out.to_owned()
        .into_shape_with_order((m, n))
        .expect("reshape")
}

/// Generic helper: compile WGSL → ONNX → tract, run with N-dimensional shaped inputs.
///
/// Sets concrete input shapes on the inference model *before* typing so that
/// complex ops (Conv, MaxPool, etc.) can resolve symbolic dimensions.
fn run_onnx(wgsl_source: &str, inputs: Vec<(Vec<f32>, Vec<usize>)>) -> Vec<f32> {
    let output = common::compile_wgsl(wgsl_source, &OnnxBackend, 1);
    let bytes = match &output.files[0].content {
        OutputContent::Binary(b) => b.clone(),
        _ => panic!("expected binary ONNX output"),
    };

    let mut model = onnx()
        .model_for_read(&mut std::io::Cursor::new(bytes))
        .expect("failed to load ONNX model in tract");

    // Set concrete shapes on the inference model before typing.
    for (i, (_, shape)) in inputs.iter().enumerate() {
        let fact = InferenceFact::dt_shape(
            f32::datum_type(),
            shape
                .iter()
                .map(|&d| (d as i64).to_dim())
                .collect::<TVec<_>>(),
        );
        model
            .set_input_fact(i, fact)
            .expect("failed to set input fact");
    }

    let model = model
        .into_typed()
        .expect("failed to type model")
        .into_optimized()
        .expect("failed to optimize model")
        .into_runnable()
        .expect("failed to make runnable model");

    let tract_inputs: Vec<TValue> = inputs
        .into_iter()
        .map(|(data, shape)| {
            tract_ndarray::ArrayD::from_shape_vec(tract_ndarray::IxDyn(&shape), data)
                .unwrap()
                .into_tensor()
                .into()
        })
        .collect();

    let result = model.run(tract_inputs.into()).expect("inference failed");
    result[0]
        .to_array_view::<f32>()
        .expect("failed to get f32 output")
        .iter()
        .copied()
        .collect()
}

const TOL: f32 = 1e-5;

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < TOL,
            "mismatch at index {i}: got {a}, expected {e}"
        );
    }
}

// --- VecAdd ---

#[test]
fn vecadd_numerical() {
    let source = common::load_example("vecadd");
    let result = run_onnx_1d(&source, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    assert_close(&result, &[5.0, 7.0, 9.0]);
}

// --- VecSub ---

#[test]
fn vecsub_numerical() {
    let source = common::load_example("vecsub");
    let result = run_onnx_1d(&source, vec![vec![5.0, 8.0, 3.0], vec![1.0, 2.0, 1.0]]);
    assert_close(&result, &[4.0, 6.0, 2.0]);
}

// --- VecMul ---

#[test]
fn vecmul_numerical() {
    let source = common::load_example("vecmul");
    let result = run_onnx_1d(&source, vec![vec![2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0]]);
    assert_close(&result, &[10.0, 18.0, 28.0]);
}

// --- MatMul ---

#[test]
fn matmul_numerical() {
    let source = common::load_example("matmul");
    let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Array2::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let c = run_onnx_matmul(&source, a, b);

    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    let expected = Array2::from_shape_vec((2, 2), vec![19.0, 22.0, 43.0, 50.0]).unwrap();
    for (actual, exp) in c.iter().zip(expected.iter()) {
        assert!(
            (actual - exp).abs() < TOL,
            "matmul mismatch: got {actual}, expected {exp}"
        );
    }
}

// --- ReLU ---

#[test]
fn relu_numerical() {
    let source = common::load_example("relu");
    let result = run_onnx_1d(&source, vec![vec![-1.0, 0.0, 1.0, -0.5, 2.0]]);
    assert_close(&result, &[0.0, 0.0, 1.0, 0.0, 2.0]);
}

// --- Tanh ---

#[test]
fn tanh_numerical() {
    let source = common::load_example("tanh_act");
    let input = vec![-1.0_f32, 0.0, 1.0, 2.0];
    let expected: Vec<f32> = input.iter().map(|x| x.tanh()).collect();
    let result = run_onnx_1d(&source, vec![input]);
    assert_close(&result, &expected);
}

// --- ReduceSum ---

#[test]
#[ignore = "ReduceSum axes emitted as attribute but opset 13 requires it as an input; tract reduces all dims"]
fn reduce_sum_numerical() {
    let source = common::load_example("reduce_sum");
    // 3×3 matrix; reduce_sum over axis=1 (row sums)
    // [[1,2,3],[4,5,6],[7,8,9]] → [6, 15, 24]
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let result = run_onnx(&source, vec![(input, vec![3, 3])]);
    assert_close(&result, &[6.0, 15.0, 24.0]);
}

// --- Transpose ---

#[test]
#[ignore = "ONNX backend does not recognize the transpose WGSL pattern yet"]
fn transpose_numerical() {
    let source = common::load_example("transpose");
    // 2×3 matrix transposed → 3×2
    // [[1,2,3],[4,5,6]] → [[1,4],[2,5],[3,6]]
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = run_onnx(&source, vec![(input, vec![2, 3])]);
    assert_close(&result, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

// --- Conv2D ---

#[test]
fn conv2d_numerical() {
    let source = common::load_example("conv2d");
    // Input: 1×1×5×5, Weight: 1×1×3×3 (center-only kernel)
    // The lowering hardcodes kernel_shape=[3,3], so we must use a 3×3 kernel.
    // Valid convolution (no padding, stride 1) → 1×1×3×3 output
    let input: Vec<f32> = (1..=25).map(|x| x as f32).collect();
    #[rustfmt::skip]
    let weight = vec![
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0,
    ];
    let result = run_onnx(
        &source,
        vec![(input, vec![1, 1, 5, 5]), (weight, vec![1, 1, 3, 3])],
    );
    // Center-only kernel picks the center element of each 3×3 patch:
    // row0: 7, 8, 9  |  row1: 12, 13, 14  |  row2: 17, 18, 19
    assert_close(
        &result,
        &[7.0, 8.0, 9.0, 12.0, 13.0, 14.0, 17.0, 18.0, 19.0],
    );
}

// --- MaxPool ---

#[test]
fn maxpool_numerical() {
    let source = common::load_example("maxpool");
    // Input: 1×1×4×4 (values 1..16), MaxPool 2×2 stride 2 → 1×1×2×2
    let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let result = run_onnx(&source, vec![(input, vec![1, 1, 4, 4])]);
    // max(1,2,5,6)=6, max(3,4,7,8)=8, max(9,10,13,14)=14, max(11,12,15,16)=16
    assert_close(&result, &[6.0, 8.0, 14.0, 16.0]);
}

// --- BatchNorm ---

#[test]
#[ignore = "ONNX graph references running_mean/running_var without defining them as inputs or initializers"]
fn batchnorm_numerical() {
    let source = common::load_example("batchnorm");
    // X: [1,2,1,2], gamma: [2], beta: [2]
    // Channel 0: [1,2] mean=1.5 var=0.25
    // Channel 1: [3,4] mean=3.5 var=0.25
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0, 1.0];
    let beta = vec![0.0, 0.0];
    let result = run_onnx(
        &source,
        vec![(x, vec![1, 2, 1, 2]), (gamma, vec![2]), (beta, vec![2])],
    );
    // y = gamma * (x - mean) / sqrt(var + 1e-5) + beta
    let s = (0.25_f32 + 1e-5).sqrt();
    let expected = vec![-0.5 / s, 0.5 / s, -0.5 / s, 0.5 / s];
    assert_close(&result, &expected);
}

// --- Attention ---

#[test]
fn attention_numerical() {
    let source = common::load_example("attention");
    // Q=K=V: 2×2 identity matrix; the lowering bakes sqrt_dk = sqrt(64) = 8.0
    let eye = vec![1.0, 0.0, 0.0, 1.0];
    let result = run_onnx(
        &source,
        vec![
            (eye.clone(), vec![2, 2]),
            (eye.clone(), vec![2, 2]),
            (eye, vec![2, 2]),
        ],
    );
    // scores = Q·K^T / 8 = I / 8 = [[0.125, 0], [0, 0.125]]
    // softmax row0: [e^0.125/(e^0.125+1), 1/(e^0.125+1)]
    let e = (0.125_f32).exp();
    let s = e + 1.0;
    let w_hi = e / s;
    let w_lo = 1.0 / s;
    // output = attn_weights · V(=I) = attn_weights
    let expected = vec![w_hi, w_lo, w_lo, w_hi];
    assert_close(&result, &expected);
}
