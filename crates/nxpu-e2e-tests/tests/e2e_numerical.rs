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
