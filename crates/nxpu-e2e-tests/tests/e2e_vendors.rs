mod common;

use nxpu_backend_amd::AmdBackend;
use nxpu_backend_arm_ethos::ArmEthosBackend;
use nxpu_backend_ceva::CevaBackend;
use nxpu_backend_coreml::CoreMlBackend;
use nxpu_backend_intel::IntelBackend;
use nxpu_backend_mediatek::MediaTekBackend;
use nxpu_backend_onnx::OnnxBackend;
use nxpu_backend_qualcomm::QualcommBackend;
use nxpu_backend_rockchip::RockchipBackend;
use nxpu_backend_samsung::SamsungBackend;
use nxpu_backend_stablehlo::StableHloBackend;
use nxpu_backend_tflite::TfLiteBackend;

macro_rules! vendor_test {
    ($name:ident, $backend:expr, $example:expr) => {
        #[test]
        fn $name() {
            let source = common::load_example($example);
            let output = common::compile_wgsl(&source, &$backend, 1);
            assert!(!output.files.is_empty(), "backend produced no output files");
        }
    };
}

// MatMul tests for all backends
vendor_test!(matmul_onnx, OnnxBackend, "matmul");
vendor_test!(matmul_tflite, TfLiteBackend, "matmul");
vendor_test!(matmul_coreml, CoreMlBackend, "matmul");
vendor_test!(matmul_stablehlo, StableHloBackend, "matmul");
vendor_test!(matmul_samsung, SamsungBackend, "matmul");
vendor_test!(matmul_mediatek, MediaTekBackend, "matmul");
vendor_test!(matmul_intel, IntelBackend, "matmul");
vendor_test!(matmul_amd, AmdBackend, "matmul");
vendor_test!(matmul_qualcomm, QualcommBackend, "matmul");
vendor_test!(matmul_arm_ethos, ArmEthosBackend, "matmul");
vendor_test!(matmul_ceva, CevaBackend, "matmul");
vendor_test!(matmul_rockchip, RockchipBackend, "matmul");

// VecAdd tests for all backends
vendor_test!(vecadd_onnx, OnnxBackend, "vecadd");
vendor_test!(vecadd_tflite, TfLiteBackend, "vecadd");
vendor_test!(vecadd_coreml, CoreMlBackend, "vecadd");
vendor_test!(vecadd_stablehlo, StableHloBackend, "vecadd");
vendor_test!(vecadd_samsung, SamsungBackend, "vecadd");
vendor_test!(vecadd_mediatek, MediaTekBackend, "vecadd");
vendor_test!(vecadd_intel, IntelBackend, "vecadd");
vendor_test!(vecadd_amd, AmdBackend, "vecadd");
vendor_test!(vecadd_qualcomm, QualcommBackend, "vecadd");
vendor_test!(vecadd_arm_ethos, ArmEthosBackend, "vecadd");
vendor_test!(vecadd_ceva, CevaBackend, "vecadd");
vendor_test!(vecadd_rockchip, RockchipBackend, "vecadd");

// Conv2D tests for all backends
vendor_test!(conv2d_onnx, OnnxBackend, "conv2d");
vendor_test!(conv2d_tflite, TfLiteBackend, "conv2d");
vendor_test!(conv2d_samsung, SamsungBackend, "conv2d");
vendor_test!(conv2d_mediatek, MediaTekBackend, "conv2d");
vendor_test!(conv2d_intel, IntelBackend, "conv2d");
vendor_test!(conv2d_amd, AmdBackend, "conv2d");
vendor_test!(conv2d_qualcomm, QualcommBackend, "conv2d");
vendor_test!(conv2d_arm_ethos, ArmEthosBackend, "conv2d");
vendor_test!(conv2d_ceva, CevaBackend, "conv2d");
vendor_test!(conv2d_rockchip, RockchipBackend, "conv2d");

// Relu tests for all backends
vendor_test!(relu_onnx, OnnxBackend, "relu");
vendor_test!(relu_tflite, TfLiteBackend, "relu");
vendor_test!(relu_samsung, SamsungBackend, "relu");
vendor_test!(relu_mediatek, MediaTekBackend, "relu");
vendor_test!(relu_intel, IntelBackend, "relu");
vendor_test!(relu_amd, AmdBackend, "relu");
vendor_test!(relu_qualcomm, QualcommBackend, "relu");
vendor_test!(relu_arm_ethos, ArmEthosBackend, "relu");
vendor_test!(relu_ceva, CevaBackend, "relu");
vendor_test!(relu_rockchip, RockchipBackend, "relu");

// Attention tests for all backends
vendor_test!(attention_onnx, OnnxBackend, "attention");
vendor_test!(attention_tflite, TfLiteBackend, "attention");
vendor_test!(attention_samsung, SamsungBackend, "attention");
vendor_test!(attention_mediatek, MediaTekBackend, "attention");
vendor_test!(attention_intel, IntelBackend, "attention");
vendor_test!(attention_amd, AmdBackend, "attention");
vendor_test!(attention_qualcomm, QualcommBackend, "attention");
vendor_test!(attention_arm_ethos, ArmEthosBackend, "attention");
vendor_test!(attention_ceva, CevaBackend, "attention");
vendor_test!(attention_rockchip, RockchipBackend, "attention");

// MaxPool tests for vendor backends
vendor_test!(maxpool_intel, IntelBackend, "maxpool");
vendor_test!(maxpool_amd, AmdBackend, "maxpool");
vendor_test!(maxpool_qualcomm, QualcommBackend, "maxpool");
vendor_test!(maxpool_samsung, SamsungBackend, "maxpool");

// ---- Vendor-specific validation tests ----

/// Intel backend should emit OpenVINO IR XML alongside ONNX.
#[test]
fn intel_emits_openvino_ir() {
    let source = common::load_example("matmul");
    let output = common::compile_wgsl(&source, &IntelBackend, 1);
    let names: Vec<&str> = output.files.iter().map(|f| f.name.as_str()).collect();
    assert!(names.contains(&"model.xml"), "missing model.xml");
    assert!(names.contains(&"model.bin"), "missing model.bin");
    assert!(names.contains(&"output.onnx"), "missing output.onnx");
}

/// AMD backend should include XDNA metadata in ONNX output.
#[test]
fn amd_has_xdna_metadata() {
    use nxpu_backend_onnx::proto::ModelProto;
    use prost::Message;

    let source = common::load_example("matmul");
    let output = common::compile_wgsl(&source, &AmdBackend, 1);
    let bytes = common::first_binary(&output);
    let model = ModelProto::decode(bytes).unwrap();
    let keys: Vec<&str> = model
        .metadata_props
        .iter()
        .map(|p| p.key.as_str())
        .collect();
    assert!(
        keys.contains(&"xdna:target_device"),
        "missing xdna:target_device metadata"
    );
}

/// Vendor backends should produce validation diagnostics.
#[test]
fn vendor_backends_produce_diagnostics() {
    let source = common::load_example("matmul");

    for (name, backend) in [
        ("Intel", &IntelBackend as &dyn nxpu_backend_core::Backend),
        ("AMD", &AmdBackend),
        ("Samsung", &SamsungBackend),
        ("Qualcomm", &QualcommBackend),
        ("CEVA", &CevaBackend),
        ("Rockchip", &RockchipBackend),
        ("MediaTek", &MediaTekBackend),
        ("Arm Ethos", &ArmEthosBackend),
    ] {
        let output = common::compile_wgsl(&source, backend, 1);
        assert!(
            !output.diagnostics.is_empty(),
            "{name} backend produced no diagnostics"
        );
    }
}
