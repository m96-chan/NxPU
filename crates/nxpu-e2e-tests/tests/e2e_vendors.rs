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
