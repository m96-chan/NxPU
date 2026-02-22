mod common;

use nxpu_backend_core::IrDumpBackend;
use nxpu_backend_onnx::OnnxBackend;

#[test]
fn invalid_wgsl_is_rejected() {
    let result = nxpu_parser::parse("this is not valid WGSL @@@ {{{");
    assert!(result.is_err());
}

#[test]
fn fragment_shader_rejected() {
    // Fragment shaders are not compute shaders — the parser should still
    // produce a module, but it will have no compute entry points.
    let source = r#"
@fragment
fn main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
"#;
    let result = nxpu_parser::parse(source);
    // The parser may reject this (no compute shader) or produce an empty module.
    match result {
        Err(_) => {} // expected
        Ok(module) => {
            // If parsing succeeds, there should be no compute entry points,
            // so the backend should fail.
            let result = common::try_compile_wgsl_from_module(&module, &OnnxBackend, 1);
            assert!(result.is_err());
        }
    }
}

#[test]
fn empty_module_backend_error() {
    // An empty module (no entry points) should produce a backend error.
    let module = nxpu_ir::Module::default();
    let opts = nxpu_backend_core::BackendOptions::default();
    let result = OnnxBackend.compile(&module, &opts);
    assert!(result.is_err());
}

#[test]
fn ir_dump_empty_module_ok() {
    // IrDumpBackend should handle an empty module gracefully.
    let module = nxpu_ir::Module::default();
    let opts = nxpu_backend_core::BackendOptions::default();
    let result = IrDumpBackend.compile(&module, &opts);
    assert!(result.is_ok());
}

#[test]
fn syntax_error_gives_useful_message() {
    let result = nxpu_parser::parse("fn main( {}");
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    // Should contain some indication of what went wrong.
    assert!(!err_msg.is_empty());
}

use nxpu_backend_core::Backend;
