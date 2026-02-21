use nxpu_backend_core::{Backend, BackendOptions, BackendOutput, OutputContent};
#[allow(unused_imports)]
use nxpu_ir::Module;
use nxpu_opt::{OptLevel, PassManager};

/// Parse WGSL source, optimize at the given level, and compile with the backend.
#[allow(dead_code)]
pub fn compile_wgsl(source: &str, backend: &dyn Backend, opt_level: u8) -> BackendOutput {
    let mut module = nxpu_parser::parse(source).expect("WGSL parse failed");
    let level = match opt_level {
        0 => OptLevel::O0,
        2 => OptLevel::O2,
        _ => OptLevel::O1,
    };
    PassManager::for_level(level).run(&mut module);
    backend
        .compile(
            &module,
            &BackendOptions {
                opt_level,
                ..Default::default()
            },
        )
        .expect("backend compilation failed")
}

/// Parse WGSL source, optimize, run a quantization pass, and compile.
#[allow(dead_code)]
pub fn compile_wgsl_with_pass(
    source: &str,
    pass: &dyn nxpu_opt::Pass,
    backend: &dyn Backend,
    opt_level: u8,
) -> BackendOutput {
    let mut module = nxpu_parser::parse(source).expect("WGSL parse failed");
    let level = match opt_level {
        0 => OptLevel::O0,
        2 => OptLevel::O2,
        _ => OptLevel::O1,
    };
    PassManager::for_level(level).run(&mut module);
    pass.run(&mut module);
    backend
        .compile(
            &module,
            &BackendOptions {
                opt_level,
                ..Default::default()
            },
        )
        .expect("backend compilation failed")
}

/// Load an example WGSL file by name (without extension).
#[allow(dead_code)]
pub fn load_example(name: &str) -> String {
    let path = format!("{}/../../examples/{name}.wgsl", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to load {path}: {e}"))
}

/// Extract the first binary output from a `BackendOutput`.
#[allow(dead_code)]
pub fn first_binary(output: &BackendOutput) -> &[u8] {
    match &output.files[0].content {
        OutputContent::Binary(b) => b,
        OutputContent::Text(_) => panic!("expected binary output, got text"),
    }
}

/// Extract the first text output from a `BackendOutput`.
#[allow(dead_code)]
pub fn first_text(output: &BackendOutput) -> &str {
    match &output.files[0].content {
        OutputContent::Text(t) => t,
        OutputContent::Binary(_) => panic!("expected text output, got binary"),
    }
}

/// Like `compile_wgsl` but returns a Result instead of panicking.
#[allow(dead_code)]
pub fn try_compile_wgsl(
    source: &str,
    backend: &dyn Backend,
    opt_level: u8,
) -> Result<BackendOutput, nxpu_backend_core::BackendError> {
    let mut module = nxpu_parser::parse(source).expect("WGSL parse failed");
    let level = match opt_level {
        0 => OptLevel::O0,
        2 => OptLevel::O2,
        _ => OptLevel::O1,
    };
    PassManager::for_level(level).run(&mut module);
    backend.compile(
        &module,
        &BackendOptions {
            opt_level,
            ..Default::default()
        },
    )
}

/// Parse WGSL to Module (no optimization).
#[allow(dead_code)]
pub fn parse_wgsl(source: &str) -> Module {
    nxpu_parser::parse(source).expect("WGSL parse failed")
}
