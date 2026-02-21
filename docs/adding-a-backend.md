# Adding a Backend

This guide walks through adding a new NPU backend to NxPU.

## 1. Create the Crate

```sh
cargo init --lib crates/nxpu-backend-myvendor
```

Add it to the workspace in the root `Cargo.toml`:

```toml
members = [
    # ...existing members...
    "crates/nxpu-backend-myvendor",
]
```

Set up `crates/nxpu-backend-myvendor/Cargo.toml`:

```toml
[package]
name = "nxpu-backend-myvendor"
description = "MyVendor NPU backend for NxPU"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true

[dependencies]
nxpu-ir = { path = "../nxpu-ir" }
nxpu-backend-core = { path = "../nxpu-backend-core" }

[dev-dependencies]
nxpu-parser = { path = "../nxpu-parser" }
```

## 2. Implement the Backend Trait

Create `crates/nxpu-backend-myvendor/src/lib.rs`:

```rust
use nxpu_backend_core::{
    Backend, BackendError, BackendOptions, BackendOutput,
    Diagnostic, DiagnosticLevel, OutputContent, OutputFile, Precision,
};
use nxpu_ir::Module;

#[derive(Debug)]
pub struct MyVendorBackend;

impl Backend for MyVendorBackend {
    fn name(&self) -> &str {
        "MyVendor NPU"
    }

    fn targets(&self) -> &[&str] {
        &["myvendor"]
    }

    fn preferred_precision(&self) -> Precision {
        Precision::Int8 // or F16, BF16, F32
    }

    fn compile(
        &self,
        module: &Module,
        opts: &BackendOptions,
    ) -> Result<BackendOutput, BackendError> {
        if module.entry_points.is_empty() {
            return Err(BackendError::Other("no entry points".into()));
        }

        // Use the shared analysis to classify entry points
        // (add nxpu-backend-onnx as a dependency if needed)

        let files = vec![OutputFile {
            name: "output.bin".into(),
            content: OutputContent::Binary(vec![/* your binary format */]),
        }];

        Ok(BackendOutput {
            files,
            diagnostics: vec![Diagnostic {
                level: DiagnosticLevel::Info,
                message: "compiled for MyVendor NPU".into(),
            }],
        })
    }
}
```

## 3. Register in the CLI

Add your backend to `crates/nxpu-cli/src/main.rs`:

```rust
// In the run() function, after other register() calls:
registry.register(Box::new(nxpu_backend_myvendor::MyVendorBackend));
```

And add the dependency to `crates/nxpu-cli/Cargo.toml`:

```toml
nxpu-backend-myvendor = { path = "../nxpu-backend-myvendor" }
```

## 4. Using Shared Analysis

Most backends reuse the pattern analysis from `nxpu-backend-onnx::analyze`:

```rust
use nxpu_backend_onnx::analyze;

fn compile(&self, module: &Module, _opts: &BackendOptions) -> Result<BackendOutput, BackendError> {
    for (i, ep) in module.entry_points.iter().enumerate() {
        let pattern = analyze::classify_entry_point(module, i)
            .map_err(|e| BackendError::Unsupported(format!("{e}")))?;

        match &pattern {
            analyze::KernelPattern::MatMul { inputs, output, shape } => {
                // Emit vendor-specific MatMul
            }
            analyze::KernelPattern::ElementWise { op, inputs, output, .. } => {
                // Emit vendor-specific elementwise op
            }
        }
    }
    // ...
}
```

## 5. Delegating to an Existing Backend

For vendor backends that build on TFLite or ONNX, delegate and post-process:

```rust
use nxpu_backend_tflite::TfLiteBackend;

fn compile(&self, module: &Module, opts: &BackendOptions) -> Result<BackendOutput, BackendError> {
    let mut output = TfLiteBackend.compile(module, opts)?;
    // Post-process: invoke vendor SDK, transform output, etc.
    Ok(output)
}
```

## 6. Add Tests

Add a test that compiles the example `matmul.wgsl`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use nxpu_backend_core::BackendOptions;

    #[test]
    fn compile_matmul() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/matmul.wgsl"
        )).unwrap();
        let module = nxpu_parser::parse(&source).unwrap();
        let output = MyVendorBackend.compile(&module, &BackendOptions::default()).unwrap();
        assert!(!output.files.is_empty());
    }
}
```

## 7. Run Tests

```sh
cargo test -p nxpu-backend-myvendor
cargo test  # full workspace
```
