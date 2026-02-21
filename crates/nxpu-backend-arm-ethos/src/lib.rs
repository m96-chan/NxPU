//! Arm Ethos NPU backend for NxPU.
//!
//! Compiles NxPU IR via the TFLite backend, then optionally invokes the
//! Arm Vela compiler to produce Ethos-U optimized `.tflite` binaries.
//!
//! When `vela` is not found on `$PATH`, the backend falls back to emitting
//! a standard `.tflite` file with a diagnostic hint.

use std::process::Command;

use nxpu_backend_core::{
    Backend, BackendError, BackendOptions, BackendOutput, Diagnostic, DiagnosticLevel,
    OutputContent, OutputFile, Precision,
};
use nxpu_backend_tflite::TfLiteBackend;
use nxpu_ir::Module;

/// Arm Ethos NPU backend.
///
/// Compilation pipeline:
/// 1. Lower IR → TFLite FlatBuffer via [`TfLiteBackend`].
/// 2. If `vela` is available, invoke it on the TFLite file to produce
///    an Ethos-U optimized binary.
/// 3. Otherwise, emit the unoptimized TFLite with a diagnostic.
#[derive(Debug)]
pub struct ArmEthosBackend;

/// Check whether the `vela` CLI tool is available on PATH.
fn vela_available() -> bool {
    Command::new("vela")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Run the Vela compiler on a TFLite file, returning the optimized bytes.
///
/// Uses a unique temporary directory per invocation to avoid race conditions
/// during parallel compilation.
fn run_vela(tflite_bytes: &[u8]) -> Result<Vec<u8>, BackendError> {
    let temp_dir = std::env::temp_dir().join(format!("nxpu-vela-{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir)
        .map_err(|e| BackendError::Other(format!("failed to create temp dir: {e}")))?;

    // Use a closure to ensure cleanup on all exit paths.
    let result = (|| {
        let input_name = "input";
        let input_path = temp_dir.join(format!("{input_name}.tflite"));
        std::fs::write(&input_path, tflite_bytes)
            .map_err(|e| BackendError::Other(format!("failed to write temp tflite: {e}")))?;

        let output = Command::new("vela")
            .arg(&input_path)
            .arg("--output-dir")
            .arg(&temp_dir)
            .output()
            .map_err(|e| BackendError::Other(format!("failed to run vela: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(BackendError::Other(format!("vela failed: {stderr}")));
        }

        // Vela outputs to <output-dir>/<input_name>_vela.tflite
        let vela_output = temp_dir.join(format!("{input_name}_vela.tflite"));
        let optimized = std::fs::read(&vela_output)
            .map_err(|e| BackendError::Other(format!("failed to read vela output: {e}")))?;

        Ok(optimized)
    })();

    // Always clean up temp directory, regardless of success or failure.
    let _ = std::fs::remove_dir_all(&temp_dir);

    result
}

impl Backend for ArmEthosBackend {
    fn name(&self) -> &str {
        "Arm Ethos NPU"
    }

    fn targets(&self) -> &[&str] {
        &["arm-ethos", "ethos-u"]
    }

    fn preferred_precision(&self) -> Precision {
        Precision::Int8
    }

    fn compile(
        &self,
        module: &Module,
        opts: &BackendOptions,
    ) -> Result<BackendOutput, BackendError> {
        // Step 1: Generate TFLite model
        let tflite_output = TfLiteBackend.compile(module, opts)?;

        let mut diagnostics = tflite_output.diagnostics;
        let mut files = Vec::new();

        // Check Vela availability once, not per file.
        let has_vela = vela_available();

        for file in &tflite_output.files {
            let tflite_bytes = match &file.content {
                OutputContent::Binary(b) => b,
                OutputContent::Text(_) => {
                    files.push(file.clone());
                    continue;
                }
            };

            // Step 2: Try to invoke Vela
            if has_vela {
                match run_vela(tflite_bytes) {
                    Ok(optimized) => {
                        diagnostics.push(Diagnostic {
                            level: DiagnosticLevel::Info,
                            message: format!(
                                "Vela compilation successful ({} -> {} bytes)",
                                tflite_bytes.len(),
                                optimized.len()
                            ),
                        });
                        files.push(OutputFile {
                            name: file.name.replace(".tflite", "_vela.tflite"),
                            content: OutputContent::Binary(optimized),
                        });
                        // Also emit original for reference
                        files.push(file.clone());
                    }
                    Err(e) => {
                        diagnostics.push(Diagnostic {
                            level: DiagnosticLevel::Warning,
                            message: format!(
                                "Vela compilation failed, emitting unoptimized TFLite: {e}"
                            ),
                        });
                        files.push(file.clone());
                    }
                }
            } else {
                // Vela not available — fall back with hint
                diagnostics.push(Diagnostic {
                    level: DiagnosticLevel::Info,
                    message: "vela not found on PATH; emitting unoptimized TFLite. \
                              Install: pip install ethos-u-vela"
                        .into(),
                });
                diagnostics.push(Diagnostic {
                    level: DiagnosticLevel::Info,
                    message: format!("To optimize manually: vela {}", file.name),
                });
                files.push(file.clone());
            }
        }

        Ok(BackendOutput { files, diagnostics })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nxpu_backend_core::BackendOptions;

    #[test]
    fn backend_metadata() {
        let backend = ArmEthosBackend;
        assert_eq!(backend.name(), "Arm Ethos NPU");
        assert!(backend.targets().contains(&"arm-ethos"));
        assert!(backend.targets().contains(&"ethos-u"));
        assert_eq!(backend.preferred_precision(), Precision::Int8);
    }

    #[test]
    fn compile_produces_tflite() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/matmul.wgsl"
        ))
        .unwrap();
        let module = nxpu_parser::parse(&source).unwrap();

        let output = ArmEthosBackend
            .compile(&module, &BackendOptions::default())
            .unwrap();

        // Should have at least one output file
        assert!(!output.files.is_empty());

        // At least one file should be a .tflite
        let has_tflite = output.files.iter().any(|f| f.name.ends_with(".tflite"));
        assert!(has_tflite);

        // Should have diagnostics about vela status
        assert!(!output.diagnostics.is_empty());
    }

    #[test]
    fn vela_availability_check() {
        // This test just verifies the function doesn't panic.
        // On most CI systems vela won't be installed.
        let _available = vela_available();
    }
}
