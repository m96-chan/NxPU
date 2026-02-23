#![warn(missing_docs)]
//! Backend trait and plugin architecture for NxPU.
//!
//! Defines the [`Backend`] trait that all NPU code emitters implement,
//! along with supporting types ([`BackendOptions`], [`BackendOutput`],
//! [`BackendError`]) and a [`BackendRegistry`] for CLI dispatch.

use std::fmt::{self, Debug};

use nxpu_ir::Module;

/// Target precision for NPU compilation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// 32-bit floating point (no conversion).
    F32,
    /// 16-bit floating point.
    F16,
    /// Brain floating point (16-bit).
    BF16,
    /// 8-bit integer (quantized).
    Int8,
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::Int8 => "I8",
        })
    }
}

/// Policy for choosing precision during compilation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrecisionPolicy {
    /// Keep the original precision from the WGSL source.
    Keep,
    /// Use a specific precision regardless of backend preference.
    Explicit(Precision),
    /// Automatically select based on the backend's preferred precision.
    #[default]
    Auto,
}

impl fmt::Display for PrecisionPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Keep => f.write_str("Keep"),
            Self::Explicit(p) => write!(f, "Explicit({p})"),
            Self::Auto => f.write_str("Auto"),
        }
    }
}

/// A backend that compiles NxPU IR to target-specific output.
pub trait Backend: Debug + Send + Sync {
    /// Human-readable name (e.g. "apple-ane").
    fn name(&self) -> &str;

    /// Target identifiers this backend handles (for `--target` dispatch).
    fn targets(&self) -> &[&str];

    /// Compile an optimized IR module to backend-specific output.
    fn compile(
        &self,
        module: &Module,
        opts: &BackendOptions,
    ) -> Result<BackendOutput, BackendError>;

    /// The precision this backend prefers for optimal NPU execution.
    fn preferred_precision(&self) -> Precision {
        Precision::F32
    }
}

/// Options passed to a backend during compilation.
///
/// The `precision` field controls the target quantization precision.
/// In the typical workflow, the caller (e.g. `nxpu-cli`) applies the
/// appropriate quantization pass (`F32ToF16`, `F32ToBf16`, `F32ToInt8`,
/// or `MixedPrecisionPass`) to the IR module *before* calling
/// `Backend::compile`. The `precision` field is informational — backends
/// can read it to emit diagnostics or choose format-specific options,
/// but the IR has already been rewritten by the quantization pass.
#[derive(Clone, Debug, Default)]
pub struct BackendOptions {
    /// Optimization level (0 = none, 1 = basic, 2 = aggressive).
    pub opt_level: u8,
    /// Precision policy for quantization.
    ///
    /// The CLI applies the corresponding quantization pass to the IR before
    /// compilation. Backends may use this to emit precision-related
    /// diagnostics or metadata, but should not re-quantize the IR.
    pub precision: PrecisionPolicy,
}

impl fmt::Display for BackendOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BackendOptions {{ opt_level: {}, precision: {} }}",
            self.opt_level, self.precision
        )
    }
}

/// The output produced by a backend.
#[derive(Clone, Debug)]
pub struct BackendOutput {
    /// One or more output files.
    pub files: Vec<OutputFile>,
    /// Non-fatal diagnostics.
    pub diagnostics: Vec<Diagnostic>,
}

impl fmt::Display for BackendOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} file(s), {} diagnostic(s)",
            self.files.len(),
            self.diagnostics.len()
        )
    }
}

/// A single output file.
#[derive(Clone, Debug)]
pub struct OutputFile {
    /// Suggested filename (e.g. "output.bin", "module.ir").
    pub name: String,
    /// The file content.
    pub content: OutputContent,
}

impl fmt::Display for OutputFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

/// Content of an output file.
#[derive(Clone, Debug)]
pub enum OutputContent {
    /// UTF-8 text.
    Text(String),
    /// Raw binary data.
    Binary(Vec<u8>),
}

impl fmt::Display for OutputContent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text(s) => write!(f, "Text({} chars)", s.len()),
            Self::Binary(b) => write!(f, "Binary({} bytes)", b.len()),
        }
    }
}

/// A non-fatal diagnostic message from a backend.
#[derive(Clone, Debug)]
pub struct Diagnostic {
    /// Severity level.
    pub level: DiagnosticLevel,
    /// Human-readable message.
    pub message: String,
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.level, self.message)
    }
}

/// Severity level for diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagnosticLevel {
    /// A warning that does not prevent compilation.
    Warning,
    /// An informational note.
    Info,
}

impl fmt::Display for DiagnosticLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Warning => "Warning",
            Self::Info => "Info",
        })
    }
}

/// Errors that can occur during backend compilation.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// The module uses an IR feature not supported by this backend.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// A general backend error.
    #[error("{0}")]
    Other(String),
}

/// Registry of available backends, used for CLI `--target` dispatch.
pub struct BackendRegistry {
    backends: Vec<Box<dyn Backend>>,
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl BackendRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
        }
    }

    /// Creates a registry pre-populated with built-in backends.
    pub fn with_builtins() -> Self {
        let mut reg = Self::new();
        reg.register(Box::new(IrDumpBackend));
        reg
    }

    /// Registers a backend.
    pub fn register(&mut self, backend: Box<dyn Backend>) {
        self.backends.push(backend);
    }

    /// Finds a backend that handles the given target identifier.
    pub fn find(&self, target: &str) -> Option<&dyn Backend> {
        self.backends
            .iter()
            .find(|b| b.targets().contains(&target))
            .map(|b| &**b)
    }

    /// Lists all supported target identifiers.
    pub fn list_targets(&self) -> Vec<&str> {
        self.backends
            .iter()
            .flat_map(|b| b.targets().iter().copied())
            .collect()
    }
}

/// Built-in backend that dumps the IR as text using [`nxpu_ir::dump_module`].
#[derive(Debug)]
pub struct IrDumpBackend;

impl Backend for IrDumpBackend {
    fn name(&self) -> &str {
        "IR Dump"
    }

    fn targets(&self) -> &[&str] {
        &["ir-dump", "ir"]
    }

    fn compile(
        &self,
        module: &Module,
        _opts: &BackendOptions,
    ) -> Result<BackendOutput, BackendError> {
        let text = nxpu_ir::dump_module(module);
        Ok(BackendOutput {
            files: vec![OutputFile {
                name: "module.ir".into(),
                content: OutputContent::Text(text),
            }],
            diagnostics: vec![],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ir_dump_backend_targets() {
        let backend = IrDumpBackend;
        assert_eq!(backend.name(), "IR Dump");
        assert!(backend.targets().contains(&"ir-dump"));
        assert!(backend.targets().contains(&"ir"));
    }

    #[test]
    fn ir_dump_backend_compile() {
        let module = Module::default();
        let opts = BackendOptions::default();
        let output = IrDumpBackend.compile(&module, &opts).unwrap();
        assert_eq!(output.files.len(), 1);
        assert_eq!(output.files[0].name, "module.ir");
        match &output.files[0].content {
            OutputContent::Text(text) => assert!(text.contains("Types:")),
            _ => panic!("expected text output"),
        }
    }

    #[test]
    fn registry_find_builtin() {
        let reg = BackendRegistry::with_builtins();
        assert!(reg.find("ir-dump").is_some());
        assert!(reg.find("ir").is_some());
        assert!(reg.find("nonexistent").is_none());
    }

    #[test]
    fn registry_list_targets() {
        let reg = BackendRegistry::with_builtins();
        let targets = reg.list_targets();
        assert!(targets.contains(&"ir-dump"));
        assert!(targets.contains(&"ir"));
    }

    #[test]
    fn registry_custom_backend() {
        #[derive(Debug)]
        struct TestBackend;
        impl Backend for TestBackend {
            fn name(&self) -> &str {
                "test"
            }
            fn targets(&self) -> &[&str] {
                &["test-target"]
            }
            fn compile(
                &self,
                _module: &Module,
                _opts: &BackendOptions,
            ) -> Result<BackendOutput, BackendError> {
                Ok(BackendOutput {
                    files: vec![],
                    diagnostics: vec![],
                })
            }
        }

        let mut reg = BackendRegistry::new();
        reg.register(Box::new(TestBackend));
        assert!(reg.find("test-target").is_some());
    }

    #[test]
    fn display_precision_all_variants() {
        assert_eq!(format!("{}", Precision::F32), "F32");
        assert_eq!(format!("{}", Precision::F16), "F16");
        assert_eq!(format!("{}", Precision::BF16), "BF16");
        assert_eq!(format!("{}", Precision::Int8), "I8");
    }

    #[test]
    fn display_precision_policy_all_variants() {
        assert_eq!(format!("{}", PrecisionPolicy::Keep), "Keep");
        assert_eq!(format!("{}", PrecisionPolicy::Auto), "Auto");
        assert_eq!(
            format!("{}", PrecisionPolicy::Explicit(Precision::F16)),
            "Explicit(F16)"
        );
    }

    #[test]
    fn display_backend_options() {
        let opts = BackendOptions {
            opt_level: 2,
            precision: PrecisionPolicy::Explicit(Precision::Int8),
        };
        let s = format!("{opts}");
        assert!(s.contains("opt_level: 2"));
        assert!(s.contains("Explicit(I8)"));
    }

    #[test]
    fn display_backend_output() {
        let output = BackendOutput {
            files: vec![
                OutputFile {
                    name: "a.bin".into(),
                    content: OutputContent::Binary(vec![1, 2, 3]),
                },
                OutputFile {
                    name: "b.txt".into(),
                    content: OutputContent::Text("hello".into()),
                },
            ],
            diagnostics: vec![Diagnostic {
                level: DiagnosticLevel::Info,
                message: "done".into(),
            }],
        };
        assert_eq!(format!("{output}"), "2 file(s), 1 diagnostic(s)");
    }

    #[test]
    fn display_output_file() {
        let f = OutputFile {
            name: "model.onnx".into(),
            content: OutputContent::Binary(vec![]),
        };
        assert_eq!(format!("{f}"), "model.onnx");
    }

    #[test]
    fn display_output_content_all_variants() {
        assert_eq!(
            format!("{}", OutputContent::Text("abc".into())),
            "Text(3 chars)"
        );
        assert_eq!(
            format!("{}", OutputContent::Binary(vec![0; 100])),
            "Binary(100 bytes)"
        );
    }

    #[test]
    fn display_diagnostic_and_level() {
        let warn = Diagnostic {
            level: DiagnosticLevel::Warning,
            message: "deprecated op".into(),
        };
        assert_eq!(format!("{warn}"), "[Warning] deprecated op");

        let info = Diagnostic {
            level: DiagnosticLevel::Info,
            message: "classified as Add".into(),
        };
        assert_eq!(format!("{info}"), "[Info] classified as Add");
    }

    #[test]
    fn display_diagnostic_level_all_variants() {
        assert_eq!(format!("{}", DiagnosticLevel::Warning), "Warning");
        assert_eq!(format!("{}", DiagnosticLevel::Info), "Info");
    }

    #[test]
    fn registry_empty_list_targets() {
        let reg = BackendRegistry::new();
        assert!(reg.list_targets().is_empty());
    }

    #[test]
    fn registry_default_is_empty() {
        let reg = BackendRegistry::default();
        assert!(reg.list_targets().is_empty());
    }

    #[test]
    fn preferred_precision_default() {
        let backend = IrDumpBackend;
        assert_eq!(backend.preferred_precision(), Precision::F32);
    }

    #[test]
    fn backend_error_display() {
        let e1 = BackendError::Unsupported("int64 tensors".into());
        assert_eq!(format!("{e1}"), "unsupported: int64 tensors");

        let e2 = BackendError::Other("internal failure".into());
        assert_eq!(format!("{e2}"), "internal failure");
    }
}
