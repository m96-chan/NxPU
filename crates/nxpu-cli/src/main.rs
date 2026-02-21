use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use miette::{Context, IntoDiagnostic};

use nxpu_backend_core::{BackendOptions, BackendRegistry, OutputContent};
use nxpu_opt::{OptLevel, PassManager};

/// NxPU — WGSL to NPU transpiler
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Input WGSL file
    input: PathBuf,

    /// Target backend (default: ir-dump)
    #[arg(short, long, default_value = "ir-dump")]
    target: String,

    /// Output path (default: stdout)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Optimization level: 0, 1, or 2
    #[arg(long, default_value = "1", value_parser = parse_opt_level)]
    opt_level: OptLevel,

    /// Dump IR to stderr before backend compilation
    #[arg(long)]
    emit_ir: bool,

    /// Validate and optimize without producing output
    #[arg(long)]
    dry_run: bool,
}

fn parse_opt_level(s: &str) -> Result<OptLevel, String> {
    match s {
        "0" => Ok(OptLevel::O0),
        "1" => Ok(OptLevel::O1),
        "2" => Ok(OptLevel::O2),
        _ => Err(format!(
            "invalid optimization level '{s}', expected 0, 1, or 2"
        )),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("Error: {err:?}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> miette::Result<()> {
    let cli = Cli::parse();

    // 1. Read source file.
    let source = std::fs::read_to_string(&cli.input)
        .into_diagnostic()
        .wrap_err_with(|| format!("failed to read {}", cli.input.display()))?;

    // 2. Parse WGSL to IR.
    let mut module = nxpu_parser::parse(&source)
        .map_err(|e| miette::miette!("{e}"))
        .wrap_err("WGSL parse failed")?;

    // 3. Optimize.
    PassManager::for_level(cli.opt_level).run(&mut module);

    // 4. Optionally dump IR to stderr.
    if cli.emit_ir {
        eprintln!("{}", nxpu_ir::dump_module(&module));
    }

    // 5. Dry-run: stop here.
    if cli.dry_run {
        return Ok(());
    }

    // 6. Backend dispatch.
    let mut registry = BackendRegistry::with_builtins();
    registry.register(Box::new(nxpu_backend_onnx::OnnxBackend));
    registry.register(Box::new(nxpu_backend_tflite::TfLiteBackend));
    registry.register(Box::new(nxpu_backend_coreml::CoreMlBackend));
    registry.register(Box::new(nxpu_backend_stablehlo::StableHloBackend));
    registry.register(Box::new(nxpu_backend_samsung::SamsungBackend));
    registry.register(Box::new(nxpu_backend_mediatek::MediaTekBackend));
    registry.register(Box::new(nxpu_backend_intel::IntelBackend));
    registry.register(Box::new(nxpu_backend_amd::AmdBackend));
    let backend = registry.find(&cli.target).ok_or_else(|| {
        let available = registry.list_targets().join(", ");
        miette::miette!("unknown target '{}' (available: {})", cli.target, available)
    })?;

    let opts = BackendOptions {
        opt_level: match cli.opt_level {
            OptLevel::O0 => 0,
            OptLevel::O1 => 1,
            OptLevel::O2 => 2,
        },
    };

    let output = backend
        .compile(&module, &opts)
        .map_err(|e| miette::miette!("{e}"))
        .wrap_err("backend compilation failed")?;

    // 7. Print diagnostics.
    for diag in &output.diagnostics {
        eprintln!("{:?}: {}", diag.level, diag.message);
    }

    // 8. Write output.
    for file in &output.files {
        match (&cli.output, &file.content) {
            (Some(path), OutputContent::Text(text)) => {
                std::fs::write(path, text)
                    .into_diagnostic()
                    .wrap_err_with(|| format!("failed to write {}", path.display()))?;
            }
            (Some(path), OutputContent::Binary(data)) => {
                std::fs::write(path, data)
                    .into_diagnostic()
                    .wrap_err_with(|| format!("failed to write {}", path.display()))?;
            }
            (None, OutputContent::Text(text)) => {
                print!("{text}");
            }
            (None, OutputContent::Binary(_)) => {
                return Err(miette::miette!(
                    "backend produced binary output but no --output path was specified"
                ));
            }
        }
    }

    Ok(())
}
