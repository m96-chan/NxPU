use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use miette::{Context, IntoDiagnostic};

use nxpu_backend_core::{
    BackendOptions, BackendRegistry, OutputContent, Precision, PrecisionPolicy,
};
use nxpu_opt::{OptLevel, PassManager};

/// NxPU — WGSL to NPU transpiler
#[derive(Parser)]
#[command(version, about)]
#[allow(clippy::struct_excessive_bools)]
struct Cli {
    /// Input WGSL file
    input: Option<PathBuf>,

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

    /// Precision policy: keep, f16, bf16, int8, or auto (default: auto)
    #[arg(long, default_value = "auto", value_parser = parse_precision)]
    precision: PrecisionPolicy,

    /// Directory containing calibration data (.bin files with f32 values)
    #[arg(long)]
    calibration_data: Option<PathBuf>,

    /// Calibration method: minmax, percentile, kl-divergence (default: minmax)
    #[arg(long, default_value = "minmax", value_parser = parse_calibration_method)]
    calibration_method: nxpu_opt::CalibrationMethod,

    /// Verbose output (print calibration statistics, etc.)
    #[arg(short, long)]
    verbose: bool,

    /// List all available target backends and exit
    #[arg(long)]
    list_targets: bool,
}

fn parse_calibration_method(s: &str) -> Result<nxpu_opt::CalibrationMethod, String> {
    nxpu_opt::CalibrationMethod::from_str_name(s).ok_or_else(|| {
        format!("invalid calibration method '{s}', expected minmax, percentile, or kl-divergence")
    })
}

fn parse_precision(s: &str) -> Result<PrecisionPolicy, String> {
    match s {
        "keep" => Ok(PrecisionPolicy::Keep),
        "f16" => Ok(PrecisionPolicy::Explicit(Precision::F16)),
        "bf16" => Ok(PrecisionPolicy::Explicit(Precision::BF16)),
        "int8" => Ok(PrecisionPolicy::Explicit(Precision::Int8)),
        "auto" => Ok(PrecisionPolicy::Auto),
        _ => Err(format!(
            "invalid precision '{s}', expected keep, f16, bf16, int8, or auto"
        )),
    }
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

fn build_registry() -> BackendRegistry {
    #[allow(unused_mut)]
    let mut registry = BackendRegistry::with_builtins();
    #[cfg(feature = "backend-onnx")]
    registry.register(Box::new(nxpu_backend_onnx::OnnxBackend));
    #[cfg(feature = "backend-tflite")]
    registry.register(Box::new(nxpu_backend_tflite::TfLiteBackend));
    #[cfg(feature = "backend-coreml")]
    registry.register(Box::new(nxpu_backend_coreml::CoreMlBackend));
    #[cfg(feature = "backend-stablehlo")]
    registry.register(Box::new(nxpu_backend_stablehlo::StableHloBackend));
    #[cfg(feature = "backend-samsung")]
    registry.register(Box::new(nxpu_backend_samsung::SamsungBackend));
    #[cfg(feature = "backend-mediatek")]
    registry.register(Box::new(nxpu_backend_mediatek::MediaTekBackend));
    #[cfg(feature = "backend-intel")]
    registry.register(Box::new(nxpu_backend_intel::IntelBackend));
    #[cfg(feature = "backend-amd")]
    registry.register(Box::new(nxpu_backend_amd::AmdBackend));
    #[cfg(feature = "backend-qualcomm")]
    registry.register(Box::new(nxpu_backend_qualcomm::QualcommBackend));
    #[cfg(feature = "backend-arm-ethos")]
    registry.register(Box::new(nxpu_backend_arm_ethos::ArmEthosBackend));
    #[cfg(feature = "backend-ceva")]
    registry.register(Box::new(nxpu_backend_ceva::CevaBackend));
    #[cfg(feature = "backend-rockchip")]
    registry.register(Box::new(nxpu_backend_rockchip::RockchipBackend));
    registry
}

fn run() -> miette::Result<()> {
    env_logger::try_init().ok();

    let cli = Cli::parse();

    // --list-targets: print available backends and exit.
    if cli.list_targets {
        let registry = build_registry();
        for target in registry.list_targets() {
            println!("{target}");
        }
        return Ok(());
    }

    let input = cli.input.ok_or_else(|| {
        miette::miette!("input file is required (use --list-targets to list backends)")
    })?;

    // 1. Read source file.
    let source = std::fs::read_to_string(&input)
        .into_diagnostic()
        .wrap_err_with(|| format!("failed to read {}", input.display()))?;

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
    let registry = build_registry();
    let backend = registry.find(&cli.target).ok_or_else(|| {
        let available = registry.list_targets().join(", ");
        miette::miette!("unknown target '{}' (available: {})", cli.target, available)
    })?;

    // 6b. Resolve precision and run quantization pass.
    let resolved_precision = match cli.precision {
        PrecisionPolicy::Keep => None,
        PrecisionPolicy::Explicit(p) => Some(p),
        PrecisionPolicy::Auto => {
            let pref = backend.preferred_precision();
            if pref == Precision::F32 {
                None
            } else {
                Some(pref)
            }
        }
    };

    if let Some(precision) = resolved_precision {
        use nxpu_opt::Pass;
        match precision {
            Precision::F16 => {
                nxpu_opt::F32ToF16.run(&mut module);
            }
            Precision::BF16 => {
                nxpu_opt::F32ToBf16.run(&mut module);
            }
            Precision::Int8 => {
                // If calibration data directory is provided, run the calibration pipeline.
                if let Some(cal_dir) = &cli.calibration_data {
                    let dataset = nxpu_opt::CalibrationDataset::load_from_dir(cal_dir)
                        .map_err(|e| miette::miette!("calibration failed: {e}"))?;

                    if cli.verbose {
                        eprintln!(
                            "Loaded {} calibration samples from {}",
                            dataset.num_samples(),
                            cal_dir.display()
                        );
                    }

                    let cal_result = nxpu_opt::run_calibration(
                        &dataset,
                        &cli.calibration_method,
                        true, // symmetric for INT8
                    )
                    .map_err(|e| miette::miette!("calibration failed: {e}"))?;

                    if cli.verbose {
                        eprintln!("Calibration results:");
                        for (name, params) in &cal_result.tensor_params {
                            eprintln!(
                                "  {}: scale={:.6}, zero_point={}",
                                name, params.scale, params.zero_point
                            );
                        }
                    }

                    nxpu_opt::F32ToInt8::with_calibration_result(cal_result).run(&mut module);
                } else {
                    nxpu_opt::F32ToInt8::default().run(&mut module);
                }
            }
            Precision::F32 => {}
        }
    }

    let opts = BackendOptions {
        opt_level: match cli.opt_level {
            OptLevel::O0 => 0,
            OptLevel::O1 => 1,
            OptLevel::O2 => 2,
        },
        precision: cli.precision,
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
    if let Some(base) = &cli.output {
        if output.files.len() > 1 {
            // Multi-file output: derive per-file paths from the base output path.
            let stem = base
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "output".into());
            let parent = base.parent().unwrap_or_else(|| std::path::Path::new("."));

            for file in &output.files {
                let dest = parent.join(format!("{stem}_{}", file.name));
                write_output_file(&dest, &file.content)?;
            }
        } else {
            for file in &output.files {
                write_output_file(base, &file.content)?;
            }
        }
    } else {
        for file in &output.files {
            match &file.content {
                OutputContent::Text(text) => print!("{text}"),
                OutputContent::Binary(_) => {
                    return Err(miette::miette!(
                        "backend produced binary output but no --output path was specified"
                    ));
                }
            }
        }
    }

    Ok(())
}

fn write_output_file(path: &std::path::Path, content: &OutputContent) -> miette::Result<()> {
    match content {
        OutputContent::Text(text) => std::fs::write(path, text),
        OutputContent::Binary(data) => std::fs::write(path, data),
    }
    .into_diagnostic()
    .wrap_err_with(|| format!("failed to write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    // ---- Argument parsing ----

    #[test]
    fn cli_defaults() {
        let cli = Cli::try_parse_from(["nxpu", "input.wgsl"]).unwrap();
        assert_eq!(cli.input.unwrap(), PathBuf::from("input.wgsl"));
        assert_eq!(cli.target, "ir-dump");
        assert!(cli.output.is_none());
        assert_eq!(cli.opt_level, OptLevel::O1);
        assert!(!cli.emit_ir);
        assert!(!cli.dry_run);
        assert_eq!(cli.precision, PrecisionPolicy::Auto);
        assert!(cli.calibration_data.is_none());
        assert_eq!(cli.calibration_method, nxpu_opt::CalibrationMethod::MinMax);
        assert!(!cli.verbose);
        assert!(!cli.list_targets);
    }

    #[test]
    fn cli_all_flags() {
        let cli = Cli::try_parse_from([
            "nxpu",
            "model.wgsl",
            "--target",
            "onnx",
            "--output",
            "out.onnx",
            "--opt-level",
            "2",
            "--emit-ir",
            "--precision",
            "f16",
        ])
        .unwrap();
        assert_eq!(cli.input.unwrap(), PathBuf::from("model.wgsl"));
        assert_eq!(cli.target, "onnx");
        assert_eq!(cli.output.unwrap(), PathBuf::from("out.onnx"));
        assert_eq!(cli.opt_level, OptLevel::O2);
        assert!(cli.emit_ir);
        assert_eq!(cli.precision, PrecisionPolicy::Explicit(Precision::F16));
    }

    #[test]
    fn cli_short_flags() {
        let cli =
            Cli::try_parse_from(["nxpu", "in.wgsl", "-t", "tflite", "-o", "out.tflite"]).unwrap();
        assert_eq!(cli.target, "tflite");
        assert_eq!(cli.output.unwrap(), PathBuf::from("out.tflite"));
    }

    #[test]
    fn cli_list_targets_no_input() {
        let cli = Cli::try_parse_from(["nxpu", "--list-targets"]).unwrap();
        assert!(cli.list_targets);
        assert!(cli.input.is_none());
    }

    #[test]
    fn cli_invalid_opt_level() {
        let result = Cli::try_parse_from(["nxpu", "in.wgsl", "--opt-level", "3"]);
        assert!(result.is_err());
    }

    #[test]
    fn cli_invalid_precision() {
        let result = Cli::try_parse_from(["nxpu", "in.wgsl", "--precision", "f64"]);
        assert!(result.is_err());
    }

    // ---- parse_precision ----

    #[test]
    fn precision_valid_values() {
        assert_eq!(parse_precision("keep").unwrap(), PrecisionPolicy::Keep);
        assert_eq!(
            parse_precision("f16").unwrap(),
            PrecisionPolicy::Explicit(Precision::F16)
        );
        assert_eq!(
            parse_precision("bf16").unwrap(),
            PrecisionPolicy::Explicit(Precision::BF16)
        );
        assert_eq!(
            parse_precision("int8").unwrap(),
            PrecisionPolicy::Explicit(Precision::Int8)
        );
        assert_eq!(parse_precision("auto").unwrap(), PrecisionPolicy::Auto);
    }

    #[test]
    fn precision_invalid_value() {
        let err = parse_precision("f64").unwrap_err();
        assert!(err.contains("invalid precision"));
        assert!(err.contains("f64"));
    }

    // ---- parse_opt_level ----

    #[test]
    fn opt_level_valid_values() {
        assert_eq!(parse_opt_level("0").unwrap(), OptLevel::O0);
        assert_eq!(parse_opt_level("1").unwrap(), OptLevel::O1);
        assert_eq!(parse_opt_level("2").unwrap(), OptLevel::O2);
    }

    #[test]
    fn opt_level_invalid_value() {
        let err = parse_opt_level("3").unwrap_err();
        assert!(err.contains("invalid optimization level"));
        assert!(err.contains('3'));
    }

    // ---- Target validation (build_registry) ----

    #[test]
    fn registry_always_has_ir_dump() {
        let registry = build_registry();
        assert!(
            registry.find("ir-dump").is_some(),
            "ir-dump should always be available"
        );
    }

    #[test]
    fn registry_unknown_target_returns_none() {
        let registry = build_registry();
        assert!(registry.find("nonexistent-backend").is_none());
    }

    #[test]
    fn registry_list_targets_includes_ir_dump() {
        let registry = build_registry();
        let targets = registry.list_targets();
        assert!(targets.contains(&"ir-dump"));
    }

    #[cfg(feature = "backend-onnx")]
    #[test]
    fn registry_has_onnx_when_enabled() {
        let registry = build_registry();
        assert!(registry.find("onnx").is_some());
    }

    #[cfg(feature = "backend-tflite")]
    #[test]
    fn registry_has_tflite_when_enabled() {
        let registry = build_registry();
        assert!(registry.find("tflite").is_some());
    }

    // ---- Output path generation ----

    #[test]
    fn multi_file_output_path_derivation() {
        let base = PathBuf::from("/tmp/model.onnx");
        let stem = base
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "output".into());
        let parent = base.parent().unwrap_or_else(|| std::path::Path::new("."));

        let dest = parent.join(format!("{stem}_{}", "weights.bin"));
        assert_eq!(dest, PathBuf::from("/tmp/model_weights.bin"));
    }

    #[test]
    fn multi_file_output_path_no_extension() {
        let base = PathBuf::from("output");
        let stem = base
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "output".into());
        let parent = base.parent().unwrap_or_else(|| std::path::Path::new("."));

        let dest = parent.join(format!("{stem}_{}", "data.bin"));
        assert_eq!(dest, PathBuf::from("output_data.bin"));
    }

    // ---- Error formatting ----

    #[test]
    fn unknown_target_error_lists_available() {
        let registry = build_registry();
        let result = registry.find("bogus");
        assert!(result.is_none());
        let available = registry.list_targets().join(", ");
        let msg = format!("unknown target 'bogus' (available: {available})");
        assert!(msg.contains("bogus"));
        assert!(msg.contains("ir-dump"));
    }

    #[test]
    fn missing_input_error_message() {
        let err = miette::miette!("input file is required (use --list-targets to list backends)");
        let msg = format!("{err}");
        assert!(msg.contains("input file is required"));
        assert!(msg.contains("--list-targets"));
    }

    // ---- Calibration CLI flags ----

    #[test]
    fn cli_calibration_defaults() {
        let cli = Cli::try_parse_from(["nxpu", "input.wgsl"]).unwrap();
        assert!(cli.calibration_data.is_none());
        assert_eq!(cli.calibration_method, nxpu_opt::CalibrationMethod::MinMax);
        assert!(!cli.verbose);
    }

    #[test]
    fn cli_calibration_data_flag() {
        let cli =
            Cli::try_parse_from(["nxpu", "input.wgsl", "--calibration-data", "/tmp/cal_data"])
                .unwrap();
        assert_eq!(
            cli.calibration_data.unwrap(),
            PathBuf::from("/tmp/cal_data")
        );
    }

    #[test]
    fn cli_calibration_method_minmax() {
        let cli =
            Cli::try_parse_from(["nxpu", "input.wgsl", "--calibration-method", "minmax"]).unwrap();
        assert_eq!(cli.calibration_method, nxpu_opt::CalibrationMethod::MinMax);
    }

    #[test]
    fn cli_calibration_method_percentile() {
        let cli = Cli::try_parse_from(["nxpu", "input.wgsl", "--calibration-method", "percentile"])
            .unwrap();
        assert_eq!(
            cli.calibration_method,
            nxpu_opt::CalibrationMethod::Percentile(99.99)
        );
    }

    #[test]
    fn cli_calibration_method_kl() {
        let cli = Cli::try_parse_from([
            "nxpu",
            "input.wgsl",
            "--calibration-method",
            "kl-divergence",
        ])
        .unwrap();
        assert_eq!(
            cli.calibration_method,
            nxpu_opt::CalibrationMethod::KlDivergence
        );
    }

    #[test]
    fn cli_calibration_method_invalid() {
        let result = Cli::try_parse_from(["nxpu", "input.wgsl", "--calibration-method", "invalid"]);
        assert!(result.is_err());
    }

    #[test]
    fn cli_verbose_flag() {
        let cli = Cli::try_parse_from(["nxpu", "input.wgsl", "--verbose"]).unwrap();
        assert!(cli.verbose);
    }

    #[test]
    fn cli_verbose_short_flag() {
        let cli = Cli::try_parse_from(["nxpu", "input.wgsl", "-v"]).unwrap();
        assert!(cli.verbose);
    }

    // ---- parse_calibration_method ----

    #[test]
    fn calibration_method_valid_values() {
        assert_eq!(
            parse_calibration_method("minmax").unwrap(),
            nxpu_opt::CalibrationMethod::MinMax
        );
        assert_eq!(
            parse_calibration_method("percentile").unwrap(),
            nxpu_opt::CalibrationMethod::Percentile(99.99)
        );
        assert_eq!(
            parse_calibration_method("kl-divergence").unwrap(),
            nxpu_opt::CalibrationMethod::KlDivergence
        );
    }

    #[test]
    fn calibration_method_invalid_value() {
        let err = parse_calibration_method("bogus").unwrap_err();
        assert!(err.contains("invalid calibration method"));
        assert!(err.contains("bogus"));
    }
}
