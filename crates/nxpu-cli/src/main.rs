use clap::Parser;

/// NxPU — WGSL to NPU transpiler
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Input WGSL file
    input: std::path::PathBuf,

    /// Target NPU backend
    #[arg(short, long)]
    target: String,

    /// Output path
    #[arg(short, long)]
    output: Option<std::path::PathBuf>,
}

fn main() {
    let _cli = Cli::parse();
    eprintln!("NxPU is not yet implemented");
    std::process::exit(1);
}
