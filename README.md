<p align="center">
  <h1 align="center">NxPU</h1>
  <p align="center">
    WGSL &rarr; NPU transpiler for multi-vendor neural processing units
    <br />
    <a href="https://github.com/m96-chan/NxPU/blob/main/docs/architecture.md"><strong>Architecture</strong></a>
    &middot;
    <a href="https://github.com/m96-chan/NxPU/blob/main/CHANGELOG.md"><strong>Changelog</strong></a>
    &middot;
    <a href="https://github.com/m96-chan/NxPU/issues"><strong>Issues</strong></a>
  </p>
</p>

<p align="center">
  <a href="https://github.com/m96-chan/NxPU/actions"><img src="https://github.com/m96-chan/NxPU/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/m96-chan/NxPU/releases/latest"><img src="https://img.shields.io/github/v/release/m96-chan/NxPU" alt="Release"></a>
  <a href="https://github.com/m96-chan/NxPU/blob/main/LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue" alt="License"></a>
  <img src="https://img.shields.io/badge/rust-1.87%2B-orange" alt="MSRV">
</p>

---

NxPU takes WGSL compute shaders and transpiles them into native formats for NPUs across vendors — ONNX, TFLite, CoreML, StableHLO, and more. Write your ML kernels once in WGSL and deploy to any supported NPU without vendor lock-in.

## Quick Start

```sh
# Install
cargo install --path crates/nxpu-cli

# Transpile WGSL to ONNX
nxpu examples/vecadd.wgsl --target onnx -o vecadd.onnx

# Transpile to TFLite with int8 quantization
nxpu examples/matmul.wgsl --target tflite --precision int8 -o matmul.tflite

# Dump IR for debugging
nxpu examples/relu.wgsl --target ir-dump

# List all available backends
nxpu --list-targets
```

## Supported Backends

| Target | Aliases | Format | Status |
|--------|---------|--------|--------|
| `onnx` | — | `.onnx` (protobuf) | Fully functional |
| `tflite` | `litert` | `.tflite` (FlatBuffers) | Fully functional |
| `coreml` | `apple-ane` | `.mlmodel` | Fully functional |
| `stablehlo` | `xla` | `.mlir` (text) | Fully functional |
| `ir-dump` | `ir` | Text (stdout) | Fully functional |
| `samsung` | `exynos` | ONNX + SDK hint | Stub (SDK required) |
| `mediatek` | `neuropilot` | TFLite + SDK hint | Stub (SDK required) |
| `intel-npu` | `openvino` | ONNX + SDK hint | Stub (SDK required) |
| `amd-xdna` | `amd-npu` | ONNX + SDK hint | Stub (SDK required) |
| `qualcomm` | `hexagon-npu` | ONNX + SDK hint | Stub (SDK required) |
| `arm-ethos` | `ethos-u` | TFLite + SDK hint | Stub (SDK required) |
| `ceva` | `neupro` | ONNX + SDK hint | Stub (SDK required) |
| `rockchip` | `rknn` | ONNX + SDK hint | Stub (SDK required) |

## Recognized ML Patterns

NxPU analyzes WGSL compute kernels and classifies them into ML operations:

| Category | Operations |
|----------|-----------|
| **Linear Algebra** | MatMul, element-wise Add/Sub/Mul |
| **Convolution** | Conv2D |
| **Pooling** | MaxPool |
| **Activation** | ReLU, Tanh, Sigmoid |
| **Normalization** | BatchNorm |
| **Reduction** | ReduceSum |
| **Tensor Ops** | Transpose, Reshape, Concat, Split |
| **Attention** | Scaled dot-product attention |

## Architecture

```
WGSL Source
    |
    v
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐
│  Parser   │───>│    IR    │───>│ Optimize │───>│   Backend    │
│  (naga)   │    │ (SSA IR) │    │  Passes  │    │   Emitter    │
└──────────┘    └──────────┘    └──────────┘    └──────────────┘
                                     |                  |
                              ┌──────┴──────┐    ┌──────┴──────┐
                              │  Const fold │    │ ONNX/TFLite │
                              │  FMA fusion │    │ CoreML/HLO  │
                              │  DCE / CSE  │    │ Vendor SDKs │
                              │  Quantize   │    └─────────────┘
                              └─────────────┘
```

## Project Structure

```
crates/
├── nxpu-parser/         # WGSL parsing via naga, lowering to NxPU IR
├── nxpu-ir/             # Arena-based SSA intermediate representation
├── nxpu-opt/            # Optimization passes (const fold, FMA, DCE, quantize)
├── nxpu-analysis/       # Pattern classification and fusion
├── nxpu-backend-core/   # Backend trait, plugin registry, IR dump
├── nxpu-backend-onnx/   # ONNX protobuf emitter
├── nxpu-backend-tflite/ # TFLite FlatBuffers emitter
├── nxpu-backend-coreml/ # CoreML emitter
├── nxpu-backend-stablehlo/ # StableHLO MLIR emitter
├── nxpu-backend-*/      # Vendor-specific backends (8 vendors)
├── nxpu-cli/            # Command-line interface
└── nxpu-e2e-tests/      # End-to-end numerical correctness tests
examples/                # WGSL sample kernels
docs/                    # Architecture and contributor guides
```

## CLI Reference

```
nxpu [OPTIONS] <INPUT>

Arguments:
  <INPUT>                  Input WGSL file

Options:
  -t, --target <TARGET>    Target backend [default: ir-dump]
  -o, --output <OUTPUT>    Output file path (default: stdout)
      --opt-level <N>      Optimization level: 0, 1, or 2 [default: 1]
      --precision <MODE>   Precision: keep, f16, bf16, int8, auto [default: auto]
      --emit-ir            Dump IR to stderr before backend compilation
      --dry-run            Validate and optimize without output
      --list-targets       List available backends and exit
  -h, --help               Print help
  -V, --version            Print version
```

## Example

**Input** (`examples/vecadd.wgsl`):

```wgsl
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  c[idx] = a[idx] + b[idx];
}
```

**Transpile**:

```sh
$ nxpu examples/vecadd.wgsl --target onnx -o vecadd.onnx
Info: entry point 'main': classified as Add
```

The output `vecadd.onnx` can be loaded directly into any ONNX runtime.

## Building from Source

### Prerequisites

- Rust 1.87+ (edition 2024)

### Build & Test

```sh
cargo build              # Build all crates
cargo test               # Run all tests
cargo clippy              # Lint
cargo fmt --check         # Check formatting
```

### Feature Flags

Backend crates are enabled by default. To build with specific backends only:

```sh
cargo build -p nxpu-cli --no-default-features --features backend-onnx,backend-tflite
```

Available features: `backend-onnx`, `backend-tflite`, `backend-coreml`, `backend-stablehlo`, `backend-samsung`, `backend-mediatek`, `backend-intel`, `backend-amd`, `backend-qualcomm`, `backend-arm-ethos`, `backend-ceva`, `backend-rockchip`

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Licensed under either of

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.
