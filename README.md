<div align="center">

```
    _   __     ____  __  __
   / | / /  __/ __ \/ / / /
  /  |/ / |/_/ /_/ / / / /
 / /|  />  </ ____/ /_/ /
/_/ |_/_/|_/_/    \____/
```

**WGSL &rarr; NPU transpiler for multi-vendor neural processing units**

Write ML kernels once in WGSL. Deploy to any NPU.

[![CI](https://github.com/m96-chan/NxPU/actions/workflows/ci.yml/badge.svg)](https://github.com/m96-chan/NxPU/actions)
[![Coverage](https://codecov.io/gh/m96-chan/NxPU/graph/badge.svg)](https://codecov.io/gh/m96-chan/NxPU)
[![Release](https://img.shields.io/github/v/release/m96-chan/NxPU?color=%23007ec6)](https://github.com/m96-chan/NxPU/releases/latest)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](#license)
[![Rust](https://img.shields.io/badge/rust-1.87%2B-f74c00)](https://www.rust-lang.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](CONTRIBUTING.md)

[Architecture](docs/architecture.md) · [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md)

</div>

---

## Why NxPU?

NPU hardware is fragmented — every vendor ships a different SDK, model format, and toolchain. NxPU solves this by providing a **single compilation pipeline** from WGSL compute shaders to native NPU formats.

- **One language, many targets** — Write WGSL once, emit ONNX, TFLite, CoreML, StableHLO, or vendor-specific formats
- **Pattern recognition** — Automatically classifies compute kernels into MatMul, Conv2D, Attention, and 10+ other ML operations
- **Optimization passes** — Constant folding, FMA fusion, dead code elimination, common subexpression elimination, and quantization
- **Vendor-aware validation** — Operator support matrices for 8 NPU vendors with native/emulated/unsupported classification
- **Pluggable backends** — Add new NPU targets by implementing a single trait

## Quick Start

```sh
cargo install --path crates/nxpu-cli
```

```sh
# Transpile WGSL → ONNX
nxpu examples/vecadd.wgsl --target onnx -o vecadd.onnx

# Transpile → TFLite with int8 quantization
nxpu examples/matmul.wgsl --target tflite --precision int8 -o matmul.tflite

# Dump the intermediate representation
nxpu examples/relu.wgsl --target ir-dump

# List all available backends
nxpu --list-targets
```

## Example

<table>
<tr>
<td> <b>Input</b> — <code>examples/vecadd.wgsl</code> </td>
<td> <b>Output</b> </td>
</tr>
<tr>
<td>

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

</td>
<td>

```sh
$ nxpu examples/vecadd.wgsl --target onnx -o vecadd.onnx
Info: entry point 'main': classified as Add
```

The output `vecadd.onnx` can be loaded directly into any ONNX runtime.

</td>
</tr>
</table>

## Supported Backends

| Target | Aliases | Format | Status |
|:-------|:--------|:-------|:------:|
| `onnx` | — | `.onnx` (protobuf) | :white_check_mark: |
| `tflite` | `litert` | `.tflite` (FlatBuffers) | :white_check_mark: |
| `coreml` | `apple-ane` | `.mlmodel` | :white_check_mark: |
| `stablehlo` | `xla` | `.mlir` (text) | :white_check_mark: |
| `ir-dump` | `ir` | Text (stdout) | :white_check_mark: |
| `intel-npu` | `openvino` | OpenVINO IR `.xml` + `.onnx` | :white_check_mark: |
| `amd-xdna` | `amd-npu` | ONNX + XDNA metadata | :white_check_mark: |
| `arm-ethos` | `ethos-u` | TFLite + optional Vela | :white_check_mark: |
| `samsung` | `exynos` | ONNX + ONE toolchain hints | :white_check_mark: |
| `qualcomm` | `hexagon-npu` | ONNX + QNN SDK hints | :white_check_mark: |
| `mediatek` | `neuropilot` | TFLite + NeuroPilot hints | :white_check_mark: |
| `rockchip` | `rknn` | ONNX + RKNN Toolkit hints | :white_check_mark: |
| `ceva` | `neupro` | ONNX + CDNN compiler hints | :white_check_mark: |

Each vendor backend includes an **operator support matrix** that validates patterns against the target NPU's capabilities, emitting warnings for emulated or unsupported operations.

<details>
<summary><b>Vendor backend details</b></summary>

| Vendor | NPU Hardware | Native Precision | Output Format | SDK Toolchain |
|:-------|:-------------|:-----------------|:--------------|:--------------|
| **Intel** | Meteor Lake / Arrow Lake NPU | F16 | OpenVINO IR v11 (`.xml` + `.bin`) + ONNX fallback | OpenVINO (`ov::Core::read_model`) |
| **AMD** | Ryzen AI XDNA | Int8, F16 | ONNX with XDNA metadata props | Vitis AI EP / ONNX Runtime |
| **Arm** | Ethos-U55 (128 MAC) / U65 (512 MAC) | Int8 (U55), Int8+Int16 (U65) | TFLite + optional Vela compilation | `ethos-u-vela` compiler |
| **Samsung** | Exynos NPU | F16, Int8 | ONNX | ONE toolchain (`one-import-onnx`, `one-codegen`) |
| **Qualcomm** | Hexagon NPU | Int8, F16 | ONNX | QNN SDK (`qnn-onnx-converter`) |
| **MediaTek** | Dimensity APU | Int8, F16 | TFLite | NeuroPilot SDK (`ncc-tflite`) |
| **Rockchip** | RK3588 NPU (3 TOPS) | Int8, F16 | ONNX | RKNN Toolkit 2 (Python API) |
| **CEVA** | NeuPro-S | Int8 | ONNX | CDNN compiler (`cdnn_cli`) |

</details>

## Recognized ML Patterns

NxPU analyzes WGSL compute kernels and classifies them into ML operations:

| Category | Operations |
|:---------|:-----------|
| Linear Algebra | MatMul, element-wise Add / Sub / Mul |
| Convolution | Conv2D |
| Pooling | MaxPool |
| Activation | ReLU, Tanh, Sigmoid |
| Normalization | BatchNorm |
| Reduction | ReduceSum |
| Tensor Ops | Transpose, Reshape, Concat, Split |
| Attention | Scaled dot-product attention |

## Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │             Optimization Passes             │
                         │  ┌───────┐ ┌───────┐ ┌───────┐ ┌────────┐  │
                         │  │ Const │ │  FMA  │ │ DCE / │ │Quantize│  │
                         │  │ Fold  │ │Fusion │ │  CSE  │ │        │  │
                         │  └───────┘ └───────┘ └───────┘ └────────┘  │
                         └──────────────────┬──────────────────────────┘
                                            │
  ┌──────────┐     ┌──────────┐     ┌───────┴──┐     ┌──────────────┐
  │   WGSL   │────>│  Parser  │────>│  SSA IR  │────>│   Backend    │
  │  Source   │     │  (naga)  │     │          │     │   Emitter    │
  └──────────┘     └──────────┘     └──────────┘     └──────┬───────┘
                                                            │
                                          ┌─────────────────┼─────────────────┐
                                          │                 │                 │
                                     ┌────┴───┐       ┌─────┴────┐      ┌────┴────┐
                                     │  ONNX  │       │  TFLite  │      │ CoreML  │
                                     │  HLO   │       │          │      │ Vendors │
                                     └────────┘       └──────────┘      └─────────┘
```

## Project Structure

```
crates/
├── nxpu-parser/              WGSL parsing via naga, lowering to NxPU IR
├── nxpu-ir/                  Arena-based SSA intermediate representation
├── nxpu-opt/                 Optimization passes (const fold, FMA, DCE, quantize)
├── nxpu-analysis/            Pattern classification and fusion
├── nxpu-backend-core/        Backend trait, plugin registry, IR dump
├── nxpu-backend-onnx/        ONNX protobuf emitter
├── nxpu-backend-tflite/      TFLite FlatBuffers emitter
├── nxpu-backend-coreml/      CoreML emitter
├── nxpu-backend-stablehlo/   StableHLO MLIR emitter
├── nxpu-backend-*/           Vendor-specific backends (8 vendors)
├── nxpu-cli/                 Command-line interface
└── nxpu-e2e-tests/           End-to-end numerical correctness tests
examples/                     WGSL sample kernels (14 examples)
docs/                         Architecture and contributor guides
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

## Building from Source

**Prerequisites:** Rust 1.87+ (edition 2024)

```sh
cargo build            # Build all crates
cargo test             # Run all tests
cargo clippy           # Lint
cargo fmt --check      # Check formatting
```

To build with specific backends only:

```sh
cargo build -p nxpu-cli --no-default-features --features backend-onnx,backend-tflite
```

<details>
<summary><b>Available feature flags</b></summary>

`backend-onnx` · `backend-tflite` · `backend-coreml` · `backend-stablehlo` · `backend-samsung` · `backend-mediatek` · `backend-intel` · `backend-amd` · `backend-qualcomm` · `backend-arm-ethos` · `backend-ceva` · `backend-rockchip`

</details>

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and [docs/adding-a-backend.md](docs/adding-a-backend.md) for backend implementation guides.

## License

Licensed under either of

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.
