# NxPU

A transpiler that converts WGSL (WebGPU Shading Language) into native code for NPU processors across vendors.

## Overview

NxPU transpiles WGSL compute shaders into native formats for various NPUs (Neural Processing Units), enabling high-efficiency inference and computation pipelines without GPU dependency.

## Target Processors

### Tier 1 — Mobile / Edge

| Vendor    | NPU                  | Devices                       |
| --------- | -------------------- | ----------------------------- |
| Apple     | Apple Neural Engine  | iPhone, Apple Silicon Mac     |
| Samsung   | Exynos NPU           | Galaxy series                 |
| MediaTek  | APU                  | Dimensity series              |
| Google    | TPU / Tensor         | Pixel, Google Cloud           |

### Tier 2 — Desktop / Server / Embedded

| Vendor    | NPU                  | Use Cases                     |
| --------- | -------------------- | ----------------------------- |
| Intel     | Intel NPU            | Core Ultra, Lunar Lake+       |
| AMD       | AMD NPU (XDNA)      | Ryzen AI series               |
| Qualcomm  | Hexagon NPU          | Snapdragon X Elite, etc.      |
| Arm       | Ethos NPU            | Cortex-M / Cortex-A embedded  |
| CEVA      | CEVA NPU             | IoT / Edge AI                 |
| Rockchip  | Rockchip NPU (RKNN)  | RK3588, SBCs                  |

## Architecture

```
┌─────────┐     ┌────────┐     ┌───────────┐     ┌────────────────┐
│  WGSL   │────▶│ Parser │────▶│    IR     │────▶│  NPU Backend   │
│  Source  │     │        │     │ (NxPU-IR) │     │  Code Emitter  │
└─────────┘     └────────┘     └───────────┘     └────────────────┘
                                     │
                                     ├── Apple ANE (CoreML)
                                     ├── Samsung Exynos (ENN SDK)
                                     ├── MediaTek APU (NeuroPilot)
                                     ├── Google TPU (XLA / TFLite)
                                     ├── Intel NPU (OpenVINO)
                                     ├── AMD XDNA (Vitis AI)
                                     ├── Qualcomm Hexagon (QNN SDK)
                                     ├── Arm Ethos (Vela)
                                     ├── CEVA (CDNN)
                                     └── Rockchip RKNN (RKNN-Toolkit)
```

## Project Structure

```
NxPU/
├── crates/
│   ├── nxpu-parser/       # WGSL parser
│   ├── nxpu-ir/           # Intermediate representation
│   ├── nxpu-opt/          # IR optimization passes
│   ├── nxpu-backend-core/ # Backend trait and plugin architecture
│   ├── nxpu-analysis/     # Shared pattern analysis and fusion
│   ├── nxpu-cli/          # Command-line interface
│   ├── nxpu-e2e-tests/    # End-to-end tests
│   └── nxpu-backend-*/    # NPU backend emitters
├── examples/              # WGSL samples
└── docs/                  # Documentation
```

## Getting Started

### Prerequisites

- Rust 1.87+
- Cargo

### Build

```sh
cargo build
```

### Run

```sh
# Transpile a WGSL file for a specific NPU backend
nxpu compile input.wgsl --target apple-ane -o output
```

### Test

```sh
cargo test
```

## License

MIT OR Apache-2.0
