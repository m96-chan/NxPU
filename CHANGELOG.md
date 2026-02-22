# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-22

### Added
- WGSL-to-NPU transpiler with arena-based SSA IR.
- `nxpu-parser`: WGSL parsing via naga with lowering to NxPU IR.
- `nxpu-ir`: Intermediate representation with types, expressions, statements, and compute graphs.
- `nxpu-opt`: Optimization pass framework with fixed-point iteration.
  - Constant folding (binary, unary, math functions).
  - FMA fusion (`a * b + c` to `fma(a, b, c)`) with use-count safety.
  - Dead code elimination (unused emits, dead stores, unreferenced locals).
  - IR validation pass (operand bounds, workgroup size checks).
  - Quantization passes (F32 to F16, BF16, Int8, mixed-precision).
  - Shape inference and memory layout assignment (NHWC/NCHW).
- `nxpu-analysis`: Pattern classification (MatMul, Conv2D, ElementWise, Pool, Activation, Reduce, Transpose, Reshape, Normalization, Concat, Split, Attention) and fusion (Conv+BatchNorm, Base+Activation) with tensor name connectivity.
- `nxpu-backend-core`: Backend trait, plugin registry, IR dump backend.
- Backend emitters: ONNX, TFLite, CoreML, StableHLO.
- Vendor backends: Samsung, MediaTek, Intel, AMD, Qualcomm, Arm Ethos, CEVA, Rockchip.
- `nxpu-cli`: Command-line interface with target selection, optimization levels, precision control, `--list-targets`, and `RUST_LOG` support.
- End-to-end tests covering all backends, error paths, and numerical validation.

[Unreleased]: https://github.com/m96-chan/NxPU/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/m96-chan/NxPU/releases/tag/v0.1.0
