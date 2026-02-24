# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-02-24

### Added
- Embed constant weights from `GlobalVariable` initializers into ONNX output.
- Real vendor NPU backends with operator support matrices (Samsung Exynos, MediaTek Dimensity, Intel OpenVINO, AMD XDNA, Qualcomm Hexagon, Arm Ethos, CEVA NeuPro, Rockchip RKNN).
- `OperatorSupport` trait and `validate_patterns` helper in `nxpu-backend-core`.
- Comprehensive test coverage for all vendor backends.
- Fuzz testing targets for WGSL parser (`fuzz_parse`, `fuzz_lower`).
- Display and validation tests for backend-core types.
- Example kernels: `conv2d_5x5.wgsl`, `vecadd_const.wgsl`.

### Fixed
- Extract loop bounds from naga 28 `If`/`else{Break}` pattern for Conv2D shape inference.
- Use `let-else` instead of let-chains for MSRV 1.87 compatibility.

### Changed
- Classify 1-input + embedded weight operations as `ElementWise` for end-to-end ONNX path.

## [0.2.0] - 2026-02-22

### Added
- Numerical correctness tests for tanh, conv2d, maxpool, and attention ops (#101).
- CLI unit tests for argument parsing, precision, and opt-level validation (#99).
- Workspace-level lint configuration with clippy pedantic (#98).
- `cargo-deny` license audit in CI (#97).
- Feature flags for backend selection (`backend-onnx`, `backend-tflite`, etc.) (#95).

### Fixed
- Move miette `fancy` feature to CLI-only to reduce dependency footprint (#96).
- Fix MSRV CI and doc env for Windows (#102, #103).
- Fix security audit CI job permissions.
- Fix formatting across workspace.
- Resolve release quality issues #85-#92.

### Changed
- Annotate dev-only workspace dependencies (#93).

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

[Unreleased]: https://github.com/m96-chan/NxPU/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/m96-chan/NxPU/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/m96-chan/NxPU/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/m96-chan/NxPU/releases/tag/v0.1.0
