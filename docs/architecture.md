# NxPU Architecture

NxPU is a WGSL-to-NPU transpiler that converts WebGPU Shading Language compute shaders into formats consumable by neural processing unit hardware and ML inference frameworks.

## Compilation Pipeline

```
WGSL Source
    |
    v
[nxpu-parser] --- naga WGSL frontend
    |
    v
NxPU IR (Module)
    |
    v
[nxpu-opt] --- optimization passes (constant folding, FMA fusion, DCE, quantization, layout)
    |
    v
Optimized IR
    |
    v
[nxpu-backend-*] --- target-specific code generation
    |
    v
Output (.onnx, .tflite, .mlmodel, .stablehlo, vendor binary)
```

### Stage 1: Parsing (`nxpu-parser`)

The parser uses the `naga` crate's WGSL frontend to parse source into naga's IR, then lowers it to NxPU's own IR (`nxpu_ir::Module`). The lowering step (`lower.rs`) maps naga types, globals, expressions, and statements to their NxPU equivalents.

### Stage 2: IR (`nxpu-ir`)

The intermediate representation is an arena-based SSA IR modeled after naga but tailored for NPU lowering. Key types:

- **`Module`** — Top-level container holding types, globals, expressions, functions, and entry points.
- **`Type` / `TypeInner`** — Type system including scalars, vectors, matrices, arrays, structs, and tensors with mixed static/dynamic shapes.
- **`GlobalVariable`** — Module-scope variables with address space, resource binding, and optional memory layout annotations.
- **`Function` / `EntryPoint`** — Functions with SSA expressions and structured control flow.
- **`ComputeGraph`** — Graph-level IR for multi-operation models (DAGs of tensor operations).

### Stage 3: Optimization (`nxpu-opt`)

The `PassManager` runs optimization passes in a fixed-point loop:

| Pass | Description |
|------|-------------|
| `ConstantFolding` | Evaluates constant expressions at compile time |
| `FmaFusion` | Fuses multiply-add sequences into FMA |
| `DeadCodeElimination` | Removes unused expressions and variables |
| `F32ToF16` / `F32ToBf16` / `F32ToInt8` | Precision conversion (quantization) |
| `MixedPrecisionPass` | Per-variable precision based on sensitivity analysis |
| `LayoutTransform` | Assigns memory layouts (NHWC/NCHW) to tensor globals |

### Stage 4: Backend Code Generation

Each backend implements the `Backend` trait from `nxpu-backend-core`:

```rust
pub trait Backend: Debug + Send + Sync {
    fn name(&self) -> &str;
    fn targets(&self) -> &[&str];
    fn compile(&self, module: &Module, opts: &BackendOptions) -> Result<BackendOutput, BackendError>;
    fn preferred_precision(&self) -> Precision { Precision::F32 }
}
```

The `BackendRegistry` dispatches `--target` flags to the appropriate backend.

## Crate Dependency Graph

```
nxpu-ir          (core types, no deps)
  ^
  |
nxpu-parser      (naga -> nxpu-ir lowering)
nxpu-opt         (optimization passes)
nxpu-backend-core (Backend trait)
  ^
  |
nxpu-backend-onnx    (ONNX protobuf emission + IR analysis)
nxpu-backend-tflite  (TFLite FlatBuffer emission)
nxpu-backend-coreml  (Core ML protobuf emission)
nxpu-backend-stablehlo (StableHLO text emission)
  ^
  |
nxpu-backend-{vendor} (thin vendor wrappers)
  ^
  |
nxpu-cli             (CLI entry point, registers all backends)
```

## IR Analysis (`nxpu-backend-onnx::analyze`)

The analysis module classifies entry points into known patterns:

- **`MatMul`** — Loop + accumulation over 2 input arrays + 1 output array
- **`ElementWise`** — Binary operation on arrays (Add, Sub, Mul, Div)

This classification drives both ONNX and TFLite code generation.

## Design Decisions

1. **Arena-based IR** — Types are deduplicated via `UniqueArena`, expressions stored in `Arena` with handles for O(1) lookup.
2. **Pattern-based lowering** — Rather than general-purpose code generation, backends recognize high-level patterns (MatMul, elementwise) and emit optimal operator sequences.
3. **Shared analysis** — The ONNX backend's `analyze` module is reused by TFLite and vendor backends, avoiding duplication.
4. **Plugin architecture** — New backends are added by implementing the `Backend` trait and registering in `main.rs`.
