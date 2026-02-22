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
[nxpu-opt] --- optimization passes (validation, constant folding, FMA fusion, DCE, quantization, layout)
    |
    v
[nxpu-analysis] --- pattern classification + fusion
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
- **`ComputeGraph`** — Graph-level IR for DAG-of-operations representation (multi-operation models).

### Stage 3: Optimization (`nxpu-opt`)

The `PassManager` runs optimization passes in a fixed-point loop (up to 10 iterations). All passes implement the `Pass` trait and report whether they modified the module. The pass manager logs iteration progress via `log::debug!`.

| Pass | Level | Description |
|------|-------|-------------|
| `IrValidation` | O1+ | Validates expression operand bounds, type handle bounds, workgroup sizes > 0 |
| `ConstantFolding` | O1+ | Evaluates constant expressions at compile time (including global expressions) |
| `FmaFusion` | O1+ | Fuses multiply-add sequences into FMA (with use-count safety) |
| `DeadCodeElimination` | O1+ | Removes unused expressions, dead stores, and unreferenced local variables |
| `F32ToF16` / `F32ToBf16` / `F32ToInt8` | Manual | Precision conversion (quantization) |
| `MixedPrecisionPass` | Manual | Per-variable precision based on sensitivity analysis |
| `ShapeInference` | Manual | Infers tensor shapes from array types and naming conventions |
| `LayoutTransform` | Manual | Assigns memory layouts (NHWC/NCHW) to tensor globals |

### Stage 4: Pattern Analysis (`nxpu-analysis`)

The analysis module classifies entry points into known kernel patterns:

| Pattern | Description |
|---------|-------------|
| `MatMul` | Loop + accumulation over 2 input arrays + 1 output array |
| `ElementWise` | Binary operation on arrays (Add, Sub, Mul, Div) |
| `Conv2D` | 2D convolution: nested loops + kernel window + accumulation |
| `Pool` | Pooling: nested loops + reduction over spatial window (Max, Avg) |
| `Activation` | Unary activation function (Relu, Sigmoid, Tanh, Softmax) |
| `Normalization` | Batch normalization: mean + variance + scale + bias |
| `Reduce` | Reduction over an axis (Sum, Mean, Max, Min) |
| `Transpose` | Permute tensor axes |
| `Reshape` | Change tensor shape without data copy |
| `Concat` | Concatenation of multiple inputs along an axis |
| `Split` | Split a single input into multiple outputs along an axis |
| `Attention` | Scaled dot-product attention (Q, K, V) |
| `Unknown` | Unrecognized pattern |

#### Fusion

After classification, `fuse_patterns()` performs greedy adjacent fusion:

- **Conv2D + Normalization** -> `ConvBatchNorm` (if output/input tensor names match)
- **Any + Activation(Relu)** -> `WithActivation { base, Relu }` (if tensor names connect)

Fusion is gated on tensor name connectivity: the producer's output tensor name must match the consumer's input tensor name.

### Stage 5: Backend Code Generation

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
nxpu-analysis    (pattern classification + fusion)
  ^
  |
nxpu-backend-onnx    (ONNX protobuf emission)
nxpu-backend-tflite  (TFLite FlatBuffer emission)
nxpu-backend-coreml  (Core ML protobuf emission)
nxpu-backend-stablehlo (StableHLO text emission)
  ^
  |
nxpu-backend-{vendor} (thin vendor wrappers: samsung, mediatek, intel, amd, qualcomm, arm-ethos, ceva, rockchip)
  ^
  |
nxpu-cli             (CLI entry point, registers all backends)
```

## Design Decisions

1. **Arena-based IR** — Types are deduplicated via `UniqueArena`, expressions stored in `Arena` with handles for O(1) lookup.
2. **Pattern-based lowering** — Rather than general-purpose code generation, backends recognize high-level patterns (MatMul, Conv2D, elementwise, etc.) and emit optimal operator sequences.
3. **Shared analysis** — The `nxpu-analysis` crate provides pattern classification and fusion used by all backends, avoiding duplication.
4. **Plugin architecture** — New backends are added by implementing the `Backend` trait and registering in `main.rs`.
5. **Workspace dependencies** — Common dependencies (thiserror, prost, naga, etc.) are centralized in root `[workspace.dependencies]` for version consistency.
