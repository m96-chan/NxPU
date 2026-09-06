# Measured operator matrices

Each file here is what **one phone** answered when it was asked, one operator at
a time, whether it would take that operator. They are measurements, not
documentation.

**A cell is a claim about a model, not about an operator.** It is silent on
several axes at once — the shape the operator was given, whether it was fused
with anything, how fast it ran, and every phone that was not this one. To those,
add two more:

> **How an operand is supplied.** Every weight in the converter-built reference
> models is a compile-time constant, because that is what a converter emits. A
> driver can require that: on MT6899, `mtk-neuron_shim` accelerates a
> convolution whose filter is a constant and refuses the same convolution, at
> the same shapes, whose filter is a graph input.
>
> **What occupying that engine costs the rest of the device.** A `✓` under
> `gpu` says the delegate took the operator. It does not say the phone stayed
> usable while it did. The GPU is the same engine the display composites on, so
> a long compute workload there contends with everything drawing on screen and
> can stall it; the NPU is a dedicated part and does not. That is why "send
> everything the NPU will take" is a defensible policy even where the GPU is
> the faster engine per operator — and it is not visible in any cell here, on
> any row, for either column.

So `CONV_2D | float32 | accelerated` means *this driver took that model*. It
does not mean the driver takes every CONV_2D, and reading it that way cost a day
of device sweeps before the rule was found. Two matrices can disagree about
"CONV_2D on `mtk-neuron_shim`" and both be right, because they handed it
different convolutions.

**They are not vendor documentation, and they are not a substitute for it.** A
vendor's table describes a family of parts across driver versions and shapes; a
file in this directory describes one handset, on one driver, on one day, at one
shape. Where the two disagree, that is a fact worth chasing — not a licence to
overwrite either with the other.

**Nothing here is written into `crates/nxpu-backend-*/src/support.rs`, and
nothing here should be.** Those tables are the compiler's beliefs about eight
vendors' silicon; these are evidence about one device on a desk. Turning the
second into the first is a decision about how much a single measurement is
allowed to say, and it has not been made. A `support.rs` that had quietly
absorbed this table would be worse than the hand-written one it replaced,
because it would look measured while being narrower than the datasheet it lost.

## What is here

| file | device | SoC | drivers |
| --- | --- | --- | --- |
| `xiaomi-2511fpc34g.json` | Xiaomi 2511FPC34G | Mediatek MT6899 | `mtk-neuron_shim`, `mtk-mdla_shim`, `mtk-dsp_shim`, `nnapi-reference` |

The header of every file written here names the device, the SoC, the Android
SDK level and the DroidRunner build that answered, because a table nobody can
trace back to a build is a table nobody should stand behind.

Two accelerators a phone has are **not** in these files, because they are not
NNAPI drivers and `droidrunner-device devices` does not list them: `gpu`,
TFLite's own GPU delegate, and `qnn-htp`, Qualcomm's runtime. On MT6899 the GPU
accepts more of what this compiler emits than any NNAPI driver does — 18 of 19
against the best driver's 15 — so a reader treating this table as the list of
ways to reach the silicon is missing the widest one. Pass them to the workflow's
`drivers` input by name.

## Regenerating one

Run the **Operator matrix** workflow (`.github/workflows/op-matrix.yml`) from
the Actions tab. It is dispatch-only: the device job costs a real phone's time.

It does three things, in three jobs, on two kinds of machine:

1. **`ubuntu-latest`** builds the models. One set comes from TensorFlow's
   converter — one operator per model, at conventional NHWC shapes. The other
   is the output of `nxpu --target tflite` for every kernel in `examples/`,
   taken exactly as the compiler emitted it.
2. **The phone** runs every model twice: once with no accelerator at all, and
   once pinned to each driver asked for. `iterations: 0` throughout — the
   question is whether the graph was accepted, and that is answered as soon as
   tensors are allocated.
3. **`ubuntu-latest`** reduces the two sweeps to `matrix.md` and `matrix.json`,
   prints the table into the run summary, and compares it against the file
   committed here for the same phone.

To commit a new table, take `matrix.json` from the run's `op-matrix` artifact
and write it to the path `tools/op-matrix/baseline-path.py` names for it. The
next run is compared against it, so a driver that stops taking an operator turns
the build red. When a cell moves, say in the commit message what moved and why
it was the driver's doing and not ours.

## The two halves of the table, and why they are separate

**"The driver, asked directly"** is a statement about the silicon. One operator
per model, built by the same converter everyone else's models come from. If a
cell here says no, the driver will not take that operator at that precision.

**"What NxPU emits, asked the same way"** is a statement about *us*. The models
are the compiler's own bytes. A cell here can be worse than the reference cell
for the same operator for reasons that have nothing to do with the driver — a
shape it emits, an option it leaves at zero, a tensor it does not write.

Keeping them apart is the whole point. Together they say which of the two is at
fault; either alone says only that something is.

## The control column, and what "excluded" means

Every model is also run with **no accelerator at all**. If it will not run on
the CPU either, the model is broken, its row is excluded, and the table says so
— it is never filed as an operator some driver refused. A defect of ours must
not end up in a table other people compile against.

For the reference half an exclusion is a bug in the generator. For the NxPU half
it is a finding: a model this compiler emitted that no TFLite runtime will load.
`matrix.json` carries the runtime's own words for each under `controlDetail`.

## What a cell says

| mark | meaning |
| --- | --- |
| `✓` | the delegate took every node |
| `~` | it took some of them — for a one-operator model, read the JSON |
| `✗` | it took nothing; the CPU ran the model |
| `·` | this phone has no such driver |
| `!` | the run failed for a reason that is neither of those |
| `—` | excluded: the model did not run on the CPU either |

`matrix.json` carries the same cells with the device's own attribution beside
them — `executedBy`, naming both the delegate and the driver it was pinned to,
and whether the phone was thermally stable while it answered.

## What a cell does not say

- **Nothing about a different shape.** Each operator is measured at one shape.
  Acceleration is routinely conditional on rank, kernel size, channel count or
  stride, and a driver that takes `CONV_2D` here may refuse yours. This is the
  sharpest limitation of the whole exercise, and the reason the two halves
  exist: NxPU emits rank-1 and rank-2 tensors where a vendor's table was written
  against rank-4.
- **Nothing about fused patterns.** Drivers match patterns, not just operators.
  An operator refused alone can still run inside a fusion the driver
  recognises, and one accepted alone can still be refused in a graph whose
  layout does not suit it. `✓` is not a promise about your network.
- **Nothing about speed.** The sweep asks only whether the graph was accepted.
- **Nothing about arithmetic.** `accelerator` says who executed the graph, not
  what precision they used to do it. On this Hexagon an f32 `ADD` is not
  bit-exact with the CPU; see the notes in `.github/workflows/device-tflite.yml`.
- **Nothing about a phone you did not run it on.**

## Compatibility with DroidRunner

`matrix.json` is DroidRunner's `matrix.json` schema 1, field for field, so a
table produced here can be sent to [m96-chan/DroidRunner](https://github.com/m96-chan/DroidRunner)
and compared against theirs with either repository's `compare.py`. The fields
this repository adds to a row — `source`, `builtin`, `kernel`, `nxpuOp` — are
additions, which that schema allows.

Rows are keyed on `(operator, precision)`, so the NxPU half writes its operator
as `ADD (nxpu:vecadd)`: two kernels can lower to the same builtin and are not
the same test. Device identity is `(manufacturer, model)` and deliberately not
the SoC — a matrix taken before DroidRunner reported an SoC does not carry one,
and keying on it refuses to compare a phone against itself across exactly the
build where the comparison matters.
