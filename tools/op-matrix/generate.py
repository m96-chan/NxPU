#!/usr/bin/env python3
"""The models a support sweep asks about, one operator at a time.

NxPU carries hand-written operator support tables in
`crates/nxpu-backend-*/src/support.rs` — every entry read off a datasheet and
never once checked against silicon. This builds the models that check them.

A whole network cannot answer the question. TFLite partitions a graph, and one
refused node pushes its entire partition back to the CPU, so a model that runs
on the CPU proves some operator wrong without saying which. A model containing
exactly one operator has nowhere to hide the answer: the driver took it or it
did not.

Two sources of models, because they answer two different questions.

`reference` — built by TensorFlow's own converter at conventional NHWC shapes,
one operator per model, restricted to the builtin operators NxPU's TFLite
backend actually emits. This measures the driver.

It does not measure "whether this silicon takes CONV_2D", which is what that
sentence used to say and what a reader will assume anyway. Every weight below
is a `tf.constant`, and on MT6899 that is the difference between accelerated
and refused: `mtk-neuron_shim` takes a convolution whose filter is a constant
and refuses the same convolution, at the same shapes, whose filter is a graph
input. So the row measures CONV_2D *with a constant filter*, which is one
instance of the operator and not the operator.

That is not a flaw in the row. It is the reason the two tables exist and are
kept apart: a driver's answer is about the model it was handed, and two
matrices can disagree about "CONV_2D" while both being correct because they
handed it different models.

`nxpu` — the bytes `nxpu --target tflite` produces for the kernels in
`examples/`, ingested from a directory the caller has already filled. This
measures *us*: whether the models this compiler emits are ones the driver will
take. A row here is only comparable to the reference row for the same operator,
and the two are kept apart in the table for that reason.

Building the reference models by hand instead of with the converter is a
tempting way to drop the TensorFlow dependency, and it is the one shortcut that
would destroy the result. A graph a driver rejects because our quantization was
invalid is indistinguishable, from outside, from an operator the driver does
not support — which is precisely the wrong answer this exists to stop us
publishing. So the reference models come from the converter everyone else's
models come from, and each is read back and checked to contain the single
operator it claims.

Usage: generate.py --out DIR [--nxpu DIR] [--only OP,OP]

Shape and much of the reasoning are ported from m96-chan/DroidRunner's
`tools/op-matrix/generate.py`. The operator set, the NxPU model source and the
flatbuffer reader below are ours.
"""

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow.lite.python import schema_py_generated as schema  # noqa: E402

SPATIAL = (1, 16, 16, 8)
FLAT = (1, 64)
BATCHED = (1, 32, 64)


def const(*shape, seed=0):
    """Weights that are the same on every machine that runs this.

    A matrix built from models that differ run to run cannot be compared with
    the one published last month, which is most of the point of having one.
    """
    return tf.constant(
        np.random.default_rng(seed).standard_normal(shape).astype("float32") * 0.1
    )


# The builtin operators `crates/nxpu-backend-tflite/src/lower.rs` emits, each as
# a graph containing exactly that one operator. CUSTOM is emitted too and is
# absent on purpose: it carries no builtin code for a driver to have an opinion
# about, and NxPU currently writes it with no custom_code at all.
#
# Each entry is (input shapes, graph, the name `support.rs` keys on). The third
# field is what makes this table NxPU's rather than a generic one: it is the
# join between a measurement and the claim it bears on. Nothing here reads
# `support.rs` — the tables stay hand-written until somebody decides, on
# purpose, how a measurement becomes a compiler's belief.
OPS = {
    "ADD": ([SPATIAL, SPATIAL], lambda a, b: a + b, "Add"),
    "SUB": ([SPATIAL, SPATIAL], lambda a, b: a - b, "Sub"),
    "MUL": ([SPATIAL, SPATIAL], lambda a, b: a * b, "Mul"),
    # Bare, with no guard against a small divisor: anything that keeps the
    # denominator away from zero is another node, and a two-node model cannot
    # say which node was refused.
    "DIV": ([SPATIAL, SPATIAL], lambda a, b: a / b, "Div"),
    "CONV_2D": ([SPATIAL], lambda x: tf.nn.conv2d(
        x, const(3, 3, 8, 16, seed=1), strides=1, padding="SAME")
        + const(16, seed=2), "Conv"),
    "AVERAGE_POOL_2D": ([SPATIAL], lambda x: tf.nn.avg_pool2d(x, 2, 2, "VALID"),
                        "AveragePool"),
    "MAX_POOL_2D": ([SPATIAL], lambda x: tf.nn.max_pool2d(x, 2, 2, "VALID"),
                    "MaxPool"),
    "MEAN": ([SPATIAL], lambda x: tf.reduce_mean(x, axis=[1, 2], keepdims=True),
             "ReduceMean"),
    "SUM": ([SPATIAL], lambda x: tf.reduce_sum(x, axis=[1, 2], keepdims=True),
            "ReduceSum"),
    "REDUCE_MAX": ([SPATIAL], lambda x: tf.reduce_max(x, axis=[1, 2], keepdims=True),
                   "ReduceMax"),
    "REDUCE_MIN": ([SPATIAL], lambda x: tf.reduce_min(x, axis=[1, 2], keepdims=True),
                   "ReduceMin"),
    "SOFTMAX": ([FLAT], tf.nn.softmax, "Softmax"),
    "LOGISTIC": ([SPATIAL], tf.sigmoid, "Sigmoid"),
    "TANH": ([SPATIAL], tf.tanh, "Tanh"),
    "RELU": ([SPATIAL], tf.nn.relu, "Relu"),
    "CONCATENATION": ([SPATIAL, SPATIAL], lambda a, b: tf.concat([a, b], axis=3),
                      "Concat"),
    "RESHAPE": ([SPATIAL], lambda x: tf.reshape(x, (1, -1)), "Reshape"),
    "TRANSPOSE": ([SPATIAL], lambda x: tf.transpose(x, [0, 3, 1, 2]), "Transpose"),
    "BATCH_MATMUL": ([BATCHED], lambda x: tf.matmul(x, const(1, 64, 32, seed=3)),
                     "MatMul"),
    "SPLIT": ([SPATIAL], lambda x: tf.split(x, 2, axis=3), "Split"),
    "GATHER": ([SPATIAL], lambda x: tf.gather(
        x, tf.constant([0, 2, 4, 6], dtype=tf.int32), axis=3), "Gather"),
    "SCATTER_ND": ([(4, 8)], lambda u: tf.scatter_nd(
        tf.constant([[0], [2], [4], [6]], dtype=tf.int32), u, (16, 8)), "ScatterND"),
}

PRECISIONS = ("float32", "int8")

# Every builtin `lower.rs` can write. Kept beside OPS so the two can be compared
# by anything that cares, and so a builtin gaining a lowering without gaining a
# model here is visible rather than silently unmeasured.
EMITTED_BUILTINS = set(OPS) | {"CUSTOM"}


def representative(shapes):
    def dataset():
        rng = np.random.default_rng(7)
        for _ in range(16):
            yield [rng.standard_normal(s).astype("float32") for s in shapes]

    return dataset


def convert(shapes, build, precision):
    fn = tf.function(build).get_concrete_function(
        *[tf.TensorSpec(s, tf.float32) for s in shapes]
    )
    converter = tf.lite.TFLiteConverter.from_concrete_functions([fn])
    if precision == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative(shapes)
        # Refuse a float fallback rather than accept one: a graph that quietly
        # keeps a float kernel is not the int8 row it would be filed under.
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        # NNAPI accepts per-channel weights for convolutions and not for dense
        # layers, and the converter quantizes both per channel by default. Every
        # driver on every phone DroidRunner tried answered the resulting model
        # with ANEURALNETWORKS_BAD_DATA — one operator reading as broken
        # everywhere, which is what a defect of ours looks like next to a
        # driver's opinion. Convolutions keep their per-channel scales.
        converter._experimental_disable_per_channel_quantization_for_dense_layers = True
    return converter.convert()


BUILTIN_NAMES = {
    value: name
    for name, value in vars(schema.BuiltinOperator).items()
    if isinstance(value, int)
}


def operators_in(blob):
    """The builtin operators a model contains, read out of the flatbuffer.

    Deliberately not through `tf.lite.Interpreter`. An interpreter has to be
    able to *load* the model to answer, and a good half of what NxPU emits
    today cannot be loaded at all — a CONV_2D with a zero stride, a SUM with no
    axes tensor. Those are exactly the rows the sweep exists to record, and
    identifying them by name is the difference between "the driver refused
    CONV_2D" and "we never gave it one".

    It is also the only reading that is a fact about the file rather than about
    the host: an interpreter applies XNNPACK while allocating, and every model
    then appears to contain a node called DELEGATE.
    """
    model = schema.Model.GetRootAs(bytearray(blob), 0)
    codes = [model.OperatorCodes(i) for i in range(model.OperatorCodesLength())]

    def name_of(code):
        # `builtin_code` is the field that counts; `deprecated_builtin_code` is
        # an int8 and saturates at 127, so anything above that reads as
        # PLACEHOLDER_FOR_GREATER_OP_CODES when the two disagree.
        builtin = code.BuiltinCode() or code.DeprecatedBuiltinCode()
        return BUILTIN_NAMES.get(builtin, f"opcode-{builtin}")

    names = []
    for s in range(model.SubgraphsLength()):
        graph = model.Subgraphs(s)
        for i in range(graph.OperatorsLength()):
            names.append(name_of(codes[graph.Operators(i).OpcodeIndex()]))
    return names


def reference_models(wanted, out):
    """One converter-built model per (operator, precision)."""
    generated, skipped = [], []
    for operator in wanted:
        shapes, build, nxpu_op = OPS[operator]
        for precision in PRECISIONS:
            name = f"ref-{operator.lower()}-{precision}"
            try:
                blob = convert(shapes, build, precision)
                emitted = operators_in(blob)
            except Exception as failure:  # the converter is the authority
                skipped.append({
                    "id": name, "operator": operator, "precision": precision,
                    "source": "reference",
                    "reason": f"the converter would not produce it: "
                              f"{type(failure).__name__}: {failure}"[:300]})
                continue
            # Anything but one node of the expected kind is unusable here: a
            # refusal could then belong to either node, which is the very
            # ambiguity these models exist to remove.
            if emitted != [operator]:
                skipped.append({
                    "id": name, "operator": operator, "precision": precision,
                    "source": "reference",
                    "reason": f"the converter emitted {emitted or 'nothing'}, "
                              f"not one {operator}"})
                continue
            (out / f"{name}.tflite").write_bytes(blob)
            generated.append({
                "id": name, "operator": operator, "precision": precision,
                "source": "reference", "nxpuOp": nxpu_op,
                "file": f"{name}.tflite", "bytes": len(blob),
                "inputShapes": [list(s) for s in shapes]})
    return generated, skipped


def nxpu_models(directory, out):
    """Models the compiler itself produced, taken as they are.

    Not converted, not repaired, not filtered by whether they load. The
    question this half asks is whether the accelerator takes what NxPU emits,
    and every repair made here is a question not asked.
    """
    generated, skipped = [], []
    for path in sorted(pathlib.Path(directory).glob("*.tflite")):
        kernel = path.stem
        blob = path.read_bytes()
        try:
            emitted = operators_in(blob)
        except Exception as failure:
            skipped.append({
                "id": f"nxpu-{kernel}", "operator": "?", "precision": "float32",
                "source": "nxpu",
                "reason": f"not a readable TFLite model: "
                          f"{type(failure).__name__}: {failure}"[:300]})
            continue
        if len(emitted) != 1:
            # Multi-operator kernels are real output and worth knowing about,
            # but a refusal cannot be attributed to one of their nodes. Recorded
            # as skipped, with the operators named, rather than dropped.
            skipped.append({
                "id": f"nxpu-{kernel}", "operator": "+".join(emitted) or "none",
                "precision": "float32", "source": "nxpu",
                "reason": f"the kernel lowers to {len(emitted)} operators "
                          f"({', '.join(emitted) or 'none'}); a refusal could "
                          f"not be attributed to one of them"})
            continue
        operator = emitted[0]
        name = f"nxpu-{kernel}"
        (out / f"{name}.tflite").write_bytes(blob)
        generated.append({
            "id": name, "operator": operator, "precision": "float32",
            "source": "nxpu", "kernel": kernel,
            "nxpuOp": OPS.get(operator, (None, None, None))[2],
            "file": f"{name}.tflite", "bytes": len(blob)})
    return generated, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--nxpu", help="directory of .tflite files nxpu emitted")
    parser.add_argument("--only", default="", help="comma-separated operator names")
    args = parser.parse_args()

    wanted = [o.strip().upper() for o in args.only.split(",") if o.strip()] or list(OPS)
    unknown = [o for o in wanted if o not in OPS]
    if unknown:
        print(f"unknown operator(s): {', '.join(unknown)}", file=sys.stderr)
        return 1

    out = pathlib.Path(args.out)
    # Flat, and the index names files relative to this directory. The whole
    # directory becomes one artifact and the device job joins its download path
    # to what the index says, so a subdirectory here becomes models/models/
    # there — every path wrong, every row excluded, and the build green. That
    # happened to DroidRunner and the silence is the failure mode.
    out.mkdir(parents=True, exist_ok=True)

    generated, skipped = reference_models(wanted, out)
    if args.nxpu:
        more, more_skipped = nxpu_models(args.nxpu, out)
        generated += more
        skipped += more_skipped

    (out / "models.json").write_text(json.dumps({
        "schema": 1,
        "generatedBy": f"tensorflow {tf.__version__}",
        "models": generated,
        "skipped": skipped,
    }, indent=2) + "\n")
    # A tab-separated index beside it, because the job that reads this runs on
    # the phone, and the phone has neither python3 nor jq.
    with (out / "models.index").open("w") as index:
        for model in generated:
            index.write(f"{model['id']}\t{model['file']}\t{model['operator']}\t"
                        f"{model['precision']}\t{model['source']}\n")

    print(f"{len(generated)} models, {len(skipped)} skipped", file=sys.stderr)
    for entry in skipped:
        print(f"  skipped {entry['id']}: {entry['reason']}", file=sys.stderr)
    # Nothing to sweep is not a sweep. Said here rather than discovered three
    # jobs later on a phone that was woken for it.
    if not generated:
        print("no models were generated; there is nothing to ask the device",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
