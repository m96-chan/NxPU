#!/usr/bin/env python3
"""Load every model the TFLite backend emits, in a real interpreter.

The suite asserts that emitted files begin with `TFL3` and stops there, which
is how a backend shipped models that no interpreter could load: strides
omitted because the value equalled the default argument, options tables tagged
with the wrong union index, index tensors typed u32 where the kernel wants
i32. None of it is visible in the bytes without a runtime that tries.

Each model is loaded in its own process because a mis-tagged options table
does not raise — it segfaults, and a sweep that dies on the first one reports
nothing about the rest.

Usage: tools/tflite-loadable.py [--update BASELINE]
Exit: 0 all models load or match the baseline, 1 a regression, 2 cannot run.
"""

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
NXPU = ROOT / "target/release/nxpu"
BASELINE = ROOT / "tools/tflite-loadable.json"

LOADER = """
import sys
from ai_edge_litert.interpreter import Interpreter
try:
    it = Interpreter(model_path=sys.argv[1])
    it.allocate_tensors()
    print("ok")
except Exception as e:
    print(str(e).replace("\\n", " ")[:120])
"""


def classify(model: pathlib.Path, loader: pathlib.Path) -> str:
    try:
        r = subprocess.run(
            [sys.executable, str(loader), str(model)],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return "timeout"
    if r.returncode != 0:
        # A crash is a distinct outcome from a rejection and the more serious
        # one: it means the runtime followed a table into somewhere it should
        # not have been.
        return f"crash (exit {r.returncode})"
    return r.stdout.strip() or "no output"


def main() -> int:
    if not NXPU.exists():
        print("build first: cargo build --release -p nxpu-cli", file=sys.stderr)
        return 2
    try:
        import ai_edge_litert  # noqa: F401
    except ImportError:
        print("needs ai-edge-litert: pip install ai-edge-litert", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        loader = pathlib.Path(tmp) / "load.py"
        loader.write_text(LOADER)
        results = {}
        for wgsl in sorted((ROOT / "examples").glob("*.wgsl")):
            # Two extents, because a shape defect can be invisible at one.
            # This ran only at 8, where a convolution whose every dimension had
            # been given the same extent has an im2col of 8^6 -- comfortably
            # inside TFLite's 32-bit limit. At 64 the same model needs 2^36 and
            # no interpreter will load it, which is what a MediaTek MT6899
            # reported and this check did not.
            outcome = "ok"
            for extent in ("8", "64"):
                model = pathlib.Path(tmp) / f"m{extent}.tflite"
                built = subprocess.run(
                    [str(NXPU), str(wgsl), "--target", "tflite", "--precision", "keep",
                     "--symbolic-dim", extent, "-o", str(model)],
                    capture_output=True, text=True,
                )
                if built.returncode != 0:
                    outcome = "does not compile"
                    break
                verdict = classify(model, loader)
                if verdict != "ok":
                    outcome = f"at --symbolic-dim {extent}: {verdict}"
                    break
            results[wgsl.stem] = outcome

    loads = sorted(k for k, v in results.items() if v == "ok")
    print(f"{len(loads)} of {len(results)} load\n")
    for name, outcome in results.items():
        mark = "ok  " if outcome == "ok" else "FAIL"
        print(f"  {mark} {name:20s} {'' if outcome == 'ok' else outcome}")

    if "--update" in sys.argv:
        BASELINE.write_text(json.dumps({"loads": loads}, indent=2) + "\n")
        print(f"\nbaseline written: {len(loads)} models")
        return 0

    if not BASELINE.exists():
        print("\nno baseline; run with --update", file=sys.stderr)
        return 2
    expected = set(json.loads(BASELINE.read_text())["loads"])
    actual = set(loads)
    lost = sorted(expected - actual)
    gained = sorted(actual - expected)
    if gained:
        print(f"\nnewly loading (update the baseline): {', '.join(gained)}")
    if lost:
        print(f"\nREGRESSION — these loaded before: {', '.join(lost)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
