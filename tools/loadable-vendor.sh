#!/usr/bin/env bash
#
# Of the vendored kernels this compiler accepts, how many produce a model a
# TFLite interpreter can load?
#
# `tools/tflite-loadable.py` asks that of `examples/`, which were written to
# exercise the compiler. This asks it of the kernels written to run something,
# which is the number that matters and the less flattering one.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
nxpu="$root/target/release/nxpu"
ops="$root/vendor/web-xpu-ops/ops"
# 64, matching what the device workflows compile at. This was 16, with a note
# saying a convolution overflowed a 32-bit im2col at 64 because "every dimension
# takes the same extent" and that this was "the flag's doing, not the emitter's".
# The second half was wrong: the emitter wrote one shape vector to a
# convolution's input, weight and output alike, so the model was incoherent at
# every extent and merely small enough to load at 16. A MediaTek MT6899 refused
# it on all four drivers while accepting the same operator from TensorFlow's
# converter, which is how it was found.
extent="${1:-64}"

[ -x "$nxpu" ] || { echo "build first: cargo build --release -p nxpu-cli" >&2; exit 2; }
[ -d "$ops" ] || { echo "submodule missing: git submodule update --init" >&2; exit 2; }
python3 -c 'import ai_edge_litert' 2>/dev/null || {
    echo "needs ai-edge-litert: pip install ai-edge-litert" >&2; exit 2; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cat > "$tmp/load.py" <<'PY'
import sys
from ai_edge_litert.interpreter import Interpreter
try:
    it = Interpreter(model_path=sys.argv[1]); it.allocate_tensors()
    print("ok")
except Exception as e:
    print(str(e).replace("\n", " ")[:100])
PY

ok=0 compiled=0 total=0
for f in "$ops"/*/wgsl/*.wgsl; do
    total=$((total + 1))
    name="$(basename "$(dirname "$(dirname "$f")")")/$(basename "$f" .wgsl)"
    "$nxpu" "$f" --target tflite --precision keep --symbolic-dim "$extent" \
        -o "$tmp/m.tflite" >/dev/null 2>&1 || continue
    compiled=$((compiled + 1))
    # Own process: a mis-tagged options table crashes rather than raising.
    if out="$(timeout 60 python3 "$tmp/load.py" "$tmp/m.tflite" 2>/dev/null)"; then
        :
    else
        out="crash (exit $?)"
    fi
    if [ "$out" = "ok" ]; then
        ok=$((ok + 1))
    else
        printf '  %-26s %s\n' "$name" "$out"
    fi
done

printf '\n%d of %d compile, %d of those load (%d of %d end to end)\n' \
    "$compiled" "$total" "$ok" "$ok" "$total"
