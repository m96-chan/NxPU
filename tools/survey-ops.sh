#!/usr/bin/env bash
#
# What fraction of the kernels NxPU exists to compile does it compile?
#
# `examples/` holds kernels written to exercise the compiler. vendor/web-xpu-ops
# holds the ones actually written to run something, which is a different and
# less flattering question. This asks the second one.
#
# Usage: tools/survey-ops.sh [--verbose]
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
nxpu="$root/target/release/nxpu"
ops="$root/vendor/web-xpu-ops/ops"

[ -x "$nxpu" ] || { echo "build first: cargo build --release -p nxpu-cli" >&2; exit 1; }
[ -d "$ops" ] || { echo "submodule missing: git submodule update --init" >&2; exit 1; }

ok=0 fail=0
reasons="$(mktemp)"
trap 'rm -f "$reasons"' EXIT

for f in "$ops"/*/wgsl/*.wgsl; do
    name="${f#"$ops"/}"
    if "$nxpu" "$f" --target tflite --precision keep --symbolic-dim 64 -o /dev/null >/dev/null 2>&1; then
        ok=$((ok + 1))
        [ "${1:-}" = "--verbose" ] && printf '  ok    %s\n' "$name"
    else
        fail=$((fail + 1))
        # The detail after miette's arrow is the part worth grouping on; the
        # rest is box drawing.
        "$nxpu" "$f" --target tflite --precision keep -o /dev/null 2>&1 |
            grep -oE '╰─▶ .*' | sed 's/╰─▶ //' >> "$reasons" || true
        [ "${1:-}" = "--verbose" ] && printf '  FAIL  %s\n' "$name"
    fi
done

printf '\n%d of %d compile\n\n' "$ok" "$((ok + fail))"
printf 'why the rest do not:\n'
sed 's/[0-9][0-9]*/N/g' "$reasons" | sort | uniq -c | sort -rn | sed 's/^/  /'
