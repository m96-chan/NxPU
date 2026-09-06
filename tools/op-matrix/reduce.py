#!/usr/bin/env python3
"""Turns a sweep into an operator support matrix.

One row per model, one column per driver, and each cell is what the device
said rather than what a datasheet says. The input is the raw envelope from
`droidrunner-device test batch`, one file per driver, plus the CPU control
sweep — every model run with no device at all.

The control is not a formality. If a model does not run on the CPU either then
the model is broken, and filing that under "the driver does not support this
operator" would put a defect of ours into a table other people compile
against. Such a row is excluded and said to be excluded — which for the `nxpu`
half of the sweep is not a rare accident but a finding, since it is where a
model NxPU emitted turns out to be one no runtime can load.

Output is DroidRunner's `matrix.json` schema 1, field for field, so a table
produced here can be sent back to m96-chan/DroidRunner and compared against
theirs. The extra fields we add — `source`, `builtin`, `kernel`, `nxpuOp` —
are additions to a row, which that schema allows.

Usage: reduce.py --models models.json --control sweep-cpu.json \
                 --sweep DRIVER=sweep-DRIVER.json [--sweep ...] \
                 [--capabilities capabilities.json] --out DIR

Ported from m96-chan/DroidRunner's `tools/op-matrix/reduce.py`; the two model
sources and the rendering below are ours.
"""

import argparse
import json
import pathlib
import sys

# What a cell can say. `unsupported` and `absent` are different answers and a
# matrix that renders them the same is not worth compiling against: one is a
# driver that exists and said no, the other is a driver this phone does not
# have.
ACCELERATED = "accelerated"
PARTIAL = "partial"
UNSUPPORTED = "unsupported"
ABSENT = "absent"
ERROR = "error"
EXCLUDED = "excluded"

MARK = {ACCELERATED: "✓", PARTIAL: "~", UNSUPPORTED: "✗",
        ABSENT: "·", ERROR: "!", EXCLUDED: "—"}

LEGEND = (f"{MARK[ACCELERATED]} accelerated  {MARK[PARTIAL]} partial  "
          f"{MARK[UNSUPPORTED]} not taken  {MARK[ABSENT]} no such driver  "
          f"{MARK[ERROR]} error  {MARK[EXCLUDED]} excluded (see below)")

# Drivers that are the CPU whatever route reached them. Needed only for the
# fallback below: a result carrying `executed` has already been through the
# agent's own version of this judgement.
CPU_DRIVERS = {"nnapi-reference"}


def results_by_id(path):
    envelope = json.loads(pathlib.Path(path).read_text())
    return {row.get("id", str(index)): row
            for index, row in enumerate(envelope.get("results", []))}


def executed_of(row, driver):
    """`executed`, or what the delegation says when the field is not there.

    The Qualcomm path reports `delegation` but no `executed` — the contract's
    headline field missing from the one route that reaches an NPU. Treating
    that absence as "the driver took nothing" reported every operator as
    unsupported on a Hexagon that had taken all of them. So it is derived
    instead, with the same rule the agent applies.
    """
    # Whatever the device said, this driver is the CPU: the name says so, and
    # it is a fact about NNAPI rather than a measurement.
    if driver in CPU_DRIVERS:
        return "cpu"
    if row.get("executed"):
        return row["executed"]
    delegation = row.get("delegation")
    if not delegation:
        return None
    if delegation.get("partial"):
        return "partial"
    return "accelerator" if delegation.get("delegated", 0) > 0 else "cpu-fallback"


def classify(row, driver=None):
    """One result, as one of the words above."""
    if row is None:
        return ERROR, "the sweep returned no row for this model"
    if row.get("ok"):
        executed = executed_of(row, driver)
        if executed is None:
            return ERROR, "the device did not say what executed this"
        if executed == "accelerator":
            return ACCELERATED, row.get("executedBy") or row.get("requestedDevice", "")
        if executed == "partial":
            # A single-operator model should not be able to split. When one
            # does, the model holds more than we think it does, and that is
            # worth seeing rather than rounding to yes or no.
            return PARTIAL, row.get("delegation", {}).get("describe", "")
        return UNSUPPORTED, row.get("executedBy", "")
    code = row.get("code")
    if code == "refused":
        return UNSUPPORTED, row.get("error", "")
    if code in ("unknown-device", "not-installed"):
        return ABSENT, row.get("error", "")
    # `message` is the layer below speaking in its own words — the half that
    # names the tensor and the field. Kept ahead of our prose because for an
    # excluded row it is the entire finding.
    return ERROR, row.get("message", "") or row.get("error", "")


def row_name(model):
    """The `operator` field, unique across both model sources.

    DroidRunner's `compare.py` keys a cell on (operator, precision, driver), so
    two rows sharing an operator name would silently become one. The reference
    half keeps the bare builtin name, which is what makes those rows comparable
    with a matrix taken there; the NxPU half names the kernel it came from,
    because `vecadd` and `vecadd_const` are both ADD and are not the same test.
    """
    if model["source"] == "reference":
        return model["operator"]
    return f"{model['operator']} (nxpu:{model['kernel']})"


def device_of(capabilities_path):
    if not capabilities_path:
        return {}
    capabilities = json.loads(pathlib.Path(capabilities_path).read_text())
    hardware = capabilities.get("device", {})
    return {key: value for key, value in (
        ("manufacturer", hardware.get("manufacturer")),
        ("model", hardware.get("model")),
        ("soc", hardware.get("soc")),
        ("sdk", capabilities.get("android", {}).get("sdk")),
        # A table nobody can trace back to a build is a table nobody should
        # stand behind.
        ("droidrunner", " ".join(filter(None, (
            capabilities.get("appVersion"), capabilities.get("appBuild")))) or None),
    ) if value is not None}


def mark_of(cell):
    """The cell as one field: the mark, the time, and how far the tail runs.

    A timed sweep answers a different question from an untimed one -- not
    whether an engine will take the operator but which engine is faster. Both
    belong in the same cell, because a time without an attribution is a number
    about the CPU as often as not.

    The tail is here because the median alone gives the wrong answer. On an
    MT6899 the GPU beats the NPU in the middle on 8 of 12 operators and is
    twice as spread at p99, with `conditions.stable` true throughout -- so it
    is contention and not a throttle. It is the engine the display composites
    on, and occupying it is not free to the rest of the device in a way no
    cell in this table can show.
    """
    mark = MARK[cell["status"]]
    median = cell.get("medianUs")
    if median is None:
        return mark
    cell_text = f"{mark} {median / 1000:.2f}ms"
    p90 = cell.get("p90Us")
    if p90 and median:
        cell_text += f" p90 {p90 / median:.1f}x"
    return cell_text


def table(lines, rows, drivers, first_column, cell_of):
    lines += ["| " + first_column + " | precision | " + " | ".join(drivers) + " |",
              "| --- | --- | " + " | ".join("---" for _ in drivers) + " |"]
    for row in rows:
        marks = " | ".join(mark_of(row["drivers"][d]) for d in drivers)
        lines.append(f"| {cell_of(row)} | {row['precision']} | {marks} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--sweep", action="append", default=[], metavar="DRIVER=FILE")
    parser.add_argument("--capabilities")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    catalogue = json.loads(pathlib.Path(args.models).read_text())
    control = results_by_id(args.control)

    sweeps = {}
    for pair in args.sweep:
        driver, _, path = pair.partition("=")
        if not path:
            print(f"--sweep wants DRIVER=FILE, got {pair!r}", file=sys.stderr)
            return 2
        sweeps[driver] = results_by_id(path)

    device = device_of(args.capabilities)

    rows, excluded = [], 0
    for model in catalogue["models"]:
        control_status, control_detail = classify(control.get(model["id"]))
        # The control ran with no device, so `unsupported` is the CPU's normal
        # answer — there was no delegate to take anything. Only a model that
        # could not be run at all disqualifies its row.
        usable = control_status != ERROR
        if not usable:
            excluded += 1
        cells = {}
        for driver, sweep in sweeps.items():
            if not usable:
                cells[driver] = {"status": EXCLUDED, "detail": control_detail}
                continue
            row = sweep.get(model["id"])
            status, detail = classify(row, driver)
            cell = {"status": status, "detail": detail}
            # Carried so a comparison can tell "this driver changed its mind"
            # from "the phone was in a state it could not describe".
            stable = (row or {}).get("conditions", {}).get("stable")
            if stable is not None:
                cell["stable"] = stable
            # Absent unless the sweep asked for iterations. Carried but never
            # compared: `compare.py` keys on status, because a median that
            # moved is a phone that got warm and a status that moved is a
            # driver that changed its mind.
            median = (row or {}).get("medianUs")
            if median is not None:
                cell["medianUs"] = median
            # The tail, because on this silicon it is where the two engines
            # actually differ. Measured on MT6899 with `conditions.stable` true
            # throughout, so not a throttle: the GPU is faster than the NPU in
            # the middle on 8 of 12 operators and is twice as spread at p99 --
            # 2.04x its own median against the NPU's 1.27x, worst case 6.23x
            # against 2.69x. It is the engine the display composites on, so it
            # is contended by construction, and a schedule chosen on medians
            # alone picks it every time.
            p90 = (row or {}).get("p90Us")
            if p90 is not None:
                cell["p90Us"] = p90
            cells[driver] = cell
        rows.append({"operator": row_name(model), "precision": model["precision"],
                     "id": model["id"], "usable": usable,
                     "controlDetail": control_detail if not usable else None,
                     "source": model["source"], "builtin": model["operator"],
                     "kernel": model.get("kernel"), "nxpuOp": model.get("nxpuOp"),
                     "drivers": cells})

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    drivers = list(sweeps)
    (out / "matrix.json").write_text(json.dumps({
        "schema": 1, "device": device, "drivers": drivers,
        "generatedBy": catalogue.get("generatedBy"),
        "skippedModels": catalogue.get("skipped", []),
        "rows": rows,
    }, indent=2) + "\n")

    reference = [r for r in rows if r["source"] == "reference"]
    ours = [r for r in rows if r["source"] == "nxpu"]

    lines = ["# Operator support matrix", ""]
    if device:
        lines += ["  ".join(f"**{key}** {value}" for key, value in device.items()), ""]
    lines += [LEGEND, "",
              "## The driver, asked directly", "",
              "One operator per model, built by TensorFlow's converter at "
              "conventional NHWC shapes, and **every weight is a constant**. "
              "A cell is what this driver did with *that model*, which is one "
              "instance of the operator and not the operator: on MT6899 the "
              "same convolution is accelerated with a constant filter and "
              "refused with a filter the graph takes as an input.", ""]
    table(lines, reference, drivers, "operator",
          lambda row: f"{row['builtin']}"
                      + (f" <sub>{row['nxpuOp']}</sub>" if row["nxpuOp"] else ""))

    lines += ["", "## What NxPU emits, asked the same way", "",
              "The bytes `nxpu --target tflite` produced for each kernel in "
              "`examples/`. A row excluded here is a model no runtime would "
              "load — a defect of ours, not an opinion of the driver's.", ""]
    table(lines, ours, drivers, "kernel",
          lambda row: f"`{row['kernel']}` → {row['builtin']}")

    if excluded:
        lines += ["", f"{excluded} row(s) excluded: the model did not run on the CPU "
                      "either, so nothing about a driver can be concluded from it. "
                      "What the runtime said about each is in `matrix.json` under "
                      "`controlDetail`.", ""]
        for row in rows:
            if not row["usable"]:
                lines.append(f"- `{row['id']}` — {row['controlDetail']}")
    if catalogue.get("skipped"):
        lines += ["", "Models that were never built, and why:", ""]
        for entry in catalogue["skipped"]:
            lines.append(f"- `{entry['id']}` — {entry['reason']}")
    lines += ["", "Every cell is what the device reported for that model. What a "
                  "cell does and does not claim is in `docs/matrices/README.md`."]
    (out / "matrix.md").write_text("\n".join(lines) + "\n")

    counted = {}
    for row in rows:
        for cell in row["drivers"].values():
            counted[cell["status"]] = counted.get(cell["status"], 0) + 1
    print(f"{len(rows)} rows x {len(drivers)} drivers: "
          + ", ".join(f"{count} {status}" for status, count in sorted(counted.items())),
          file=sys.stderr)

    # An all-excluded table is not a matrix, and publishing one as though it
    # were is worse than publishing nothing: it looks like an answer. The first
    # run of this on hardware — in DroidRunner, from models the agent could not
    # find because an artifact had been uploaded one directory too high — was
    # exactly that, and it came back green.
    if rows and excluded == len(rows):
        print("every row was excluded: no model ran on the CPU, so this sweep "
              "says nothing about any driver", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
