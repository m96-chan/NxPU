#!/usr/bin/env python3
"""Compares two operator matrices and fails on a regression.

A vendor driver is not a fixed thing. It arrives with the system image and is
replaced by OTA updates nobody asks permission for, so a driver that used to
take an operator can quietly stop. Finding that out needs the same phone, kept,
and asked the same question twice — which is the one check a fleet of rented
cloud machines cannot perform.

Exit status:
  0  nothing got worse
  1  an operator that was accelerated no longer is
  2  the two matrices cannot be compared at all

Three distinct codes, and every path returns one deliberately. `sys.exit(str)`
prints the string and exits 1, which quietly turns "cannot compare" into
"regression" — so nothing here exits with a string.

Usage: compare.py BEFORE.json AFTER.json [--json OUT]

Ported from m96-chan/DroidRunner's `tools/op-matrix/compare.py`.
"""

import argparse
import json
import pathlib
import sys

CANNOT_COMPARE = 2

# What counts as the driver having taken the work. `partial` is in here because
# a single-operator model that splits is already odd; losing it entirely is
# still a step down, and the report names which it was.
TAKEN = {"accelerated", "partial"}

# Not comparable: the model did not run on the CPU either, so neither side says
# anything about a driver.
SKIP = {"excluded"}


def load(path):
    try:
        return json.loads(pathlib.Path(path).read_text())
    except (OSError, json.JSONDecodeError) as failure:
        print(f"cannot read {path}: {failure}", file=sys.stderr)
        # A file that could not be read is a comparison that did not happen,
        # not a regression that was found.
        sys.exit(CANNOT_COMPARE)


def key(matrix):
    """What makes two matrices the same phone.

    Manufacturer and model, not SoC — even though the SoC is what determines
    the drivers. A matrix taken before DroidRunner reported `soc` does not
    carry one, and identifying by SoC refuses to compare a phone against itself
    across the build that started reporting it, which is exactly when the
    comparison matters most.
    """
    device = matrix.get("device", {})
    return (device.get("manufacturer", "?"), device.get("model", "?"))


def describe(matrix):
    device = matrix.get("device", {})
    return device.get("soc") or device.get("model") or "unknown device"


def cells(matrix):
    return {(row["operator"], row["precision"], driver): (row, cell)
            for row in matrix.get("rows", [])
            for driver, cell in row.get("drivers", {}).items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--json", dest="out")
    args = parser.parse_args()

    before, after = load(args.before), load(args.after)

    # Two different phones produce two different answers for reasons that have
    # nothing to do with anything changing.
    if key(before) != key(after):
        print(f"these are different devices: {describe(before)} and "
              f"{describe(after)}", file=sys.stderr)
        return CANNOT_COMPARE

    build_before = before.get("device", {}).get("droidrunner")
    build_after = after.get("device", {}).get("droidrunner")
    # A cell that moved between two DroidRunner builds may be their change and
    # not the driver's, and saying so is the difference between a report
    # someone acts on and one they argue with.
    same_build = build_before == build_after

    old, new = cells(before), cells(after)
    regressions, improvements, unverified, appeared, vanished = [], [], [], [], []

    for cell_key in sorted(set(old) | set(new)):
        operator, precision, driver = cell_key
        was, now = old.get(cell_key), new.get(cell_key)
        if was is None:
            appeared.append((operator, precision, driver, now[1]["status"]))
            continue
        if now is None:
            vanished.append((operator, precision, driver, was[1]["status"]))
            continue
        before_status, after_status = was[1]["status"], now[1]["status"]
        if before_status == after_status:
            continue
        if before_status in SKIP or after_status in SKIP:
            continue
        change = {"operator": operator, "precision": precision, "driver": driver,
                  "from": before_status, "to": after_status,
                  "detail": now[1].get("detail", "")}
        if before_status in TAKEN and after_status not in TAKEN:
            # A phone that could not say what state it was in has not earned
            # the right to fail somebody's build.
            if now[1].get("stable") is False or was[1].get("stable") is False:
                change["reason"] = "the run was not thermally stable"
                unverified.append(change)
            elif not same_build:
                change["reason"] = f"builds differ: {build_before} -> {build_after}"
                unverified.append(change)
            else:
                regressions.append(change)
        elif after_status in TAKEN and before_status not in TAKEN:
            improvements.append(change)
        else:
            unverified.append({**change, "reason": "neither side was accelerated"})

    report = {"schema": 1, "device": describe(after),
              "builds": {"before": build_before, "after": build_after},
              "regressions": regressions, "improvements": improvements,
              "unverified": unverified,
              "driversAppeared": [list(a) for a in appeared],
              "driversGone": [list(v) for v in vanished]}
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    print(f"{describe(after)}: {len(regressions)} regressions, "
          f"{len(improvements)} improvements, {len(unverified)} not compared")
    if not same_build:
        print(f"  builds differ: {build_before} -> {build_after} — "
              "a cell that moved may be theirs and not the driver's")
    for change in regressions:
        print(f"  REGRESSION  {change['driver']}  {change['operator']} "
              f"{change['precision']}  {change['from']} -> {change['to']}")
    for change in improvements:
        print(f"  better      {change['driver']}  {change['operator']} "
              f"{change['precision']}  {change['from']} -> {change['to']}")
    for change in unverified:
        print(f"  not judged  {change['driver']}  {change['operator']} "
              f"{change['precision']}  {change['from']} -> {change['to']}"
              f"  ({change['reason']})")
    for operator, precision, driver, status in appeared:
        print(f"  new cell    {driver}  {operator} {precision}  {status}")
    for operator, precision, driver, status in vanished:
        print(f"  gone        {driver}  {operator} {precision}  was {status}")

    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
