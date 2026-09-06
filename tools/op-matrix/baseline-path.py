#!/usr/bin/env python3
"""Where the committed matrix for a given phone lives.

Its own script rather than a heredoc inside the workflow: the naming has to
agree between the job that checks a matrix and the person committing one, and a
shell step that embeds Python inside YAML is where that agreement goes to die.

Named by manufacturer and model, matching how `compare.py` decides two matrices
are the same phone. Not by SoC — a matrix taken before DroidRunner reported one
does not carry it, and keying on SoC then refuses to compare a phone against
itself across exactly the build where the comparison matters.

Usage: baseline-path.py MATRIX.json
"""

import json
import pathlib
import re
import sys


def slug(value):
    return re.sub(r"[^a-z0-9]+", "-", (value or "unknown").lower()).strip("-")


def main():
    if len(sys.argv) != 2:
        print("usage: baseline-path.py MATRIX.json", file=sys.stderr)
        return 2
    device = json.loads(pathlib.Path(sys.argv[1]).read_text()).get("device", {})
    print(f"docs/matrices/{slug(device.get('manufacturer'))}-"
          f"{slug(device.get('model'))}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
