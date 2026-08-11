"""Score an open-plane run along every axis it was designed to test.

Each slice gets its OWN best-constant baseline, because a slice whose answers
cluster can be scored well by a model that ignores the video — a mistake this
project has already made twice.

    .venv/bin/python plane_analyze.py runs/pl_s2 runs/pl_base
"""

import argparse
import collections
import glob
import json
import statistics as st
from pathlib import Path

import make_probes as MP

ROOT = Path(__file__).resolve().parent
HELD_OUT_COLOURS = ("pink", "white", "brown")
TRAINED_TASKS = ("bearing_final", "bearing_start")


def load(run):
    recs = {}
    for f in glob.glob(str(ROOT / run / "raw_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            recs[r["id"]] = r
    return recs


def constant(gts):
    c = min(range(360), key=lambda k: sum(MP.circ_err(k, b) for b in gts))
    return c, sum(MP.circ_err(c, b) for b in gts) / len(gts)


def table(title, groups, runs, plans):
    print(f"\n{title}")
    head = f"{'slice':24s} {'n':>4} {'constant':>12}"
    for name in runs:
        head += f" {name:>13}"
    print(head)
    for key in sorted(groups):
        ids = groups[key]
        gts = [plans[i]["bearing_gt_deg"] for i in ids]
        c, cerr = constant(gts)
        row = f"{str(key):24s} {len(ids):>4} {cerr:>8.1f}@{c:03d}"
        for name in runs:
            errs = [runs[name][i]["err"] for i in ids
                    if i in runs[name] and runs[name][i]["err"] is not None]
            row += f" {st.mean(errs):>13.1f}" if errs else f" {'-':>13}"
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--probes", default="probes/plane_probe_all.json")
    args = ap.parse_args()

    doc = json.loads((ROOT / args.probes).read_text())
    plans = {p["id"]: p for p in doc["plans"]}
    runs = {Path(r).name: load(r) for r in args.runs}

    def group(fn):
        g = collections.defaultdict(list)
        for i, p in plans.items():
            k = fn(p)
            if k is not None:
                g[k].append(i)
        return g

    table("BY QUESTION TYPE  (mid / obj-to-obj / back-to-start never trained)",
          group(lambda p: p["task"]), runs, plans)
    table("BY PATH DIFFICULTY  (P4 never trained)",
          group(lambda p: p.get("level_path")), runs, plans)
    table("TRAINED TYPES ONLY, by target colour",
          group(lambda p: (("held-out colour"
                            if any(c in str(p.get("object", ""))
                                   for c in HELD_OUT_COLOURS)
                            else "trained colour")
                           if p["task"] in TRAINED_TASKS else None)),
          runs, plans)
    table("bearing_final ONLY, by whether the object is on screen at the end",
          group(lambda p: (("visible at end" if p.get("visible_at_end")
                            else "off screen at end")
                           if p["task"] == "bearing_final" else None)),
          runs, plans)

    print("\nANSWER SPREAD (one dominant answer means the video is ignored)")
    for name, R in runs.items():
        c = collections.Counter(r["answer"] for r in R.values()
                                if r["answer"] is not None)
        unparsed = sum(r["answer"] is None for r in R.values())
        top = ", ".join(f"{a:.0f}x{n}" for a, n in c.most_common(3))
        print(f"  {name:14s} n={len(R):4d}  distinct={len(c):4d}  "
              f"unparsed={unparsed:3d}  top: {top}")


if __name__ == "__main__":
    main()
