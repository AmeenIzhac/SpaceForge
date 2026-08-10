"""Score a bearing run against the benchmark's own baselines.

A mean error in degrees means little on its own — on this answer distribution
several strategies that involve no spatial reasoning at all already score well
under chance. So every run is reported next to them:

    uniform guess      90 deg by construction
    always "behind"    answer 180 every time (`behind_err_deg`)
    nearest cardinal   the best of 0/90/180/270 per sample, an oracle-ish
                       ceiling on "round number" answering (`cardinal_err_deg`)
    equal legs         track the turns, ignore how long each straight ran
                       (`equal_leg_err_deg`) — the headline axis: beating this
                       is the only evidence of actual path integration

Then the same error sliced by the axes the benchmark was stratified over.

    .venv/bin/python analyze.py runs/base
    .venv/bin/python analyze.py runs/base --diag       # diagnose.py output
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def circ_err(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def stats(vals):
    if not vals:
        return float("nan"), float("nan")
    s = sorted(vals)
    return sum(vals) / len(vals), s[len(s) // 2]


def add_baselines(plans):
    """Fill in the shortcut-baseline columns for plan files that lack them.

    make_probes.py writes them; make_trainset.py does not, so an easier set
    rendered from training plans would otherwise be unscoreable against the
    same yardsticks. They are all derivable from turns/legs."""
    import sft_data

    for p in plans:
        if "behind_err_deg" not in p:
            p["behind_err_deg"] = round(circ_err(p["bearing_gt_deg"], 180.0), 2)
        if "cardinal_err_deg" not in p:
            p["cardinal_err_deg"] = round(
                min(circ_err(p["bearing_gt_deg"], c)
                    for c in (0, 90, 180, 270)), 2)
        if "equal_leg_err_deg" not in p:
            legs = p["legs"]
            flat = max(1, int(round(sum(legs) / len(legs))))
            eq, *_ = sft_data.solve(p["turns"], [flat] * len(legs))
            p["equal_leg_err_deg"] = round(circ_err(p["bearing_gt_deg"], eq), 2)
    return plans


def load(run_dir):
    recs = []
    p = run_dir / "results.json"
    if p.exists():
        recs = json.loads(p.read_text())
    else:
        for f in sorted(run_dir.glob("raw_*.jsonl")):
            recs += [json.loads(l) for l in f.read_text().splitlines() if l]
    seen, out = set(), []
    for r in sorted(recs, key=lambda r: r.get("id", 0)):
        key = (r.get("task"), r.get("id"), r.get("frames"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def report_bearing(recs, plans, label):
    by_id = {p["id"]: p for p in plans}
    live = [r for r in recs if r["id"] in by_id]
    answered = [r for r in live if r.get("answer") is not None]
    n_un = len(live) - len(answered)

    errs = [r["err"] for r in answered]
    mean, med = stats(errs)

    # baselines on exactly the samples the model answered, so the comparison
    # is like for like even on a partial run
    ids = [r["id"] for r in answered]
    base = {
        "model": errs,
        "always 180": [by_id[i]["behind_err_deg"] for i in ids],
        "nearest cardinal": [by_id[i]["cardinal_err_deg"] for i in ids],
        "equal legs": [by_id[i]["equal_leg_err_deg"] for i in ids],
    }

    print(f"\n=== {label} ===")
    n_forced = sum(bool(r.get("forced")) for r in live)
    print(f"{len(live)} probes, {n_un} unparsed"
          + (f", {n_forced} ran out of thinking budget and were forced "
             f"to commit" if n_forced else ""))
    print(f"\n  {'strategy':20s} {'mean':>7s} {'median':>7s} "
          f"{'<=30 deg':>9s} {'<=45 deg':>9s}")
    for k, v in base.items():
        m, md = stats(v)
        p30 = 100 * sum(e <= 30 for e in v) / len(v) if v else float("nan")
        p45 = 100 * sum(e <= 45 for e in v) / len(v) if v else float("nan")
        print(f"  {k:20s} {m:7.1f} {md:7.1f} {p30:8.0f}% {p45:8.0f}%")
    print(f"  {'uniform guess':20s} {90.0:7.1f} {90.0:7.1f} "
          f"{100 / 6:8.0f}% {25.0:8.0f}%")

    # is the model's answer even correlated with the truth?
    if len(answered) > 3:
        gts = [math.radians(r["gt"]) for r in answered]
        ans = [math.radians(r["answer"]) for r in answered]
        # mean resultant length of the error angle: 1 = perfect, 0 = unrelated
        dx = sum(math.cos(a - g) for a, g in zip(ans, gts)) / len(gts)
        dy = sum(math.sin(a - g) for a, g in zip(ans, gts)) / len(gts)
        print(f"\n  circular concentration R = {math.hypot(dx, dy):.3f} "
              f"(0 = answers unrelated to truth, 1 = perfect)")

    for axis, key in (("turns", "n_turns"), ("tier", "tier"),
                      ("pattern", "pattern")):
        groups = defaultdict(list)
        for r in answered:
            if key not in by_id[r["id"]]:
                continue
            groups[by_id[r["id"]][key]].append(r["err"])
        if len(groups) < 2:
            continue
        print(f"\n  by {axis}:")
        for k in sorted(groups, key=str):
            m, md = stats(groups[k])
            print(f"    {str(k):10s} n={len(groups[k]):3d} "
                  f"mean {m:6.1f}  median {md:6.1f}")

    # what did it actually say? a model that always answers 180 looks fine on
    # mean error alone if the answers happen to sit behind the walker
    hist = defaultdict(int)
    for r in answered:
        hist[int(r["answer"] // 45) * 45] += 1
    print("\n  answers by 45-degree bin: " +
          " ".join(f"{k:03d}:{hist.get(k, 0)}" for k in range(0, 360, 45)))
    ghist = defaultdict(int)
    for r in answered:
        ghist[int(r["gt"] // 45) * 45] += 1
    print("  truth   by 45-degree bin: " +
          " ".join(f"{k:03d}:{ghist.get(k, 0)}" for k in range(0, 360, 45)))


def report_diag(recs, label):
    print(f"\n=== {label} ===")
    cells = defaultdict(list)
    for r in recs:
        cells[(r["task"], r["frames"], r["width"])].append(r)
    print(f"  {'task':14s} {'frames':>7s} {'res':>9s} {'slot_s':>7s} "
          f"{'n':>4s} {'acc':>6s} {'MAE':>6s} {'unparsed':>9s}")
    for (task, fr, w), rs in sorted(cells.items()):
        acc = sum(bool(r["correct"]) for r in rs) / len(rs)
        errs = [r["err"] for r in rs if r["err"] is not None]
        mae = sum(errs) / len(errs) if errs else float("nan")
        slot = sum(r["slot_s"] for r in rs) / len(rs)
        res = f"{w}x{rs[0]['height']}"
        print(f"  {task:14s} {fr:7d} {res:>9s} {slot:7.2f} {len(rs):4d} "
              f"{acc:6.2f} {mae:6.2f} {len(rs) - len(errs):9d}")

    for task in sorted({r["task"] for r in recs}):
        rs = [r for r in recs if r["task"] == task and r["pred"] is not None]
        if not rs:
            continue
        if task == "count":
            print(f"\n  {task}: prediction histogram vs truth")
            pv = defaultdict(int)
            gv = defaultdict(int)
            for r in rs:
                pv[r["pred"]] += 1
                gv[r["gt"]] += 1
            keys = sorted(set(pv) | set(gv))
            print("    value: " + " ".join(f"{k:>4}" for k in keys))
            print("    pred : " + " ".join(f"{pv.get(k, 0):>4}" for k in keys))
            print("    truth: " + " ".join(f"{gv.get(k, 0):>4}" for k in keys))
            # does it track the truth at all?
            n = len(rs)
            mp = sum(r["pred"] for r in rs) / n
            mg = sum(r["gt"] for r in rs) / n
            cov = sum((r["pred"] - mp) * (r["gt"] - mg) for r in rs)
            vp = math.sqrt(sum((r["pred"] - mp) ** 2 for r in rs))
            vg = math.sqrt(sum((r["gt"] - mg) ** 2 for r in rs))
            if vp and vg:
                print(f"    correlation with truth r = {cov / (vp * vg):+.3f}")
        if task == "coarse":
            conf = defaultdict(int)
            for r in rs:
                conf[(r["gt"], r["pred"])] += 1
            labs = ["AHEAD", "RIGHT", "BEHIND", "LEFT"]
            print(f"\n  {task}: rows = truth, cols = predicted")
            print("           " + " ".join(f"{c:>7s}" for c in labs))
            for g in labs:
                print(f"    {g:7s} " +
                      " ".join(f"{conf.get((g, p), 0):>7d}" for p in labs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="+")
    ap.add_argument("--plans", default="probes/bearing_probes.json")
    ap.add_argument("--diag", action="store_true")
    args = ap.parse_args()

    plans = add_baselines(json.loads((ROOT / args.plans).read_text())["plans"])
    for run in args.run:
        d = ROOT / run
        recs = load(d)
        if not recs:
            print(f"{run}: no results")
            continue
        if args.diag or "task" in recs[0]:
            report_diag(recs, run)
        else:
            report_bearing(recs, plans, run)


if __name__ == "__main__":
    main()
