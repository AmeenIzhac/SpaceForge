"""Mid-walk bearing probes: where was the X just before the final corner?

Reuses existing rendered walks and asks about a moment the model was never
trained to report: one second before the last turn begins. Answering needs a
*running* spatial state — a model that only ever forms "X relative to my final
pose" has nothing to look up. Ground truth comes from the same baked camera
track the videos were rendered from.

One-turn walks are kept but trivial by construction (before the only corner
the walker has gone straight from the X, so the answer is exactly 180) — they
check the question is understood; the 2-and-3-turn rows carry the test.

    .venv/bin/python make_midpoint_probes.py probes/test_clean.json \
        --out probes/mid_test_clean.json
"""

import argparse
import json
from pathlib import Path

import generate as G
import make_probes as MP
import make_trainset as MT

ROOT = Path(__file__).resolve().parent

QUESTION = (
    "This is a first-person video of someone walking through corridors. A "
    "large red X is painted on the floor at the spot where the walk begins; "
    "it is visible in the opening seconds, and the walker then leaves it "
    "behind. Now think back to the moment IN THE MIDDLE of the walk, one "
    "second BEFORE the walker started to turn the FINAL corner. Freeze on "
    "that moment — not the end of the video. Treat the direction the camera "
    "was facing at that moment as north (000 degrees). What was the compass "
    "bearing from the camera's position at that moment to the red X? Answer "
    "in degrees clockwise from that facing, 0-359: 090 is directly to the "
    "right, 180 directly behind, 270 directly to the left.")


def build(plans):
    out = []
    for p in plans:
        if not p["turns"]:
            continue                      # no corner to be "before"
        _marker, poses, ev, _lt = MT.timing(p["turns"], p["legs"])
        marker, _poly, poses2 = MP.walk_of(p["turns"], p["legs"])
        t_q = max(0.5, ev[-1]["t_start"] - 1.0)
        pose = poses2[min(int(round(t_q * G.FPS)), len(poses2) - 1)]
        out.append({
            "id": p["id"],
            "video": f"{p['id']:03d}.mp4",
            "level": f"mid_{p['n_turns']}turn",
            "turns": p["turns"], "legs": p["legs"],
            "n_turns": p["n_turns"],
            "t_question_s": round(t_q, 2),
            "bearing_final_deg": p["bearing_gt_deg"],   # the trained target
            "bearing_gt_deg": round(G.bearing_deg(pose, marker), 2),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plans")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    plans = json.loads((ROOT / args.plans).read_text())["plans"]
    rows = build(plans)

    gts = [r["bearing_gt_deg"] for r in rows]
    c = min(range(360), key=lambda k: sum(MP.circ_err(k, b) for b in gts))
    cerr = sum(MP.circ_err(c, b) for b in gts) / len(gts)
    # how close is the mid-walk answer to the end-of-walk answer? If they are
    # near, a model could score well by ignoring the question and answering
    # the end state — this number is the "answer the trained question anyway"
    # baseline, and it must be beaten for the test to mean anything.
    lazy = sum(MP.circ_err(r["bearing_gt_deg"], r["bearing_final_deg"])
               for r in rows) / len(rows)

    doc = {
        "question": QUESTION,
        "note": ("Mid-walk state probe over existing videos; bearing_gt_deg "
                 "is measured 1 s before the final turn starts."),
        "const_baseline_deg": round(cerr, 2),
        "const_baseline_answer": c,
        "answer_final_anyway_deg": round(lazy, 2),
        "plans": rows,
    }
    (ROOT / args.out).write_text(json.dumps(doc, indent=1))
    by = {}
    for r in rows:
        by.setdefault(r["n_turns"], []).append(r)
    print(f"{len(rows)} probes -> {args.out}")
    print(f"  best constant {cerr:.1f}@{c:03d}; answering the FINAL bearing "
          f"anyway scores {lazy:.1f}")
    for k in sorted(by):
        print(f"  {k}-turn: n={len(by[k])}")


if __name__ == "__main__":
    main()
