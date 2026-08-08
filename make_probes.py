"""Bearing-to-start probe set: 100 route plans + ground truth, no rendering.

The task each plan poses: watch the walk, then at the final frame report where
the red X (painted on the floor at the very start) lies as a compass bearing,
taking your own forward direction as north. Answering needs path integration —
the turn sequence alone is not enough, because the leg lengths decide how far
each turn displaces you.

Plans are stratified over the axes that plausibly make that hard:

  * turn count      1, 2, 3, 5, 8 — how many rotations to compose
  * straight ratio  the five RATIO_BANDS, 1:1 through 1:10 — how unequal the
                    legs are, i.e. how much the distances matter
  * turn pattern    alternating / same-direction spiral / random — whether the
                    heading oscillates or winds up past a full circle

5 x 5 x 4 replicates = 100. Every plan carries the features needed to slice
results afterwards, including two baselines that say what a shortcut would
score: `equal_leg_err_deg` is how wrong you land by tracking turns but
ignoring how long each straight was, and `cardinal_err_deg` is how wrong the
nearest round number (0/90/180/270) is.

Plans are renderable: generate.make_sample(name, turns, legs, seed) draws the
same walk this file measured.

    .venv/bin/python make_probes.py     -> probes/
"""

import csv
import json
import math
import random
from pathlib import Path

import numpy as np

import generate as G
import world

OUT_DIR = Path(__file__).resolve().parent / "probes"
SEED = 2029

TURN_LEVELS = [1, 2, 3, 5, 8]
PATTERNS = ["alt", "same", "rand", "rand"]      # 4 replicates per cell
TOTAL_MAX = 34                                  # lattice steps, caps run length
BEARING_BINS = 12                               # 30-degree answer bins
CANDIDATES = 24                                 # routes drawn per cell

QUESTION = (
    "This is a first-person video of someone walking through corridors. At the "
    "very start of the walk a large red X is painted on the floor, straight "
    "ahead of the camera. Freeze on the final frame and treat the direction "
    "the camera is facing as north (000 degrees). What is the compass bearing "
    "from the camera to the red X? Answer in degrees clockwise from that "
    "forward direction, 0-359: 090 is directly to the right, 180 directly "
    "behind, 270 directly to the left."
)
SCORING = ("Absolute circular error in degrees between the answer and "
           "bearing_gt_deg: err = min(|a - gt|, 360 - |a - gt|). Chance level "
           "for a uniform guess is 90 degrees.")


def bearing_bin(b):
    return min(int(b // (360.0 / BEARING_BINS)), BEARING_BINS - 1)


def wrap360(a):
    a %= 360.0
    return 0.0 if a >= 360.0 else a     # a hair below zero rounds up to 360


def circ_err(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def turn_string(rng, n_turns, pattern):
    if pattern == "same":
        return rng.choice("LR") * n_turns
    if pattern == "alt":
        first = rng.choice("LR")
        return "".join(first if i % 2 == 0 else "LR"[first == "L"]
                       for i in range(n_turns))
    return "".join(rng.choice("LR") for _ in range(n_turns))


def draw_plan(rng, n_turns, band, pattern, tries=800):
    """(turns, legs, sorted_legs_flag, pattern) for a self-avoiding route with
    exactly `n_turns` turns, or None if the pattern admits none.

    A same-direction spiral only stays self-avoiding if it keeps opening out,
    so if the random leg order collides we retry with the legs sorted
    ascending — a genuinely expanding spiral — and flag it."""
    n_legs = n_turns + 1
    for _ in range(tries):
        legs = G.draw_legs(rng, n_legs, band,
                           total_min=n_legs + 2, total_max=TOTAL_MAX)
        turns = turn_string(rng, n_turns, pattern)
        for ordered in (legs, sorted(legs)):
            route = G.build_route(turns, ordered)
            if len(set(route)) == len(route):
                return turns, ordered, ordered != legs, pattern
    return None


def walk_of(turns, legs):
    route, _ = G.normalize(G.build_route(turns, legs), [])
    _cc, _legs_idx, marker, poly, poses = G.solve_walk(route)
    return marker, poly, poses


def bearing_of(turns, legs):
    marker, _poly, poses = walk_of(turns, legs)
    return G.bearing_deg(poses[-1], marker)


def measure(turns, legs):
    """Ground truth + geometry features for one plan."""
    marker, poly, poses = walk_of(turns, legs)
    end = poses[-1]

    bearing = G.bearing_deg(end, marker)
    dist = math.hypot(marker[0] - end[0], marker[1] - end[1])
    path_len = float(world.polyline_length(poly))

    # what you get from the turn sequence alone, with every straight the same
    # length: the gap to the true bearing is how much this sample depends on
    # actually tracking the distances
    flat = max(1, int(round(sum(legs) / len(legs))))
    equal_leg_bearing = bearing_of(turns, [flat] * len(legs))

    cum_turn = sum(90.0 if t == "R" else -90.0 for t in turns)
    return {
        "bearing_gt_deg": round(bearing, 2),
        "distance_to_x": round(dist, 2),
        "final_yaw_deg": round(wrap360(math.degrees(end[2])), 2),
        "end_xy": [round(end[0], 2), round(end[1], 2)],
        "marker_xy": [round(marker[0], 2), round(marker[1], 2)],
        "path_len": round(path_len, 2),
        "duration_s": round(len(poses) / G.FPS, 1),
        "n_frames": len(poses),
        "cum_turn_deg": cum_turn,
        "net_turn_deg": ((cum_turn + 180) % 360) - 180,
        "crow_over_path": round(dist / max(path_len, 1e-9), 3),
        # diagnostics: how wrong the obvious shortcuts are on this sample
        "equal_leg_bearing_deg": round(equal_leg_bearing, 2),
        "equal_leg_err_deg": round(circ_err(bearing, equal_leg_bearing), 2),
        "cardinal_err_deg": round(min(circ_err(bearing, c)
                                      for c in (0, 90, 180, 270)), 2),
        "behind_err_deg": round(circ_err(bearing, 180.0), 2),
        # a 1-unit position error swings the bearing this far
        "sensitivity_deg_per_unit": round(math.degrees(1.0 / max(dist, 1e-9)), 2),
    }


def build():
    """Fill the strata, and inside each cell pick the candidate that best
    flattens the answer distribution.

    Left alone the answers pile up behind the walker — you generally end a walk
    away from where you started — and a model that always guesses "roughly
    behind me" scores far better than it deserves. Drawing CANDIDATES routes
    per cell and keeping the one landing in the emptiest bearing bin spreads
    the answers around the circle without disturbing the design: the cell still
    gets its assigned turn count, ratio band and pattern. One-turn routes are
    the exception — the X is always somewhere behind you after a single turn,
    so those cells simply cannot reach the forward half."""
    rng = random.Random(SEED)
    plans, fallbacks, spirals = [], 0, 0
    bins = [0] * BEARING_BINS
    for n_turns in TURN_LEVELS:
        for bi, band in enumerate(G.RATIO_BANDS):
            for rep, pattern in enumerate(PATTERNS):
                cands = []
                for _ in range(CANDIDATES):
                    # a spiral that admits no self-avoiding route at this turn
                    # count opens up to a free pattern instead
                    got = (draw_plan(rng, n_turns, band, pattern)
                           or draw_plan(rng, n_turns, band, "rand"))
                    if got is None:
                        raise RuntimeError(f"no route: turns={n_turns} {band}")
                    cands.append(got)
                pick = min(cands,
                           key=lambda c: bins[bearing_bin(bearing_of(*c[:2]))])
                turns, legs, sorted_legs, used = pick
                fallbacks += used != pattern
                bins[bearing_bin(bearing_of(turns, legs))] += 1
                spirals += sorted_legs
                pid = len(plans)
                plans.append({
                    "id": pid,
                    "seed": SEED + pid,
                    "turns": turns,
                    "legs": legs,
                    "n_turns": n_turns,
                    "ratio_band": list(band),
                    "ratio": round(max(legs) / min(legs), 2),
                    "total_steps": sum(legs),
                    "pattern_requested": pattern,
                    "pattern": used,
                    "legs_sorted_ascending": sorted_legs,
                    "replicate": rep,
                    **measure(turns, legs),
                })
    return plans, fallbacks, spirals


def add_difficulty(plans):
    """Rank-normalise the four features that should predict trouble, average
    them, and cut the result into terciles. Percentile-based on purpose: no
    magic thresholds, and it stays meaningful if the strata change."""
    keys = ["n_turns", "equal_leg_err_deg", "cardinal_err_deg",
            "sensitivity_deg_per_unit"]
    n = len(plans)
    ranks = {}
    for k in keys:
        order = sorted(range(n), key=lambda i: plans[i][k])
        r = [0.0] * n
        for pos, i in enumerate(order):
            r[i] = pos / (n - 1)
        ranks[k] = r
    for i, p in enumerate(plans):
        score = sum(ranks[k][i] for k in keys) / len(keys)
        p["difficulty"] = round(score, 3)
    cuts = sorted(p["difficulty"] for p in plans)
    lo, hi = cuts[n // 3], cuts[2 * n // 3]
    for p in plans:
        p["tier"] = ("easy" if p["difficulty"] < lo
                     else "hard" if p["difficulty"] >= hi else "medium")


def report(plans, fallbacks, spirals):
    print(f"{len(plans)} plans: {len(TURN_LEVELS)} turn levels x "
          f"{len(G.RATIO_BANDS)} ratio bands x {len(PATTERNS)} replicates")
    if fallbacks:
        print(f"  {fallbacks} cells fell back to a random turn pattern "
              f"(a same-direction spiral had no self-avoiding route there)")
    if spirals:
        print(f"  {spirals} routes needed legs sorted ascending to stay "
              f"self-avoiding (expanding spirals)")

    hist = [0] * BEARING_BINS
    for p in plans:
        hist[bearing_bin(p["bearing_gt_deg"])] += 1
    print("  answer spread, 30-degree bins from 000: " +
          " ".join(str(h) for h in hist))

    def rng_of(k):
        v = [p[k] for p in plans]
        return f"{min(v):.1f} .. {max(v):.1f}  (median {sorted(v)[len(v) // 2]:.1f})"

    for k in ("distance_to_x", "duration_s", "equal_leg_err_deg",
              "cardinal_err_deg", "behind_err_deg", "sensitivity_deg_per_unit"):
        print(f"  {k:26s} {rng_of(k)}")

    for tier in ("easy", "medium", "hard"):
        sub = [p for p in plans if p["tier"] == tier]
        turns = sorted({p["n_turns"] for p in sub})
        mean_eq = sum(p["equal_leg_err_deg"] for p in sub) / len(sub)
        print(f"  {tier:6s} n={len(sub):3d}  turns {turns}  "
              f"mean equal-leg error {mean_eq:5.1f} deg")

    total_s = sum(p["duration_s"] for p in plans)
    print(f"  rendering all 100 would be {total_s / 60:.0f} min of video")


def main():
    plans, fallbacks, spirals = build()
    add_difficulty(plans)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    doc = {
        "question": QUESTION,
        "scoring": SCORING,
        "answer_field": "bearing_gt_deg",
        "note": ("Every field except id/turns/legs/seed is ground truth or a "
                 "derived feature — strip them before showing a model. "
                 "probes_prompts.jsonl holds the model-facing side alone."),
        "convention": ("World axes are x east, y south; the camera looks along "
                       "(cos yaw, sin yaw), so bearing 090 is (-sin, cos), its "
                       "right. One lattice step is 3 world units."),
        "generator": {"seed": SEED, "turn_levels": TURN_LEVELS,
                      "ratio_bands": [list(b) for b in G.RATIO_BANDS],
                      "patterns": PATTERNS, "total_steps_max": TOTAL_MAX,
                      "mood": G.MOOD, "junctions": G.JUNCTIONS,
                      "marker_ahead": G.MARKER_AHEAD, "speed": G.SPEED,
                      "fps": G.FPS},
        "plans": plans,
    }
    (OUT_DIR / "bearing_probes.json").write_text(json.dumps(doc, indent=2))

    with open(OUT_DIR / "bearing_probes.csv", "w", newline="") as f:
        cols = [k for k in plans[0] if k not in ("ratio_band", "end_xy",
                                                 "marker_xy", "legs")]
        w = csv.DictWriter(f, fieldnames=["legs_str"] + cols)
        w.writeheader()
        for p in plans:
            w.writerow({"legs_str": "-".join(map(str, p["legs"])),
                        **{k: p[k] for k in cols}})

    with open(OUT_DIR / "probes_prompts.jsonl", "w") as f:
        for p in plans:
            f.write(json.dumps({"id": p["id"], "video": f"{p['id']:03d}.mp4",
                                "question": QUESTION}) + "\n")

    report(plans, fallbacks, spirals)
    print(f"-> {OUT_DIR}")


if __name__ == "__main__":
    main()
