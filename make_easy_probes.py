"""An easy ladder for the bearing task: four levels, each a strict floor.

The benchmark set scores at chance, and a single number at chance says nothing
about *where* the task breaks — question comprehension, turn tracking, and leg
integration all fail the same way. So this builds four levels that isolate
them, ordered so each adds exactly one demand:

  L0 straight     0 turns.  The X is directly behind you at the end, always.
                  Answering needs no spatial reasoning at all, only reading the
                  question. A model that says 000 here has not understood it.
  L1 one_turn     1 turn, legs within 1:1.2.  Needs the turn's direction; the
                  two legs are near enough equal that their lengths hardly
                  matter.
  L2 two_turns    2 turns, legs within 1:1.2.  Compose two rotations, still
                  without integrating distances.
  L3 three_turns  3 turns, legs 1:1.8-2.6.  The first level where how long each
                  straight ran actually moves the answer.

Walks are short on purpose — 4 to 15 lattice steps, roughly 10-40 s, against
the benchmark's 44-81 s. At the eval's 2 fps / 128 frame budget a walk this
short is sampled whole, so a wrong answer cannot be blamed on frames the model
never saw.

Every level records `const_baseline_deg`: the error of the best single constant
answer for that level. L0's is 0 by construction. Score a level against that,
not against 90 - chance for a uniform guess is not chance for a level whose
answers cluster.

    .venv/bin/python make_easy_probes.py          -> probes/easy_probes.json
    .venv/bin/python render_probes.py --plans probes/easy_probes.json \
        --out out/easy
"""

import argparse
import json
import math
import random
from pathlib import Path

import generate as G
import make_probes as MP

ROOT = Path(__file__).resolve().parent
SEED = 7717
SEED_BASE = 900_000          # scenery seeds, clear of benchmark and trainset
CANDIDATES = 24              # routes drawn per slot, best-spread one kept

# (name, n_turns, ratio band, total lattice steps, how many)
LEVELS = [
    ("L0_straight",    0, (1.0, 1.2), (4, 10), 10),
    ("L1_one_turn",    1, (1.0, 1.2), (4, 9),  15),
    ("L2_two_turns",   2, (1.0, 1.2), (6, 12), 15),
    ("L3_three_turns", 3, (1.8, 2.6), (9, 15), 15),
]

QUESTION = (
    "This is a first-person video of someone walking through corridors. A "
    "large red X is painted on the floor at the spot where the walk begins; it "
    "is visible in the opening seconds, and the walker then leaves it behind. "
    "Freeze on the final frame and treat the direction the camera is facing at "
    "that moment as north (000 degrees). What is the compass bearing from the "
    "camera's final position to the red X? Answer in degrees clockwise from "
    "that forward direction, 0-359: 090 is directly to the right, 180 directly "
    "behind, 270 directly to the left.")


def draw_legs(rng, n_legs, band, total):
    """Leg lengths inside `band` summing inside `total`.

    G.draw_legs places the ratio extremes on two distinct legs, which needs at
    least two of them; a level with no turns has one, and its ratio is 1:1
    whatever it draws."""
    lo, hi = total
    if n_legs == 1:
        return [rng.randint(max(1, lo), hi)]
    return G.draw_legs(rng, n_legs, band, total_min=lo, total_max=hi)


def draw_plan(rng, n_turns, band, total, pattern, tries=400):
    """A self-avoiding route with exactly `n_turns` turns, or None."""
    for _ in range(tries):
        try:
            legs = draw_legs(rng, n_turns + 1, band, total)
        except RuntimeError:
            continue
        turns = MP.turn_string(rng, n_turns, pattern) if n_turns else ""
        for ordered in (legs, sorted(legs)):
            route = G.build_route(turns, ordered)
            if len(set(route)) == len(route):
                return turns, ordered
    return None


def const_baseline(bearings):
    """Error of the best single constant answer over these bearings — what a
    model scores by ignoring the video and always saying the same thing."""
    if not bearings:
        return None
    best = min(range(360),
               key=lambda c: sum(MP.circ_err(c, b) for b in bearings))
    return best, sum(MP.circ_err(best, b) for b in bearings) / len(bearings)


def build():
    rng = random.Random(SEED)
    plans, seen = [], set()
    for name, n_turns, band, total, count in LEVELS:
        bins = [0] * MP.BEARING_BINS           # spread answers within a level
        for k in range(count):
            # alternate the requested pattern so L/R and spiral/zigzag are
            # both represented at every level that has more than one turn
            pattern = ["alt", "same", "rand"][k % 3]
            cands = []
            for _ in range(CANDIDATES):
                got = (draw_plan(rng, n_turns, band, total, pattern)
                       or draw_plan(rng, n_turns, band, total, "rand"))
                if got is None:
                    raise RuntimeError(f"no route for {name}")
                cands.append(got)
            # prefer a route this set has not used, but L0 has only a handful
            # of distinct geometries (one leg, a few lengths) and repeating one
            # under a different scenery seed is a different video and a fair
            # sample — so fall back rather than fail
            fresh = [c for c in cands if (c[0], tuple(c[1])) not in seen]
            pick = min(fresh or cands, key=lambda c:
                       bins[MP.bearing_bin(MP.bearing_of(*c))])
            turns, legs = pick
            seen.add((turns, tuple(legs)))
            bins[MP.bearing_bin(MP.bearing_of(turns, legs))] += 1
            pid = len(plans)
            plans.append({
                "id": pid,
                "seed": SEED_BASE + pid,
                "level": name,
                "turns": turns,
                "legs": legs,
                "n_turns": n_turns,
                "ratio_band": list(band),
                "ratio": round(max(legs) / min(legs), 2),
                "total_steps": sum(legs),
                "pattern": pattern,
                **MP.measure(turns, legs),
            })

    # per-level constant-answer baseline, stamped on every plan in the level
    for name, *_ in LEVELS:
        sub = [p for p in plans if p["level"] == name]
        c, err = const_baseline([p["bearing_gt_deg"] for p in sub])
        for p in sub:
            p["const_baseline_deg"] = round(err, 2)
            p["const_baseline_answer"] = c
    return plans


def report(plans):
    held = {(p["turns"], tuple(p["legs"])) for p in json.loads(
        (ROOT / "probes/bearing_probes.json").read_text())["plans"]}
    overlap = sum((p["turns"], tuple(p["legs"])) in held for p in plans)

    print(f"{len(plans)} plans over {len(LEVELS)} levels"
          + (f"  ({overlap} share a route with the benchmark)"
             if overlap else ""))
    hdr = (f"{'level':16s} {'n':>3} {'turns':>5} {'steps':>9} {'video s':>11} "
           f"{'const base':>10} {'equal-leg':>9} {'cardinal':>8}")
    print(hdr)
    for name, *_ in LEVELS:
        sub = [p for p in plans if p["level"] == name]
        rng_of = lambda k: (min(p[k] for p in sub), max(p[k] for p in sub))
        s0, s1 = rng_of("total_steps")
        d0, d1 = rng_of("duration_s")
        mean = lambda k: sum(p[k] for p in sub) / len(sub)
        print(f"{name:16s} {len(sub):>3} {sub[0]['n_turns']:>5} "
              f"{f'{s0}-{s1}':>9} {f'{d0:.0f}-{d1:.0f}':>11} "
              f"{sub[0]['const_baseline_deg']:>7.1f}@{sub[0]['const_baseline_answer']:03d} "
              f"{mean('equal_leg_err_deg'):>9.1f} {mean('cardinal_err_deg'):>8.1f}")

    hist = [0] * MP.BEARING_BINS
    for p in plans:
        hist[MP.bearing_bin(p["bearing_gt_deg"])] += 1
    print("  answer spread, 30-degree bins from 000: "
          + " ".join(str(h) for h in hist))
    total_s = sum(p["duration_s"] for p in plans)
    print(f"  rendering all {len(plans)} would be {total_s / 60:.0f} min of "
          f"video ({total_s / sum(1 for _ in plans):.0f} s each on average)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="probes/easy_probes.json")
    args = ap.parse_args()

    plans = build()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "question": QUESTION,
        "scoring": MP.SCORING + (" Levels cluster their answers, so compare "
                                 "against const_baseline_deg, not against 90."),
        "answer_field": "bearing_gt_deg",
        "note": ("Easy ladder, not the benchmark. Levels isolate question "
                 "comprehension (L0), turn direction (L1), composing turns "
                 "(L2) and leg integration (L3)."),
        "convention": ("World axes are x east, y south; the camera looks along "
                       "(cos yaw, sin yaw), so bearing 090 is (-sin, cos), its "
                       "right. One lattice step is 3 world units."),
        "generator": {"seed": SEED, "seed_base": SEED_BASE,
                      "levels": [list(l) for l in LEVELS],
                      "mood": G.MOOD, "junctions": G.JUNCTIONS,
                      "marker_ahead": G.MARKER_AHEAD, "speed": G.SPEED,
                      "fps": G.FPS},
        "plans": plans,
    }, indent=2))

    report(plans)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
