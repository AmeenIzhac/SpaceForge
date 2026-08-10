"""Training plans for the bearing task — same generator, disjoint seeds.

Writes plans in the same shape as `make_probes.py`, so `render_probes.py`
draws them unchanged, plus the extra timing ground truth a supervision trace
needs: when each corner was turned and how long each straight ran, in seconds
of video. Those are the quantities a model could in principle read off the
frame timestamps, so a trace built from them asks for nothing unobservable.

Seeds start at TRAIN_SEED_BASE, far from the benchmark's 2029..2128, and the
`(turns, legs)` route of every plan is checked against the benchmark's so no
training walk is a rendered twin of a test walk.

    .venv/bin/python make_trainset.py --n 800 --out probes/trainset.json
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

import generate as G
import world

ROOT = Path(__file__).resolve().parent
TRAIN_SEED_BASE = 500_000

# The benchmark runs 1..8 turns over 34 lattice steps, up to 81 s of video.
# Training leans short: a 10-step walk is ~24 s, which at a fixed frame budget
# is nearly three times the temporal resolution per corner, and renders and
# trains proportionally faster. The long tail keeps the distribution touching
# the benchmark's range so the model is not purely trained out of domain.
TIERS = [
    # (weight, turn choices, total-step bounds)
    (0.45, [1, 2],       (4, 10)),
    (0.35, [3, 4],       (6, 14)),
    (0.15, [5, 6],       (9, 20)),
    (0.05, [7, 8],       (12, 30)),
]
PATTERNS = ["alt", "same", "rand", "rand"]


def turn_string(rng, n_turns, pattern):
    if pattern == "same":
        return rng.choice("LR") * n_turns
    if pattern == "alt":
        first = rng.choice("LR")
        return "".join(first if i % 2 == 0 else "LR"[first == "L"]
                       for i in range(n_turns))
    return "".join(rng.choice("LR") for _ in range(n_turns))


def draw_plan(rng, n_turns, band, pattern, total_bounds, tries=600):
    n_legs = n_turns + 1
    lo, hi = total_bounds
    for _ in range(tries):
        try:
            # a no-turn walk is a single straight: G.draw_legs places the
            # ratio extremes on two different legs and there is only one
            legs = ([rng.randint(max(1, lo), hi)] if n_legs == 1 else
                    G.draw_legs(rng, n_legs, band,
                                total_min=max(lo, n_legs), total_max=hi))
        except RuntimeError:
            continue
        turns = turn_string(rng, n_turns, pattern)
        for ordered in (legs, sorted(legs)):
            route = G.build_route(turns, ordered)
            if len(set(route)) == len(route):
                return turns, ordered
    return None


def timing(turns, legs):
    """Per-leg and per-turn wall-clock, straight off the baked camera track.

    The turn arc is found in the yaw trace rather than assumed, so the numbers
    match the video frame for frame — including the corner rounding, which
    eats the last of one leg and the start of the next."""
    route, _ = G.normalize(G.build_route(turns, legs), [])
    _cc, _legs_idx, marker, poly, poses = G.solve_walk(route)

    yaw = np.unwrap([p[2] for p in poses])
    moving = np.abs(np.diff(yaw)) > math.radians(0.3)
    runs, start = [], None
    for i, t in enumerate(moving):
        if t and start is None:
            start = i
        elif not t and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(moving)))

    fps = G.FPS
    turn_events = [{"t_start": round(a / fps, 2), "t_end": round(b / fps, 2),
                    "dir": "L" if yaw[b] < yaw[a] else "R",
                    "delta_deg": round(math.degrees(yaw[b] - yaw[a]))}
                   for a, b in runs]
    bounds = [0.0] + [e["t_end"] for e in turn_events] + [len(poses) / fps]
    starts = [0.0] + [e["t_end"] for e in turn_events]
    ends = [e["t_start"] for e in turn_events] + [len(poses) / fps]
    leg_times = [{"t_start": round(s, 2), "t_end": round(e, 2),
                  "seconds": round(e - s, 2)} for s, e in zip(starts, ends)]
    del bounds
    return marker, poses, turn_events, leg_times


def measure(turns, legs):
    """Ground truth for every task the trainer supervises."""
    marker, poses, turn_events, leg_times = timing(turns, legs)
    end = poses[-1]
    bearing = G.bearing_deg(end, marker)
    dist = math.hypot(marker[0] - end[0], marker[1] - end[1])
    route, _ = G.normalize(G.build_route(turns, legs), [])
    _cc, _li, _m, poly, _p = G.solve_walk(route)
    cum = sum(90.0 if t == "R" else -90.0 for t in turns)
    return {
        "bearing_gt_deg": round(bearing, 2),
        "distance_to_x": round(dist, 2),
        "final_yaw_deg": round(math.degrees(end[2]) % 360.0, 2),
        "end_xy": [round(end[0], 2), round(end[1], 2)],
        "marker_xy": [round(marker[0], 2), round(marker[1], 2)],
        "path_len": round(float(world.polyline_length(poly)), 2),
        "duration_s": round(len(poses) / G.FPS, 1),
        "n_frames": len(poses),
        "cum_turn_deg": cum,
        "net_turn_deg": ((cum + 180) % 360) - 180,
        "crow_over_path": round(dist / max(float(world.polyline_length(poly)),
                                           1e-9), 3),
        "turn_events": turn_events,
        "leg_times": leg_times,
    }


def build(n, seed, exclude_routes, tiers=None, bands=None, tag=None,
          seed_base=TRAIN_SEED_BASE, forbidden=None):
    """`forbidden` is the holdout: routes that must never be emitted, however
    dry the draw gets. `exclude_routes` is only the routes this set has already
    used, which the stuck-fallback below is allowed to repeat. Keeping them in
    one set — as this did — means a narrow tier that exhausts its geometries
    starts handing back benchmark routes, and 21 of 40 test routes end up in
    training."""
    forbidden = forbidden or set()
    rng = random.Random(seed)
    tiers = tiers or TIERS
    bands = bands or G.RATIO_BANDS
    weights = [t[0] for t in tiers]
    plans, dupes = [], 0
    # counted per turn count, not globally: a 0-turn walk over 4-10 steps has
    # only seven distinct routes and would otherwise never clear a shared
    # counter that the plentiful 2-turn draws keep resetting, leaving the easy
    # rungs of a curriculum set almost empty
    stuck = {}
    while len(plans) < n:
        _w, turn_choices, bounds = rng.choices(tiers, weights=weights)[0]
        n_turns = rng.choice(turn_choices)
        band = rng.choice(bands)
        pattern = rng.choice(PATTERNS)
        got = (draw_plan(rng, n_turns, band, pattern, bounds)
               or draw_plan(rng, n_turns, band, "rand", bounds))
        if got is None:
            continue
        turns, legs = got
        if (turns, tuple(legs)) in forbidden:
            dupes += 1
            continue                      # holdout: never emit, at any cost
        if (turns, tuple(legs)) in exclude_routes:
            dupes += 1
            # A narrow curriculum tier has few distinct routes — 0-1 turns over
            # 4-10 steps is a few dozen — so insisting on unique geometry caps
            # the set far below the requested size. Once the draws are clearly
            # exhausted, allow a repeat: the scenery seed differs, so it is a
            # different corridor and a different video, only the route repeats.
            stuck[n_turns] = stuck.get(n_turns, 0) + 1
            if stuck[n_turns] < 30:
                continue
        else:
            exclude_routes.add((turns, tuple(legs)))
        stuck[n_turns] = 0
        pid = len(plans)
        plans.append({
            "id": pid, "seed": seed_base + pid,
            "turns": turns, "legs": legs, "n_turns": n_turns,
            "ratio_band": list(band),
            "ratio": round(max(legs) / min(legs), 2),
            "total_steps": sum(legs), "pattern": pattern,
            **({"level": tag} if tag else {}),
            **measure(turns, legs),
        })
    return plans, dupes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=97)
    ap.add_argument("--out", default="probes/trainset.json")
    # Curriculum knobs. The default TIERS span the benchmark's range; pass
    # these to mint a set at one difficulty instead — e.g. the level the model
    # was found to have some grip on, or one rung below it.
    ap.add_argument("--turns", default=None,
                    help="comma-separated turn counts, e.g. 0,1 (overrides "
                         "the default tier mix)")
    ap.add_argument("--steps", default=None,
                    help="min,max total lattice steps, e.g. 4,10")
    ap.add_argument("--bands", default=None,
                    help="indices into generate.RATIO_BANDS, e.g. 0,1 for "
                         "near-equal legs only")
    ap.add_argument("--tag", default=None,
                    help="stamped on every plan as `level`")
    ap.add_argument("--seed-base", type=int, default=TRAIN_SEED_BASE,
                    help="scenery seeds start here; give a later batch its "
                         "own range or it redraws the first batch's corridors")
    ap.add_argument("--holdout",
                    default="probes/bearing_probes.json,probes/easy_probes.json",
                    help="comma-separated plan files whose routes must stay "
                         "out of the training set")
    args = ap.parse_args()

    held = set()
    for path in filter(None, args.holdout.split(",")):
        f = ROOT / path
        if not f.exists():
            continue
        for p in json.loads(f.read_text())["plans"]:
            held.add((p["turns"], tuple(p["legs"])))
    n_held = len(held)

    tiers = bands = None
    if args.turns or args.steps:
        turn_choices = ([int(t) for t in args.turns.split(",")]
                        if args.turns else [1, 2])
        lo, hi = ((int(x) for x in args.steps.split(","))
                  if args.steps else (4, 12))
        tiers = [(1.0, turn_choices, (lo, hi))]
    if args.bands:
        bands = [G.RATIO_BANDS[int(i)] for i in args.bands.split(",")]

    plans, dupes = build(args.n, args.seed, set(), tiers, bands, args.tag,
                         args.seed_base, forbidden=held)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "note": "training plans; disjoint routes and seeds from the benchmark",
        "seed_base": TRAIN_SEED_BASE, "tiers": [list(t) for t in TIERS],
        "plans": plans}, indent=2))

    secs = sum(p["duration_s"] for p in plans)
    turns = {}
    for p in plans:
        turns[p["n_turns"]] = turns.get(p["n_turns"], 0) + 1
    print(f"{len(plans)} plans, {dupes} draws rejected as repeats of a "
          f"benchmark route or of an earlier training route "
          f"({n_held} benchmark routes held out)")
    print("  turns:", " ".join(f"{k}:{v}" for k, v in sorted(turns.items())))
    print(f"  video: {secs / 60:.0f} min total, "
          f"{min(p['duration_s'] for p in plans):.0f}"
          f"..{max(p['duration_s'] for p in plans):.0f} s each "
          f"(median {sorted(p['duration_s'] for p in plans)[len(plans) // 2]:.0f})")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
