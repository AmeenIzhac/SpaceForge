"""Refined tail of the sphere->corridor morph, measured one step at a time.

The first pass removed every object in one go and jumped 15.9 -> 34.7 degrees,
which located the break only to within "the scene got empty". This walks the
same ground in small steps so the break can be pinned to a count, and it
rebuilds the walls properly: continuous slabs like a real corridor rather than
a grid of blocks with a jagged silhouette, and short enough at first to see
over.

Every step starts from M5 (target is a floor X at the start of the walk,
corridor wording, rigid constant-speed motion, straight legs and 90 degree
turns) and changes exactly one thing:

    N12 N8 N5 N3 N2 N1 N0   how many objects besides the target remain
    F   fence walls, 1.2 high — structure you can see over
    W   full-height walls, 2.7
    C   + ceiling and indoor light
    P   + five objects put back inside the corridor

    .venv/bin/python morph2.py --steps N12,N8,N5 --n 70
"""

import argparse
import json
import math
import random
from pathlib import Path

import morph_ablation as MA
import plane_gen as PG
import sphere_gen as SG

ROOT = Path(__file__).resolve().parent
HALF_W = MA.HALF_W
WALL_T = 0.3
FENCE_H = 1.2
FULL_H = 2.7

STEPS = {                      # name -> (objects kept, wall height or None, ceiling)
    "N12": (12, None, False), "N8": (8, None, False), "N5": (5, None, False),
    "N3": (3, None, False), "N2": (2, None, False), "N1": (1, None, False),
    "N0": (0, None, False),
    "F_fence": (0, FENCE_H, False), "W_full": (0, FULL_H, False),
    "C_ceiling": (0, FULL_H, True), "P_props": (5, FULL_H, True),
    # bisect the biggest single drop, and a control: an empty scene whose
    # GROUND is strongly textured. If the loss is about trackable visual
    # structure rather than objects as such, the checker should recover some
    # of it without putting a single object back.
    "N10": (10, None, False),
    "N0_checker": (0, None, False),
    "N0_checker_fence": (0, FENCE_H, False),
}
CHECKER_STEPS = {"N0_checker", "N0_checker_fence"}


def smooth_walls(poses, height):
    """Continuous wall slabs along each straight leg, mitred at the corners:
    the wall on the inside of a turn stops short of the corner and the one on
    the outside runs past it, so the two legs' walls meet in a flat line. A
    grid of blocks (the first attempt) leaves a stepped silhouette and can put
    a block where the camera walks."""
    legs = MA.legs_of(poses)
    if not legs:
        return []
    head = [math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in legs]
    out = []
    for i, ((ax, ay), (bx, by)) in enumerate(legs):
        th = head[i]
        dx, dy = math.cos(th), math.sin(th)
        px, py = -math.sin(th), math.cos(th)        # the camera's right
        L = math.hypot(bx - ax, by - ay)
        for sgn in (1, -1):
            s0, s1 = 0.0, L
            if i > 0:                                # joint at this leg's start
                inner = (math.sin(th - head[i - 1]) > 0) == (sgn > 0)
                s0 = HALF_W if inner else -HALF_W - 0.8
            if i < len(legs) - 1:                    # joint at its end
                inner = (math.sin(head[i + 1] - th) > 0) == (sgn > 0)
                s1 = L - HALF_W if inner else L + HALF_W + 0.8
            if s1 - s0 < 0.4:
                continue
            mid = (s0 + s1) / 2
            out.append({"name": "wall", "shape": "building",
                        "colour": MA.WALL_COL, "colour_name": "grey",
                        "x": round(ax + dx * mid + px * HALF_W * sgn, 3),
                        "y": round(ay + dy * mid + py * HALF_W * sgn, 3),
                        "yaw": round(-th, 4), "scale": 1.0, "no_shadow": True,
                        "dims": [round(s1 - s0, 2), height, WALL_T]})
    for (a, b), th, at_start in ((legs[0], head[0], True),
                                 (legs[-1], head[-1], False)):
        x, y = (a if at_start else b)
        # push the cap clear of the walker: the camera stands on the leg's
        # end point, so a cap 0.3 away is literally in its face
        d = -1.4 if at_start else 1.4
        out.append({"name": "endcap", "shape": "building",
                    "colour": MA.WALL_COL, "colour_name": "grey",
                    "x": round(x + math.cos(th) * d, 3),
                    "y": round(y + math.sin(th) * d, 3),
                    "yaw": round(-th + math.pi / 2, 4), "scale": 1.0,
                    "no_shadow": True,
                    "dims": [2 * HALF_W + WALL_T, height, WALL_T]})
    return out


def build(step, rng, sid):
    keep, wall_h, ceiling = STEPS[step]
    cfg = MA.cfg_for("M5_lattice_path")            # everything up to M5 is on
    s = MA.build_scene("M5_lattice_path", cfg, rng, sid)
    if s is None:
        return None
    s["level"] = step
    objs = s["objects"]
    tgt = [o for o in objs if o.get("is_target")]
    others = [o for o in objs if not o.get("is_target")]
    rng.shuffle(others)
    objs = tgt + others[:keep]
    if wall_h:
        objs = objs + smooth_walls([(p[0], p[1], p[2]) for p in s["poses"]],
                                   wall_h)
    if ceiling:
        c = MA.ceiling_object([(p[0], p[1], p[2]) for p in s["poses"]])
        c["y_off"] = FULL_H + 0.05
        objs = objs + [c]
        s["indoor"] = True
    if step in CHECKER_STEPS:
        s["ground_style"] = "checker"
        s["ground_scale"] = 150
    s["objects"] = objs
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--n", type=int, default=70)
    ap.add_argument("--seed", type=int, default=8080)
    args = ap.parse_args()

    for step in args.steps.split(","):
        rng = random.Random(args.seed + sum(map(ord, step)))
        sid = 400000 + list(STEPS).index(step) * 1000
        scenes = []
        while len(scenes) < args.n:
            s = build(step, rng, sid)
            if s:
                scenes.append(s)
                sid += 1
        (ROOT / f"probes/m2_{step}.json").write_text(
            json.dumps({"note": step, "scenes": scenes}))
        keep, wh, ce = STEPS[step]
        print(f"{step:10s} {len(scenes)} scenes | {keep} other objects | "
              f"walls {wh or '-'} | ceiling {ce}")


if __name__ == "__main__":
    main()
