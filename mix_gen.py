"""Training scenes for the mixed-domain run: plain, fenced plain, corridor.

Three worlds, one task. Every scene asks the same kind of question — where is
a named thing, relative to the camera, at the end (or start) of the walk — but
the world around it changes:

    plain    open ground, objects scattered around          (as before)
    fence    the same, walled in at 1.2 m: you can see over
    indoor   full-height walls and a ceiling, props inside

Camera motion is the natural kind throughout (variable speed, pauses, backward
stretches, sway, pitch, roll), so the world is the only thing that varies.

    .venv/bin/python mix_gen.py --world fence --n 420 --out probes/mix_fence.json
"""

import argparse
import json
import math
import random
from pathlib import Path

import morph2 as M2
import natural_motion as NM
import plane_gen as PG
import sphere_gen as SG

ROOT = Path(__file__).resolve().parent


def build(world, rng, sid, secs=10.0):
    poses, turns = NM.natural_path(rng, secs * rng.uniform(0.9, 1.1))
    flat = [(p[0], p[1], p[2]) for p in poses]
    saved = PG.camera_path
    PG.camera_path = lambda *a, **k: (flat, turns)
    try:
        s = SG.draw(rng, sid)
    finally:
        PG.camera_path = saved
    if s is None:
        return None
    s["poses"] = poses
    s["turns"], s["n_turns"] = turns, len(turns)
    s["secs"] = round(len(poses) / PG.FPS, 2)
    s["level"] = world

    if world != "plain":
        # a fenced or roofed corridor needs a corridor-shaped route, and the
        # natural path wanders: keep the wandering, wall what it actually did
        h = M2.FENCE_H if world == "fence" else M2.FULL_H
        s["objects"] = s["objects"] + M2.smooth_walls(flat, h)
        if world == "indoor":
            import morph_ablation as MA
            c = MA.ceiling_object(flat)
            c["y_off"] = M2.FULL_H + 0.05
            s["objects"] = s["objects"] + [c]
            s["indoor"] = True
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", choices=("plain", "fence", "indoor"), required=True)
    ap.add_argument("--n", type=int, default=420)
    ap.add_argument("--seed", type=int, default=606)
    ap.add_argument("--seed0", type=int, default=500000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    scenes = []
    while len(scenes) < args.n:
        s = build(args.world, rng, args.seed0 + len(scenes))
        if s:
            scenes.append(s)
    (ROOT / args.out).write_text(json.dumps({"note": f"mix {args.world}",
                                             "scenes": scenes}))
    print(f"{args.world}: {len(scenes)} scenes -> {args.out} "
          f"({sum(len(s['poses']) for s in scenes)} frames, "
          f"{sum(len(s['objects']) for s in scenes)//len(scenes)} objects/scene)")


if __name__ == "__main__":
    main()
