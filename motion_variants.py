"""Does the sphere model do path integration, or match one motion template?

Training used exactly one motion shape: straight leg, one eased turn, straight
leg, stop. Only the turn size, direction, speed and leg lengths varied. A model
could fit that with a narrow recipe rather than by integrating motion, and the
in-domain score would look identical either way.

So: same scenes, same question, same everything — only the camera's *motion
shape* changes, in ways never trained. Objects, ground, sky and question are
drawn by `sphere_gen` exactly as in training.

    same        the trained shape (control)
    two_turns   two eased turns instead of one
    no_turn     straight line, no turn at all
    curved      one continuous gentle arc, no discrete turn
    stop_go     the trained shape plus a mid-leg stop and a speed change
    fast_turn   the turn happens in 0.4 s instead of 1.3 s
    backtrack   walks forward, turns, then walks back the way it came

    .venv/bin/python motion_variants.py --n 40
"""

import argparse
import json
import math
import random
from pathlib import Path

import plane_gen as PG
import sphere_gen as SG

ROOT = Path(__file__).resolve().parent
FPS = PG.FPS


def path(kind, rng, secs, speed):
    yaw = rng.uniform(0, 2 * math.pi)
    x = y = 0.0
    poses, turns = [], []

    def step(sp, dy):
        nonlocal x, y, yaw
        yaw += dy
        x += math.cos(yaw) * sp / FPS
        y += math.sin(yaw) * sp / FPS
        poses.append((round(x, 4), round(y, 4), round(yaw, 5)))

    def walk(t, sp=None):
        for _ in range(max(1, int(t * FPS))):
            step(speed if sp is None else sp, 0.0)

    def turn(deg, dur):
        n = max(2, int(dur * FPS))
        d = math.radians(deg)
        turns.append({"start_frame": len(poses), "dir": "R" if d > 0 else "L",
                      "deg": round(deg, 1)})
        for i in range(n):
            w = math.sin(math.pi * (i + 0.5) / n)
            step(speed * 0.25, d * w / (n * 2 / math.pi))

    amt = lambda: rng.uniform(35, 115) * rng.choice((-1, 1))

    if kind == "same":
        leg = (secs - 1.3 - 0.6) / 2
        walk(leg); turn(amt(), 1.3); walk(leg)
    elif kind == "two_turns":
        leg = (secs - 2.6 - 0.6) / 3
        walk(leg); turn(amt(), 1.3); walk(leg); turn(amt(), 1.3); walk(leg)
    elif kind == "no_turn":
        walk(secs - 0.6)
    elif kind == "curved":                       # constant slow rotation, no turn
        total = math.radians(rng.uniform(40, 110)) * rng.choice((-1, 1))
        n = int((secs - 0.6) * FPS)
        for _ in range(n):
            step(speed, total / n)
    elif kind == "stop_go":
        leg = (secs - 1.3 - 0.6 - 1.0) / 2
        walk(leg * 0.5)
        for _ in range(int(1.0 * FPS)):          # stand still mid-leg
            step(0.0, 0.0)
        walk(leg * 0.5, sp=speed * 1.7)          # then speed up
        turn(amt(), 1.3)
        walk(leg * 0.6, sp=speed * 0.5)
        walk(leg * 0.4, sp=speed * 1.4)
    elif kind == "fast_turn":
        leg = (secs - 0.4 - 0.6) / 2
        walk(leg); turn(amt(), 0.4); walk(leg)
    elif kind == "backtrack":
        leg = (secs - 1.3 - 0.6) / 2
        walk(leg); turn(175.0 * rng.choice((-1, 1)), 1.3); walk(leg)
    else:
        raise KeyError(kind)
    for _ in range(int(0.6 * FPS)):
        step(0.0, 0.0)
    return poses, turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--kinds", default="same,two_turns,no_turn,curved,"
                                       "stop_go,fast_turn,backtrack")
    args = ap.parse_args()

    sid = 80000
    for kind in args.kinds.split(","):
        rng = random.Random(args.seed + hash(kind) % 9999)
        scenes = []
        while len(scenes) < args.n:
            fov = rng.uniform(58, 78)
            hfov = PG.half_fov(fov)
            speed = rng.uniform(1.0, 1.8)
            poses, turns = path(kind, rng, rng.uniform(6.0, 9.0), speed)
            # reuse sphere_gen's object placement verbatim so only motion differs
            s = {"id": sid, "level": kind, "n_turns": len(turns), "turns": turns,
                 "fps": FPS, "secs": round(len(poses) / FPS, 2),
                 "eye": round(rng.uniform(1.4, 1.8), 2), "fov": round(fov, 1),
                 "hfov": round(hfov, 1), "speed": round(speed, 2),
                 "ground_style": "noise",
                 "ground_hue": round(rng.uniform(0.24, 0.35), 3),
                 "ground_scale": rng.randint(70, 180),
                 "sky_hue": round(rng.uniform(0.53, 0.63), 3),
                 "sun_az": round(rng.uniform(0, 6.28), 2),
                 "sun_el": round(rng.uniform(0.55, 1.15), 2),
                 "poses": poses}
            objs = SG.draw.__wrapped__ if False else None
            # place objects with sphere_gen's own rules
            saved = PG.camera_path
            PG.camera_path = lambda *a, **k: (poses, turns)
            try:
                built = SG.draw(rng, sid)
            finally:
                PG.camera_path = saved
            if built is None:
                continue
            built.update({k: s[k] for k in ("level", "n_turns", "turns")})
            scenes.append(built)
            sid += 1
        out = ROOT / f"probes/mv_{kind}.json"
        out.write_text(json.dumps({"note": f"motion variant {kind}",
                                   "scenes": scenes}))
        print(f"{kind:10s} {len(scenes)} scenes -> {out.name} "
              f"({sum(len(x['poses']) for x in scenes)} frames)")


if __name__ == "__main__":
    main()
