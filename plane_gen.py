"""Open-plane spatial task: scene geometry + ground truth. No questions here.

No corridors, no walls, no fixed target. A camera walks across an open plain
scattered with objects. `plane_tasks.py` turns each scene into questions; this
file owns only what is measurable: camera poses, object placement, per-object
visibility, and the randomisation that stops a scene being memorable.

Randomised per scene so no fixed look can be learned: ground hue and pattern
scale, sky hue, sun azimuth/elevation, camera eye height, field of view,
walking speed, object count, object sizes and rotations.

Difficulty is the camera path alone:

    P0  no turns          walk straight
    P1  one turn
    P2  two turns
    P3  three turns, longer walk
    P4  four turns        never trained — extrapolation only

Colours are split: three are reserved for test scenes, so a test question can
name a thing whose colour the model has never been asked to localise.

    .venv/bin/python plane_gen.py --split train --n 400 --out probes/plane_train.json
"""

import argparse
import json
import math
import random
from pathlib import Path

import generate as G

ROOT = Path(__file__).resolve().parent
FPS = 30

SHAPES = ["cube", "sphere", "cylinder", "cone", "torus", "capsule",
          "octahedron", "pyramid"]
COLOURS = {
    "red": "#d32f2f", "blue": "#1f5fd0", "green": "#2e9e46",
    "yellow": "#e6c018", "orange": "#e8761f", "purple": "#7a3fbf",
    "cyan": "#1fb6c9", "pink": "#e070a8", "white": "#eceae4",
    "brown": "#8a5a34",
}
TEST_ONLY_COLOURS = ["pink", "white", "brown"]
TRAIN_COLOURS = [c for c in COLOURS if c not in TEST_ONLY_COLOURS]

LEVELS = {"P0": (0, 4.5), "P1": (1, 6.5), "P2": (2, 8.5),
          "P3": (3, 10.5), "P4": (4, 12.0)}


def half_fov(fov_deg, w=640, h=360):
    """Horizontal half-angle; three.js takes a *vertical* fov."""
    return math.degrees(math.atan(math.tan(math.radians(fov_deg) / 2) * w / h))


def camera_path(rng, n_turns, secs, speed):
    """Straight legs joined by eased free rotations — not corridor corners:
    turn sizes vary continuously and the walker keeps drifting while turning."""
    turn_t = 1.3
    walk_t = max(1.0, (secs - n_turns * turn_t - 0.6) / (n_turns + 1))
    yaw = rng.uniform(0, 2 * math.pi)
    x = y = 0.0
    poses, turns = [], []

    def step(sp, dy):
        nonlocal x, y, yaw
        yaw += dy
        x += math.cos(yaw) * sp / FPS
        y += math.sin(yaw) * sp / FPS
        poses.append((round(x, 4), round(y, 4), round(yaw, 5)))

    for t in range(n_turns + 1):
        for _ in range(int(walk_t * FPS)):
            step(speed, 0.0)
        if t < n_turns:
            d = math.radians(rng.uniform(35, 115)) * rng.choice((-1, 1))
            n = int(turn_t * FPS)
            turns.append({"start_frame": len(poses),
                          "dir": "R" if d > 0 else "L",
                          "deg": round(math.degrees(d), 1)})
            for i in range(n):
                w = math.sin(math.pi * (i + 0.5) / n)
                step(speed * 0.25, d * w / (n * 2 / math.pi))
    for _ in range(int(0.6 * FPS)):
        step(0.0, 0.0)
    return poses, turns


def place_objects(rng, poses, palette, n_obj):
    pts = [(p[0], p[1]) for p in poses[::10]]
    objs, combos = [], [(s, c) for s in SHAPES for c in palette]
    rng.shuffle(combos)
    for shape, colour in combos:
        if len(objs) >= n_obj:
            break
        for _ in range(60):
            ax, ay = pts[rng.randrange(len(pts))]
            ang, rad = rng.uniform(0, 6.283), rng.uniform(3.0, 12.0)
            ox, oy = ax + rad * math.cos(ang), ay + rad * math.sin(ang)
            if min(math.hypot(px - ox, py - oy) for px, py in pts) < 2.2:
                continue
            if any(math.hypot(ox - o["x"], oy - o["y"]) < 3.2 for o in objs):
                continue
            objs.append({"name": f"{colour} {shape}", "shape": shape,
                         "colour": COLOURS[colour], "colour_name": colour,
                         "x": round(ox, 3), "y": round(oy, 3),
                         "scale": round(rng.uniform(0.75, 1.5), 2),
                         "yaw": round(rng.uniform(0, 6.28), 3)})
            break
    return objs


def annotate(o, poses, hfov):
    """When the object is on screen, and where it ends up."""
    vis = [i for i, (x, y, yaw) in enumerate(poses)
           if min(G.bearing_deg((x, y, yaw), (o["x"], o["y"])),
                  360 - G.bearing_deg((x, y, yaw), (o["x"], o["y"]))) <= hfov]
    last = poses[-1]
    b = G.bearing_deg((last[0], last[1], last[2]), (o["x"], o["y"]))
    o["seen_frames"] = len(vis)
    o["first_seen"] = vis[0] if vis else None
    o["seen_early"] = bool(vis and vis[0] < len(poses) * 0.6 and len(vis) >= 10)
    o["bearing_final"] = round(b, 2)
    o["visible_at_end"] = bool(min(b, 360 - b) <= hfov)
    o["distance_end"] = round(math.hypot(o["x"] - last[0],
                                         o["y"] - last[1]), 2)
    return o


def draw_scene(rng, level, palette, sid):
    n_turns, secs = LEVELS[level]
    fov = rng.uniform(58, 78)
    hfov = half_fov(fov)
    speed = rng.uniform(0.95, 1.8)
    poses, turns = camera_path(rng, n_turns, secs, speed)
    objs = [annotate(o, poses, hfov)
            for o in place_objects(rng, poses, palette, rng.randint(3, 6))]
    if not any(o["seen_early"] for o in objs):
        return None
    return {"id": sid, "level": level, "n_turns": n_turns, "turns": turns,
            "fps": FPS, "secs": round(len(poses) / FPS, 2),
            "eye": round(rng.uniform(1.35, 1.8), 2), "fov": round(fov, 1),
            "hfov": round(hfov, 1), "speed": round(speed, 2),
            "ground_hue": round(rng.uniform(0, 1), 3),
            "ground_scale": rng.randint(60, 220),
            "sky_hue": round(rng.uniform(0.5, 0.68), 3),
            "sun_az": round(rng.uniform(0, 6.28), 2),
            "sun_el": round(rng.uniform(0.5, 1.2), 2),
            "poses": poses, "objects": objs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "test"), required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--levels", default="P0:1,P1:3,P2:3,P3:2")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    weights = {}
    for part in args.levels.split(","):
        k, w = part.split(":")
        weights[k] = float(w)
    palette = TRAIN_COLOURS if args.split == "train" else list(COLOURS)

    rng = random.Random(args.seed)
    scenes, per = [], {k: 0 for k in weights}
    while len(scenes) < args.n:
        lv = min(per, key=lambda k: per[k] / weights[k])
        s = draw_scene(rng, lv, palette, args.seed0 + len(scenes))
        if s:
            per[lv] += 1
            scenes.append(s)

    (ROOT / args.out).write_text(json.dumps(
        {"note": f"open-plane {args.split} scenes", "scenes": scenes}))
    frames = sum(len(s["poses"]) for s in scenes)
    print(f"{len(scenes)} scenes -> {args.out}")
    print("  levels:", {k: per[k] for k in sorted(per)})
    print(f"  {frames} frames, {frames / FPS / 60:.0f} min of video")
    print(f"  objects/scene {sum(len(s['objects']) for s in scenes)/len(scenes):.1f}"
          f", seen-early {sum(sum(o['seen_early'] for o in s['objects']) for s in scenes)/len(scenes):.1f}")


if __name__ == "__main__":
    main()
