"""Specs + ground truth for the Three.js open-plane transfer test.

The scene itself is drawn by photoreal/plane3.js (real Three.js, sky, sun,
shadows); this file owns everything measurable: camera poses per frame,
object placement, and the bearing ground truth via the same G.bearing_deg
used everywhere else. World frame is x east, y south, yaw as in generate.py;
the JS maps it to three.js as (X=x, Y=up, Z=y) so the convention is identical.

    .venv/bin/python plane3_make.py        # -> photoreal/plane3_spec.json
                                           #    probes/plane3_probes.json
"""

import json
import math
import random
from pathlib import Path

import numpy as np

import generate as G
import make_probes as MP

ROOT = Path(__file__).resolve().parent
FPS = 30
N_SCENES = 12
SEED0 = 880_000
EYE = 1.6                      # metres; a person, not the corridor dwarf-cam

OBJECTS = [
    ("red cube",       "box",      [0.9, 0.9, 0.9], "#d42824"),
    ("blue ball",      "sphere",   [0.55],          "#2858dc"),
    ("green pillar",   "cylinder", [0.35, 0.35, 2.2], "#1e9640"),
    ("yellow crate",   "box",      [1.3, 0.8, 0.9], "#e8b216"),
    ("orange cone",    "cone",     [0.55, 1.4],     "#ec7420"),
]

Q_FINAL = (
    "This is a first-person video of someone walking across an open plain "
    "with several objects standing on the ground. Freeze on the FINAL frame "
    "and treat the direction the camera is facing at that moment as north "
    "(000 degrees). What is the compass bearing from the camera's final "
    "position to the {name}? Answer in degrees clockwise from that forward "
    "direction, 0-359: 090 is directly to the right, 180 directly behind, "
    "270 directly to the left.")

Q_START = (
    "This is a first-person video of someone walking across an open plain "
    "with several objects standing on the ground. Think back to the very "
    "FIRST frame of the video. Treat the direction the camera was facing AT "
    "THAT FIRST MOMENT as north (000 degrees). From the camera's STARTING "
    "position, what was the compass bearing to the {name}? Answer in degrees "
    "clockwise from that facing, 0-359: 090 is directly to the right, 180 "
    "directly behind, 270 directly to the left.")


def camera_path(rng, variance=0):
    """~10 s: walk, one eased rotation, walk. `variance` scales future
    versions (more rotations, curved legs); v0 keeps it gentle."""
    yaw = rng.uniform(0, 2 * math.pi)
    dyaw = math.radians(rng.uniform(40, 85)) * rng.choice((-1, 1))
    v, v_rot = 1.35, 0.3
    x = y = 0.0
    poses = []

    def step(speed, dy):
        nonlocal x, y, yaw
        yaw += dy
        x += math.cos(yaw) * speed / FPS
        y += math.sin(yaw) * speed / FPS
        poses.append((round(x, 4), round(y, 4), round(yaw, 5)))

    n1, nr, n2, nh = int(3.4 * FPS), int(1.9 * FPS), int(3.6 * FPS), int(0.8 * FPS)
    for _ in range(n1):
        step(v, 0.0)
    for i in range(nr):
        w = math.sin(math.pi * (i + 0.5) / nr)          # eased turn
        step(v_rot, dyaw * w / (nr * 2 / math.pi))
    for _ in range(n2):
        step(v, 0.0)
    for _ in range(nh):
        step(0.0, 0.0)
    return poses


def place_objects(rng, poses):
    pts = np.array([(p[0], p[1]) for p in poses[::15]])
    placed = []
    for name, shape, dims, color in OBJECTS:
        for _ in range(500):
            # anchor on a random point of the walk so objects line the whole
            # route, not just its centroid
            ax, ay = pts[rng.randrange(len(pts))]
            ang = rng.uniform(0, 2 * math.pi)
            rad = rng.uniform(3.5, 11.0)
            cx = ax + rad * math.cos(ang)
            cy = ay + rad * math.sin(ang)
            if np.min(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)) < 2.0:
                continue
            if any(math.hypot(cx - o["x"], cy - o["y"]) < 3.5 for o in placed):
                continue
            placed.append({"name": name, "shape": shape, "dims": dims,
                           "color": color, "x": round(cx, 3),
                           "y": round(cy, 3),
                           "yaw": round(rng.uniform(0, 6.28), 3)})
            break
        else:
            raise RuntimeError(f"no spot for {name}")
    return placed


def main():
    scenes, rows = [], []
    for sid in range(N_SCENES):
        rng = random.Random(SEED0 + sid)
        poses = camera_path(rng)
        objs = place_objects(rng, poses)
        # guarantee at least one object in view at the end, so the
        # perception-sanity row exists in every scene
        last = poses[-1]
        def off(o):
            b = G.bearing_deg((last[0], last[1], last[2]), (o["x"], o["y"]))
            return min(b, 360 - b)
        if min(off(o) for o in objs) > 28:
            o = objs[-1]
            b = math.radians(rng.uniform(-20, 20))
            d = rng.uniform(5.0, 10.0)
            yawf = last[2] + b
            o["x"] = round(last[0] + d * math.cos(yawf), 3)
            o["y"] = round(last[1] + d * math.sin(yawf), 3)
        scenes.append({"id": sid, "fps": FPS, "eye": EYE, "seed": SEED0 + sid,
                       "poses": poses, "objects": objs})
        first, last = poses[0], poses[-1]
        rng2 = random.Random(9500 + sid)
        for o in objs:                                   # final frame, all 5
            b = G.bearing_deg((last[0], last[1], last[2]), (o["x"], o["y"]))
            off = min(b, 360 - b)
            rows.append({
                "id": len(rows), "video": f"{sid:03d}.mp4", "scene": sid,
                "object": o["name"], "frame": "final",
                "level": "final_visible" if off <= 30 else "final_hidden",
                "question": Q_FINAL.format(name=o["name"]),
                "bearing_gt_deg": round(b, 2),
                "distance": round(math.hypot(o["x"] - last[0],
                                             o["y"] - last[1]), 2)})
        for o in rng2.sample(objs, 2):                   # start frame, two
            b = G.bearing_deg((first[0], first[1], first[2]),
                              (o["x"], o["y"]))
            rows.append({
                "id": len(rows), "video": f"{sid:03d}.mp4", "scene": sid,
                "object": o["name"], "frame": "start", "level": "start",
                "question": Q_START.format(name=o["name"]),
                "bearing_gt_deg": round(b, 2),
                "distance": round(math.hypot(o["x"] - first[0],
                                             o["y"] - first[1]), 2)})

    (ROOT / "photoreal/plane3_spec.json").write_text(json.dumps(scenes))
    gts = [r["bearing_gt_deg"] for r in rows]
    c = min(range(360), key=lambda k: sum(MP.circ_err(k, b) for b in gts))
    doc = {"note": "Three.js open-plain transfer probes; eval-only.",
           "const_baseline_deg": round(
               sum(MP.circ_err(c, b) for b in gts) / len(gts), 2),
           "const_baseline_answer": c, "plans": rows}
    (ROOT / "probes/plane3_probes.json").write_text(json.dumps(doc, indent=1))
    import collections
    print(f"{len(scenes)} scenes, {len(rows)} probes; levels",
          dict(collections.Counter(r['level'] for r in rows)),
          f"constant {doc['const_baseline_deg']}@{c}")


if __name__ == "__main__":
    main()
