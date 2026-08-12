"""Walk the sphere task into the corridor task one change at a time.

The sphere model is excellent on its own world and scores nothing on corridors.
Between those two points sit a dozen differences at once — target shape, where
the target sits, the wording, the camera motion, the path geometry, the
clutter, walls, a ceiling, the renderer itself. A single before/after cannot
say which of them costs the ability.

So: a chain of test sets. Each step is the previous step plus **exactly one**
change, and each is scored against its own best-constant baseline. Wherever the
error jumps, that is the change that breaks it.

    M0  natural sphere task, as trained
    M1  + target moved to the start of the walk (still a sphere, ~3.5 ahead)
    M2  + target becomes a red X painted flat on the ground
    M3  + the corridor task's wording
    M4  + rigid camera: constant speed, no sway, pitch, roll or bob
    M5  + lattice geometry: straight legs, exact 90 degree turns
    M6  + the distractor objects are removed
    M7  + walls along the route
    M8  + a ceiling and indoor light (fully enclosed)
    M9  the real corridor task (different renderer, textures, props, scale)

M9 is the existing corridor test set, not generated here.

    .venv/bin/python morph_ablation.py --n 70
"""

import argparse
import json
import math
import random
from pathlib import Path

import generate as G
import make_probes as MP
import natural_motion as NM
import plane_gen as PG
import sphere_gen as SG

ROOT = Path(__file__).resolve().parent
FPS = PG.FPS

HALF_W = 1.6            # corridor half width, world units
WALL_H = 2.7
WALL_T = 0.35
WALL_COL = "#b9b0a2"
CEIL_COL = "#d8d4cc"

STEPS = ["M0_sphere_natural", "M1_target_at_start", "M2_target_is_floor_X",
         "M3_corridor_wording", "M4_rigid_motion", "M5_lattice_path",
         "M6_no_distractors", "M7_walls", "M8_ceiling_indoor"]


def cfg_for(step):
    """Cumulative: every flag stays on once its step is reached."""
    i = STEPS.index(step)
    return {"at_start": i >= 1, "floor_x": i >= 2, "corridor_words": i >= 3,
            "rigid": i >= 4, "lattice": i >= 5, "bare": i >= 6,
            "walls": i >= 7, "ceiling": i >= 8}


# --------------------------------------------------------------- camera paths
def rigid_path(rng, secs, lattice):
    """Constant speed, discrete turns, nothing else moving. With `lattice` the
    turns are exactly 90 degrees, so legs run on a grid like a corridor."""
    speed = rng.uniform(1.1, 1.5)
    n_turns = rng.randint(1, 2)
    turn_t = 1.1
    leg = max(1.2, (secs - n_turns * turn_t - 0.6) / (n_turns + 1))
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
        for _ in range(int(leg * FPS)):
            step(speed, 0.0)
        if t < n_turns:
            deg = (90.0 if lattice else rng.uniform(55, 115)) * rng.choice((-1, 1))
            w = int(turn_t * FPS)
            turns.append({"start_frame": len(poses),
                          "dir": "R" if deg > 0 else "L", "deg": deg})
            for i in range(w):
                ease = math.sin(math.pi * (i + 0.5) / w)
                step(speed * 0.3, math.radians(deg) * ease / (w * 2 / math.pi))
    for _ in range(int(0.6 * FPS)):
        step(0.0, 0.0)
    return poses, turns


def legs_of(poses, tol=1e-3):
    """Straight runs of the path, as ((ax, ay), (bx, by)) — what to wall."""
    out, start = [], 0
    for i in range(1, len(poses) - 1):
        if abs(((poses[i + 1][2] - poses[i][2]) + math.pi) % (2 * math.pi)
               - math.pi) > tol:
            if i - start > 3:
                out.append((poses[start][:2], poses[i][:2]))
            start = i + 1
    if len(poses) - 1 - start > 3:
        out.append((poses[start][:2], poses[-1][:2]))
    return out


CELL = 0.7


def wall_objects(poses):
    """Wall blocks on a grid around the walked corridor. Cells whose centre is
    within HALF_W of the path are walkable; the shell just outside becomes
    wall. This closes corners and dead-ends automatically and, unlike per-leg
    slabs, can never place a wall where the camera goes."""
    import numpy as np
    pts = np.array([(p[0], p[1]) for p in poses], "f8")
    lo = pts.min(0) - (HALF_W + 3.2)
    hi = pts.max(0) + (HALF_W + 3.2)
    gx = np.arange(lo[0], hi[0] + CELL, CELL)
    gy = np.arange(lo[1], hi[1] + CELL, CELL)
    gxx, gyy = np.meshgrid(gx, gy, indexing="ij")
    flat = np.stack([gxx.ravel(), gyy.ravel()], 1)
    # min distance from each grid centre to any pose on the path
    d = np.min(np.hypot(flat[:, None, 0] - pts[None, :, 0],
                        flat[:, None, 1] - pts[None, :, 1]), axis=1)
    keep = (d >= HALF_W) & (d < HALF_W + 2.8)   # >=4 cells deep, no gaps
    cells = [[round(float(a), 2), round(float(b), 2)] for a, b in flat[keep]]
    return [{"name": "walls", "shape": "wallgrid", "colour": WALL_COL,
             "colour_name": "grey", "x": 0.0, "y": 0.0, "yaw": 0.0,
             "scale": 1.0, "cells": cells, "cell": CELL, "h": WALL_H,
             "no_shadow": True}]


def ceiling_object(poses):
    xs = [p[0] for p in poses]
    ys = [p[1] for p in poses]
    return {"name": "ceiling", "shape": "building", "colour": CEIL_COL,
            "colour_name": "grey",
            "x": round((min(xs) + max(xs)) / 2, 2),
            "y": round((min(ys) + max(ys)) / 2, 2), "yaw": 0.0, "scale": 1.0,
            "y_off": WALL_H + 0.1, "no_shadow": True,
            "dims": [max(xs) - min(xs) + 8 * HALF_W, 0.2,
                     max(ys) - min(ys) + 8 * HALF_W]}


# ------------------------------------------------------------------ questions
CORRIDOR_Q = (
    "This is a first-person video of someone walking. A large red X is painted "
    "on the floor at the spot where the walk begins; it is visible in the "
    "opening seconds, and the walker then leaves it behind and keeps going. By "
    "the final frame the X is out of shot — it is somewhere off camera, and you "
    "have to work out where from the route that was walked. Freeze on the final "
    "frame and treat the direction the camera is facing at that moment as north "
    "(000 degrees). What is the compass bearing from the camera's final "
    "position back to the red X?")


# once the target is a floor marker the sphere wording is nonsense ("the red X,
# the only red sphere here"), which measures a broken prompt rather than the
# change under test. Same shape of sentence, correct noun.
Q_FINAL_X = [
    " One of the things here is a {name} — a marking painted flat on the "
    "ground, the only one of its kind in the scene. It comes into view during "
    "the walk, but by the final frame it may be out of shot. Freeze on the "
    "final frame and treat the direction the camera is facing at that moment "
    "as north (000 degrees). What is the compass bearing from the camera's "
    "final position to the {name}?",
    " Look for the {name}, a marking painted flat on the ground — the only "
    "one here. At the end of the video, taking the camera's own facing as "
    "north (000 degrees), which compass bearing is it in?",
]


def make_rows(scene, cfg, rng, vids):
    poses = scene["poses"]
    first, last = poses[0], poses[-1]
    tgt = next(o for o in scene["objects"] if o.get("is_target"))
    rows = []
    for kind in ("bearing_final", "bearing_start"):
        if kind == "bearing_final":
            b = G.bearing_deg((last[0], last[1], last[2]), (tgt["x"], tgt["y"]))
        else:
            b = G.bearing_deg((first[0], first[1], first[2]),
                              (tgt["x"], tgt["y"]))
        if cfg["corridor_words"]:
            if kind == "bearing_start":
                continue                       # the corridor task has no such question
            q = CORRIDOR_Q + SG.DEG
        else:
            tag, intro = rng.choice(SG.INTRO)
            if cfg["floor_x"] and kind == "bearing_final":
                tmpl = rng.choice(Q_FINAL_X)
            else:
                tmpl = rng.choice(SG.Q_FINAL if kind == "bearing_final"
                                  else SG.Q_START)
            q = intro + tmpl.format(name=tgt["name"],
                                    colour=tgt.get("colour_name", "red")) + SG.DEG
        rows.append({"video": f"{scene['id']:04d}.mp4", "scene": scene["id"],
                     "task": kind, "level": scene["level"], "object": tgt["name"],
                     "question": q, "think": "",
                     "answer": f"ANSWER: {round(b) % 360:03d}",
                     "bearing_gt_deg": round(b, 2)})
    return rows


# ---------------------------------------------------------------------- build
def build_scene(step, cfg, rng, sid):
    secs = rng.uniform(9.0, 11.0)
    if cfg["rigid"]:
        poses3, turns = rigid_path(rng, secs, cfg["lattice"])
        poses = poses3
    else:
        poses, turns = NM.natural_path(rng, secs)
        poses3 = [(p[0], p[1], p[2]) for p in poses]

    saved = PG.camera_path
    PG.camera_path = lambda *a, **k: (poses3, turns)
    try:
        s = SG.draw(rng, sid)
    finally:
        PG.camera_path = saved
    if s is None:
        return None
    s["poses"] = poses
    s["turns"], s["n_turns"], s["level"] = turns, len(turns), step
    s["secs"] = round(len(poses) / FPS, 2)

    objs = s["objects"]
    if cfg["at_start"]:
        # target sits just ahead of where the walk began, like the corridor X
        x0, y0, yaw0 = poses3[0]
        d = rng.uniform(3.0, 4.2)
        tx, ty = x0 + math.cos(yaw0) * d, y0 + math.sin(yaw0) * d
        objs = [o for o in objs if o["shape"] != "sphere"]
        tgt = {"name": "red sphere", "shape": "sphere", "colour": "#d32f2f",
               "colour_name": "red", "x": round(tx, 3), "y": round(ty, 3),
               "yaw": 0.0, "scale": 1.3, "is_target": True}
        objs = [tgt] + objs
    else:
        tgt = next(o for o in objs if o["name"] == s["targets"][0])
        tgt["is_target"] = True

    if cfg["floor_x"]:
        tgt.update({"shape": "xmark", "scale": 2.0, "colour": "#d32f2f",
                    "colour_name": "red", "name": "red X"})
    if cfg["bare"]:
        objs = [o for o in objs if o.get("is_target")]
    if cfg["walls"]:
        objs = objs + wall_objects(poses3)
    if cfg["ceiling"]:
        objs = objs + [ceiling_object(poses3)]
        s["indoor"] = True
        s["sky_hue"] = 0.58
    s["objects"] = objs
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=70)
    ap.add_argument("--seed", type=int, default=4711)
    args = ap.parse_args()

    sid = 300000
    for step in STEPS:
        cfg = cfg_for(step)
        rng = random.Random(args.seed + STEPS.index(step) * 1000)
        scenes = []
        while len(scenes) < args.n:
            s = build_scene(step, cfg, rng, sid)
            if s:
                scenes.append(s)
                sid += 1
        (ROOT / f"probes/mo_{step}.json").write_text(
            json.dumps({"note": step, "scenes": scenes}))
        print(f"{step:22s} {len(scenes)} scenes, "
              f"{sum(len(x['objects']) for x in scenes)//len(scenes)} objects/scene")


if __name__ == "__main__":
    main()
