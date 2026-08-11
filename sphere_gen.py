"""One-turn open-plane scenes with an unambiguous sphere target.

Narrowed on purpose. Every scene is a green plain with a noise ground (no
checkerboard — a checker is a metric ruler painted on the floor, and distance
walked can be counted off it instead of integrated), exactly one turn, and a
clutter of objects that changes scene to scene.

The target is always a sphere, and **no two spheres in a scene share a colour**,
so "the red sphere" names exactly one thing. Non-sphere objects may reuse
colours freely — they are distractors that make the colour alone insufficient.

Two questions only, both about where the target is relative to the camera:

    bearing_final   from the camera's pose in the LAST frame
    bearing_start   from its pose in the FIRST frame

    .venv/bin/python sphere_gen.py --split train --n 900 --out probes/sph_train.json
"""

import argparse
import json
import math
import random
from pathlib import Path

import generate as G
import make_probes as MP
import plane_gen as PG

ROOT = Path(__file__).resolve().parent
FPS = PG.FPS

COLOURS = {"red": "#d32f2f", "blue": "#1f5fd0", "green": "#2e9e46",
           "yellow": "#e6c018", "orange": "#e8761f", "purple": "#7a3fbf",
           "cyan": "#1fb6c9", "pink": "#e070a8", "white": "#eceae4",
           "black": "#2b2b2e", "brown": "#8a5a34", "grey": "#9aa0a6"}
DISTRACTOR_SHAPES = ["cube", "cylinder", "cone", "torus", "capsule",
                     "octahedron", "pyramid"]
BUILDING_COLS = ["#8d8377", "#9a9086", "#7d7469", "#a89c8c", "#6f6a63", "#bfae95"]

INTRO = [
    ("plain", "This is a first-person video of someone walking across an open "
              "green plain with objects scattered around."),
    ("terse", "A first-person video of a walk across open ground past several "
              "objects."),
    ("story", "Someone walks across an open field, filming as they go, passing "
              "various objects standing on the ground."),
]
Q_FINAL = [
    " One of the objects is a {name} — the only {colour} sphere in the scene. "
    "It comes into view during the walk, but by the final frame it may be out "
    "of shot. Freeze on the final frame and treat the direction the camera is "
    "facing at that moment as north (000 degrees). What is the compass bearing "
    "from the camera's final position to the {name}?",
    " Look for the {name}, the only {colour} sphere here. At the end of the "
    "video, taking the camera's own facing as north (000 degrees), which "
    "compass bearing is it in?",
]
Q_START = [
    " One of the objects is a {name} — the only {colour} sphere in the scene. "
    "Think back to the very first frame of the video and treat the direction "
    "the camera was facing at that moment as north (000 degrees). From the "
    "camera's STARTING position, what was the compass bearing to the {name}?",
    " Find the {name}, the only {colour} sphere here. Rewind to the opening "
    "frame; taking the camera's facing there as north (000 degrees), in which "
    "compass direction did it lie from where the walk began?",
]
DEG = (" Answer in degrees clockwise from that forward direction, 0-359: 090 "
       "is directly to the right, 180 directly behind, 270 directly to the "
       "left.\n\nAny integer from 0 to 359 is a valid answer — the object is "
       "usually not at an exact multiple of 90, so give your best estimate "
       "rather than rounding to the nearest quarter turn.\nEnd your reply "
       "with exactly one line, and nothing after it:\nANSWER: <degrees>")


def draw(rng, sid):
    fov = rng.uniform(58, 78)
    hfov = PG.half_fov(fov)
    speed = rng.uniform(1.0, 1.8)
    poses, turns = PG.camera_path(rng, 1, rng.uniform(6.0, 9.0), speed)
    pts = [(p[0], p[1]) for p in poses[::8]]
    objs = []

    def free(x, y, r=1.8):
        if min(math.hypot(px - x, py - y) for px, py in pts) < r:
            return False
        return all(math.hypot(x - o["x"], y - o["y"]) > 3.0 for o in objs)

    def spot(lo, hi):
        for _ in range(160):
            ax, ay = pts[rng.randrange(len(pts))]
            a, rad = rng.uniform(0, 6.283), rng.uniform(lo, hi)
            x, y = ax + rad * math.cos(a), ay + rad * math.sin(a)
            if free(x, y):
                return x, y
        return None

    # spheres: distinct colours, so naming a colour names one sphere
    sphere_cols = rng.sample(list(COLOURS), rng.randint(3, 5))
    for c in sphere_cols:
        p = spot(3.0, 14.0)
        if p:
            objs.append({"name": f"{c} sphere", "shape": "sphere",
                         "colour": COLOURS[c], "colour_name": c,
                         "x": round(p[0], 2), "y": round(p[1], 2),
                         "yaw": 0.0, "scale": round(rng.uniform(0.9, 1.6), 2)})
    # distractors: any shape, any colour — colour alone is not enough
    for _ in range(rng.randint(6, 12)):
        p = spot(2.5, 16.0)
        if p:
            c = rng.choice(list(COLOURS))
            objs.append({"name": f"{c} {rng.choice(DISTRACTOR_SHAPES)}",
                         "shape": rng.choice(DISTRACTOR_SHAPES),
                         "colour": COLOURS[c], "colour_name": c,
                         "x": round(p[0], 2), "y": round(p[1], 2),
                         "yaw": round(rng.uniform(0, 6.28), 3),
                         "scale": round(rng.uniform(0.6, 1.5), 2)})
    for _ in range(rng.randint(3, 7)):                       # distant skyline
        p = spot(26.0, 70.0)
        if p:
            w, d = rng.uniform(5, 13), rng.uniform(5, 13)
            objs.append({"name": "building", "shape": "building",
                         "colour": rng.choice(BUILDING_COLS),
                         "colour_name": "grey",
                         "x": round(p[0], 2), "y": round(p[1], 2),
                         "yaw": round(rng.uniform(0, 6.28), 3), "scale": 1.0,
                         "dims": [round(w, 1), round(rng.uniform(7, 30), 1),
                                  round(d, 1)]})

    objs = [PG.annotate(o, poses, hfov) for o in objs]
    cands = [o for o in objs if o["shape"] == "sphere" and o["seen_early"]]
    if not cands:
        return None
    return {"id": sid, "level": "S1", "n_turns": 1, "turns": turns, "fps": FPS,
            "secs": round(len(poses) / FPS, 2),
            "eye": round(rng.uniform(1.4, 1.8), 2), "fov": round(fov, 1),
            "hfov": round(hfov, 1), "speed": round(speed, 2),
            "ground_style": "noise",
            "ground_hue": round(rng.uniform(0.24, 0.35), 3),   # greens
            "ground_scale": rng.randint(70, 180),
            "sky_hue": round(rng.uniform(0.53, 0.63), 3),
            "sun_az": round(rng.uniform(0, 6.28), 2),
            "sun_el": round(rng.uniform(0.55, 1.15), 2),
            "poses": poses, "objects": objs,
            "targets": [o["name"] for o in cands]}


def rows(scene, rng, video_dir, for_probe):
    poses = scene["poses"]
    first, last = poses[0], poses[-1]
    out = []
    names = list(scene["targets"])
    rng.shuffle(names)
    for kind, names_k in (("bearing_final", names[:2]),
                          ("bearing_start", names[:1])):
        for nm in names_k:
            o = next(x for x in scene["objects"] if x["name"] == nm)
            if kind == "bearing_final":
                b = o["bearing_final"]
                tmpl = rng.choice(Q_FINAL)
            else:
                b = G.bearing_deg((first[0], first[1], first[2]),
                                  (o["x"], o["y"]))
                tmpl = rng.choice(Q_START)
            tag, intro = rng.choice(INTRO)
            q = intro + tmpl.format(name=nm, colour=o["colour_name"]) + DEG
            out.append({
                "video": (f"{scene['id']:04d}.mp4" if for_probe
                          else str(ROOT / video_dir / f"{scene['id']:04d}.mp4")),
                "scene": scene["id"], "task": kind, "level": kind,
                "paraphrase": tag, "object": nm, "question": q, "think": "",
                "answer": f"ANSWER: {round(b) % 360:03d}",
                "bearing_gt_deg": round(b, 2),
                "visible_at_end": o["visible_at_end"],
                "distance": o["distance_end"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "test"), required=True)
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--seed", type=int, default=555)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    scenes = []
    while len(scenes) < args.n:
        s = draw(rng, args.seed0 + len(scenes))
        if s:
            scenes.append(s)
    (ROOT / args.out).write_text(json.dumps({"note": f"sphere {args.split}",
                                             "scenes": scenes}))
    f = sum(len(s["poses"]) for s in scenes)
    ob = sum(len(s["objects"]) for s in scenes) / len(scenes)
    print(f"{len(scenes)} scenes -> {args.out}")
    print(f"  {f} frames ({f/FPS/60:.0f} min), {ob:.0f} objects/scene, "
          f"{sum(len(s['targets']) for s in scenes)/len(scenes):.1f} usable "
          f"sphere targets/scene")


if __name__ == "__main__":
    main()
