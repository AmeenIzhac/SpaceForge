"""Dense open-plane scenes: buildings, mid-size props and small clutter.

Same spec format as plane_gen.py (so plane_gl.py renders it and the bearing
ground truth is computed the same way), but the object list is far richer:
a few large buildings placed out at distance, a middle band of props, and a
scatter of small objects near the path.

    .venv/bin/python plane_city.py --n 6 --out probes/plane_city.json
"""
import argparse, json, math, random
from pathlib import Path
import plane_gen as PG

GROUND_STYLE = None        # set by --ground

BUILDING_COLS = ["#8d8377", "#9a9086", "#7d7469", "#a89c8c", "#6f6a63",
                 "#b0a header"]
BUILDING_COLS = ["#8d8377", "#9a9086", "#7d7469", "#a89c8c", "#6f6a63", "#b3a housing"]
BUILDING_COLS = ["#8d8377", "#9a9086", "#7d7469", "#a89c8c", "#6f6a63", "#bfae95"]
SMALL_COLS = list(PG.COLOURS.values())

def scene(rng, sid, n_turns=2, secs=11.0):
    speed = rng.uniform(1.2, 1.9)
    poses, turns = PG.camera_path(rng, n_turns, secs, speed)
    pts = [(p[0], p[1]) for p in poses[::10]]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    objs = []

    def free(x, y, r):
        if min(math.hypot(px - x, py - y) for px, py in pts) < r + 1.6:
            return False
        return all(math.hypot(x - o["x"], y - o["y"]) >
                   r + o.get("_r", 1.0) + 1.0 for o in objs)

    def add(shape, colour, x, y, dims=None, scale=1.0, r=1.0):
        objs.append({"name": f"obj{len(objs)}", "shape": shape, "colour": colour,
                     "colour_name": "grey", "x": round(x, 2), "y": round(y, 2),
                     "yaw": round(rng.uniform(0, 6.28), 3), "scale": scale,
                     **({"dims": dims} if dims else {}), "_r": r,
                     "seen_frames": 0, "first_seen": None, "seen_early": False,
                     "bearing_final": 0.0, "visible_at_end": False,
                     "distance_end": 0.0})

    for _ in range(rng.randint(10, 16)):            # buildings, further out
        for _try in range(200):
            a, rad = rng.uniform(0, 6.283), rng.uniform(22, 75)
            x, y = cx + rad * math.cos(a), cy + rad * math.sin(a)
            w, d = rng.uniform(5, 14), rng.uniform(5, 14)
            h = rng.uniform(7, 34)
            if free(x, y, max(w, d) / 2):
                add("building", rng.choice(BUILDING_COLS), x, y,
                    dims=[w, h, d], r=max(w, d) / 2)
                break
    for _ in range(rng.randint(10, 16)):            # mid-size props
        for _try in range(200):
            a, rad = rng.uniform(0, 6.283), rng.uniform(8, 30)
            x, y = cx + rad * math.cos(a), cy + rad * math.sin(a)
            if free(x, y, 1.6):
                sh = rng.choice(["cylinder", "cone", "cube"])
                add(sh, rng.choice(SMALL_COLS), x, y,
                    dims=[rng.uniform(1.4, 3.2), rng.uniform(2.5, 7.0),
                          rng.uniform(1.4, 3.2)], r=1.8)
                break
    for _ in range(rng.randint(22, 34)):            # small clutter near the path
        for _try in range(200):
            ax, ay = pts[rng.randrange(len(pts))]
            a, rad = rng.uniform(0, 6.283), rng.uniform(2.6, 14)
            x, y = ax + rad * math.cos(a), ay + rad * math.sin(a)
            if free(x, y, 0.7):
                add(rng.choice(["sphere", "cube", "cone", "torus", "capsule",
                                "octahedron", "pyramid"]),
                    rng.choice(SMALL_COLS), x, y,
                    scale=rng.uniform(0.5, 1.4), r=0.9)
                break
    for o in objs:
        o.pop("_r", None)
    return {"id": sid, "level": f"city{n_turns}", "n_turns": n_turns,
            "turns": turns, "fps": PG.FPS,
            "secs": round(len(poses) / PG.FPS, 2),
            "eye": round(rng.uniform(1.5, 1.8), 2),
            "fov": round(rng.uniform(62, 76), 1), "speed": round(speed, 2),
            "ground_hue": round(rng.uniform(0, 1), 3),
            "ground_scale": rng.randint(70, 180),
            "ground_style": GROUND_STYLE or rng.choice(
                ["noise", "patchy", "plain", "checker"]),
            "sky_hue": round(rng.uniform(0.5, 0.68), 3),
            "sun_az": round(rng.uniform(0, 6.28), 2),
            "sun_el": round(rng.uniform(0.55, 1.15), 2),
            "poses": poses, "objects": objs}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", default="probes/plane_city.json")
    ap.add_argument("--ground", default=None,
                    choices=("checker", "plain", "noise", "patchy"),
                    help="fix the ground style (default: random per scene)")
    args = ap.parse_args()
    global GROUND_STYLE
    GROUND_STYLE = args.ground
    rng = random.Random(args.seed)
    scenes = [scene(rng, 7000 + i, rng.choice([1, 2, 3]), rng.uniform(9, 13))
              for i in range(args.n)]
    Path(args.out).write_text(json.dumps({"note": "dense city-ish scenes",
                                          "scenes": scenes}))
    n = sum(len(s["objects"]) for s in scenes)
    f = sum(len(s["poses"]) for s in scenes)
    print(f"{len(scenes)} scenes -> {args.out}")
    print(f"  {n} objects total ({n/len(scenes):.0f} per scene), {f} frames")

if __name__ == "__main__":
    main()
