"""Open-arena transfer probes: a different world, the same question.

Every corridor crutch is removed at once: no corridors, no red X, camera
rotation decoupled from any junction, several objects instead of one target.
A big open hall (26x26 world units) holds five distinct objects; the camera
walks a gentle two-leg path with one free rotation in the middle, ~10 s. The
questions ask for compass bearings from the FINAL pose (some objects in view,
some behind) and from the STARTING pose — all in the corridor task's own
answer format, so the trained skill either transfers or it does not.

Ground truth is the same G.bearing_deg over hand-built poses; nothing is
learned from these scenes, they are eval-only.

    .venv/bin/python plane_probe.py --render          # -> out/plane + probes
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np

import generate as G
import make_probes as MP
import props
import textures
import world

ROOT = Path(__file__).resolve().parent
ARENA = 26                     # world units per side
EYE_FOV = 68.0
FPS = 30
N_SCENES = 12
SEED0 = 770_000

OBJECTS = [
    ("red cube",      lambda: props.box(0.9, 0.9, 0.9, (214, 40, 36),
                                        emissive=0.15)),
    ("blue ball",     lambda: props.blob(0.55, (40, 90, 220), 0, 0.55, 0,
                                         emissive=0.15)),
    ("green pillar",  lambda: props.cylinder(0.34, 0.30, 1.8, (30, 150, 60),
                                             emissive=0.15)),
    ("yellow crate",  lambda: props.box(1.25, 0.75, 0.85, (232, 178, 22),
                                        emissive=0.15)),
    ("orange barrel", lambda: props.cylinder(0.45, 0.45, 1.1, (236, 118, 28),
                                             emissive=0.15)),
]

Q_FINAL = (
    "This is a first-person video of someone walking across a large open "
    "hall with several objects standing on the floor. Freeze on the FINAL "
    "frame and treat the direction the camera is facing at that moment as "
    "north (000 degrees). What is the compass bearing from the camera's "
    "final position to the {name}? Answer in degrees clockwise from that "
    "forward direction, 0-359: 090 is directly to the right, 180 directly "
    "behind, 270 directly to the left.")

Q_START = (
    "This is a first-person video of someone walking across a large open "
    "hall with several objects standing on the floor. Think back to the "
    "very FIRST frame of the video. Treat the direction the camera was "
    "facing AT THAT FIRST MOMENT as north (000 degrees). From the camera's "
    "STARTING position, what was the compass bearing to the {name}? Answer "
    "in degrees clockwise from that facing, 0-359: 090 is directly to the "
    "right, 180 directly behind, 270 directly to the left.")


def camera_path(rng):
    """(x, y, yaw, s) at 30 fps: leg, free rotation, leg — no corridors."""
    cx = cy = ARENA / 2.0
    yaw0 = rng.uniform(0, 2 * math.pi)
    dyaw = math.radians(rng.uniform(38, 85)) * rng.choice((-1, 1))
    v, v_rot = 1.1, 0.25
    t_leg1, t_rot, t_leg2, t_hold = 3.4, 1.8, 3.6, 0.8

    x = cx - math.cos(yaw0) * 3.8
    y = cy - math.sin(yaw0) * 3.8
    poses, s, yaw = [], 0.0, yaw0
    bob = lambda: None

    def step(dt, speed, dy):
        nonlocal x, y, yaw, s
        yaw += dy
        x += math.cos(yaw) * speed * dt
        y += math.sin(yaw) * speed * dt
        s += speed * dt
        poses.append((x, y, yaw, s))

    n1, nr, n2, nh = (int(t * FPS) for t in (t_leg1, t_rot, t_leg2, t_hold))
    for _ in range(n1):
        step(1 / FPS, v, 0.0)
    for i in range(nr):
        # ease the rotation in and out, like a person turning to look
        w = math.sin(math.pi * (i + 0.5) / nr)
        step(1 / FPS, v_rot, dyaw * w / (nr * 0.6366))
    for _ in range(n2):
        step(1 / FPS, v, 0.0)
    for _ in range(nh):
        step(1 / FPS, 0.0, 0.0)
    return poses


def place_objects(rng, poses):
    """Five objects, spread in angle around the path, clear of it."""
    pts = np.array([(p[0], p[1]) for p in poses[:: FPS // 2]])
    placed = []
    for name, _ in OBJECTS:
        for _try in range(400):
            ox = rng.uniform(3.0, ARENA - 3.0)
            oy = rng.uniform(3.0, ARENA - 3.0)
            if np.min(np.hypot(pts[:, 0] - ox, pts[:, 1] - oy)) < 1.6:
                continue
            if any(math.hypot(ox - a, oy - b) < 3.0 for a, b, *_ in placed):
                continue
            placed.append((ox, oy, name))
            break
        else:
            raise RuntimeError("could not place " + name)
    return placed


def in_view(pose, pt, fov_deg=EYE_FOV):
    b = G.bearing_deg(pose, pt)
    off = min(b, 360 - b)
    return off <= fov_deg / 2 * 0.9


def render_scene(sid, out_dir):
    import imageio.v2 as imageio
    import gl_renderer

    rng = random.Random(SEED0 + sid)
    nrng = np.random.default_rng(SEED0 + sid)

    poses = camera_path(rng)
    objs = place_objects(rng, poses)

    open_ = np.zeros((ARENA + 4, ARENA + 4), bool)
    open_[2:-2, 2:-2] = True
    pal = textures.build_palette(G.MOOD, nrng, n_walls=G.N_WALL_VARIANTS,
                                 baked_lights=False)
    atlas = pal["walls"] + [pal["door"]]
    wmap = world.wmap_from_open(open_, n_variants=len(pal["walls"]),
                                door_id=len(atlas), salt=SEED0 + sid,
                                door_prob=0.0)

    mesh = None
    for (ox, oy, name), (_n, builder) in zip(objs, OBJECTS):
        m = props.transform(builder(), yaw=rng.uniform(0, 6.28),
                            dx=ox + 2, dz=oy + 2)     # +2: arena border pad
        mesh = m if mesh is None else props.merge(mesh, m)

    light_cells = [(cx, cy) for cx in range(4, ARENA + 2, 5)
                   for cy in range(4, ARENA + 2, 5)]
    shadows = [(ox + 2, oy + 2, 0.85) for ox, oy, _ in objs]
    rend = gl_renderer.GLRenderer(
        wmap, atlas, pal["floor"], pal["ceil"], pal["floor_spec"], pal,
        mesh, shadows, [], width=G.WIDTH, height=G.HEIGHT, fov_deg=EYE_FOV,
        wall_h=G.WALL_H, eye=G.EYE_H, bob_amp=0.006, stride=0.7,
        light_cells=light_cells)

    out = out_dir / f"{sid:03d}.mp4"
    writer = imageio.get_writer(str(out), fps=FPS, codec="libx264",
                                quality=8, pixelformat="yuv420p")
    try:
        for (x, y, yaw, s) in poses:
            writer.append_data(rend.render(x + 2, y + 2, yaw, s))
    finally:
        writer.close()
    rend.release()
    return poses, objs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--out", default="out/plane")
    ap.add_argument("--gpu", default="2")
    args = ap.parse_args()

    os.environ.setdefault("EGL_DEVICE_ID", args.gpu)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sid in range(N_SCENES):
        poses, objs = render_scene(sid, out_dir) if args.render else (None, None)
        if poses is None:
            rng = random.Random(SEED0 + sid)
            poses = camera_path(rng)
            objs = place_objects(rng, poses)
        first, last = poses[0], poses[-1]
        rng2 = random.Random(9000 + sid)
        finals = list(objs)                      # every object, final frame
        starts = rng2.sample(objs, 2)
        for ox, oy, name in finals:
            vis = in_view(last, (ox, oy))
            rows.append({
                "id": len(rows), "video": f"{sid:03d}.mp4", "scene": sid,
                "object": name, "frame": "final",
                "level": "final_visible" if vis else "final_hidden",
                "question": Q_FINAL.format(name=name),
                "bearing_gt_deg": round(G.bearing_deg(last, (ox, oy)), 2),
                "distance": round(math.hypot(ox - last[0], oy - last[1]), 2),
            })
        for ox, oy, name in starts:
            rows.append({
                "id": len(rows), "video": f"{sid:03d}.mp4", "scene": sid,
                "object": name, "frame": "start", "level": "start",
                "question": Q_START.format(name=name),
                "bearing_gt_deg": round(G.bearing_deg(first, (ox, oy)), 2),
                "distance": round(math.hypot(ox - first[0], oy - first[1]), 2),
            })

    gts = [r["bearing_gt_deg"] for r in rows]
    c = min(range(360), key=lambda k: sum(MP.circ_err(k, b) for b in gts))
    doc = {"note": "open-arena transfer probes; eval-only",
           "const_baseline_deg": round(
               sum(MP.circ_err(c, b) for b in gts) / len(gts), 2),
           "const_baseline_answer": c,
           "plans": rows}
    (ROOT / "probes/plane_probes.json").write_text(json.dumps(doc, indent=1))
    import collections
    lv = collections.Counter(r["level"] for r in rows)
    print(f"{len(rows)} probes over {N_SCENES} scenes -> probes/plane_probes.json")
    print("  levels:", dict(lv), " best constant:",
          doc["const_baseline_deg"], "@", c)


if __name__ == "__main__":
    main()
