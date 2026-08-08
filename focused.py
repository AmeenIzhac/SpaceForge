"""Focused samples: bare corridors with hand-authored turn sequences.

No rooms, no doors, no diversions, and no mid-walk yaws — the camera only
turns where the path turns. Each straight leg gets its own wall colour as an
orientation cue. Junction variants add crossroads at every bend, so each
turn visibly could have gone left, right, or straight (dead-end stubs carve
the unchosen directions).

Outputs are hardcoded: out/focused/<n>.mp4
"""

import random
from pathlib import Path

import numpy as np
import imageio.v2 as imageio

import decor
import floorplan
import generate as G
import gl_renderer
import pathgen
import props
import textures
import world

OUT_DIR = Path(__file__).resolve().parent / "out" / "focused"
SEED = 11

# (out name, turn sequence, lattice leg lengths, junctions at bends, mood)
SAMPLES = [
    ("0.mp4", "LRL",   (2, 2, 2, 2),       False, "office"),
    ("1.mp4", "LRRL",  (2, 2, 2, 2, 2),    False, "hotel"),
    ("2.mp4", "LRLLR", (2, 2, 2, 2, 2, 2), False, "lab"),
    ("3.mp4", "RLR",   (2, 2, 2, 2),       True,  "brick"),
    ("4.mp4", "LRLR",  (2, 2, 2, 2, 2),    True,  "office"),
    ("5.mp4", "RLRRL", (2, 2, 2, 2, 3, 2), True,  "hotel"),
]

TURN_NAME = {"L": "left", "R": "right"}


def build_route(turns, legs):
    """Lattice route from a turn string. Starts heading east; y grows
    downward, so left of (dx, dy) is (dy, -dx) and right is (-dy, dx)."""
    d = (1, 0)
    pos = (0, 0)
    route = [pos]
    for leg, t in zip(legs, list(turns) + [None]):
        for _ in range(leg):
            pos = (pos[0] + d[0], pos[1] + d[1])
            route.append(pos)
        if t == "L":
            d = (d[1], -d[0])
        elif t == "R":
            d = (-d[1], d[0])
    return route


def leg_spans(route):
    """(start_idx, end_idx) per straight leg, split at bends."""
    legs, start = [], 0
    for i in range(1, len(route) - 1):
        d1 = (route[i][0] - route[i - 1][0], route[i][1] - route[i - 1][1])
        d2 = (route[i + 1][0] - route[i][0], route[i + 1][1] - route[i][1])
        if d1 != d2:
            legs.append((start, i))
            start = i
    legs.append((start, len(route) - 1))
    return legs


def junction_stubs(route):
    """At each bend: 1-step dead-end stubs in the two directions not taken
    (straight ahead + the other turn), making the bend a full crossroads.
    Returns (bend_index_in_route, stub_edge) pairs."""
    stubs = []
    for i in range(1, len(route) - 1):
        d1 = (route[i][0] - route[i - 1][0], route[i][1] - route[i - 1][1])
        d2 = (route[i + 1][0] - route[i][0], route[i + 1][1] - route[i][1])
        if d1 == d2:
            continue
        for d in (d1, (d1[1], -d1[0]), (-d1[1], d1[0])):
            if d != d2:
                stubs.append((i, (route[i],
                                  (route[i][0] + d[0], route[i][1] + d[1]))))
    return stubs


def make_sample(name, turns, legs, junctions, mood):
    seed = SEED + int(name.split(".")[0])
    prng = random.Random(seed)
    nrng = np.random.default_rng(seed)

    route = build_route(turns, legs)
    assert len(set(route)) == len(route), "route revisits a node"
    got = [t["turn"] for t in pathgen.route_turns(route)]
    want = [TURN_NAME[t] for t in turns]
    assert got == want, (got, want)

    stubs = junction_stubs(route) if junctions else []
    all_nodes = [c for c in route] + [e[1][1] for e in stubs]
    x0 = min(c[0] for c in all_nodes)
    y0 = min(c[1] for c in all_nodes)
    route = [(x - x0, y - y0) for x, y in route]
    stubs = [(i, ((a[0] - x0, a[1] - y0), (b[0] - x0, b[1] - y0)))
             for i, (a, b) in stubs]

    s, off = floorplan.S, floorplan.PAD
    size = (max(max(c) for c in route + [e[1] for _, e in stubs]) * s
            + 2 * off + 1)
    open_ = np.zeros((size, size), bool)
    light_cells = []
    edges = list(zip(route[:-1], route[1:])) + [e for _, e in stubs]
    for a, b in edges:
        dx, dy = b[0] - a[0], b[1] - a[1]
        for t in range(s + 1):
            open_[off + a[1] * s + t * dy, off + a[0] * s + t * dx] = True
        light_cells.append((off + a[0] * s + dx * (s // 2),
                            off + a[1] * s + dy * (s // 2)))

    # one wall colour per straight leg: every wall cell next to a leg's
    # corridor takes that leg's variant. Stubs painted first with the
    # incoming leg's colour (a straight stub reads as the same corridor
    # continuing, then dead-ending); legs after, so route walls stay clean.
    legs_idx = leg_spans(route)
    variant_map = np.ones((size, size), np.int16)

    def paint_edge(a, b, v):
        dx, dy = b[0] - a[0], b[1] - a[1]
        for t in range(s + 1):
            cx = off + a[0] * s + t * dx
            cy = off + a[1] * s + t * dy
            variant_map[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2] = v

    leg_of_bend = {i1: li for li, (i0, i1) in enumerate(legs_idx)}
    for bend_i, (a, b) in stubs:
        paint_edge(a, b, (2 * leg_of_bend[bend_i]) % 8 + 1)
    for li, (i0, i1) in enumerate(legs_idx):
        for k in range(i0, i1):
            paint_edge(route[k], route[k + 1], (2 * li) % 8 + 1)

    pal = textures.build_palette(mood, nrng, n_walls=8, baked_lights=False)
    atlas = pal["walls"] + [pal["door"]]
    wmap = world.wmap_from_open(open_, n_variants=len(pal["walls"]),
                                door_id=len(atlas), salt=seed,
                                door_prob=0.0,        # no doors — no rooms behind
                                variant_map=variant_map)

    exclude = np.zeros_like(open_)
    lms = world.place_landmarks_wmap(wmap, exclude, prng, len(props.PROP_ORDER),
                                     density=G.LANDMARK_DENSITY,
                                     min_gap=G.LANDMARK_GAP)
    wall_decor = decor.place_wall_decor(wmap, door_id=len(atlas), rng=prng,
                                        density=G.DECOR_DENSITY,
                                        min_gap=G.DECOR_GAP)

    mesh, shadows, plights = props.build_props(lms, prng)
    wall_mesh, wall_lights = decor.build_wall_meshes(wall_decor, prng)
    mesh = props.merge(mesh, wall_mesh)
    decal_mesh, decal_texs = decor.build_decals(wall_decor, nrng)

    rend = gl_renderer.GLRenderer(
        wmap, atlas, pal["floor"], pal["ceil"], pal["floor_spec"], pal,
        mesh, shadows, plights + wall_lights,
        width=G.WIDTH, height=G.HEIGHT, fov_deg=G.FOV_DEG,
        wall_h=G.WALL_H, eye=G.EYE_H, bob_amp=0.006, stride=0.7,
        decal_mesh=decal_mesh, decal_texs=decal_texs,
        light_cells=light_cells)

    # no mid-walk yaws: no glances, no hesitations, no stops — the camera
    # only turns where the path turns
    cc = lambda c: floorplan.lat_center(c, off)
    poly = world.round_corners(np.array([cc(c) for c in route]))
    poses = world.walk_poses(poly, G.SPEED, G.FPS,
                             slow_s=(), glances=(), stops=())

    out = OUT_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=G.FPS, codec="libx264",
                                quality=8, pixelformat="yuv420p")
    try:
        for (x, y, yaw, sv) in poses:
            writer.append_data(rend.render(x, y, yaw, sv))
    finally:
        writer.close()
    rend.release()
    print(f"{name}: turns {want}  junctions={junctions}  mood={mood}  "
          f"{len(poses)} frames ({len(poses) / G.FPS:.1f}s)")


def main():
    for spec in SAMPLES:
        make_sample(*spec)
    print(f"-> {OUT_DIR}")


if __name__ == "__main__":
    main()
