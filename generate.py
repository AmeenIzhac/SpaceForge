"""SpaceForge corridor probes — main generator.

Bare corridors with algorithmically generated turn sequences. No rooms, no
doors, no diversions, and no mid-walk yaws: the camera only turns where the
path turns. Each straight leg gets its own wall colour as an orientation cue,
and a red X is painted on the floor at the start of the walk, set a little
ahead of the walker so it passes right under the nose in full view.

Straight lengths are drawn per sample, not hardcoded: leg lengths span
LEG_MIN..LEG_MAX lattice steps, so the ratio between any two straights in a
sample can be anything from 1:1 to 1:10. Each sample targets a different band
of that range (see RATIO_BANDS), so the set as a whole exercises the whole
span instead of clustering in the middle.

Everything is seed-deterministic in MASTER_SEED. Outputs: out/main/<n>.mp4

    .venv/bin/python generate.py

The two earlier pipelines are kept as backups: dataset/generate_dataset.py
(full floor plans + ground truth) and photoreal/ (Three.js realism track).
"""

import math
import os
import random
from pathlib import Path

import numpy as np
import imageio.v2 as imageio

import decor
import floorplan
import gl_renderer
import pathgen
import props
import textures
import world

# ------------------------------- CONFIG -------------------------------------
OUT_DIR = Path(__file__).resolve().parent / "out" / "main"

N_SAMPLES = 5
MASTER_SEED = 11
MOOD = "office"           # one palette for the whole set
JUNCTIONS = False         # True: crossroads at every bend (dead-end stubs in
                          # the directions not taken), so each turn visibly
                          # could have gone left, right or straight

# Route shape. One lattice step = floorplan.S world cells = one corridor width
# times S, so a leg of L steps is 3L world units and takes 3L/SPEED seconds.
TURNS_MIN, TURNS_MAX = 3, 5       # turns per sample (legs = turns + 1)
LEG_MIN, LEG_MAX = 1, 10          # straight length in lattice steps
# Per-sample target for max:min straight ratio — together these span the whole
# 1:1 .. 1:10 range the generator is allowed to produce.
RATIO_BANDS = [(1.0, 1.2), (1.8, 2.6), (3.5, 4.5), (6.0, 7.5), (9.0, 10.0)]
TOTAL_STEPS_MIN, TOTAL_STEPS_MAX = 14, 30     # bounds the video length

# Floor marker at the start of the walk
MARKER_AHEAD = 3.0        # world units past the start, along the first leg
MARKER_ARM = 0.92         # arm length; the X spans ~0.71x this across the floor
MARKER_THICK = 0.15       # arm width
MARKER_RGB = (214, 30, 26)
MARKER_EMIT = 0.45        # keeps it legible in dim corridors

# Render / motion (self-contained: this script no longer borrows the dataset
# pipeline's CONFIG)
WIDTH, HEIGHT = 960, 544
FPS = 30
FOV_DEG = 68.0
WALL_H = 1.25             # corridor height, in corridor-widths
EYE_H = 0.60
SPEED = 1.26              # world units/s
N_WALL_VARIANTS = 8
LANDMARK_DENSITY = 0.14
LANDMARK_GAP = 1.6
DECOR_DENSITY = 0.16
DECOR_GAP = 1.7
# -----------------------------------------------------------------------------

TURN_NAME = {"L": "left", "R": "right"}
# wall variants in a maximally-separated order, so neighbouring legs never
# land on lookalike shades even when a sample runs to six legs
VARIANT_ORDER = [1, 5, 3, 7, 2, 6, 4, 8]


# --------------------------------------------------------------- route search
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


def draw_legs(rng, n_legs, band, total_min=None, total_max=None):
    """Leg lengths (lattice steps) whose max:min ratio lands inside `band`.

    The extremes are placed explicitly and the rest drawn between them, so the
    realized ratio is exactly max/min rather than whatever a blind uniform
    draw happened to produce. The probe planner passes its own total bounds,
    since it fixes the turn count instead of letting it float."""
    total_min = TOTAL_STEPS_MIN if total_min is None else total_min
    total_max = TOTAL_STEPS_MAX if total_max is None else total_max
    for _ in range(4000):
        lo = rng.randint(LEG_MIN, LEG_MAX)
        hi = int(round(lo * rng.uniform(*band)))
        if not lo <= hi <= LEG_MAX:
            continue
        legs = [rng.randint(lo, hi) for _ in range(n_legs)]
        i, j = rng.sample(range(n_legs), 2)
        legs[i], legs[j] = lo, hi
        if not total_min <= sum(legs) <= total_max:
            continue
        if band[0] <= hi / lo <= band[1]:
            return legs
    raise RuntimeError(f"no leg set for band {band} with {n_legs} legs")


def draw_route(rng, band):
    """(turns, legs, route, stubs) for a self-avoiding walk in this ratio band.

    Rejects routes that revisit a lattice node, and — when junctions are on —
    routes whose dead-end stubs would land on the route itself and silently
    turn a decoy into a real branch."""
    for _ in range(4000):
        n_turns = rng.randint(TURNS_MIN, TURNS_MAX)
        legs = draw_legs(rng, n_turns + 1, band)
        turns = "".join(rng.choice("LR") for _ in range(n_turns))
        route = build_route(turns, legs)
        if len(set(route)) != len(route):
            continue
        stubs = junction_stubs(route) if JUNCTIONS else []
        tips = [b for _, (_a, b) in stubs]
        if len(set(tips)) != len(tips) or set(tips) & set(route):
            continue
        return turns, legs, route, stubs
    raise RuntimeError(f"no self-avoiding route for band {band}")


# -------------------------------------------------------------- floor marker
def marker_mesh(x, y, yaw):
    """A red X painted flat on the floor, centred on (x, y) and squared up
    with the corridor running along `yaw`."""
    arm = props.box(MARKER_ARM, 0.02, MARKER_THICK, MARKER_RGB,
                    y0=0.012, emissive=MARKER_EMIT)
    cross = props.merge(props.transform(arm, yaw=math.pi / 4),
                        props.transform(arm, yaw=-math.pi / 4))
    return props.transform(cross, yaw=yaw, dx=x, dz=y)


def start_marker(route, legs_idx, cc):
    """World point + heading for the single start-of-walk marker, pushed
    MARKER_AHEAD down the first leg so it sits in front of the camera instead
    of directly underneath it: at this eye height and FOV the floor only
    enters frame about 1.6 units out. On a one-step first leg there isn't room
    for the full offset, so it shrinks to keep clear of the first bend."""
    i0, i1 = legs_idx[0]
    ax, ay = cc(route[i0])
    bx, by = cc(route[i0 + 1])
    h = math.hypot(bx - ax, by - ay)
    ux, uy = (bx - ax) / h, (by - ay) / h
    span = math.hypot(*(np.subtract(cc(route[i1]), (ax, ay))))
    ahead = min(MARKER_AHEAD, max(1.8, span - 1.0))
    return (ax + ux * ahead, ay + uy * ahead, -math.atan2(uy, ux))


# ------------------------------------------------------------------ geometry
def normalize(route, stubs):
    """Shift a route and its stubs so every lattice coordinate is >= 0."""
    all_nodes = list(route) + [e[1][1] for e in stubs]
    x0 = min(c[0] for c in all_nodes)
    y0 = min(c[1] for c in all_nodes)
    return ([(x - x0, y - y0) for x, y in route],
            [(i, ((a[0] - x0, a[1] - y0), (b[0] - x0, b[1] - y0)))
             for i, (a, b) in stubs])


def solve_walk(route):
    """Everything the renderer and the probe planner both need from a
    normalized route, with no scenery involved: the lattice->world map, the
    per-leg spans, the start marker, and the baked camera track.

    No mid-walk yaws — no glances, hesitations or stops — so the camera only
    turns where the path turns."""
    cc = lambda c: floorplan.lat_center(c, floorplan.PAD)
    legs_idx = leg_spans(route)
    marker = start_marker(route, legs_idx, cc)
    poly = world.round_corners(np.array([cc(c) for c in route]))
    poses = world.walk_poses(poly, SPEED, FPS, slow_s=(), glances=(), stops=())
    return cc, legs_idx, marker, poly, poses


def bearing_deg(pose, point):
    """Clockwise bearing in degrees from the walker's heading to `point` —
    the probe answer, with the walker's own forward taken as north.

    World axes are x east, y south, and the renderer aims the camera at
    (x + cos yaw, y + sin yaw), so forward is (cos, sin) and the camera's
    right — bearing 90 — is that turned a quarter clockwise, (-sin, cos)."""
    px, py, yaw = pose[0], pose[1], pose[2]
    fx, fy = math.cos(yaw), math.sin(yaw)
    rx, ry = -math.sin(yaw), math.cos(yaw)
    vx, vy = point[0] - px, point[1] - py
    b = math.degrees(math.atan2(vx * rx + vy * ry, vx * fx + vy * fy)) % 360.0
    return 0.0 if b >= 360.0 else b      # a hair below zero rounds up to 360


# ------------------------------------------------------------------- sample
def make_sample(name, turns, legs, seed):
    """Render one plan. `turns`/`legs` come from draw_route (or a saved probe
    plan); `seed` drives the scenery — palette, landmarks, wall decor."""
    prng = random.Random(seed)
    nrng = np.random.default_rng(seed)

    route = build_route(turns, legs)
    got = [t["turn"] for t in pathgen.route_turns(route)]
    want = [TURN_NAME[t] for t in turns]
    assert got == want, (got, want)

    stubs = junction_stubs(route) if JUNCTIONS else []
    route, stubs = normalize(route, stubs)

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
    cc, legs_idx, marker, poly, poses = solve_walk(route)
    variant_map = np.ones((size, size), np.int16)

    def paint_edge(a, b, v):
        dx, dy = b[0] - a[0], b[1] - a[1]
        for t in range(s + 1):
            cx = off + a[0] * s + t * dx
            cy = off + a[1] * s + t * dy
            variant_map[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2] = v

    def leg_variant(li):
        return VARIANT_ORDER[li % len(VARIANT_ORDER)]

    leg_of_bend = {i1: li for li, (i0, i1) in enumerate(legs_idx)}
    for bend_i, (a, b) in stubs:
        paint_edge(a, b, leg_variant(leg_of_bend[bend_i]))
    for li, (i0, i1) in enumerate(legs_idx):
        for k in range(i0, i1):
            paint_edge(route[k], route[k + 1], leg_variant(li))

    pal = textures.build_palette(MOOD, nrng, n_walls=N_WALL_VARIANTS,
                                 baked_lights=False)
    atlas = pal["walls"] + [pal["door"]]
    wmap = world.wmap_from_open(open_, n_variants=len(pal["walls"]),
                                door_id=len(atlas), salt=seed,
                                door_prob=0.0,        # no doors — no rooms behind
                                variant_map=variant_map)

    exclude = np.zeros_like(open_)
    lms = world.place_landmarks_wmap(wmap, exclude, prng, len(props.PROP_ORDER),
                                     density=LANDMARK_DENSITY,
                                     min_gap=LANDMARK_GAP,
                                     avoid_pts=[marker[:2]], avoid_r=1.3)
    wall_decor = decor.place_wall_decor(wmap, door_id=len(atlas), rng=prng,
                                        density=DECOR_DENSITY,
                                        min_gap=DECOR_GAP)

    mesh, shadows, plights = props.build_props(lms, prng)
    wall_mesh, wall_lights = decor.build_wall_meshes(wall_decor, prng)
    mesh = props.merge(mesh, wall_mesh, marker_mesh(*marker))
    decal_mesh, decal_texs = decor.build_decals(wall_decor, nrng)

    rend = gl_renderer.GLRenderer(
        wmap, atlas, pal["floor"], pal["ceil"], pal["floor_spec"], pal,
        mesh, shadows, plights + wall_lights,
        width=WIDTH, height=HEIGHT, fov_deg=FOV_DEG,
        wall_h=WALL_H, eye=EYE_H, bob_amp=0.006, stride=0.7,
        decal_mesh=decal_mesh, decal_texs=decal_texs,
        light_cells=light_cells)

    out = OUT_DIR / f"{name}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    # libx264 will otherwise grab every core it can see — with a few render
    # workers in parallel that starves anything else sharing the box
    threads = os.environ.get("SPACEFORGE_FFMPEG_THREADS")
    writer = imageio.get_writer(str(out), fps=FPS, codec="libx264",
                                quality=8, pixelformat="yuv420p",
                                output_params=(["-threads", threads]
                                               if threads else None))
    try:
        for (x, y, yaw, sv) in poses:
            writer.append_data(rend.render(x, y, yaw, sv))
    finally:
        writer.close()
    rend.release()

    print(f"{name}.mp4: turns {want}  legs {legs} steps  "
          f"ratio {max(legs) / min(legs):.1f}:1  junctions={JUNCTIONS}  "
          f"bearing {bearing_deg(poses[-1], marker):.1f} deg  "
          f"{len(poses)} frames ({len(poses) / FPS:.1f}s)", flush=True)
    return legs


def main():
    all_legs = []
    for i in range(N_SAMPLES):
        seed = MASTER_SEED + i
        turns, legs, _route, _stubs = draw_route(random.Random(seed),
                                                 RATIO_BANDS[i % len(RATIO_BANDS)])
        all_legs += make_sample(str(i), turns, legs, seed)
    print(f"straights: {min(all_legs)}..{max(all_legs)} lattice steps "
          f"({3 * min(all_legs)}..{3 * max(all_legs)} world units), "
          f"extreme ratio {max(all_legs) / min(all_legs):.0f}:1")
    print(f"-> {OUT_DIR}")


if __name__ == "__main__":
    main()
