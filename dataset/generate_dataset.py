"""SpaceForge corridor simulator — full dataset generator (backup pipeline).

The paper-faithful track: Algorithm 1 grid paths instantiated as furnished
floor plans, rendered as an outbound walk plus a return walk, with the full
spatial ground truth. Superseded as the default entry point by ../generate.py
(controlled corridor probes), kept here as the dataset-scale pipeline.

Edit the CONFIG block below (no CLI arguments), then run:

    .venv/bin/python dataset/generate_dataset.py

Each sample lands in out/dataset/sample_XX/ with:
    outbound.mp4         egocentric walk from start (green) to stop (red)
    return.mp4           egocentric walk back along the return route
    map.png              top-down diagram (Figure-4 style)
    meta.json            paths, junctions, ground-truth turn sequences, landmarks
    poses_outbound.csv   per-frame camera pose (frame, t, x, y, yaw)
    poses_return.csv

Rendering uses the moderngl GPU engine (real 3D props, point lighting,
specular floors, contact shadows, MSAA); if moderngl is unavailable it falls
back to the numpy raycaster. Samples render in parallel worker processes.
"""

import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # shared library

import decor
import floorplan
import mapfig
import pathgen
import props
import raycaster
import rooms
import textures
import world

# ------------------------------- CONFIG -------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "out" / "dataset"

NUM_SAMPLES = 6
WORKERS = 3           # parallel sample processes (1 = serial)
MASTER_SEED = 7
RENDERER = "gl"       # "gl" (moderngl engine) with automatic raycaster fallback

# Floor plan: tightly packed rectangles; corridors run along the wall lines
# between them (one lattice step = floorplan.S world units)
COARSE_W_RANGE = (8, 10)      # packing extent in coarse cells
COARSE_H_RANGE = (6, 8)
N_ROOMS_RANGE = (6, 9)        # rectangles to pack

# Algorithm 1 on the wall-line graph — lengths are lattice steps
OUT_LEN_RANGE = (4, 6)        # L — very short skeleton: with the tight turn
                              # budget, most turning happens inside rooms
N_CHORDS_RANGE = (1, 2)       # k
MIN_LEN_FRAC = 0.55           # Lmin = max(4, frac * L)
MAX_ATTEMPTS = 900            # T
MIN_RETURN_LEN = 3            # extra knob: forces a non-trivial return
MAX_RETURN_FACTOR = 1.5       # reject returns much longer than the outbound
BIAS_RETURN_JUNCTION = True   # deviation from line 17: make loops on P_ret bends
MIN_RETURN_DECISIONS = 2      # aim for this many left/right choices on the return
STRAIGHT_BIAS = 0.55          # deviation from line 8: prefer straight runs (0 = faithful)
TURN_PENALTY = 0.10           # keep return route bendy enough to host junctions
MIN_DOORED_ROOMS = 3          # retry layouts until this many rooms open onto corridors

WIDTH, HEIGHT = 960, 544
FPS = 30
FOV_DEG = 68.0
WALL_H = 1.25         # corridor height, in corridor-widths
EYE_H = 0.60
SPEED = 1.26          # walking speed, world units/s (0.7x — unhurried stroll)
RENDER_RETURN = True
PALETTE_ORDER = ["office", "brick", "lab", "hotel"]

N_WALL_VARIANTS = 8       # distinct wall variants; rooms claim one each
LANDMARK_DENSITY = 0.14   # probability of an object on each straight corridor cell
LANDMARK_GAP = 1.6        # min spacing between objects, world units
DECOR_DENSITY = 0.16      # wall decorations per open cell (corridors + rooms)
DECOR_GAP = 1.7           # min spacing between wall decorations, world units


# regular, consistent locomotion: no sway, no speed wobble, no gaze noise —
# the head only turns for deliberate glances and path turns
SWAY = 0.0
YAW_WANDER = 0.0
SPEED_WANDER = 0.0
CROSS_PROB = 0.7      # chance to cut through a bordering room (in one door,
MAX_CROSSINGS = 2     # out another) per opportunity, capped per walk
VISIT_PROB = 0.4      # chance to step into a room, look around, and leave
MAX_VISITS = 1        # through the same door, capped per walk
TURNS_MIN = 2         # total outbound turns (corridor bends + in-room turns:
TURNS_MAX = 5         # a crossing counts ~4, a visit ~3)
GLANCE_PROB = 0.45    # chance to glance at a passed landmark
SIDE_GLANCE_PROB = 1.0    # look down every side corridor passed (1.0 = always)
STOP_COUNTS = (0,)    # no random stops — only room visits pause (deliberately)
# -----------------------------------------------------------------------------


def _movement_events(route, poly, adj, landmarks, prng, cc, side_doors=(),
                     slow_pts=()):
    """Junction slowdowns, glances (side corridors, open room doors, landmarks),
    and full stops. cc maps a lattice node to its world-space center. slow_pts
    are extra hesitation points (doorways the walker crosses through). Route
    nodes bypassed by a room crossing are skipped via the near-route check."""
    total = world.polyline_length(poly)
    parr = np.asarray(poly, np.float64)
    seg = np.diff(parr, axis=0)
    seglen2 = np.maximum(seg[:, 0] ** 2 + seg[:, 1] ** 2, 1e-12)

    def near_route(px, py, r):
        t = np.clip(((px - parr[:-1, 0]) * seg[:, 0] +
                     (py - parr[:-1, 1]) * seg[:, 1]) / seglen2, 0.0, 1.0)
        d2 = (parr[:-1, 0] + t * seg[:, 0] - px) ** 2 + \
             (parr[:-1, 1] + t * seg[:, 1] - py) ** 2
        return float(d2.min()) < r * r

    junctions = [c for c in route[1:-1]
                 if len(adj[c]) >= 3 and near_route(*cc(c), 0.6)]
    slow_s = world.arclengths_at(poly, [cc(c) for c in junctions])
    slow_s += world.arclengths_at(poly, list(slow_pts))

    # side-corridor glances: look down every passable turn the walker goes past
    side = []
    for i in range(1, len(route) - 1):
        c = route[i]
        branches = [b for b in adj[c] if b not in (route[i - 1], route[i + 1])]
        if not branches or prng.random() > SIDE_GLANCE_PROB:
            continue
        if not near_route(*cc(c), 0.6):     # bypassed by a room crossing
            continue
        sj = world.arclengths_at(poly, [cc(c)])[0]
        bx, by = cc(prng.choice(branches))
        side.append({"s0": max(sj - 1.5, 0.2), "s1": sj + 0.25, "x": bx, "y": by})

    # open-door glances: look into every side room the walker passes (door
    # passages sit exactly 1 unit off the corridor centerline); the walker
    # also slows down a little while peeking in
    for d in side_doors:
        if prng.random() > SIDE_GLANCE_PROB or not near_route(d["x"], d["y"], 1.35):
            continue
        sd = world.arclengths_at(poly, [(d["x"], d["y"])])[0]
        side.append({"s0": max(sd - 1.0, 0.2), "s1": sd + 0.65,
                     "x": d["cx"], "y": d["cy"]})
        slow_s.append(sd)

    # landmark glances (lower priority — kept only if room remains)
    lm_glances = []
    for lm in landmarks:
        if prng.random() > GLANCE_PROB:
            continue
        if not near_route(lm["x"], lm["y"], 0.8):   # on a different corridor
            continue
        s_lm = world.arclengths_at(poly, [(lm["x"], lm["y"])])[0]
        if s_lm < 1.0:
            continue
        lm_glances.append({"s0": max(s_lm - 2.0, 0.2), "s1": s_lm - 0.25,
                           "x": lm["x"], "y": lm["y"]})
    glances = sorted(side + lm_glances, key=lambda g: g["s0"])

    stops = []
    n_stops = prng.choice(STOP_COUNTS)
    for _ in range(n_stops * 3):
        if len(stops) >= n_stops:
            break
        s_stop = prng.uniform(2.0, max(total - 2.0, 2.5))
        if all(abs(s_stop - sj) > 1.3 for sj in slow_s) and \
           all(abs(s_stop - s0) > 1.5 for s0, _ in stops):
            stops.append((s_stop, prng.uniform(0.4, 1.0)))

    return dict(slow_s=slow_s, glances=glances, stops=sorted(stops))


def build_scene(i):
    """Everything needed to render sample i (deterministic in MASTER_SEED)."""
    nrng = np.random.default_rng(MASTER_SEED * 997 + i)

    # retry until the packing, the walk, and enough doored rooms all work out
    path = built = None
    for attempt in range(10):
        prng = random.Random(MASTER_SEED * 1009 + i + attempt * 7919)
        plan = floorplan.make_plan(prng, prng.randint(*COARSE_W_RANGE),
                                   prng.randint(*COARSE_H_RANGE),
                                   prng.randint(*N_ROOMS_RANGE))
        if plan is None:
            continue
        out_len = prng.randint(*OUT_LEN_RANGE)
        min_len = max(4, int(MIN_LEN_FRAC * out_len))
        n_chords = prng.randint(*N_CHORDS_RANGE)
        path = pathgen.generate_returnable_path_graph(
            plan["adj"], out_len, min_len, MAX_ATTEMPTS, n_chords, prng,
            min_return_len=MIN_RETURN_LEN,
            max_return_len=int(out_len * MAX_RETURN_FACTOR) + 1,
            bias_return_junction=BIAS_RETURN_JUNCTION,
            min_return_decisions=MIN_RETURN_DECISIONS,
            straight_bias=STRAIGHT_BIAS, turn_penalty=TURN_PENALTY)
        if path is None:
            continue
        built = floorplan.build_world(plan, path, prng)
        crossings = floorplan.plan_crossings(plan, built, path, prng,
                                             prob=CROSS_PROB,
                                             max_per_route=MAX_CROSSINGS,
                                             visit_prob=VISIT_PROB,
                                             max_visits=MAX_VISITS)

        def turn_total():
            return len(pathgen.route_turns(path["out"])) + \
                sum(4 if c["kind"] == "cross" else 3 for c in crossings["out"])

        while turn_total() > TURNS_MAX and crossings["out"]:
            crossings["out"].pop()          # shed diversions to fit the budget
        if built["n_doored"] >= MIN_DOORED_ROOMS and \
                ((crossings["out"] and TURNS_MIN <= turn_total() <= TURNS_MAX)
                 or attempt >= 7):
            break
    if path is None or built is None:
        raise RuntimeError("floor-plan generation failed: no walkable wall graph")
    adj = pathgen.corridor_graph(path)
    off = built["off"]
    cc = lambda c: floorplan.lat_center(c, off)

    pname = PALETTE_ORDER[i % len(PALETTE_ORDER)]
    use_gl = RENDERER == "gl"
    if use_gl:
        try:
            import gl_renderer
            if gl_renderer.moderngl is None:
                raise RuntimeError("moderngl missing")
        except Exception as e:
            print(f"  [sample {i:02d}] GL unavailable ({e}); using raycaster", flush=True)
            use_gl = False

    pal = textures.build_palette(pname, nrng, n_walls=N_WALL_VARIANTS,
                                 baked_lights=not use_gl)
    atlas = pal["walls"] + [pal["door"]]

    rooms_list = built["rooms"]

    def _div_doors(c):
        return (c["d_in"], c["d_out"]) if c["kind"] == "cross" else (c["door"],)

    cross_cells = {(int(d["x"]), int(d["y"]))
                   for cs in crossings.values() for c in cs for d in _div_doors(c)}
    side_doors = [dict(d, cx=r["center"][0], cy=r["center"][1])
                  for r in rooms_list for d in r["doors"]
                  if (int(d["x"]), int(d["y"])) not in cross_cells]

    # per-room wall colours: walls take the variant of the room that owns
    # them (accent side gets a contrasting variant); unowned walls (building
    # perimeter) stay on variant 1
    nv = len(pal["walls"])
    owner = built["owner"]
    base_v = (np.where(owner >= 1000, owner - 1000, np.maximum(owner, 0)) % nv) + 1
    accent_v = ((owner - 1000 + 3) % nv) + 1
    variant_map = np.where(owner >= 1000, accent_v,
                           np.where(owner >= 0, base_v, 1))
    wmap = world.wmap_from_open(built["open"], n_variants=nv,
                                door_id=len(atlas),
                                salt=MASTER_SEED * 31 + i * 7,
                                door_prob=pal["door_prob"],
                                variant_map=variant_map)

    exclude = built["room_mask"].copy()
    for (x, y) in built["passages"]:
        exclude[y, x] = True
    junction_pts = [cc(c) for c, nb in adj.items() if len(nb) >= 3]
    lms = world.place_landmarks_wmap(
        wmap, exclude, prng, len(props.PROP_ORDER),
        density=LANDMARK_DENSITY, min_gap=LANDMARK_GAP,
        avoid_pts=[(d["x"], d["y"]) for d in built["doors"]] + junction_pts,
        avoid_r=1.4)
    exterior = set(map(tuple, np.argwhere((~built["open"]) & (owner == -1))[:, ::-1]))
    wall_decor = decor.place_wall_decor(wmap, door_id=len(atlas), rng=prng,
                                        density=DECOR_DENSITY, min_gap=DECOR_GAP,
                                        skip_cells=built["passages"],
                                        exterior_cells=exterior)
    furn_mesh, furn_shadows, furn_lights, furniture = \
        rooms.build_room_contents(rooms_list, pname, prng)
    bob_amp = 0.005 + 0.003 * prng.random()   # subtle, smooth walking bob
    stride = 0.62 + 0.18 * prng.random()

    if use_gl:
        import gl_renderer
        mesh, shadows, plights = props.build_props(lms, prng)
        wall_mesh, wall_lights = decor.build_wall_meshes(wall_decor, prng)
        frame_mesh = rooms.build_door_frames(rooms_list, prng, WALL_H,
                                             frame_rgb=pal["door_rgb"])
        mesh = props.merge(mesh, wall_mesh, furn_mesh, frame_mesh)
        shadows = shadows + furn_shadows
        decal_mesh, decal_texs = decor.build_decals(wall_decor, nrng)
        rend = gl_renderer.GLRenderer(
            wmap, atlas, pal["floor"], pal["ceil"], pal["floor_spec"], pal,
            mesh, shadows, plights + wall_lights + furn_lights,
            width=WIDTH, height=HEIGHT, fov_deg=FOV_DEG, wall_h=WALL_H, eye=EYE_H,
            bob_amp=bob_amp, stride=stride,
            decal_mesh=decal_mesh, decal_texs=decal_texs,
            room_mask=built["room_mask"], room_centers=built["panel_pts"],
            room_floor_tex=pal["floor2"], room_floor_spec=pal["floor2_spec"],
            light_cells=built["light_cells"])
    else:
        atlas = decor.bake_decals(atlas, wmap, wall_decor, nrng, WALL_H)
        cat = textures.sprite_catalog(nrng)
        sprites = [dict(x=l["x"], y=l["y"], tex=cat[l["t"]]["tex"],
                        h=cat[l["t"]]["h"], w=cat[l["t"]]["w"]) for l in lms]
        rend = raycaster.Renderer(
            wmap, np.stack(atlas, 0), pal["floor"], pal["ceil"], sprites,
            width=WIDTH, height=HEIGHT, fov_deg=FOV_DEG, wall_h=WALL_H, eye=EYE_H,
            fog_d=pal["fog_d"], fog_rgb=pal["fog_rgb"], side_shade=pal["side_shade"],
            bob_amp=bob_amp, stride=stride)

    glance_targets = lms + [{"x": d["x"] + d["nx"] * 0.06, "y": d["y"] + d["ny"] * 0.06}
                            for d in wall_decor]
    wander = dict(rng=prng, sway=SWAY, yaw_wander=YAW_WANDER, speed_wander=SPEED_WANDER)

    def _route_motion(route, divs, initial_yaw=None):
        wp = floorplan.route_waypoints(route, divs, off)
        poly = world.round_corners(np.array(wp))
        slow = [(d["x"], d["y"]) for c in divs for d in _div_doors(c)]
        ev = _movement_events(route, poly, adj, glance_targets, prng, cc,
                              side_doors=side_doors, slow_pts=slow)
        # room visits pause deep inside the room for a natural look-around
        visit_stops = [(world.arclengths_at(poly, [c["deep"]])[0], c["look"], c["amp"])
                       for c in divs if c["kind"] == "visit"]
        ev["stops"] = sorted(ev["stops"] + visit_stops)
        poses = world.walk_poses(poly, SPEED, FPS, initial_yaw=initial_yaw,
                                 **wander, **ev)
        return wp, poses, ev

    wp_out, poses_out, ev_out = _route_motion(path["out"], crossings["out"])
    poses_ret, ev_ret = None, None
    if RENDER_RETURN:
        wp_ret, poses_ret, ev_ret = _route_motion(path["ret"], crossings["ret"],
                                                  initial_yaw=poses_out[-1][2])

    routes_world = {"out": wp_out,
                    "ret": wp_ret if RENDER_RETURN else [cc(c) for c in path["ret"]],
                    "chords": [[cc(c) for c in ch] for ch in path["chords"]]}
    return dict(path=path, adj=adj, palette=pname, rend=rend,
                backend="gl" if use_gl else "raycast",
                poses_out=poses_out, poses_ret=poses_ret, landmarks=lms,
                rooms=rooms_list, furniture=furniture, routes_world=routes_world,
                crossings=crossings, world_size=built["size"],
                wall_decor=wall_decor, wall_style=pal.get("wall_style", ""),
                params=dict(L=out_len, Lmin=min_len, T=MAX_ATTEMPTS,
                            k=n_chords, min_return_len=MIN_RETURN_LEN,
                            bias_return_junction=BIAS_RETURN_JUNCTION,
                            min_return_decisions=MIN_RETURN_DECISIONS,
                            straight_bias=STRAIGHT_BIAS, turn_penalty=TURN_PENALTY,
                            lattice_scale=floorplan.S),
                events=dict(outbound=ev_out, ret=ev_ret))


def render_video(rend, poses, out_path, label):
    writer = imageio.get_writer(str(out_path), fps=FPS, codec="libx264",
                                quality=8, pixelformat="yuv420p")
    t0 = time.time()
    try:
        for i, (x, y, yaw, s) in enumerate(poses):
            writer.append_data(rend.render(x, y, yaw, s))
            if (i + 1) % 300 == 0:
                fps = (i + 1) / (time.time() - t0)
                print(f"    {label}: frame {i + 1}/{len(poses)} ({fps:.1f} fps)", flush=True)
    finally:
        writer.close()
    print(f"    {label}: {len(poses)} frames ({len(poses) / FPS:.1f} s) "
          f"rendered in {time.time() - t0:.1f} s", flush=True)


def write_poses(poses, out_path):
    with open(out_path, "w") as fh:
        fh.write("frame,t,x,y,yaw\n")
        for i, (x, y, yaw, _s) in enumerate(poses):
            fh.write(f"{i},{i / FPS:.3f},{x:.4f},{y:.4f},{yaw:.4f}\n")


def make_sample(i):
    d = OUT_DIR / f"sample_{i:02d}"
    d.mkdir(parents=True, exist_ok=True)
    sc = build_scene(i)
    path, adj = sc["path"], sc["adj"]
    p = sc["params"]

    n_doored = sum(1 for r in sc["rooms"] if not r.get("sealed"))
    print(f"  [{i:02d}] {sc['backend']} palette={sc['palette']}/{sc['wall_style']} "
          f"outbound={len(path['out'])} return={len(path['ret'])} "
          f"chords={[len(c) for c in path['chords']]} landmarks={len(sc['landmarks'])} "
          f"decor={len(sc['wall_decor'])} rooms={n_doored}/{len(sc['rooms'])}doored "
          f"furniture={len(sc['furniture'])} "
          f"crossings={len(sc['crossings']['out'])}+{len(sc['crossings']['ret'])}",
          flush=True)

    render_video(sc["rend"], sc["poses_out"], d / "outbound.mp4", f"[{i:02d}] outbound")
    write_poses(sc["poses_out"], d / "poses_outbound.csv")
    if sc["poses_ret"] is not None:
        render_video(sc["rend"], sc["poses_ret"], d / "return.mp4", f"[{i:02d}] return")
        write_poses(sc["poses_ret"], d / "poses_return.csv")
    if hasattr(sc["rend"], "release"):
        sc["rend"].release()

    mapfig.draw_plan(sc["world_size"], sc["rooms"], sc["routes_world"], d / "map.png")

    junctions = sorted(c for c, nb in adj.items() if len(nb) >= 3)
    decisions = pathgen.route_decisions(path["ret"], adj)
    meta = {
        "sample": i,
        "master_seed": MASTER_SEED,
        "renderer": sc["backend"],
        "palette": sc["palette"],
        "algorithm1": p,
        "p_out": [list(c) for c in path["out"]],
        "p_ret": [list(c) for c in path["ret"]],
        "chords": [[list(c) for c in ch] for ch in path["chords"]],
        "junctions": [list(c) for c in junctions],
        "return_decisions": decisions,
        "return_turns": pathgen.route_turns(path["ret"]),
        "outbound_turns": pathgen.route_turns(path["out"]),
        "landmarks": [{"name": props.PROP_ORDER[l["t"] % len(props.PROP_ORDER)],
                       "cell": list(l["cell"]),
                       "world": [round(l["x"], 3), round(l["y"], 3)]}
                      for l in sc["landmarks"]],
        "wall_style": sc["wall_style"],
        "world_size": list(sc["world_size"]),
        "rooms": [{"type": r.get("rtype", "sealed"),
                   "rect_world": list(r["rect"]),
                   "center": [round(v, 2) for v in r["center"]],
                   "doorways": [[round(dd["x"], 2), round(dd["y"], 2)]
                                for dd in r["doors"]]}
                  for r in sc["rooms"]],
        "furniture": sc["furniture"],
        "room_diversions": {rn: [
            {"kind": c["kind"], "room": c["room"],
             **({"door_in": [round(c["d_in"]["x"], 2), round(c["d_in"]["y"], 2)],
                 "door_out": [round(c["d_out"]["x"], 2), round(c["d_out"]["y"], 2)]}
                if c["kind"] == "cross" else
                {"door": [round(c["door"]["x"], 2), round(c["door"]["y"], 2)],
                 "deep": [round(c["deep"][0], 2), round(c["deep"][1], 2)]})}
            for c in cs] for rn, cs in sc["crossings"].items()},
        "wall_decor": [{"type": d["type"],
                        "world": [round(d["x"], 3), round(d["y"], 3)],
                        "facing": round(float(np.arctan2(d["ny"], d["nx"])), 4)}
                       for d in sc["wall_decor"]],
        "movement": {
            "outbound": {"stops": len(sc["events"]["outbound"]["stops"]),
                         "glances": len(sc["events"]["outbound"]["glances"])},
            "return": None if sc["events"]["ret"] is None else
                      {"stops": len(sc["events"]["ret"]["stops"]),
                       "glances": len(sc["events"]["ret"]["glances"])},
        },
        "camera": {"resolution": [WIDTH, HEIGHT], "fps": FPS, "fov_deg": FOV_DEG,
                   "eye_height": EYE_H, "wall_height": WALL_H, "speed": SPEED},
        "world": f"routes are wall-lattice nodes; node (i,j) center = "
                 f"({floorplan.PAD}+{floorplan.S}*i+0.5, {floorplan.PAD}+{floorplan.S}*j+0.5) "
                 f"world units; corridor width = 1 unit; y grows downward (south); "
                 f"yaw = atan2(dy, dx)",
    }
    with open(d / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    dec = "".join(m["action"][0].upper() for m in decisions) or "-"
    print(f"  [{i:02d}] junctions={len(junctions)} return decisions: {dec}", flush=True)
    return i


def main():
    t0 = time.time()
    print(f"SpaceForge corridor sim: {NUM_SAMPLES} samples, {WORKERS} workers "
          f"-> {OUT_DIR}", flush=True)
    if WORKERS > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(WORKERS, maxtasksperchild=1) as pool:
            pool.map(make_sample, range(NUM_SAMPLES))
    else:
        for i in range(NUM_SAMPLES):
            make_sample(i)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
