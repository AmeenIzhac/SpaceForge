"""Rooms: rectangular spaces carved off the corridor network.

Two kinds:
  * through rooms — centered on a straight stretch of the route, so the agent
    enters through a doorway, crosses the room, and leaves through the
    opposite doorway;
  * side rooms — reached through a single 1-cell doorway punched in a
    corridor wall, so the agent can look inside while walking past. They are
    dead ends: the corridor topology (junctions, turn ground truth) is
    untouched.

Every doorway gets real frame geometry (jambs + header + usually an open door
leaf swung into the room). Rooms are furnished by type (office / lounge /
storage / bedroom) with pieces placed along the walls, keep-outs around
doorways and the route line, and get their own floor texture and a centered
ceiling light panel.

World-cell convention matches world.py: route cell (x, y) sits at world cell
(2x+1, 2y+1); continuous point of world cell (wx, wy) center = (wx+.5, wy+.5).
"""

import numpy as np

import props

# furniture / prop footprints: name -> (depth along facing +X, width along wall)
FOOT = {
    "table": (0.62, 0.95), "desk": (0.62, 0.95), "chair": (0.46, 0.46),
    "shelf": (0.34, 0.95), "sofa": (0.74, 1.18), "bed": (1.00, 1.82),
    "cabinet": (0.40, 0.60), "rug": (0.92, 1.36),
    "tv": (0.44, 1.02), "coffee_table": (0.54, 0.94), "wardrobe": (0.58, 1.02),
    "plant": (0.44, 0.44), "bin": (0.30, 0.30), "crate": (0.38, 0.38),
    "boxes": (0.44, 0.38), "barrel": (0.34, 0.34), "lamp": (0.28, 0.28),
}

ROOM_TYPES = {
    "office": ["desk", "chair", "chair", "cabinet", "shelf", "plant", "bin", "boxes"],
    "lounge": ["sofa", "table", "plant", "lamp", "chair", "shelf", "tv", "coffee_table"],
    "storage": ["shelf", "crate", "boxes", "barrel", "bin", "crate", "boxes"],
    "bedroom": ["bed", "cabinet", "lamp", "plant", "chair", "wardrobe"],
}
MOOD_ROOMS = {
    "office": ["office", "office", "lounge", "storage"],
    "lab": ["office", "storage", "office"],
    "brick": ["storage", "storage", "office"],
    "hotel": ["bedroom", "bedroom", "lounge", "storage"],
}
CENTER_RUG = {"lounge", "bedroom"}   # room types that may get a centered rug


def _rect_cells(x0, y0, x1, y1):
    return [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]


def _ring_cells(x0, y0, x1, y1):
    """4-adjacent border of the rect. Diagonal corners are deliberately NOT
    included: touching corners don't connect in a 4-connected grid world."""
    out = []
    for x in range(x0, x1 + 1):
        out += [(x, y0 - 1), (x, y1 + 1)]
    for y in range(y0, y1 + 1):
        out += [(x0 - 1, y), (x1 + 1, y)]
    return out


def place_rooms(n, path, open_mask, rng, n_through=2, n_side=3):
    """Carve rooms into the wall space around the corridors.

    Returns dict(rooms=[...], extra_open=set, room_cells=set, passages=set).
    Each room: {"rect": (x0,y0,x1,y1), "kind", "center": (cx,cy),
                "doors": [{"x","y","nx","ny","depth","kind"}]}.
    """
    size = open_mask.shape[0]
    open_ = open_mask.copy()
    rooms, extra_open, room_cells, passages = [], set(), set(), set()

    def free(cells, allowed=()):
        for (x, y) in cells:
            if not (1 <= x <= size - 2 and 1 <= y <= size - 2):
                return False
            if open_[y, x] and (x, y) not in allowed:
                return False
        return True

    def carve(cells):
        for (x, y) in cells:
            open_[y, x] = True
            extra_open.add((x, y))

    # --- through rooms: centered on straight interior route cells -----------
    cands = []
    for route in (path["out"], path["ret"]):
        for i in range(1, len(route) - 1):
            a, b, c = route[i - 1], route[i], route[i + 1]
            if (b[0] - a[0], b[1] - a[1]) == (c[0] - b[0], c[1] - b[1]):
                cands.append((b, (c[0] - b[0], c[1] - b[1])))
    rng.shuffle(cands)
    for b, d in cands:
        if sum(1 for r in rooms if r["kind"] == "through") >= n_through:
            break
        wx, wy = 2 * b[0] + 1, 2 * b[1] + 1
        along = (abs(d[0]), abs(d[1]))                    # unit world axis
        lat = (along[1], along[0])
        # lateral extent (lo..hi) around the route line: centered or shifted to
        # one side, so rooms fit on the outer edge of the loop too
        opts = [(-1, 1), (0, 2), (-2, 0)]
        rng.shuffle(opts)
        if rng.random() < 0.3:
            opts.insert(0, (0, 3) if rng.random() < 0.5 else (-3, 0))
        for lo, hi in opts:
            xa = wx - along[0] + lo * lat[0]
            ya = wy - along[1] + lo * lat[1]
            xb = wx + along[0] + hi * lat[0]
            yb = wy + along[1] + hi * lat[1]
            x0, x1 = min(xa, xb), max(xa, xb)
            y0, y1 = min(ya, yb), max(ya, yb)
            line = {(wx - along[0], wy - along[1]), (wx, wy),
                    (wx + along[0], wy + along[1])}
            ring_ok = {(wx - 2 * along[0], wy - 2 * along[1]),
                       (wx + 2 * along[0], wy + 2 * along[1])}
            rect = _rect_cells(x0, y0, x1, y1)
            if not free(rect, allowed=line):
                continue
            if not free(_ring_cells(x0, y0, x1, y1), allowed=ring_ok):
                continue
            carve([c_ for c_ in rect if not open_[c_[1], c_[0]]])
            room_cells.update(rect)
            doors = []
            for s in (-1, 1):
                fx = wx + (1.5 * s + 0.5) * along[0] + 0.5 * lat[0]
                fy = wy + (1.5 * s + 0.5) * along[1] + 0.5 * lat[1]
                doors.append({"x": fx, "y": fy, "nx": -s * along[0],
                              "ny": -s * along[1], "depth": 0.14, "kind": "through"})
            rooms.append({"rect": (x0, y0, x1, y1), "kind": "through",
                          "center": ((x0 + x1 + 1) / 2, (y0 + y1 + 1) / 2),
                          "axis": along,
                          "route_lat": (wy + 0.5) if along[0] else (wx + 0.5),
                          "doors": doors})
            break

    # --- side rooms: one doorway punched in a corridor wall ------------------
    # prefer walls of the walked routes (out/ret) so the agent actually passes
    # the open door; chords only as fallback. Doors attach perpendicular to
    # STRAIGHT route cells only — on a bend the walker turns away just as it
    # passes, so it could never look inside.
    def straight_dirs(route):
        out = {}
        for i in range(1, len(route) - 1):
            d1 = (route[i][0] - route[i - 1][0], route[i][1] - route[i - 1][1])
            d2 = (route[i + 1][0] - route[i][0], route[i + 1][1] - route[i][1])
            if d1 == d2:
                out[route[i]] = d1
        return out

    walked, fallback = [], []
    for route, pool in ((path["out"], walked), (path["ret"], walked),
                        *((ch, fallback) for ch in path["chords"])):
        for c, d in straight_dirs(route).items():
            for dw in ((d[1], d[0]), (-d[1], -d[0])):
                pool.append((c, dw, pool is walked))
    rng.shuffle(walked)
    rng.shuffle(fallback)
    cands = walked + fallback
    door_pts = [d for r in rooms for d in r["doors"]]
    for c, dw, on_walked in cands:
        if sum(1 for r in rooms if r["kind"] == "side") >= n_side:
            break
        cw = (2 * c[0] + 1, 2 * c[1] + 1)
        w = (cw[0] + dw[0], cw[1] + dw[1])                # doorway passage cell
        lat = (abs(dw[1]), abs(dw[0]))
        dx, dy = abs(dw[0]), abs(dw[1])
        depth, halfw = (3, 1) if rng.random() < 0.7 else (4, 1)
        sx, sy = cw[0] + 2 * dw[0], cw[1] + 2 * dw[1]     # room cell at the door
        x0 = min(sx, sx + (depth - 1) * dw[0]) - halfw * lat[0]
        y0 = min(sy, sy + (depth - 1) * dw[1]) - halfw * lat[1]
        x1 = max(sx, sx + (depth - 1) * dw[0]) + halfw * lat[0]
        y1 = max(sy, sy + (depth - 1) * dw[1]) + halfw * lat[1]
        dpt = (w[0] + 0.5, w[1] + 0.5)
        if any((dpt[0] - q["x"]) ** 2 + (dpt[1] - q["y"]) ** 2 < 2.2 ** 2
               for q in door_pts):
            continue
        rect = _rect_cells(x0, y0, x1, y1)
        if not free(rect) or not free(_ring_cells(x0, y0, x1, y1), allowed={w}):
            continue
        carve(rect + [w])
        room_cells.update(rect)
        passages.add(w)
        door = {"x": dpt[0], "y": dpt[1], "nx": dw[0], "ny": dw[1],
                "depth": 1.04, "kind": "side"}
        door_pts.append(door)
        rooms.append({"rect": (x0, y0, x1, y1), "kind": "side",
                      "center": ((x0 + x1 + 1) / 2, (y0 + y1 + 1) / 2),
                      "axis": (dx, dy), "on_walked": on_walked, "doors": [door]})

    return dict(rooms=rooms, extra_open=extra_open,
                room_cells=room_cells, passages=passages)


# ---------------------------------------------------------------------------
# Furniture fill
# ---------------------------------------------------------------------------

def _seg_dist2(px, py, a, b):
    """Squared distance from point to segment ab."""
    vx, vy = b[0] - a[0], b[1] - a[1]
    l2 = max(vx * vx + vy * vy, 1e-9)
    t = max(0.0, min(1.0, ((px - a[0]) * vx + (py - a[1]) * vy) / l2))
    qx, qy = a[0] + t * vx, a[1] + t * vy
    return (px - qx) ** 2 + (py - qy) ** 2


def build_room_contents(rooms, mood, rng):
    """Furnish every room. Returns (mesh, shadows, lights, records)."""
    all_b = {**props.BUILDERS, **props.FURNITURE}
    meshes, shadows, lights, records = [], [], [], []
    for ri, room in enumerate(rooms):
        if room.get("sealed"):
            continue
        rtype = rng.choice(MOOD_ROOMS[mood])
        room["rtype"] = rtype
        x0, y0, x1, y1 = room["rect"]
        bx0, by0, bx1, by1 = x0, y0, x1 + 1.0, y1 + 1.0
        cx, cy = room["center"]
        axis = room.get("axis", (1, 0))
        placed = []

        def ok(px, py, half, skip_route=False):
            if not (bx0 + half + 0.04 <= px <= bx1 - half - 0.04 and
                    by0 + half + 0.04 <= py <= by1 - half - 0.04):
                return False
            for d in room["doors"]:
                if (px - d["x"]) ** 2 + (py - d["y"]) ** 2 < (0.85 + half * 0.5) ** 2:
                    return False
            for a, b in room.get("cross_segs", ()):   # walker cuts through here
                if _seg_dist2(px, py, a, b) < (0.55 + half * 0.4) ** 2:
                    return False
            if room["kind"] == "through" and not skip_route:
                lc = py if axis[0] else px
                if abs(lc - room["route_lat"]) < 0.55 + half * 0.4:
                    return False
            return all((px - q[0]) ** 2 + (py - q[1]) ** 2 >= (half + q[2] + 0.10) ** 2
                       for q in placed)

        def put(name, px, py, yaw):
            m, shadow_r, light = all_b[name](rng)
            meshes.append(props.transform(m, yaw=yaw, dx=px, dz=py,
                                          tint=0.92 + 0.16 * rng.random()))
            if shadow_r > 0:
                shadows.append((px, py, shadow_r))
            if light is not None:
                lx, lh, lz = light
                ca, sa = np.cos(yaw), np.sin(yaw)
                lights.append((px + lx * ca + lz * sa, lh, py - lx * sa + lz * ca))
            records.append({"name": name, "room": ri,
                            "world": [round(px, 2), round(py, 2)]})

        pool = ROOM_TYPES[rtype]
        # centered rug (flat: the route may cross it in through rooms)
        if rtype in CENTER_RUG and rng.random() < 0.75:
            ryaw = 0.0 if axis[0] else np.pi / 2
            put("rug", cx, cy, ryaw + rng.uniform(-0.06, 0.06))
        # centered table for lounges when there is room off the route line
        if rtype == "lounge" and rng.random() < 0.6:
            half = max(FOOT["table"]) / 2
            tx, ty = cx, cy
            if room["kind"] == "through":
                tx += axis[1] * 1.05
                ty += axis[0] * 1.05
            if ok(tx, ty, half, skip_route=False):
                put("table", tx, ty, rng.uniform(0, np.pi))
                placed.append((tx, ty, half))

        area = (bx1 - bx0) * (by1 - by0)
        budget = min(8, max(2, int(area * 0.14))) - len(placed)
        walls = [((1, 0), bx0), ((-1, 0), bx1), ((0, 1), by0), ((0, -1), by1)]
        for _ in range(60):
            if budget <= 0:
                break
            name = rng.choice(pool)
            (fd, fw) = FOOT[name]
            half = max(fd, fw) / 2
            (nx, ny), wall_pos = walls[int(rng.random() * 4)]
            inset = fd / 2 + 0.06
            if nx:
                px = wall_pos + nx * inset
                py = rng.uniform(by0 + fw / 2 + 0.10, by1 - fw / 2 - 0.10)
            else:
                py = wall_pos + ny * inset
                px = rng.uniform(bx0 + fw / 2 + 0.10, bx1 - fw / 2 - 0.10)
            if not ok(px, py, half):
                continue
            yaw = -float(np.arctan2(ny, nx)) + rng.uniform(-0.05, 0.05)
            put(name, px, py, yaw)
            placed.append((px, py, half))
            budget -= 1

    return (props.merge(*meshes) if meshes else None), shadows, lights, records


# ---------------------------------------------------------------------------
# Door frames: jambs + header + an open leaf swung into the room
# ---------------------------------------------------------------------------

def build_door_frames(rooms, rng, wall_h, frame_rgb=(92, 70, 48), leaf_rgb=None):
    leaf_rgb = leaf_rgb or tuple(int(v * 1.15) for v in frame_rgb)
    meshes = []
    for room in rooms:
        for d in room["doors"]:
            depth = d["depth"]
            parts = [
                props.box(depth, wall_h - 0.97, 1.02, frame_rgb, y0=0.97),
                props.transform(props.box(depth, 0.97, 0.15, frame_rgb), dz=0.435),
                props.transform(props.box(depth, 0.97, 0.15, frame_rgb), dz=-0.435),
            ]
            flat = d.get("leaf_flat", False)
            if flat or rng.random() < 0.85:               # open door leaf
                hinge = 0.36 if rng.random() < 0.5 else -0.36
                # doors the walker passes through are pinned nearly flat
                # against the interior wall, out of the camera's path
                lo, span = (2.78, 0.28) if flat else (1.75, 0.9)
                alpha = (lo + span * rng.random()) * (1.0 if hinge > 0 else -1.0)
                leaf = props.transform(
                    props.box(0.035, 0.93, 0.70, leaf_rgb, y0=0.02),
                    dz=-0.35 if hinge > 0 else 0.35)
                leaf = props.transform(leaf, yaw=alpha)
                leaf = props.transform(leaf, dx=depth / 2 + 0.03, dz=hinge)
                parts.append(leaf)
            yaw = -float(np.arctan2(d["ny"], d["nx"]))
            meshes.append(props.transform(props.merge(*parts),
                                          yaw=yaw, dx=d["x"], dz=d["y"]))
    return props.merge(*meshes) if meshes else None
