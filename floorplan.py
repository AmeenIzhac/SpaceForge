"""Floor-plan world: tightly packed rectangles with corridors along the lines
between them.

Stage 1 packs coarse rectangles (rooms) with no gaps between them — new rects
must share edges with the existing block and may not create holes, so the
union is an irregular but solid silhouette (like a real building footprint).

Stage 2 extracts the wall-line graph: lattice nodes at rectangle corners,
edges along every wall segment separating two different rooms. Algorithm 1
runs on this graph, and the walked subset of wall lines is carved into
1-unit-wide corridors (wall | corridor | wall replaces the shared wall).

Rooms open onto the carved corridors through framed doorways; rooms whose
walls carry no corridor stay sealed (they still shape the world). One lattice
step = S world cells, so a coarse cell is S units across.
"""

from collections import deque

import numpy as np

S = 3          # world cells per lattice step (rooms ~2 coarse cells across
               # => living-room-sized interiors of ~3-4 units)
PAD = 2        # solid border around the building


def lat_center(c, off):
    """World-space center of the corridor cell at lattice node c."""
    return (off + c[0] * S + 0.5, off + c[1] * S + 0.5)


def _clamp_mid(c0, c1, lo, hi):
    """Middle of the overlap of [c0, c1] and [lo, hi]; None if empty."""
    a, b = max(c0, lo), min(c1, hi)
    if a > b:
        return None
    return (a + b) // 2


# ---------------------------------------------------------------------------
# Stage 1: tight rectangle packing
# ---------------------------------------------------------------------------

def _has_hole(grid):
    empty = grid == -1
    h, w = grid.shape
    seen = np.zeros((h, w), bool)
    dq = deque()
    for x in range(w):
        for y in (0, h - 1):
            if empty[y, x] and not seen[y, x]:
                seen[y, x] = True
                dq.append((x, y))
    for y in range(h):
        for x in (0, w - 1):
            if empty[y, x] and not seen[y, x]:
                seen[y, x] = True
                dq.append((x, y))
    while dq:
        x, y = dq.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h and empty[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                dq.append((nx, ny))
    return bool((empty & ~seen).any())


def pack_rooms(rng, cw, ch, target, tries=400):
    """Pack `target` rectangles tightly on a cw x ch coarse grid; sides are
    mostly 2 coarse cells (living-room scale), occasionally 3.
    Returns (grid of room ids, [(x0, y0, w, h), ...])."""
    def side():
        return rng.choice((2, 2, 2, 3))

    grid = np.full((ch, cw), -1, int)
    w, h = side(), side()
    x0, y0 = (cw - w) // 2, (ch - h) // 2
    grid[y0:y0 + h, x0:x0 + w] = 0
    rects = [(x0, y0, w, h)]
    for _ in range(tries):
        if len(rects) >= target:
            break
        w, h = side(), side()
        x0 = rng.randint(0, cw - w)
        y0 = rng.randint(0, ch - h)
        if (grid[y0:y0 + h, x0:x0 + w] != -1).any():
            continue
        contact = 0
        for x in range(x0, x0 + w):
            for y in (y0 - 1, y0 + h):
                if 0 <= y < ch and grid[y, x] != -1:
                    contact += 1
        for y in range(y0, y0 + h):
            for x in (x0 - 1, x0 + w):
                if 0 <= x < cw and grid[y, x] != -1:
                    contact += 1
        if contact < 2:
            continue
        rid = len(rects)
        grid[y0:y0 + h, x0:x0 + w] = rid
        if _has_hole(grid):
            grid[y0:y0 + h, x0:x0 + w] = -1
            continue
        rects.append((x0, y0, w, h))
    return grid, rects


# ---------------------------------------------------------------------------
# Stage 2: wall-line graph (corridor candidates)
# ---------------------------------------------------------------------------

def wall_graph(grid):
    """Lattice graph of wall lines: segments separating two different rooms
    (internal walls) plus segments separating a room from the outside (the
    building outline, i.e. perimeter corridors). The outline closes the big
    cycle that internal walls chord, so the graph is loop-rich enough for
    returnable paths."""
    ch, cw = grid.shape
    adj = {}

    def cell(i, j):
        return grid[j, i] if 0 <= i < cw and 0 <= j < ch else -1

    def add(a, b):
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    for j in range(ch + 1):           # horizontal segments along y = j
        for i in range(cw):
            a, b = cell(i, j - 1), cell(i, j)
            if a != b and (a != -1 or b != -1):
                add((i, j), (i + 1, j))
    for i in range(cw + 1):           # vertical segments along x = i
        for j in range(ch):
            a, b = cell(i - 1, j), cell(i, j)
            if a != b and (a != -1 or b != -1):
                add((i, j), (i, j + 1))
    if not adj:
        return {}
    # largest connected component
    best = set()
    seen = set()
    for start in adj:
        if start in seen:
            continue
        comp = {start}
        dq = deque([start])
        while dq:
            c = dq.popleft()
            for nb in adj[c]:
                if nb not in comp:
                    comp.add(nb)
                    dq.append(nb)
        seen |= comp
        if len(comp) > len(best):
            best = comp
    return {c: sorted(v & best) for c, v in adj.items() if c in best}


def make_plan(rng, cw, ch, n_rooms):
    grid, rects = pack_rooms(rng, cw, ch, n_rooms)
    adj = wall_graph(grid)
    if len(rects) < 4 or len(adj) < 10:
        return None
    return dict(grid=grid, rects=rects, adj=adj, cw=cw, ch=ch)


# ---------------------------------------------------------------------------
# Stage 3: carve the walked wall lines into corridors, open rooms onto them
# ---------------------------------------------------------------------------

def build_world(plan, path, rng):
    """Carve corridors along the walked wall lines, inset room interiors, and
    punch doorways. Returns everything generate.py needs."""
    grid, rects = plan["grid"], plan["rects"]
    ch, cw = grid.shape
    off = PAD
    W = cw * S + 2 * PAD + 1
    H = ch * S + 2 * PAD + 1
    open_ = np.zeros((H, W), bool)
    room_mask = np.zeros((H, W), bool)

    # wall ownership: each room claims the walls of its coarse region (first
    # claim wins on shared lines), and one random side becomes an accent wall
    # (owner + 1000) — per-room colours make walls distinguishable
    owner = np.full((H, W), -1, np.int32)
    for rid, (x0, y0, w, h) in enumerate(rects):
        ra = owner[off + y0 * S:off + (y0 + h) * S + 1,
                   off + x0 * S:off + (x0 + w) * S + 1]
        ra[ra == -1] = rid
    for rid, (x0, y0, w, h) in enumerate(rects):
        if rng.random() < 0.8:
            side = rng.choice(("T", "B", "L", "R"))
            if side == "T":
                owner[off + y0 * S, off + x0 * S:off + (x0 + w) * S + 1] = 1000 + rid
            elif side == "B":
                owner[off + (y0 + h) * S, off + x0 * S:off + (x0 + w) * S + 1] = 1000 + rid
            elif side == "L":
                owner[off + y0 * S:off + (y0 + h) * S + 1, off + x0 * S] = 1000 + rid
            else:
                owner[off + y0 * S:off + (y0 + h) * S + 1, off + (x0 + w) * S] = 1000 + rid

    routes = [path["out"], path["ret"], *path["chords"]]
    carved = set()
    walked = set()
    for ri, r in enumerate(routes):
        for a, b in zip(r[:-1], r[1:]):
            e = frozenset((a, b))
            carved.add(e)
            if ri < 2:
                walked.add(e)

    # corridor strips along carved lattice edges
    light_cells = []
    for e in carved:
        a, b = sorted(e)
        dx, dy = b[0] - a[0], b[1] - a[1]
        for t in range(S + 1):
            open_[off + a[1] * S + t * dy, off + a[0] * S + t * dx] = True
        light_cells.append((off + a[0] * S + dx * (S // 2),
                            off + a[1] * S + dy * (S // 2)))

    def seg_carved(vert, line, k):
        """Is the unit wall segment on line `line` from k to k+1 carved?"""
        if vert:
            return frozenset(((line, k), (line, k + 1))) in carved
        return frozenset(((k, line), (k + 1, line))) in carved

    def side_spans(vert, line, k0, k1, edges):
        return [k for k in range(k0, k1) if seg_carved(vert, line, k) and
                (not edges or frozenset((((line, k), (line, k + 1)) if vert else
                                         ((k, line), (k + 1, line)))) in edges)]

    rooms_out = []
    passages = set()
    all_doors = []
    for (x0, y0, w, h) in rects:
        sides = {
            "L": bool(side_spans(True, x0, y0, y0 + h, None)),
            "R": bool(side_spans(True, x0 + w, y0, y0 + h, None)),
            "T": bool(side_spans(False, y0, x0, x0 + w, None)),
            "B": bool(side_spans(False, y0 + h, x0, x0 + w, None)),
        }
        ins = {s: 2 if sides[s] else 1 for s in sides}
        xa = off + x0 * S + ins["L"]
        xb = off + (x0 + w) * S - ins["R"]
        ya = off + y0 * S + ins["T"]
        yb = off + (y0 + h) * S - ins["B"]
        open_[ya:yb + 1, xa:xb + 1] = True
        room_mask[ya:yb + 1, xa:xb + 1] = True

        # doorway candidates per corridor side, walked corridors first
        cands = []
        for sname, vert, line, k0, k1, nrm in (
                ("L", True, x0, y0, y0 + h, (1, 0)),
                ("R", True, x0 + w, y0, y0 + h, (-1, 0)),
                ("T", False, y0, x0, x0 + w, (0, 1)),
                ("B", False, y0 + h, x0, x0 + w, (0, -1))):
            for pref, edges in ((0, walked), (1, None)):
                for k in side_spans(vert, line, k0, k1, edges):
                    if vert:
                        px = off + line * S + nrm[0]
                        py = _clamp_mid(off + k * S + 1, off + (k + 1) * S - 1, ya, yb)
                    else:
                        py = off + line * S + nrm[1]
                        px = _clamp_mid(off + k * S + 1, off + (k + 1) * S - 1, xa, xb)
                    if px is None or py is None:
                        continue
                    cands.append((pref, rng.random(), px, py, nrm))
        seen_pos = set()
        cands = [c for c in sorted(cands)
                 if not (c[2], c[3]) in seen_pos and not seen_pos.add((c[2], c[3]))]
        doors = []
        n_doors = 1 if rng.random() < 0.75 else 2
        for pref, _, px, py, nrm in cands[:n_doors]:
            open_[py, px] = True
            passages.add((px, py))
            doors.append({"x": px + 0.5, "y": py + 0.5, "nx": nrm[0], "ny": nrm[1],
                          "depth": 1.04, "kind": "side"})
        all_doors += doors

        # ceiling panel points spread over the interior
        pw, ph = xb - xa + 1, yb - ya + 1
        nx = max(1, round(pw / 4.5))
        ny = max(1, round(ph / 4.5))
        panels = [(xa + (ix + 0.5) * pw / nx, ya + (iy + 0.5) * ph / ny)
                  for ix in range(nx) for iy in range(ny)]

        rooms_out.append({"rect": (xa, ya, xb, yb), "kind": "side",
                          "center": ((xa + xb + 1) / 2, (ya + yb + 1) / 2),
                          "doors": doors, "sealed": not doors,
                          "panels": panels})

    panel_pts = [p for r in rooms_out if not r["sealed"] for p in r["panels"]]
    return dict(open=open_, room_mask=room_mask, passages=passages,
                rooms=rooms_out, doors=all_doors, light_cells=light_cells,
                panel_pts=panel_pts, off=off, size=(W, H), owner=owner,
                n_doored=sum(1 for r in rooms_out if not r["sealed"]))


# ---------------------------------------------------------------------------
# Stage 4: room crossings — the walker cuts through a bordering room, in one
# door and out another, instead of following the corridor around it
# ---------------------------------------------------------------------------

def _edge_room_sides(grid, a, b):
    """[(room_id, normal-into-room), ...] for the two coarse cells flanking
    lattice edge a-b."""
    ch, cw = grid.shape

    def cell(i, j):
        return grid[j, i] if 0 <= i < cw and 0 <= j < ch else -1

    out = []
    if a[0] == b[0]:                                   # vertical edge, line x=a[0]
        k = min(a[1], b[1])
        out.append((cell(a[0] - 1, k), (-1, 0)))
        out.append((cell(a[0], k), (1, 0)))
    else:                                              # horizontal, line y=a[1]
        k = min(a[0], b[0])
        out.append((cell(k, a[1] - 1), (0, -1)))
        out.append((cell(k, a[1]), (0, 1)))
    return [(int(rid), nrm) for rid, nrm in out if rid >= 0]


def _door_for_edge(built, rid, edge, nrm):
    """Find or punch a doorway for room `rid` on the wall segment of `edge`.
    Returns the door dict, or None if the geometry doesn't allow one."""
    off = built["off"]
    room = built["rooms"][rid]
    xa, ya, xb, yb = room["rect"]
    a, b = edge
    if a[0] == b[0]:                                   # vertical line
        k = min(a[1], b[1])
        px = off + a[0] * S + nrm[0]
        py = _clamp_mid(off + k * S + 1, off + (k + 1) * S - 1, ya, yb)
    else:
        k = min(a[0], b[0])
        py = off + a[1] * S + nrm[1]
        px = _clamp_mid(off + k * S + 1, off + (k + 1) * S - 1, xa, xb)
    if px is None or py is None:
        return None
    for d in room["doors"]:                            # reuse an existing door
        if int(d["x"]) == px and int(d["y"]) == py:
            return d
    door = {"x": px + 0.5, "y": py + 0.5, "nx": nrm[0], "ny": nrm[1],
            "depth": 1.04, "kind": "side"}
    built["open"][py, px] = True
    built["passages"].add((px, py))
    room["doors"].append(door)
    room["sealed"] = False
    built["doors"].append(door)
    return door


def _clamp_pt(p, rect, inset=0.7):
    xa, ya, xb, yb = rect
    return (min(max(p[0], xa + inset), xb + 1 - inset),
            min(max(p[1], ya + inset), yb + 1 - inset))


def plan_crossings(plan, built, path, rng, prob=0.7, max_per_route=3,
                   visit_prob=0.4, max_visits=2):
    """Room diversions for each walked route. Two kinds:

      * cross — where the route follows a room boundary for >= 2 wall
        segments: in a door near the start of the span, diagonally through the
        room's middle (bowed toward the center, jittered), out a door near the
        end;
      * visit — enter a bordering room through one door, walk well inside,
        stop and look around, then turn and leave through the same door.

    Mutates `built` (doors, passages, open cells, per-room furniture
    keep-outs). Returns {"out": [diversion, ...], "ret": [...]}."""
    grid = plan["grid"]
    out = {"out": [], "ret": []}
    for rname in ("out", "ret"):
        route = path[rname]
        edges = list(zip(route[:-1], route[1:]))
        borders = {}
        for ei, e in enumerate(edges):
            for rid, nrm in _edge_room_sides(grid, *e):
                borders.setdefault(rid, []).append((ei, nrm))
        cands = []
        for rid, lst in borders.items():
            run = [lst[0]]
            runs = []
            for v in lst[1:]:
                if v[0] == run[-1][0] + 1:
                    run.append(v)
                else:
                    runs.append(run)
                    run = [v]
            runs.append(run)
            for r_ in runs:
                cands.append((rid, r_[:4]))            # cap the bypass length
        rng.shuffle(cands)
        used = set()
        spans = []
        n_cross = n_visit = 0

        def overlaps(i0, i1):
            return any(not (i1 < c0 or i0 > c1) for c0, c1 in spans)

        for rid, run in cands:
            room = built["rooms"][rid]
            center = room["center"]
            # -------- crossing: two doors, diagonal through the middle -------
            if len(run) >= 2 and n_cross < max_per_route and rid not in used \
                    and rng.random() < prob:
                (i0, n0), (i1, n1) = run[0], run[-1]
                if overlaps(i0, i1):
                    continue
                d_in = _door_for_edge(built, rid, edges[i0], n0)
                d_out = _door_for_edge(built, rid, edges[i1], n1)
                if d_in is None or d_out is None or d_in is d_out:
                    continue
                if (d_in["x"] - d_out["x"]) ** 2 + (d_in["y"] - d_out["y"]) ** 2 < 2.0 ** 2:
                    continue
                d_in["leaf_flat"] = d_out["leaf_flat"] = True   # keep walked doors clear
                p_in = (d_in["x"] + d_in["nx"] * 0.9, d_in["y"] + d_in["ny"] * 0.9)
                p_out = (d_out["x"] + d_out["nx"] * 0.9, d_out["y"] + d_out["ny"] * 0.9)
                base = ((p_in[0] + p_out[0]) / 2, (p_in[1] + p_out[1]) / 2)
                t = rng.uniform(0.45, 0.85)
                mid = _clamp_pt((base[0] + (center[0] - base[0]) * t + rng.uniform(-0.3, 0.3),
                                 base[1] + (center[1] - base[1]) * t + rng.uniform(-0.3, 0.3)),
                                room["rect"])
                used.add(rid)
                spans.append((i0, i1))
                n_cross += 1
                room.setdefault("cross_segs", []).extend([(p_in, mid), (mid, p_out)])
                out[rname].append(dict(kind="cross", room=rid, i0=i0, i1=i1,
                                       d_in=d_in, d_out=d_out, mid=mid))
                continue
            # -------- visit: walk in, look around, walk back out -------------
            if n_visit < max_visits and rid not in used and rng.random() < visit_prob:
                ei, nrm = run[len(run) // 2]
                if overlaps(ei, ei):
                    continue
                door = _door_for_edge(built, rid, edges[ei], nrm)
                if door is None:
                    continue
                pas = (door["x"], door["y"])
                t = rng.uniform(0.65, 0.9)
                deep = _clamp_pt((pas[0] + (center[0] - pas[0]) * t + rng.uniform(-0.25, 0.25),
                                  pas[1] + (center[1] - pas[1]) * t + rng.uniform(-0.25, 0.25)),
                                 room["rect"], inset=0.75)
                if (deep[0] - pas[0]) ** 2 + (deep[1] - pas[1]) ** 2 < 1.2 ** 2:
                    continue
                door["leaf_flat"] = True                # keep walked doors clear
                used.add(rid)
                spans.append((ei, ei))
                n_visit += 1
                room.setdefault("cross_segs", []).append((pas, deep))
                out[rname].append(dict(kind="visit", room=rid, i0=ei, i1=ei,
                                       door=door, deep=deep,
                                       look=rng.uniform(1.2, 2.2),
                                       amp=rng.uniform(0.35, 0.55)))
    return out


def route_waypoints(route, diversions, off):
    """World waypoints for a route, taking its planned room diversions."""
    W = [lat_center(c, off) for c in route]
    if not diversions:
        return W
    out = []
    pos = 0
    for c in sorted(diversions, key=lambda c: c["i0"]):
        i0, i1 = c["i0"], c["i1"]
        out += W[pos:i0 + 1]
        if c["kind"] == "cross":
            din, dout = c["d_in"], c["d_out"]
            out += [(din["x"] - din["nx"], din["y"] - din["ny"]),
                    (din["x"], din["y"]),
                    (din["x"] + din["nx"] * 0.9, din["y"] + din["ny"] * 0.9),
                    c["mid"],
                    (dout["x"] + dout["nx"] * 0.9, dout["y"] + dout["ny"] * 0.9),
                    (dout["x"], dout["y"]),
                    (dout["x"] - dout["nx"], dout["y"] - dout["ny"])]
        else:                                          # visit: in and back out
            d = c["door"]
            cor = (d["x"] - d["nx"], d["y"] - d["ny"])
            pas = (d["x"], d["y"])
            # straighten through the doorway before angling toward the deep
            # point, so the camera never cuts the jamb or the open leaf
            ins = (d["x"] + d["nx"] * 0.85, d["y"] + d["ny"] * 0.85)
            out += [cor, pas, ins, c["deep"], ins, pas, cor]
        pos = i1 + 1
    out += W[pos:]
    dedup = [out[0]]
    for p in out[1:]:
        if (p[0] - dedup[-1][0]) ** 2 + (p[1] - dedup[-1][1]) ** 2 > 0.09:
            dedup.append(p)
    return dedup
