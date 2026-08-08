"""Abstract grid path -> concrete corridor world + smooth egocentric trajectory.

World grid: each abstract cell (x, y) maps to world cell (2x+1, 2y+1) and each
path *edge* opens the world cell between its endpoints, so corridors exist only
where the routes actually ran — two parallel corridors one grid row apart stay
separated by a wall. Corridor width = 1 world unit. Cell (x, y) center sits at
(2x + 1.5, 2y + 1.5). y grows downward (south); the renderer uses the same
convention, so "left/right" turn labels match what the camera sees.
"""

import numpy as np


def world_size(n):
    return 2 * n + 1


def cell_center(c):
    return (2 * c[0] + 1.5, 2 * c[1] + 1.5)


def carve_open(n, path):
    """Bool mask of open corridor cells (route cells + edge connectors)."""
    size = world_size(n)
    open_ = np.zeros((size, size), bool)
    for r in [path["out"], path["ret"], *path["chords"]]:
        for (cx, cy) in r:
            open_[2 * cy + 1, 2 * cx + 1] = True
        for a, b in zip(r[:-1], r[1:]):
            open_[a[1] + b[1] + 1, a[0] + b[0] + 1] = True
    return open_


def wmap_from_open(open_, n_variants, door_id, salt, door_prob=0.10,
                   variant_map=None):
    """int16 map from an open-cell mask: 0 = open, 1..n_variants = wall
    texture variants, door_id = door decal on wall cells that face an open
    cell. Variants come from `variant_map` (per-cell 1..n, e.g. per-room wall
    ownership) or a per-2x2-block stable hash."""
    h, w = open_.shape
    ys, xs = np.mgrid[0:h, 0:w]
    if variant_map is None:
        coarse = ((xs // 2) * 73856093 ^ (ys // 2) * 19349663 ^ np.int64(salt) * 2654435761) & 0x7FFFFFFF
        variant = (1 + coarse % n_variants).astype(np.int16)
    else:
        variant = variant_map.astype(np.int16)
    wmap = np.where(open_, np.int16(0), variant)

    fine = (xs * 9781 ^ ys * 6151 ^ np.int64(salt) * 911) & 0x7FFFFFFF
    adjo = np.zeros_like(open_)
    adjo[1:, :] |= open_[:-1, :]
    adjo[:-1, :] |= open_[1:, :]
    adjo[:, 1:] |= open_[:, :-1]
    adjo[:, :-1] |= open_[:, 1:]
    wmap[(~open_) & adjo & ((fine % 1000) < int(door_prob * 1000))] = door_id
    return wmap


def build_world_map(n, path, n_variants, door_id, salt, door_prob=0.10,
                    extra_open=None):
    """int16 map for the grid pipeline: carve the routes, then hash variants."""
    open_ = carve_open(n, path)
    if extra_open:
        for (x, y) in extra_open:
            open_[y, x] = True
    return wmap_from_open(open_, n_variants, door_id, salt, door_prob)


def place_landmarks_wmap(wmap, exclude_mask, rng, n_types, density=0.22,
                         min_gap=1.6, avoid_pts=(), avoid_r=1.4):
    """Corridor clutter for arbitrary layouts: scan the world map for straight
    corridor cells (open along exactly one axis) outside `exclude_mask`
    (rooms, doorway passages) and drop objects offset toward a wall.

    Returns the same record shape as place_landmarks."""
    h, w = wmap.shape
    cands = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if wmap[y, x] != 0 or exclude_mask[y, x]:
                continue
            ew = wmap[y, x - 1] == 0 and wmap[y, x + 1] == 0
            ns = wmap[y - 1, x] == 0 and wmap[y + 1, x] == 0
            if ew and not ns:
                cands.append((x, y, (1.0, 0.0)))
            elif ns and not ew:
                cands.append((x, y, (0.0, 1.0)))
    rng.shuffle(cands)
    deck = list(range(n_types))
    rng.shuffle(deck)
    di = 0
    placed = []
    for x, y, along in cands:
        if rng.random() > density:
            continue
        perp = (along[1], along[0])
        side = 1.0 if rng.random() < 0.5 else -1.0
        om = 0.30 + 0.08 * rng.random()
        aj = (rng.random() - 0.5) * 0.6
        px = x + 0.5 + perp[0] * side * om + along[0] * aj
        py = y + 0.5 + perp[1] * side * om + along[1] * aj
        if any((px - q["x"]) ** 2 + (py - q["y"]) ** 2 < min_gap ** 2 for q in placed):
            continue
        if any((px - ax) ** 2 + (py - ay) ** 2 < avoid_r ** 2 for ax, ay in avoid_pts):
            continue
        placed.append({"t": deck[di], "cell": (x, y), "x": px, "y": py,
                       "rot": float(np.arctan2(along[1], along[0])), "side": side})
        di += 1
        if di == len(deck):
            rng.shuffle(deck)
            di = 0
    return placed


def place_landmarks(path, adj, n_types, rng, density=0.80, min_gap=1.15, offset=0.30,
                    skip_world=frozenset(), avoid_pts=(), avoid_r=1.0):
    """Clutter objects on straight corridor cells, offset toward one wall with
    jitter both across and along the corridor. Cells whose world cell is in
    skip_world (rooms handle their own props) and spots within avoid_r of any
    avoid point (doorways) are left clear.

    Returns [{"t": type_idx, "cell": (x, y), "x": wx, "y": wy}, ...].
    """
    start, end = path["out"][0], path["out"][-1]
    seen, order = set(), []
    for r in [path["out"], path["ret"], *path["chords"]]:
        for c in r[1:-1]:
            if c not in seen:
                seen.add(c)
                order.append(c)

    deck = list(range(n_types))
    rng.shuffle(deck)
    di = 0
    placed = []
    for c in order:
        if c in (start, end):
            continue
        if (2 * c[0] + 1, 2 * c[1] + 1) in skip_world:
            continue
        nb = adj.get(c, [])
        if len(nb) != 2:
            continue
        d = (nb[1][0] - nb[0][0], nb[1][1] - nb[0][1])
        if abs(d[0]) == 2 and d[1] == 0:
            perp, along = (0.0, 1.0), (1.0, 0.0)   # corridor runs east-west
        elif abs(d[1]) == 2 and d[0] == 0:
            perp, along = (1.0, 0.0), (0.0, 1.0)
        else:
            continue                    # corner cell
        if rng.random() > density:
            continue
        side = 1.0 if rng.random() < 0.5 else -1.0
        om = offset + 0.08 * rng.random()          # distance from centerline
        aj = (rng.random() - 0.5) * 0.6            # slide along the corridor
        cx, cy = cell_center(c)
        px = cx + perp[0] * side * om + along[0] * aj
        py = cy + perp[1] * side * om + along[1] * aj
        if any((px - q["x"]) ** 2 + (py - q["y"]) ** 2 < min_gap ** 2 for q in placed):
            continue
        if any((px - ax) ** 2 + (py - ay) ** 2 < avoid_r ** 2 for ax, ay in avoid_pts):
            continue
        placed.append({"t": deck[di], "cell": c, "x": px, "y": py,
                       "rot": float(np.arctan2(along[1], along[0])), "side": side})
        di += 1
        if di == len(deck):
            rng.shuffle(deck)
            di = 0
    return placed


# ---------------------------------------------------------------------------
# Trajectory: rounded polyline + constant-speed walk with look-ahead heading
# ---------------------------------------------------------------------------

def route_polyline(route):
    return np.array([cell_center(c) for c in route], np.float64)


def round_corners(pts, r=0.48, k=7):
    """Replace each bend with a sampled quadratic Bezier of radius r (shrunk
    on short segments so neighbouring arcs never overlap). Wider arcs = the
    camera sweeps gently through corners instead of pivoting."""
    if len(pts) < 3:
        return np.asarray(pts, np.float64)
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        l1 = max(np.hypot(*(b - a)), 1e-9)
        l2 = max(np.hypot(*(c - b)), 1e-9)
        u1 = (b - a) / l1
        u2 = (c - b) / l2
        if abs(u1[0] * u2[1] - u1[1] * u2[0]) < 1e-9:
            out.append(b)
            continue
        re = min(r, 0.42 * l1, 0.42 * l2)
        p1, p2 = b - u1 * re, b + u2 * re
        for t in np.linspace(0.0, 1.0, k):
            out.append((1 - t) ** 2 * p1 + 2 * (1 - t) * t * b + t ** 2 * p2)
    out.append(pts[-1])
    return np.array(out)


def _angdiff(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def polyline_length(poly):
    seg = np.diff(np.asarray(poly, np.float64), axis=0)
    return float(np.hypot(seg[:, 0], seg[:, 1]).sum())


def arclengths_at(poly, points):
    """Arclength along `poly` of the closest point (true segment projection)."""
    poly = np.asarray(poly, np.float64)
    seg = np.diff(poly, axis=0)
    seglen2 = np.maximum(seg[:, 0] ** 2 + seg[:, 1] ** 2, 1e-12)
    cd = np.concatenate([[0.0], np.cumsum(np.sqrt(seglen2))])
    out = []
    for px, py in points:
        t = np.clip(((px - poly[:-1, 0]) * seg[:, 0] +
                     (py - poly[:-1, 1]) * seg[:, 1]) / seglen2, 0.0, 1.0)
        qx = poly[:-1, 0] + t * seg[:, 0]
        qy = poly[:-1, 1] + t * seg[:, 1]
        d2 = (qx - px) ** 2 + (qy - py) ** 2
        i = int(np.argmin(d2))
        out.append(float(cd[i] + t[i] * np.sqrt(seglen2[i])))
    return out


def _noise_keys(rng, total, spacing, amp):
    """Piecewise-linear noise over walked distance: keypoints every `spacing`
    units with uniform values in ±amp."""
    n = max(3, int(total / spacing) + 2)
    ks = np.linspace(0.0, max(total, 1e-6), n)
    vs = np.array([rng.uniform(-1.0, 1.0) for _ in range(n)]) * amp
    return ks, vs


def walk_poses(poly, speed, fps, lookahead=1.25, hold_start=0.4, hold_end=0.7,
               initial_yaw=None, turn_time=0.9, max_frames=9000,
               rng=None, sway=0.08, yaw_wander=0.06, speed_wander=0.25,
               slow_s=(), glances=(), stops=()):
    """Per-frame camera poses (x, y, yaw, s) walking the polyline.

    s is cumulative distance walked (drives head-bob). Heading tracks a
    look-ahead point on the path, which smooths turns the way a walker
    anticipates corners. If initial_yaw is given, the walk is prefixed with an
    in-place turn from that heading (the agent turning around at its stop point).

    If rng (random.Random) is given, human-like wander is added: smooth speed
    wobble (±speed_wander), lateral drift within the corridor (±sway world
    units), and gaze wander (±yaw_wander rad) — all keyed on distance walked,
    so runs stay deterministic per seed.

    Higher-level behaviours (all optional):
      slow_s   — arclengths (junctions) where the walker hesitates: speed dips
                 to ~45% in a gaussian window around each.
      glances  — [{"s0", "s1", "x", "y"}]: inside the window the gaze eases
                 toward world point (x, y) and back (looking at an object or
                 down a side corridor) while walking on.
      stops    — [(s_stop, duration_s)]: full stops with a small look-around.
    """
    poly = np.asarray(poly, np.float64)
    seg = np.diff(poly, axis=0)
    seglen = np.maximum(np.hypot(seg[:, 0], seg[:, 1]), 1e-9)
    cd = np.concatenate([[0.0], np.cumsum(seglen)])
    total = cd[-1]

    if rng is not None:
        ks_v, vs_v = _noise_keys(rng, total, 2.0, speed_wander)
        ks_l, vs_l = _noise_keys(rng, total, 1.6, sway)
        ks_y, vs_y = _noise_keys(rng, total, 1.3, yaw_wander)

        def wander(sv):
            return (float(np.interp(sv, ks_v, vs_v)),
                    float(np.interp(sv, ks_l, vs_l)),
                    float(np.interp(sv, ks_y, vs_y)))
    else:
        def wander(sv):
            return 0.0, 0.0, 0.0

    def point(sv):
        return np.array([np.interp(sv, cd, poly[:, 0]), np.interp(sv, cd, poly[:, 1])])

    poses = []
    p0 = point(0.0)
    tgt = point(min(lookahead, total))
    yaw = float(np.arctan2(tgt[1] - p0[1], tgt[0] - p0[0]))

    if initial_yaw is not None and abs(_angdiff(yaw, initial_yaw)) > 0.05:
        d = _angdiff(yaw, initial_yaw)
        # unhurried turn-around: duration scales with the angle (~60 deg/s)
        nf = max(int(turn_time * fps), int(abs(d) / 1.05 * fps), 1)
        for j in range(nf):
            u = (j + 1) / nf
            u = u * u * (3 - 2 * u)  # smoothstep
            poses.append((p0[0], p0[1], initial_yaw + d * u, 0.0))

    for _ in range(int(hold_start * fps)):
        poses.append((p0[0], p0[1], yaw, 0.0))

    slow_s = sorted(slow_s)
    # stops: (s, duration) or (s, duration, sweep_amp) — a big amp means a
    # deliberate look-around (slow pan both ways), small means a brief pause
    pending_stops = sorted(tuple(float(v) for v in st) for st in stops)

    def hesitation(sv):
        # gentle, broad easing near junctions/doorways — no abrupt speed dips
        f = 1.0
        for sj in slow_s:
            if abs(sv - sj) < 1.8:
                f -= 0.25 * float(np.exp(-((sv - sj) / 0.65) ** 2))
        return max(f, 0.55)

    def glance_offset(sv, p, base_yaw):
        # blend all active glances by window weight — a hard max-switch between
        # overlapping windows snaps the head by tens of degrees in one frame.
        # Targets nearly behind the walker fade out instead of being clipped:
        # the wrapped angle flips sign at 180deg, which would snap the head.
        num = den = 0.0
        for g in glances:
            if g["s0"] <= sv <= g["s1"]:
                w = float(np.sin(np.pi * (sv - g["s0"]) / max(g["s1"] - g["s0"], 1e-6)) ** 2)
                if w < 1e-4:
                    continue
                tgt = float(np.arctan2(g["y"] - p[1], g["x"] - p[0]))
                off = _angdiff(tgt, base_yaw)
                fade = min(1.0, max(0.0, (2.4 - abs(off)) / 0.5))
                if fade <= 0.0:
                    continue
                num += w * fade * float(np.clip(off, -1.15, 1.15))
                den += w * fade
        if den < 1e-6:
            return 0.0
        return (num / den) * min(1.0, den)

    s, dt, cur = 0.0, 1.0 / fps, yaw
    max_step = 1.26 * dt                            # yaw rate limit (~72 deg/s)
    last_stop_s = -1e9
    while s < total - 1e-4 and len(poses) < max_frames:
        if pending_stops and s >= pending_stops[0][0]:
            last_stop_s = pending_stops[0][0]
            st = pending_stops.pop(0)
            dur = st[1]
            amp = st[2] if len(st) > 2 else 0.09
            nf = max(int(dur * fps), 1)
            px_, py_, pyaw, _ = poses[-1] if poses else (p0[0], p0[1], cur, 0.0)
            for j in range(nf):                     # stand still
                u = (j + 1) / nf
                if amp > 0.2:                       # look around: slow pan both ways
                    sweep = amp * float(np.sin(2 * np.pi * u) * np.sin(np.pi * u))
                else:                               # brief pause: barely a drift
                    sweep = amp * float(np.sin(np.pi * u))
                sweep *= 1 if len(pending_stops) % 2 else -1
                _, _, wy_ = wander(s + 0.31 * j / fps)
                poses.append((px_, py_, pyaw + sweep + wy_, s))
            # face where we're going before setting off again (e.g. turning
            # around after walking into a room)
            tp2 = point(min(s + lookahead, total))
            if np.hypot(tp2[0] - px_, tp2[1] - py_) > 0.05:
                desired = float(np.arctan2(tp2[1] - py_, tp2[0] - px_))
                dd = _angdiff(desired, pyaw)
                if abs(dd) > 0.45:
                    nf2 = max(8, int(abs(dd) / 0.9 * fps))   # ~52 deg/s
                    for j in range(nf2):
                        u = (j + 1) / nf2
                        u = u * u * (3 - 2 * u)
                        poses.append((px_, py_, pyaw + dd * u, s))
                cur = desired
        wv, wl, wy = wander(s)
        v = speed * (1.0 + wv) * hesitation(s) \
            * min(1.0, 0.18 + s / 1.2, 0.08 + (total - s) / 1.2)
        s = min(total, s + max(v, 0.12 * speed) * dt)
        p = point(s)
        # look-ahead never crosses the next stop: you look at where you are
        # about to stop, and only turn further after stopping there
        s_look = min(s + lookahead, total,
                     pending_stops[0][0] if pending_stops else total)
        tp = point(s_look)
        dx, dy = tp[0] - p[0], tp[1] - p[1]
        if dx * dx + dy * dy > 0.03 ** 2:
            step = _angdiff(float(np.arctan2(dy, dx)), cur) * min(1.0, 10.0 * dt)
            cur = cur + float(np.clip(step, -max_step, max_step))
        yaw_out = cur + wy + glance_offset(s, p, cur)
        # fade lateral sway around stops: the heading can reverse there, which
        # would otherwise flip the sway offset to the other side in one frame
        sw_f = min(1.0, abs(s - last_stop_s),
                   abs(pending_stops[0][0] - s) if pending_stops else 1.0)
        nx, ny = -np.sin(cur), np.cos(cur)          # corridor-normal drift
        poses.append((float(p[0] + nx * wl * sw_f),
                      float(p[1] + ny * wl * sw_f), yaw_out, s))

    lx, ly = poses[-1][0], poses[-1][1]
    for _ in range(int(hold_end * fps)):
        poses.append((lx, ly, cur, s))

    # global yaw shaping: cap the turn rate (~80 deg/s) AND the turn
    # acceleration, so the camera eases into turns and — crucially — ramps
    # through zero when reversing direction instead of snapping. The braking
    # term (sqrt) slows the rate early enough to settle without overshoot;
    # near the target the sqrt curve is steeper than one accel step, so cap
    # the rate at |d|/frame there or the follower limit-cycles (~5 deg/s
    # jitter that never decays).
    max_rate = 1.4 * dt          # rad/frame  (~80 deg/s)
    max_acc = 4.5 * dt * dt      # rad/frame^2 (full reversal takes ~0.6 s)
    smoothed = []
    prev = poses[0][2]
    rate = 0.0
    for (px, py, yw, sv) in poses:
        d = _angdiff(yw, prev)
        tgt = float(np.sign(d)) * min(max_rate, float(np.sqrt(2.0 * max_acc * abs(d))), abs(d))
        rate += float(np.clip(tgt - rate, -max_acc, max_acc))
        prev += rate
        smoothed.append((px, py, prev, sv))
    return smoothed
