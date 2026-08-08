"""Procedural 3D meshes for landmark props (GL renderer).

Each prop type is assembled from primitive meshes (cylinders, boxes,
noise-displaced icospheres). All instances of a scene are baked — rotated,
translated, tinted — into one static vertex buffer, so the whole prop set is
a single draw call. Mesh dict: {"v": (N,3) pos, "n": (N,3) normal,
"c": (N,3) color 0..1, "e": (N,) emissive}. Ground plane is world (x, y);
height is GL +Y, so a world point (x, y) sits at GL (x, h, y).
"""

import numpy as np


def _rgb(c):
    return np.asarray(c, np.float64) / 255.0


def _mesh(v, n, c, e=None):
    v = np.asarray(v, np.float64).reshape(-1, 3)
    return {"v": v, "n": np.asarray(n, np.float64).reshape(-1, 3),
            "c": np.asarray(c, np.float64).reshape(-1, 3),
            "e": np.zeros(len(v)) if e is None else np.full(len(v), float(e))}


def merge(*meshes):
    meshes = [m for m in meshes if m is not None and len(m["v"])]
    if not meshes:
        return _mesh(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)))
    return {k: np.concatenate([m[k] for m in meshes]) for k in ("v", "n", "c", "e")}


def transform(m, yaw=0.0, dx=0.0, dy=0.0, dz=0.0, tint=1.0):
    ca, sa = np.cos(yaw), np.sin(yaw)
    r = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
    return {"v": m["v"] @ r.T + np.array([dx, dy, dz]),
            "n": m["n"] @ r.T,
            "c": np.clip(m["c"] * tint, 0.0, 1.0),
            "e": m["e"].copy()}


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def cylinder(rb, rt, h, color, seg=20, y0=0.0, cap_top=True, cap_bottom=False,
             top_color=None, emissive=0.0):
    col = _rgb(color)
    tcol = col if top_color is None else _rgb(top_color)
    ang = np.linspace(0.0, 2 * np.pi, seg + 1)
    cs, sn = np.cos(ang), np.sin(ang)
    slope = (rb - rt) / max(h, 1e-6)
    V, N, C = [], [], []
    for i in range(seg):
        b0 = (rb * cs[i], y0, rb * sn[i])
        b1 = (rb * cs[i + 1], y0, rb * sn[i + 1])
        t0 = (rt * cs[i], y0 + h, rt * sn[i])
        t1 = (rt * cs[i + 1], y0 + h, rt * sn[i + 1])
        n0 = _norm((cs[i], slope, sn[i]))
        n1 = _norm((cs[i + 1], slope, sn[i + 1]))
        V += [b0, b1, t1, b0, t1, t0]
        N += [n0, n1, n1, n0, n1, n0]
        C += [col] * 6
    if cap_top and rt > 1e-4:
        for i in range(seg):
            V += [(0, y0 + h, 0), (rt * cs[i], y0 + h, rt * sn[i]),
                  (rt * cs[i + 1], y0 + h, rt * sn[i + 1])]
            N += [(0, 1, 0)] * 3
            C += [tcol] * 3
    if cap_bottom and rb > 1e-4:
        for i in range(seg):
            V += [(0, y0, 0), (rb * cs[i + 1], y0, rb * sn[i + 1]),
                  (rb * cs[i], y0, rb * sn[i])]
            N += [(0, -1, 0)] * 3
            C += [col * 0.8] * 3
    return _mesh(V, N, C, emissive)


def _norm(v):
    v = np.asarray(v, np.float64)
    return tuple(v / max(np.linalg.norm(v), 1e-9))


def box(sx, sy, sz, color, y0=0.0, top_f=1.14, emissive=0.0):
    col = _rgb(color)
    hx, hz = sx / 2, sz / 2
    y1 = y0 + sy
    V, N, C = [], [], []

    def face(p0, p1, p2, p3, n, f):
        V.extend([p0, p1, p2, p0, p2, p3])
        N.extend([n] * 6)
        C.extend([np.clip(col * f, 0, 1)] * 6)

    face((-hx, y0, -hz), (hx, y0, -hz), (hx, y1, -hz), (-hx, y1, -hz), (0, 0, -1), 1.0)
    face((hx, y0, hz), (-hx, y0, hz), (-hx, y1, hz), (hx, y1, hz), (0, 0, 1), 0.96)
    face((hx, y0, -hz), (hx, y0, hz), (hx, y1, hz), (hx, y1, -hz), (1, 0, 0), 1.04)
    face((-hx, y0, hz), (-hx, y0, -hz), (-hx, y1, -hz), (-hx, y1, hz), (-1, 0, 0), 0.92)
    face((-hx, y1, -hz), (hx, y1, -hz), (hx, y1, hz), (-hx, y1, hz), (0, 1, 0), top_f)
    return _mesh(V, N, C, emissive)


_ICO = None


def _icosphere(sub=1):
    global _ICO
    if _ICO is None:
        t = (1 + 5 ** 0.5) / 2
        vs = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
              (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
              (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
        fs = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
              (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
              (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
              (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
        vs = [np.array(v) / np.linalg.norm(v) for v in vs]
        for _ in range(sub):
            nf, cache = [], {}

            def mid(a, b):
                key = (min(a, b), max(a, b))
                if key not in cache:
                    m = vs[a] + vs[b]
                    vs.append(m / np.linalg.norm(m))
                    cache[key] = len(vs) - 1
                return cache[key]

            for a, b, c in fs:
                ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
                nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
            fs = nf
        _ICO = (np.array(vs), fs)
    return _ICO


def blob(r, color, cx, cy, cz, squash=0.82, seed=0.0, emissive=0.0):
    vs, fs = _icosphere(1)
    disp = 1.0 + 0.16 * np.sin(vs[:, 0] * 7.1 + vs[:, 1] * 9.3 + vs[:, 2] * 5.7 + seed)
    pts = vs * disp[:, None] * np.array([r, r * squash, r]) + np.array([cx, cy, cz])
    col = _rgb(color)
    V, N, C = [], [], []
    for a, b, c in fs:
        V += [pts[a], pts[b], pts[c]]
        N += [vs[a], vs[b], vs[c]]
        C += [col] * 3
    return _mesh(V, N, C, emissive)


# ---------------------------------------------------------------------------
# Prop types (sizes match the old billboard sprites; heights in world units)
# ---------------------------------------------------------------------------

def _barrel(rng):
    c = (88, 122, 204)
    m = merge(cylinder(0.15, 0.15, 0.46, c, cap_top=True, top_color=(52, 72, 130)),
              cylinder(0.156, 0.156, 0.045, (48, 66, 118), y0=0.10),
              cylinder(0.156, 0.156, 0.045, (48, 66, 118), y0=0.30))
    return m, 0.22, None


def _crate(rng):
    return box(0.34, 0.38, 0.34, (172, 126, 64)), 0.26, None


def _boxes(rng):
    m = merge(box(0.32, 0.24, 0.30, (186, 148, 94)),
              transform(box(0.24, 0.22, 0.22, (160, 120, 72)),
                        yaw=0.3, dx=0.02, dy=0.24, dz=-0.01))
    return m, 0.26, None


def _plant(rng):
    m = merge(cylinder(0.085, 0.115, 0.17, (128, 84, 54), cap_top=True, top_color=(80, 52, 34)),
              blob(0.15, (56, 128, 62), 0.0, 0.34, 0.0, seed=rng.uniform(0, 9)),
              blob(0.12, (82, 156, 70), 0.07, 0.44, 0.04, seed=rng.uniform(0, 9)),
              blob(0.11, (44, 108, 52), -0.07, 0.42, -0.05, seed=rng.uniform(0, 9)))
    return m, 0.20, None


def _cone(rng):
    m = merge(cylinder(0.125, 0.02, 0.32, (236, 116, 32), y0=0.02),
              cylinder(0.085, 0.075, 0.05, (238, 234, 226), y0=0.14),
              box(0.26, 0.025, 0.26, (196, 92, 24)))
    return m, 0.17, None


def _face_z(m, dy):
    """Rotate a Y-axis cylinder to face along Z and lift it to height dy."""
    r = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    return {"v": m["v"] @ r.T + np.array([0.0, dy, 0.0]),
            "n": m["n"] @ r.T, "c": m["c"], "e": m["e"]}


def _sign(rng):
    pole = cylinder(0.016, 0.016, 0.62, (110, 112, 116))
    back = _face_z(cylinder(0.17, 0.17, 0.020, (236, 240, 240), seg=24,
                            cap_top=True, cap_bottom=True), 0.64)
    front = _face_z(cylinder(0.14, 0.14, 0.030, (44, 172, 162), seg=24,
                             cap_top=True, cap_bottom=True), 0.64)
    return merge(pole, back, front), 0.12, None


def _extinguisher(rng):
    m = merge(cylinder(0.055, 0.055, 0.28, (204, 42, 38), y0=0.02,
                       cap_top=True, top_color=(150, 30, 28)),
              box(0.05, 0.07, 0.05, (52, 52, 56), y0=0.30),
              cylinder(0.012, 0.012, 0.10, (52, 52, 56), y0=0.24))
    return m, 0.10, None


def _bin(rng):
    m = merge(cylinder(0.105, 0.135, 0.40, (108, 112, 118), y0=0.01),
              cylinder(0.142, 0.142, 0.035, (70, 72, 76), y0=0.40,
                       cap_top=True, top_color=(60, 62, 66)))
    return m, 0.18, None


def _lamp(rng):
    body = merge(cylinder(0.115, 0.115, 0.022, (56, 54, 52), cap_top=True),
                 cylinder(0.013, 0.013, 0.66, (70, 66, 62), y0=0.02))
    shade = cylinder(0.155, 0.10, 0.20, (240, 214, 152), y0=0.66,
                     cap_top=True, top_color=(240, 214, 152), emissive=0.85)
    light = (0.0, 0.74, 0.0)   # local offset of the light this prop emits
    return merge(body, shade), 0.16, light


# ---------------------------------------------------------------------------
# Room furniture (same signature as landmark builders: rng -> (mesh, shadow_r,
# light-or-None); every piece faces local +X with its back at -X, so a yaw of
# -atan2(ny, nx) puts its back against the wall whose normal is (nx, ny))
# ---------------------------------------------------------------------------

def _d(c, f):
    return tuple(int(v * f) for v in c)


def _table(rng):
    c = (170, 126, 78)
    m = [box(0.62, 0.055, 0.95, c, y0=0.40)]
    for dx in (-0.26, 0.26):
        for dz in (-0.42, 0.42):
            m.append(transform(box(0.05, 0.40, 0.05, _d(c, 0.7)), dx=dx, dz=dz))
    return merge(*m), 0.45, None


def _desk(rng):
    base, _, _ = _table(rng)
    mon = merge(
        transform(box(0.24, 0.02, 0.16, (70, 72, 76)), dx=-0.06, dy=0.455),
        transform(box(0.06, 0.10, 0.05, (70, 72, 76)), dx=-0.08, dy=0.475),
        transform(box(0.045, 0.30, 0.50, (26, 28, 32), top_f=1.0), dx=-0.10, dy=0.565))
    return merge(base, mon), 0.45, None


def _chair(rng):
    c = (128, 92, 60)
    m = [box(0.42, 0.05, 0.42, c, y0=0.25)]
    for dx in (-0.17, 0.17):
        for dz in (-0.17, 0.17):
            m.append(transform(box(0.04, 0.25, 0.04, _d(c, 0.7)), dx=dx, dz=dz))
    m.append(transform(box(0.05, 0.45, 0.42, _d(c, 0.9)), dx=-0.185, dy=0.30))
    return merge(*m), 0.30, None


def _shelf(rng):
    c = (124, 92, 60)
    m = [box(0.30, 1.05, 0.92, c)]
    for h in (0.34, 0.64, 0.92):
        m.append(transform(box(0.02, 0.03, 0.90, _d(c, 1.25)), dx=0.15, dy=h))
    books = [(168, 60, 52), (52, 96, 168), (44, 128, 84),
             (216, 150, 40), (94, 70, 148), (182, 180, 172)]
    for level in (0.37, 0.67):
        z = -0.40
        while z < 0.32:
            bw = 0.05 + 0.05 * rng.random()
            if rng.random() < 0.78:
                bh = 0.15 + 0.07 * rng.random()
                col = books[int(rng.random() * len(books))]
                m.append(transform(box(0.06, bh, bw, col), dx=0.115,
                                   dy=level, dz=z + bw / 2))
            z += bw + 0.015
    return merge(*m), 0.45, None


def _sofa(rng):
    cs = [(152, 64, 66), (84, 104, 148), (122, 118, 108)]
    c = cs[int(rng.random() * len(cs))]
    m = [box(0.55, 0.32, 1.15, c, y0=0.04),
         transform(box(0.16, 0.60, 1.15, _d(c, 0.85)), dx=-0.20),
         transform(box(0.50, 0.46, 0.14, _d(c, 0.92)), dz=0.505),
         transform(box(0.50, 0.46, 0.14, _d(c, 0.92)), dz=-0.505),
         transform(box(0.50, 0.02, 0.02, _d(c, 0.7)), dy=0.36)]
    return merge(*m), 0.62, None


def _bed(rng):
    m = [box(0.98, 0.20, 1.80, (112, 82, 52)),
         transform(box(0.92, 0.13, 1.70, (228, 224, 212)), dy=0.20),
         transform(box(0.06, 0.52, 0.98, (96, 68, 42)), dx=-0.46),
         transform(box(0.30, 0.09, 0.62, (242, 242, 236)), dx=-0.26, dy=0.33),
         transform(box(0.52, 0.05, 1.72, (146, 62, 58)), dx=0.20, dy=0.33)]
    return merge(*m), 0.85, None


def _cabinet(rng):
    c = (100, 106, 114)
    m = [box(0.38, 1.05, 0.58, c),
         transform(box(0.02, 0.95, 0.015, _d(c, 0.6)), dx=0.19, dy=0.05),
         transform(box(0.03, 0.05, 0.03, (222, 222, 216)), dx=0.19, dy=0.55, dz=0.06),
         transform(box(0.03, 0.05, 0.03, (222, 222, 216)), dx=0.19, dy=0.55, dz=-0.06)]
    return merge(*m), 0.34, None


def _rug(rng):
    pairs = [((150, 60, 58), (208, 188, 158)), ((84, 104, 148), (206, 212, 220)),
             ((110, 96, 70), (196, 184, 158))]
    c1, c2 = pairs[int(rng.random() * len(pairs))]
    return merge(box(0.90, 0.012, 1.35, c1),
                 box(0.72, 0.018, 1.15, c2)), 0.0, None


def _tv(rng):
    m = [box(0.36, 0.30, 1.00, (92, 72, 52)),
         transform(box(0.05, 0.50, 0.92, (16, 18, 22), top_f=1.0, emissive=0.10),
                   dx=-0.06, dy=0.34),
         transform(box(0.06, 0.02, 0.30, (60, 60, 64)), dx=-0.06, dy=0.30)]
    return merge(*m), 0.45, None


def _coffee_table(rng):
    c = (152, 112, 72)
    m = [box(0.52, 0.045, 0.92, c, y0=0.24)]
    for dx in (-0.21, 0.21):
        for dz in (-0.40, 0.40):
            m.append(transform(box(0.05, 0.24, 0.05, _d(c, 0.7)), dx=dx, dz=dz))
    return merge(*m), 0.42, None


def _wardrobe(rng):
    c = (142, 106, 70)
    m = [box(0.55, 1.15, 1.00, c),
         transform(box(0.02, 1.05, 0.02, _d(c, 0.55)), dx=0.275, dy=0.05),
         transform(box(0.035, 0.09, 0.035, (216, 214, 208)), dx=0.275, dy=0.60, dz=0.09),
         transform(box(0.035, 0.09, 0.035, (216, 214, 208)), dx=0.275, dy=0.60, dz=-0.09)]
    return merge(*m), 0.55, None


BUILDERS = {
    "barrel": _barrel, "crate": _crate, "plant": _plant, "cone": _cone,
    "sign": _sign, "extinguisher": _extinguisher, "bin": _bin,
    "boxes": _boxes, "lamp": _lamp,
}
FURNITURE = {
    "table": _table, "desk": _desk, "chair": _chair, "shelf": _shelf,
    "sofa": _sofa, "bed": _bed, "cabinet": _cabinet, "rug": _rug,
    "tv": _tv, "coffee_table": _coffee_table, "wardrobe": _wardrobe,
}
PROP_ORDER = ["barrel", "crate", "plant", "cone", "sign", "extinguisher",
              "bin", "boxes", "lamp"]


def build_props(landmarks, rng):
    """Bake all landmark instances into one mesh.

    Returns (mesh, shadows, lights) where shadows = [(x, y, r), ...] blob
    ground shadows and lights = [(x, h, y, warm)] extra point lights from
    emissive props (lamps).
    """
    meshes, shadows, lights = [], [], []
    for lm in landmarks:
        name = PROP_ORDER[lm["t"] % len(PROP_ORDER)]
        m, shadow_r, light = BUILDERS[name](rng)
        yaw = lm["rot"] + rng.uniform(-0.15, 0.15)
        if name in ("sign", "extinguisher"):        # face along the corridor
            yaw = lm["rot"]
        tint = 0.9 + 0.2 * rng.random()
        meshes.append(transform(m, yaw=yaw, dx=lm["x"], dz=lm["y"], tint=tint))
        shadows.append((lm["x"], lm["y"], shadow_r))
        if light is not None:
            lx, lh, lz = light
            ca, sa = np.cos(yaw), np.sin(yaw)
            lights.append((lm["x"] + lx * ca + lz * sa, lh,
                           lm["y"] - lx * sa + lz * ca))
    return merge(*meshes), shadows, lights
