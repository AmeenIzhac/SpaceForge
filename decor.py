"""Wall decoration: the stuff a normal corridor actually has.

Two kinds, placed on corridor-facing wall faces (never on doors):
  * decals  — flat textured quads: posters, framed pictures, notice boards,
              whiteboards, exit signs, hazard signs (procedural textures from
              textures.py, every instance unique);
  * meshes  — wall-mounted 3D fixtures from props.py primitives: radiators,
              glowing wall sconces (which add real point lights), fire alarms.

Placement returns face records {type, x, y, nx, ny, cellw}; (x, y) is the face
center on the ground plane and (nx, ny) the wall normal pointing into the
corridor. For the raycaster fallback, decals are baked into per-cell copies of
the wall texture instead (mesh fixtures are GL-only).
"""

import numpy as np
from PIL import Image

import props
import textures

RS = getattr(Image, "Resampling", Image)

# name -> (width, height, center height, texture builder)
DECALS = {
    "poster": (0.42, 0.56, 0.70, textures.decal_poster),
    "frame": (0.48, 0.38, 0.74, textures.decal_frame),
    "notice": (0.66, 0.46, 0.74, textures.decal_notice),
    "whiteboard": (0.78, 0.48, 0.72, textures.decal_whiteboard),
    "exit": (0.34, 0.16, 1.10, textures.decal_exit),
    "hazard": (0.24, 0.24, 0.80, textures.decal_hazard),
    "window": (0.96, 0.74, 0.72, textures.decal_window),
}

_WEIGHTS = [("poster", 20), ("frame", 15), ("notice", 12), ("whiteboard", 8),
            ("exit", 12), ("hazard", 9), ("radiator", 9), ("sconce", 11),
            ("alarm", 4)]


def place_wall_decor(wmap, door_id, rng, density=0.30, min_gap=1.4,
                     skip_cells=frozenset(), exterior_cells=frozenset()):
    """rng is random.Random. density = decorations per open corridor cell.
    skip_cells: open cells (e.g. doorway passages) that get no decoration.
    exterior_cells: wall cells on the building outline — faces on them mostly
    get daylight windows instead of interior decor."""
    h, w = wmap.shape
    cands = []
    n_open = 0
    for oy in range(1, h - 1):
        for ox in range(1, w - 1):
            if wmap[oy, ox] != 0 or (ox, oy) in skip_cells:
                continue
            n_open += 1
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                v = wmap[oy + dy, ox + dx]
                if v > 0 and v != door_id:
                    cands.append((ox, oy, dx, dy))
    rng.shuffle(cands)
    names = [n for n, _ in _WEIGHTS]
    weights = [wt for _, wt in _WEIGHTS]
    target = int(n_open * density)
    placed = []
    for ox, oy, dx, dy in cands:
        if len(placed) >= target:
            break
        fx, fy = ox + 0.5 + dx * 0.5, oy + 0.5 + dy * 0.5
        if any((fx - p["x"]) ** 2 + (fy - p["y"]) ** 2 < min_gap ** 2 for p in placed):
            continue
        if (ox + dx, oy + dy) in exterior_cells and rng.random() < 0.7:
            t = "window"
        else:
            t = rng.choices(names, weights)[0]
        placed.append({"type": t,
                       "x": fx, "y": fy, "nx": -dx, "ny": -dy,
                       "cellw": (ox + dx, oy + dy)})
    return placed


# ---------------------------------------------------------------------------
# GL: decal quads (level-shader vertex layout: pos3 norm3 uv2 layer1 spec1)
# ---------------------------------------------------------------------------

def build_decals(placed, rng):
    """Returns (verts float32 array or None, [texture, ...])."""
    rows, texs = [], []
    for p in placed:
        if p["type"] not in DECALS:
            continue
        w, hgt, hc, builder = DECALS[p["type"]]
        layer = len(texs)
        texs.append(builder(rng))
        nx, ny = p["nx"], p["ny"]
        tx, ty = -ny, nx                             # horizontal axis of the face
        cx, cy = p["x"] + nx * 0.008, p["y"] + ny * 0.008
        y0, y1 = hc - hgt / 2, hc + hgt / 2
        bl = (cx - tx * w / 2, y0, cy - ty * w / 2)
        br = (cx + tx * w / 2, y0, cy + ty * w / 2)
        tr = (cx + tx * w / 2, y1, cy + ty * w / 2)
        tl = (cx - tx * w / 2, y1, cy - ty * w / 2)
        for pt, uv in ((bl, (0, 1)), (br, (1, 1)), (tr, (1, 0)),
                       (bl, (0, 1)), (tr, (1, 0)), (tl, (0, 0))):
            rows.append((*pt, nx, 0.0, ny, *uv, float(layer), 0.12))
    return (np.asarray(rows, np.float32) if rows else None), texs


# ---------------------------------------------------------------------------
# GL: wall-mounted 3D fixtures (local frame: +X points out of the wall)
# ---------------------------------------------------------------------------

def _radiator(rng):
    col = (222, 220, 214)
    body = props.transform(props.box(0.06, 0.30, 0.50, col, y0=0.10), dx=0.045)
    fins = [props.transform(props.box(0.025, 0.24, 0.045, (198, 196, 190), y0=0.13),
                            dx=0.085, dz=-0.20 + k * 0.0665) for k in range(7)]
    return props.merge(body, *fins), None


def _sconce(rng):
    bracket = props.transform(props.box(0.045, 0.07, 0.09, (96, 94, 92), y0=0.86),
                              dx=0.022)
    shade = props.transform(
        props.cylinder(0.052, 0.072, 0.13, (242, 216, 152), y0=0.86,
                       cap_top=True, emissive=0.75), dx=0.085)
    return props.merge(bracket, shade), (0.10, 1.04)


def _alarm(rng):
    return props.merge(
        props.transform(props.box(0.05, 0.13, 0.10, (200, 42, 38), y0=0.80), dx=0.026),
        props.transform(props.box(0.02, 0.05, 0.04, (240, 238, 230), y0=0.84), dx=0.055),
    ), None


_WALL_BUILDERS = {"radiator": _radiator, "sconce": _sconce, "alarm": _alarm}


def build_wall_meshes(placed, rng):
    """Returns (merged mesh or None, [(x, h, y) sconce lights])."""
    meshes, lights = [], []
    for p in placed:
        if p["type"] not in _WALL_BUILDERS:
            continue
        m, light = _WALL_BUILDERS[p["type"]](rng)
        # props.transform maps local +X to (cos -yaw, sin -yaw) on the ground
        # plane, so negate to aim +X along the wall normal
        yaw = -float(np.arctan2(p["ny"], p["nx"]))
        meshes.append(props.transform(m, yaw=yaw, dx=p["x"], dz=p["y"]))
        if light is not None:
            off, hgt = light
            lights.append((p["x"] + p["nx"] * off, hgt, p["y"] + p["ny"] * off))
    return (props.merge(*meshes) if meshes else None), lights


# ---------------------------------------------------------------------------
# Raycaster fallback: bake decals into per-cell wall texture copies
# ---------------------------------------------------------------------------

def bake_decals(atlas, wmap, placed, rng, wall_h):
    """Mutates wmap; returns the grown atlas list. One decal per wall cell."""
    atlas = list(atlas)
    done = set()
    for p in placed:
        if p["type"] not in DECALS:
            continue
        wx, wy = p["cellw"]
        if (wx, wy) in done:
            continue
        done.add((wx, wy))
        w, hgt, hc, builder = DECALS[p["type"]]
        img = builder(rng)
        base = atlas[int(wmap[wy, wx]) - 1].copy()
        t = base.shape[0]
        r0 = max(0, int(t * (1 - (hc + hgt / 2) / wall_h)))
        r1 = min(t, int(t * (1 - (hc - hgt / 2) / wall_h)))
        c0 = max(0, int(t * (0.5 - w / 2)))
        c1 = min(t, int(t * (0.5 + w / 2)))
        if r1 <= r0 or c1 <= c0:
            continue
        pim = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        pim = pim.resize((c1 - c0, r1 - r0), RS.BILINEAR)
        base[r0:r1, c0:c1] = np.asarray(pim, np.float32) / 255.0
        atlas.append(base)
        wmap[wy, wx] = len(atlas)
    return atlas
