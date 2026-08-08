"""Procedural textures, palettes, and billboard sprites. No image assets needed.

All textures are float32 RGB arrays in [0, 1] (ceiling light panels exceed 1 to
survive fog and clip to a glow). Sprites are float32 RGBA arrays; each sprite
canvas has the same aspect ratio as its world-space bounding box so shapes stay
undistorted when billboarded.
"""

import numpy as np
from PIL import Image, ImageDraw

RS = getattr(Image, "Resampling", Image)  # Pillow 9/10 compatibility

T = 128    # decal / sprite texture resolution
TEX = 512  # wall/floor/ceiling atlas resolution (photo-based)
K = TEX // 128
S = 96     # sprite texture height in pixels


def _rgb(c):
    return np.asarray(c, np.float32) / 255.0


def _noise(rng, cells=9, lo=0.92, hi=1.08, size=T):
    a = (rng.random((cells, cells)) * 255).astype(np.uint8)
    img = Image.fromarray(a, "L").resize((size, size), RS.BILINEAR)
    n = np.asarray(img, np.float32) / 255.0
    return (lo + (hi - lo) * n)[..., None]


def _base(c, size=T):
    return np.ones((size, size, 3), np.float32) * _rgb(c)


def _lum(img):
    l = img.mean(axis=2, keepdims=True)
    return l / max(float(l.mean()), 1e-4)


def _tint(img, c):
    """Recolour a photo texture: keep its detail, take the accent colour."""
    return np.clip(_lum(img) * _rgb(c), 0.0, 1.0).astype(np.float32)


def _photo(rng, names):
    import assets
    name = names[int(rng.integers(len(names)))] if not isinstance(names, str) else names
    return assets.get(name, TEX)


def _skirt(tex, c=(50, 48, 46), h=10 * K):
    tex[-h:, :] = _rgb(c)
    tex[-h - 2 * K:-h, :] *= 0.7
    return tex


# ---------------------------------------------------------------------------
# Wall / floor / ceiling textures (texture row 0 = top of wall, last row =
# floor). Photo albedos (CC0, cached one-time by assets.py) recoloured per
# accent; plain-noise fallback if an asset is missing.
# ---------------------------------------------------------------------------

def _wall_base(rng, c, names=("Plaster001", "Plaster003")):
    ph = _photo(rng, names)
    if ph is None:
        return _base(c, TEX) * _noise(rng, 9, 0.93, 1.07, size=TEX)
    return _tint(ph, c)


def plaster(rng, c, skirt=True):
    tex = _wall_base(rng, c)
    return _skirt(tex) if skirt else tex


def brick(rng, c, mortar=None):
    return _skirt(_wall_base(rng, c, ("Bricks023", "Bricks066")))


def panels(rng, c, seam=0.55, rail=True, pw=32 * K):
    tex = _wall_base(rng, c)
    xx = np.arange(TEX)
    tex[:, (xx % pw) < 2 * K] *= seam
    if rail:
        tex[84 * K:88 * K] *= 0.62
    return _skirt(tex)


def wainscot(rng, top_c, bot_c, split=74 * K):
    tex = _wall_base(rng, top_c)
    wood = _photo(rng, "Wood051")
    if wood is None:
        bot = _base(bot_c, TEX) * _noise(rng, 12, 0.94, 1.06, size=TEX)
    else:
        bot = _tint(wood, bot_c)
    tex[split:] = bot[split:]
    xx = np.arange(TEX)
    tex[split:, (xx % (24 * K)) < 2 * K] *= 0.6
    tex[split:split + 4 * K] *= 0.55
    return _skirt(tex)


def wallpaper(rng, c1, c2, sw=16 * K):
    ph = _photo(rng, ("Plaster001", "Plaster003"))
    lum = _lum(ph) if ph is not None else _noise(rng, 10, 0.9, 1.1, size=TEX)
    xx = np.arange(TEX)
    stripe = ((xx // sw) % 2) == 0
    col = np.where(stripe[None, :, None], _base(c1, TEX), _base(c2, TEX))
    return _skirt(np.clip(lum * col, 0, 1).astype(np.float32))


def blocks(rng, c):
    tex = _wall_base(rng, c, "Concrete034")
    yy, xx = np.mgrid[0:TEX, 0:TEX]
    m = ((yy % (32 * K)) < 2 * K) | (((xx + 32 * K * ((yy // (32 * K)) % 2)) % (64 * K)) < 2 * K)
    tex[m] *= 0.55
    return _skirt(tex)


def metal(rng, c):
    tex = _base(c, TEX) * _noise(rng, 6, 0.96, 1.04, size=TEX)
    tex *= (0.96 + 0.08 * rng.random((1, TEX, 1))).astype(np.float32)
    tex[:, (np.arange(TEX) % (43 * K)) < 2 * K] *= 0.6
    return _skirt(tex)


def wood_floor(rng, c=None):
    ph = _photo(rng, "WoodFloor041")
    if ph is None:
        return _base(c or (150, 110, 70), TEX) * _noise(rng, 20, 0.85, 1.15, size=TEX)
    return (ph * (0.9 + 0.2 * float(rng.random()))).astype(np.float32)


def door(rng, wall_c, door_c):
    tex = _wall_base(rng, wall_c)
    wood = _photo(rng, "Wood051")
    leaf = _tint(wood, door_c) if wood is not None else _base(door_c, TEX)
    x0, x1 = 22 * K, 106 * K
    tex[6 * K:, x0:x1] = _rgb(tuple(int(v * 0.5) for v in door_c))    # frame
    tex[10 * K:, x0 + 4 * K:x1 - 4 * K] = leaf[10 * K:, x0 + 4 * K:x1 - 4 * K]
    for py0, py1 in ((26 * K, 60 * K), (74 * K, 112 * K)):            # inset panels
        tex[py0:py1, x0 + 14 * K:x1 - 14 * K] *= 0.82
    tex[70 * K:77 * K, x1 - 16 * K:x1 - 9 * K] = _rgb((222, 208, 128))  # handle
    return tex


def tile_floor(rng, a=(200, 200, 196), b=None, grout=None):
    ph = _photo(rng, ("Tiles074", "Tiles101"))
    if ph is None:
        yy, xx = np.mgrid[0:TEX, 0:TEX]
        tex = _base(a, TEX) * _noise(rng, 16, 0.95, 1.05, size=TEX)
        tex[(xx % (64 * K) < 2 * K) | (yy % (64 * K) < 2 * K)] *= 0.7
        return tex
    return _tint(ph, a)


def carpet(rng, c):
    ph = _photo(rng, ("Carpet004", "Carpet008"))
    if ph is None:
        return _base(c, TEX) * _noise(rng, 32, 0.88, 1.12, size=TEX)
    return _tint(ph, c)


def concrete(rng, c):
    ph = _photo(rng, "Concrete034")
    if ph is None:
        return _base(c, TEX) * _noise(rng, 6, 0.9, 1.1, size=TEX)
    return _tint(ph, c)


def ceiling_tiles(rng, c=(232, 232, 228), light=True):
    """One texture spans 2x2 world units (renderer ceil_scale = 0.5)."""
    ph = _photo(rng, "OfficeCeiling005")
    if ph is None:
        yy, xx = np.mgrid[0:TEX, 0:TEX]
        tex = _base(c, TEX) * _noise(rng, 10, 0.97, 1.03, size=TEX)
        tex[(xx % (64 * K) < 2 * K) | (yy % (64 * K) < 2 * K)] *= 0.72
    else:
        tex = _tint(ph, c)
    if light:            # baked light panel (raycaster fallback only)
        yy, xx = np.mgrid[0:TEX, 0:TEX]
        m = (xx >= 8 * K) & (xx < 56 * K) & (yy >= 8 * K) & (yy < 56 * K)
        tex = tex.copy()
        tex[m] = np.array([1.9, 1.9, 1.75], np.float32)
    return tex


# ---------------------------------------------------------------------------
# Themes (domain randomisation). Each mood defines base colours; a sample draws
# a shuffled pool of wall styles with jittered colours, a random floor/ceiling,
# and jittered fog, so no two samples — and no two corridor stretches — match.
# ---------------------------------------------------------------------------

def _jit(rng, c, s=0.12):
    """Randomly scale each channel by ±s."""
    return tuple(int(np.clip(v * (1 + s * (2 * rng.random() - 1)), 8, 250)) for v in c)


MOODS = {
    "office": dict(wall=(222, 216, 205), panel=(203, 196, 183), brick=(168, 132, 104),
                   wains=(150, 112, 70), stripes=((200, 190, 170), (176, 166, 148)),
                   block=(192, 188, 180), metal=(170, 174, 180), door=(146, 96, 52),
                   floors=(("carpet", (108, 116, 128)), ("wood", (160, 118, 76)),
                           ("tile", ((200, 200, 196), (172, 174, 172)))),
                   ceil=("tiles", (232, 232, 228)),
                   fog=(0.78, 0.79, 0.81), fog_d=16.0, side=0.82, door_prob=0.13,
                   ambient=0.34, light_rgb=(1.0, 0.97, 0.90), light_every=1, light_int=0.85),
    "brick": dict(wall=(150, 146, 140), panel=(110, 106, 100), brick=(146, 74, 58),
                  wains=(96, 70, 50), stripes=((140, 134, 126), (118, 112, 104)),
                  block=(124, 120, 114), metal=(110, 114, 118), door=(72, 96, 70),
                  floors=(("concrete", (96, 94, 92)), ("wood", (110, 84, 58)),
                          ("tile", ((140, 138, 134), (118, 116, 112)))),
                  ceil=("concrete", (70, 68, 66)),
                  fog=(0.06, 0.06, 0.07), fog_d=11.0, side=0.78, door_prob=0.08,
                  ambient=0.26, light_rgb=(1.0, 0.78, 0.55), light_every=1, light_int=0.95),
    "lab": dict(wall=(228, 230, 232), panel=(214, 218, 222), brick=(196, 200, 204),
                wains=(168, 182, 196), stripes=((222, 226, 230), (200, 208, 216)),
                block=(206, 210, 214), metal=(184, 190, 198), door=(140, 150, 158),
                floors=(("tile", ((208, 208, 204), (184, 187, 186))),
                        ("concrete", (150, 152, 154)), ("carpet", (140, 148, 158))),
                ceil=("tiles", (236, 238, 240)),
                fog=(0.84, 0.86, 0.90), fog_d=18.0, side=0.85, door_prob=0.13,
                ambient=0.40, light_rgb=(0.90, 0.95, 1.0), light_every=1, light_int=0.9),
    "hotel": dict(wall=(208, 188, 160), panel=(122, 58, 54), brick=(152, 96, 72),
                  wains=(110, 70, 48), stripes=((196, 178, 150), (170, 148, 120)),
                  block=(176, 160, 138), metal=(150, 140, 124), door=(88, 58, 38),
                  floors=(("carpet", (128, 52, 48)), ("wood", (134, 96, 60)),
                          ("carpet", (96, 84, 110))),
                  ceil=("tiles", (214, 200, 182)),
                  fog=(0.38, 0.34, 0.30), fog_d=13.0, side=0.80, door_prob=0.14,
                  ambient=0.26, light_rgb=(1.0, 0.84, 0.62), light_every=1, light_int=0.72),
}

_FLOOR_SPEC = {"tile": 0.50, "wood": 0.30, "carpet": 0.06, "concrete": 0.15}


_STYLE_SETS = {
    "office": ("plaster", "panels", "wainscot", "wallpaper"),
    "brick": ("brick", "blocks", "plaster"),
    "lab": ("panels", "plaster", "blocks"),
    "hotel": ("wallpaper", "plaster", "wainscot", "panels"),
}

# distinct-but-tasteful wall colours; each room claims one, so every wall
# tells you which room it belongs to (orientation cue when turning around)
_ACCENTS = {
    "office": [(222, 216, 205), (178, 192, 172), (168, 184, 200), (200, 188, 170),
               (204, 158, 128), (196, 196, 160), (208, 210, 208), (188, 180, 200)],
    "brick": [(146, 74, 58), (110, 106, 102), (170, 140, 104), (96, 116, 92),
              (160, 96, 60), (110, 122, 134), (176, 142, 80), (120, 88, 96)],
    "lab": [(228, 230, 232), (196, 212, 224), (190, 214, 200), (204, 208, 210),
            (172, 204, 204), (198, 196, 214), (224, 218, 206), (170, 180, 190)],
    "hotel": [(208, 188, 160), (122, 58, 54), (86, 108, 84), (86, 96, 124),
              (190, 158, 100), (172, 162, 150), (186, 120, 86), (222, 210, 188)],
}


def build_palette(name, rng, n_walls=8, baked_lights=True):
    """n_walls DISTINCT wall variants for one mood: styles cycle through the
    mood's family and each variant gets its own accent colour, so different
    rooms (and the corridor walls they back onto) are tellable apart."""
    m = MOODS[name]
    styles = _STYLE_SETS[name]
    accents = _ACCENTS[name]

    def build_wall(style, col):
        if style == "plaster":
            return plaster(rng, col)
        if style == "panels":
            return panels(rng, col)
        if style == "brick":
            return brick(rng, col)
        if style == "blocks":
            return blocks(rng, col)
        if style == "wainscot":
            return wainscot(rng, col, _jit(rng, m["wains"], 0.08))
        return wallpaper(rng, col, tuple(int(v * 0.86) for v in col))

    shift = int(rng.integers(len(accents)))
    walls = [build_wall(styles[i % len(styles)],
                        _jit(rng, accents[(i + shift) % len(accents)], 0.04))
             for i in range(n_walls)]
    style = styles[0]

    def build_floor(fname, fc):
        if fname == "tile":
            return tile_floor(rng, _jit(rng, fc[0], 0.06), _jit(rng, fc[1], 0.06))
        return {"carpet": carpet, "wood": wood_floor,
                "concrete": concrete}[fname](rng, _jit(rng, fc))

    idx = int(rng.integers(len(m["floors"])))
    fname, fc = m["floors"][idx]
    floor = build_floor(fname, fc)
    j = (idx + 1 + int(rng.integers(max(len(m["floors"]) - 1, 1)))) % len(m["floors"])
    f2name, f2c = m["floors"][j]
    floor2 = build_floor(f2name, f2c)      # rooms get their own floor

    cname, cc = m["ceil"]
    if cname == "tiles":
        ceil = ceiling_tiles(rng, _jit(rng, cc, 0.05),
                             light=baked_lights and bool(rng.random() < 0.85))
    else:
        ceil = concrete(rng, _jit(rng, cc))

    return dict(name=name, walls=walls, wall_style=style,
                door=door(rng, m["wall"], _jit(rng, m["door"])),
                door_rgb=m["door"],
                floor=floor, ceil=ceil, fog_rgb=m["fog"],
                fog_d=m["fog_d"] * (0.75 + 0.5 * float(rng.random())),
                side_shade=m["side"], door_prob=m["door_prob"],
                floor_spec=_FLOOR_SPEC[fname],
                floor2=floor2, floor2_spec=_FLOOR_SPEC[f2name],
                ambient=m["ambient"],
                light_rgb=m["light_rgb"], light_every=m["light_every"],
                light_int=m["light_int"])


# ---------------------------------------------------------------------------
# Wall decal textures (posters, signs, boards) — what a normal corridor has
# ---------------------------------------------------------------------------

_PAPER = [(238, 232, 210), (222, 236, 244), (240, 224, 228), (226, 240, 222),
          (244, 238, 220)]
_INK = [(52, 96, 168), (168, 60, 52), (44, 128, 84), (216, 150, 40), (94, 70, 148)]


def _to_f(img):
    return np.asarray(img, np.float32) / 255.0


def decal_poster(rng):
    bg = _PAPER[int(rng.integers(len(_PAPER)))]
    ink = _INK[int(rng.integers(len(_INK)))]
    img = Image.new("RGB", (T, T), bg)
    d = ImageDraw.Draw(img)
    d.rectangle([3, 3, T - 4, T - 4], outline=tuple(int(v * 0.72) for v in bg), width=3)
    kind = int(rng.integers(3))
    if kind == 0:
        d.ellipse([28, 16, 100, 88], fill=ink)
    elif kind == 1:
        d.polygon([(64, 14), (18, 88), (110, 88)], fill=ink)
    else:
        d.polygon([(64, 12), (108, 52), (64, 92), (20, 52)], fill=ink)
    y = 98
    for _ in range(3):
        w = 40 + int(rng.integers(56))
        d.rectangle([14, y, 14 + w, y + 5], fill=(70, 70, 74))
        y += 10
    return _to_f(img)


def decal_frame(rng):
    img = Image.new("RGB", (T, T), (88, 62, 40))
    d = ImageDraw.Draw(img)
    d.rectangle([4, 4, T - 5, T - 5], outline=(122, 90, 60), width=3)
    c1 = _INK[int(rng.integers(len(_INK)))]
    c2 = _PAPER[int(rng.integers(len(_PAPER)))]
    for y in range(14, T - 14):
        t = (y - 14) / (T - 28)
        col = tuple(int(c1[k] * (1 - t) + c2[k] * t) for k in range(3))
        d.line([14, y, T - 15, y], fill=col)
    for _ in range(2):
        ex, ey = 24 + int(rng.integers(56)), 24 + int(rng.integers(50))
        er = 8 + int(rng.integers(12))
        d.ellipse([ex - er, ey - er // 2, ex + er, ey + er // 2],
                  fill=_INK[int(rng.integers(len(_INK)))])
    return _to_f(img)


def decal_notice(rng):
    cork = (156, 116, 74)
    img = Image.new("RGB", (T, T), cork)
    d = ImageDraw.Draw(img)
    d.rectangle([2, 2, T - 3, T - 3], outline=(110, 80, 50), width=4)
    for _ in range(5):
        pw, ph = 24 + int(rng.integers(16)), 18 + int(rng.integers(12))
        paper = Image.new("RGB", (pw, ph), _PAPER[int(rng.integers(len(_PAPER)))])
        pd = ImageDraw.Draw(paper)
        for k in range(3):
            yy = 4 + k * max(ph // 4, 4)
            pd.line([3, yy, pw - 4, yy], fill=(122, 122, 128))
        rot = paper.rotate(float(rng.uniform(-9, 9)), expand=True, fillcolor=cork)
        px = 8 + int(rng.integers(max(T - 16 - rot.width, 1)))
        py = 10 + int(rng.integers(max(T - 20 - rot.height, 1)))
        img.paste(rot, (px, py))
    return _to_f(img)


def decal_whiteboard(rng):
    img = Image.new("RGB", (T, T), (243, 244, 242))
    d = ImageDraw.Draw(img)
    d.rectangle([2, 2, T - 3, T - 3], outline=(168, 170, 172), width=5)
    for _ in range(3):
        col = _INK[int(rng.integers(len(_INK)))]
        x, y = float(rng.uniform(14, 48)), float(rng.uniform(16, 100))
        pts = [(x, y)]
        for _k in range(5):
            x = float(np.clip(x + rng.uniform(8, 26), 10, T - 12))
            y = float(np.clip(y + rng.uniform(-14, 14), 12, T - 16))
            pts.append((x, y))
        d.line(pts, fill=col, width=2, joint="curve")
    d.rectangle([30, T - 9, T - 31, T - 5], fill=(150, 152, 154))
    return _to_f(img)


def decal_exit(rng):
    # drawn 2:1 then stretched, so it reads right on the wide sign quad
    img = Image.new("RGB", (T, T // 2), (16, 96, 44))
    d = ImageDraw.Draw(img)
    d.rectangle([2, 2, T - 3, T // 2 - 3], outline=(232, 240, 234), width=3)
    for k in range(4):                       # abstract EXIT lettering
        x0 = 26 + k * 21
        d.rectangle([x0, 10, x0 + 12, 30], fill=(238, 244, 240))
    if rng.random() < 0.5:
        d.polygon([(14, 46), (34, 36), (34, 56)], fill=(238, 244, 240))
        d.rectangle([34, 42, 104, 50], fill=(238, 244, 240))
    else:
        d.polygon([(114, 46), (94, 36), (94, 56)], fill=(238, 244, 240))
        d.rectangle([24, 42, 94, 50], fill=(238, 244, 240))
    return _to_f(img.resize((T, T), RS.BILINEAR))


def decal_window(rng):
    """Bright daylight window — near-white sky so the bloom pass makes it glow."""
    img = Image.new("RGB", (T, T), (240, 244, 248))
    d = ImageDraw.Draw(img)
    for y in range(8, T - 14):
        t = (y - 8) / (T - 22)
        col = (int(168 + 60 * t), int(200 + 40 * t), int(238 + 14 * t))
        d.line([8, y, T - 9, y], fill=col)
    if rng.random() < 0.6:                      # skyline hint
        x = 12
        while x < T - 20:
            bw = 8 + int(rng.integers(14))
            bh = 10 + int(rng.integers(22))
            d.rectangle([x, T - 14 - bh, x + bw, T - 14],
                        fill=(150 + int(rng.integers(30)),) * 3)
            x += bw + 2 + int(rng.integers(6))
    fr = (226, 228, 230)
    d.rectangle([0, 0, T - 1, T - 1], outline=fr, width=8)
    d.rectangle([T // 2 - 3, 0, T // 2 + 3, T - 1], fill=fr)
    d.rectangle([0, T // 2 - 3, T - 1, T // 2 + 3], fill=fr)
    d.rectangle([0, T - 10, T - 1, T - 1], fill=(210, 212, 214))   # sill
    return _to_f(img)


def decal_hazard(rng):
    img = Image.new("RGB", (T, T), (240, 238, 232))
    d = ImageDraw.Draw(img)
    d.rectangle([2, 2, T - 3, T - 3], outline=(150, 150, 148), width=3)
    tri = [(64, 12), (118, 108), (10, 108)]
    d.polygon(tri, fill=(240, 196, 30))
    d.line(tri + [tri[0]], fill=(30, 30, 30), width=5, joint="curve")
    d.rectangle([59, 42, 69, 78], fill=(30, 30, 30))
    d.ellipse([58, 86, 70, 98], fill=(30, 30, 30))
    return _to_f(img)


# ---------------------------------------------------------------------------
# Landmark sprites (billboards). Canvas aspect == world box aspect.
# ---------------------------------------------------------------------------

def _mk(wr):
    cw = max(32, int(256 * wr))
    img = Image.new("RGBA", (cw, 256), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img), cw


def _fin(img, wr):
    img = img.resize((max(8, int(S * wr)), S), RS.LANCZOS)
    return np.asarray(img, np.float32) / 255.0


def _dk(c, f=0.55):
    return tuple(int(v * f) for v in c)


def _lt(c, f=1.3):
    return tuple(min(255, int(v * f)) for v in c)


def spr_barrel(rng, c=(88, 122, 204), wr=0.65):
    img, d, cw = _mk(wr)
    d.rounded_rectangle([2, 14, cw - 3, 250], radius=int(cw * 0.18), fill=c + (255,))
    d.rectangle([int(cw * 0.62), 26, cw - 5, 240], fill=_dk(c, 0.72) + (255,))
    for y0 in (66, 150):
        d.rectangle([2, y0, cw - 3, y0 + 14], fill=_dk(c, 0.5) + (255,))
    d.rectangle([int(cw * 0.14), 22, int(cw * 0.26), 244], fill=_lt(c) + (255,))
    d.ellipse([2, 2, cw - 3, 34], fill=_dk(c, 0.82) + (255,), outline=_dk(c, 0.5) + (255,), width=3)
    return _fin(img, wr)


def spr_crate(rng, c=(178, 130, 66), wr=1.0):
    img, d, cw = _mk(wr)
    d.rectangle([8, 8, cw - 9, 248], fill=c + (255,))
    d.rectangle([8, 8, cw - 9, 248], outline=_dk(c) + (255,), width=16)
    d.line([20, 20, cw - 21, 236], fill=_dk(c, 0.7) + (255,), width=12)
    d.line([cw - 21, 20, 20, 236], fill=_dk(c, 0.7) + (255,), width=12)
    d.rectangle([8, 8, cw - 9, 34], fill=_lt(c, 1.2) + (255,))
    return _fin(img, wr)


def spr_plant(rng, wr=0.75):
    img, d, cw = _mk(wr)
    pot = (122, 82, 54)
    d.polygon([(int(cw * 0.28), 176), (int(cw * 0.72), 176), (int(cw * 0.62), 250), (int(cw * 0.38), 250)],
              fill=pot + (255,))
    d.rectangle([int(cw * 0.25), 168, int(cw * 0.75), 182], fill=_dk(pot, 0.75) + (255,))
    greens = [(58, 132, 64), (84, 160, 70), (46, 112, 56)]
    for i in range(10):
        gx = 0.5 + 0.62 * (rng.random() - 0.5)
        gy = 0.30 + 0.42 * (rng.random() - 0.5)
        r = 0.10 + 0.10 * rng.random()
        g = greens[i % 3]
        d.ellipse([int((gx - r) * cw), int((gy - r * wr) * 256),
                   int((gx + r) * cw), int((gy + r * wr) * 256)], fill=g + (255,))
    return _fin(img, wr)


def spr_cone(rng, c=(236, 120, 36), wr=0.75):
    img, d, cw = _mk(wr)
    d.polygon([(cw // 2, 10), (int(cw * 0.08), 232), (int(cw * 0.92), 232)], fill=c + (255,))

    def half_w(y):
        return 0.5 * (0.08 + (0.92 - 0.08) * (y - 10) / 222)

    y0, y1 = 120, 158
    d.polygon([(int(cw * (0.5 - half_w(y0))), y0), (int(cw * (0.5 + half_w(y0))), y0),
               (int(cw * (0.5 + half_w(y1))), y1), (int(cw * (0.5 - half_w(y1))), y1)],
              fill=(238, 234, 226, 255))
    d.rectangle([4, 226, cw - 5, 250], fill=_dk(c, 0.6) + (255,))
    return _fin(img, wr)


def spr_sign(rng, c=(52, 178, 168), wr=0.40):
    img, d, cw = _mk(wr)
    d.rectangle([int(cw * 0.42), 40, int(cw * 0.58), 250], fill=(110, 112, 116, 255))
    d.ellipse([3, 4, cw - 4, cw - 3], fill=c + (255,), outline=(240, 244, 244, 255), width=7)
    return _fin(img, wr)


def spr_ext(rng, c=(206, 44, 40), wr=0.55):
    img, d, cw = _mk(wr)
    d.rounded_rectangle([int(cw * 0.24), 66, int(cw * 0.76), 246], radius=18, fill=c + (255,))
    d.rectangle([int(cw * 0.36), 30, int(cw * 0.64), 74], fill=(52, 52, 56, 255))
    d.line([int(cw * 0.60), 40, int(cw * 0.92), 84], fill=(52, 52, 56, 255), width=10)
    d.rectangle([int(cw * 0.32), 120, int(cw * 0.68), 168], fill=(238, 236, 230, 255))
    d.rectangle([int(cw * 0.28), 70, int(cw * 0.38), 242], fill=_lt(c, 1.25) + (255,))
    return _fin(img, wr)


def spr_bin(rng, c=(112, 116, 122), wr=0.62):
    img, d, cw = _mk(wr)
    d.polygon([(int(cw * 0.10), 34), (int(cw * 0.90), 34),
               (int(cw * 0.80), 246), (int(cw * 0.20), 246)], fill=c + (255,))
    d.rectangle([int(cw * 0.66), 44, int(cw * 0.82), 240], fill=_dk(c, 0.72) + (255,))
    d.rectangle([int(cw * 0.16), 44, int(cw * 0.30), 240], fill=_lt(c, 1.2) + (255,))
    d.rectangle([2, 16, cw - 3, 40], fill=_dk(c, 0.6) + (255,))
    d.rectangle([int(cw * 0.34), 4, int(cw * 0.66), 20], fill=_dk(c, 0.5) + (255,))
    return _fin(img, wr)


def spr_boxes(rng, c=(190, 152, 98), wr=0.85):
    img, d, cw = _mk(wr)
    c2 = _jit_px(rng, c)
    d.rectangle([4, 128, cw - 5, 250], fill=c + (255,))
    d.rectangle([4, 128, cw - 5, 250], outline=_dk(c, 0.65) + (255,), width=10)
    d.line([cw // 2, 132, cw // 2, 246], fill=_dk(c, 0.75) + (255,), width=8)
    d.rectangle([int(cw * 0.18), 18, int(cw * 0.78), 132], fill=c2 + (255,))
    d.rectangle([int(cw * 0.18), 18, int(cw * 0.78), 132], outline=_dk(c2, 0.65) + (255,), width=8)
    d.rectangle([int(cw * 0.18), 62, int(cw * 0.78), 78], fill=(226, 222, 210, 255))
    return _fin(img, wr)


def spr_lamp(rng, wr=0.42):
    img, d, cw = _mk(wr)
    shade = (232, 208, 150)
    d.rectangle([int(cw * 0.45), 60, int(cw * 0.55), 236], fill=(70, 66, 62, 255))
    d.ellipse([int(cw * 0.16), 230, int(cw * 0.84), 252], fill=(60, 58, 56, 255))
    d.polygon([(int(cw * 0.24), 66), (int(cw * 0.76), 66),
               (int(cw * 0.92), 6), (int(cw * 0.08), 6)], fill=shade + (255,))
    d.polygon([(int(cw * 0.28), 64), (int(cw * 0.72), 64),
               (int(cw * 0.60), 40), (int(cw * 0.40), 40)], fill=_lt(shade, 1.25) + (255,))
    return _fin(img, wr)


def _jit_px(rng, c, s=0.18):
    return tuple(int(np.clip(v * (1 + s * (2 * rng.random() - 1)), 8, 250)) for v in c)


def sprite_catalog(rng):
    """Distinctly coloured landmark objects. h/w are world-space sizes
    (1 unit = corridor width)."""
    items = [
        ("barrel", spr_barrel, 0.46, 0.65),
        ("crate", spr_crate, 0.38, 1.00),
        ("plant", spr_plant, 0.56, 0.75),
        ("cone", spr_cone, 0.34, 0.75),
        ("sign", spr_sign, 0.80, 0.40),
        ("extinguisher", spr_ext, 0.38, 0.55),
        ("bin", spr_bin, 0.44, 0.62),
        ("boxes", spr_boxes, 0.52, 0.85),
        ("lamp", spr_lamp, 0.88, 0.42),
    ]
    return [dict(name=n, tex=f(rng), h=h, w=round(h * wr, 3)) for n, f, h, wr in items]
