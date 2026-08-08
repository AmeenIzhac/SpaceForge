"""Top-down map figure in the style of the proposal's Figure 4:
outbound route blue, return route + chord corridors orange,
start cell green, stopping point red."""

from PIL import Image, ImageDraw

BLUE = (59, 111, 212)
ORANGE = (242, 161, 60)
GREEN = (46, 158, 68)
RED = (212, 59, 59)
DOT = (225, 225, 228)


ROOM_FILL = (226, 229, 236)
ROOM_EDGE = (201, 205, 214)


def draw_plan(world_size, rooms, routes, out_png, ppu=14, margin=26):
    """Floor-plan map: packed rooms (world-space interiors), corridors along
    the wall lines (outbound blue, return + chords orange), doorway dots."""
    W, H = world_size
    img = Image.new("RGB", (2 * margin + W * ppu, 2 * margin + H * ppu),
                    (255, 255, 255))
    d = ImageDraw.Draw(img)

    def P(u, v):
        return (margin + u * ppu, margin + v * ppu)

    for room in rooms:
        xa, ya, xb, yb = room["rect"]
        d.rounded_rectangle([*P(xa, ya), *P(xb + 1, yb + 1)], radius=5,
                            fill=ROOM_FILL, outline=ROOM_EDGE, width=2)

    def draw_route(pts, color, lw):
        for a, b in zip(pts[:-1], pts[1:]):
            d.line([P(*a), P(*b)], fill=color, width=lw)
        for p in pts:
            x, y = P(*p)
            r = lw // 2
            d.ellipse([x - r, y - r, x + r, y + r], fill=color)

    lw = max(6, int(ppu * 0.65))
    for chord in routes["chords"]:
        draw_route(chord, ORANGE, lw)
    draw_route(routes["ret"], ORANGE, lw)
    draw_route(routes["out"], BLUE, lw)

    for room in rooms:
        for dr in room["doors"]:
            x, y = P(dr["x"], dr["y"])
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 255, 255),
                      outline=ROOM_EDGE, width=2)

    for p, color in ((routes["out"][0], GREEN), (routes["out"][-1], RED)):
        x, y = P(*p)
        d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=color,
                  outline=(255, 255, 255), width=2)
    img.save(out_png)


def draw_map(n, path, out_png, cell=44, margin=40, lw=16, rooms=None):
    size = 2 * margin + (n - 1) * cell
    img = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(img)

    def pix(c):
        return (margin + c[0] * cell, margin + c[1] * cell)

    def wpix(u):                       # continuous world coordinate -> pixels
        return margin + (u - 1.5) * cell / 2

    for room in rooms or []:
        x0, y0, x1, y1 = room["rect"]
        d.rounded_rectangle([wpix(x0), wpix(y0), wpix(x1 + 1), wpix(y1 + 1)],
                            radius=6, fill=ROOM_FILL, outline=ROOM_EDGE, width=3)
        for dr in room["doors"]:       # stub connecting the doorway to the hall
            px, py = wpix(dr["x"]), wpix(dr["y"])
            ex = px + dr["nx"] * cell * 0.30
            ey = py + dr["ny"] * cell * 0.30
            d.line([px - dr["nx"] * cell * 0.30, py - dr["ny"] * cell * 0.30, ex, ey],
                   fill=ROOM_EDGE, width=9)
            d.ellipse([px - 5, py - 5, px + 5, py + 5], fill=(255, 255, 255),
                      outline=ROOM_EDGE, width=2)

    for gx in range(n):
        for gy in range(n):
            x, y = pix((gx, gy))
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=DOT)

    def draw_route(route, color):
        for a, b in zip(route[:-1], route[1:]):
            d.line([pix(a), pix(b)], fill=color, width=lw)
        for c in route:
            x, y = pix(c)
            r = lw // 2
            d.ellipse([x - r, y - r, x + r, y + r], fill=color)

    for chord in path["chords"]:
        draw_route(chord, ORANGE)
    draw_route(path["ret"], ORANGE)
    draw_route(path["out"], BLUE)

    for c, color in ((path["out"][0], GREEN), (path["out"][-1], RED)):
        x, y = pix(c)
        d.ellipse([x - 13, y - 13, x + 13, y + 13], fill=color, outline=(255, 255, 255), width=3)

    img.save(out_png)
