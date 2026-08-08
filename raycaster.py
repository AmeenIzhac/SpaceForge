"""Purpose-built corridor renderer: a vectorised numpy raycaster.

DDA raycasting over the world grid (0 = open, >0 = wall texture id) with
perspective-correct textured walls, floor/ceiling casting, billboard sprites
with per-column depth testing, exponential distance fog, subtle head-bob and
vignette. Every quantity needed for supervision (pose, depth, layout) is exact.
"""

import numpy as np


class Renderer:
    def __init__(self, wmap, wall_atlas, floor_tex, ceil_tex, sprites,
                 width=640, height=480, fov_deg=68.0, wall_h=1.25, eye=0.60,
                 fog_d=15.0, fog_rgb=(0.8, 0.8, 0.82), side_shade=0.82,
                 floor_scale=1.0, ceil_scale=0.5, bob_amp=0.010, stride=0.72):
        self.map = np.ascontiguousarray(wmap, np.int16)
        self.mh, self.mw = self.map.shape
        self.atlas = np.ascontiguousarray(wall_atlas, np.float32)          # (K,T,T,3)
        self.fc = np.stack([np.asarray(floor_tex, np.float32),
                            np.asarray(ceil_tex, np.float32)])             # (2,T,T,3)
        self.sprites = sprites
        self.W, self.H = width, height
        self.tanh = float(np.tan(np.radians(fov_deg) / 2))
        self.f = (width / 2) / self.tanh                                   # focal length, px
        self.hor = height / 2
        self.wall_h = wall_h
        self.eye0 = eye
        self.fog_d = fog_d
        self.fog = np.asarray(fog_rgb, np.float32)
        self.side_shade = side_shade
        self.fs, self.cs = floor_scale, ceil_scale
        self.bob_amp, self.stride = bob_amp, stride
        self.camx = np.linspace(-1.0, 1.0, width).astype(np.float32)
        self.yc = (np.arange(height, dtype=np.float32) + 0.5)[:, None]     # pixel-row centers
        xn = np.linspace(-1, 1, width, dtype=np.float32)[None, :]
        yn = np.linspace(-1, 1, height, dtype=np.float32)[:, None]
        self.vig = (1.0 - 0.16 * (xn ** 2 + yn ** 2))[..., None]
        self.T = self.atlas.shape[1]
        self.Tf = self.fc.shape[1]
        self.maxiter = 2 * (self.mw + self.mh)

    def render(self, x, y, yaw, s=0.0):
        W, H, f, hor = self.W, self.H, self.f, self.hor
        eye = self.eye0 + self.bob_amp * np.sin(2 * np.pi * s / self.stride)
        dirx, diry = float(np.cos(yaw)), float(np.sin(yaw))
        px, py = -diry * self.tanh, dirx * self.tanh                        # camera plane

        rdx = (dirx + px * self.camx).astype(np.float32)
        rdy = (diry + py * self.camx).astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            ddx = np.abs(1.0 / rdx)
            ddy = np.abs(1.0 / rdy)
            mapx = np.full(W, int(np.floor(x)), np.int32)
            mapy = np.full(W, int(np.floor(y)), np.int32)
            stepx = np.where(rdx < 0, -1, 1).astype(np.int32)
            stepy = np.where(rdy < 0, -1, 1).astype(np.int32)
            sdx = np.where(rdx < 0, x - mapx, mapx + 1.0 - x).astype(np.float32) * ddx
            sdy = np.where(rdy < 0, y - mapy, mapy + 1.0 - y).astype(np.float32) * ddy
        sdx = np.nan_to_num(sdx, nan=np.inf, posinf=np.inf)
        sdy = np.nan_to_num(sdy, nan=np.inf, posinf=np.inf)

        side = np.zeros(W, bool)   # False = x-side wall, True = y-side wall
        hit = np.zeros(W, bool)
        for _ in range(self.maxiter):
            act = ~hit
            if not act.any():
                break
            gx = act & (sdx <= sdy)
            gy = act & (sdx > sdy)
            mapx[gx] += stepx[gx]
            sdx[gx] += ddx[gx]
            side[gx] = False
            mapy[gy] += stepy[gy]
            sdy[gy] += ddy[gy]
            side[gy] = True
            cx = np.clip(mapx, 0, self.mw - 1)
            cy = np.clip(mapy, 0, self.mh - 1)
            oob = (mapx != cx) | (mapy != cy)
            hit |= act & ((self.map[cy, cx] > 0) | oob)

        with np.errstate(invalid="ignore"):
            perp = np.where(side, sdy - ddy, sdx - ddx).astype(np.float32)
        perp = np.maximum(np.nan_to_num(perp, nan=1e-4, posinf=1e4), 1e-4)
        cx = np.clip(mapx, 0, self.mw - 1)
        cy = np.clip(mapy, 0, self.mh - 1)
        tid = np.maximum(self.map[cy, cx].astype(np.int32), 1) - 1

        wx_hit = np.where(side, x + perp * rdx, y + perp * rdy)
        wallx = wx_hit - np.floor(wx_hit)
        flip = (~side & (rdx > 0)) | (side & (rdy < 0))
        wallx = np.where(flip, 1.0 - wallx, wallx).astype(np.float32)

        # --- walls -----------------------------------------------------------
        topf = (hor - (self.wall_h - eye) * f / perp).astype(np.float32)
        botf = (hor + eye * f / perp).astype(np.float32)
        onw = (self.yc >= topf[None, :]) & (self.yc < botf[None, :])
        worldz = (botf[None, :] - self.yc) * (perp[None, :] / f)
        v = np.clip(worldz / self.wall_h, 0.0, 0.999)
        ty = ((1.0 - v) * (self.T - 1)).astype(np.int32)
        tx = np.minimum((wallx * self.T).astype(np.int32), self.T - 1)
        wallpix = self.atlas[tid[None, :], ty, tx[None, :]]
        shade = np.where(side, self.side_shade, 1.0).astype(np.float32)
        wallpix *= shade[None, :, None]

        # --- floor / ceiling ---------------------------------------------------
        below = self.yc > hor
        dyf = np.maximum(self.yc - hor, 0.5)
        dyc = np.maximum(hor - self.yc, 0.5)
        rowd = np.where(below, eye * f / dyf, (self.wall_h - eye) * f / dyc).astype(np.float32)
        gx_ = (x + rowd * rdx[None, :])
        gy_ = (y + rowd * rdy[None, :])
        scale = np.where(below, self.fs, self.cs).astype(np.float32)
        ux = gx_ * scale
        uy = gy_ * scale
        txf = np.minimum(((ux - np.floor(ux)) * self.Tf).astype(np.int32), self.Tf - 1)
        tyf = np.minimum(((uy - np.floor(uy)) * self.Tf).astype(np.int32), self.Tf - 1)
        sel = (~below).astype(np.int32)                                    # 0 floor, 1 ceiling
        base = self.fc[sel, tyf, txf]

        img = np.where(onw[..., None], wallpix, base)
        depth = np.where(onw, perp[None, :], rowd).astype(np.float32)
        fogf = np.exp(-depth / self.fog_d)[..., None]
        img = img * fogf + self.fog * (1.0 - fogf)

        # --- billboard sprites (painter's order, per-column wall depth test) --
        invdet = 1.0 / (px * diry - dirx * py)
        vis = []
        for sp in self.sprites:
            relx, rely = sp["x"] - x, sp["y"] - y
            d = invdet * (-py * relx + px * rely)
            if d > 0.30:
                vis.append((d, invdet * (diry * relx - dirx * rely), sp))
        for d, tcx, sp in sorted(vis, key=lambda o: -o[0]):
            sxc = (W / 2) * (1 + tcx / d)
            wpix = sp["w"] * f / d
            if wpix < 1:
                continue
            top = hor + (eye - sp["h"]) * f / d
            bot = hor + eye * f / d
            x0 = max(int(np.floor(sxc - wpix / 2)), 0)
            x1 = min(int(np.ceil(sxc + wpix / 2)), W)
            r0 = max(int(np.floor(top)), 0)
            r1 = min(int(np.ceil(bot)), H)
            if x0 >= x1 or r0 >= r1:
                continue
            occl = d < perp[x0:x1]
            if not occl.any():
                continue
            st = sp["tex"]
            cols = np.arange(x0, x1, dtype=np.float32)
            u = np.clip((cols + 0.5 - (sxc - wpix / 2)) / wpix, 0.0, 0.999)
            rows = np.arange(r0, r1, dtype=np.float32) + 0.5
            vv = np.clip((rows - top) / max(bot - top, 1e-6), 0.0, 0.999)
            su = (u * st.shape[1]).astype(np.int32)
            sv = (vv * st.shape[0]).astype(np.int32)
            patch = st[sv[:, None], su[None, :]]
            a = patch[..., 3:] * occl[None, :, None]
            if a.max() <= 0:
                continue
            fsp = float(np.exp(-d / self.fog_d))
            rgb = patch[..., :3] * fsp + self.fog * (1.0 - fsp)
            reg = img[r0:r1, x0:x1]
            img[r0:r1, x0:x1] = reg * (1.0 - a) + rgb * a

        img *= self.vig
        return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
