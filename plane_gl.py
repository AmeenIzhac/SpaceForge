"""Cheap renderer for the open-plane scenes: raw OpenGL, no browser.

Reads exactly the same scene spec `plane_gen.py` writes, so a video from here
is interchangeable with the three.js one and the ground truth is untouched.
The point is cost: three.js runs inside headless Chrome and every frame makes
a CDP screenshot round-trip, which dominates. Here the GPU draws straight into
an offscreen framebuffer and the pixels go to ffmpeg over a pipe.

Deliberately close to the three.js look rather than identical — same camera
convention, same sun direction, same object palette, procedural checker
ground, Lambert shading with a cheap planar shadow per object.

    .venv/bin/python plane_gl.py --scenes probes/plane_test.json \
        --out out/plane_gl --limit 6
"""

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
W, H = 640, 360

VERT = """
#version 330
uniform mat4 mvp;
uniform mat4 model;
in vec3 in_pos;
in vec3 in_norm;
out vec3 v_world;
out vec3 v_norm;
void main() {
    vec4 w = model * vec4(in_pos, 1.0);
    v_world = w.xyz;
    v_norm = mat3(model) * in_norm;
    gl_Position = mvp * w;
}
"""

FRAG = """
#version 330
uniform vec3 colour;
uniform vec3 sun_dir;
uniform vec3 sky;
uniform int is_ground;
uniform vec3 ground_a;
uniform vec3 ground_b;
uniform float ground_scale;
uniform int ground_style;      // 0 checker, 1 plain, 2 noise, 3 patchy
uniform float ambient;         // raised indoors, where there is no sky
uniform float diffuse_k;
uniform float fog_near;
uniform float fog_far;
uniform vec3 cam;
in vec3 v_world;
in vec3 v_norm;
out vec4 f_colour;
float hash21(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i), b = hash21(i + vec2(1, 0));
    float c = hash21(i + vec2(0, 1)), d = hash21(i + vec2(1, 1));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
float fbm(vec2 p) {
    float v = 0.0, amp = 0.5;
    for (int i = 0; i < 4; i++) { v += amp * vnoise(p); p *= 2.03; amp *= 0.5; }
    return v;
}

void main() {
    vec3 base = colour;
    if (is_ground == 1) {
        vec2 w = v_world.xz;
        if (ground_style == 0) {                 // checker
            vec2 c = floor(w * ground_scale);
            base = mod(c.x + c.y, 2.0) < 0.5 ? ground_a : ground_b;
        } else if (ground_style == 1) {          // plain, faint grain only
            base = mix(ground_a, ground_b, 0.06 * fbm(w * 3.0));
        } else if (ground_style == 2) {          // soft noise, no grid at all
            base = mix(ground_a, ground_b, smoothstep(0.3, 0.7, fbm(w * 0.7)));
            base *= 0.92 + 0.16 * fbm(w * 6.0);
        } else {                                 // patchy: large blotches
            float m = fbm(w * 0.16);
            base = mix(ground_a, ground_b, smoothstep(0.42, 0.58, m));
            base *= 0.90 + 0.20 * fbm(w * 2.4);
        }
    }
    vec3 n = normalize(v_norm);
    float diff = max(dot(n, sun_dir), 0.0);
    vec3 lit = base * (ambient + diffuse_k * diff);
    float d = length(v_world - cam);
    float f = clamp((d - fog_near) / (fog_far - fog_near), 0.0, 1.0);
    f_colour = vec4(mix(lit, sky, f), 1.0);
}
"""


# ----------------------------------------------------------------- geometry
def _mesh(verts, norms, idx):
    return np.asarray(verts, "f4"), np.asarray(norms, "f4"), np.asarray(idx, "i4")


def sphere(r, seg=20, ring=14):
    v, n, idx = [], [], []
    for i in range(ring + 1):
        th = math.pi * i / ring
        for j in range(seg + 1):
            ph = 2 * math.pi * j / seg
            p = (math.sin(th) * math.cos(ph), math.cos(th), math.sin(th) * math.sin(ph))
            v.append([p[0] * r, p[1] * r, p[2] * r])
            n.append(list(p))
    for i in range(ring):
        for j in range(seg):
            a = i * (seg + 1) + j
            b = a + seg + 1
            idx += [a, b, a + 1, a + 1, b, b + 1]
    return _mesh(v, n, idx)


def box(sx, sy, sz):
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    faces = [((0, 0, 1), [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]),
             ((0, 0, -1), [(hx, -hy, -hz), (-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz)]),
             ((1, 0, 0), [(hx, -hy, hz), (hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz)]),
             ((-1, 0, 0), [(-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz)]),
             ((0, 1, 0), [(-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz), (-hx, hy, -hz)]),
             ((0, -1, 0), [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)])]
    v, n, idx = [], [], []
    for nrm, quad in faces:
        b = len(v)
        for p in quad:
            v.append(list(p))
            n.append(list(nrm))
        idx += [b, b + 1, b + 2, b, b + 2, b + 3]
    return _mesh(v, n, idx)


def cylinder(r0, r1, h, seg=22):
    v, n, idx = [], [], []
    for j in range(seg + 1):
        a = 2 * math.pi * j / seg
        ca, sa = math.cos(a), math.sin(a)
        v.append([r0 * ca, -h / 2, r0 * sa]); n.append([ca, 0.25, sa])
        v.append([r1 * ca, h / 2, r1 * sa]); n.append([ca, 0.25, sa])
    for j in range(seg):
        a = 2 * j
        idx += [a, a + 1, a + 2, a + 2, a + 1, a + 3]
    top = len(v)
    v.append([0, h / 2, 0]); n.append([0, 1, 0])
    for j in range(seg + 1):
        a = 2 * math.pi * j / seg
        v.append([r1 * math.cos(a), h / 2, r1 * math.sin(a)]); n.append([0, 1, 0])
    for j in range(seg):
        idx += [top, top + 1 + j, top + 2 + j]
    return _mesh(v, n, idx)


def cone(r, h, seg=22):
    return cylinder(r, 0.001, h, seg)


def torus(R, r, seg=24, ring=14):
    v, n, idx = [], [], []
    for i in range(ring + 1):
        u = 2 * math.pi * i / ring
        for j in range(seg + 1):
            t = 2 * math.pi * j / seg
            nx, ny, nz = math.cos(t) * math.cos(u), math.sin(t), math.cos(t) * math.sin(u)
            v.append([(R + r * math.cos(t)) * math.cos(u), r * math.sin(t),
                      (R + r * math.cos(t)) * math.sin(u)])
            n.append([nx, ny, nz])
    for i in range(ring):
        for j in range(seg):
            a = i * (seg + 1) + j
            b = a + seg + 1
            idx += [a, b, a + 1, a + 1, b, b + 1]
    return _mesh(v, n, idx)


def xmark(arm, thick=0.22):
    """Two flat bars crossed at right angles, lying on the ground — the
    corridor task's floor marker, in this renderer."""
    a = box(arm * 2, 0.03, thick)
    b = box(thick, 0.03, arm * 2)
    v = np.vstack([a[0], b[0]])
    n = np.vstack([a[1], b[1]])
    i = np.concatenate([a[2], b[2] + len(a[0])])
    return v.astype("f4"), n.astype("f4"), i.astype("i4")


def wallgrid(cells, size, h):
    """One merged mesh for a whole set of wall blocks. Per-leg wall slabs
    overlap through corners and can swallow the camera; a grid of blocks around
    the walkable region cannot, and it is how the real corridor renderer does
    it. Merging keeps it to a single draw call."""
    bx, bn, bi = box(size, h, size)
    vs, ns, idx = [], [], []
    for cx, cy in cells:
        base = len(vs)
        for v in bx:
            vs.append([v[0] + cx, v[1] + h / 2, v[2] + cy])
        ns.extend(bn.tolist())
        idx.extend((bi + base).tolist())
    return (np.asarray(vs, "f4"), np.asarray(ns, "f4"),
            np.asarray(idx, "i4"))


def build(o):
    """(mesh, lift) for one object. `dims` overrides the shape's own size —
    that is how buildings get to be tall boxes rather than scaled cubes."""
    shape, s = o["shape"], o.get("scale", 1.0)
    if "dims" in o:
        w, h, d = o["dims"]
        if shape in ("building", "cube"):
            return box(w, h, d), h / 2
        if shape == "cylinder":
            return cylinder(w / 2, w / 2, h), h / 2
        if shape == "cone":
            return cone(w / 2, h), h / 2
    if shape == "wallgrid":
        return wallgrid(o["cells"], o["cell"], o["h"]), 0.0
    if shape == "xmark":     return xmark(0.75 * s), 0.02
    if shape == "cube":       return box(.95 * s, .95 * s, .95 * s), .475 * s
    if shape == "sphere":     return sphere(.55 * s), .55 * s
    if shape == "cylinder":   return cylinder(.38 * s, .38 * s, 1.9 * s), .95 * s
    if shape == "cone":       return cone(.6 * s, 1.5 * s), .75 * s
    if shape == "torus":      return torus(.6 * s, .21 * s), .81 * s
    if shape == "capsule":    return cylinder(.35 * s, .35 * s, 1.3 * s), .85 * s
    if shape == "octahedron": return sphere(.75 * s, 4, 4), .75 * s
    if shape == "pyramid":    return cone(.8 * s, 1.4 * s, 4), .70 * s
    raise KeyError(shape)


# ------------------------------------------------------------------- matrices
def perspective(fovy, aspect, near, far):
    f = 1 / math.tan(math.radians(fovy) / 2)
    m = np.zeros((4, 4), "f4")
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def view_from_angles(eye, yaw, pitch, roll):
    """View matrix from yaw/pitch/roll. Same handedness as look_at, and with
    pitch=roll=0 it is exactly look_at(eye, eye+(cos yaw, 0, sin yaw)) — so the
    bearing convention (090 = the camera's right) is unchanged and the existing
    ground truth still applies."""
    cf = math.cos(pitch)
    f = np.array([math.cos(yaw) * cf, math.sin(pitch),
                  math.sin(yaw) * cf], "f4")
    f /= np.linalg.norm(f)
    s = np.cross(f, np.array([0, 1, 0], "f4"))
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    cr, sr = math.cos(roll), math.sin(roll)
    s, u = s * cr + u * sr, u * cr - s * sr          # roll about the view axis
    m = np.eye(4, dtype="f4")
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ np.array(eye, "f4")
    return m


def look_at(eye, tgt, up=(0, 1, 0)):
    f = np.array(tgt, "f4") - np.array(eye, "f4")
    f /= np.linalg.norm(f)
    s = np.cross(f, np.array(up, "f4"))
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype="f4")
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ np.array(eye, "f4")
    return m


def model_mat(x, y, z, yaw=0.0, sy=1.0):
    c, s = math.cos(yaw), math.sin(yaw)
    m = np.eye(4, dtype="f4")
    m[0, 0], m[0, 2] = c, s
    m[2, 0], m[2, 2] = -s, c
    m[1, 1] = sy
    m[:3, 3] = (x, y, z)
    return m


def hsl(h, s, l):
    import colorsys
    return colorsys.hls_to_rgb(h, l, s)


def render_scene(ctx, prog, scene, out_path, fps=30):
    import moderngl
    fbo = ctx.simple_framebuffer((W, H))
    fbo.use()
    ctx.enable(moderngl.DEPTH_TEST)

    sky = np.array(hsl(scene["sky_hue"], 0.42, 0.72), "f4")
    ga = np.array(hsl(scene["ground_hue"], 0.30, 0.46), "f4")
    gb = np.array(hsl((scene["ground_hue"] + 0.04) % 1, 0.28, 0.52), "f4")
    az, el = scene["sun_az"], scene["sun_el"]
    sun = np.array([math.cos(az) * math.cos(el), math.sin(el),
                    math.sin(az) * math.cos(el)], "f4")
    sun /= np.linalg.norm(sun)

    gv, gn, gi = box(900, 0.02, 900)
    ground = ctx.vertex_array(prog, [
        (ctx.buffer(np.hstack([gv, gn]).astype("f4").tobytes()),
         "3f 3f", "in_pos", "in_norm")], ctx.buffer(gi.tobytes()))

    objs = []
    for o in scene["objects"]:
        (v, n, i), lift = build(o)
        lift = o.get("y_off", lift)          # walls/ceilings set this directly
        vao = ctx.vertex_array(prog, [
            (ctx.buffer(np.hstack([v, n]).astype("f4").tobytes()),
             "3f 3f", "in_pos", "in_norm")], ctx.buffer(i.tobytes()))
        rgb = np.array([int(o["colour"][k:k + 2], 16) / 255 for k in (1, 3, 5)], "f4")
        objs.append((vao, rgb, o, lift))

    prog["sky"].value = tuple(sky)
    prog["sun_dir"].value = tuple(sun)
    prog["ground_a"].value = tuple(ga)
    prog["ground_b"].value = tuple(gb)
    # match the three.js checker size: its 256px/8-checker texture tiles
    # `ground_scale` times across a 900-unit plane
    prog["ground_scale"].value = scene["ground_scale"] * 8 / 900
    indoor = bool(scene.get("indoor"))
    prog["ambient"].value = 0.62 if indoor else 0.42
    prog["diffuse_k"].value = 0.35 if indoor else 0.75
    if indoor:
        prog["fog_near"].value = 12.0
        prog["fog_far"].value = 55.0
    prog["ground_style"].value = {"checker": 0, "plain": 1, "noise": 2,
                                  "patchy": 3}[scene.get("ground_style",
                                                         "checker")]
    prog["fog_near"].value = 45.0
    prog["fog_far"].value = 190.0

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-framerate", str(fps), "-i", "-",
         # OpenGL's framebuffer origin is bottom-left, so rows come out
         # upside down; flipping in ffmpeg costs nothing on this path
         "-vf", "vflip", "-c:v", "libx264",
         "-preset", "veryfast", "-pix_fmt", "yuv420p", "-crf", "23", "-threads", "1",
         str(out_path)], stdin=subprocess.PIPE)

    proj = perspective(scene["fov"], W / H, 0.05, 400)
    for k, pose in enumerate(scene["poses"]):
        # 3-element poses are the old (x, y, yaw); 6-element ones carry
        # pitch, roll and a per-frame eye height as well
        if len(pose) >= 6:
            x, y, yaw, pitch, roll, h = pose[:6]
        else:
            x, y, yaw = pose[:3]
            pitch = roll = 0.0
            h = scene["eye"] + 0.018 * math.sin(k * 0.42)
        view = view_from_angles((x, h, y), yaw, pitch, roll)
        vp = proj @ view
        fbo.clear(*sky, 1.0)
        prog["cam"].value = (x, h, y)

        prog["is_ground"].value = 1
        m = model_mat(0, -0.01, 0)
        prog["model"].write(m.T.tobytes())
        prog["mvp"].write(vp.T.tobytes())
        prog["colour"].value = (0.5, 0.5, 0.5)
        ground.render()

        prog["is_ground"].value = 0
        for vao, rgb, o, lift in objs:          # flat shadow, then the object
            # structure (walls, ceiling) casts no blob shadow: the ceiling is
            # the size of the whole scene, so its shadow would paint the floor
            # black — which is exactly what it did
            if not o.get("no_shadow"):
                m = model_mat(o["x"], 0.012, o["y"], o["yaw"], sy=0.02)
                prog["model"].write(m.T.tobytes())
                prog["mvp"].write(vp.T.tobytes())
                prog["colour"].value = (0.22, 0.22, 0.20)
                vao.render()
            m = model_mat(o["x"], lift, o["y"], o["yaw"])
            prog["model"].write(m.T.tobytes())
            prog["mvp"].write(vp.T.tobytes())
            prog["colour"].value = tuple(rgb)
            vao.render()

        ff.stdin.write(fbo.read(components=3))
    ff.stdin.close()
    ff.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="probes/plane_test.json")
    ap.add_argument("--out", default="out/plane_gl")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--gpu", default="2")
    args = ap.parse_args()
    os.environ.setdefault("EGL_DEVICE_ID", args.gpu)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

    import moderngl
    import gl_renderer
    ctx = gl_renderer._standalone_context()
    prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)

    scenes = json.loads((ROOT / args.scenes).read_text())["scenes"]
    if args.ids:
        keep = {int(i) for i in args.ids.split(",")}
        scenes = [s for s in scenes if s["id"] in keep]
    scenes = scenes[: args.limit]
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    frames = 0
    for s in scenes:
        render_scene(ctx, prog, s, out / f"{s['id']:04d}.mp4", s["fps"])
        frames += len(s["poses"])
    dt = time.time() - t0
    print(f"{len(scenes)} scenes, {frames} frames in {dt:.1f}s "
          f"-> {frames/dt:.0f} fps, {dt/len(scenes):.2f}s per scene")


if __name__ == "__main__":
    main()
