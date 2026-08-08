"""GPU corridor renderer (moderngl / OpenGL 3.3, headless).

Real 3D geometry: textured wall/floor/ceiling quads built from the world map,
solid prop meshes, emissive ceiling light panels. Per-pixel Blinn-Phong point
lighting from the panels (plus lamp props), specular floors, exponential fog,
contact shadows under props, 4x MSAA, head-bob. Same render(x, y, yaw, s)
interface as the numpy raycaster, so the generator can use either.

World ground plane (x, y) maps to GL (x, z); height is GL +Y.
"""

import numpy as np

try:
    import moderngl
except ImportError:      # generator falls back to the raycaster
    moderngl = None

MAX_LIGHTS = 64

_LEVEL_VS = """
#version 330
uniform mat4 u_vp;
in vec3 in_pos; in vec3 in_norm; in vec2 in_uv; in float in_layer; in float in_spec;
out vec3 v_pos; out vec3 v_norm; out vec2 v_uv; out float v_layer; out float v_spec;
void main() {
    v_pos = in_pos; v_norm = in_norm; v_uv = in_uv;
    v_layer = in_layer; v_spec = in_spec;
    gl_Position = u_vp * vec4(in_pos, 1.0);
}
"""

_LIGHT_FN = """
uniform vec3 u_cam; uniform vec3 u_amb; uniform vec3 u_fogc; uniform float u_fogd;
uniform int u_nl; uniform vec4 u_lpos[64]; uniform vec4 u_lcol[64];
void shade(vec3 pos, vec3 n, out vec3 diff, out vec3 spec) {
    diff = vec3(0.0); spec = vec3(0.0);
    vec3 vd = normalize(u_cam - pos);
    for (int i = 0; i < u_nl; i++) {
        vec3 ld = u_lpos[i].xyz - pos;
        float d2 = dot(ld, ld);
        vec3 l = ld * inversesqrt(max(d2, 1e-6));
        float att = u_lpos[i].w / (1.0 + 1.1 * d2);
        diff += u_lcol[i].rgb * (max(dot(n, l), 0.0) * att);
        vec3 h = normalize(l + vd);
        spec += u_lcol[i].rgb * (pow(max(dot(n, h), 0.0), 48.0) * att);
    }
}
vec3 fogmix(vec3 c, vec3 pos) {
    float f = exp(-distance(u_cam, pos) / u_fogd);
    return mix(u_fogc, c, f);
}
"""

_LEVEL_FS = """
#version 330
uniform sampler2DArray u_tex;
in vec3 v_pos; in vec3 v_norm; in vec2 v_uv; in float v_layer; in float v_spec;
out vec4 f_color;
""" + _LIGHT_FN + """
void main() {
    vec3 base = texture(u_tex, vec3(v_uv, v_layer)).rgb;
    vec3 diff, spec;
    shade(v_pos, normalize(v_norm), diff, spec);
    vec3 c = base * (u_amb + diff) + spec * v_spec;
    f_color = vec4(fogmix(c, v_pos), 1.0);
}
"""

_COLOR_VS = """
#version 330
uniform mat4 u_vp;
in vec3 in_pos; in vec3 in_norm; in vec3 in_col; in float in_emit;
out vec3 v_pos; out vec3 v_norm; out vec3 v_col; out float v_emit;
void main() {
    v_pos = in_pos; v_norm = in_norm; v_col = in_col; v_emit = in_emit;
    gl_Position = u_vp * vec4(in_pos, 1.0);
}
"""

_COLOR_FS = """
#version 330
in vec3 v_pos; in vec3 v_norm; in vec3 v_col; in float v_emit;
out vec4 f_color;
""" + _LIGHT_FN + """
void main() {
    vec3 diff, spec;
    shade(v_pos, normalize(v_norm), diff, spec);
    vec3 lit = v_col * (u_amb + diff) + spec * 0.22;
    vec3 c = mix(lit, v_col * 1.15, v_emit);
    f_color = vec4(fogmix(c, v_pos), 1.0);
}
"""

_SHADOW_VS = """
#version 330
uniform mat4 u_vp;
in vec3 in_pos; in vec2 in_uv;
out vec3 v_pos; out vec2 v_uv;
void main() { v_pos = in_pos; v_uv = in_uv; gl_Position = u_vp * vec4(in_pos, 1.0); }
"""

_SHADOW_FS = """
#version 330
uniform vec3 u_cam; uniform float u_fogd;
in vec3 v_pos; in vec2 v_uv;
out vec4 f_color;
void main() {
    float r = length(v_uv);
    float a = 0.42 * pow(clamp(1.0 - r, 0.0, 1.0), 1.4);
    a *= exp(-distance(u_cam, v_pos) / u_fogd);
    f_color = vec4(0.0, 0.0, 0.0, a);
}
"""

_QUAD_VS = """
#version 330
out vec2 v_uv;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

_BRIGHT_FS = """
#version 330
uniform sampler2D u_src;
in vec2 v_uv; out vec4 f_color;
void main() {
    vec3 c = texture(u_src, v_uv).rgb;
    f_color = vec4(max(c - 0.72, 0.0) * 2.2, 1.0);
}
"""

_BLUR_FS = """
#version 330
uniform sampler2D u_src; uniform vec2 u_dir;
in vec2 v_uv; out vec4 f_color;
void main() {
    vec3 c = texture(u_src, v_uv).rgb * 0.294;
    c += texture(u_src, v_uv + u_dir * 1.33).rgb * 0.235;
    c += texture(u_src, v_uv - u_dir * 1.33).rgb * 0.235;
    c += texture(u_src, v_uv + u_dir * 3.11).rgb * 0.118;
    c += texture(u_src, v_uv - u_dir * 3.11).rgb * 0.118;
    f_color = vec4(c, 1.0);
}
"""

_POST_FS = """
#version 330
uniform sampler2D u_scene; uniform sampler2D u_bloom;
uniform float u_exposure; uniform float u_bloom_amt;
in vec2 v_uv; out vec4 f_color;
vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
void main() {
    vec3 c = texture(u_scene, v_uv).rgb * u_exposure
           + texture(u_bloom, v_uv).rgb * u_bloom_amt;
    c = mix(c, aces(c), 0.55);            // soft filmic, keeps shadows readable
    c = pow(c, vec3(1.0 / 1.06));
    vec2 d = v_uv - 0.5;
    c *= 1.0 - 0.18 * dot(d, d);
    f_color = vec4(c, 1.0);
}
"""


def _perspective(fovy, aspect, znear, zfar):
    f = 1.0 / np.tan(fovy / 2)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = 2 * zfar * znear / (znear - zfar)
    m[3, 2] = -1.0
    return m


def _look_at(eye, center, up=(0.0, 1.0, 0.0)):
    eye, center, up = (np.asarray(v, np.float64) for v in (eye, center, up))
    f = center - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[0, 3], m[1, 3], m[2, 3] = -s @ eye, -u @ eye, f @ eye
    return m


class GLRenderer:
    def __init__(self, wmap, wall_atlas, floor_tex, ceil_tex, floor_spec, mood,
                 prop_mesh, shadows, prop_lights,
                 width=960, height=544, fov_deg=68.0, wall_h=1.25, eye=0.60,
                 bob_amp=0.010, stride=0.72, decal_mesh=None, decal_texs=None,
                 room_mask=None, room_centers=(), room_floor_tex=None,
                 room_floor_spec=0.30, light_cells=None):
        if moderngl is None:
            raise RuntimeError("moderngl not installed")
        self.W, self.H = width, height
        self.wall_h, self.eye0 = wall_h, eye
        self.bob_amp, self.stride = bob_amp, stride
        self.fog = np.asarray(mood["fog_rgb"], np.float32)
        fovy = 2 * np.arctan(np.tan(np.radians(fov_deg) / 2) * height / width)
        self.proj = _perspective(fovy, width / height, 0.03, 80.0)

        ctx = self.ctx = moderngl.create_context(standalone=True)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        self.prog_level = ctx.program(vertex_shader=_LEVEL_VS, fragment_shader=_LEVEL_FS)
        self.prog_color = ctx.program(vertex_shader=_COLOR_VS, fragment_shader=_COLOR_FS)
        self.prog_shadow = ctx.program(vertex_shader=_SHADOW_VS, fragment_shader=_SHADOW_FS)

        # --- texture array: wall variants + door + floor + ceiling (+ rooms) --
        layers = list(wall_atlas) + [floor_tex, ceil_tex]
        if room_floor_tex is not None:
            layers.append(room_floor_tex)
        data = (np.clip(np.stack(layers), 0.0, 1.0) * 255).astype(np.uint8)
        tsize = data.shape[1]
        self.tex = ctx.texture_array((tsize, tsize, data.shape[0]), 3, data.tobytes())
        self.tex.build_mipmaps()
        self.tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.tex.repeat_x = self.tex.repeat_y = True
        try:
            self.tex.anisotropy = 8.0
        except Exception:
            pass
        floor_layer = len(wall_atlas)      # after walls+door
        ceil_layer = len(wall_atlas) + 1
        rfl_layer = len(wall_atlas) + 2 if room_floor_tex is not None else floor_layer
        rfl_spec = room_floor_spec if room_floor_tex is not None else floor_spec

        # --- level mesh ------------------------------------------------------
        lv = self._build_level(wmap, wall_h, floor_layer, ceil_layer, floor_spec,
                               room_mask, rfl_layer, rfl_spec)
        self.vbo_level = ctx.buffer(lv.astype("f4").tobytes())
        self.vao_level = ctx.vertex_array(
            self.prog_level, [(self.vbo_level, "3f 3f 2f 1f 1f",
                               "in_pos", "in_norm", "in_uv", "in_layer", "in_spec")])

        # --- wall decals (posters, signs) — reuse the level shader -----------
        self.vao_decal = None
        if decal_mesh is not None and len(decal_mesh) and decal_texs:
            dd = (np.clip(np.stack(decal_texs), 0.0, 1.0) * 255).astype(np.uint8)
            self.tex_decal = ctx.texture_array(
                (dd.shape[1], dd.shape[2], dd.shape[0]), 3, dd.tobytes())
            self.tex_decal.build_mipmaps()
            self.tex_decal.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            self.tex_decal.repeat_x = self.tex_decal.repeat_y = False
            self.vbo_decal = ctx.buffer(decal_mesh.astype("f4").tobytes())
            self.vao_decal = ctx.vertex_array(
                self.prog_level, [(self.vbo_decal, "3f 3f 2f 1f 1f",
                                   "in_pos", "in_norm", "in_uv", "in_layer", "in_spec")])

        # --- lights + emissive ceiling panels --------------------------------
        if light_cells is None:
            light_cells = self._light_cells(wmap, mood.get("light_every", 1), room_mask)
        lrgb = np.asarray(mood["light_rgb"], np.float64)
        lint = float(mood.get("light_int", 0.85))
        panels = [(x + 0.5, y + 0.5, 0.36) for (x, y) in light_cells]
        panels += [(cx, cy, 0.52) for (cx, cy) in room_centers]
        lights = [(cx, wall_h - 0.08, cy, lint) for (cx, cy, _r) in panels]
        n_white = len(lights)
        lights += [(x, h, z, 0.45) for (x, h, z) in prop_lights]
        if len(lights) > MAX_LIGHTS:
            idx = np.linspace(0, len(lights) - 1, MAX_LIGHTS).astype(int)
            lights = [lights[i] for i in idx]
        lpos = np.zeros((MAX_LIGHTS, 4), np.float32)
        lcol = np.zeros((MAX_LIGHTS, 4), np.float32)
        for i, (x, h, z, inten) in enumerate(lights):
            lpos[i] = (x, h, z, inten)
            lcol[i, :3] = lrgb if i < n_white else (1.0, 0.82, 0.60)
        self._set_static_uniforms(lpos, lcol, len(lights),
                                  mood.get("ambient", 0.3), mood["fog_d"])

        panel = self._panel_mesh(panels, wall_h, lrgb)
        pm = {k: np.concatenate([prop_mesh[k], panel[k]]) for k in ("v", "n", "c", "e")} \
            if len(panel["v"]) else prop_mesh
        cm = np.hstack([pm["v"], pm["n"], pm["c"], pm["e"][:, None]])
        self.vao_color = None
        if len(cm):
            self.vbo_color = ctx.buffer(cm.astype("f4").tobytes())
            self.vao_color = ctx.vertex_array(
                self.prog_color, [(self.vbo_color, "3f 3f 3f 1f",
                                   "in_pos", "in_norm", "in_col", "in_emit")])

        # --- contact shadows --------------------------------------------------
        self.vao_shadow = None
        sh = []
        for (x, y, r) in shadows:
            r *= 1.35
            c00 = (x - r, 0.006, y - r, -1, -1)
            c10 = (x + r, 0.006, y - r, 1, -1)
            c11 = (x + r, 0.006, y + r, 1, 1)
            c01 = (x - r, 0.006, y + r, -1, 1)
            sh += [c00, c10, c11, c00, c11, c01]
        if sh:
            self.vbo_shadow = ctx.buffer(np.asarray(sh, np.float32).tobytes())
            self.vao_shadow = ctx.vertex_array(
                self.prog_shadow, [(self.vbo_shadow, "3f 2f", "in_pos", "in_uv")])

        # --- framebuffers (MSAA -> resolve -> bloom -> tonemap) ---------------
        self.fbo_ms = ctx.framebuffer(
            color_attachments=[ctx.renderbuffer((width, height), samples=4)],
            depth_attachment=ctx.depth_renderbuffer((width, height), samples=4))
        self.tex_scene = ctx.texture((width, height), 3)
        bw, bh = width // 2, height // 2
        self.tex_b0 = ctx.texture((bw, bh), 3)
        self.tex_b1 = ctx.texture((bw, bh), 3)
        for t in (self.tex_scene, self.tex_b0, self.tex_b1):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
        self.fbo_res = ctx.framebuffer(color_attachments=[self.tex_scene])
        self.fbo_b0 = ctx.framebuffer(color_attachments=[self.tex_b0])
        self.fbo_b1 = ctx.framebuffer(color_attachments=[self.tex_b1])
        self.fbo_out = ctx.framebuffer(color_attachments=[ctx.renderbuffer((width, height))])

        self.prog_bright = ctx.program(vertex_shader=_QUAD_VS, fragment_shader=_BRIGHT_FS)
        self.prog_blur = ctx.program(vertex_shader=_QUAD_VS, fragment_shader=_BLUR_FS)
        self.prog_post = ctx.program(vertex_shader=_QUAD_VS, fragment_shader=_POST_FS)
        self.vao_bright = ctx.vertex_array(self.prog_bright, [])
        self.vao_blur = ctx.vertex_array(self.prog_blur, [])
        self.vao_post = ctx.vertex_array(self.prog_post, [])
        self.prog_bright["u_src"].value = 0
        self.prog_blur["u_src"].value = 0
        self.prog_post["u_scene"].value = 0
        self.prog_post["u_bloom"].value = 1
        self.prog_post["u_exposure"].value = 1.28
        self.prog_post["u_bloom_amt"].value = 0.50
        self._px = (1.0 / bw, 1.0 / bh)

    # -------------------------------------------------------------------------

    @staticmethod
    def _light_cells(wmap, every, room_mask=None):
        cells = []
        h, w = wmap.shape
        for wy in range(1, h, 2):
            for wx in range(1, w, 2):
                if wmap[wy, wx] != 0:
                    continue
                if room_mask is not None and room_mask[wy, wx]:
                    continue           # rooms get one centered panel instead
                if ((wx // 2 + wy // 2) % max(every, 1)) == 0:
                    cells.append((wx, wy))
        return cells

    @staticmethod
    def _panel_mesh(panels, wall_h, lrgb):
        V, N, C, E = [], [], [], []
        y = wall_h - 0.012
        col = np.clip(lrgb, 0, 1)
        for (cx, cz, r) in panels:
            q = [(cx - r, y, cz - r), (cx + r, y, cz - r),
                 (cx + r, y, cz + r), (cx - r, y, cz + r)]
            for p in (q[0], q[1], q[2], q[0], q[2], q[3]):
                V.append(p)
                N.append((0.0, -1.0, 0.0))
                C.append(col)
                E.append(1.0)
        return {"v": np.array(V).reshape(-1, 3), "n": np.array(N).reshape(-1, 3),
                "c": np.array(C).reshape(-1, 3), "e": np.array(E)}

    @staticmethod
    def _build_level(wmap, wall_h, floor_layer, ceil_layer, floor_spec,
                     room_mask=None, rfl_layer=None, rfl_spec=None):
        rows = []

        def quad(p0, p1, p2, p3, nrm, uvs, layer, spec):
            pts = (p0, p1, p2, p0, p2, p3)
            us = (uvs[0], uvs[1], uvs[2], uvs[0], uvs[2], uvs[3])
            for p, uv in zip(pts, us):
                rows.append((*p, *nrm, *uv, layer, spec))

        h, w = wmap.shape
        for wy in range(h):
            for wx in range(w):
                if wmap[wy, wx] != 0:
                    continue
                in_room = room_mask is not None and room_mask[wy, wx]
                fl = rfl_layer if in_room else floor_layer
                fs = rfl_spec if in_room else floor_spec
                x0, x1, z0, z1 = wx, wx + 1, wy, wy + 1
                quad((x0, 0, z0), (x1, 0, z0), (x1, 0, z1), (x0, 0, z1), (0, 1, 0),
                     ((x0, z0), (x1, z0), (x1, z1), (x0, z1)), fl, fs)
                quad((x0, wall_h, z1), (x1, wall_h, z1), (x1, wall_h, z0), (x0, wall_h, z0),
                     (0, -1, 0), ((x0 * .5, z1 * .5), (x1 * .5, z1 * .5),
                                  (x1 * .5, z0 * .5), (x0 * .5, z0 * .5)), ceil_layer, 0.05)

                def wall(b0, b1, nrm, layer):
                    t0 = (b0[0], wall_h, b0[2])
                    t1 = (b1[0], wall_h, b1[2])
                    quad(b0, b1, t1, t0, nrm, ((0, 1), (1, 1), (1, 0), (0, 0)), layer, 0.06)

                if wmap[wy - 1, wx] != 0:      # north wall
                    wall((x0, 0, z0), (x1, 0, z0), (0, 0, 1), wmap[wy - 1, wx] - 1)
                if wmap[wy + 1, wx] != 0:      # south wall
                    wall((x1, 0, z1), (x0, 0, z1), (0, 0, -1), wmap[wy + 1, wx] - 1)
                if wmap[wy, wx - 1] != 0:      # west wall
                    wall((x0, 0, z1), (x0, 0, z0), (1, 0, 0), wmap[wy, wx - 1] - 1)
                if wmap[wy, wx + 1] != 0:      # east wall
                    wall((x1, 0, z0), (x1, 0, z1), (-1, 0, 0), wmap[wy, wx + 1] - 1)
        return np.asarray(rows, np.float64)

    def _set_static_uniforms(self, lpos, lcol, nl, ambient, fog_d):
        for prog in (self.prog_level, self.prog_color):
            prog["u_lpos"].write(lpos.tobytes())
            prog["u_lcol"].write(lcol.tobytes())
            prog["u_nl"].value = nl
            prog["u_amb"].value = (ambient, ambient, ambient)
            prog["u_fogc"].value = tuple(float(v) for v in self.fog)
            prog["u_fogd"].value = float(fog_d)
        self.prog_shadow["u_fogd"].value = float(fog_d)
        self.prog_level["u_tex"].value = 0

    # -------------------------------------------------------------------------

    def render(self, x, y, yaw, s=0.0):
        eye_h = self.eye0 + self.bob_amp * np.sin(2 * np.pi * s / self.stride)
        eye = (x, eye_h, y)
        center = (x + np.cos(yaw), eye_h, y + np.sin(yaw))
        vp = self.proj @ _look_at(eye, center)
        vp_bytes = vp.T.astype("f4").tobytes()
        cam = (float(x), float(eye_h), float(y))

        self.fbo_ms.use()
        self.ctx.clear(*(float(v) for v in self.fog), depth=1.0)
        for prog in (self.prog_level, self.prog_color, self.prog_shadow):
            prog["u_vp"].write(vp_bytes)
            prog["u_cam"].value = cam
        self.tex.use(0)
        self.vao_level.render(moderngl.TRIANGLES)
        if self.vao_decal is not None:
            self.tex_decal.use(0)
            self.vao_decal.render(moderngl.TRIANGLES)
        if self.vao_color is not None:
            self.vao_color.render(moderngl.TRIANGLES)
        if self.vao_shadow is not None:
            self.ctx.enable(moderngl.BLEND)
            self.vao_shadow.render(moderngl.TRIANGLES)
            self.ctx.disable(moderngl.BLEND)

        # post: resolve -> bright pass -> separable blur -> tonemap + vignette
        self.ctx.copy_framebuffer(self.fbo_res, self.fbo_ms)
        self.fbo_b0.use()
        self.tex_scene.use(0)
        self.vao_bright.render(moderngl.TRIANGLES, vertices=3)
        self.prog_blur["u_dir"].value = (self._px[0], 0.0)
        self.fbo_b1.use()
        self.tex_b0.use(0)
        self.vao_blur.render(moderngl.TRIANGLES, vertices=3)
        self.prog_blur["u_dir"].value = (0.0, self._px[1])
        self.fbo_b0.use()
        self.tex_b1.use(0)
        self.vao_blur.render(moderngl.TRIANGLES, vertices=3)
        self.fbo_out.use()
        self.tex_scene.use(0)
        self.tex_b0.use(1)
        self.vao_post.render(moderngl.TRIANGLES, vertices=3)
        buf = self.fbo_out.read(components=3, alignment=1)
        img = np.frombuffer(buf, np.uint8).reshape(self.H, self.W, 3)[::-1]
        return np.ascontiguousarray(img)

    def release(self):
        for r in (self.vao_level, self.vao_color, self.vao_shadow, self.vao_decal,
                  self.vao_bright, self.vao_blur, self.vao_post):
            if r is not None:
                r.release()
        for name in ("vbo_level", "vbo_color", "vbo_shadow", "vbo_decal", "tex_decal",
                     "tex_scene", "tex_b0", "tex_b1",
                     "fbo_res", "fbo_b0", "fbo_b1", "fbo_out"):
            if hasattr(self, name):
                getattr(self, name).release()
        self.tex.release()
        self.fbo_ms.release()
        self.ctx.release()
