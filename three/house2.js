// Realistic single-storey house + first-person tour through every room.
// Runs inside headless Chrome; render_house.mjs calls window.renderFrame(i).
//
// Bungalow: entry hall spine, living room, kitchen-diner, bedroom, bathroom.
// Real doorway openings (walk-through), open door leaves, windows with
// frames/sills/curtains and a garden outside, sun with shadow mapping,
// per-room paint + floors, primitive-built furniture with PBR textures.
// Motion: the cleaned-up walk model (look-ahead heading, accel-limited yaw)
// plus tour stops: pause in each room, pan around, turn in place, move on.

import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";

const SEED = 7;
const H = 2.5;                // ceiling height
const T = 0.2;                // wall thickness
const EYE = 1.60;
const SPEED = 1.1;
const FPS = 30;
const FOV = 58;

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rng = mulberry32(SEED);
const angdiff = (a, b) => {
  let d = a - b;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return d;
};

// ---------------------------------------------------------------- renderer
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xa6c5e3);
scene.fog = new THREE.FogExp2(0xbccfdd, 0.0045);

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
if ("environmentIntensity" in scene) scene.environmentIntensity = 0.28;

const camera = new THREE.PerspectiveCamera(FOV, window.innerWidth / window.innerHeight, 0.05, 80);
camera.rotation.order = "YXZ";

// ---------------------------------------------------------------- textures
const texLoader = new THREE.TextureLoader();
function tex(file, { srgb = false } = {}) {
  const t = texLoader.load(`/assets/${file}`);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = 8;
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  t.channel = 0;
  return t;
}
function pbr(name, color, extra = {}) {
  const m = new THREE.MeshStandardMaterial({
    color,
    map: tex(`${name}_Color.jpg`, { srgb: true }),
    normalMap: tex(`${name}_NormalGL.jpg`),
    roughnessMap: tex(`${name}_Roughness.jpg`),
    ...extra,
  });
  m.shadowSide = THREE.DoubleSide;
  return m;
}

// per-room wall paint (same plaster maps, different tint)
const paint = (c) => pbr("Plaster001", c, { normalScale: new THREE.Vector2(0.55, 0.55) });
const MAT = {
  LIV: paint(0xe7dfd1), KIT: paint(0xeae2d2), HALL: paint(0xe0d8ca),
  BED: paint(0xccd5c9), BATH: paint(0xdfe7e9), OUT: paint(0xd7d1c5),
  ceil: paint(0xf2f0ea),
  floorWood: pbr("WoodFloor041", 0xd9c6ab),
  floorCarpet: pbr("Carpet008", 0xcabca6, { aoMap: tex("Carpet008_AmbientOcclusion.jpg"), aoMapIntensity: 0.7 }),
  floorTileK: pbr("Tiles074", 0xd8d4cc),
  floorTileB: pbr("Tiles101", 0xdfe4e5),
  rug: pbr("Carpet004", 0x8f7a63, { aoMap: tex("Carpet004_AmbientOcclusion.jpg") }),
  wood: pbr("Wood051", 0xd8b98e, { roughness: 0.85 }),
  grass: pbr("Grass001", 0x87a066, { envMapIntensity: 0.25 }),
  trim: new THREE.MeshStandardMaterial({ color: 0xf4f1ea, roughness: 0.5 }),
  fabricSofa: new THREE.MeshStandardMaterial({ color: 0x76828e, roughness: 1 }),
  fabricChair: new THREE.MeshStandardMaterial({ color: 0x8e8274, roughness: 1 }),
  duvet: new THREE.MeshStandardMaterial({ color: 0xe3ddd2, roughness: 1 }),
  pillow: new THREE.MeshStandardMaterial({ color: 0xf3f0e9, roughness: 1 }),
  curtain: new THREE.MeshStandardMaterial({ color: 0xb5a78e, roughness: 0.95 }),
  counter: new THREE.MeshStandardMaterial({ color: 0x33363a, roughness: 0.3 }),
  metal: new THREE.MeshStandardMaterial({ color: 0xc9ccd0, metalness: 0.8, roughness: 0.35 }),
  brass: new THREE.MeshStandardMaterial({ color: 0xb08d57, metalness: 0.9, roughness: 0.35 }),
  white: new THREE.MeshStandardMaterial({ color: 0xf6f5f2, roughness: 0.25 }),
  tvDark: new THREE.MeshStandardMaterial({ color: 0x0b0c0e, metalness: 0.6, roughness: 0.1 }),
  darkWood: new THREE.MeshStandardMaterial({ color: 0x4a382a, roughness: 0.6 }),
  glass: new THREE.MeshPhysicalMaterial({
    color: 0xdfeaf2, transparent: true, opacity: 0.08, roughness: 0.05, side: THREE.DoubleSide,
  }),
  hedge: new THREE.MeshStandardMaterial({ color: 0x3f5c33, roughness: 1 }),
  leaf: new THREE.MeshStandardMaterial({ color: 0x4d6b3a, roughness: 1 }),
  trunk: new THREE.MeshStandardMaterial({ color: 0x6b4f37, roughness: 0.95 }),
};
for (const m of Object.values(MAT)) m.shadowSide = THREE.DoubleSide;

// ------------------------------------------------------------- quad sinks
function makeQuadSink() {
  const pos = [], nrm = [], uv = [];
  return {
    add(p0, u, v, su = 2.5, sv = 2.5) {
      const uh = new THREE.Vector3().copy(u).normalize();
      const vh = new THREE.Vector3().copy(v).normalize();
      const n = new THREE.Vector3().crossVectors(u, v).normalize();
      const c = [
        p0,
        new THREE.Vector3().addVectors(p0, u),
        new THREE.Vector3().addVectors(p0, u).add(v),
        new THREE.Vector3().addVectors(p0, v),
      ];
      for (const idx of [0, 1, 2, 0, 2, 3]) {
        const p = c[idx];
        pos.push(p.x, p.y, p.z);
        nrm.push(n.x, n.y, n.z);
        uv.push(p.dot(uh) / su, p.dot(vh) / sv);
      }
    },
    build(material, { cast = true, recv = true } = {}) {
      if (!pos.length) return null;
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
      g.setAttribute("normal", new THREE.Float32BufferAttribute(nrm, 3));
      g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
      const mesh = new THREE.Mesh(g, material);
      mesh.castShadow = cast;
      mesh.receiveShadow = recv;
      scene.add(mesh);
      return mesh;
    },
  };
}
const sinks = new Map();
function sink(matKey) {
  if (!sinks.has(matKey)) sinks.set(matKey, makeQuadSink());
  return sinks.get(matKey);
}

// generic box helper (y = bottom)
function box(mat, w, h, d, x, y, z, ry = 0, opts = {}) {
  const g = opts.round
    ? new RoundedBoxGeometry(w, h, d, 3, opts.round)
    : new THREE.BoxGeometry(w, h, d);
  const m = new THREE.Mesh(g, mat);
  m.position.set(x, y + h / 2, z);
  m.rotation.y = ry;
  m.castShadow = opts.cast !== false;
  m.receiveShadow = opts.recv !== false;
  scene.add(m);
  return m;
}
function cyl(mat, r1, r2, h, x, y, z, seg = 20) {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(r1, r2, h, seg), mat);
  m.position.set(x, y + h / 2, z);
  m.castShadow = true;
  m.receiveShadow = true;
  scene.add(m);
  return m;
}

// ------------------------------------------------------------------ walls
// wall: dir 'x' -> runs along x, thickness band z in [at, at+T]
//       dir 'z' -> runs along z, thickness band x in [at, at+T]
// faces: neg = face at the lower coord side, pos = other. Each face lists
// {a, b, room} sub-ranges (gaps behind abutting walls are skipped).
// openings: {a, b, y0, y1, kind: 'door'|'doorC'|'window'|'winHi'}
const WALLS = [
  { dir: "x", at: -0.2, faces: { neg: [{ a: -0.2, b: 11.8, room: "OUT" }],
      pos: [{ a: 0, b: 4.8, room: "LIV" }, { a: 5.0, b: 6.6, room: "HALL" }, { a: 6.8, b: 11.6, room: "KIT" }] },
    openings: [
      { a: 5.25, b: 6.15, y0: 0, y1: 2.06, kind: "doorC" },
      { a: 1.3, b: 3.5, y0: 0.9, y1: 2.1, kind: "window" },
      { a: 8.0, b: 9.4, y0: 0.9, y1: 2.1, kind: "window" },
      { a: 10.0, b: 11.2, y0: 0.9, y1: 2.1, kind: "window" },
    ] },
  { dir: "x", at: 4.2, faces: { neg: [{ a: 6.8, b: 11.6, room: "KIT" }],
      pos: [{ a: 6.8, b: 9.2, room: "BATH" }, { a: 9.4, b: 11.8, room: "OUT" }] }, openings: [] },
  { dir: "x", at: 4.2, faces: { neg: [{ a: 0, b: 4.8, room: "LIV" }],
      pos: [{ a: 0, b: 4.8, room: "BED" }] }, openings: [], span: [-0.2, 5.0] },
  { dir: "x", at: 6.4, faces: { neg: [{ a: 5.0, b: 6.6, room: "HALL" }],
      pos: [{ a: 4.8, b: 6.8, room: "OUT" }] },
    openings: [{ a: 5.4, b: 6.2, y0: 0.9, y1: 2.1, kind: "window" }] },
  { dir: "x", at: 6.6, faces: { neg: [{ a: 6.8, b: 9.2, room: "BATH" }],
      pos: [{ a: 6.8, b: 9.4, room: "OUT" }] },
    openings: [{ a: 7.6, b: 8.6, y0: 1.35, y1: 2.1, kind: "winHi" }] },
  { dir: "x", at: 8.4, faces: { neg: [{ a: 0, b: 4.8, room: "BED" }],
      pos: [{ a: -0.2, b: 5.0, room: "OUT" }] },
    openings: [{ a: 1.4, b: 3.2, y0: 0.9, y1: 2.1, kind: "window" }] },
  { dir: "z", at: -0.2, faces: { neg: [{ a: -0.2, b: 8.6, room: "OUT" }],
      pos: [{ a: 0, b: 4.2, room: "LIV" }, { a: 4.4, b: 8.4, room: "BED" }] },
    openings: [
      { a: 1.6, b: 3.4, y0: 0.9, y1: 2.1, kind: "window" },
      { a: 5.6, b: 7.4, y0: 0.9, y1: 2.1, kind: "window" },
    ] },
  { dir: "z", at: 11.6, faces: { neg: [{ a: 0, b: 4.2, room: "KIT" }],
      pos: [{ a: -0.2, b: 4.4, room: "OUT" }] },
    openings: [{ a: 1.4, b: 3.0, y0: 0.9, y1: 2.1, kind: "window" }] },
  { dir: "z", at: 4.8, span: [0, 4.2], faces: { neg: [{ a: 0, b: 4.2, room: "LIV" }],
      pos: [{ a: 0, b: 4.2, room: "HALL" }] },
    openings: [{ a: 1.3, b: 2.2, y0: 0, y1: 2.06, kind: "door" }] },
  { dir: "z", at: 6.6, span: [0, 4.2], faces: { neg: [{ a: 0, b: 4.2, room: "HALL" }],
      pos: [{ a: 0, b: 4.2, room: "KIT" }] },
    openings: [{ a: 1.3, b: 2.2, y0: 0, y1: 2.06, kind: "door" }] },
  { dir: "z", at: 4.8, span: [4.4, 6.4], faces: { neg: [{ a: 4.4, b: 6.4, room: "BED" }],
      pos: [{ a: 4.4, b: 6.4, room: "HALL" }] },
    openings: [{ a: 4.85, b: 5.75, y0: 0, y1: 2.06, kind: "door" }] },
  { dir: "z", at: 4.8, span: [6.6, 8.6], faces: { neg: [{ a: 6.6, b: 8.4, room: "BED" }],
      pos: [{ a: 6.4, b: 8.6, room: "OUT" }] },
    openings: [{ a: 7.2, b: 8.0, y0: 0.9, y1: 2.1, kind: "window" }] },
  { dir: "z", at: 6.6, span: [4.4, 6.4], faces: { neg: [{ a: 4.4, b: 6.4, room: "HALL" }],
      pos: [{ a: 4.4, b: 6.4, room: "BATH" }] },
    openings: [{ a: 4.85, b: 5.75, y0: 0, y1: 2.06, kind: "door" }] },
  { dir: "z", at: 9.2, span: [4.4, 6.8], faces: { neg: [{ a: 4.4, b: 6.6, room: "BATH" }],
      pos: [{ a: 4.2, b: 6.8, room: "OUT" }] }, openings: [] },
];

// emit one rectangle of a wall face
function faceRect(w, side, a, b, y0, y1, room) {
  if (b - a < 1e-4 || y1 - y0 < 1e-4) return;
  const s = sink(room === "OUT" ? "OUT" : room);
  const at = side === "neg" ? w.at : w.at + T;
  const v = new THREE.Vector3(0, y1 - y0, 0);
  if (w.dir === "x") {
    if (side === "neg") s.add(new THREE.Vector3(b, y0, at), new THREE.Vector3(-(b - a), 0, 0), v);
    else s.add(new THREE.Vector3(a, y0, at), new THREE.Vector3(b - a, 0, 0), v);
  } else {
    if (side === "neg") s.add(new THREE.Vector3(at, y0, a), new THREE.Vector3(0, 0, b - a), v);
    else s.add(new THREE.Vector3(at, y0, b), new THREE.Vector3(0, 0, -(b - a)), v);
  }
}

function wallXZ(w, along, off) { // point on wall: coordinate along run + across band
  return w.dir === "x" ? [along, w.at + off] : [w.at + off, along];
}

const doorLeaves = []; // filled from wall pass, leaves added after

for (const w of WALLS) {
  const ops = [...w.openings].sort((p, q) => p.a - q.a);
  for (const side of ["neg", "pos"]) {
    for (const fr of w.faces[side]) {
      // vertical strips between openings
      let cur = fr.a;
      for (const o of ops) {
        const a = Math.max(o.a, fr.a), b = Math.min(o.b, fr.b);
        if (b <= a) continue;
        faceRect(w, side, cur, a, 0, H, fr.room);
        if (o.y0 > 0) faceRect(w, side, a, b, 0, o.y0, fr.room);
        faceRect(w, side, a, b, o.y1, H, fr.room);
        cur = b;
      }
      faceRect(w, side, cur, fr.b, 0, H, fr.room);
      // baseboard: skip at door openings
      const doorSpans = ops.filter((o) => o.y0 === 0);
      let bcur = fr.a;
      const bbAt = side === "neg" ? w.at - 0.009 : w.at + T + 0.009;
      const emitBB = (a, b) => {
        if (b - a < 0.05 || fr.room === "OUT" || fr.room === "BATH") return;
        const len = b - a, mid = (a + b) / 2;
        const [px, pz] = wallXZ(w, mid, side === "neg" ? -0.009 : T + 0.009);
        box(MAT.trim, w.dir === "x" ? len : 0.016, 0.09, w.dir === "x" ? 0.016 : len,
          px, 0, pz, 0, { cast: false });
      };
      for (const o of doorSpans) {
        const a = Math.max(o.a, fr.a), b = Math.min(o.b, fr.b);
        if (b <= a) continue;
        emitBB(bcur, a - 0.06);
        bcur = b + 0.06;
      }
      emitBB(bcur, fr.b);
      void bbAt;
    }
  }
  // openings: frames, reveals, glass, leaves
  for (const o of ops) {
    const mid = (o.a + o.b) / 2;
    if (o.kind === "window" || o.kind === "winHi") {
      const s = sink("trimQ");
      const t2 = T;
      if (w.dir === "x") {
        s.add(new THREE.Vector3(o.a, o.y0, w.at + t2), new THREE.Vector3(0, 0, -t2), new THREE.Vector3(0, o.y1 - o.y0, 0), 1, 1);
        s.add(new THREE.Vector3(o.b, o.y0, w.at), new THREE.Vector3(0, 0, t2), new THREE.Vector3(0, o.y1 - o.y0, 0), 1, 1);
        s.add(new THREE.Vector3(o.a, o.y1, w.at), new THREE.Vector3(o.b - o.a, 0, 0), new THREE.Vector3(0, 0, t2), 1, 1);
        s.add(new THREE.Vector3(o.a, o.y0, w.at + t2), new THREE.Vector3(o.b - o.a, 0, 0), new THREE.Vector3(0, 0, -t2), 1, 1);
      } else {
        s.add(new THREE.Vector3(w.at, o.y0, o.a), new THREE.Vector3(t2, 0, 0), new THREE.Vector3(0, o.y1 - o.y0, 0), 1, 1);
        s.add(new THREE.Vector3(w.at + t2, o.y0, o.b), new THREE.Vector3(-t2, 0, 0), new THREE.Vector3(0, o.y1 - o.y0, 0), 1, 1);
        s.add(new THREE.Vector3(w.at + t2, o.y1, o.a), new THREE.Vector3(0, 0, o.b - o.a), new THREE.Vector3(-t2, 0, 0), 1, 1);
        s.add(new THREE.Vector3(w.at, o.y0, o.a), new THREE.Vector3(0, 0, o.b - o.a), new THREE.Vector3(t2, 0, 0), 1, 1);
      }
      // frame: border strips + center mullion + glass, at mid thickness
      const fd = 0.055, fw = 0.06;
      const len = o.b - o.a, hgt = o.y1 - o.y0;
      const [cx, cz] = wallXZ(w, mid, T / 2);
      const ry = w.dir === "x" ? 0 : Math.PI / 2;
      box(MAT.trim, len, fw, fd, cx, o.y1 - fw, cz, ry);
      box(MAT.trim, len, fw, fd, cx, o.y0, cz, ry);
      const [lx, lz] = wallXZ(w, o.a + fw / 2, T / 2);
      const [rx, rz] = wallXZ(w, o.b - fw / 2, T / 2);
      box(MAT.trim, fw, hgt, fd, lx, o.y0, lz, ry);
      box(MAT.trim, fw, hgt, fd, rx, o.y0, rz, ry);
      box(MAT.trim, 0.045, hgt - 2 * fw, fd - 0.01, cx, o.y0 + fw, cz, ry);
      const glass = new THREE.Mesh(new THREE.PlaneGeometry(len - 0.1, hgt - 0.1), MAT.glass);
      glass.position.set(cx, (o.y0 + o.y1) / 2, cz);
      glass.rotation.y = ry;
      scene.add(glass);
      // sill board proud into the room on the non-OUT side(s)
      for (const side of ["neg", "pos"]) {
        const fr = w.faces[side].find((f) => mid > f.a && mid < f.b);
        if (!fr || fr.room === "OUT") continue;
        const [sx, sz] = wallXZ(w, mid, side === "neg" ? -0.06 : T + 0.06);
        box(MAT.trim, w.dir === "x" ? len + 0.12 : 0.14, 0.03,
          w.dir === "x" ? 0.14 : len + 0.12, sx, o.y0 - 0.03, sz, 0, { cast: false });
      }
    } else {
      // door frame: jambs + head (covers the reveal), white trim
      const [cx, cz] = wallXZ(w, mid, T / 2);
      const ry = w.dir === "x" ? 0 : Math.PI / 2;
      const [ax, az] = wallXZ(w, o.a - 0.033, T / 2);
      const [bx, bz] = wallXZ(w, o.b + 0.033, T / 2);
      box(MAT.trim, 0.066, 2.09, T + 0.05, ax, 0, az, ry);
      box(MAT.trim, 0.066, 2.09, T + 0.05, bx, 0, bz, ry);
      box(MAT.trim, o.b - o.a + 0.132, 0.07, T + 0.05, cx, 2.06, cz, ry);
      if (o.kind === "doorC") {
        // closed front door: white slab + inset panels + knob
        box(MAT.trim, o.b - o.a, 2.05, 0.055, cx, 0, cz, ry);
        for (const py of [0.35, 1.15]) {
          box(MAT.trim, (o.b - o.a) * 0.62, 0.62, 0.075, cx, py, cz, ry, { cast: false });
        }
        const [kx, kz] = wallXZ(w, o.b - 0.12, T / 2 + 0.055);
        cyl(MAT.brass, 0.032, 0.032, 0.04, kx, 1.0, kz, 16).rotation.x = Math.PI / 2;
      } else {
        doorLeaves.push({ w, o });
      }
    }
  }
}
// interior open door leaves (hand-tuned hinge + swing so nothing is blocked)
const LEAF_POSE = {
  "4.8|1.3": { hx: 4.9, hz: 1.3, ry: Math.PI / 2 + 0.22 },     // living
  "6.6|1.3": { hx: 6.7, hz: 1.3, ry: Math.PI / 2 - 0.22 },     // kitchen
  "4.8|4.85": { hx: 4.9, hz: 5.75, ry: -Math.PI / 2 - 0.22 },  // bedroom
  "6.6|4.85": { hx: 6.7, hz: 5.75, ry: -Math.PI / 2 + 0.06 },  // bathroom (hugs wall)
};
for (const { w, o } of doorLeaves) {
  const pose = LEAF_POSE[`${w.at}|${o.a}`];
  if (!pose) continue;
  const g = new THREE.Group();
  g.position.set(pose.hx, 0, pose.hz);
  g.rotation.y = pose.ry;
  const leafW = o.b - o.a - 0.04;
  const leaf = new THREE.Mesh(new THREE.BoxGeometry(leafW, 2.02, 0.042), MAT.wood);
  leaf.position.set(leafW / 2, 1.02, 0);
  leaf.castShadow = leaf.receiveShadow = true;
  g.add(leaf);
  for (const px of [-0.19, 0.19]) {
    for (const py of [-0.62, 0.02, 0.66]) {
      const p = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.5, 0.012), MAT.wood);
      p.position.set(leafW / 2 + px, 1.02 + py, 0.027);
      g.add(p);
    }
  }
  const kn = new THREE.Mesh(new THREE.CylinderGeometry(0.009, 0.009, 0.11, 10), MAT.brass);
  kn.rotation.z = Math.PI / 2;
  kn.position.set(leafW - 0.07, 1.0, 0.05);
  g.add(kn);
  scene.add(g);
}

// -------------------------------------------------------- floors, ceilings
const ROOMS = {
  LIV: { x0: 0, x1: 4.8, z0: 0, z1: 4.2, floor: "floorWood", fsu: 2.2 },
  HALL: { x0: 5.0, x1: 6.6, z0: 0, z1: 6.4, floor: "floorWood", fsu: 2.2 },
  KIT: { x0: 6.8, x1: 11.6, z0: 0, z1: 4.2, floor: "floorWood", fsu: 2.2 },
  BED: { x0: 0, x1: 4.8, z0: 4.4, z1: 8.4, floor: "floorCarpet", fsu: 2.6 },
  BATH: { x0: 6.8, x1: 9.2, z0: 4.4, z1: 6.6, floor: "floorTileK", fsu: 0.85 },
};
for (const r of Object.values(ROOMS)) {
  const fs = sink(r.floor), cs = sink("ceil");
  const dx = r.x1 - r.x0, dz = r.z1 - r.z0;
  fs.add(new THREE.Vector3(r.x0, 0, r.z0), new THREE.Vector3(0, 0, dz), new THREE.Vector3(dx, 0, 0), r.fsu, r.fsu);
  cs.add(new THREE.Vector3(r.x0, H, r.z0), new THREE.Vector3(dx, 0, 0), new THREE.Vector3(0, 0, dz), 2.5, 2.5);
}
// floor under door openings (thresholds)
for (const [x, z, dir] of [[4.9, 1.75, "z"], [6.7, 1.75, "z"], [4.9, 5.3, "z"], [6.7, 5.3, "z"], [5.7, -0.1, "x"]]) {
  const fs = sink("floorWood");
  if (dir === "z") fs.add(new THREE.Vector3(x - T / 2 - 0.01, 0.001, z - 0.47), new THREE.Vector3(0, 0, 0.94), new THREE.Vector3(T + 0.02, 0, 0), 2.2, 2.2);
  else fs.add(new THREE.Vector3(x - 0.47, 0.001, z - T / 2 - 0.01), new THREE.Vector3(0, 0, T + 0.02), new THREE.Vector3(0.94, 0, 0), 2.2, 2.2);
}
// roof slab above ceilings: blocks the sun from leaking through wall bands
{
  const roof = new THREE.Mesh(new THREE.BoxGeometry(12.4, 0.1, 9.2), MAT.OUT);
  roof.position.set(5.8, H + 0.09, 4.2);
  roof.castShadow = true;
  scene.add(roof);
}

// ------------------------------------------------------------------ garden
{
  const gs = sink("grass");
  gs.add(new THREE.Vector3(-24, -0.02, -22), new THREE.Vector3(0, 0, 52), new THREE.Vector3(58, 0, 0), 3.2, 3.2);
  const hedgeAt = (x, z, w, d) =>
    box(MAT.hedge, w, 0.65 + rng() * 0.45, d, x, -0.02, z, 0, { round: 0.08 });
  for (let x = -2; x <= 13; x += 2.1) hedgeAt(x, -4.4, 1.7 + rng() * 0.4, 0.75);
  for (let z = -3; z <= 10; z += 2.2) hedgeAt(-4.3, z, 0.75, 1.8 + rng() * 0.4);
  const tree = (x, z, s) => {
    cyl(MAT.trunk, 0.13 * s, 0.17 * s, 1.9 * s, x, -0.02, z, 10);
    for (const [ox, oy, oz, r] of [[0, 2.6, 0, 1.35], [0.8, 2.1, 0.3, 0.95], [-0.7, 2.2, -0.4, 0.9]]) {
      const m = new THREE.Mesh(new THREE.IcosahedronGeometry(r * s, 1), MAT.leaf);
      m.position.set(x + ox * s, oy * s, z + oz * s);
      m.castShadow = true;
      scene.add(m);
    }
  };
  tree(-3.6, -2.2, 1.1); tree(14.6, 1.4, 1.0); tree(13.8, 8.0, 0.85); tree(-3.2, 11.0, 0.9);
}

// ---------------------------------------------------------------- lighting
{
  const sun = new THREE.DirectionalLight(0xfff0da, 4.2);
  sun.position.set(2.5, 9.5, -12.5);
  sun.target.position.set(5.8, 0, 4.0);
  scene.add(sun.target);
  sun.castShadow = true;
  sun.shadow.mapSize.set(4096, 4096);
  sun.shadow.camera.left = -9.5; sun.shadow.camera.right = 9.5;
  sun.shadow.camera.top = 9.5; sun.shadow.camera.bottom = -9.5;
  sun.shadow.camera.near = 1; sun.shadow.camera.far = 45;
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.035;
  sun.shadow.camera.updateProjectionMatrix(); // frustum extents changed above
  scene.add(sun);
  scene.add(new THREE.HemisphereLight(0xcfe0f0, 0x8b7d6a, 0.28));
}
function ceilingLamp(x, z, intensity = 2.0, dist = 9) {
  cyl(MAT.white, 0.16, 0.16, 0.05, x, H - 0.05, z, 24);
  const disc = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.13, 0.012, 24),
    new THREE.MeshStandardMaterial({ color: 0xfff6e6, emissive: 0xffe9c8, emissiveIntensity: 1.7 }));
  disc.position.set(x, H - 0.062, z);
  scene.add(disc);
  const pl = new THREE.PointLight(0xffe6c4, intensity, dist, 2);
  pl.position.set(x, H - 0.28, z);
  scene.add(pl);
  return pl;
}
ceilingLamp(2.4, 2.1, 2.2);
ceilingLamp(9.2, 2.1, 2.2);
ceilingLamp(2.4, 6.4, 2.0);
ceilingLamp(8.0, 5.5, 1.6, 6);
ceilingLamp(5.8, 1.6, 1.4, 6);
ceilingLamp(5.8, 4.8, 1.4, 6);

// ----------------------------------------------------------------- living
{
  const r = MAT;
  const rug = sink("rug");
  rug.add(new THREE.Vector3(0.9, 0.006, 1.6), new THREE.Vector3(0, 0, 2.2), new THREE.Vector3(1.8, 0, 0), 2.4, 2.4);
  // sofa along the west wall, under the window, facing the TV wall
  box(r.fabricSofa, 0.85, 0.42, 2.1, 0.705, 0.1, 2.65, 0, { round: 0.05 });
  box(r.fabricSofa, 0.2, 0.6, 2.1, 0.38, 0.5, 2.65, 0, { round: 0.05 });
  box(r.fabricSofa, 0.85, 0.62, 0.24, 0.705, 0.3, 1.72, 0, { round: 0.05 });
  box(r.fabricSofa, 0.85, 0.62, 0.24, 0.705, 0.3, 3.58, 0, { round: 0.05 });
  for (const pz of [2.34, 2.96]) box(r.pillow, 0.58, 0.15, 0.56, 0.78, 0.52, pz, 0, { round: 0.06 });
  box(r.pillow, 0.14, 0.4, 0.4, 0.52, 0.55, 1.98, 0.15, { round: 0.06 });
  box(r.pillow, 0.14, 0.4, 0.4, 0.52, 0.55, 3.28, -0.2, { round: 0.06 });
  // coffee table (long axis along the sofa)
  box(r.wood, 0.5, 0.045, 0.95, 1.75, 0.36, 2.7);
  for (const [lx, lz] of [[1.57, 2.3], [1.93, 2.3], [1.57, 3.1], [1.93, 3.1]]) {
    box(r.darkWood, 0.045, 0.36, 0.045, lx, 0, lz);
  }
  box(r.darkWood, 0.3, 0.025, 0.2, 1.78, 0.405, 2.55, 0.3, { cast: false }); // book
  // tv unit + tv on the east wall (south of the doorway)
  box(r.wood, 0.4, 0.42, 1.5, 4.58, 0, 3.4);
  box(r.tvDark, 0.045, 0.75, 1.3, 4.66, 0.5, 3.4);
  // bookshelf on the north wall, east of the window
  box(r.wood, 0.9, 1.8, 0.26, 4.22, 0, 0.34);
  for (let i = 0; i < 4; i++) {
    box(r.wood, 0.86, 0.022, 0.24, 4.22, 0.32 + i * 0.42, 0.34, 0, { cast: false });
    let bx = 3.82 + rng() * 0.06;
    while (bx < 4.56) {
      const bw = 0.035 + rng() * 0.035, bh = 0.24 + rng() * 0.1;
      box(new THREE.MeshStandardMaterial({
        color: new THREE.Color().setHSL(rng(), 0.35 + rng() * 0.3, 0.38 + rng() * 0.25), roughness: 0.8,
      }), bw, bh, 0.19, bx + bw / 2, 0.345 + i * 0.42, 0.35, 0, { cast: false });
      bx += bw + 0.006 + rng() * 0.02;
    }
  }
  // armchair (NW corner, angled toward the TV) + floor lamp behind it
  const ary = 1.02;
  box(r.fabricChair, 0.72, 0.4, 0.72, 1.5, 0.09, 1.5, ary, { round: 0.05 });
  box(r.fabricChair, 0.72, 0.55, 0.2, 1.5 - Math.sin(ary) * 0.27, 0.45, 1.5 - Math.cos(ary) * 0.27, ary, { round: 0.05 });
  cyl(r.metal, 0.014, 0.014, 1.35, 0.45, 0, 0.6, 10);
  cyl(r.metal, 0.16, 0.16, 0.02, 0.45, 0.02, 0.6, 18);
  const shade = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.18, 0.26, 20, 1, true),
    new THREE.MeshStandardMaterial({ color: 0xf0e4cd, emissive: 0xffdfae, emissiveIntensity: 0.85, side: THREE.DoubleSide }));
  shade.position.set(0.45, 1.42, 0.6);
  scene.add(shade);
  const lampL = new THREE.PointLight(0xffddaa, 1.3, 7, 2);
  lampL.position.set(0.45, 1.4, 0.6);
  lampL.castShadow = true;
  lampL.shadow.mapSize.set(1024, 1024);
  lampL.shadow.bias = -0.004;
  scene.add(lampL);
}
// wall art (procedural canvases)
function art(x, z, ry, w = 0.56, h = 0.72) {
  const cv = document.createElement("canvas");
  cv.width = cv.height = 256;
  const g = cv.getContext("2d");
  const hue = 15 + rng() * 220;
  const grad = g.createLinearGradient(0, 0, 256, 256);
  grad.addColorStop(0, `hsl(${hue}, 28%, 74%)`);
  grad.addColorStop(1, `hsl(${hue + 45}, 22%, 46%)`);
  g.fillStyle = grad;
  g.fillRect(0, 0, 256, 256);
  for (let i = 0; i < 5; i++) {
    g.fillStyle = `hsla(${hue + rng() * 90 - 45}, 32%, ${32 + rng() * 48}%, 0.38)`;
    g.beginPath();
    g.ellipse(rng() * 256, rng() * 256, 22 + rng() * 60, 16 + rng() * 46, rng() * 3, 0, 7);
    g.fill();
  }
  const t = new THREE.CanvasTexture(cv);
  t.colorSpace = THREE.SRGBColorSpace;
  const grp = new THREE.Group();
  const bd = 0.032;
  for (const [fx, fy, fw2, fh2] of [
    [0, h / 2 - bd / 2, w, bd], [0, -h / 2 + bd / 2, w, bd],
    [-w / 2 + bd / 2, 0, bd, h - 2 * bd], [w / 2 - bd / 2, 0, bd, h - 2 * bd]]) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(fw2, fh2, 0.028), MAT.darkWood);
    m.position.set(fx, fy, 0.014);
    grp.add(m);
  }
  const c = new THREE.Mesh(new THREE.PlaneGeometry(w - 2 * bd, h - 2 * bd),
    new THREE.MeshStandardMaterial({ map: t, roughness: 0.85 }));
  c.position.z = 0.011;
  grp.add(c);
  grp.position.set(x, 1.55, z);
  grp.rotation.y = ry;
  scene.add(grp);
}
art(2.3, 4.185, Math.PI, 0.9, 0.62);       // living, south wall
art(4.985, 3.1, -Math.PI / 2, 0.5, 0.66);   // hall west wall
art(4.985, 4.0, -Math.PI / 2, 0.5, 0.66);
art(2.0, 8.385, Math.PI, 0.62, 0.5);        // bedroom south wall
art(8.4, 4.185, Math.PI, 0.7, 0.55);        // kitchen south wall
art(9.5, 4.185, Math.PI, 0.55, 0.7);
{ // kitchen wall clock (east wall)
  const clk = new THREE.Group();
  const face = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 0.03, 32), MAT.white);
  face.rotation.z = Math.PI / 2;
  clk.add(face);
  const rim = new THREE.Mesh(new THREE.TorusGeometry(0.16, 0.014, 8, 32), MAT.darkWood);
  rim.rotation.y = Math.PI / 2;
  clk.add(rim);
  const hand = (len, wd, ang) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(0.02, wd, len), MAT.darkWood);
    m.position.x = -0.017;
    m.rotation.x = ang;
    m.translateZ(len / 2);
    clk.add(m);
  };
  hand(0.1, 0.012, 0.9);
  hand(0.13, 0.008, -1.8);
  clk.position.set(11.58, 1.85, 3.5);
  scene.add(clk);
}
{ // fruit bowl on the dining table
  cyl(MAT.white, 0.13, 0.09, 0.06, 10.2, 0.745, 2.1, 20);
  const fruit = [[0xd9822b, -0.05, 0.02], [0xc23b22, 0.05, 0.03], [0x9aa832, 0, -0.05], [0xd9a13b, 0.01, 0.07]];
  for (const [c, ox, oz] of fruit) {
    const f = new THREE.Mesh(new THREE.SphereGeometry(0.038, 12, 10),
      new THREE.MeshStandardMaterial({ color: c, roughness: 0.55 }));
    f.position.set(10.2 + ox, 0.835, 2.1 + oz);
    f.castShadow = true;
    scene.add(f);
  }
}

// ---------------------------------------------------------------- kitchen
{
  const r = MAT;
  box(r.wood, 4.55, 0.86, 0.6, 9.22, 0, 0.51);                 // base run
  box(r.counter, 4.62, 0.04, 0.66, 9.22, 0.86, 0.52);          // worktop
  box(r.white, 4.55, 0.55, 0.03, 9.22, 0.92, 0.225, 0, { cast: false }); // backsplash
  for (let x = 7.2; x < 11.4; x += 0.62) {                     // cabinet handles
    box(r.metal, 0.14, 0.018, 0.02, x, 0.78, 0.82, 0, { cast: false });
  }
  box(r.wood, 1.0, 0.66, 0.34, 7.45, 1.5, 0.38);               // uppers (skip window)
  box(r.wood, 1.9, 0.66, 0.34, 10.55, 1.5, 0.38);
  // sink + faucet under window
  box(r.metal, 0.56, 0.02, 0.4, 8.7, 0.875, 0.5, 0, { cast: false });
  cyl(r.metal, 0.016, 0.016, 0.24, 8.55, 0.88, 0.36, 10);
  const spout = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.2, 8), r.metal);
  spout.rotation.x = Math.PI / 2;
  spout.position.set(8.55, 1.12, 0.46);
  scene.add(spout);
  // hob
  box(r.tvDark, 0.58, 0.012, 0.5, 10.35, 0.865, 0.5, 0, { cast: false });
  for (const [ox, oz] of [[-0.14, -0.11], [0.14, -0.11], [-0.14, 0.12], [0.14, 0.12]]) {
    cyl(r.counter, 0.055, 0.055, 0.006, 10.35 + ox, 0.877, 0.5 + oz, 16);
  }
  // fridge
  box(r.metal, 0.75, 1.82, 0.68, 7.28, 0, 3.7);
  box(r.metal, 0.02, 0.9, 0.04, 7.62, 0.6, 3.34, 0, { cast: false });
  // dining table + chairs
  box(r.wood, 1.25, 0.045, 1.0, 10.2, 0.7, 2.1);
  for (const [lx, lz] of [[9.66, 1.68], [10.74, 1.68], [9.66, 2.52], [10.74, 2.52]]) {
    box(r.darkWood, 0.05, 0.7, 0.05, lx, 0, lz);
  }
  const chair = (x, z, ry) => {
    box(r.wood, 0.42, 0.04, 0.42, x, 0.44, z, ry);
    const bx = x - Math.sin(ry) * 0.19, bz = z - Math.cos(ry) * 0.19;
    box(r.wood, 0.42, 0.45, 0.035, bx, 0.48, bz, ry);
    for (const [ox, oz] of [[-0.18, -0.18], [0.18, -0.18], [-0.18, 0.18], [0.18, 0.18]]) {
      const c = Math.cos(ry), s = Math.sin(ry);
      box(r.darkWood, 0.035, 0.44, 0.035, x + ox * c + oz * s, 0, z - ox * s + oz * c);
    }
  };
  chair(10.2, 1.35, Math.PI);
  chair(10.2, 2.85, 0);
  chair(9.5, 2.1, Math.PI / 2);
  chair(10.9, 2.1, -Math.PI / 2);
  // pendant over the table
  cyl(r.metal, 0.006, 0.006, 0.55, 10.2, 1.95, 2.1, 8);
  const shade = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.19, 0.2, 24, 1, true),
    new THREE.MeshStandardMaterial({ color: 0x2f3236, roughness: 0.5, side: THREE.DoubleSide }));
  shade.position.set(10.2, 1.85, 2.1);
  scene.add(shade);
  const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.035, 12, 8),
    new THREE.MeshStandardMaterial({ emissive: 0xffe2b0, emissiveIntensity: 2.6, color: 0x111111 }));
  bulb.position.set(10.2, 1.83, 2.1);
  scene.add(bulb);
  const pd = new THREE.PointLight(0xffdfa8, 1.7, 6.5, 2);
  pd.position.set(10.2, 1.78, 2.1);
  pd.castShadow = true;
  pd.shadow.mapSize.set(1024, 1024);
  pd.shadow.bias = -0.004;
  scene.add(pd);
  // plant on the counter
  cyl(r.darkWood, 0.07, 0.055, 0.11, 7.35, 0.88, 0.42, 12);
  const pl1 = new THREE.Mesh(new THREE.IcosahedronGeometry(0.11, 1), r.leaf);
  pl1.position.set(7.35, 1.08, 0.42);
  scene.add(pl1);
}

// ---------------------------------------------------------------- bedroom
{
  const r = MAT;
  box(r.wood, 1.7, 0.3, 2.15, 2.2, 0, 5.58);                    // bed frame
  box(r.wood, 1.7, 0.85, 0.06, 2.2, 0, 4.52);                   // headboard
  box(r.white, 1.6, 0.2, 2.0, 2.2, 0.3, 5.6, 0, { round: 0.05 }); // mattress
  box(r.duvet, 1.66, 0.16, 1.4, 2.2, 0.48, 5.95, 0, { round: 0.06 });
  for (const px of [1.85, 2.55]) box(r.pillow, 0.58, 0.14, 0.38, px, 0.5, 4.78, 0, { round: 0.05 });
  const stand = (x) => {
    box(r.wood, 0.42, 0.48, 0.38, x, 0, 4.72);
    cyl(r.brass, 0.012, 0.012, 0.2, x, 0.48, 4.72, 8);
    const sh = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.11, 0.13, 16, 1, true),
      new THREE.MeshStandardMaterial({ color: 0xf0e4cd, emissive: 0xffdfae, emissiveIntensity: 0.8, side: THREE.DoubleSide }));
    sh.position.set(x, 0.72, 4.72);
    scene.add(sh);
    const pl = new THREE.PointLight(0xffddaa, 0.45, 4, 2);
    pl.position.set(x, 0.7, 4.72);
    scene.add(pl);
  };
  stand(1.08);
  stand(3.32);
  box(r.wood, 0.38, 2.0, 1.5, 4.59, 0, 7.55);                   // wardrobe (east wall)
  box(r.metal, 0.02, 0.5, 0.025, 4.38, 0.85, 7.2, 0, { cast: false });
  box(r.metal, 0.02, 0.5, 0.025, 4.38, 0.85, 7.9, 0, { cast: false });
  box(r.wood, 1.1, 0.8, 0.4, 3.9, 0, 8.18);                     // dresser (south wall)
  const rug = sink("rug");
  rug.add(new THREE.Vector3(1.6, 0.006, 6.9), new THREE.Vector3(0, 0, 1.05), new THREE.Vector3(1.9, 0, 0), 2.4, 2.4);
}

// --------------------------------------------------------------- bathroom
{
  const r = MAT;
  box(r.wood, 0.95, 0.78, 0.46, 7.5, 0, 4.73);                  // vanity
  box(r.white, 1.0, 0.045, 0.5, 7.5, 0.78, 4.74);
  box(r.white, 0.5, 0.11, 0.34, 7.5, 0.825, 4.72, 0, { round: 0.04 });
  cyl(r.metal, 0.013, 0.013, 0.2, 7.5, 0.83, 4.58, 10);
  const mir = new THREE.Mesh(new THREE.PlaneGeometry(0.7, 0.85),
    new THREE.MeshStandardMaterial({ color: 0xcfd8dc, metalness: 0.95, roughness: 0.06 }));
  mir.position.set(7.5, 1.55, 4.515);
  scene.add(mir);
  box(MAT.metal, 0.74, 0.03, 0.03, 7.5, 1.09, 4.53, 0, { cast: false }); // mirror shelf
  // toilet
  box(r.white, 0.42, 0.5, 0.2, 8.85, 0, 5.32, Math.PI / 2);
  box(r.white, 0.4, 0.38, 0.55, 8.72, 0, 5.32, 0, { round: 0.09 });
  box(r.white, 0.42, 0.05, 0.58, 8.72, 0.4, 5.32, 0, { round: 0.02 });
  // bathtub along the south wall
  box(r.white, 1.85, 0.55, 0.78, 7.85, 0, 6.14, 0, { round: 0.07 });
  box(new THREE.MeshStandardMaterial({ color: 0xdfe7ea, roughness: 0.15 }),
    1.6, 0.02, 0.55, 7.85, 0.44, 6.14, 0, { cast: false });
  cyl(r.metal, 0.014, 0.014, 0.18, 7.2, 0.55, 6.4, 10);
  // towels
  box(r.metal, 0.02, 0.02, 0.6, 6.86, 1.15, 5.3, 0, { cast: false });
  box(new THREE.MeshStandardMaterial({ color: 0xb9ccd4, roughness: 1 }), 0.03, 0.5, 0.55, 6.88, 0.68, 5.3);
}

// -------------------------------------------------------------------- hall
{
  const r = MAT;
  box(r.wood, 0.3, 0.045, 1.0, 6.42, 0.72, 3.35);               // console table
  for (const [lx, lz] of [[6.42, 2.92], [6.42, 3.78]]) {
    box(r.darkWood, 0.04, 0.72, 0.04, lx, 0, lz);
    box(r.darkWood, 0.04, 0.72, 0.04, lx - 0.0, 0, lz); // (twin legs hidden against wall)
  }
  cyl(r.darkWood, 0.06, 0.045, 0.14, 6.42, 0.765, 3.15, 12);    // vase
  for (let i = 0; i < 5; i++) {
    const st = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.004, 0.3 + rng() * 0.15, 6), MAT.leaf);
    st.position.set(6.4 + (rng() - 0.5) * 0.05, 1.05, 3.15 + (rng() - 0.5) * 0.05);
    st.rotation.z = (rng() - 0.5) * 0.5;
    st.rotation.x = (rng() - 0.5) * 0.5;
    scene.add(st);
  }
  const mat2 = new THREE.Mesh(new THREE.PlaneGeometry(0.95, 0.6),
    new THREE.MeshStandardMaterial({ color: 0x4c443a, roughness: 1 }));
  mat2.rotation.x = -Math.PI / 2;
  mat2.position.set(5.7, 0.004, 0.55);
  mat2.receiveShadow = true;
  scene.add(mat2);
  // coat hooks
  box(r.wood, 0.5, 0.07, 0.02, 5.05, 1.6, 0.7, Math.PI / 2, { cast: false });
  for (const hz of [0.55, 0.7, 0.85]) cyl(r.brass, 0.012, 0.012, 0.05, 5.06, 1.56, hz, 8).rotation.z = Math.PI / 2;
}

// build all quad sinks
const SINK_MAT = {
  LIV: MAT.LIV, KIT: MAT.KIT, HALL: MAT.HALL, BED: MAT.BED, BATH: MAT.BATH, OUT: MAT.OUT,
  ceil: MAT.ceil, trimQ: MAT.trim, grass: MAT.grass, rug: MAT.rug,
  floorWood: MAT.floorWood, floorCarpet: MAT.floorCarpet, floorTileK: MAT.floorTileK, floorTileB: MAT.floorTileB,
};
for (const [k, s] of sinks) s.build(SINK_MAT[k]);

// curtains beside the bigger windows
function curtains(cx, cz, ry, span) {
  const rail = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, span + 0.5, 8), MAT.brass);
  rail.rotation.z = Math.PI / 2;
  rail.rotation.y = ry;
  rail.position.set(cx, 2.24, cz);
  scene.add(rail);
  for (const s of [-1, 1]) {
    const p = new THREE.Mesh(new RoundedBoxGeometry(0.26, 1.95, 0.1, 3, 0.05), MAT.curtain);
    p.position.set(cx + Math.cos(ry) * s * (span / 2 + 0.1), 1.24, cz - Math.sin(ry) * s * (span / 2 + 0.1));
    p.rotation.y = ry;
    p.castShadow = true;
    scene.add(p);
  }
}
curtains(2.4, 0.16, 0, 2.2);      // living north
curtains(0.16, 2.5, Math.PI / 2, 1.8);  // living west
curtains(0.16, 6.5, Math.PI / 2, 1.8);  // bedroom west
curtains(2.3, 8.24, 0, 1.8);      // bedroom south

// --------------------------------------------------------------- tour path
const WP = [
  [5.8, 0.85], [5.8, 1.75], [4.35, 1.75], [3.3, 1.95], [2.4, 2.15],
  [3.3, 1.95], [4.35, 1.75], [5.45, 1.75], [6.15, 1.75], [7.25, 1.75], [8.3, 2.05],
  [7.25, 1.75], [6.15, 1.75], [5.8, 2.0], [5.8, 4.5], [5.8, 5.3],
  [5.45, 5.3], [4.35, 5.3], [3.5, 5.95], [2.7, 7.2],
  [3.5, 5.95], [4.35, 5.3], [5.45, 5.3], [6.15, 5.3], [7.25, 5.3], [8.0, 5.45],
  [7.25, 5.3], [6.15, 5.3], [5.8, 5.1], [5.8, 3.0], [5.8, 1.5],
];
const STOPS = [ // {pt, dur (s), amp (rad)} — matched to nearest path point
  { pt: [2.4, 2.15], dur: 3.4, amp: 1.25 },  // living
  { pt: [8.3, 2.05], dur: 3.2, amp: 1.15 },  // kitchen
  { pt: [2.7, 7.2], dur: 3.2, amp: 1.1 },    // bedroom
  { pt: [8.0, 5.45], dur: 2.4, amp: 0.8 },   // bathroom
];

function roundCorners(pts, r = 0.4, k = 8) {
  const out = [pts[0].clone()];
  for (let i = 1; i < pts.length - 1; i++) {
    const p = pts[i];
    const din = new THREE.Vector3().subVectors(p, pts[i - 1]);
    const dout = new THREE.Vector3().subVectors(pts[i + 1], p);
    const li = din.length(), lo = dout.length();
    din.normalize(); dout.normalize();
    if (din.dot(dout) < -0.99) { out.push(p.clone()); continue; } // reversal apex
    const rr = Math.min(r, 0.42 * Math.min(li, lo));
    if (rr < 0.02 || din.dot(dout) > 0.995) { out.push(p.clone()); continue; }
    const p1 = new THREE.Vector3().copy(p).addScaledVector(din, -rr);
    const p2 = new THREE.Vector3().copy(p).addScaledVector(dout, rr);
    out.push(p1);
    for (let j = 1; j < k; j++) {
      const t = j / k, mt = 1 - t;
      out.push(new THREE.Vector3(
        mt * mt * p1.x + 2 * mt * t * p.x + t * t * p2.x, 0,
        mt * mt * p1.z + 2 * mt * t * p.z + t * t * p2.z));
    }
    out.push(p2);
  }
  out.push(pts[pts.length - 1].clone());
  return out;
}
const poly = roundCorners(WP.map(([x, z]) => new THREE.Vector3(x, 0, z)));
const cum = [0];
for (let i = 1; i < poly.length; i++) cum.push(cum[i - 1] + poly[i].distanceTo(poly[i - 1]));
const TOTAL = cum[cum.length - 1];
function atS(s) {
  s = Math.max(0, Math.min(s, TOTAL));
  let i = 1;
  while (i < cum.length - 1 && cum[i] < s) i++;
  const t = (s - cum[i - 1]) / Math.max(1e-9, cum[i] - cum[i - 1]);
  return new THREE.Vector3().lerpVectors(poly[i - 1], poly[i], t);
}
// arclength of the closest poly vertex to each stop point
const stops = STOPS.map((st) => {
  let best = 0, bd = 1e9;
  for (let i = 0; i < poly.length; i++) {
    const d = (poly[i].x - st.pt[0]) ** 2 + (poly[i].z - st.pt[1]) ** 2;
    if (d < bd) { bd = d; best = i; }
  }
  return { s: cum[best], dur: st.dur, amp: st.amp };
}).sort((a, b) => a.s - b.s);

// ------------------------------------------------------------- pose bake
// Humanized camera model, parameters fit to a handheld house-tour video
// (camera path recovered with Depth Anything 3; see three/camera_ref.json).
// Measured targets: pitch mean ~-8 deg wandering +-5; roll rms ~1.3 deg;
// yaw-rate rms ~26 deg/s walking and ~10 deg/s standing, energy mostly
// < 0.5 Hz (slow scanning, not jitter); gaze decoupled from velocity
// (looking at features while walking past); speed surging 0.55-1.6x with
// lulls at turns and full stops; lateral sway ~40 mm rms at stride freq;
// vertical bob ~9 mm rms; turns peaking 45-65 deg/s at ~56 deg/s^2.
const dt = 1 / FPS;
const poses = [];
{
  // seeded gaussian + Ornstein-Uhlenbeck banks (stationary rms = sigma)
  let spare = null;
  const gauss = () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    const u1 = Math.max(rng(), 1e-12), u2 = rng();
    const m = Math.sqrt(-2 * Math.log(u1));
    spare = m * Math.sin(2 * Math.PI * u2);
    return m * Math.cos(2 * Math.PI * u2);
  };
  const makeOU = (tau, sigma) => {
    let x = 0;
    return () => {
      x += (-x * dt) / tau + sigma * Math.sqrt((2 * dt) / tau) * gauss();
      return x;
    };
  };
  const ouSpeed = makeOU(2.5, 0.20);    // walking-pace surges
  const ouGaze = makeOU(2.4, 0.16);     // rad: slow scanning wander (never still)
  const ouPitch = makeOU(3.0, 0.06);    // rad, around the downward base
  const ouRoll = makeOU(1.6, 0.017);    // rad
  const ouX = makeOU(0.45, 0.005), ouY = makeOU(0.5, 0.004), ouZ = makeOU(0.45, 0.005);
  const ouH = makeOU(6.0, 0.02);        // slow eye-height drift
  const BASE_PITCH = -0.115;            // ~-6.6 deg: tours look slightly down

  const accel = 0.55;
  // yaw follower fit to measured turns: peak ~60 deg/s, accel ~150 deg/s^2
  const maxRate = 1.05 * dt, maxAcc = 2.6 * dt * dt;

  const headingAt = (ss) => {
    const p = atS(Math.max(0, ss - 0.2)), q = atS(Math.min(ss + 0.45, TOTAL));
    const d = new THREE.Vector3().subVectors(q, p);
    return d.lengthSq() > 1e-8 ? Math.atan2(-d.x, -d.z) : null;
  };
  // suppress look-asides near doorways (don't stare into a jamb mid-transit)
  const doorS = [[4.9, 1.75], [6.7, 1.75], [4.9, 5.3], [6.7, 5.3]].map(([x, z]) => {
    let best = 0, bd = 1e9;
    for (let i = 0; i < poly.length; i++) {
      const d2 = (poly[i].x - x) ** 2 + (poly[i].z - z) ** 2;
      if (d2 < bd) { bd = d2; best = i; }
    }
    return cum[best];
  });
  const nearDoor = (ss) => doorS.some((sd) => Math.abs(ss - sd) < 1.0);

  let s = 0, rate = 0, phase = "walk", phaseT = 0, stopIdx = 0, panBase = 0;
  let yaw = headingAt(0.05) ?? 0;
  let pitch = BASE_PITCH, roll = 0;
  let gait = 0;                          // step phase (1 unit = 1 step)
  let regardOff = 0, regardHold = 0, regardWait = 2.0 + rng() * 2.5;
  let guard = 0;
  while (guard++ < 30000) {
    const prevStopS = stopIdx > 0 ? stops[stopIdx - 1].s : 0;
    const nextStopS = stopIdx < stops.length ? stops[stopIdx].s : TOTAL;
    let targetYaw = yaw, sv = 0;
    if (phase === "walk") {
      // speed: stop-boundary ramps * curvature lull * OU surge * gait pulse
      let v = Math.min(SPEED,
        Math.sqrt(2 * accel * (s - prevStopS + 0.03)),
        Math.sqrt(2 * accel * Math.max(0, nextStopS - s) + 0.03));
      const h1 = headingAt(s + 0.35), h0 = headingAt(s - 0.35);
      if (h1 !== null && h0 !== null) {
        const kappa = Math.abs(angdiff(h1, h0)) / 0.7;
        v *= 1 / (1 + 1.3 * kappa);
      }
      v *= Math.min(1.55, Math.max(0.55, 1 + ouSpeed()));
      v *= 1 + 0.10 * Math.sin(2 * Math.PI * gait);
      s = Math.min(s + v * dt, nextStopS);
      sv = v / SPEED;
      gait += (1.65 + 0.4 * Math.min(sv, 1.2)) * dt;
      // gaze: path direction + slow wander + occasional look-asides
      regardWait -= dt;
      if (regardHold > 0) {
        regardHold -= dt;
        if (regardHold <= 0) regardOff = 0;
      } else if (regardWait <= 0 && !nearDoor(s) && nextStopS - s > 1.4) {
        regardOff = (rng() < 0.5 ? -1 : 1) * (0.5 + rng() * 0.9);
        regardHold = 1.1 + rng() * 1.4;
        regardWait = 2.2 + rng() * 3.2;
      }
      const base = headingAt(s + 1.0) ?? yaw;
      const off = Math.max(-1.5, Math.min(1.5, ouGaze() + regardOff));
      targetYaw = base + off;
      if (s >= nextStopS - 1e-9) {
        regardOff = 0; regardHold = 0;
        if (stopIdx < stops.length) { phase = "stop"; phaseT = 0; panBase = yaw; }
        else break;
      }
    } else if (phase === "stop") {
      phaseT += dt;
      const st = stops[stopIdx];
      const u = Math.min(1, phaseT / st.dur);
      // scripted room pan + the standing wander seen in the reference
      targetYaw = panBase + st.amp * Math.sin(2 * Math.PI * u) * Math.sin(Math.PI * u) + ouGaze();
      if (phaseT >= st.dur) { phase = "turn"; phaseT = 0; }
    } else { // turn in place toward the resumed path
      targetYaw = (headingAt(s + 0.02) ?? yaw) + 0.4 * ouGaze();
      phaseT += dt;
      if (Math.abs(angdiff(targetYaw, yaw)) < 0.14 || phaseT > 4) {
        phase = "walk";
        stopIdx++;
      }
    }
    // yaw: rate+accel-limited pursuit
    const err = angdiff(targetYaw, yaw);
    const tgt = Math.sign(err) * Math.min(maxRate, Math.sqrt(2 * maxAcc * Math.abs(err)), Math.abs(err));
    rate += Math.max(-maxAcc, Math.min(maxAcc, tgt - rate));
    yaw += rate;
    // pitch: slow wander about a downward base + faint step coupling
    const pitchT = BASE_PITCH + ouPitch() + 0.003 * Math.sin(2 * Math.PI * gait + 1.3);
    pitch += (Math.max(-0.44, Math.min(0.05, pitchT)) - pitch) * Math.min(1, dt / 0.45);
    // roll: OU + stride-locked lean
    roll = ouRoll() + 0.004 * Math.sin(Math.PI * gait + 0.7);
    // position: path + stride sway/bob + handheld noise floor
    const p = atS(s);
    const hd = headingAt(s) ?? yaw;
    const sway = 0.042 * (0.35 + 0.65 * sv) * Math.sin(Math.PI * gait);
    const bob = (0.012 * Math.sin(2 * Math.PI * gait)
      + 0.004 * Math.sin(4 * Math.PI * gait + 0.8)) * (0.3 + 0.7 * sv);
    const jf = phase === "walk" ? 1 : 0.55;   // handheld never fully rests
    const px = p.x + Math.cos(hd) * sway + jf * ouX();
    const pz = p.z - Math.sin(hd) * sway + jf * ouZ();
    const py = EYE + ouH() + bob + jf * ouY();
    poses.push([px, py, pz, yaw, pitch, roll]);
  }
  // settle at the end: hold position, keep the micro-noise breathing
  const last = poses[poses.length - 1];
  for (let i = 0; i < 30; i++) {
    poses.push([last[0] + 0.6 * ouX(), last[1] + 0.6 * ouY(), last[2] + 0.6 * ouZ(),
      last[3], last[4], ouRoll()]);
  }
}

// ---------------------------------------------------------------- post/api
const dsize = renderer.getDrawingBufferSize(new THREE.Vector2());
const rt = new THREE.WebGLRenderTarget(dsize.x, dsize.y, { samples: 4, type: THREE.HalfFloatType });
const composer = new EffectComposer(renderer, rt);
composer.addPass(new RenderPass(scene, camera));
composer.addPass(new UnrealBloomPass(dsize.clone(), 0.11, 0.4, 1.1));
composer.addPass(new OutputPass());

window.renderFrame = (i) => {
  const k = Math.min(i, poses.length - 1);
  const [x, y, z, yaw, pitch, roll] = poses[k];
  camera.position.set(x, y, z);
  camera.rotation.set(pitch, yaw, roll);
  composer.render();
  return true;
};

const glCtx = renderer.getContext();
const dbg = glCtx.getExtension("WEBGL_debug_renderer_info");
const glName = dbg ? glCtx.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : "unknown";

window.__scene = scene; // debug hooks
window.__camera = camera;
window.__renderNow = () => composer.render();

THREE.DefaultLoadingManager.onLoad = () => {
  window.renderFrame(0);
  window.__info = { gl: glName, frames: poses.length };
  window.__ready = true;
};
