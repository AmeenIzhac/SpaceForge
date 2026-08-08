// Realistic corridor scene + deterministic first-person walk.
// Runs inside headless Chrome; render.mjs calls window.renderFrame(i).
//
// Hotel-ish corridor, left-right-left turns, no rooms. PBR materials from
// ambientCG (plaster walls, carpet floor, wood doors), recessed ceiling
// panels with RectAreaLights, a few shadow-casting points, ACES + bloom.
// Motion is the cleaned-up model from world.py: constant speed with ramps,
// look-ahead heading, acceleration-limited yaw (no glances, no stops).

import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RectAreaLightUniformsLib } from "three/addons/lights/RectAreaLightUniformsLib.js";

const SEED = 11;
const CS = 0.85;              // grid cell (m); corridor is 2 cells wide
const WALL_H = 2.6;
const EYE = 1.62;
const SPEED = 1.2;            // m/s
const FPS = 30;
const FOV = 55;               // vertical

// waypoints in cell units (x, z); legs 8 cells = 6.8 m. Turns: L, R, L.
const WPTS = [[0, 0], [8, 0], [8, -8], [16, -8], [16, -16]];

// ---------------------------------------------------------------- utilities
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

// ------------------------------------------------------------ occupancy grid
// carve 2-cell-wide corridors along each leg (1 cell each side of the
// centerline, which runs on cell boundaries), extended 1 cell past the ends
const open = new Set();
const key = (ix, iz) => `${ix},${iz}`;
for (let i = 0; i < WPTS.length - 1; i++) {
  const [ax, az] = WPTS[i], [bx, bz] = WPTS[i + 1];
  if (az === bz) {
    const lo = Math.min(ax, bx) - 1, hi = Math.max(ax, bx);
    for (let ix = lo; ix <= hi; ix++) { open.add(key(ix, az - 1)); open.add(key(ix, az)); }
  } else {
    const lo = Math.min(az, bz) - 1, hi = Math.max(az, bz);
    for (let iz = lo; iz <= hi; iz++) { open.add(key(ax - 1, iz)); open.add(key(ax, iz)); }
  }
}

// wall runs: merge open/closed boundary faces into long quads.
// each run: {axis: 'x'|'z' (plane normal axis), at, lo, hi, n (+1/-1 into corridor)}
function wallRuns() {
  const faces = new Map(); // groupKey -> sorted list of span starts
  for (const k of open) {
    const [ix, iz] = k.split(",").map(Number);
    const nb = [
      [ix - 1, iz, "x", ix, +1, iz], [ix + 1, iz, "x", ix + 1, -1, iz],
      [ix, iz - 1, "z", iz, +1, ix], [ix, iz + 1, "z", iz + 1, -1, ix],
    ];
    for (const [nx, nz, axis, at, n, span] of nb) {
      if (!open.has(key(nx, nz))) {
        const g = `${axis}|${at}|${n}`;
        if (!faces.has(g)) faces.set(g, []);
        faces.get(g).push(span);
      }
    }
  }
  const runs = [];
  for (const [g, spans] of faces) {
    const [axis, at, n] = g.split("|");
    spans.sort((a, b) => a - b);
    let lo = spans[0], prev = spans[0];
    for (let i = 1; i <= spans.length; i++) {
      if (i === spans.length || spans[i] !== prev + 1) {
        runs.push({ axis, at: +at * CS, lo: lo * CS, hi: (prev + 1) * CS, n: +n });
        if (i < spans.length) { lo = spans[i]; prev = spans[i]; }
      } else prev = spans[i];
    }
  }
  return runs;
}

// floor strips: merge consecutive open cells per row
function floorStrips() {
  const rows = new Map();
  for (const k of open) {
    const [ix, iz] = k.split(",").map(Number);
    if (!rows.has(iz)) rows.set(iz, []);
    rows.get(iz).push(ix);
  }
  const strips = [];
  for (const [iz, xs] of rows) {
    xs.sort((a, b) => a - b);
    let lo = xs[0], prev = xs[0];
    for (let i = 1; i <= xs.length; i++) {
      if (i === xs.length || xs[i] !== prev + 1) {
        strips.push({ x0: lo * CS, x1: (prev + 1) * CS, z0: iz * CS, z1: (iz + 1) * CS });
        if (i < xs.length) { lo = xs[i]; prev = xs[i]; }
      } else prev = xs[i];
    }
  }
  return strips;
}

// --------------------------------------------------------------- geometry IO
// quad builder with world-space UVs: origin p0, edge u (horizontal), edge v.
// winding p0, p0+u, p0+u+v, p0+v is CCW w.r.t. cross(u, v).
function makeQuadSink() {
  const pos = [], nrm = [], uv = [];
  return {
    add(p0, u, v, su, sv) {
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
    build(material) {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
      g.setAttribute("normal", new THREE.Float32BufferAttribute(nrm, 3));
      g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
      return new THREE.Mesh(g, material);
    },
  };
}

// ------------------------------------------------------------------ textures
const texLoader = new THREE.TextureLoader();
function tex(file, { srgb = false, rx = 1, ry = 1 } = {}) {
  const t = texLoader.load(`/assets/${file}`);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(rx, ry);
  t.anisotropy = 8;
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  t.channel = 0;
  return t;
}

const wallMat = new THREE.MeshStandardMaterial({
  color: 0xded5c8,
  map: tex("Plaster001_Color.jpg", { srgb: true }),
  normalMap: tex("Plaster001_NormalGL.jpg"),
  roughnessMap: tex("Plaster001_Roughness.jpg"),
  normalScale: new THREE.Vector2(0.7, 0.7),
});
const ceilMat = new THREE.MeshStandardMaterial({
  color: 0xf1efe9,
  map: tex("Plaster001_Color.jpg", { srgb: true }),
  normalMap: tex("Plaster001_NormalGL.jpg"),
  roughnessMap: tex("Plaster001_Roughness.jpg"),
  normalScale: new THREE.Vector2(0.4, 0.4),
});
const floorMat = new THREE.MeshStandardMaterial({
  color: 0xcabca6,
  map: tex("Carpet008_Color.jpg", { srgb: true }),
  normalMap: tex("Carpet008_NormalGL.jpg"),
  roughnessMap: tex("Carpet008_Roughness.jpg"),
  aoMap: tex("Carpet008_AmbientOcclusion.jpg"),
  aoMapIntensity: 0.7,
});
const woodMat = new THREE.MeshStandardMaterial({
  color: 0xe2c298,
  map: tex("Wood051_Color.jpg", { srgb: true, rx: 0.9, ry: 0.9 }),
  normalMap: tex("Wood051_NormalGL.jpg", { rx: 0.9, ry: 0.9 }),
  roughnessMap: tex("Wood051_Roughness.jpg", { rx: 0.9, ry: 0.9 }),
  roughness: 0.85,
});
const trimMat = new THREE.MeshStandardMaterial({ color: 0xf4f1ea, roughness: 0.5 });
const brassMat = new THREE.MeshStandardMaterial({ color: 0xb08d57, metalness: 0.9, roughness: 0.35 });
const aluMat = new THREE.MeshStandardMaterial({ color: 0xcfcfcf, metalness: 0.7, roughness: 0.4 });
const darkWoodMat = new THREE.MeshStandardMaterial({ color: 0x4a382a, roughness: 0.6 });

// -------------------------------------------------------------------- scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x060504);
scene.fog = new THREE.FogExp2(0x0b0908, 0.012);

const runs = wallRuns();
const walls = makeQuadSink(), floors = makeQuadSink(), ceils = makeQuadSink();
for (const r of runs) {
  const len = r.hi - r.lo;
  // winding: front face is the cross(u, v) side, and the sign of
  // cross(u, (0,H,0)) differs between x- and z-normal walls
  const p0 = r.axis === "x"
    ? new THREE.Vector3(r.at, 0, r.n > 0 ? r.hi : r.lo)
    : new THREE.Vector3(r.n > 0 ? r.lo : r.hi, 0, r.at);
  const u = r.axis === "x"
    ? new THREE.Vector3(0, 0, r.n > 0 ? -len : len)
    : new THREE.Vector3(r.n > 0 ? len : -len, 0, 0);
  walls.add(p0, u, new THREE.Vector3(0, WALL_H, 0), 2.6, 2.6);
}
for (const s of floorStrips()) {
  const dx = s.x1 - s.x0, dz = s.z1 - s.z0;
  floors.add(new THREE.Vector3(s.x0, 0, s.z0),
    new THREE.Vector3(0, 0, dz), new THREE.Vector3(dx, 0, 0), 2.6, 2.6);
  ceils.add(new THREE.Vector3(s.x0, WALL_H, s.z0),
    new THREE.Vector3(dx, 0, 0), new THREE.Vector3(0, 0, dz), 2.6, 2.6);
}
const wallMesh = walls.build(wallMat);
wallMesh.receiveShadow = true;
scene.add(wallMesh);
const floorMesh = floors.build(floorMat);
floorMesh.receiveShadow = true;
scene.add(floorMesh);
scene.add(ceils.build(ceilMat));

// --------------------------------------------------- trim, doors, wall art
function alongDir(run) { // unit vector along the wall span
  return run.axis === "x" ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(1, 0, 0);
}
function normalDir(run) {
  return run.axis === "x" ? new THREE.Vector3(run.n, 0, 0) : new THREE.Vector3(0, 0, run.n);
}
function wallPoint(run, t, y, proud) { // t = distance from run.lo along span
  const p = new THREE.Vector3()
    .addScaledVector(alongDir(run), run.lo + t)
    .addScaledVector(normalDir(run), proud);
  if (run.axis === "x") p.x += run.at; else p.z += run.at;
  p.y = y;
  return p;
}
function yawOf(run) { // rotation so a box's local x lies along the run
  return run.axis === "x" ? Math.PI / 2 : 0;
}

// baseboards
for (const r of runs) {
  const len = r.hi - r.lo;
  const bb = new THREE.Mesh(new THREE.BoxGeometry(len, 0.095, 0.016), trimMat);
  bb.position.copy(wallPoint(r, len / 2, 0.0475, 0.008));
  bb.rotation.y = yawOf(r);
  scene.add(bb);
}

// doors + occasional framed prints on long runs
const artCanvasTex = (() => {
  const variants = [];
  for (let v = 0; v < 3; v++) {
    const cv = document.createElement("canvas");
    cv.width = cv.height = 256;
    const g = cv.getContext("2d");
    const hue = 20 + rng() * 200;
    const grad = g.createLinearGradient(0, 0, 256, 256);
    grad.addColorStop(0, `hsl(${hue}, 25%, 72%)`);
    grad.addColorStop(1, `hsl(${hue + 40}, 20%, 45%)`);
    g.fillStyle = grad;
    g.fillRect(0, 0, 256, 256);
    for (let i = 0; i < 5; i++) {
      g.fillStyle = `hsla(${hue + rng() * 80 - 40}, 30%, ${30 + rng() * 50}%, 0.35)`;
      g.beginPath();
      g.ellipse(rng() * 256, rng() * 256, 20 + rng() * 60, 15 + rng() * 45, rng() * 3, 0, 7);
      g.fill();
    }
    const t = new THREE.CanvasTexture(cv);
    t.colorSpace = THREE.SRGBColorSpace;
    variants.push(t);
  }
  return variants;
})();

function addDoor(run, t) {
  const yaw = yawOf(run);
  const g = new THREE.Group();
  // casing: two legs + head around a 0.98 x 2.08 opening
  const legG = new THREE.BoxGeometry(0.075, 2.08, 0.028);
  for (const s of [-1, 1]) {
    const leg = new THREE.Mesh(legG, trimMat);
    leg.position.set(s * (0.49 + 0.0375), 1.04, 0.014);
    g.add(leg);
  }
  const head = new THREE.Mesh(new THREE.BoxGeometry(1.13, 0.075, 0.028), trimMat);
  head.position.set(0, 2.08 + 0.0375, 0.014);
  g.add(head);
  // slab, slightly recessed behind the casing
  const slab = new THREE.Mesh(new THREE.BoxGeometry(0.94, 2.05, 0.042), woodMat);
  slab.position.set(0, 1.025, 0.004);
  slab.castShadow = true;
  g.add(slab);
  // raised panels, 2 x 3
  const panelG = new THREE.BoxGeometry(0.34, 0.52, 0.014);
  for (const px of [-0.21, 0.21]) {
    for (const py of [0.45, 1.06, 1.67]) {
      const p = new THREE.Mesh(panelG, woodMat);
      p.position.set(px, py, 0.028);
      g.add(p);
    }
  }
  // brass lever handle
  const rose = new THREE.Mesh(new THREE.CylinderGeometry(0.024, 0.024, 0.014, 20), brassMat);
  rose.rotation.x = Math.PI / 2;
  rose.position.set(0.36, 1.02, 0.032);
  g.add(rose);
  const lever = new THREE.Mesh(new THREE.CylinderGeometry(0.009, 0.009, 0.12, 12), brassMat);
  lever.rotation.z = Math.PI / 2;
  lever.position.set(0.3, 1.02, 0.045);
  g.add(lever);

  const p = wallPoint(run, t, 0, 0.02);
  g.position.copy(p);
  g.rotation.y = yaw + (run.axis === "x" && run.n < 0 ? Math.PI : 0) + (run.axis === "z" && run.n < 0 ? Math.PI : 0);
  scene.add(g);
}

function addArt(run, t) {
  const g = new THREE.Group();
  const fw = 0.56, fh = 0.72, bd = 0.035;
  const frame = [
    [0, fh / 2 - bd / 2, fw, bd], [0, -fh / 2 + bd / 2, fw, bd],
    [-fw / 2 + bd / 2, 0, bd, fh - 2 * bd], [fw / 2 - bd / 2, 0, bd, fh - 2 * bd],
  ];
  for (const [x, y, w, h] of frame) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, 0.03), darkWoodMat);
    m.position.set(x, y, 0.015);
    g.add(m);
  }
  const canvas = new THREE.Mesh(
    new THREE.PlaneGeometry(fw - 2 * bd, fh - 2 * bd),
    new THREE.MeshStandardMaterial({ map: artCanvasTex[Math.floor(rng() * 3)], roughness: 0.85 }),
  );
  canvas.position.z = 0.012;
  g.add(canvas);
  const p = wallPoint(run, t, 1.55, 0.0);
  g.position.copy(p);
  g.rotation.y = yawOf(run) + ((run.axis === "x" && run.n < 0) || (run.axis === "z" && run.n < 0) ? Math.PI : 0);
  scene.add(g);
}

for (const r of runs) {
  const len = r.hi - r.lo;
  if (len < 3.2) continue;
  const margin = 1.15;
  const nDoors = Math.max(1, Math.floor((len - 2 * margin) / 3.3));
  const doorTs = [];
  for (let i = 0; i < nDoors; i++) {
    const t = margin + (i + 0.5) * ((len - 2 * margin) / nDoors) + (rng() - 0.5) * 0.4;
    doorTs.push(t);
    addDoor(r, t);
  }
  for (const dt of doorTs) {
    const t = dt + 1.75;
    if (t < len - margin && doorTs.every((o) => Math.abs(o - t) > 1.2) && rng() < 0.65) addArt(r, t);
  }
}

// --------------------------------------------------------------- lighting
RectAreaLightUniformsLib.init();
scene.add(new THREE.HemisphereLight(0xfff4e4, 0x6a6156, 0.4));

let panelIdx = 0;
for (let i = 0; i < WPTS.length - 1; i++) {
  const a = new THREE.Vector3(WPTS[i][0] * CS, 0, WPTS[i][1] * CS);
  const b = new THREE.Vector3(WPTS[i + 1][0] * CS, 0, WPTS[i + 1][1] * CS);
  const dir = new THREE.Vector3().subVectors(b, a).normalize();
  const len = a.distanceTo(b);
  // object yaw that maps a box's local +x onto the leg direction
  const ringYaw = Math.atan2(-dir.z, dir.x);
  const panelS = [];
  for (let s = 1.7; s < len - 0.8; s += 3.4) panelS.push(s);
  if (i === WPTS.length - 2) panelS.push(len + 0.2); // light the dead end
  for (const s of panelS) {
    const c = new THREE.Vector3().copy(a).addScaledVector(dir, s);
    // trim ring + warm emissive panel, flush with the ceiling
    const ring = new THREE.Mesh(new THREE.BoxGeometry(1.18, 0.02, 0.62), aluMat);
    ring.position.set(c.x, WALL_H - 0.012, c.z);
    ring.rotation.y = ringYaw;
    scene.add(ring);
    const panel = new THREE.Mesh(
      new THREE.PlaneGeometry(1.08, 0.52),
      new THREE.MeshStandardMaterial({
        color: 0x222222, emissive: 0xfff1dd, emissiveIntensity: 2.0,
      }),
    );
    // Rz spins the plane's long axis onto the leg, then Rx tips it face-down
    panel.rotation.set(Math.PI / 2, 0, Math.atan2(dir.z, dir.x));
    panel.position.set(c.x, WALL_H - 0.024, c.z);
    scene.add(panel);

    const ral = new THREE.RectAreaLight(0xfff1dd, 4.5, 1.08, 0.52);
    ral.position.set(c.x, WALL_H - 0.03, c.z);
    ral.lookAt(c.x, 0, c.z);
    scene.add(ral);

    if (panelIdx % 2 === 0) {
      const pl = new THREE.PointLight(0xffe9d0, 4.5, 8, 2);
      pl.position.set(c.x, WALL_H - 0.5, c.z);
      pl.castShadow = true;
      pl.shadow.mapSize.set(1024, 1024);
      pl.shadow.bias = -0.004;
      scene.add(pl);
    }
    panelIdx++;
  }
}

// ------------------------------------------------------------------- motion
function roundCorners(pts, r = 0.62, k = 9) {
  const out = [pts[0].clone()];
  for (let i = 1; i < pts.length - 1; i++) {
    const p = pts[i];
    const din = new THREE.Vector3().subVectors(p, pts[i - 1]);
    const dout = new THREE.Vector3().subVectors(pts[i + 1], p);
    const li = din.length(), lo = dout.length();
    din.normalize(); dout.normalize();
    const rr = Math.min(r, 0.42 * Math.min(li, lo));
    const p1 = new THREE.Vector3().copy(p).addScaledVector(din, -rr);
    const p2 = new THREE.Vector3().copy(p).addScaledVector(dout, rr);
    out.push(p1);
    for (let j = 1; j < k; j++) { // quadratic bezier p1 -> p -> p2
      const t = j / k, mt = 1 - t;
      out.push(new THREE.Vector3(
        mt * mt * p1.x + 2 * mt * t * p.x + t * t * p2.x, 0,
        mt * mt * p1.z + 2 * mt * t * p.z + t * t * p2.z,
      ));
    }
    out.push(p2);
  }
  out.push(pts[pts.length - 1].clone());
  return out;
}

const poly = roundCorners(WPTS.map(([x, z]) => new THREE.Vector3(x * CS, 0, z * CS)));
const cum = [0];
for (let i = 1; i < poly.length; i++) cum.push(cum[i - 1] + poly[i].distanceTo(poly[i - 1]));
const TOTAL = cum[cum.length - 1] - 2.4; // stop with the end wall still ~3 m off
function atS(s) {
  s = Math.max(0, Math.min(s, cum[cum.length - 1]));
  let i = 1;
  while (i < cum.length - 1 && cum[i] < s) i++;
  const t = (s - cum[i - 1]) / Math.max(1e-9, cum[i] - cum[i - 1]);
  return new THREE.Vector3().lerpVectors(poly[i - 1], poly[i], t);
}

const dt = 1 / FPS;
const poses = [];
{
  let s = 0;
  const accel = 0.65;
  const h0 = new THREE.Vector3().subVectors(poly[1], poly[0]).normalize();
  let yaw = Math.atan2(-h0.x, -h0.z);
  let rate = 0;
  const maxRate = 1.4 * dt, maxAcc = 4.5 * dt * dt;
  while (s < TOTAL) {
    const v = Math.min(SPEED, Math.sqrt(2 * accel * (s + 0.02)), Math.sqrt(2 * accel * (TOTAL - s + 0.02)));
    s = Math.min(s + v * dt, TOTAL);
    const p = atS(s);
    const look = atS(Math.min(s + 1.25, TOTAL + 0.35));
    const d = new THREE.Vector3().subVectors(look, p);
    const targetYaw = d.lengthSq() > 1e-6 ? Math.atan2(-d.x, -d.z) : yaw;
    const err = angdiff(targetYaw, yaw);
    const tgt = Math.sign(err) * Math.min(maxRate, Math.sqrt(2 * maxAcc * Math.abs(err)), Math.abs(err));
    rate += Math.max(-maxAcc, Math.min(maxAcc, tgt - rate));
    yaw += rate;
    poses.push([p.x, p.z, yaw, s]);
    if (poses.length > 20000) break;
  }
  for (let i = 0; i < 14; i++) poses.push(poses[poses.length - 1]); // hold at end
}

// ------------------------------------------------------------ renderer/post
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.25;
document.body.appendChild(renderer.domElement);

const camera = new THREE.PerspectiveCamera(FOV, window.innerWidth / window.innerHeight, 0.05, 60);
camera.rotation.order = "YXZ";

const dsize = renderer.getDrawingBufferSize(new THREE.Vector2());
const rt = new THREE.WebGLRenderTarget(dsize.x, dsize.y, { samples: 4, type: THREE.HalfFloatType });
const composer = new EffectComposer(renderer, rt);
composer.addPass(new RenderPass(scene, camera));
composer.addPass(new UnrealBloomPass(dsize.clone(), 0.13, 0.45, 1.05));
composer.addPass(new OutputPass());

window.renderFrame = (i) => {
  const k = Math.min(i, poses.length - 1);
  const [x, z, yaw, s] = poses[k];
  const bobY = 0.011 * Math.sin(2 * Math.PI * s / 0.75);
  const sway = 0.006 * Math.sin(2 * Math.PI * s / 1.5);
  camera.position.set(
    x + Math.cos(yaw) * sway,
    EYE + bobY,
    z - Math.sin(yaw) * sway,
  );
  camera.rotation.set(-0.02, yaw, 0);
  composer.render();
  return true;
};

// ------------------------------------------------------------------- ready
const glCtx = renderer.getContext();
const dbg = glCtx.getExtension("WEBGL_debug_renderer_info");
const glName = dbg ? glCtx.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : "unknown";

THREE.DefaultLoadingManager.onLoad = () => {
  window.renderFrame(0); // warm up pipelines/shadow maps
  window.__info = { gl: glName, frames: poses.length };
  window.__ready = true;
};
