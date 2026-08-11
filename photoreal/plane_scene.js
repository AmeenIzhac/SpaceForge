// Open-plane scene for the generalisation experiment.
//
// Reads /scenes.json (written by plane_gen.py), renders the scene given by
// ?scene=N, and exposes:
//   window.sceneReady        promise, resolves after the first frame
//   window.nFrames           number of poses
//   window.renderFrame(i)    draw pose i
//
// World frame matches generate.py: x east, y south, yaw clockwise from east,
// mapped to three.js as (X=x, Y=up, Z=y). The camera's right-hand direction
// is then (-sin yaw, cos yaw), which is exactly what G.bearing_deg calls 090,
// so bearings computed in Python are the bearings seen on screen.
import * as THREE from "three";

const SID = parseInt(new URLSearchParams(location.search).get("scene") ?? "0", 10);
const all = await (await fetch("/scenes.json")).json();
const S = all.scenes.find((s) => s.id === SID);

const W = 640, H = 360;
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(W, H);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const sky = new THREE.Color().setHSL(S.sky_hue, 0.42, 0.72);
scene.background = sky;
scene.fog = new THREE.Fog(sky, 45, 190);

const camera = new THREE.PerspectiveCamera(S.fov, W / H, 0.05, 400);

// ---- ground: two-tone patch texture, hue and scale randomised per scene
const gc = document.createElement("canvas");
gc.width = gc.height = 256;
const g = gc.getContext("2d");
const base = new THREE.Color().setHSL(S.ground_hue, 0.30, 0.46);
const alt = new THREE.Color().setHSL((S.ground_hue + 0.04) % 1, 0.28, 0.52);
for (let y = 0; y < 8; y++)
  for (let x = 0; x < 8; x++) {
    g.fillStyle = "#" + ((x + y) % 2 ? base : alt).getHexString();
    g.fillRect(x * 32, y * 32, 32, 32);
  }
g.fillStyle = "rgba(0,0,0,0.13)";
for (let i = 0; i < 420; i++)
  g.fillRect(Math.random() * 256, Math.random() * 256, 2, 2);
const gtex = new THREE.CanvasTexture(gc);
gtex.wrapS = gtex.wrapT = THREE.RepeatWrapping;
gtex.repeat.set(S.ground_scale, S.ground_scale);
gtex.colorSpace = THREE.SRGBColorSpace;
const ground = new THREE.Mesh(new THREE.PlaneGeometry(900, 900),
                              new THREE.MeshLambertMaterial({ map: gtex }));
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

// ---- light: sun at a random azimuth/elevation, plus sky fill
const sun = new THREE.DirectionalLight(0xfff4e2, 2.5);
const R = 70;
sun.position.set(Math.cos(S.sun_az) * R * Math.cos(S.sun_el),
                 R * Math.sin(S.sun_el),
                 Math.sin(S.sun_az) * R * Math.cos(S.sun_el));
sun.castShadow = true;
sun.shadow.mapSize.set(1024, 1024);
Object.assign(sun.shadow.camera, { left: -35, right: 35, top: 35, bottom: -35 });
scene.add(sun);
scene.add(new THREE.HemisphereLight(sky.getHex(), 0x555544, 0.85));

// ---- objects
const geo = (o) => {
  const s = o.scale;
  switch (o.shape) {
    case "cube":       return new THREE.BoxGeometry(0.95 * s, 0.95 * s, 0.95 * s);
    case "sphere":     return new THREE.SphereGeometry(0.55 * s, 32, 22);
    case "cylinder":   return new THREE.CylinderGeometry(0.38 * s, 0.38 * s, 1.9 * s, 26);
    case "cone":       return new THREE.ConeGeometry(0.6 * s, 1.5 * s, 26);
    case "torus":      return new THREE.TorusGeometry(0.6 * s, 0.21 * s, 16, 34);
    case "capsule":    return new THREE.CapsuleGeometry(0.35 * s, 1.0 * s, 8, 20);
    case "octahedron": return new THREE.OctahedronGeometry(0.75 * s);
    case "pyramid":    return new THREE.ConeGeometry(0.8 * s, 1.4 * s, 4);
    default: throw new Error(o.shape);
  }
};
const lift = (o) => {
  const s = o.scale;
  switch (o.shape) {
    case "cube":       return 0.475 * s;
    case "sphere":     return 0.55 * s;
    case "cylinder":   return 0.95 * s;
    case "cone":       return 0.75 * s;
    case "torus":      return 0.81 * s;
    case "capsule":    return 0.85 * s;
    case "octahedron": return 0.75 * s;
    case "pyramid":    return 0.70 * s;
    default: return 0.5 * s;
  }
};
for (const o of S.objects) {
  const m = new THREE.Mesh(geo(o), new THREE.MeshStandardMaterial(
    { color: o.colour, roughness: 0.5, metalness: 0.04 }));
  m.position.set(o.x, lift(o), o.y);
  m.rotation.y = o.yaw;
  if (o.shape === "torus") m.rotation.x = Math.PI / 2;
  m.castShadow = true;
  scene.add(m);
}

const bob = (i) => 0.018 * Math.sin(i * 0.42);
window.nFrames = S.poses.length;
window.renderFrame = (i) => {
  const [x, y, yaw] = S.poses[Math.min(i, S.poses.length - 1)];
  const h = S.eye + bob(i);
  camera.position.set(x, h, y);
  camera.lookAt(x + Math.cos(yaw), h, y + Math.sin(yaw));
  renderer.render(scene, camera);
};
window.renderFrame(0);
window.sceneReady = true;
