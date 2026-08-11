// Open-plain transfer scene: ground to the horizon, sky, sun, five objects.
// Reads /plane3_spec.json, picks the scene from ?scene=N, and exposes
//   window.sceneReady          -> promise, resolves after first render
//   window.renderFrame(i)      -> draws pose i, resolves when presented
// World frame matches generate.py: x east, y south, yaw clockwise from east;
// mapped to three.js as (X=x, Y=up, Z=y), so bearings transfer unchanged.
import * as THREE from "three";

const params = new URLSearchParams(location.search);
const SCENE_ID = parseInt(params.get("scene") ?? "0", 10);

const spec = await (await fetch("/plane3_spec.json")).json();
const S = spec[SCENE_ID];

const W = 960, H = 544;
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(W, H);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x9db8d8);            // hazy sky
scene.fog = new THREE.Fog(0x9db8d8, 60, 220);

const camera = new THREE.PerspectiveCamera(68, W / H, 0.05, 400);

// --- ground: subtle two-tone checker so motion is visible, grass-ish
const gc = document.createElement("canvas");
gc.width = gc.height = 256;
const g = gc.getContext("2d");
for (let y = 0; y < 8; y++)
  for (let x = 0; x < 8; x++) {
    g.fillStyle = (x + y) % 2 ? "#8fa876" : "#93ad7a";
    g.fillRect(x * 32, y * 32, 32, 32);
  }
g.fillStyle = "rgba(60,70,50,0.25)";
for (let i = 0; i < 300; i++)
  g.fillRect(Math.random() * 256, Math.random() * 256, 2, 2);
const gtex = new THREE.CanvasTexture(gc);
gtex.wrapS = gtex.wrapT = THREE.RepeatWrapping;
gtex.repeat.set(150, 150);
gtex.colorSpace = THREE.SRGBColorSpace;
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(600, 600),
  new THREE.MeshLambertMaterial({ map: gtex }));
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

// --- light: sun + sky dome light
const sun = new THREE.DirectionalLight(0xfff2df, 2.6);
sun.position.set(40, 60, -25);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
const sc = 30;
Object.assign(sun.shadow.camera, { left: -sc, right: sc, top: sc, bottom: -sc });
scene.add(sun);
scene.add(new THREE.HemisphereLight(0xbdd0e8, 0x6a7a55, 0.9));

// --- objects from the spec
const mkGeo = (o) => {
  if (o.shape === "box") return new THREE.BoxGeometry(...o.dims);
  if (o.shape === "sphere") return new THREE.SphereGeometry(o.dims[0], 32, 24);
  if (o.shape === "cylinder")
    return new THREE.CylinderGeometry(o.dims[0], o.dims[1], o.dims[2], 28);
  if (o.shape === "cone") return new THREE.ConeGeometry(o.dims[0], o.dims[1], 28);
  throw new Error(o.shape);
};
const lift = (o) =>
  o.shape === "box" ? o.dims[1] / 2 :
  o.shape === "sphere" ? o.dims[0] :
  o.shape === "cylinder" ? o.dims[2] / 2 : o.dims[1] / 2;
for (const o of S.objects) {
  const m = new THREE.Mesh(
    mkGeo(o),
    new THREE.MeshStandardMaterial({ color: o.color, roughness: 0.55 }));
  m.position.set(o.x, lift(o), o.y);
  m.rotation.y = o.yaw;
  m.castShadow = true;
  scene.add(m);
}

const bob = (i) => 0.015 * Math.sin(i * 0.45);           // faint head-bob
window.renderFrame = (i) => {
  const [x, y, yaw] = S.poses[Math.min(i, S.poses.length - 1)];
  camera.position.set(x, S.eye + bob(i), y);
  camera.lookAt(x + Math.cos(yaw), S.eye + bob(i), y + Math.sin(yaw));
  renderer.render(scene, camera);
  return new Promise(requestAnimationFrame);
};
window.nFrames = S.poses.length;
window.sceneReady = window.renderFrame(0);
