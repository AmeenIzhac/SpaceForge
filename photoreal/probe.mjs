// Debug probe: load house.js, optionally boost the sun, screenshot key frames.
import http from "node:http";
import { readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const OUTDIR = "/private/tmp/claude-501/-Users-ameenizhac/72fbbca6-dde6-4222-b3ff-98c213f77806/scratchpad";
const SUN_BOOST = 1; // multiply sun intensity
const FRAMES = [200, 500, 1180];

const MIME = { ".html": "text/html", ".js": "text/javascript", ".jpg": "image/jpeg" };
const INDEX = `<!doctype html><meta charset="utf-8">
<style>html,body{margin:0;overflow:hidden;background:#000}canvas{display:block}</style>
<script type="importmap">{"imports":{"three":"/node_modules/three/build/three.module.js","three/addons/":"/node_modules/three/examples/jsm/"}}</script>
<body><script type="module" src="/house.js"></script></body>`;

const server = http.createServer(async (req, res) => {
  const url = decodeURIComponent(req.url.split("?")[0]);
  if (url === "/") { res.writeHead(200, { "content-type": "text/html" }); res.end(INDEX); return; }
  const file = path.normalize(path.join(ROOT, url));
  if (!file.startsWith(ROOT) || !existsSync(file)) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { "content-type": MIME[path.extname(file)] ?? "application/octet-stream" });
  res.end(await readFile(file));
});
await new Promise((ok) => server.listen(0, "127.0.0.1", ok));

const browser = await puppeteer.launch({
  headless: true,
  args: ["--use-angle=metal", "--enable-gpu", "--hide-scrollbars"],
  defaultViewport: { width: 960, height: 544, deviceScaleFactor: 1 },
});
const page = await browser.newPage();
page.on("pageerror", (e) => console.error("[pageerror]", e.message));
await page.goto(`http://127.0.0.1:${server.address().port}/`, { waitUntil: "domcontentloaded" });
await page.waitForFunction("window.__ready === true", { timeout: 120000 });

const sunInfo = await page.evaluate((boost) => {
  const out = [];
  window.__scene.traverse((o) => {
    if (o.isDirectionalLight) {
      out.push({ intensity: o.intensity, pos: o.position.toArray(), tgt: o.target.position.toArray(),
                 shadow: o.castShadow, mapSize: o.shadow.mapSize.toArray() });
      o.intensity *= boost;
    }
  });
  return out;
}, SUN_BOOST);
console.log("sun:", JSON.stringify(sunInfo));

for (const f of FRAMES) {
  await page.evaluate((k) => window.renderFrame(k), f);
  await writeFile(path.join(OUTDIR, `probe_${f}.png`), await page.screenshot({ type: "png" }));
}
// custom shots: outside looking at the house + at the living-room beam zone
const SHOTS = [
  { name: "outside", pos: [5.8, 1.7, 14.5], rot: [0.02, Math.PI, 0] },
  { name: "livfloor", pos: [2.6, 1.5, 3.9], rot: [-0.5, 0, 0] },
  { name: "kitfloor", pos: [9.0, 1.5, 3.8], rot: [-0.45, 0.3, 0] },
];
for (const s of SHOTS) {
  await page.evaluate((sh) => {
    window.__camera.position.set(...sh.pos);
    window.__camera.rotation.set(...sh.rot);
    window.__renderNow();
  }, s);
  await writeFile(path.join(OUTDIR, `probe_${s.name}.png`), await page.screenshot({ type: "png" }));
}
console.log("done");
await browser.close();
server.close();
