// Render the open-plain Three.js scenes to mp4, headless.
//
//   node photoreal/render_plane3.mjs [firstScene [lastScene]]
//
// PLATFORM SWITCH ------------------------------------------------------------
// This repo runs both on a Mac laptop and on a Linux GPU box; the two need
// different Chrome GL flags and a different ffmpeg. Everything platform-
// specific is decided here, overridable by env:
//   FFMPEG      path to ffmpeg        (default: homebrew on mac, PATH on linux)
//   CHROME_GL   angle backend         (default: metal on mac, swiftshader on
//                                      linux — software GL, always works
//                                      headless; try "vulkan" for GPU)
// ---------------------------------------------------------------------------
import http from "node:http";
import { readFile } from "node:fs/promises";
import { existsSync, mkdirSync } from "node:fs";
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const OUTDIR = path.join(ROOT, "..", "out", "plane3");
const MAC = process.platform === "darwin";
const FFMPEG = process.env.FFMPEG ?? (MAC ? "/opt/homebrew/bin/ffmpeg" : "ffmpeg");
const GL = process.env.CHROME_GL ?? (MAC ? "metal" : "swiftshader");
const W = 960, H = 544, FPS = 30;

const first = parseInt(process.argv[2] ?? "0", 10);
const last = parseInt(process.argv[3] ?? "11", 10);

const MIME = { ".html": "text/html", ".js": "text/javascript",
               ".mjs": "text/javascript", ".json": "application/json" };
const INDEX = `<!doctype html><meta charset="utf-8">
<style>html,body{margin:0;overflow:hidden;background:#000}canvas{display:block}</style>
<script type="importmap">{"imports":{"three":"/node_modules/three/build/three.module.js","three/addons/":"/node_modules/three/examples/jsm/"}}</script>
<body><script type="module" src="/plane3.js"></script></body>`;

const server = http.createServer(async (req, res) => {
  try {
    const url = decodeURIComponent(req.url.split("?")[0]);
    if (url === "/" || url === "/index.html") {
      res.writeHead(200, { "content-type": "text/html" });
      res.end(INDEX);
      return;
    }
    const file = path.normalize(path.join(ROOT, url));
    if (!file.startsWith(ROOT) || !existsSync(file)) {
      res.writeHead(404); res.end(); return;
    }
    res.writeHead(200, { "content-type": MIME[path.extname(file)] ?? "application/octet-stream" });
    res.end(await readFile(file));
  } catch (e) { res.writeHead(500); res.end(String(e)); }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const port = server.address().port;

const browser = await puppeteer.launch({
  headless: true,
  // --no-sandbox: required on the Linux box (no user namespaces for the
  // unprivileged account); harmless on mac.
  args: [`--use-angle=${GL}`, "--enable-gpu", "--no-sandbox",
         "--hide-scrollbars", "--mute-audio",
         `--window-size=${W},${H}`],
});

mkdirSync(OUTDIR, { recursive: true });
for (let sid = first; sid <= last; sid++) {
  const page = await browser.newPage();
  await page.setViewport({ width: W, height: H });
  page.on("pageerror", (e) => console.error(`[scene ${sid}] page:`, e.message));
  await page.goto(`http://127.0.0.1:${port}/?scene=${sid}`, { waitUntil: "load" });
  // "load" fires before the module script runs; wait for the scene itself
  await page.waitForFunction("window.nFrames !== undefined", { timeout: 60000 });
  await page.evaluate(() => window.sceneReady);
  const n = await page.evaluate(() => window.nFrames);

  const out = path.join(OUTDIR, String(sid).padStart(3, "0") + ".mp4");
  const ff = spawn(FFMPEG, [
    "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", String(FPS),
    "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", out]);
  ff.stderr.pipe(process.stderr);

  const t0 = Date.now();
  for (let i = 0; i < n; i++) {
    await page.evaluate((k) => window.renderFrame(k), i);
    const png = await page.screenshot({ type: "png" });
    if (!ff.stdin.write(png)) await new Promise((r) => ff.stdin.once("drain", r));
  }
  ff.stdin.end();
  await new Promise((r) => ff.on("close", r));
  console.log(`scene ${sid}: ${n} frames in ${((Date.now() - t0) / 1000).toFixed(0)}s -> ${out}`);
  await page.close();
}
await browser.close();
server.close();
