// Headless Three.js house-tour sample -> out/photoreal/2.mp4
//
// Serves the scene module + node_modules + assets over a local HTTP server, drives a
// headless Chrome (puppeteer) frame by frame, and pipes PNG screenshots into
// ffmpeg. Everything is hardcoded; run with:  node photoreal/render_house2.mjs

import http from "node:http";
import { readFile } from "node:fs/promises";
import { existsSync, mkdirSync } from "node:fs";
import { spawn } from "node:child_process";
import { once } from "node:events";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.join(ROOT, "..", "out", "photoreal", "2.mp4");
const FFMPEG = "/opt/homebrew/bin/ffmpeg";
const W = 960, H = 544, FPS = 30, DSF = 2; // render at 2x, downscale in ffmpeg

const MIME = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
  ".json": "application/json", ".jpg": "image/jpeg", ".png": "image/png",
};

const INDEX = `<!doctype html>
<meta charset="utf-8">
<style>html,body{margin:0;padding:0;overflow:hidden;background:#000}canvas{display:block}</style>
<script type="importmap">
{"imports":{"three":"/node_modules/three/build/three.module.js",
            "three/addons/":"/node_modules/three/examples/jsm/"}}
</script>
<body><script type="module" src="/house2.js"></script></body>`;

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
      res.writeHead(404); res.end("not found"); return;
    }
    res.writeHead(200, { "content-type": MIME[path.extname(file)] ?? "application/octet-stream" });
    res.end(await readFile(file));
  } catch (e) {
    res.writeHead(500); res.end(String(e));
  }
});
await new Promise((ok) => server.listen(0, "127.0.0.1", ok));
const port = server.address().port;

const browser = await puppeteer.launch({
  headless: true,
  args: ["--use-angle=metal", "--enable-gpu", "--hide-scrollbars", "--mute-audio"],
  defaultViewport: { width: W, height: H, deviceScaleFactor: DSF },
});
const page = await browser.newPage();
page.setDefaultTimeout(120000);
page.on("console", (m) => console.log("[page]", m.text()));
page.on("pageerror", (e) => console.error("[pageerror]", e.message));

await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: "domcontentloaded" });
await page.waitForFunction("window.__ready === true", { timeout: 120000 });

const info = await page.evaluate(() => window.__info);
console.log("WebGL:", info.gl);
console.log(`frames: ${info.frames} (${(info.frames / 30).toFixed(1)}s)`);

mkdirSync(path.dirname(OUT), { recursive: true });
const ff = spawn(FFMPEG, [
  "-y", "-loglevel", "error",
  "-f", "image2pipe", "-c:v", "png", "-framerate", String(FPS), "-i", "pipe:0",
  "-vf", `scale=${W}:${H}:flags=lanczos`,
  "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
  OUT,
], { stdio: ["pipe", "inherit", "inherit"] });

const t0 = Date.now();
for (let i = 0; i < info.frames; i++) {
  await page.evaluate((k) => window.renderFrame(k), i);
  const buf = await page.screenshot({ type: "png", optimizeForSpeed: true });
  if (!ff.stdin.write(buf)) await once(ff.stdin, "drain");
  if (i % 90 === 0) {
    const dt = (Date.now() - t0) / 1000;
    console.log(`frame ${i}/${info.frames}  (${(dt / (i + 1) * 1000).toFixed(0)} ms/frame)`);
  }
}
ff.stdin.end();
await once(ff, "close");
console.log(`-> ${OUT}  (${((Date.now() - t0) / 1000).toFixed(0)}s render)`);

await browser.close();
server.close();
