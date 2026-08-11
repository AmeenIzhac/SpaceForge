// Parallel headless renderer for the open-plane scenes.
//
//   node photoreal/render_plane.mjs <scenes.json> <outdir> [workers]
//
// Runs `workers` browser pages at once, each owning whole scenes, and pipes
// JPEG frames straight into its own ffmpeg. JPEG rather than PNG and 640x360
// rather than 960x544: the model is fed 448x256, so anything larger is spent
// on encoding, and the screenshot round-trip is the bottleneck.
//
// PLATFORM SWITCH: this repo runs on a Mac laptop and on a Linux GPU box.
//   FFMPEG     ffmpeg path   (default: homebrew on mac, PATH on linux)
//   CHROME_GL  angle backend (default: metal on mac, swiftshader on linux —
//              software GL, which always works headless; "vulkan" to try GPU)
import http from "node:http";
import { readFile } from "node:fs/promises";
import { existsSync, mkdirSync } from "node:fs";
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const [scenesArg, outArg, workersArg] = process.argv.slice(2);
const SCENES = path.resolve(scenesArg);
const OUTDIR = path.resolve(outArg);
const WORKERS = parseInt(workersArg ?? "6", 10);
const MAC = process.platform === "darwin";
const FFMPEG = process.env.FFMPEG ?? (MAC ? "/opt/homebrew/bin/ffmpeg" : "ffmpeg");
const GL = process.env.CHROME_GL ?? (MAC ? "metal" : "swiftshader");
const W = 640, H = 360, FPS = 30;

const doc = JSON.parse(await readFile(SCENES, "utf8"));
const todo = doc.scenes.filter(
  (s) => !existsSync(path.join(OUTDIR, String(s.id).padStart(4, "0") + ".mp4")));
mkdirSync(OUTDIR, { recursive: true });
console.log(`${todo.length}/${doc.scenes.length} to render on ${WORKERS} workers (GL=${GL})`);

const MIME = { ".html": "text/html", ".js": "text/javascript", ".json": "application/json" };
const INDEX = `<!doctype html><meta charset="utf-8">
<style>html,body{margin:0;overflow:hidden;background:#000}canvas{display:block}</style>
<script type="importmap">{"imports":{"three":"/node_modules/three/build/three.module.js","three/addons/":"/node_modules/three/examples/jsm/"}}</script>
<body><script type="module" src="/plane_scene.js"></script></body>`;

const server = http.createServer(async (req, res) => {
  try {
    const url = decodeURIComponent(req.url.split("?")[0]);
    if (url === "/" || url === "/index.html") {
      res.writeHead(200, { "content-type": "text/html" }); res.end(INDEX); return;
    }
    if (url === "/scenes.json") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(await readFile(SCENES)); return;
    }
    const file = path.normalize(path.join(ROOT, url));
    if (!file.startsWith(ROOT) || !existsSync(file)) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { "content-type": MIME[path.extname(file)] ?? "application/octet-stream" });
    res.end(await readFile(file));
  } catch (e) { res.writeHead(500); res.end(String(e)); }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const port = server.address().port;

// One browser PER worker, not one browser with several pages: concurrent
// pages in a single Chrome share one GPU process, and under software GL they
// starve each other — pages past the first never get a WebGL context and the
// scene never becomes ready. Separate processes each get their own.
const launch = () => puppeteer.launch({
  headless: true,
  protocolTimeout: 600000,
  // --no-sandbox: unprivileged account on the Linux box has no user namespaces
  // --enable-unsafe-swiftshader: Chrome deprecated the silent fallback to
  // software WebGL, so it must be asked for explicitly
  args: [`--use-angle=${GL}`, "--enable-unsafe-swiftshader", "--no-sandbox",
         "--disable-dev-shm-usage", "--hide-scrollbars", "--mute-audio"],
});

let next = 0, done = 0;
const t0 = Date.now();

async function worker(wid) {
  const browser = await launch();
  const page = await browser.newPage();
  await page.setViewport({ width: W, height: H, deviceScaleFactor: 1 });
  page.on("pageerror", (e) => console.error(`[w${wid}] page:`, e.message));
  while (true) {
    const i = next++;
    if (i >= todo.length) break;
    const s = todo[i];
    await page.goto(`http://127.0.0.1:${port}/?scene=${s.id}`, { waitUntil: "load" });
    // "load" fires before the module script has run; wait for the scene
    await page.waitForFunction("window.sceneReady === true", { timeout: 120000 });
    const n = await page.evaluate(() => window.nFrames);
    const out = path.join(OUTDIR, String(s.id).padStart(4, "0") + ".mp4");
    const ff = spawn(FFMPEG, ["-y", "-loglevel", "error", "-f", "image2pipe",
                              "-framerate", String(FPS), "-i", "-", "-c:v", "libx264",
                              "-preset", "veryfast", "-pix_fmt", "yuv420p",
                              "-crf", "23", "-threads", "1", out]);
    ff.stderr.pipe(process.stderr);
    for (let k = 0; k < n; k++) {
      await page.evaluate((j) => window.renderFrame(j), k);
      const buf = await page.screenshot({ type: "jpeg", quality: 82,
                                          optimizeForSpeed: true });
      if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once("drain", r));
    }
    ff.stdin.end();
    await new Promise((r) => ff.on("close", r));
    done++;
    if (done % 20 === 0 || done === todo.length) {
      const el = (Date.now() - t0) / 1000;
      const eta = el * (todo.length - done) / done;
      console.log(`${done}/${todo.length}  ${el.toFixed(0)}s elapsed, eta ${eta.toFixed(0)}s`);
    }
  }
  await page.close();
  await browser.close();
}

await Promise.all(Array.from({ length: WORKERS }, (_, i) => worker(i)));
server.close();
console.log(`done in ${((Date.now() - t0) / 60000).toFixed(1)} min -> ${OUTDIR}`);
