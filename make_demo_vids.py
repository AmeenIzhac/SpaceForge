"""Demo videos: first-person view | top-down map | banner with Q&A.

Left: the video the model saw. Right: an animated top-down map — the full
route in faint grey, the walked part so far in blue, a heading arrow at the
camera, the red X where the walk began. Bottom: the question, the model's
answer, and the truth. The map is drawn from the same poses the video was
rendered from, so what you see is exactly what was measured.

    .venv/bin/python make_demo_vids.py --pick 1t:runs/fb_01 2t:runs/fb_2 ...
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import make_probes as MP

ROOT = Path(__file__).resolve().parent
FPS = 30
PANEL = 544                     # square map panel, same height as the video
BANNER = 130

F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
FB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)

PROMPT = ("Q: a red X marks where the walk began. At the final frame, taking "
          "your facing as north (000°), what bearing is the X?")


def map_transform(poly, marker, size, pad=54):
    xs = list(poly[:, 0]) + [marker[0]]
    ys = list(poly[:, 1]) + [marker[1]]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    span = max(x1 - x0, y1 - y0, 1e-6)
    sc = (size - 2 * pad) / span
    ox = (size - sc * (x1 - x0)) / 2 - sc * x0
    oy = (size - sc * (y1 - y0)) / 2 - sc * y0
    return lambda x, y: (int(round(sc * x + ox)), int(round(sc * y + oy)))


def draw_map(tf, poly, marker, poses, upto, size):
    img = np.full((size, size, 3), 252, np.uint8)
    pts = np.array([tf(x, y) for x, y in poly], np.int32)
    cv2.polylines(img, [pts], False, (215, 214, 209), 3, cv2.LINE_AA)
    k = max(2, int(upto * len(poly) / len(poses)))
    cv2.polylines(img, [pts[:k]], False, (214, 120, 42)[::-1], 3, cv2.LINE_AA)

    mx, my = tf(marker[0], marker[1])
    for a in (45, -45):                                   # the X itself
        d = 11
        dx, dy = int(d * math.cos(math.radians(a))), int(d * math.sin(math.radians(a)))
        cv2.line(img, (mx - dx, my - dy), (mx + dx, my + dy),
                 (36, 40, 214)[::-1], 4, cv2.LINE_AA)

    x, y, yaw = poses[upto][0], poses[upto][1], poses[upto][2]
    cx, cy = tf(x, y)
    hx = int(cx + 22 * math.cos(yaw))
    hy = int(cy + 22 * math.sin(yaw))
    cv2.circle(img, (cx, cy), 7, (11, 11, 11), -1, cv2.LINE_AA)
    cv2.arrowedLine(img, (cx, cy), (hx, hy), (11, 11, 11), 3,
                    cv2.LINE_AA, tipLength=0.45)
    cv2.putText(img, "top-down (X = start)", (14, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (110, 110, 110), 1, cv2.LINE_AA)
    return img


def banner_img(width, rec):
    img = Image.new("RGB", (width, BANNER), "white")
    d = ImageDraw.Draw(img)
    d.text((18, 10), PROMPT, fill=(70, 70, 70), font=F)
    ok = rec["err"] <= 10
    d.text((18, 48),
           f"Model: {rec['answer']:.0f}°   Truth: {rec['gt']:.0f}°   "
           f"(off by {rec['err']:.0f}°)",
           fill=(0, 128, 0) if ok else (188, 0, 0), font=FB)
    lbl = rec["label"]
    d.text((width - 18 - d.textlength(lbl, font=F), 88), lbl,
           fill=(120, 120, 120), font=F)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def build_one(plan, rec, video, out):
    marker, poly, poses = MP.walk_of(plan["turns"], plan["legs"])
    tf = map_transform(poly, marker, PANEL)

    cap = cv2.VideoCapture(str(video))
    W = int(cap.get(3))
    n = int(cap.get(7))
    ban = banner_img(W + PANEL, rec)
    wr = cv2.VideoWriter(str(out) + ".tmp.mp4",
                         cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                         (W + PANEL, PANEL + BANNER))
    i = 0
    while True:
        okf, fr = cap.read()
        if not okf:
            break
        pi = min(int(i * len(poses) / max(n, 1)), len(poses) - 1)
        m = draw_map(tf, poly, marker, poses, pi, PANEL)
        top = np.hstack([fr, m])
        wr.write(np.vstack([top, ban]))
        i += 1
    cap.release()
    wr.release()
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(out) + ".tmp.mp4", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "22", str(out)], check=True)
    Path(str(out) + ".tmp.mp4").unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="+", required=True,
                    help="label:plans.json:videos_dir:results.json:probe_id ...")
    ap.add_argument("--outdir", default="demo_vids")
    args = ap.parse_args()

    for job in args.jobs:
        label, plans_f, vids, results_f, pid = job.split(":")
        pid = int(pid)
        plans = {p["id"]: p for p in
                 json.loads((ROOT / plans_f).read_text())["plans"]}
        recs = {r["id"]: r for r in
                json.loads((ROOT / results_f).read_text())}
        plan, r = plans[pid], recs[pid]
        rec = {"answer": r["answer"], "gt": r["gt"], "err": r["err"],
               "label": f"{label}: {plan['turns'] or 'straight'} "
                        f"({plan['n_turns']} turns)"}
        out = ROOT / args.outdir / f"map_{label}.mp4"
        out.parent.mkdir(exist_ok=True)
        build_one(plan, rec, ROOT / vids / plan.get(
            "video", f"{pid:03d}.mp4"), out)
        print(out.name, f"ans={r['answer']:.0f} gt={r['gt']:.0f} err={r['err']:.0f}")


if __name__ == "__main__":
    main()
