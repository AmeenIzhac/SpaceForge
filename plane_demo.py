"""Demo videos for the open-plane task: view | top-down map | question & verdict.

Left is what the model saw. Right is the truth: the walked path so far in
orange, the rest in grey, every object as a coloured dot with the asked one
ringed, and the camera's position and facing as an arrow. The banner carries
the question type, the model's answer and the ground truth.

    .venv/bin/python plane_demo.py --run runs/pl_tuned --n 6
"""

import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
PANEL, BANNER = 360, 120
F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
FB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)

HEX = {"red": (47, 47, 211), "blue": (208, 95, 31), "green": (70, 158, 46),
       "yellow": (24, 192, 230), "orange": (31, 118, 232),
       "purple": (191, 63, 122), "cyan": (201, 182, 31),
       "pink": (168, 112, 224), "white": (228, 234, 236),
       "brown": (52, 90, 138)}                      # BGR


def wrap(d, text, font, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=font) <= maxw:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + [cur]


def build(scene, rec, plan, video, out):
    poses, objs = scene["poses"], scene["objects"]
    xs = [p[0] for p in poses] + [o["x"] for o in objs]
    ys = [p[1] for p in poses] + [o["y"] for o in objs]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    span = max(x1 - x0, y1 - y0, 1e-6)
    pad = 34
    sc = (PANEL - 2 * pad) / span
    tf = lambda x, y: (int(sc * (x - x0) + (PANEL - sc * (x1 - x0)) / 2),
                       int(sc * (y - y0) + (PANEL - sc * (y1 - y0)) / 2))
    pts = np.array([tf(p[0], p[1]) for p in poses], np.int32)

    cap = cv2.VideoCapture(str(video))
    W = int(cap.get(3))
    n = int(cap.get(7))

    img = Image.new("RGB", (W + PANEL, BANNER), "white")
    d = ImageDraw.Draw(img)
    q = plan["question"].split("\n")[0]
    q = q[q.index(".") + 1:].strip() if "." in q else q
    y = 8
    for ln in wrap(d, q, F, W + PANEL - 36)[:2]:
        d.text((18, y), ln, fill=(70, 70, 70), font=F)
        y += 21
    ok = rec["err"] <= 20
    d.text((18, y + 6),
           f"Model: {rec['answer']:.0f}°   Truth: {rec['gt']:.0f}°   "
           f"(off by {rec['err']:.0f}°)",
           fill=(0, 128, 0) if ok else (188, 0, 0), font=FB)
    tag = f"{plan['task']}  ·  {plan.get('level_path', '')}"
    d.text((W + PANEL - 18 - d.textlength(tag, font=F), y + 10), tag,
           fill=(120, 120, 120), font=F)
    ban = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    wr = cv2.VideoWriter(str(out) + ".tmp.mp4",
                         cv2.VideoWriter_fourcc(*"mp4v"), scene["fps"],
                         (W + PANEL, max(PANEL, 360) + BANNER))
    target = str(plan.get("object", ""))
    i = 0
    while True:
        okf, fr = cap.read()
        if not okf:
            break
        m = np.full((PANEL, PANEL, 3), 252, np.uint8)
        cv2.polylines(m, [pts], False, (215, 214, 209), 2, cv2.LINE_AA)
        k = max(2, int(i * len(poses) / max(n, 1)))
        cv2.polylines(m, [pts[:k]], False, (42, 120, 214), 2, cv2.LINE_AA)
        for o in objs:
            cx, cy = tf(o["x"], o["y"])
            cv2.circle(m, (cx, cy), 7, HEX.get(o["colour_name"], (80, 80, 80)),
                       -1, cv2.LINE_AA)
            if o["name"] in target:
                cv2.circle(m, (cx, cy), 12, (11, 11, 11), 2, cv2.LINE_AA)
        pi = min(int(i * len(poses) / max(n, 1)), len(poses) - 1)
        x, yy, yaw = poses[pi]
        cx, cy = tf(x, yy)
        cv2.circle(m, (cx, cy), 5, (11, 11, 11), -1, cv2.LINE_AA)
        cv2.arrowedLine(m, (cx, cy),
                        (int(cx + 18 * math.cos(yaw)), int(cy + 18 * math.sin(yaw))),
                        (11, 11, 11), 2, cv2.LINE_AA, tipLength=0.45)
        cv2.putText(m, "top-down (ring = asked)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (110, 110, 110), 1, cv2.LINE_AA)
        pad_h = max(0, fr.shape[0] - PANEL)
        mm = np.vstack([m, np.full((pad_h, PANEL, 3), 252, np.uint8)]) if pad_h else m
        wr.write(np.vstack([np.hstack([fr, mm]), ban]))
        i += 1
    cap.release()
    wr.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out) + ".tmp.mp4",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                    str(out)], check=True)
    Path(str(out) + ".tmp.mp4").unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--probes", default="probes/plane_probe_all.json")
    ap.add_argument("--scenes", default="probes/plane_test.json")
    ap.add_argument("--videos", default="out/plane_test")
    ap.add_argument("--outdir", default="demo_vids")
    args = ap.parse_args()

    plans = {p["id"]: p for p in
             json.loads((ROOT / args.probes).read_text())["plans"]}
    scenes = {s["id"]: s for s in
              json.loads((ROOT / args.scenes).read_text())["scenes"]}
    recs = {}
    import glob
    for f in glob.glob(str(ROOT / args.run / "raw_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            recs[r["id"]] = r

    # one per question type: the median-error example, so nothing is cherry-picked
    picks = {}
    for task in ("bearing_final", "bearing_start", "bearing_mid",
                 "bearing_obj", "back_to_start"):
        ids = [i for i, p in plans.items()
               if p["task"] == task and i in recs and recs[i]["err"] is not None]
        if not ids:
            continue
        ids.sort(key=lambda i: recs[i]["err"])
        picks[task] = ids[len(ids) // 2]

    outdir = ROOT / args.outdir
    outdir.mkdir(exist_ok=True)
    for task, pid in picks.items():
        p, r = plans[pid], recs[pid]
        out = outdir / f"plane_{task}.mp4"
        build(scenes[p["scene"]], r, p, ROOT / args.videos / p["video"], out)
        print(f"{out.name}  model={r['answer']:.0f} truth={r['gt']:.0f} "
              f"err={r['err']:.0f} ({p.get('level_path')})")


if __name__ == "__main__":
    main()
