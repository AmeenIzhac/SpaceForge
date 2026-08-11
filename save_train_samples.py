"""Save inspectable copies of actual training examples.

For each sample: the video the model was fed, beside a top-down map of the
same scene, with a banner carrying the exact question text and the exact
target string the model was trained to emit. Nothing here is a model
prediction — it is the supervision itself, so the labels can be checked by
eye against the geometry.

    .venv/bin/python save_train_samples.py --n-per-type 2
"""

import argparse
import json
import math
import random
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
PANEL, BANNER = 360, 250      # tall enough that the last line
                              # clears a player progress bar
RULE = (214, 212, 206)          # subtle separator, BGR
F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
FM = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 20)
FB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)

HEX = {"red": (47, 47, 211), "blue": (208, 95, 31), "green": (70, 158, 46),
       "yellow": (24, 192, 230), "orange": (31, 118, 232),
       "purple": (191, 63, 122), "cyan": (201, 182, 31),
       "pink": (168, 112, 224), "white": (228, 234, 236),
       "brown": (52, 90, 138)}


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


def make(row, scene, out):
    poses, objs = scene["poses"], scene["objects"]
    xs = [p[0] for p in poses] + [o["x"] for o in objs]
    ys = [p[1] for p in poses] + [o["y"] for o in objs]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    span = max(x1 - x0, y1 - y0, 1e-6)
    sc = (PANEL - 68) / span
    tf = lambda x, y: (int(sc * (x - x0) + (PANEL - sc * (x1 - x0)) / 2),
                       int(sc * (y - y0) + (PANEL - sc * (y1 - y0)) / 2))
    pts = np.array([tf(p[0], p[1]) for p in poses], np.int32)

    cap = cv2.VideoCapture(row["video"])
    W = int(cap.get(3))
    n = int(cap.get(7))

    img = Image.new("RGB", (W + PANEL, BANNER), "white")
    d = ImageDraw.Draw(img)
    d.text((16, 8), f"TRAINING EXAMPLE  ·  task: {row['task']}  ·  path: "
                    f"{row['level']}  ·  wording: {row['paraphrase']}",
           fill=(120, 120, 120), font=F)
    y = 32
    for ln in wrap(d, row["question"].split("\n")[0], F, W + PANEL - 32)[:5]:
        d.text((16, y), ln, fill=(60, 60, 60), font=F)
        y += 20
    # everything on ONE line, nothing below it: the player's progress bar
    # sits over the bottom of the frame
    pred = row.get("prediction")
    d.text((16, y + 6), "target", fill=(120, 120, 120), font=F)
    d.text((16, y + 26), row["answer"], fill=(0, 110, 0), font=FM)
    x = 16 + max(d.textlength(row["answer"], font=FM), 90) + 46
    if pred:
        d.text((x, y + 6), "model", fill=(120, 120, 120), font=F)
        d.text((x, y + 26), pred["text"],
               fill=(0, 110, 0) if pred["ok"] else (180, 0, 0), font=FM)
        if pred.get("note"):
            d.text((x + d.textlength(pred["text"], font=FM) + 10, y + 28),
                   pred["note"], fill=(120, 120, 120), font=F)
    if row.get("object"):
        d.text((W + PANEL - 16 - d.textlength(f"asked about: {row['object']}",
                                              font=F), y + 28),
               f"asked about: {row['object']}", fill=(120, 120, 120), font=F)
    d.line([(0, 0), (W + PANEL, 0)], fill=(206, 204, 198), width=2)
    ban = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    wr = cv2.VideoWriter(str(out) + ".tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                         scene["fps"], (W + PANEL, max(PANEL, 360) + BANNER))
    target = str(row.get("object", ""))
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
            if target and o["name"] in target:
                cv2.circle(m, (cx, cy), 12, (11, 11, 11), 2, cv2.LINE_AA)
        pi = min(int(i * len(poses) / max(n, 1)), len(poses) - 1)
        x, yy, yaw = poses[pi]
        cx, cy = tf(x, yy)
        cv2.circle(m, (cx, cy), 5, (11, 11, 11), -1, cv2.LINE_AA)
        cv2.arrowedLine(m, (cx, cy), (int(cx + 18 * math.cos(yaw)),
                                      int(cy + 18 * math.sin(yaw))),
                        (11, 11, 11), 2, cv2.LINE_AA, tipLength=0.45)
        cv2.putText(m, "top-down: path so far (blue), objects, ring = asked",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (110, 110, 110), 1,
                    cv2.LINE_AA)
        pad_h = max(0, fr.shape[0] - PANEL)
        mm = np.vstack([m, np.full((pad_h, PANEL, 3), 252, np.uint8)]) if pad_h else m
        mm[:, :2] = RULE                      # divider between view and map
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
    ap.add_argument("--rows", default="data/plane_train.jsonl")
    ap.add_argument("--scenes", default="probes/plane_train.json")
    ap.add_argument("--outdir", default="train_samples")
    ap.add_argument("--n-per-type", type=int, default=2)
    ap.add_argument("--preds", default=None,
                    help="results.json from running the model on these very "
                         "examples (they are training examples, so this shows "
                         "fit, not generalisation)")
    ap.add_argument("--manifest-order", action="store_true",
                    help="reuse the existing manifest's example list")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(ROOT / args.rows)]
    scenes = {s["id"]: s for s in
              json.loads((ROOT / args.scenes).read_text())["scenes"]}
    out = ROOT / args.outdir
    out.mkdir(exist_ok=True)

    preds = {}
    if args.preds:
        man = json.loads((out / "manifest.json").read_text())
        res = {r["id"]: r for r in json.loads((ROOT / args.preds).read_text())}
        for i, m in enumerate(man):
            reply = res[i]["reply"].strip().splitlines()[-1].strip()
            got = reply.split(":")[-1].strip()
            want = m["target_answer"].split(":")[-1].strip()
            note = ""
            if m["task"].startswith("bearing"):
                d = abs(int(got) - int(want)) % 360
                d = min(d, 360 - d)
                ok, note = d <= 20, f"(off by {d}\u00b0)"
            else:
                ok = got.upper() == want.upper()
            preds[m["file"]] = {"text": reply, "ok": ok, "note": note}

    rng = random.Random(3)
    picked, manifest = [], []
    for task in ("bearing_final", "bearing_start", "quadrant", "count", "turns"):
        cand = [r for r in rows if r["task"] == task]
        rng.shuffle(cand)
        # spread over path difficulty so the samples are not all trivial
        seen_lv = set()
        for r in cand:
            if len([p for p in picked if p["task"] == task]) >= args.n_per_type:
                break
            if r["level"] in seen_lv:
                continue
            seen_lv.add(r["level"])
            picked.append(r)

    for i, r in enumerate(picked):
        name = f"{r['task']}_{r['level']}.mp4"
        r["prediction"] = preds.get(name)
        make(r, scenes[r["scene"]], out / name)
        manifest.append({"file": name, "task": r["task"], "path_level": r["level"],
                         "wording": r["paraphrase"], "object": r.get("object"),
                         "question": r["question"], "target_answer": r["answer"],
                         "source_video": r["video"]})
        print(name)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "README.md").write_text(
        "# Training examples (supervision, not predictions)\n\n"
        "Each mp4 is one training example exactly as fed to the model:\n\n"
        "* left  — the video (the model sees it at 4 fps, 448x256, no timestamps)\n"
        "* right — top-down map: path so far in blue, objects as coloured dots,\n"
        "          black ring on the object the question names, arrow = camera\n"
        "* below — the exact question text and the exact target string\n\n"
        "`manifest.json` has the full question and target for each.\n\n"
        "The model is trained to emit ONLY the target line (e.g. `ANSWER: 214`).\n"
        "No reasoning, no working, no timestamps in the prompt.\n")
    print(f"\n-> {out}  ({len(manifest)} samples + manifest.json + README.md)")


if __name__ == "__main__":
    main()
