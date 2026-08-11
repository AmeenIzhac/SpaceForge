"""Rebuild the open-plain probes with a prompt matched to the corridor one.

Three flaws in v1, all mine, all in the test rather than the model:

  1. The corridor prompt tells the model the target is off-screen at the end
     and must be recovered from the route walked. The plain prompt said no
     such thing, so "where is the green pillar" reads as a perception
     question, not a path-integration one.
  2. `eval_probes.ANSWER_FORMAT` — appended to every question — talks about
     "the X", which does not exist in this world. Here the question carries
     its own format line instead (run with `--no-format-hint`).
  3. 25 of 84 v1 probes asked about an object that never appeared in its
     video: unanswerable, not failed. Only objects that are on screen at
     some point are asked about now.

Visibility also needed fixing: three.js takes a VERTICAL fov, so the 68 here
is 100 degrees horizontal (half-angle 50), not the corridor renderer's 34.
v1 labelled "visible" with a 30-degree threshold and mis-sorted the rows.

    .venv/bin/python plane3_probes_v2.py    -> probes/plane3_v2.json
"""

import json
import math
from pathlib import Path

import generate as G
import make_probes as MP

ROOT = Path(__file__).resolve().parent
HALF_FOV = math.degrees(math.atan(math.tan(math.radians(68) / 2) * 960 / 544))

FORMAT = ("\n\nAny integer from 0 to 359 is a valid answer — the object is "
          "usually not at an exact multiple of 90, so give your best estimate "
          "rather than rounding to the nearest quarter turn.\n"
          "End your reply with exactly one line, and nothing after it:\n"
          "ANSWER: <degrees>")

Q_FINAL = (
    "This is a first-person video of someone walking across an open plain "
    "with several objects standing on the ground. One of them is a {name}; "
    "it is visible at some point during the walk, but by the final frame it "
    "may be out of shot — if it is, you have to work out where it is from "
    "the route that was walked. Freeze on the final frame and treat the "
    "direction the camera is facing at that moment as north (000 degrees). "
    "What is the compass bearing from the camera's final position to the "
    "{name}? Answer in degrees clockwise from that forward direction, 0-359: "
    "090 is directly to the right, 180 directly behind, 270 directly to the "
    "left.") + FORMAT

Q_START = (
    "This is a first-person video of someone walking across an open plain "
    "with several objects standing on the ground. One of them is a {name}; "
    "it is visible at some point during the walk. Think back to the very "
    "first frame of the video, and treat the direction the camera was facing "
    "at that moment as north (000 degrees). From the camera's starting "
    "position, what was the compass bearing to the {name}? You have to work "
    "it out from where the object appears and the route that was walked. "
    "Answer in degrees clockwise from that facing, 0-359: 090 is directly to "
    "the right, 180 directly behind, 270 directly to the left.") + FORMAT


def ever_seen(poses, o):
    for (x, y, yaw) in poses:
        b = G.bearing_deg((x, y, yaw), (o["x"], o["y"]))
        if min(b, 360 - b) <= HALF_FOV:
            return True
    return False


def main():
    scenes = json.load(open(ROOT / "photoreal/plane3_spec.json"))
    rows = []
    for S in scenes:
        poses = S["poses"]
        first, last = poses[0], poses[-1]
        seen = [o for o in S["objects"] if ever_seen(poses, o)]
        for o in seen:
            b = G.bearing_deg((last[0], last[1], last[2]), (o["x"], o["y"]))
            rows.append({
                "id": len(rows), "video": f"{S['id']:03d}.mp4",
                "scene": S["id"], "object": o["name"], "frame": "final",
                "level": ("final_visible"
                          if min(b, 360 - b) <= HALF_FOV else "final_hidden"),
                "question": Q_FINAL.format(name=o["name"]),
                "bearing_gt_deg": round(b, 2),
                "distance": round(math.hypot(o["x"] - last[0],
                                             o["y"] - last[1]), 2)})
        for o in seen[:2]:
            b = G.bearing_deg((first[0], first[1], first[2]),
                              (o["x"], o["y"]))
            rows.append({
                "id": len(rows), "video": f"{S['id']:03d}.mp4",
                "scene": S["id"], "object": o["name"], "frame": "start",
                "level": "start",
                "question": Q_START.format(name=o["name"]),
                "bearing_gt_deg": round(b, 2),
                "distance": round(math.hypot(o["x"] - first[0],
                                             o["y"] - first[1]), 2)})

    gts = [r["bearing_gt_deg"] for r in rows]
    c = min(range(360), key=lambda k: sum(MP.circ_err(k, b) for b in gts))
    doc = {"note": "open-plain transfer probes v2: matched prompt, "
                   "self-contained format line (use --no-format-hint), "
                   "only ever-visible objects, FOV-correct visibility labels",
           "const_baseline_deg": round(
               sum(MP.circ_err(c, b) for b in gts) / len(gts), 2),
           "const_baseline_answer": c, "plans": rows}
    (ROOT / "probes/plane3_v2.json").write_text(json.dumps(doc, indent=1))

    import collections
    print(f"{len(rows)} probes (v1 had 84, of which 25 were unanswerable)")
    print("  levels:", dict(collections.Counter(r["level"] for r in rows)))
    print(f"  best constant {doc['const_baseline_deg']}@{c:03d}")


if __name__ == "__main__":
    main()
