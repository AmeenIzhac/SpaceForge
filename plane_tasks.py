"""Questions over open-plane scenes — many kinds, some never trained.

The corridor model learned one question in one world and could not answer a
temporal variant of it (§18). So here the *type* of question varies, not just
its wording, and three types are withheld from training entirely. If the model
built a reusable spatial state it can answer them; if it learned a mapping
from video to number, it cannot.

  trained             bearing_final   object's bearing from the last frame
                      bearing_start   ... from the first frame (other frame
                                      of reference, same scene)
                      quadrant        AHEAD/RIGHT/BEHIND/LEFT at the end
                      count           how many objects in the scene
                      turns           how many times the walk turned

  held out (test)     bearing_mid     bearing one second before the last turn
                      bearing_obj     bearing from one OBJECT to another,
                                      in the final camera's frame — allocentric,
                                      neither endpoint is the camera
                      back_to_start   bearing to where the walk began (the
                                      corridor task, in a world with no marker)

Wording varies too: three paraphrases in training, a fourth reserved for test.
Several questions are asked per scene about *different* objects, so the video
alone never determines the answer — the named object has to be bound.

    .venv/bin/python plane_tasks.py --scenes probes/plane_train.json \
        --split train --videos out/plane_train --out data/plane_train.jsonl
"""

import argparse
import json
import math
import random
from pathlib import Path

import generate as G
import make_probes as MP

ROOT = Path(__file__).resolve().parent

DEG = (" Answer in degrees clockwise from that forward direction, 0-359: 090 "
       "is directly to the right, 180 directly behind, 270 directly to the "
       "left.\n\nAny integer from 0 to 359 is a valid answer — the object is "
       "usually not at an exact multiple of 90, so give your best estimate "
       "rather than rounding to the nearest quarter turn.\nEnd your reply "
       "with exactly one line, and nothing after it:\nANSWER: <degrees>")

INTRO = [
    ("plain", "This is a first-person video of someone walking across an open "
              "plain with objects scattered around."),
    ("terse", "A first-person video of a walk across open ground past several "
              "objects."),
    ("story", "Someone walks across an open field, filming as they go, "
              "passing various objects standing on the ground."),
    ("held_out", "In this first-person video the camera travels across an "
                 "open landscape dotted with objects."),   # test split only
]

BEARING_FINAL = [
    " One of them is a {name}; it comes into view at some point during the "
    "walk, but by the final frame it may be out of shot — if so, work out "
    "where it is from the route that was walked. Freeze on the final frame "
    "and treat the direction the camera is facing at that moment as north "
    "(000 degrees). What is the compass bearing from the camera's final "
    "position to the {name}?",
    " At the end of the video, taking the camera's own facing as north (000 "
    "degrees), which compass bearing is the {name} in? It was in view earlier "
    "and may be off screen now, so work it out from how the camera moved.",
    " Consider the very last frame, and call the direction the camera looks "
    "there north (000 degrees). Relative to that, in what compass direction "
    "does the {name} lie?",
]
BEARING_START = [
    " Think back to the very first frame of the video and treat the direction "
    "the camera was facing at that moment as north (000 degrees). From the "
    "camera's STARTING position, what was the compass bearing to the {name}?",
    " Rewind to the beginning. Taking the camera's facing in the opening "
    "frame as north (000 degrees), in which compass direction did the {name} "
    "lie from where the walk started?",
]
QUADRANT = (" Freeze on the final frame. Relative to the direction the camera "
            "is facing at that moment, roughly where is the {name}? Choose "
            "one: AHEAD, RIGHT, BEHIND, LEFT.\n\nEnd your reply with exactly: "
            "ANSWER: <one of AHEAD, RIGHT, BEHIND, LEFT>")
COUNT = (" How many different objects appear on screen at some point during "
         "the video? Count each object once, however often it appears.\n\nEnd "
         "your reply with exactly: ANSWER: <integer>")
TURNS = (" How many times did the walker change direction during the walk? "
         "Count only the turns, not the straight stretches.\n\nEnd your reply "
         "with exactly: ANSWER: <integer>")

# held-out types
BEARING_MID = (" Think back to the moment one second BEFORE the walker began "
               "their final change of direction — a moment in the MIDDLE of "
               "the walk, not the end. Treat the direction the camera was "
               "facing at that moment as north (000 degrees). What was the "
               "compass bearing from the camera's position at that moment to "
               "the {name}?")
BEARING_OBJ = (" Two of the objects are a {a} and a {b}. Treat the direction "
               "the camera is facing in the FINAL frame as north (000 "
               "degrees). Using that same north, what is the compass bearing "
               "from the {a} to the {b}? This is the direction you would walk "
               "if you were standing at the {a} and heading for the {b}.")
BACK_TO_START = (" Freeze on the final frame and treat the direction the "
                 "camera is facing at that moment as north (000 degrees). "
                 "What is the compass bearing from the camera's final "
                 "position back to the spot where the walk began?")

QUAD = ["AHEAD", "RIGHT", "BEHIND", "LEFT"]


def quadrant(b):
    return QUAD[int(((b + 45.0) % 360.0) // 90.0)]


def mid_pose(scene):
    """One second before the last turn starts (or mid-walk if no turns)."""
    poses, turns = scene["poses"], scene["turns"]
    f = (turns[-1]["start_frame"] - scene["fps"]) if turns else len(poses) // 2
    return poses[max(0, min(f, len(poses) - 1))]


def rows_for_scene(scene, split, rng, video_dir):
    intros = INTRO[:3] if split == "train" else INTRO[3:]
    vid = str(ROOT / video_dir / f"{scene['id']:04d}.mp4")
    seen = [o for o in scene["objects"] if o["seen_early"]]
    if not seen:
        return []
    poses = scene["poses"]
    first, last = poses[0], poses[-1]
    out = []

    def add(kind, text, answer, gt=None, extra=None):
        tag, intro = intros[rng.randrange(len(intros))]
        out.append({"video": vid, "scene": scene["id"], "task": kind,
                    "level": scene["level"], "paraphrase": tag,
                    "question": intro + text, "think": "",
                    "answer": f"ANSWER: {answer}",
                    "bearing_gt_deg": gt, **(extra or {})})

    # --- trained types -----------------------------------------------------
    for o in rng.sample(seen, min(2, len(seen))):
        b = o["bearing_final"]
        add("bearing_final",
            rng.choice(BEARING_FINAL).format(name=o["name"]) + DEG,
            f"{round(b) % 360:03d}", b,
            {"object": o["name"], "visible_at_end": o["visible_at_end"]})
    o = rng.choice(seen)
    b = G.bearing_deg((first[0], first[1], first[2]), (o["x"], o["y"]))
    add("bearing_start", rng.choice(BEARING_START).format(name=o["name"]) + DEG,
        f"{round(b) % 360:03d}", round(b, 2), {"object": o["name"]})
    o = rng.choice(seen)
    add("quadrant", QUADRANT.format(name=o["name"]),
        quadrant(o["bearing_final"]), None, {"object": o["name"]})
    # only objects that are actually seen: an object behind the camera
    # for the whole walk cannot be counted, and supervising it would be
    # teaching a guess
    add("count", COUNT, sum(o["seen_frames"] > 0 for o in scene["objects"]))
    add("turns", TURNS, scene["n_turns"])

    # --- held-out types (test split only) ----------------------------------
    if split == "test":
        o = rng.choice(seen)
        mp = mid_pose(scene)
        b = G.bearing_deg((mp[0], mp[1], mp[2]), (o["x"], o["y"]))
        add("bearing_mid", BEARING_MID.format(name=o["name"]) + DEG,
            f"{round(b) % 360:03d}", round(b, 2), {"object": o["name"]})
        if len(seen) >= 2:
            a, c = rng.sample(seen, 2)
            # bearing from a to c, expressed in the final camera's frame
            b = G.bearing_deg((a["x"], a["y"], last[2]), (c["x"], c["y"]))
            add("bearing_obj",
                BEARING_OBJ.format(a=a["name"], b=c["name"]) + DEG,
                f"{round(b) % 360:03d}", round(b, 2),
                {"object": f"{a['name']}->{c['name']}"})
        b = G.bearing_deg((last[0], last[1], last[2]), (first[0], first[1]))
        add("back_to_start", BACK_TO_START + DEG,
            f"{round(b) % 360:03d}", round(b, 2), {"object": "start point"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--split", choices=("train", "test"), required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    scenes = json.loads((ROOT / args.scenes).read_text())["scenes"]
    rng = random.Random(args.seed)
    rows = []
    for s in scenes:
        if not (ROOT / args.videos / f"{s['id']:04d}.mp4").exists():
            continue
        rows += rows_for_scene(s, args.split, rng, args.videos)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".jsonl":
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    else:                                   # probe file for eval_probes
        for i, r in enumerate(rows):
            r["id"] = i
            r["video"] = Path(r["video"]).name
            r["level_path"] = r["level"]
        bear = [r for r in rows if r["bearing_gt_deg"] is not None]
        gts = [r["bearing_gt_deg"] for r in bear]
        c = min(range(360), key=lambda k: sum(MP.circ_err(k, b) for b in gts))
        out.write_text(json.dumps(
            {"note": "open-plane probes", "const_baseline_deg": round(
                sum(MP.circ_err(c, b) for b in gts) / len(gts), 2),
             "const_baseline_answer": c,
             "plans": [dict(r, level=r["task"]) for r in bear]}, indent=1))

    import collections
    print(f"{len(rows)} rows -> {args.out}")
    print("  by task:", dict(collections.Counter(r["task"] for r in rows)))
    print("  by level:", dict(sorted(collections.Counter(
        r["level"] for r in rows).items())))
    bear = [r for r in rows if r["bearing_gt_deg"] is not None]
    if bear:
        gts = [r["bearing_gt_deg"] for r in bear]
        c = min(range(360), key=lambda k: sum(MP.circ_err(k, b) for b in gts))
        print(f"  bearing rows {len(bear)}, best constant "
              f"{sum(MP.circ_err(c, b) for b in gts)/len(gts):.1f}@{c:03d} "
              f"(uniform guess 90)")


if __name__ == "__main__":
    main()
