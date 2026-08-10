"""Turn rendered plans into supervised examples, with worked traces.

One video supports several questions — how many corners, which way each went,
roughly where the start is, the exact bearing — so each rendered walk pays for
four or so training examples instead of one.

The traces are not free text. Each is generated from the plan's own geometry
and its arithmetic is checked against the simulator's ground truth before the
example is emitted (`verify_trace`), so a trace can never talk its way to an
answer that disagrees with the label.

Everything in a trace is expressed in quantities the model could read off the
video: elapsed seconds, and turn directions. The walker's speed is constant,
so "seconds of walking" doubles as the unit of distance and the bearing comes
out scale-free — no world units, no lattice steps, nothing the model has no
way to see.

    .venv/bin/python sft_data.py --plans probes/trainset.json \
        --videos out/train --out data/sft_train.jsonl
"""

import argparse
import json
import math
import random
from pathlib import Path

import generate as G

ROOT = Path(__file__).resolve().parent

# constant walking speed turns seconds into a distance unit
SEC_PER_STEP = G.floorplan.S / G.SPEED          # one lattice step, in seconds
MARKER_SEC = G.MARKER_AHEAD / G.SPEED           # X sits this far up leg 1

TURN_WORD = {"L": "left", "R": "right"}


def leg_headings(turns):
    """Heading of each leg in degrees clockwise, leg 1 = 0."""
    h, out = 0.0, [0.0]
    for t in turns:
        h += 90.0 if t == "R" else -90.0
        out.append(h)
    return out


def marker_seconds(legs):
    """How far up leg 1 the X sits, in walking-seconds.

    generate.start_marker shortens the offset when leg 1 is too short to hold
    it; mirror that here or the trace and the render disagree."""
    span = legs[0] * G.floorplan.S
    ahead = min(G.MARKER_AHEAD, max(1.8, span - 1.0))
    return ahead / G.SPEED


def solve(turns, legs):
    """Bearing to the X from the end, worked in the final camera's frame.

    Returns the bearing plus the per-leg terms, so the trace can show the same
    numbers it used rather than a plausible-looking reconstruction."""
    headings = leg_headings(turns)
    final = headings[-1]
    # heading of each leg relative to the final heading, which the question
    # calls north; a leg walked "north" carries the walker away from the end
    terms = []
    for L, h in zip(legs, headings):
        rel = (h - final) % 360.0
        secs = L * SEC_PER_STEP
        terms.append({"secs": secs, "rel_deg": rel,
                      "dx": secs * math.sin(math.radians(rel)),
                      "dy": secs * math.cos(math.radians(rel))})
    # start -> end displacement, in the final frame (x right, y forward)
    ex = sum(t["dx"] for t in terms)
    ey = sum(t["dy"] for t in terms)
    # the X is up leg 1, i.e. along leg 1's relative heading
    m_rel = (headings[0] - final) % 360.0
    m_sec = marker_seconds(legs)
    mx = m_sec * math.sin(math.radians(m_rel))
    my = m_sec * math.cos(math.radians(m_rel))
    # from the end, back to the X
    vx, vy = mx - ex, my - ey
    bearing = math.degrees(math.atan2(vx, vy)) % 360.0
    return bearing, terms, (vx, vy), (m_sec, m_rel)


def verify_trace(plan, tol=1.5):
    """Does the trace's own arithmetic land on the simulator's answer?"""
    got, *_ = solve(plan["turns"], plan["legs"])
    gt = plan["bearing_gt_deg"]
    d = abs(((got - gt + 180) % 360) - 180)
    return d <= tol, round(got, 2), round(d, 2)


def fmt(x, n=1):
    s = f"{x:.{n}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "-") else s


def trace_bearing(plan):
    turns, legs = plan["turns"], plan["legs"]
    bearing, terms, (vx, vy), (m_sec, m_rel) = solve(turns, legs)
    ev = plan.get("turn_events") or []

    lines = []
    if turns:
        when = ", ".join(f"{TURN_WORD[t['dir']]} at {fmt(t['t_start'])}s"
                         for t in ev) if ev else ", ".join(
                             TURN_WORD[t] for t in turns)
        lines.append(f"The walk turns {len(turns)} time"
                     f"{'s' if len(turns) != 1 else ''}: {when}.")
    else:
        lines.append("The walk never turns.")
    lines.append("Taking the final heading as north, and using seconds of "
                 "walking as the unit of distance:")
    for i, t in enumerate(terms, 1):
        lines.append(f"  leg {i}: {fmt(t['secs'])}s at {t['rel_deg']:03.0f} "
                     f"-> forward {fmt(t['dy'], 2)}, right {fmt(t['dx'], 2)}")
    ex = sum(t["dx"] for t in terms)
    ey = sum(t["dy"] for t in terms)
    lines.append(f"Start to end: forward {fmt(ey, 2)}, right {fmt(ex, 2)}.")
    lines.append(f"The X sits {fmt(m_sec)}s up leg 1, at "
                 f"forward {fmt(m_sec * math.cos(math.radians(m_rel)), 2)}, "
                 f"right {fmt(m_sec * math.sin(math.radians(m_rel)), 2)}.")
    lines.append(f"From the end back to the X: forward {fmt(vy, 2)}, "
                 f"right {fmt(vx, 2)}.")
    lines.append(f"Bearing = atan2(right, forward) = {bearing:.0f} degrees.")
    return "\n".join(lines), f"ANSWER: {round(bearing) % 360:03d}"


def trace_count(plan):
    ev = plan.get("turn_events") or []
    n = plan["n_turns"]
    if ev:
        when = ", ".join(f"{fmt(t['t_start'])}s" for t in ev)
        body = (f"The corridor changes direction at {when}." if ev
                else "The corridor never changes direction.")
    else:
        body = f"The corridor changes direction {n} times."
    return body, f"ANSWER: {n}"


def trace_seq(plan):
    ev = plan.get("turn_events") or []
    turns = plan["turns"]
    if ev:
        body = " ".join(f"{fmt(t['t_start'])}s: {TURN_WORD[t['dir']]}."
                        for t in ev)
    else:
        body = " ".join(f"turn {i}: {TURN_WORD[t]}."
                        for i, t in enumerate(turns, 1))
    return body, f"ANSWER: {turns if turns else 'none'}"


def trace_coarse(plan):
    b = plan["bearing_gt_deg"]
    quad = ["AHEAD", "RIGHT", "BEHIND", "LEFT"][int(((b + 45.0) % 360.0) // 90)]
    body = (f"Following the turns and the length of each straight puts the X "
            f"at a bearing of about {b:.0f} degrees from the final heading, "
            f"which is nearest {quad.lower()}.")
    return body, f"ANSWER: {quad}"


import diagnose  # noqa: E402  (question text lives with the diagnostics)
import eval_probes  # noqa: E402  (the bearing question must match the eval's)

TASKS = {
    "count": (diagnose.TASKS["count"]["q"] + diagnose.TASKS["count"]["fmt"],
              trace_count),
    "seq": (diagnose.TASKS["seq"]["q"] + diagnose.TASKS["seq"]["fmt"],
            trace_seq),
    "coarse": (diagnose.TASKS["coarse"]["q"] + diagnose.TASKS["coarse"]["fmt"],
               trace_coarse),
    "bearing": (None, trace_bearing),     # filled from the benchmark's own text
}


def build(plans, video_dir, question, tasks, mix, seed, with_trace=True):
    rng = random.Random(seed)
    # exactly the string the eval sends, imported rather than restated: train
    # on one wording and score on another and the gap is measuring the wording
    TASKS["bearing"] = (question + eval_probes.ANSWER_FORMAT, trace_bearing)

    rows, no_video, bad_trace = [], 0, 0
    for plan in plans:
        vid = video_dir / f"{plan['id']:03d}.mp4"
        if not vid.exists():
            no_video += 1
            continue
        ok, got, d = verify_trace(plan)
        if not ok:
            bad_trace += 1
            continue
        for task in tasks:
            if rng.random() > mix.get(task, 1.0):
                continue
            q, tracer = TASKS[task]
            body, answer = tracer(plan)
            rows.append({
                "video": str(vid), "task": task, "question": q,
                "think": body if with_trace else "",
                "answer": answer,
                "plan_id": plan["id"], "n_turns": plan["n_turns"],
                "duration_s": plan["duration_s"],
            })
    return rows, no_video, bad_trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", default="probes/trainset.json")
    ap.add_argument("--videos", default="out/train")
    ap.add_argument("--out", default="data/sft_train.jsonl")
    ap.add_argument("--tasks", default="count,seq,coarse,bearing")
    ap.add_argument("--bearing-frac", type=float, default=1.0)
    ap.add_argument("--aux-frac", type=float, default=0.5,
                    help="probability of emitting each non-bearing task")
    ap.add_argument("--question", choices=("neutral", "original"),
                    default="neutral")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--no-trace", dest="with_trace", action="store_false")
    args = ap.parse_args()

    plans = json.loads((ROOT / args.plans).read_text())["plans"]
    question = (eval_probes.NEUTRAL_QUESTION if args.question == "neutral"
                else json.loads((ROOT / "probes/bearing_probes.json"
                                 ).read_text())["question"])
    tasks = args.tasks.split(",")
    mix = {t: args.aux_frac for t in tasks}
    mix["bearing"] = args.bearing_frac

    rows, no_video, bad_trace = build(plans, ROOT / args.videos, question,
                                      tasks, mix, args.seed, args.with_trace)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    counts = {}
    for r in rows:
        counts[r["task"]] = counts.get(r["task"], 0) + 1
    print(f"{len(rows)} examples from "
          f"{len(plans) - no_video - bad_trace} videos "
          f"({no_video} plans not rendered, {bad_trace} dropped for a trace "
          f"that disagreed with the label)")
    print("  " + "  ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
