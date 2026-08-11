"""Scenes with natural, messy camera motion instead of one rigid template.

The sphere training set moved the same way every time: straight, one eased
turn, straight, stop. Everything here is loosened, and each clip mixes several
kinds of randomness rather than exercising one:

    translation   speed varies smoothly, with occasional pauses and short
                  backward stretches; plus lateral sway, so the walker drifts
                  sideways as well as forward — motion is not locked to facing
    yaw           one to three deliberate turns of varying size and duration,
                  on top of a continuous wandering drift
    pitch         looking up and down, smooth, with occasional glances down
    roll          slight head tilt, a couple of degrees, always changing
    bob           amplitude and rate vary per clip and scale with speed

Ground truth is untouched: a bearing is a horizontal angle, so pitch and roll
do not enter it, and yaw still means what it meant. Object placement and the
questions come from `sphere_gen` unchanged.

    .venv/bin/python natural_motion.py --n 8 --out probes/nat.json
"""

import argparse
import json
import math
import random
from pathlib import Path

import plane_gen as PG
import sphere_gen as SG

ROOT = Path(__file__).resolve().parent
FPS = PG.FPS


def smooth(v, win):
    """Symmetric moving average applied twice — near-Gaussian, so the result
    has no steps and a bounded derivative. This is where 'acceleration exists'
    gets enforced: targets are written as if they could change instantly, then
    every one of them is passed through here."""
    win = max(1, int(win))
    for _ in range(2):
        n = len(v)
        out = []
        for i in range(n):
            a, b = max(0, i - win), min(n, i + win + 1)
            out.append(sum(v[a:b]) / (b - a))
        v = out
    return v


def vnoise(rng, n, period, amp):
    """Smooth random signal: control points every `period` frames, eased."""
    k = max(2, int(n / period) + 2)
    ctrl = [rng.uniform(-1, 1) for _ in range(k)]
    out = []
    for i in range(n):
        t = i / period
        a = min(int(t), k - 2)
        f = t - a
        f = f * f * (3 - 2 * f)
        out.append(amp * (ctrl[a] * (1 - f) + ctrl[a + 1] * f))
    return out


def natural_path(rng, secs=10.0):
    """(poses, turns). Poses are (x, y, yaw, pitch, roll, eye)."""
    n = int(secs * FPS)
    base_speed = rng.uniform(1.0, 1.7)
    eye0 = rng.uniform(1.45, 1.8)

    # --- yaw: deliberate turns on top of a wandering drift ------------------
    yaw_rate = vnoise(rng, n, rng.uniform(45, 90), math.radians(rng.uniform(6, 16)) / FPS)
    turns = []
    for _ in range(rng.randint(1, 3)):
        deg = rng.uniform(30, 110) * rng.choice((-1, 1))
        # a bigger turn takes longer — a person does not pivot 110 degrees in
        # the time they take for 30
        dur = 1.0 + abs(deg) / 55.0 * rng.uniform(0.8, 1.25)
        start = rng.randint(int(0.15 * n), max(int(0.15 * n) + 1, int(0.75 * n)))
        w = max(2, int(dur * FPS))
        turns.append({"start_frame": start, "dir": "R" if deg > 0 else "L",
                      "deg": round(deg, 1)})
        for i in range(w):
            j = start + i
            if j < n:
                # eased profile, integrates to exactly `deg`
                yaw_rate[j] += (math.radians(deg) * math.sin(math.pi * (i + 0.5) / w)
                                / (w * 2 / math.pi))

    # --- speed: smooth variation, pauses, and a chance of walking backwards --
    sp = [base_speed * (1 + v) for v in vnoise(rng, n, rng.uniform(40, 80), 0.45)]
    for _ in range(rng.randint(0, 2)):                       # slow to a halt
        a = rng.randint(0, max(0, n - 60))
        for j in range(a, min(n, a + int(rng.uniform(0.6, 1.4) * FPS))):
            sp[j] = 0.0
    if rng.random() < 0.45:                                  # back up a little
        a = rng.randint(int(0.25 * n), max(int(0.25 * n) + 1, int(0.75 * n)))
        for j in range(a, min(n, a + int(rng.uniform(0.7, 1.5) * FPS))):
            sp[j] = -abs(sp[j]) * rng.uniform(0.35, 0.6)

    # --- lateral sway: movement is not locked to where the camera looks -----
    strafe = vnoise(rng, n, rng.uniform(35, 70), base_speed * rng.uniform(0.12, 0.4))

    pitch = vnoise(rng, n, rng.uniform(50, 100), math.radians(rng.uniform(4, 9)))
    if rng.random() < 0.5:                                   # a glance down
        a = rng.randint(0, max(0, n - 30))
        for j in range(a, min(n, a + int(rng.uniform(0.5, 1.1) * FPS))):
            pitch[j] -= math.radians(rng.uniform(6, 14))
    roll = vnoise(rng, n, rng.uniform(30, 70), math.radians(rng.uniform(1.2, 3.0)))

    # every target is smoothed before it is used, so speed ramps instead of
    # jumping, turns ease in and out, and the gaze drifts rather than snapping
    sp = smooth(sp, 0.45 * FPS)
    strafe = smooth(strafe, 0.40 * FPS)
    yaw_rate = smooth(yaw_rate, 0.30 * FPS)
    pitch = smooth(pitch, 0.35 * FPS)
    roll = smooth(roll, 0.35 * FPS)

    # a gentle footfall, not a shudder: a few millimetres, ~2 steps a second
    bob_amp = rng.uniform(0.003, 0.009)
    bob_rate = rng.uniform(0.24, 0.34)

    x = y = 0.0
    yaw = rng.uniform(0, 2 * math.pi)
    poses = []
    for i in range(n):
        yaw += yaw_rate[i]
        fx, fy = math.cos(yaw), math.sin(yaw)
        rx, ry = -math.sin(yaw), math.cos(yaw)
        x += (fx * sp[i] + rx * strafe[i]) / FPS
        y += (fy * sp[i] + ry * strafe[i]) / FPS
        moving = min(1.0, abs(sp[i]) / max(base_speed, 1e-6))
        h = eye0 + bob_amp * moving * math.sin(i * bob_rate)
        r = roll[i]
        poses.append((round(x, 4), round(y, 4), round(yaw, 5),
                      round(pitch[i], 5), round(r, 5), round(h, 4)))
    return poses, sorted(turns, key=lambda t: t["start_frame"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--secs", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--seed0", type=int, default=90000)
    ap.add_argument("--out", default="probes/nat.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    scenes = []
    while len(scenes) < args.n:
        poses, turns = natural_path(rng, args.secs * rng.uniform(0.9, 1.1))
        saved = PG.camera_path
        # sphere_gen places objects around whatever path it is given; feed it
        # a flat (x, y, yaw) view of ours so its visibility maths still works
        flat = [(p[0], p[1], p[2]) for p in poses]
        PG.camera_path = lambda *a, **k: (flat, turns)
        try:
            s = SG.draw(rng, args.seed0 + len(scenes))
        finally:
            PG.camera_path = saved
        if s is None:
            continue
        s["poses"] = poses                     # full 6-element poses to render
        s["n_turns"] = len(turns)
        s["turns"] = turns
        s["level"] = "natural"
        s["secs"] = round(len(poses) / FPS, 2)
        scenes.append(s)

    (ROOT / args.out).write_text(json.dumps({"note": "natural camera motion",
                                             "scenes": scenes}))
    import statistics as st
    print(f"{len(scenes)} scenes -> {args.out}")
    print(f"  {st.mean(s['secs'] for s in scenes):.1f}s average, "
          f"{st.mean(s['n_turns'] for s in scenes):.1f} deliberate turns/clip, "
          f"{st.mean(len(s['objects']) for s in scenes):.0f} objects/scene")


if __name__ == "__main__":
    main()
