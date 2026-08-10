"""Render the probe plans to video.

make_probes.py writes plans and ground truth but no pixels; this draws them.
Each plan's `turns`/`legs`/`seed` go straight into generate.make_sample, which
is the same call that measured the bearing, so the video and the ground truth
cannot drift apart. A rendered plan is skipped on re-runs, so an interrupted
batch resumes where it stopped.

    .venv/bin/python render_probes.py                    # -> out/probes/
    .venv/bin/python render_probes.py --workers 3        # one per GPU
    .venv/bin/python render_probes.py --plans probes/easy_probes.json \
        --out out/easy
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def render_one(job):
    """Render one plan in this worker process. Returns (id, seconds, error)."""
    plan, out_dir, gpu, speed = job
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # EGL picks its device independently of CUDA_VISIBLE_DEVICES
        os.environ["EGL_DEVICE_ID"] = str(gpu)

    import generate as G

    # Speed only changes how fast the camera traverses the polyline — the
    # route, the final pose and therefore every bearing answer are untouched.
    # So the same plan file can be re-rendered short without restating any
    # ground truth: a 3x walk turns a 24 s clip into an 8 s one, which renders
    # 3x faster and, sampled at 3x the fps, hands the model the same frames.
    if speed:
        G.SPEED = speed

    G.OUT_DIR = Path(out_dir)
    name = f"{plan['id']:03d}"
    t0 = time.time()
    try:
        G.make_sample(name, plan["turns"], plan["legs"], plan["seed"])
    except Exception as exc:                      # keep the batch going
        return plan["id"], time.time() - t0, f"{type(exc).__name__}: {exc}"
    return plan["id"], time.time() - t0, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", default="probes/bearing_probes.json")
    ap.add_argument("--out", default="out/probes")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--gpus", default="0,1,2",
                    help="round-robin EGL devices, or 'none'")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--speed", type=float, default=None,
                    help="walk speed in world units/s; higher makes the "
                         "same walk a shorter video (default 1.26)")
    ap.add_argument("--force", action="store_true",
                    help="re-render plans that already have an mp4")
    args = ap.parse_args()

    plans = json.loads((ROOT / args.plans).read_text())["plans"]
    if args.limit:
        plans = plans[: args.limit]

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in plans
            if args.force or not (out_dir / f"{p['id']:03d}.mp4").exists()]
    done_already = len(plans) - len(todo)
    if done_already:
        print(f"{done_already} already rendered, {len(todo)} to go")
    if not todo:
        print(f"nothing to do -> {out_dir}")
        return

    gpus = ([None] if args.gpus == "none"
            else [int(g) for g in args.gpus.split(",")])
    # longest first: the 40 s walks should start before the 4 s ones, or the
    # tail of the batch is one worker rendering a long video alone
    todo.sort(key=lambda p: -p["n_frames"])
    jobs = [(p, str(out_dir), gpus[i % len(gpus)], args.speed)
            for i, p in enumerate(todo)]

    total_frames = sum(p["n_frames"] for p in todo)
    print(f"rendering {len(todo)} plans, {total_frames} frames "
          f"({total_frames / 30 / 60:.0f} min of video) on {args.workers} "
          f"workers over GPUs {gpus}")

    t0 = time.time()
    failures, frames_done = [], 0
    ctx = mp.get_context("spawn")            # each worker needs its own GL ctx
    with ctx.Pool(args.workers) as pool:
        for n, (pid, secs, err) in enumerate(
                pool.imap_unordered(render_one, jobs), 1):
            frames_done += next(p["n_frames"] for p in todo if p["id"] == pid)
            if err:
                failures.append((pid, err))
                print(f"[{n}/{len(todo)}] {pid:03d} FAILED {err}", flush=True)
            else:
                el = time.time() - t0
                eta = el * (total_frames - frames_done) / max(frames_done, 1)
                print(f"[{n}/{len(todo)}] {pid:03d} {secs:5.1f}s  "
                      f"elapsed {el / 60:4.1f}m  eta {eta / 60:4.1f}m",
                      flush=True)

    print(f"\n{len(todo) - len(failures)}/{len(todo)} rendered in "
          f"{(time.time() - t0) / 60:.1f} min -> {out_dir}")
    for pid, err in failures:
        print(f"  FAILED {pid:03d}: {err}")


if __name__ == "__main__":
    main()
