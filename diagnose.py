"""Decompose the bearing probe into the abilities it is built out of.

A single number on the full task says "it failed" without saying where. The
bearing answer needs four things to go right:

    see      the red X is at the start, and the walk ends somewhere else
    count    how many corners the walk turned
    sign     which way each corner went
    integrate how long each straight leg ran, combined into a displacement

so this asks each one separately, on the same videos, and also sweeps the
frame sampling — a turn takes about 1.6 s, and at the default 2 fps with
temporal patching the model gets one view every 2.4 s, which can alias a
corner away entirely. A model that cannot count corners it never saw is a
preprocessing result, not a model result.

    .venv/bin/python diagnose.py --task count --sweep --gpus 0,1,2
    .venv/bin/python diagnose.py --task count,seq,coarse --frames 192 \
        --width 384 --height 224 --thinking
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DEFAULT = "Qwen/Qwen3.5-9B"

PREAMBLE = ("This is a first-person video of someone walking through "
            "corridors. ")

TASKS = {
    "count": {
        "q": PREAMBLE + "How many times did the walker turn a corner during "
             "the walk? Count only the corners where the corridor changed "
             "direction. Reply with a single integer.",
        "fmt": "\n\nEnd your reply with exactly: ANSWER: <integer>",
    },
    "seq": {
        "q": PREAMBLE + "List every corner the walker turned, in order, as "
             "the letters L (left) and R (right) with no spaces — for "
             "example LRR for a left then two rights.",
        "fmt": "\n\nEnd your reply with exactly: ANSWER: <letters>",
    },
    "coarse": {
        # same de-contaminated wording as eval_probes.NEUTRAL_QUESTION: saying
        # the X is "straight ahead" invites the model to answer AHEAD from the
        # prompt alone
        "q": PREAMBLE + "A large red X is painted on the floor at the spot "
             "where the walk begins; it is visible in the opening seconds and "
             "the walker then leaves it behind, so by the final frame it is "
             "out of shot. Freeze on the final frame. Relative to the "
             "direction the camera is facing at that moment, roughly where is "
             "the red X? Choose one: AHEAD, RIGHT, BEHIND, LEFT.",
        "fmt": "\n\nEnd your reply with exactly: ANSWER: <one of AHEAD, "
               "RIGHT, BEHIND, LEFT>",
    },
    "return_secs": {
        "q": PREAMBLE + "A large red X is painted on the floor at the spot "
             "where the walk begins; the walker leaves it behind. The walker's "
             "pace is constant throughout the video. From the final position, "
             "how many seconds of walking at that same pace would it take to "
             "walk in a straight line directly back to the red X?",
        "fmt": "\n\nEnd your reply with exactly: ANSWER: <seconds>",
    },
    "final_heading": {
        "q": PREAMBLE + "Compare the direction the camera faces in the very "
             "first frame with the direction it faces in the very last "
             "frame. By how many degrees did the camera's heading rotate in "
             "total over the walk, and in which direction? Reply with a "
             "signed angle in degrees, clockwise positive, in the range "
             "-360 to 360.",
        "fmt": "\n\nEnd your reply with exactly: ANSWER: <signed integer>",
    },
}

QUADRANTS = ["AHEAD", "RIGHT", "BEHIND", "LEFT"]


def quadrant(bearing):
    """0/90/180/270 +-45 -> AHEAD/RIGHT/BEHIND/LEFT."""
    return QUADRANTS[int(((bearing + 45.0) % 360.0) // 90.0)]


def truth(task, plan):
    if task == "count":
        return plan["n_turns"]
    if task == "seq":
        return plan["turns"]
    if task == "coarse":
        return quadrant(plan["bearing_gt_deg"])
    if task == "final_heading":
        return plan["net_turn_deg"]
    if task == "return_secs":
        return round(plan["distance_to_x"] / 1.26, 1)   # generate.SPEED
    raise KeyError(task)


def parse(task, text):
    m = re.findall(r"ANSWER\s*[:\-]?\s*([A-Za-z0-9+\-]+)", text)
    tok = m[-1] if m else None
    if task in ("count", "final_heading"):
        nums = re.findall(r"[+-]?\d+", tok) if tok else re.findall(r"[+-]?\d+", text)
        return int(nums[-1]) if nums else None
    if task == "return_secs":
        nums = re.findall(r"[+-]?\d+(?:\.\d+)?", tok or text)
        return float(nums[-1]) if nums else None
    if task == "seq":
        # "none" is the right answer for a walk with no corners, and the
        # trained model says exactly that — scoring it as unparsed marked 10
        # correct answers wrong
        if tok and tok.lower() in ("none", "nothing", "n/a"):
            return ""
        if tok and re.fullmatch(r"[LRlr]*", tok):
            return tok.upper()
        m2 = re.findall(r"\b([LRlr]{1,12})\b", text)
        return m2[-1].upper() if m2 else None
    if task == "coarse":
        if tok and tok.upper() in QUADRANTS:
            return tok.upper()
        hits = [w for w in re.findall(r"[A-Za-z]+", text.upper())
                if w in QUADRANTS]
        return hits[-1] if hits else None
    return None


def score(task, pred, gt):
    """Task-appropriate correctness plus a graded error where one exists."""
    if pred is None:
        return {"correct": False, "err": None}
    if task == "count":
        return {"correct": pred == gt, "err": abs(pred - gt)}
    if task == "seq":
        # exact string, plus how many positions line up when lengths match
        hits = sum(a == b for a, b in zip(pred, gt))
        return {"correct": pred == gt, "err": len(gt) - hits,
                "len_ok": len(pred) == len(gt)}
    if task == "coarse":
        return {"correct": pred == gt, "err": 0 if pred == gt else 1}
    if task == "final_heading":
        d = abs(((pred - gt + 180) % 360) - 180)
        return {"correct": d <= 20, "err": d}
    if task == "return_secs":
        d = abs(pred - gt)
        return {"correct": d <= max(1.5, 0.15 * gt), "err": round(d, 2)}
    return {"correct": False, "err": None}


def worker_main(args, gpu, shard, n_shards, jobs, out_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("HF_HUB_CACHE", args.hf_cache)
    os.environ.pop("HUGGINGFACE_HUB_CACHE", None)

    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from eval_probes import load_frames

    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation=args.attn)
    if getattr(args, "adapter", None):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.eval()

    video_dir = ROOT / args.videos
    mine = jobs[shard::n_shards]
    done = set()
    if out_path.exists():
        done = {(r["task"], r["id"], r["frames"], r["width"])
                for r in map(json.loads, out_path.read_text().splitlines()) if r}
    mine = [j for j in mine
            if (j["task"], j["plan"]["id"], j["frames"], j["width"]) not in done]

    cache = {}
    with open(out_path, "a") as f:
        for i, job in enumerate(mine, 1):
            plan, task = job["plan"], job["task"]
            key = (plan["id"], job["frames"], job["width"], job["height"])
            if key not in cache:
                cache.clear()
                cache[key] = load_frames(
                    video_dir / f"{plan['id']:03d}.mp4", args.fps,
                    job["frames"], (job["width"], job["height"]))
            frames, meta, duration = cache[key]

            spec = TASKS[task]
            text = processor.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "video"},
                    {"type": "text", "text": spec["q"] + spec["fmt"]}]}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=args.thinking)
            inputs = processor(text=[text], videos=[frames],
                               video_metadata=[meta], do_sample_frames=False,
                               return_tensors="pt").to(model.device)
            n_in = inputs["input_ids"].shape[1]

            t0 = time.time()
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=True, temperature=1.0,
                                     top_p=0.95, top_k=20)
            n_gen = int(out.shape[1] - n_in)
            reply = processor.tokenizer.decode(out[0][n_in:],
                                               skip_special_tokens=True)
            finished = (not args.thinking) or ("</think>" in reply)
            body = reply.split("</think>")[-1] if "</think>" in reply else reply
            pred = parse(task, body) if finished else None
            gt = truth(task, plan)

            rec = {"task": task, "id": plan["id"], "frames": job["frames"],
                   "width": job["width"], "height": job["height"],
                   "slot_s": round(duration / max(len(frames) / 2, 1), 2),
                   "n_turns": plan["n_turns"], "video_s": round(duration, 1),
                   "gt": gt, "pred": pred, **score(task, pred, gt),
                   "prompt_tokens": n_in, "gen_tokens": n_gen,
                   "secs": round(time.time() - t0, 1), "reply": reply}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[gpu{gpu} {i}/{len(mine)}] {task} id={plan['id']:03d} "
                  f"f={job['frames']} gt={gt} pred={pred} "
                  f"{'OK ' if rec['correct'] else 'MISS'} ({rec['secs']}s)",
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--adapter", default=None,
                    help="LoRA checkpoint to score instead of the base")
    ap.add_argument("--plans", default="probes/bearing_probes.json")
    ap.add_argument("--videos", default="out/probes")
    ap.add_argument("--out", default="runs/diag")
    ap.add_argument("--task", default="count")
    ap.add_argument("--gpus", default="0,1,2")
    # 1e9 means "as many frames as fit"; 2.0 matches what the
    # trainer and eval_probes feed, which is what a tuned
    # checkpoint has actually seen
    ap.add_argument("--fps", type=float, default=1e9)
    ap.add_argument("--frames", type=int, default=128)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--sweep", action="store_true",
                    help="frame/resolution ladder instead of one setting")
    ap.add_argument("--n", type=int, default=20, help="probes per cell")
    ap.add_argument("--max-new-tokens", type=int, default=6144)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--hf-cache", default="/mnt/data0/ameen/hf_cache/hub")
    ap.add_argument("--thinking", action="store_true", default=False)
    args = ap.parse_args()

    doc = json.loads((ROOT / args.plans).read_text())
    plans = doc["plans"]
    have = {int(p.stem) for p in (ROOT / args.videos).glob("*.mp4")}
    plans = [p for p in plans if p["id"] in have]

    # spread the subset over turn counts so "can it count" is not answered
    # from a sample that is all 1-turn walks
    by_turns = {}
    for p in plans:
        by_turns.setdefault(p["n_turns"], []).append(p)
    subset, i = [], 0
    while len(subset) < min(args.n, len(plans)):
        for k in sorted(by_turns):
            if i < len(by_turns[k]) and len(subset) < args.n:
                subset.append(by_turns[k][i])
        i += 1

    if args.sweep:
        # equal token cost per cell (~7k video tokens): trade space for time
        cells = [(32, 896, 512), (64, 640, 352), (128, 448, 256),
                 (192, 384, 224), (256, 320, 192)]
    else:
        cells = [(args.frames, args.width, args.height)]

    jobs = [{"plan": p, "task": t, "frames": f, "width": w, "height": h}
            for f, w, h in cells
            for t in args.task.split(",")
            for p in subset]

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"{len(jobs)} jobs: {len(subset)} probes x {args.task.split(',')} "
          f"x {len(cells)} sampling cells, thinking={args.thinking}")

    gpus = [int(g) for g in args.gpus.split(",")]
    if len(gpus) == 1:
        worker_main(args, gpus[0], 0, 1, jobs, out_dir / "raw_0.jsonl")
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=worker_main,
                             args=(args, g, s, len(gpus), jobs,
                                   out_dir / f"raw_{s}.jsonl"))
                 for s, g in enumerate(gpus)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

    recs = []
    for f in sorted(out_dir.glob("raw_*.jsonl")):
        recs += [json.loads(l) for l in f.read_text().splitlines() if l]
    (out_dir / "results.json").write_text(json.dumps(recs, indent=2))

    print(f"\n{'task':14s} {'frames':>7s} {'slot_s':>7s} {'n':>4s} "
          f"{'acc':>6s}  mean|err|")
    seen = {}
    for r in recs:
        seen.setdefault((r["task"], r["frames"]), []).append(r)
    for (task, fr), rs in sorted(seen.items()):
        acc = sum(r["correct"] for r in rs) / len(rs)
        errs = [r["err"] for r in rs if r["err"] is not None]
        slot = sum(r["slot_s"] for r in rs) / len(rs)
        print(f"{task:14s} {fr:7d} {slot:7.2f} {len(rs):4d} {acc:6.2f}  "
              f"{sum(errs) / len(errs) if errs else float('nan'):.2f}")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
