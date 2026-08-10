"""Run a Qwen3.5-class VLM on the bearing probes.

Feeds each probe video plus its question to the model and records the bearing
it answers, then scores absolute circular error against `bearing_gt_deg`.

The question text is the benchmark's own, verbatim; only a short output-format
line is appended so the answer can be parsed without judging free text.

Shards over GPUs: each worker loads its own copy of the model and takes every
Nth probe, so three 3090s run three streams.

    .venv/bin/python eval_probes.py --gpus 0,1,2 --out runs/base
    .venv/bin/python eval_probes.py --no-thinking --out runs/base_nothink
"""

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DEFAULT = "Qwen/Qwen3.5-9B"

# The benchmark's own wording opens with "a large red X is painted on the
# floor, straight ahead of the camera". That describes the first frame, but a
# model can satisfy it by repeating it — and in non-thinking mode this one
# does, answering 000 on 92 of 100 probes while quoting the phrase back. The
# neutral variant states the same setup without ever placing the X ahead of
# the camera, so echoing the prompt is no longer a way to answer it. It also
# says outright that the X is off-screen at the end, because the other half of
# that failure was the model reporting what it could see rather than where the
# X had got to. Any gap between the two questions is prompt artefact rather
# than spatial ability, so `original` stays available to measure it.
NEUTRAL_QUESTION = (
    "This is a first-person video of someone walking through corridors. A "
    "large red X is painted on the floor at the spot where the walk begins; "
    "it is visible in the opening seconds, and the walker then leaves it "
    "behind and keeps going. By the final frame the X is out of shot — it is "
    "somewhere off camera, and you have to work out where from the route that "
    "was walked. Freeze on the final frame and treat the direction the camera "
    "is facing at that moment as north (000 degrees). What is the compass "
    "bearing from the camera's final position back to the red X? Answer in "
    "degrees clockwise from that forward direction, 0-359: 090 is directly to "
    "the right, 180 directly behind, 270 directly to the left.")

# The answers are real angles, not a four-way choice; without this the model
# snaps to 000/090/180/270 and its error is dominated by the rounding.
ANSWER_FORMAT = (
    "\n\nAny integer from 0 to 359 is a valid answer — the X is usually not at "
    "an exact multiple of 90, so give your best estimate rather than rounding "
    "to the nearest quarter turn.\n"
    "End your reply with exactly one line, and nothing after it:\n"
    "ANSWER: <degrees>"
)

# Optional. Names the quantities the answer is made of without giving any of
# them away, for separating "cannot do the geometry" from "did not think to".
SCAFFOLD = (
    "\n\nWork through it in this order before answering:\n"
    "1. How long, in seconds, was each straight stretch of the walk?\n"
    "2. At each corner, did the walker turn left or right?\n"
    "3. Starting from the X and facing the walker's initial direction, follow "
    "those stretches and turns to the final position, and note which way the "
    "camera ends up facing.\n"
    "4. From there, which direction is the X in, relative to that final "
    "facing?"
)


def circ_err(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def parse_bearing(text):
    """(bearing, how_it_was_found). `how` is recorded per probe because the
    fallbacks are not equally trustworthy: replies quote frame timestamps like
    "00:24", so scavenging the last number in the text can manufacture an
    answer out of a reply that never gave one. A run whose numbers lean on
    `last_number` is a parsing artefact, not a result."""
    m = re.findall(r"ANSWER\s*[:\-]?\s*(-?\d+(?:\.\d+)?)", text, re.I)
    how = "answer_line"
    if not m:
        # a bare "123 degrees" still counts as an attempt
        m = re.findall(r"(-?\d+(?:\.\d+)?)\s*(?:°|degrees?\b)", text, re.I)
        how = "degrees"
    if not m:
        m = re.findall(r"(-?\d+(?:\.\d+)?)", text)
        how = "last_number"
    if not m:
        return None, "none"
    try:
        return float(m[-1]) % 360.0, how
    except ValueError:
        return None, "none"


def load_frames(path, fps, max_frames, size):
    """Uniformly-spaced frames as (T, H, W, C) uint8 plus their VideoMetadata.

    Sampling is uniform over the whole clip so equal wall-clock intervals map
    to equal frame counts — the walk's leg lengths live in those durations, so
    a non-uniform sample would quietly destroy the thing being measured.

    The metadata is not optional. Qwen3-VL writes a `<t seconds>` tag before
    every frame, computed from `fps` and `frames_indices`; hand it pre-sampled
    frames with no metadata and it assumes 24 fps, so a 77-second walk is
    labelled as a 21-second one and every leg duration the model reads off is
    wrong by the same factor."""
    import numpy as np
    from decord import VideoReader, cpu
    from transformers.video_utils import VideoMetadata

    vr = VideoReader(str(path), ctx=cpu(0), num_threads=2)
    n_total = len(vr)
    native_fps = float(vr.get_avg_fps()) or 30.0
    duration = n_total / native_fps

    n_want = min(max_frames, max(4, int(round(duration * fps))), n_total)
    n_want -= n_want % 2                  # temporal_patch_size = 2
    idx = np.linspace(0, n_total - 1, n_want).round().astype(int)
    frames = vr.get_batch(idx).asnumpy()          # (T, H, W, 3) uint8

    h, w = frames.shape[1:3]
    if size:
        import cv2
        w, h = size
        frames = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
                           for f in frames])
    meta = VideoMetadata(total_num_frames=n_total, fps=native_fps,
                         width=w, height=h, duration=duration,
                         video_backend="decord", frames_indices=idx.tolist())
    return frames, meta, duration


def build_worker(args, gpu):
    """Return a fn(plan) -> record, with the model resident on `gpu`."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("HF_HUB_CACHE", args.hf_cache)
    os.environ.pop("HUGGINGFACE_HUB_CACHE", None)

    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model)
    load_kw = dict(dtype=torch.bfloat16, device_map="cuda:0",
                   attn_implementation=args.attn)
    if args.load_4bit:
        # matches the trainer's quantization, for when an adapter is being
        # scored on exactly the base it was fitted to
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["visual", "lm_head"])
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model,
                                                            **load_kw)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload() if not args.load_4bit else model
    model.eval()

    video_dir = ROOT / args.videos
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=True,
                      temperature=1.0, top_p=0.95, top_k=20)

    def run(plan):
        vid = video_dir / f"{plan['id']:03d}.mp4"
        frames, meta, duration = load_frames(
            vid, args.fps, args.max_frames, (args.width, args.height))

        question = (plan["_question"]
                    + (SCAFFOLD if args.scaffold else "")
                    + (ANSWER_FORMAT if args.format_hint else ""))
        messages = [{"role": "user", "content": [
            {"type": "video"}, {"type": "text", "text": question}]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=args.thinking)

        inputs = processor(text=[text], videos=[frames], video_metadata=[meta],
                           do_sample_frames=False, return_tensors="pt")
        inputs = inputs.to(model.device)
        n_in = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        n_gen = int(out.shape[1] - n_in)
        reply = processor.tokenizer.decode(
            out[0][n_in:], skip_special_tokens=True)

        # Budget forcing. This model reliably talks past any thinking budget
        # on this task, and a reply cut off mid-thought carries no answer —
        # scoring only the ones that happened to finish would quietly select
        # for the easy probes. So when the budget runs out, close the thought
        # and make it commit, and record that it had to be forced.
        forced = args.thinking and "</think>" not in reply
        if forced:
            close = processor.tokenizer(
                "\n</think>\n\nANSWER:", add_special_tokens=False,
                return_tensors="pt").input_ids.to(out.device)
            with torch.inference_mode():
                out2 = model.generate(
                    input_ids=torch.cat([out, close], dim=1),
                    attention_mask=torch.ones(
                        (1, out.shape[1] + close.shape[1]),
                        dtype=torch.long, device=out.device),
                    max_new_tokens=12, do_sample=False)
            tail = processor.tokenizer.decode(
                out2[0][out.shape[1]:], skip_special_tokens=True)
            reply = reply + tail
            n_gen += int(out2.shape[1] - out.shape[1] - close.shape[1])

        body = reply.split("</think>")[-1] if "</think>" in reply else reply
        ans, how = parse_bearing(body)
        gt = plan["bearing_gt_deg"]
        return {
            "id": plan["id"], "level": plan.get("level"), "gt": gt,
            "answer": ans, "parsed_by": how,
            "err": None if ans is None else round(circ_err(ans, gt), 2),
            "forced": bool(forced),
            "n_frames_fed": len(frames), "video_s": round(duration, 1),
            "prompt_tokens": n_in, "gen_tokens": n_gen,
            "secs": round(time.time() - t0, 1),
            "reply": reply,
        }

    return run


def worker_main(args, gpu, shard, n_shards, plans, out_path):
    run = build_worker(args, gpu)
    mine = plans[shard::n_shards]
    records = []
    if out_path.exists():                       # resume
        records = [json.loads(l) for l in out_path.read_text().splitlines() if l]
        done = {r["id"] for r in records}
        mine = [p for p in mine if p["id"] not in done]

    with open(out_path, "a") as f:
        for i, plan in enumerate(mine, 1):
            rec = run(plan)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[gpu{gpu} {i}/{len(mine)}] id={rec['id']:03d} "
                  f"gt={rec['gt']:6.1f} ans={rec['answer']} "
                  f"err={rec['err']} ({rec['secs']}s, "
                  f"{rec['prompt_tokens']}tok in / {rec['gen_tokens']} out)",
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--adapter", default=None,
                    help="LoRA checkpoint to score instead of the bare base")
    ap.add_argument("--load-4bit", action="store_true",
                    help="quantize the base as the trainer does")
    ap.add_argument("--plans", default="probes/bearing_probes.json")
    ap.add_argument("--videos", default="out/probes")
    ap.add_argument("--out", default="runs/base")
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=64)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=352)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--hf-cache", default="/mnt/data0/ameen/hf_cache/hub")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None, help="comma-separated subset")
    # neutral by default: the benchmark's own wording is answerable by echoing
    # it, so `original` measures the artefact, not the model
    ap.add_argument("--question", choices=("original", "neutral"),
                    default="neutral")
    ap.add_argument("--scaffold", action="store_true",
                    help="append the four-step recipe (names the quantities, "
                         "gives none of them away)")
    ap.add_argument("--no-thinking", dest="thinking", action="store_false")
    ap.add_argument("--no-format-hint", dest="format_hint",
                    action="store_false")
    args = ap.parse_args()

    doc = json.loads((ROOT / args.plans).read_text())
    plans = doc["plans"]
    # plan files built by make_trainset.py carry no question of their own
    question = (NEUTRAL_QUESTION if args.question == "neutral"
                else doc.get("question") or json.loads(
                    (ROOT / "probes/bearing_probes.json").read_text())["question"])
    for p in plans:
        p["_question"] = question
    if args.ids:
        keep = set()
        for part in args.ids.split(","):
            if "-" in part.strip("-"):
                a, b = part.split("-")
                keep |= set(range(int(a), int(b) + 1))
            else:
                keep.add(int(part))
        plans = [p for p in plans if p["id"] in keep]
    if args.limit:
        plans = plans[: args.limit]

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    gpus = [int(g) for g in args.gpus.split(",")]
    print(f"{len(plans)} probes over GPUs {gpus}, thinking={args.thinking}, "
          f"{args.fps} fps <= {args.max_frames} frames at "
          f"{args.width}x{args.height}")

    if len(gpus) == 1:
        worker_main(args, gpus[0], 0, 1, plans, out_dir / "raw_0.jsonl")
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=worker_main,
                             args=(args, g, s, len(gpus), plans,
                                   out_dir / f"raw_{s}.jsonl"))
                 for s, g in enumerate(gpus)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

    recs = []
    for f in sorted(out_dir.glob("raw_*.jsonl")):
        recs += [json.loads(l) for l in f.read_text().splitlines() if l]
    recs.sort(key=lambda r: r["id"])
    (out_dir / "results.json").write_text(json.dumps(recs, indent=2))

    summarize(recs, plans)
    print(f"-> {out_dir}")


def summarize(recs, plans):
    """Overall and per-level error, each against the baseline that level
    deserves: a constant answer beats 90 degrees wherever the answers cluster,
    and on a level like L0 it is perfect, so `mean err` alone would read as
    understanding."""
    import collections

    def block(rs, label, const):
        errs = [r["err"] for r in rs if r["err"] is not None]
        if not errs:
            print(f"{label:16s} {len(rs):>3}  no parsable answers")
            return
        errs_sorted = sorted(errs)
        base = f"{const:5.1f}" if const is not None else "   90"
        print(f"{label:16s} {len(rs):>3}  mean {sum(errs) / len(errs):5.1f}  "
              f"median {errs_sorted[len(errs) // 2]:5.1f}  "
              f"beat-constant {base}  "
              f"unparsed {sum(r['err'] is None for r in rs)}")

    const_by_level = {p.get("level"): p.get("const_baseline_deg")
                      for p in plans}
    print()
    block(recs, "all", None)
    levels = [l for l in dict.fromkeys(r.get("level") for r in recs)
              if l is not None]
    for lv in sorted(levels):
        block([r for r in recs if r.get("level") == lv], lv,
              const_by_level.get(lv))

    paths = collections.Counter(r.get("parsed_by") for r in recs)
    if set(paths) - {"answer_line"}:
        print("  answers parsed by: "
              + ", ".join(f"{k} {v}" for k, v in paths.most_common()))
    forced = sum(bool(r.get("forced")) for r in recs)
    if forced:
        print(f"  {forced} replies ran past the thinking budget and were "
              f"forced to commit")
    # a model that ignores the video shows up here, not in the mean
    top = collections.Counter(r["answer"] for r in recs).most_common(1)
    if top and top[0][1] > len(recs) * 0.25:
        print(f"  WARNING: {top[0][1]}/{len(recs)} answers are all "
              f"{top[0][0]:g} — the model is not reading the video")


if __name__ == "__main__":
    main()
