# Can Qwen3.5-9B tell where it came from? — findings

A working log of one session (8–9 Aug 2026): benchmarking the base model on the
bearing-to-start task, finding out *why* it failed, building training data
aimed at that cause, and fine-tuning. `README.md` documents the simulator
itself; this file is the experiment.

**Headline:** the base model scores at chance on every rung of the task, and the
reason is not spatial reasoning — **it cannot tell a left turn from a right
turn**. After ~20 minutes of LoRA fine-tuning on 540 generated walks with worked
solutions, left/right goes from 3/9 to 15/15 on one-turn walks, corner counting
from 35% to 78%, and one-turn bearings become genuinely correct (median error
0.4°). Two- and three-turn walks improve but still lose to a constant answer.

---

## 1. The task

A first-person video of someone walking through corridors. A red X is painted
on the floor where the walk starts; the walker leaves it behind and stops
somewhere else. Question: **at the final frame, what compass bearing is the X
at**, taking your own facing as 000? Answer 0–359, where 090 is right, 180 is
behind, 270 is left.

Answering needs path integration: the turn sequence alone is not enough,
because how long each straight ran decides how far each turn displaced you.

## 2. Why every score is quoted against a "constant baseline"

A model that ignores the video and always answers the same number scores
surprisingly well, because the answers cluster behind the walker. On a level
where the answer is *always* 180, always answering 180 is **perfect**.

So every level records `const_baseline_deg` — the error of the best single
constant answer for that level — and **only beating that counts as evidence of
anything**. Chance for a uniform guess is 90°, but 90° is the wrong yardstick
almost everywhere.

This is the single most important thing in this document. Several results that
look like successes are constants in disguise.

## 3. The difficulty ladder (`make_easy_probes.py`)

The 100-plan benchmark (`make_probes.py`) spans 1–8 turns and scored at chance,
which says "it failed" without saying where. So: four levels, 55 walks, each
adding exactly one demand.

| level | turns | walk | video | best constant | what it isolates |
|---|---|---|---|---|---|
| `L0_straight` | 0 | 4–10 steps | 13–27s | **0.0° @180** | reading the question at all |
| `L1_one_turn` | 1 | 4–8 steps | 13–22s | 28.6° @206 | one turn's direction |
| `L2_two_turns` | 2 | 6–12 steps | 17–32s | 47.0° @151 | composing two turns |
| `L3_three_turns` | 3 | 10–15 steps | 27–39s | 80.0° @166 | leg *durations* start to matter |

Walks are short enough that the 2 fps / 128-frame budget sees them whole, so a
miss cannot be blamed on frames the model never got.

## 4. Base model results

### 4.1 The prompt was contaminated

The benchmark's own wording opens with "a large red X is painted on the floor,
**straight ahead of the camera**". That describes the first frame — but the
model satisfied it by repeating it, answering **000 on 92 of 100 probes** while
quoting the phrase back.

`eval_probes.py` now defaults to a neutral wording that never places the X ahead
of the camera and states outright that it is off-screen by the end.
`--question original` still measures the artefact.

Fixing the prompt did not produce understanding — it swapped one degenerate
constant for another. The model went to answering **180 on 26 of 55 probes**.

### 4.2 Base scores on the ladder (thinking off, all 55)

| level | mean err | constant baseline | verdict |
|---|---|---|---|
| L0_straight | 30.5° | 0.0° | worse |
| L1_one_turn | 45.1° | 28.6° | worse |
| L2_two_turns | 80.6° | 47.0° | much worse |
| L3_three_turns | 81.7° | 80.0° | tied |

**No rung beats its constant.** Since L0 already has zero route difficulty,
making routes easier cannot go lower — route difficulty was never the binding
constraint.

### 4.3 The actual deficit (`diagnose.py`)

Asking about the component abilities separately, on the same videos:

| task | 0 turns | 1 turn | 2 turns | 3 turns |
|---|---|---|---|---|
| count corners | 3/3 | 2/3 | 4/9 | 0/3 |
| **which way (L/R)** | — | 3/9 | 1/3 | 0/3 |
| coarse 4-way | 4/4 | 3/3 | 0/3 | 5/9 |
| net heading change | 2/3 | 0/3 | 2/9 | 0/3 |

On one-turn walks the model answered **L eight times out of nine** while the
truth was 7×L, 2×R — **zero of the right turns correct**, and worse than always
saying "L" (7/9). On the 4-way question it answered **BEHIND on 19 of 19**; its
63% "accuracy" is just how often BEHIND happens to be true.

**The causal chain:** can't perceive turn direction → can't integrate a path →
180 is the best available constant → chance on every bearing rung.

### 4.4 The labels were verified before training on them

Optical flow across the corner on six one-turn probes: every labelled `R` shows
the scene sliding left, every `L` slides right — **6/6 match**. The ground truth
is correct; the model is genuinely wrong.

## 5. Training

- **Data** — 540 generated walks (`make_trainset.py --turns/--steps/--bands`),
  0/1/2 turns, short, routes held out from both the benchmark and the ladder.
  Four supervised tasks per walk (count, which-way, coarse, bearing) = **2160
  examples**.
- **Supervision** — a worked trace per example (`sft_data.py`), expressed only
  in quantities visible in the video (elapsed seconds, turn directions). Every
  trace's arithmetic is checked against the simulator before the example is
  emitted: **240/240 and 300/300 verified, worst deviation 0.0°**.
- **Method** — QLoRA, 4-bit NF4 base, LoRA on every linear in the text stack
  (86.5M trainable, 0.91%), vision tower frozen.

### Results at checkpoint-80 (~20 min, thinking off, all 55 probes)

**Perception:**

| | base | tuned |
|---|---|---|
| which-way (L/R), 1-turn walks | 3/9 (0 of 2 right turns) | **15/15** |
| which-way overall | 0.11 | **0.73** |
| count corners | 0.35 | **0.78** |
| coarse 4-way | 0.74 | 0.60 |

**Bearing:**

| level | base | tuned | constant baseline | beats constant? |
|---|---|---|---|---|
| L0_straight | 30.5° | 18.0° | 0.0° | no |
| L1_one_turn | 45.1° | **17.5°** (median 0.4°) | 28.6° | **yes** |
| L2_two_turns | 80.6° | 55.5° | 47.0° | no |
| L3_three_turns | 81.7° | 83.6° | 80.0° | no |
| overall | 62.1° | **46.0°** | — | |

A median of 0.4° on one-turn walks means it is landing the exact bearing, not
approximating a quadrant.

**Where it still fails, and why that's informative:** on three-turn walks the
model now gets every turn's *direction* right but **misses corners** —
predictions are prefixes of the truth (`LL` for `LLL`, `LR` for `LRL`), with
aligned-position direction accuracy 67/68. That is undercounting, not
confusion. Suspected cause: a corner takes ~1.6s and 2 fps with temporal
patching gives one view per ~2.4s, so corners can be sampled away entirely.
**Untested** — the 4 fps control had not run at time of writing.

Also note L0 and `coarse` got *worse*. Both are places where the base's score
came from a constant that happened to be right; a model that has stopped
answering a constant pays for it there.

## 6. Bugs found (each was corrupting results)

1. **Contaminated prompt** — §4.1; worth ~92% of answers on the base model.
2. **Scavenging parser** — `parse_bearing` fell back to "last number in the
   reply", and replies quote frame timestamps like `00:24`. "0 unparsed" was
   partly manufactured. It now records *how* each answer was found.
3. **`ANSWER: none` scored as unparsed** — the correct answer for a zero-turn
   walk was marked wrong on all 10, worth 0.19 of the which-way score.
4. **Train/eval frame-rate mismatch** — the trainer fed a flat 128 frames
   (`fps=1e9`, ≈6.4 fps on a 20s clip) while eval fed 2 fps: different frame
   count and different `<t seconds>` tags on the very timestamps the trace
   reasons from. Also the direct cause of the first OOM.
5. **`diagnose.py` had the same mismatch** — tuned checkpoints were being
   measured off-distribution. Both now take `--fps`.
6. **Thinking-template mismatch** — `enable_thinking=False` pre-fills an *empty*
   `<think></think>`, so a trace supervised inside that block is unreachable at
   generation time. Training with thinking on and scoring with it off would have
   wasted the run; the trace now moves into the visible reply.
7. **Training prompt ≠ eval prompt** — `sft_data.py` built bearing prompts from
   the original contaminated question. It now imports the eval's exact string.
8. **Global dedupe counter starved the easy rungs** — a 0-turn walk over 4–10
   steps has only seven distinct routes, and plentiful 2-turn draws kept
   resetting a shared counter, so curriculum sets came out nearly empty of easy
   examples. Now counted per turn-count.
9. **fp32 upcast on a 248k-entry vocabulary** — `prepare_model_for_kbit_training`
   upcasts non-quantized weights, and here the embedding table alone is 1.0B
   parameters: 2 GB of a 24 GB card. Casting back to bf16 (norms too, or the
   hidden states disagree mid-layer) took weights 11.46 → 7.66 GiB.
10. **Frozen vision tower retaining activations** — it carries no LoRA, so its
    activations were kept for a backward pass that cannot use them. Running it
    under `no_grad` was the difference between OOM and a 19.44 GiB peak.

## 7. Known-bad / incomplete

- **Resuming training from a checkpoint crashes**: bitsandbytes throws
  `Error invalid argument at line 580 in pythonInterface.cpp` when reloading
  `paged_adamw_8bit` state. A resume launched at 06:22 died instantly and was
  not noticed for ~11 hours, because the queued eval chain was waiting on a
  checkpoint that never appeared. **Run training in one uninterrupted go and
  evaluate saved checkpoints afterwards**, or switch the optimizer on resume.
- The 4 fps aliasing control (§5) never ran.
- `curriculum_c` (300 walks at 3–4 turns) is generated; 34 of 300 rendered.
- The GPU-0 shard of the base diagnostics OOM'd against another user's job, so
  §4.3 is a 73-job sample; §4.2 and §5 are complete 55-probe runs.

## 8. Reproducing

```bash
.venv/bin/python make_easy_probes.py                       # ladder plans
.venv/bin/python render_probes.py --plans probes/easy_probes.json --out out/easy

# base
.venv/bin/python eval_probes.py --plans probes/easy_probes.json --videos out/easy \
    --out runs/easy_nothink --no-thinking --max-new-tokens 512
.venv/bin/python diagnose.py --plans probes/easy_probes.json --videos out/easy \
    --out runs/easy_diag --task seq,count,coarse --fps 2.0 --n 55

# training data + fine-tune
.venv/bin/python make_trainset.py --n 240 --turns 0,1,2 --steps 4,10 --bands 0,1 \
    --tag curriculum_a --out probes/train_curriculum_a.json
.venv/bin/python render_probes.py --plans probes/train_curriculum_a.json --out out/train_a --workers 1
.venv/bin/python sft_data.py --plans probes/train_curriculum_a.json --videos out/train_a \
    --tasks count,seq,coarse,bearing --aux-frac 1.0 --out data/sft_a.jsonl
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1 \
  .venv/bin/python -m torch.distributed.run --nproc_per_node=2 train_lora.py \
    --data data/sft_stage1.jsonl --out ckpt/stage1_v2 --no-thinking --fps 2.0

# score a checkpoint the same way
.venv/bin/python eval_probes.py --plans probes/easy_probes.json --videos out/easy \
    --adapter ckpt/stage1_v2/checkpoint-N --out runs/ckptN --no-thinking
```

Rendering runs one worker per free GPU; a single worker coexists fine with
another job on the same card (~10s per short walk), three do not.

## 9. Next

1. The 4 fps control, before spending any training on L3 — if undercounting is
   aliasing, counting jumps with no retraining and the fix is cheap.
2. Finish `curriculum_c` and train on 3–4 turn walks if it isn't aliasing.
3. Watch whether L0 recovers as training continues, or whether losing the
   constant is a permanent cost.
4. Re-run the full 100-plan benchmark once the ladder is solved.
