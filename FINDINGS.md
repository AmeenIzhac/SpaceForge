# Can Qwen3.5-9B tell where it came from? — findings

A working log of 8–11 Aug 2026: benchmarking the base model on the
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

## 9. The leakage incident (and the clean result that survived it)

A stage-1 checkpoint appeared to *solve* L0–L2 (0.0°/0.3°/0.3°). An audit
before reporting found **21 of the 40 easy-ladder test routes had been emitted
as training data**: the generator's "repeat a route when draws run dry"
fallback kept the holdout and the already-used-routes in one set, so a narrow
tier that exhausted its geometries started handing back benchmark routes. The
deeper cause is structural — at 0–2 turns over 4–10 steps only ~40 distinct
routes exist, so the easy rungs cannot both be trained on and held out.

Fixes: the holdout is now un-breachable (`forbidden` is checked before the
repeat fallback), and test sets since use walk lengths the training set does
not contain. Clean re-test on 40 verified-disjoint routes, **11–16 steps —
longer than any training walk (4–10)**:

| | base | stage-1 tuned |
|---|---|---|
| overall (40 routes) | 64.9° (22/40 answers are "180") | **3.0°, median 0.7°** |
| 1 turn (17) | — | 1.0°, 17/17 within 5° |
| 2 turns (17) | — | 6.0°, 13/17 within 5° |

Generalises across routes *and* lengths at 1–2 turns. Every number in §5's
table that involved L0–L2 bearing error should be treated as contaminated;
the perception rows (seq/count) were re-confirmed on clean sets.

## 10. Shortcut audit ("is it cheating?")

"Measure the legs, do the trigonometry" is the *intended* solution; the
illegitimate shortcuts were tested one by one on checkpoint `stage1_v2/360`:

1. **Equal-leg assumption** (trained legs were near-equal, ≤2.6:1): 30 unseen
   walks with leg ratios 3.5–7.5. Assuming equal legs scores 35.5° here; the
   model scored **3.0° (median 0.5°)**. It genuinely reads leg durations, and
   extrapolates to ratios it never saw.
2. **Novel question** — "how many seconds of walking straight back to the X?"
   (the *magnitude* of the displacement vector whose *angle* it was trained to
   report; never asked in training): directional but rough — median 30%
   relative error, 19/40 within 25%, r≈0.33. The angle is solid; the full
   vector transfers only partially to new question forms.
3. **Frame-sampling aliasing** (the §5 hypothesis for 3-turn undercounting):
   **refuted.** On short-leg walks where corners fill 20–32% of the clip, 1–2
   turn bearings hold (1.8°/3.5°) but 3-turn walks still fail (58.8° vs 49.1°
   constant). The 3-turn failure is capability, not sampling.

## 11. Three turns: fitting is not learning

Stage 2 trained a fresh adapter on 3 360 examples (0–2-turn mix + 300 new
3–4-turn walks, 17–36 s, all traces verified, holdout enforced). Training loss
reached 0.02 — the model reproduces its 3-turn training traces — and its
**final** checkpoint on unseen 3-turn walks:

| 3-turn test (unseen routes) | stage-1 | stage-2 final | constant |
|---|---|---|---|
| lengths inside training range (L3, 27–39 s) | 83.6° | **50.2°** | 80.0° |
| shorter walks (U3, 12–22 s) | 58.8° | 62.5° | 49.1° |

First genuine 3-turn ability (30° under the constant), but **brittle to walk
length**. The failure mode is specific: asked "which way did each corner go?"
the stage-2 model answers the exact sequence **15/15** on the same unseen
short walks whose bearings it gets wrong; inside its bearing working it then
*narrates a different walk* ("the walk turns 2 times…"). Perception and
procedure both present; the binding breaks under length shift at 3 turns.

Stage 2b — continue-training from the stage-2 adapter (`--init-adapter`,
fresh optimizer; Trainer resume crashes reloading 8-bit optimizer state, §7)
with 120 additional *short* 3-turn walks covering the failing regime:

| test | stage-2 | stage-2b | constant |
|---|---|---|---|
| 1–2 turns (40) | 6.6° | 7.1° (median 0.4°) | ~37–56° |
| 3-turn in-length (L3) | 50.2° | **24.2°** (median 1.2°) | 80.0° |
| 3-turn short (U3) | 62.5° | **32.7°** (median 0.4°) | 49.1° |

Every rung 0–3 turns beat its constant on verified-clean held-out routes; the
base model beat none. Medians say most 3-turn walks are solved exactly (10/15
short walks within 5°); means are dragged by a few 150°+ blowups. Autopsy of
every blowup: in six of seven the model narrates the correct turn sequence and
then composes it **mirrored** — the answer is exactly 360°−truth — plus one
answered relative to the initial heading. Discrete convention flips, not
diffuse error (and themselves evidence of real computation: a guesser does not
produce the exact mirror of a correctly-perceived walk).

## 12. Working notes

- Evals and training must share fps: frames carry absolute `<t seconds>` tags
  (constant-rate sampling until `max_frames` caps it), so a checkpoint is
  scored on the temporal density it trained at.
- Walk `--speed` in `render_probes.py` re-times the same route: identical
  ground truth, 2.75× shorter videos, 7× faster renders — but it shifts every
  timestamp, so train/eval speed must match too.
- Background-shell trap that cost four runs: gating on `pgrep -f "<string>"`
  from a chain whose own command line contains that string deadlocks (or
  self-kills). Poll a captured PID instead.
- Root disk at 100% truncates files mid-`write_text` (one .py was cut mid-
  line); optimizer states are ~700 MB per checkpoint and are dead weight when
  Trainer-resume is never used — prune them.

---

# Part II — no reasoning anywhere (10 Aug onward)

The objective was restated (Ameen, 10 Aug): the model should **authentically
learn spatial reasoning in its weights** — genuinely represent where things
are — not externalize the task into written arithmetic it already knows. The
worked-trace models above are the wrong *kind* of success: their replies are
timestamp differences, per-leg vectors and an atan2. Part II reruns the
program with no reasoning in any channel: hidden thinking off **and**
supervision a bare `ANSWER: N` — six tokens, nothing else. All evals greedy
(sampling noise lands directly on answer digits once no trace pins them), all
on verified-disjoint held-out routes.

## 13. Stage A: bare answers work

Training data: 1 048 bare-answer examples over 262 walks, 0–1 turns only.

| checkpoint | 0–1-turn err (n=23) | 2-turn err, never trained (n=17) |
|---|---|---|
| base | 53.4° | 80.4° |
| 40 (~10 min) | 10.9° | 46.7° |
| 80–200 | ≈9° plateau | 34–51° |
| 360 (end) | **3.1°** | 46.4° |
| *references* | *constant 30.0° · trace model 0.7°* | *constant 55.6° · trace model 6.0°* |

Three findings: (1) the mapping can live in the weights — 3.1° with a bare
number reply, within 2.4° of the write-out-the-arithmetic model; (2)
two-phase learning — a 120-step plateau that looks exactly like convergence,
then a second drop; (3) composition is not free — 2-turn walks beat the
constant on pure transfer but plateau at ≈46°.

## 14. The `<t seconds>` crutch, removed

The processor writes a literal `<12.3 seconds>` before every frame pair, so
leg durations were readable as *text* even with bare answers. `notimestamps.py`
removes the tag at the processor level (source-patched, guarded, per-instance;
verified 22 tags → 0, vision payload identical), wired through trainer and
evals as `--no-timestamps`. Stage A rerun identically without tags:

- **Base model without tags: 150.7°** — beyond the 90° of random guessing
  (answers anti-correlate with truth) and far worse than its 53.4° with tags.
  The *base* model was leaning on the printed numbers.
- **Trained without tags: 4.6°** (vs 3.1° with) — the crutch is worth ~1.5°
  and nothing more. Duration is learnable from pixel evidence alone (frames
  are uniformly spaced, so elapsed time is visible as frame count).

Everything after this point is timestamp-free, bare-answer, thinking-off.

## 15. Overnight curriculum: 0–2, then 3–4 turns

Each stage warm-starts from the previous best (`--init-adapter`) and is
scored per-checkpoint by a rolling greedy monitor on held-out sets.

**Stage AB** (+2-turn data, 2 160 rows): 2-turn error **40.1° → 13.3°**,
0–1-turn held at 4.0°. The stage-A transfer plateau broke as soon as 2-turn
examples entered training.

**Stage ABC** (+3–4-turn data, 4 480 rows, init AB-final):

| checkpoint | 3-turn short U3 | 3-turn long L3 | 2-turn |
|---|---|---|---|
| 60 | 18.4° | 53.6° | — |
| 360 | 3.1° | 30.0° | 11.6° |
| 600 (best) | **2.2°** | **19.0°** | **9.2°** |
| *constant* | *49.1°* | *80.0°* | *55.6°* |

Three turns, no timestamps, no reasoning text: 2.2° on short walks, and the
2-turn score kept improving under 3-turn data. The Part-I length-brittleness
did not reappear (short and long both under their constants), though long
3-turn walks (54–78 frames) remain the weakest rung at 19–20°.

## 16. Anti-cheat battery (objective: transferable, not memorized)

On the AB-final checkpoint, before ABC:

- **Mirror test** (new): every frame flipped left-right, scored against the
  mirrored bearing. A model reciting route priors fails at ~2× chance; one
  reading pixels matches its unmirrored score. Result: **8.3° mirrored** vs
  ~8.0° unmirrored blend — left/right is visually grounded, not memorized.
- **Unequal legs, no timestamps**: 14.9° (median 9.5°) on ratios 3.5–7.5×
  vs 35.5° for the equal-leg assumption — it measures durations from pixels.

## 17. Stage D and the full-ladder result

Stage D added 160 five/six-turn walks (init ABC-600). Final battery, every
row a verified-unseen holdout, greedy, no timestamps, bare answers
(`ckpt/d_nots/checkpoint-600`; chart: `demo_vids/overnight_ladder.png`):

| rung | final model | best constant | base |
|---|---|---|---|
| 0–1 turn | 3.5° | 30.0° | 150.7° |
| 2 turns | 8.8° | 55.6° | 110.1° |
| 3 turns, short | **1.7°** | 49.1° | — |
| 3 turns, long | 23.2° | 80.0° | — |
| 4 turns | 16.6° | 45.0° | 130.0° |
| 5–6 turns | 20.6° | 55.5° | 115.7° |

Every rung far under its constant; the base model is *anti-correlated*
everywhere (worse than the 90° of random guessing). Notable curriculum
effects: the 2-turn score kept improving as harder data arrived
(40.1 → 13.3 → 9.2 → 8.8), and a mid-stage-D dip on 3-turn walks (2.2 → 5.8)
recovered to an all-time best 1.7° by the end — interference from new
difficulty is transient at these scales.

## 18. Authenticity battery — what is and is not real

Run on the final model (base references where they change the reading):

**Pass — the in-domain skill is genuinely visual and metric:**
- **Mirror test**: every frame flipped left-right, scored against the
  mirrored bearing: **5.5°** (its unmirrored blend ≈5–6°). Left/right comes
  from the pixels, not memorized route priors.
- **Unequal legs**: 12.2° on ratios 3.5–7.5× (equal-leg shortcut scores
  35.5°) — leg durations are measured, without timestamp text, on ratios
  never trained.
- Route and length extrapolation hold at every rung (all test sets are
  disjoint routes; several use lengths outside the training range).

**Fail — the skill is question-shaped and domain-bound:**
- **Mid-walk question** ("bearing just before the final corner"): 53.2° on
  1–2-turn walks and 73.5° on 3-turn — hugging the answer-the-END-anyway
  baselines (46.4°/72.1°) and far off the constants (18.9°/53.1°). It
  answers the trained question regardless of the temporal qualifier; there
  is no queryable running state yet. (Base is worse still: 124°.)
- **New-domain transfer** (Three.js open plain, real sky/sun/grass, five
  objects, rotation decoupled from any corridor): overall 80.9° ≈ the 81°
  constant. The failure is *degenerate on both sides*: the tuned model
  answers 180 to 39/84 questions (corridor prior: things are behind); the
  base answers 000 to 75/84 (things are ahead) — which incidentally scores
  7.9° on objects visible at the end and must not be mistaken for
  perception. Zero-shot transfer to a new visual world: absent, in both
  models.

Objective-2 verdict: not cheating *within* the world it was taught — the
mirror, unequal-leg and extrapolation tests close the shortcut routes — but
the ability does not yet leave that world or that question form on its own.
The obvious next levers: train across visual domains and question variants
(the open-plain generator and mid-walk probes are now standing infrastructure
for exactly that), and only then judge in-weights generality.

## 19. Infrastructure added for Part II

- `notimestamps.py` — guarded processor patch removing `<t seconds>` tags.
- `--greedy`, `--hflip`, `--no-timestamps`, per-plan `question`/`video`
  fields in `eval_probes.py`; `--no-timestamps` in `diagnose.py`.
- `--init-adapter` staged training (Trainer resume stays broken; this
  sidesteps it) and an optimizer-state pruner (700 MB per checkpoint of
  dead weight once resume is off the table).
- `make_midpoint_probes.py`, `plane3_make.py` + `photoreal/plane3.js` +
  `photoreal/render_plane3.mjs` (headless Three.js on mac *and* linux;
  platform switch documented in the file, node lives in
  `/mnt/data0/ameen/tools`), `make_demo_vids.py` (first-person + top-down
  map + verdict banner).
- Demo videos: `demo_vids/map_{1,2,3,4}turn.mp4`, `map_6turn_{hit,miss}.mp4`.
