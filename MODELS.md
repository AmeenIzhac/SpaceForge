# Model log

Checkpoints are large and cheap to recreate; results are not. This file is the
durable record: what each model was, how it scored, and the exact commands to
rebuild it. **Delete adapters only when asked** — this log is what makes that
safe.

Every model here is a QLoRA adapter on `Qwen/Qwen3.5-9B` (4-bit NF4 base,
LoRA on every linear in the text stack, vision tower frozen). Unless noted:
thinking off, no `<t seconds>` timestamps, bare `ANSWER: N` supervision, greedy
decoding at eval. Errors are mean absolute circular error in degrees on
held-out routes/scenes; **"const" is the best single constant answer for that
slice** and the only honest yardstick.

---

## corridor-v1 — worked-trace models (superseded)

Trained on corridor walks with a written arithmetic trace in the reply. Reached
3.0° on 1–2-turn held-out routes but by *externalising* the task into text,
which is the failure mode the project set out to avoid. Full account in
`FINDINGS.md` §5–§11. Adapters deleted; not worth rebuilding.

## corridor-nots — bare-answer corridor curriculum ⭐ best corridor model

`ckpt/d_nots/checkpoint-600` — **deleted 11 Aug 2026, reproducible below.**

Staged: A (0–1 turns) → AB (+2) → ABC (+3–4) → D (+5–6), each warm-started
from the last. Final scores, all verified-disjoint held-out routes:

| rung | model | const | base |
|---|---|---|---|
| 0–1 turn | 3.5° | 30.0° | 150.7° |
| 2 turns | 8.8° | 55.6° | 110.1° |
| 3 turns, short | 1.7° | 49.1° | — |
| 3 turns, long | 23.2° | 80.0° | — |
| 4 turns | 16.6° | 45.0° | 130.0° |
| 5–6 turns | 20.6° | 55.5° | 115.7° |
| mirror test | 5.5° | — | — |
| unequal legs 3.5–7.5× | 12.2° | 35.5° (equal-leg shortcut) | — |

Failures: mid-walk question 53–74° (answers the end-of-walk question anyway);
zero-shot to the open plain ≈ constant, with a "target is behind me" prior that
puts 153° error on objects in plain sight.

<details><summary>reproduce</summary>

```bash
# plans (holdout is un-breachable; see make_trainset.py --holdout)
.venv/bin/python make_trainset.py --n 240 --turns 0,1,2 --steps 4,10 --bands 0,1 \
    --tag curriculum_a --seed 4242 --seed-base 500000 --out probes/train_curriculum_a.json
.venv/bin/python make_trainset.py --n 300 --turns 1,2  --steps 4,10 --bands 0,1 \
    --tag curriculum_b --seed 8899 --seed-base 700000 --out probes/train_curriculum_b.json
.venv/bin/python make_trainset.py --n 300 --turns 3,4  --steps 6,14 --bands 0,1,2 \
    --tag curriculum_c --seed 5150 --seed-base 900000 --out probes/train_curriculum_c.json
.venv/bin/python make_trainset.py --n 120 --turns 3    --steps 4,9  --bands 0,1 \
    --tag c_short --seed 9091 --seed-base 950000 --out probes/train_c_short.json
.venv/bin/python make_trainset.py --n 160 --turns 5,6  --steps 8,22 --bands 0,1,2 \
    --tag turns56 --seed 6001 --seed-base 1100000 --out probes/train_turns56.json
# render each: render_probes.py --plans <plans> --out out/<dir> --workers 1
# rows (bare answers)
.venv/bin/python sft_data.py --plans <plans> --videos out/<dir> --no-trace \
    --tasks count,seq,coarse,bearing --aux-frac 1.0 --bearing-frac 1.0 --out data/<x>.jsonl
# four stages, each --init-adapter the previous best
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1 \
 .venv/bin/python -m torch.distributed.run --nproc_per_node=2 train_lora.py \
   --data data/nt_stageD.jsonl --out ckpt/d_nots --epochs 1.2 --accum 4 --lr 5e-5 \
   --init-adapter <prev> --fps 2.0 --max-frames 56 --no-thinking --no-timestamps
```
Test sets: `probes/test_clean.json`, `micro_u3`, `easy_l3`, `test_t4`,
`test_t56` (all still on disk).
</details>

## plane-s1 / plane-s2 — open-plane, five question types

`ckpt2/pl_s1/checkpoint-375`, `ckpt2/pl_s2/checkpoint-340` — **kept.**

Trained on Three.js open-plain scenes (`plane_gen.py` + `plane_tasks.py`),
5 question types; three types held out entirely as transfer tests.

| question type | const | s1 (P0+P1) | s2 (P0–P3) |
|---|---|---|---|
| bearing_final *(trained)* | 85.3° | 23.0° | **20.8°** |
| bearing_start *(trained)* | 46.5° | 13.5° | **9.2°** |
| bearing_mid *(never trained)* | 54.5° | 34.5° | **27.8°** |
| back_to_start *(never trained)* | 41.1° | 71.6° | 84.3° ✗ |
| object→object *(never trained)* | 84.1° | 61.3° | 66.1° |

Path extrapolation (s1, trained on ≤1 turn): P2 34.0°, P3 43.1°, **P4 47.0°**
against constants of 79.8/85.4/88.3. Held-out colours score the same as
trained colours (17.2 vs 16.5). Mirror test 19.5° ≈ unmirrored 20.8°.
Base model: 17 distinct answers, 201/684 unparseable, worse than constant.

<details><summary>reproduce</summary>

```bash
.venv/bin/python plane_gen.py --split train --n 400 --levels P0:1,P1:3,P2:3,P3:2 \
    --seed 2211 --seed0 0    --out probes/plane_train.json
.venv/bin/python plane_gen.py --split test  --n 120 --levels P0:1,P1:2,P2:2,P3:2,P4:2 \
    --seed 8833 --seed0 5000 --out probes/plane_test.json
node photoreal/render_plane.mjs probes/plane_train.json out/plane_train 8
.venv/bin/python plane_tasks.py --scenes probes/plane_train.json --split train \
    --videos out/plane_train --out data/plane_train.jsonl
# stage 1 = P0+P1 rows only, 3 epochs, lr 1e-4, from base
# stage 2 = all rows, 1.2 epochs, lr 6e-5, --init-adapter <stage1>
#   both: --no-thinking --no-timestamps --fps 4.0 --max-frames 64
```
</details>

## sphere-1turn — narrowed open-plane task ⭐ best model so far

`ckpt2/sph/checkpoint-612` — **kept.**

Deliberately narrow: green plain, procedural noise ground (no checkerboard —
a checker is a metric ruler painted on the floor), exactly one turn, ~18
objects per scene, and one question family only — the bearing to a named
sphere from the final or the starting frame. Every sphere in a scene has a
distinct colour so "the red sphere" names one thing; distractor shapes reuse
colours freely, so colour alone is not enough to identify it.

900 train / 150 test scenes (disjoint), 2518 bare-answer examples, 2 epochs.

| slice | n | const | tuned | base | mirror |
|---|---|---|---|---|---|
| all | 412 | 66.7° | **7.3°** (median 3.3°) | 64.4° (134 unparsed) | 8.1° |
| bearing_final | 262 | 81.9° | 8.2° | 82.9° | 9.2° |
| bearing_start | 150 | 40.0° | 5.7° | 40.1° | 6.3° |
| target visible at end | 141 | 36.4° | 4.3° | 38.9° | 4.9° |
| target off screen at end | 271 | 82.4° | 8.9° | 76.6° | 9.8° |

62% of answers within 5°, 90% within 15°. Mirror test 8.1° ≈ unmirrored 7.3°.

**Motion-template audit.** Training used one motion shape (straight, one eased
turn, straight, stop); only its size, direction, speed and durations varied. So
the same scenes were re-rendered with camera motions never trained, objects and
questions unchanged — if the model had fitted a recipe for that one shape,
these collapse toward the constant:

| motion at eval | trained | n | mean | median | that set's constant |
|---|---|---|---|---|---|
| same shape (control) | yes | 112 | 6.9° | 3.8° | 69.1° |
| two turns | no | 110 | 13.1° | 6.5° | 73.8° |
| no turn at all | no | 107 | 7.1° | 2.9° | 53.0° |
| continuous curved arc | no | 74 | 9.6° | 5.0° | 69.3° |
| stop mid-leg + speed changes | no | 110 | 8.7° | 3.9° | 66.6° |
| turn 3x faster (0.4 s) | no | 115 | 7.7° | 2.2° | 60.3° |
| backtrack (175° turn) | no | 118 | 26.6° | 10.7° | 87.5° |

Every unseen motion stays within a few degrees of the control, including the
two that break the template hardest: a continuous arc (no discrete turn event
to read off) and stop-go (no fixed time-to-distance mapping). `backtrack` is
the only real degradation, and it is geometrically nasty rather than a template
failure — after a 175° turn the target sits close and nearly behind, where a
small heading error swings the bearing a long way. Still 3x under its constant.
Conclusion: this is visual path integration, not motion-template matching.

**Corridor transfer** (never trained on corridors; target there is a painted
red X, not an object): **none.** Scored 45.0° but the set's best constant is
43.5° and the model answered 186–190° on all 40 probes — it fell back to a
"behind me" constant. Beating the base (132.2°) means nothing here because the
base is worse than useless on corridors; the constant is the only bar, and it
was not cleared.

<details><summary>reproduce</summary>

```bash
.venv/bin/python sphere_gen.py --split train --n 900 --seed 555  --seed0 0     --out probes/sph_train.json
.venv/bin/python sphere_gen.py --split test  --n 150 --seed 9090 --seed0 60000 --out probes/sph_test.json
# render with moderngl, 4 shards over 3 GPUs (~5.4 min for all 1050 scenes)
.venv/bin/python plane_gl.py --scenes <shard>.json --out out/sph_train --gpu <n>
# rows: sphere_gen.rows() per scene -> data/sph_train.jsonl / probes/sph_probe.json
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1 \
 .venv/bin/python -m torch.distributed.run --nproc_per_node=2 train_lora.py \
   --data data/sph_train.jsonl --out ckpt2/sph --epochs 2 --accum 4 --lr 1e-4 \
   --no-thinking --no-timestamps --fps 4.0 --max-frames 64
```
</details>

---

## Conventions

* Adapters live in `ckpt2/` (data disk). `ckpt/` is the old root-disk location.
* Optimizer state is ~700 MB per checkpoint and is never needed —
  `--init-adapter` restarts the optimizer, and Trainer resume is broken
  (bitsandbytes crashes reloading 8-bit state). Prune it.
* Before deleting anything, add its row here with scores and the commands.
