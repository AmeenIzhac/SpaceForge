# spaceforge-sim

Purpose-built corridor simulator for the SpaceForge project (§4.1 of the
proposal): generates egocentric videos of walking through corridor
environments instantiated from abstract grid paths, plus the exact spatial
ground truth (poses, junctions, return-turn sequences, landmark positions)
that each video pairs with. No Isaac Lab, no training — just the data
factory's video end.

## Layout

| path | what it is |
|---|---|
| `generate.py` | **main script** — controlled corridor probes (below) |
| `make_probes.py` | builds the 100-plan bearing benchmark, no rendering |
| `probes/` | the benchmark: plans, ground truth, prompts (tracked) |
| `*.py` (root) | shared library: path gen, floor plans, textures, props, renderers |
| `dataset/generate_dataset.py` | backup pipeline: full floor plans + ground truth |
| `photoreal/` | backup pipeline: Three.js realism track (was `three/`) |
| `out/` | rendered video, one subdirectory per pipeline (git-ignored) |

## Main: controlled corridor probes (`generate.py`)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python generate.py          # -> out/main/0..4.mp4
```

Bare corridors that strip out every confound: no rooms, no doors, no
diversions, and no mid-walk yaws — the camera only turns where the path turns.
Each straight leg gets its own wall colour as an orientation cue.

- **Algorithmic routes.** Turn strings and leg lengths are drawn per sample,
  not hardcoded. A route is rejected if it revisits a lattice node (and, with
  `JUNCTIONS = True`, if a decoy stub would land on the route and become a
  real branch).
- **Variable straights.** Legs span `LEG_MIN..LEG_MAX` lattice steps (1–10),
  so the ratio between any two straights in a sample can be anything from 1:1
  to 1:10. Each sample targets a different band of that range (`RATIO_BANDS`),
  so the set exercises the whole span instead of clustering mid-range. One
  lattice step = `floorplan.S` = 3 world units ≈ 2.4 s of walking.
- **Start marker.** One red X, painted flat on the floor where the walk
  begins — `MARKER_AHEAD` units down the first leg so it sits in front of the
  camera rather than underneath it (at this eye height and FOV the floor only
  enters frame ~1.6 units out). It shrinks on a one-step first leg to stay
  clear of the first bend. Nothing marks the later corridors.
- **Junctions** (`JUNCTIONS`, default off). When on, every bend becomes a full
  crossroads: 1-step dead-end stubs carve the directions not taken, so each
  turn visibly could have gone left, right, or straight.

All knobs are in the CONFIG block at the top; runs are deterministic in
`MASTER_SEED`. The current set spans 1:1 → 10:1 straight ratios across five
samples, 3–5 turns each, ~39–60 s per video at 960×544/30fps.

## Benchmark: bearing to the start (`make_probes.py`)

```bash
.venv/bin/python make_probes.py       # -> probes/
```

100 route plans with ground truth, no video. The question each one poses:

> This is a first-person video of someone walking through corridors. At the
> very start of the walk a large red X is painted on the floor, straight ahead
> of the camera. Freeze on the final frame and treat the direction the camera
> is facing as north (000°). What is the compass bearing from the camera to
> the red X? Answer in degrees clockwise from that forward direction, 0–359:
> 090 is directly to the right, 180 directly behind, 270 directly to the left.

Scored as absolute circular error in degrees; a uniform guess averages 90°.

**Design.** 5 turn counts (1, 2, 3, 5, 8) × the 5 ratio bands × 4 replicates,
the replicates cycling turn pattern (alternating / same-direction spiral /
random / random). Within each cell the generator draws 24 candidate routes and
keeps the one landing in the emptiest 30° answer bin — left alone the answers
pile up behind the walker, and "always say roughly 180" scores far better than
it should. Balanced, that baseline's median error is 84°, i.e. near chance.
One-turn routes are the exception: after a single turn the X is always behind
you, so those cells cannot reach the forward half.

**Diagnostics.** Every plan carries features to slice results by, including
what the obvious shortcuts would score on it:

| field | what it tells you |
|---|---|
| `equal_leg_err_deg` | error from tracking the turns but ignoring how long each straight was — the headline axis (median 27°, up to 173°) |
| `cardinal_err_deg` | error from answering the nearest round number (0/90/180/270) |
| `behind_err_deg` | error from always answering 180 |
| `sensitivity_deg_per_unit` | how far the bearing swings per unit of position error — high when the walk ends near the X, so the answer is ill-conditioned |
| `crow_over_path`, `cum_turn_deg`, `net_turn_deg` | how much the route doubles back, and total vs. net rotation (a spiral winds past 360°) |
| `difficulty`, `tier` | rank-normalised blend of turn count, equal-leg error, cardinal error and sensitivity, cut into terciles |

**Files.** `bearing_probes.json` (everything, `bearing_gt_deg` is the answer),
`bearing_probes.csv` (flat table), `probes_prompts.jsonl` (id + question only —
the model-facing side, with no ground truth to leak).

Plans are renderable: `generate.make_sample(name, turns, legs, seed)` draws the
same walk `make_probes.py` measured. All 100 would come to ~74 min of video.

## Backup: full dataset pipeline (`dataset/generate_dataset.py`)

```bash
.venv/bin/python dataset/generate_dataset.py    # -> out/dataset/sample_XX/
```

The paper-faithful track, kept for dataset-scale generation with ground truth.

1. **`pathgen.py` — Algorithm 1, line for line.** A random self-avoiding
   outbound walk `P_out` on an N×N grid, a BFS return route `P_ret` that may
   not reuse the outbound corridor (only its endpoints), and `k` chord
   corridors routed through unused cells, which create the extra cycles and
   junctions where a returning agent must choose left/right.
2. **`floorplan.py` + `world.py` + `rooms.py` — instantiation.** The world is
   a building floor plan: rectangles (rooms) are packed tightly — every new
   rect must share edges with the block and may not create holes, so the
   union is a solid, irregular silhouette. The wall lines *between* the
   rectangles (plus the building outline) form a lattice graph; Algorithm 1
   runs on that graph, and the walked wall lines are carved into 1-unit-wide
   corridors (wall | corridor | wall replaces the shared wall). Rooms open
   onto the carved corridors through framed doorways — jambs, header, and
   usually an open door leaf — that the agent slows at and glances through
   while walking past; rooms whose walls carry no corridor stay sealed.
   Rooms are furnished by type (office / lounge / storage / bedroom: desks,
   chairs, sofas, beds, shelves with books, cabinets, rugs) with their own
   floor texture and ceiling light panels. Landmark objects are scattered
   along straight corridor stretches, offset toward a wall. Camera
   trajectory: corner-rounded polyline along the corridor centerlines,
   walking speed with ease-in/out, look-ahead heading, hesitation at
   junctions and doorways, and glances into rooms, down side corridors, and
   at landmarks.
3. **`textures.py` + `decor.py` — domain randomisation.** Each sample picks
   ONE mood-appropriate wall style (plaster, panels, brick, wainscot,
   wallpaper, blocks) rendered as subtle shade variants, so corridors look
   consistent; visual variety comes from decoration instead: doors, posters,
   framed pictures, notice boards, whiteboards, exit signs, hazard signs
   (unique procedural decal textures), plus wall-mounted radiators, glowing
   sconces, and fire alarms. Random floors (carpet/wood/tile/concrete) and
   four mood palettes (office / brick / lab / hotel) set colours, fog, and
   lighting.
4. **Renderers.** `gl_renderer.py` (default) is a headless moderngl/OpenGL
   engine: real 3D geometry, solid procedural prop meshes (`props.py`),
   per-pixel Blinn-Phong point lighting from emissive ceiling panels (plus
   glowing floor lamps), specular floors, contact shadows, exponential fog,
   4x MSAA — ~40+ fps per worker at 960×544 on the GPU. `raycaster.py`, the
   original vectorised numpy DDA raycaster with billboard sprites, remains as
   an automatic fallback when moderngl is unavailable (`RENDERER` in CONFIG).
5. **The factory.** Renders the outbound walk (start → stop) and the return
   walk as H.264 MP4s and writes all ground truth. Samples run in parallel
   worker processes (`WORKERS`), with grid size N, outbound length L, and
   chord count k randomised per sample. Movement is humanised: speed wobble,
   lateral sway, gaze wander, head-bob, hesitation at junctions, glances at
   landmarks and down side corridors, occasional full stops — all
   seed-deterministic.

### Outputs (`out/dataset/sample_XX/`)

| file | contents |
|---|---|
| `outbound.mp4` | egocentric walk along `P_out`, start (green on map) → stop (red) |
| `return.mp4` | in-place turn at the stop point, then walk along `P_ret` |
| `map.png` | Figure-4-style top-down: outbound blue, return + chords orange |
| `meta.json` | `P_out`, `P_ret`, chords, junction cells, **ground-truth turn sequence at each junction of the return route** (`return_decisions`), all bends, landmarks with world positions, camera intrinsics |
| `poses_*.csv` | per-frame camera pose `frame, t, x, y, yaw` |

Coordinates: cell `(x, y)` has its center at world `(2x+1.5, 2y+1.5)`;
y grows downward (south), `yaw = atan2(dy, dx)`. Turn labels are egocentric
and match the video (verified by the self-test in `pathgen.py`).

### Deviations from the paper's pseudocode (all configurable)

- The chord loop (Algorithm 1 line 16) is capped at `chord_tries` attempts so
  it cannot spin forever on a saturated grid; a sample may end up with fewer
  than `k` chords.
- `MIN_RETURN_LEN` (default 6) rejects episodes whose return route is
  trivially short (e.g. stop cell adjacent to start). Set it to 2 for the
  strictly faithful behaviour.
- Chords must contain at least one genuinely unused interior cell
  (`|γ| ≥ 3`), matching the paper's "through unused cells".
- `BIAS_RETURN_JUNCTION` (default on) turns the chord phase into a loop-maker
  aimed at the return route. Instead of uniform line-17 sampling it repeatedly
  connects one *bend* of `P_ret` to another bend (falling back to any loop
  cell), so each chord closes a rectangular cycle whose corners are genuine
  left/right decisions on the way back. It keeps adding chords past `k` until
  `MIN_RETURN_DECISIONS` bends carry a junction (capped at `k + 3` to bound
  clutter), rejects returns too straight to host them, and rejects walks that
  land fewer than two such decisions. Result: every sample has several cycles
  and 2–3 return-route left/right choices. Set it to False for the uniform
  line-17 behaviour.
- `STRAIGHT_BIAS` (deviation from line 8) makes the outbound walk keep its
  direction with the given probability, and `TURN_PENALTY` routes the
  return/chords through a turn-minimising Dijkstra instead of plain BFS.
  Together they produce simpler, loopier rectangular layouts with long
  straight corridors instead of zig-zags. Both at 0 reproduce the pseudocode
  exactly.

## Backup: photoreal track (`photoreal/`)

A separate Node pipeline: a Three.js scene driven frame-by-frame in headless
Chrome (puppeteer), screenshotted at 2x and piped into ffmpeg. PBR materials
from ambientCG, shadow mapping, ACES tone mapping, bloom. No ground truth is
emitted — this branch exists to see how far graphical realism can go.

```bash
(cd photoreal && npm install)
node photoreal/render_corridor.mjs    # -> out/photoreal/0.mp4
```

| entry point | scene | output |
|---|---|---|
| `render_corridor.mjs` | `corridor.js` — hotel-ish corridor, L-R-L, no rooms | `0.mp4` |
| `render_house.mjs` | `house.js` — single-storey bungalow, tour of every room | `1.mp4` |
| `render_house2.mjs` | `house2.js` — humanised handheld camera | `2.mp4` |
| `render_house3.mjs` | `house3.js` — same motion, de-jittered (gimbal-like) | `3.mp4` |
| `render_house4.mjs` | `house4.js` — adds a staircase and upper landing | `4.mp4` |

The handheld camera model in `house2.js` onward is fit to real footage:
`da3_poses.py` recovers a reference tour's trajectory with Depth Anything 3
into `camera_ref.json`, and `analyze_path.py` characterises it (pitch, roll,
yaw-rate, speed surging, sway and bob) into the constants the scene uses.

## Where this goes next

The extension axes from the proposal map onto this codebase directly:
path variety/complexity is `pathgen.py` + the main CONFIG; graphical realism
is `textures.py`/`gl_renderer.py` (or swapping the renderer for Isaac Lab
while keeping `pathgen.py`, the world mapping, and the ground-truth format
intact).
