# spaceforge-sim

Purpose-built corridor simulator for the SpaceForge project (§4.1 of the
proposal): generates egocentric videos of walking through corridor
environments instantiated from abstract grid paths, plus the exact spatial
ground truth (poses, junctions, return-turn sequences, landmark positions)
that each video pairs with. No Isaac Lab, no training — just the data
factory's video end.

## Pipeline (abstract → concrete, as in the paper)

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
5. **`generate.py` — the factory.** Renders the outbound walk (start → stop)
   and the return walk as H.264 MP4s and writes all ground truth. Samples run
   in parallel worker processes (`WORKERS`), with grid size N, outbound
   length L, and chord count k randomised per sample. Movement is humanised:
   speed wobble, lateral sway, gaze wander, head-bob, hesitation at
   junctions, glances at landmarks and down side corridors, occasional full
   stops — all seed-deterministic.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python generate.py
```

All knobs are hardcoded in the CONFIG block at the top of `generate.py`
(grid size N, outbound length L, Lmin, attempts T, chords k, resolution, fps,
fov, walking speed, palettes, number of samples, master seed). Runs are fully
deterministic in `MASTER_SEED`.

## Outputs (`out/sample_XX/`)

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

## Deviations from the paper's pseudocode (all configurable)

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

## Where this goes next

The extension axes from the proposal map onto this codebase directly:
path variety/complexity is `pathgen.py` + CONFIG; graphical realism is
`textures.py`/`raycaster.py` (or swapping the renderer for Isaac Lab while
keeping `pathgen.py`, the world mapping, and the ground-truth format intact).
# SpaceForge
