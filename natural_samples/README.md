# Natural-motion samples (smoothed)

`natural_motion.py` -> `plane_gl.py` (moderngl). ~10 s each, ~1.6 s to render.
Same world, objects and questions as the sphere training set; only the camera
motion is loosened.

Every clip mixes several kinds of randomness:

* **translation** — speed varies smoothly throughout; slows to a halt and
  picks up again; sometimes backs up a little; lateral sway means movement is
  not locked to where the camera is facing
* **yaw** — 1-3 deliberate turns of varying size, each taking longer the
  bigger it is, on top of a continuous wandering drift
* **pitch** — looking up and down, with the occasional glance at the ground
* **roll** — a couple of degrees of head tilt, always changing
* **bob** — a few millimetres at ~2 steps a second, fading out as the walker
  slows and stopping when they stop

**Acceleration is enforced.** Every target signal (speed, sway, turn rate,
pitch, roll) is written as if it could change instantly and then passed
through a two-pass moving average, so nothing steps: stops are ramps, turns
ease in and out. Measured on these clips: max linear acceleration 2.8 u/s^2,
max yaw acceleration 66 deg/s^2, head bob 6-17 mm peak-to-peak (the first,
juddery version reached 90 mm).

Ground truth is unaffected: a bearing is a horizontal angle, so pitch and roll
do not enter it, and yaw still means what it always meant.
