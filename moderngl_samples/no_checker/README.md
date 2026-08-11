# Ground styles without the checkerboard

Three procedural alternatives, two scenes each (`plane_gl.py`, `ground_style`
in the scene spec; `plane_city.py --ground <style>`):

  7100-7101  noise    smooth two-tone fbm, fine grain on top
  7102-7103  patchy   large soft blotches, coarser mottling
  7104-7105  plain    near-flat colour, only a faint grain to avoid banding

Cost is unchanged: 2004 frames in 10.0 s = 200 fps, 1.67 s per scene — the
ground is one quad either way, and the noise is a few fbm octaves.

Why this matters beyond looks: a checkerboard is a metric ruler painted on the
floor. Tiles pass under the camera at a fixed rate, so distance walked can be
read off by counting them rather than by integrating motion. These grounds
carry no repeating unit, so displacement has to come from parallax and object
motion — closer to what a real scene offers.
