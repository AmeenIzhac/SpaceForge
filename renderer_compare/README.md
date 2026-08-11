# Renderer comparison — same scene spec, two renderers

Both read the SAME scene file (`probes/plane_test.json`), so the camera path,
object placement and ground truth are identical; only the pixels differ. The
videos are therefore interchangeable as training data.

| renderer | how | 1 process | parallel | quality notes |
|---|---|---|---|---|
| three.js | headless Chrome + puppeteer, CDP screenshot per frame | ~23 s/scene | 2.8 s/scene (8 workers) | soft shadow maps, mipmapped ground, antialiasing |
| moderngl | native OpenGL, offscreen FBO piped to ffmpeg | 1.3 s/scene | 0.41 s/scene (4 procs) | flat blob shadow, aliased checker, no mipmaps |

~18x cheaper per process, ~7x cheaper at each one's practical parallelism.
Where the cost goes in three.js: one CDP screenshot round-trip per frame,
which dominates the actual drawing.

Files: `sceneNNNN_threejs_vs_moderngl.mp4` (left three.js, right moderngl).
