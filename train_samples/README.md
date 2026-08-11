# Training examples (supervision, not predictions)

Each mp4 is one training example exactly as fed to the model:

* left  — the video (the model sees it at 4 fps, 448x256, no timestamps)
* right — top-down map: path so far in blue, objects as coloured dots,
          black ring on the object the question names, arrow = camera
* below — the exact question text and the exact target string

`manifest.json` has the full question and target for each.

The model is trained to emit ONLY the target line (e.g. `ANSWER: 214`).
No reasoning, no working, no timestamps in the prompt.
