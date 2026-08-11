"""Remove the `<t seconds>` frame tags from Qwen3-VL prompt construction.

The processor writes a literal `<12.3 seconds>` before every frame pair
(processing_qwen3_vl.py, inside `Qwen3VLProcessor.__call__`), which hands the
model each leg's duration as *text*. For the no-externalized-anything
experiments that is a crutch: the walk's distances should have to come from
the pixels (frame counts at a known uniform rate), not from reading numbers.

`install()` rebinds `__call__` with the tag line guarded behind an instance
flag, so behaviour is unchanged until a processor sets `_no_timestamps = True`.
The patch is source-level and asserts the exact line it edits still exists, so
a transformers upgrade fails loudly rather than silently re-enabling tags.

    import notimestamps
    notimestamps.install()
    processor._no_timestamps = True     # per instance, opt in
"""

import inspect
import textwrap

TAG_LINE = 'video_placeholder += f"<{curr_time:.1f} seconds>"'
_installed = False


def install():
    global _installed
    if _installed:
        return
    import transformers.models.qwen3_vl.processing_qwen3_vl as PQ

    src = textwrap.dedent(inspect.getsource(PQ.Qwen3VLProcessor.__call__))
    # getsource includes decorators (@auto_docstring); exec'ing those on a
    # bare function breaks — keep from the def onward
    lines = src.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("def __call__"))
    src = "\n".join(lines[start:])
    if TAG_LINE not in src:
        raise RuntimeError(
            "notimestamps: timestamp line not found in Qwen3VLProcessor "
            "__call__ — transformers changed; refusing to guess.")
    guarded = ('if not getattr(self, "_no_timestamps", False):\n'
               + " " * 24 + TAG_LINE)
    src = src.replace(TAG_LINE, guarded)

    ns = dict(PQ.__dict__)
    exec(compile(src, PQ.__file__ + "#notimestamps", "exec"), ns)
    PQ.Qwen3VLProcessor.__call__ = ns["__call__"]
    _installed = True
