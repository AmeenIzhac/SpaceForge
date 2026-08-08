"""Extract the camera trajectory from tour_clip.mp4 with DA3-SMALL.

Chunked inference (overlapping windows), stitched into one world frame by
Sim(3) alignment on the overlap: scale from camera-centre spread, rotation
from averaged relative rotations (robust to collinear walking segments),
translation from means. Output: camera_ref.json (positions + c2w rotations).
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

CLIP = Path(__file__).parent / "tour_clip.mp4"
OUT = Path("/Users/ameenizhac/Documents/spaceforge-sim/three/camera_ref.json")
SAMPLE_FPS = 10
WIDTH = 640
CHUNK = 24
OVERLAP = 8

# ---------------------------------------------------------------- frames
cap = cv2.VideoCapture(str(CLIP))
src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
step = max(1, round(src_fps / SAMPLE_FPS))
frames = []
i = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    if i % step == 0:
        h, w = fr.shape[:2]
        nw = WIDTH
        nh = int(round(h * nw / w / 2) * 2)
        fr = cv2.resize(fr, (nw, nh), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    i += 1
cap.release()
fps_eff = src_fps / step
print(f"frames: {len(frames)} at {fps_eff:.2f} fps", flush=True)

# ---------------------------------------------------------------- model
from depth_anything_3.api import DepthAnything3  # noqa: E402

t0 = time.time()
model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL")
device = "mps" if torch.backends.mps.is_available() else "cpu"
try:
    model = model.to(device)
except Exception as e:
    print("mps failed, cpu fallback:", e, flush=True)
    device = "cpu"
    model = model.to(device)
print(f"model on {device} ({time.time()-t0:.0f}s)", flush=True)


def infer(chunk_frames):
    with torch.no_grad():
        pred = model.inference(chunk_frames)
    ext = np.asarray(pred.extrinsics, dtype=np.float64)  # (N,3,4) w2c
    R = ext[:, :3, :3]
    t = ext[:, :3, 3]
    R_c2w = np.transpose(R, (0, 2, 1))
    C = -np.einsum("nij,nj->ni", R_c2w, t)  # camera centres
    return R_c2w, C


def avg_rotation(Rs):
    M = np.sum(Rs, axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


# ------------------------------------------------------- chunked + stitch
starts = list(range(0, max(1, len(frames) - OVERLAP), CHUNK - OVERLAP))
world_R, world_C = None, None
for k, s0 in enumerate(starts):
    idx = list(range(s0, min(s0 + CHUNK, len(frames))))
    if len(idx) < OVERLAP + 2:
        break
    tc = time.time()
    R_new, C_new = infer([frames[j] for j in idx])
    print(f"chunk {k+1}/{len(starts)} frames {idx[0]}-{idx[-1]} "
          f"({time.time()-tc:.0f}s)", flush=True)
    if world_R is None:
        world_R, world_C = list(R_new), list(C_new)
        continue
    # overlap between stitched tail and chunk head
    n_ov = len(world_R) - s0
    assert n_ov >= 2, f"no overlap at chunk {k}"
    Rp = np.array(world_R[s0:])          # prev poses on overlap
    Cp = np.array(world_C[s0:])
    Rn, Cn = R_new[:n_ov], C_new[:n_ov]
    # scale: ratio of centre spreads
    sp = np.linalg.norm(Cp - Cp.mean(0), axis=1).sum()
    sn = np.linalg.norm(Cn - Cn.mean(0), axis=1).sum()
    scale = sp / max(sn, 1e-9)
    # rotation: average of Rp_i @ Rn_i^T
    R_al = avg_rotation(np.einsum("nij,nkj->nik", Rp, Rn))
    t_al = Cp.mean(0) - scale * (R_al @ Cn.mean(0))
    # residual on overlap (sanity)
    C_fit = (scale * np.einsum("ij,nj->ni", R_al, Cn)) + t_al
    res = float(np.linalg.norm(C_fit - Cp, axis=1).mean())
    print(f"  align: scale {scale:.3f} residual {res:.4f}", flush=True)
    # blend the overlap (linear cross-fade of centres), append the rest
    C_tx = (scale * np.einsum("ij,nj->ni", R_al, C_new)) + t_al
    R_tx = np.einsum("ij,njk->nik", R_al, R_new)
    for j in range(n_ov):
        w = (j + 1) / (n_ov + 1)
        world_C[s0 + j] = (1 - w) * world_C[s0 + j] + w * C_tx[j]
        M = (1 - w) * world_R[s0 + j] + w * R_tx[j]
        world_R[s0 + j] = avg_rotation(M[None])
    world_R.extend(R_tx[n_ov:])
    world_C.extend(C_tx[n_ov:])

world_R = np.array(world_R)
world_C = np.array(world_C)
print(f"stitched {len(world_C)} poses", flush=True)

OUT.write_text(json.dumps({
    "fps": fps_eff,
    "positions": world_C.round(6).tolist(),
    "rotations_c2w": world_R.round(6).reshape(len(world_R), 9).tolist(),
}))
print(f"-> {OUT} ({OUT.stat().st_size/1024:.0f} KB)", flush=True)
