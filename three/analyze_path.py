"""Characterize the handheld camera trajectory recovered by DA3.

Gravity-aligns the stitched path, normalizes scale so median walking speed
is 1.05 m/s (typical indoor tour pace), then measures the statistics that
matter for synthesizing believable first-person motion.
"""

import json
import numpy as np
from scipy import signal

d = json.load(open("/Users/ameenizhac/Documents/spaceforge-sim/three/camera_ref.json"))
fps = d["fps"]
P = np.array(d["positions"])            # (N,3)
R = np.array(d["rotations_c2w"]).reshape(-1, 3, 3)
N = len(P)

# OpenCV cam axes in world: col0 right, col1 down, col2 forward
up_w = -R[:, :, 1].mean(0)
up_w /= np.linalg.norm(up_w)
# rotate world so up -> +Y
def rot_between(a, b):
    v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)
A = rot_between(up_w, np.array([0, 1.0, 0]))
P = P @ A.T
R = np.einsum("ij,njk->nik", A, R)

fwd = R[:, :, 2]
rightv = R[:, :, 0]
downv = R[:, :, 1]

# scale normalization: median moving speed -> 1.05 m/s
v = np.gradient(P, axis=0) * fps
sp_h = np.linalg.norm(v[:, [0, 2]], axis=1)
mov = sp_h > np.percentile(sp_h, 40)
scale = 1.05 / np.median(sp_h[mov])
P *= scale
v *= scale
sp_h *= scale
print(f"scale-normalized: median moving speed set to 1.05 m/s (x{scale:.3f})")
print(f"duration {N/fps:.1f}s  path length {np.sum(np.linalg.norm(np.diff(P,axis=0),axis=1)):.1f} m")

# angles
yaw = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2]))
pitch = np.arcsin(np.clip(-fwd[:, 1] * -1, -1, 1))  # + = looking up
pitch = np.arcsin(np.clip(fwd[:, 1], -1, 1))
roll = np.arcsin(np.clip(rightv[:, 1], -1, 1))

deg = np.degrees
yr = np.diff(yaw) * fps
pr = np.diff(pitch) * fps
rr = np.diff(roll) * fps

print("\n-- orientation --")
print(f"pitch: mean {deg(pitch.mean()):+.1f} deg, std {deg(pitch.std()):.1f}, range [{deg(pitch.min()):+.1f},{deg(pitch.max()):+.1f}]")
print(f"roll:  mean {deg(roll.mean()):+.1f} deg, std {deg(roll.std()):.2f}, range [{deg(roll.min()):+.1f},{deg(roll.max()):+.1f}]")
print(f"yaw rate: rms {deg(np.sqrt((yr**2).mean())):.1f} deg/s, p95 |.| {deg(np.percentile(np.abs(yr),95)):.1f}, max {deg(np.abs(yr).max()):.1f}")
print(f"pitch rate rms {deg(np.sqrt((pr**2).mean())):.1f} deg/s   roll rate rms {deg(np.sqrt((rr**2).mean())):.1f} deg/s")

# yaw accel (jerkiness)
ya = np.diff(yr) * fps
print(f"yaw accel rms {deg(np.sqrt((ya**2).mean())):.0f} deg/s^2, p95 {deg(np.percentile(np.abs(ya),95)):.0f}")

print("\n-- speed --")
print(f"speed: median(moving) {np.median(sp_h[mov]):.2f} m/s, p90 {np.percentile(sp_h,90):.2f}, min {sp_h.min():.2f}")
stopped = sp_h < 0.25
# stop segments
segs, s0 = [], None
for i, st in enumerate(stopped):
    if st and s0 is None: s0 = i
    if not st and s0 is not None:
        if i - s0 >= fps * 0.5: segs.append((s0, i))
        s0 = None
if s0 is not None and N - s0 >= fps * 0.5: segs.append((s0, N))
print(f"stops >=0.5s: {len(segs)}, total {sum(b-a for a,b in segs)/fps:.1f}s of {N/fps:.1f}s")
if segs:
    yr_stop = np.concatenate([yr[a:b-1] for a, b in segs])
    print(f"yaw rate while stopped: rms {deg(np.sqrt((yr_stop**2).mean())):.1f} deg/s")

# gaze vs velocity misalignment while moving
vel_dir = np.arctan2(v[:, 0], v[:, 2])
mis = np.degrees(np.arctan2(np.sin(vel_dir - yaw), np.cos(vel_dir - yaw)))
mis_mov = mis[mov & (sp_h > 0.5)]
print(f"\n-- gaze vs velocity (moving) --")
print(f"misalign: mean {mis_mov.mean():+.1f} deg, std {mis_mov.std():.1f}, p95 |.| {np.percentile(np.abs(mis_mov),95):.1f}")

# vertical bob & step frequency
y_det = signal.detrend(P[:, 1])
b, a = signal.butter(3, [1.2, 4.5], "bandpass", fs=fps)
y_band = signal.filtfilt(b, a, P[:, 1])
f, Pxx = signal.welch(y_det, fs=fps, nperseg=min(256, N))
band = (f > 1.0) & (f < 4.5)
f_step = f[band][np.argmax(Pxx[band])]
print(f"\n-- step oscillation --")
print(f"step freq peak: {f_step:.2f} Hz, vertical bob rms (1.2-4.5Hz) {y_band.std()*1000:.1f} mm")

# lateral sway (in heading frame), roll/pitch band power at step freq
lat = np.zeros(N)
head_u = np.stack([np.sin(yaw), np.zeros(N), np.cos(yaw)], 1)
side_u = np.stack([np.cos(yaw), np.zeros(N), -np.sin(yaw)], 1)
dp = np.gradient(P, axis=0)
lat_v = (dp * side_u).sum(1) * fps
b2, a2 = signal.butter(3, [0.5, 2.0], "bandpass", fs=fps)
lat_band = signal.filtfilt(b2, a2, np.cumsum(lat_v) / fps)
print(f"lateral sway rms (0.5-2Hz): {lat_band.std()*1000:.0f} mm")
for nm, sig_ in (("pitch", pitch), ("roll", roll), ("yaw", yaw)):
    s_band = signal.filtfilt(b, a, sig_)
    lo = signal.filtfilt(*signal.butter(3, 0.7, "lowpass", fs=fps), sig_)
    print(f"{nm}: step-band rms {deg(s_band.std()):.2f} deg, slow-drift rms {deg((sig_-lo).std()):.2f} | lo-band rms {deg(np.std(lo - np.mean(lo))):.2f}")

# PSD slopes for yaw rate (how energy distributes)
f2, Pyy = signal.welch(np.degrees(yr), fs=fps, nperseg=min(256, N - 1))
for flo, fhi in ((0.1, 0.5), (0.5, 1.5), (1.5, 4.0)):
    m = (f2 >= flo) & (f2 < fhi)
    print(f"yaw-rate power {flo}-{fhi} Hz: {np.trapezoid(Pyy[m], f2[m]):.1f} (deg/s)^2")

# turns: sustained |yaw rate| smoothed over 0.5s
yr_s = signal.filtfilt(*signal.butter(2, 0.8, "lowpass", fs=fps), yr)
turning = np.abs(yr_s) > np.radians(25)
tsegs, s0 = [], None
for i, st in enumerate(turning):
    if st and s0 is None: s0 = i
    if not st and s0 is not None:
        if i - s0 > fps * 0.4: tsegs.append((s0, i))
        s0 = None
print(f"\n-- turns (sustained >25 deg/s) --")
print(f"count {len(tsegs)}")
for a_, b_ in tsegs[:12]:
    seg = deg(yr_s[a_:b_])
    print(f"  t={a_/fps:5.1f}s dur {(b_-a_)/fps:.1f}s peak {np.abs(seg).max():.0f} deg/s total {np.sum(seg)/fps:+.0f} deg")

# plot for visual inspection
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
ax[0, 0].plot(P[:, 0], P[:, 2], lw=1)
ax[0, 0].scatter(P[::20, 0], P[::20, 2], s=6, c=np.arange(N)[::20], cmap="viridis")
ax[0, 0].set_title("top-down path (m)"); ax[0, 0].axis("equal")
t = np.arange(N) / fps
ax[0, 1].plot(t, deg(yaw - yaw[0]), label="yaw")
ax[0, 1].plot(t, deg(pitch), label="pitch")
ax[0, 1].plot(t, deg(roll), label="roll")
ax[0, 1].legend(); ax[0, 1].set_title("angles (deg)")
ax[1, 0].plot(t, sp_h); ax[1, 0].set_title("horizontal speed (m/s)")
ax[1, 1].plot(t, signal.detrend(P[:, 1]) * 1000)
ax[1, 1].set_title("vertical (detrended, mm)")
plt.tight_layout()
plt.savefig("/private/tmp/claude-501/-Users-ameenizhac/72fbbca6-dde6-4222-b3ff-98c213f77806/scratchpad/traj_analysis.png", dpi=90)
print("\nplot saved")
