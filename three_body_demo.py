import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from scipy.integrate import solve_ivp

# Ensure chaos_proxy can be imported
sys.path.append(os.getcwd())
import chaos_proxy

# =============================================================================
# 1. Physics Setup (Pythagorean 3-Body Problem)
# =============================================================================
m = np.array([3.0, 4.0, 5.0])
G = 1.0

# m1 (mass 3) at (1, 3)
# m2 (mass 4) at (-2, -1)
# m3 (mass 5) at (1, -1)
pos0 = np.array([[1.0, 3.0], [-2.0, -1.0], [1.0, -1.0]], dtype=np.float64)
vel0 = np.zeros((3, 2), dtype=np.float64)

y0 = np.concatenate([pos0.flatten(), vel0.flatten()])


def nbody_derivatives(t, y):
    pos = y[:6].reshape((3, 2))
    vel = y[6:].reshape((3, 2))
    acc = np.zeros((3, 2), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            if i != j:
                r_ij = pos[j] - pos[i]
                dist = np.linalg.norm(r_ij)
                acc[i] += G * m[j] * r_ij / (dist**3 + 1e-12)
    return np.concatenate([vel.flatten(), acc.flatten()])


def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# =============================================================================
# 2. Simulation & Metric Extraction
# =============================================================================
T_MAX = 73.0
DT = 0.01
t_eval = np.arange(0, T_MAX, DT)

print(f"Simulating trajectory using DOP853 to t={T_MAX}...")
sol = solve_ivp(
    nbody_derivatives,
    [0, T_MAX],
    y0,
    method="DOP853",
    t_eval=t_eval,
    rtol=1e-10,
    atol=1e-12,
)
y_all = sol.y.T  # (N, 12)
pos_all = y_all[:, :6].reshape(-1, 3, 2)

print("Calculating Mathematical LLE...")
lle = np.zeros(len(t_eval))
tangent = np.random.randn(12)
tangent /= np.linalg.norm(tangent)
epsilon = 1e-8

for i in range(len(t_eval)):
    y_base = y_all[i]
    y_base_next = rk4_step(nbody_derivatives, t_eval[i], y_base, DT)

    y_pert = y_base + epsilon * tangent
    y_pert_next = rk4_step(nbody_derivatives, t_eval[i], y_pert, DT)

    dist = np.linalg.norm(y_pert_next - y_base_next)
    gamma = np.log(max(dist / epsilon, 1e-12)) / DT
    lle[i] = gamma

    tangent = y_pert_next - y_base_next
    norm = np.linalg.norm(tangent)
    if norm > 0:
        tangent /= norm

# Smooth LLE using strict trailing moving average (no lookahead bias)
window_lle = 200
lle_smoothed = (
    pd.Series(lle).rolling(window=window_lle, min_periods=window_lle).mean().values
)

print(
    "Extracting Kinetic Energy Variance (K_var) and generating proper Delta-Sigma bits..."
)
# Calculate Kinetic Energy over time
K_all = np.zeros(len(t_eval))
for i in range(len(t_eval)):
    vel = y_all[i, 6:].reshape((3, 2))
    K_all[i] = 0.5 * np.sum(m * np.sum(vel**2, axis=1))

# Variance of Kinetic Energy highlights structural rupture vs stability
K_var = pd.Series(K_all).rolling(window=20, min_periods=1).var().fillna(0).values

# Use logarithmic scaling to ensure global normalization doesn't crush the stable phases
K_var_log = np.log1p(K_var)
K_min, K_max = np.min(K_var_log), np.max(K_var_log)
signal_norm = 0.1 + 0.8 * (K_var_log - K_min) / (K_max - K_min + 1e-12)

# Proper Signal Quantizer: Delta-Sigma Modulation
bits_array = np.zeros(len(t_eval), dtype=np.uint8)
acc = 0.0
for i in range(len(t_eval)):
    acc += signal_norm[i]
    if acc >= 1.0:
        bits_array[i] = 1
        acc -= 1.0

print("Running Symbolic Chaos Kernel...")
chaos_score = np.zeros(len(t_eval))
entropy = np.zeros(len(t_eval))

window_size = 200
windows_to_process = []
valid_indices = []

for i in range(len(t_eval)):
    if i < window_size:
        window = bits_array[: i + 1]
    else:
        window = bits_array[i - window_size + 1 : i + 1]

    pad = (8 - len(window) % 8) % 8
    if pad:
        window = np.concatenate([window, np.zeros(pad, dtype=np.uint8)])
    byte_vals = bytes(np.packbits(window).tolist())
    windows_to_process.append(byte_vals)

# Batch process with C++ kernel
results = chaos_proxy.analyze_window_batch(windows_to_process)
for i, res in enumerate(results):
    chaos_score[i] = res.fluctuation_persistence
    entropy[i] = res.entropy

# Calculate Stable Regime (Rt) simple proxy
# True if entropy is stable and hazard is very low
Rt_gate = (entropy > 0.8) & (chaos_score < 0.35)

# =============================================================================
# 3. Visualization Pipeline
# =============================================================================
print("Generating visualizations...")


# Color mapping helper for trails
def get_color(hazard):
    if hazard < 0.35:
        return "green"
    elif hazard > 0.52:
        return "red"
    else:
        return "orange"


colors = [get_color(h) for h in chaos_score]

# Static Output: 4-Panel Dashboard
plt.style.use("dark_background")
fig = plt.figure(figsize=(16, 12))

# Top-Left: Physical Reality
ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
ax1.set_title(
    "Physical Reality: Pythagorean 3-Body Problem\nColored by Chaos Proxy Metric",
    fontsize=14,
)
ax1.set_facecolor("#111111")

# Plot orbits
# We'll plot trail segments to color them dynamically
for i in range(0, len(pos_all) - 1, 5):
    h = chaos_score[i]
    c = get_color(h)
    ax1.plot(
        pos_all[i : i + 6, 0, 0],
        pos_all[i : i + 6, 0, 1],
        color=c,
        alpha=0.6,
        linewidth=1.5,
    )
    ax1.plot(
        pos_all[i : i + 6, 1, 0],
        pos_all[i : i + 6, 1, 1],
        color=c,
        alpha=0.6,
        linewidth=1.5,
    )
    ax1.plot(
        pos_all[i : i + 6, 2, 0],
        pos_all[i : i + 6, 2, 1],
        color=c,
        alpha=0.6,
        linewidth=1.5,
    )

ax1.scatter(pos_all[0, :, 0], pos_all[0, :, 1], c="white", label="Start", zorder=5)
ax1.scatter(pos_all[-1, :, 0], pos_all[-1, :, 1], c="cyan", label="Ejection", zorder=5)
ax1.legend()

# Top-Right: Scatter Plot (Proxy vs LLE)
ax2 = plt.subplot2grid((3, 2), (0, 1), rowspan=2)
ax2.set_title("Chaos Proxy vs Mathematical LLE", fontsize=14)
ax2.set_xlabel(r"Chaos Proxy ($\lambda$)")
ax2.set_ylabel("Local Lyapunov Exponent (LLE)")
ax2.set_facecolor("#111111")
sc = ax2.scatter(
    chaos_score, lle_smoothed, c=chaos_score, cmap="RdYlGn_r", alpha=0.3, s=10
)
# calculate true correlation discarding burn-in window
start_idx = window_size
valid_mask = ~np.isnan(lle_smoothed[start_idx:])
corr = np.corrcoef(
    chaos_score[start_idx:][valid_mask], lle_smoothed[start_idx:][valid_mask]
)[0, 1]
ax2.text(
    0.05,
    0.95,
    f"Pearson r = {corr:.3f}",
    transform=ax2.transAxes,
    fontsize=16,
    color="white",
    verticalalignment="top",
    bbox=dict(facecolor="black", alpha=0.5),
)

# Bottom Panel: Time-series Dashboard (clean for paper)
ax3 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
ax3.set_title("Chaos Score and Entropy Time Series", fontsize=14)
ax3.set_facecolor("#111111")
ax3.plot(t_eval, chaos_score, label="Chaos score", color="red", linewidth=1.5)
ax3.plot(t_eval, entropy, label="Entropy", color="cyan", alpha=0.8, linewidth=1.2)
ax3.set_xlabel("Time")
ax3.set_ylabel("Metric Value")
ax3.legend(loc="upper left")
ax3.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("three_body_dashboard.png", dpi=300)
print("Saved static dashboard to three_body_dashboard.png")

# =============================================================================
# 4. Animated GIF/MP4 Generation
# =============================================================================
print("Generating animation (this may take a minute)...")
fig_anim = plt.figure(figsize=(10, 10))
ax_anim = fig_anim.add_subplot(111)
ax_anim.set_facecolor("#050505")
ax_anim.set_xlim(-8, 8)
ax_anim.set_ylim(-8, 8)
ax_anim.set_title(
    "Symbolic Chaos Proxy in the Pythagorean 3-Body Problem",
    color="white",
)
ax_anim.axis("off")

# Plot elements
trail_len = 150
lines = [ax_anim.plot([], [], "-", lw=2, alpha=0.8)[0] for _ in range(3)]
heads = [ax_anim.plot([], [], "o", color="white", ms=6)[0] for _ in range(3)]
text_hazard = ax_anim.text(
    0.05, 0.95, "", transform=ax_anim.transAxes, color="white", fontsize=14
)

step_stride = 5  # Speed up rendering


def init():
    for line in lines:
        line.set_data([], [])
    for head in heads:
        head.set_data([], [])
    text_hazard.set_text("")
    return lines + heads + [text_hazard]


def update(frame):
    idx = frame * step_stride
    start_idx = max(0, idx - trail_len)

    # Current hazard dictates color
    h = chaos_score[idx]
    c = get_color(h)

    for i in range(3):
        lines[i].set_data(pos_all[start_idx:idx, i, 0], pos_all[start_idx:idx, i, 1])
        lines[i].set_color(c)
        heads[i].set_data(
            [pos_all[idx, i, 0]], [pos_all[idx, i, 1]]
        )  # sequence instead of scalar

    text_hazard.set_text(
        f"t = {t_eval[idx]:.1f}\nChaos Score ($\\lambda$): {h:.3f}\n"
        + ("STABLE" if h < 0.35 else ("MODERATE" if h < 0.52 else "HIGH CHAOS"))
    )
    text_hazard.set_color(c)

    return lines + heads + [text_hazard]


frames = len(t_eval) // step_stride
ani = FuncAnimation(
    fig_anim, update, frames=frames, init_func=init, blit=True, interval=20
)
ani.save(
    "three_body_chaos_demo.mp4",
    fps=30,
    dpi=150,
    extra_args=["-vcodec", "libx264"],
)
print("Saved animation to three_body_chaos_demo.mp4")
print("Done! Demo successfully generated.")
