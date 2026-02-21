import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import json
import argparse
import time

# Ensure we can import the C++ chaos proxy binary
sys.path.append(os.getcwd())
import chaos_proxy

# =============================================================================
# 1. Physics Setup (Vectorized PyTorch)
# =============================================================================
G = 1.0
T_MAX = 100.0  # Increased from 30 to capture full interaction before ejection
DT = 0.01


def nbody_derivatives_torch(y, m):
    """
    y: [B, 12] - [pos_x, pos_y, vel_x, vel_y ...]
    m: [B, 3]  - masses
    """
    B = y.shape[0]

    # Reshape
    pos = y[:, :6].view(B, 3, 2)
    vel = y[:, 6:].view(B, 3, 2)
    acc = torch.zeros((B, 3, 2), dtype=y.dtype, device=y.device)

    for i in range(3):
        for j in range(3):
            if i != j:
                r_ij = pos[:, j, :] - pos[:, i, :]  # [B, 2]
                dist = torch.norm(r_ij, dim=1, keepdim=True)  # [B, 1]
                m_j = m[:, j].view(B, 1)  # [B, 1]
                acc[:, i, :] += G * m_j * r_ij / (dist**3 + 1e-12)

    return torch.cat([vel.view(B, 6), acc.view(B, 6)], dim=1)


def rk4_step_torch(y, m, dt):
    k1 = nbody_derivatives_torch(y, m)
    k2 = nbody_derivatives_torch(y + 0.5 * dt * k1, m)
    k3 = nbody_derivatives_torch(y + 0.5 * dt * k2, m)
    k4 = nbody_derivatives_torch(y + dt * k3, m)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def check_ejection(y):
    """
    Returns a boolean mask [B] where True means the system has a body > 50 units away from origin.
    """
    B = y.shape[0]
    pos = y[:, :6].view(B, 3, 2)
    # dist to origin
    dist_org = torch.norm(pos, dim=2)  # [B, 3]
    max_dist, _ = torch.max(dist_org, dim=1)  # [B]
    return max_dist > 50.0


# =============================================================================
# 2. Vectorized Metric Extraction
# =============================================================================
def compute_mutual_information(x, y, bins=50):
    c_xy = np.histogram2d(x, y, bins)[0]
    return mutual_info_score(None, None, contingency=c_xy)


# =============================================================================
# 3. Main GPU Batch Runner
# =============================================================================
def generate_configs_torch(num_samples, device):
    torch.manual_seed(42)  # For reproducibility

    # m: [B, 3]
    m = torch.empty((num_samples, 3), dtype=torch.float64, device=device).uniform_(
        1.0, 10.0
    )

    # pos0: [B, 3, 2]
    pos0 = torch.empty(
        (num_samples, 3, 2), dtype=torch.float64, device=device
    ).uniform_(-5.0, 5.0)
    # Give non-zero initial velocities to represent non-cold-start states
    vel0 = torch.empty(
        (num_samples, 3, 2), dtype=torch.float64, device=device
    ).uniform_(-1.0, 1.0)

    # Adjust to center of mass frame
    total_m = torch.sum(m, dim=1, keepdim=True).unsqueeze(2)  # [B, 1, 1]
    momentum = m.unsqueeze(2) * vel0  # [B, 3, 2]
    total_momentum = torch.sum(momentum, dim=1, keepdim=True)  # [B, 1, 2]
    v_cm = total_momentum / total_m  # [B, 1, 2]
    vel0 = vel0 - v_cm

    # y0: [B, 12]
    y0 = torch.cat([pos0.view(num_samples, 6), vel0.view(num_samples, 6)], dim=1)

    return y0, m


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Batch Validation of Chaos Proxy")
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of random configurations to test",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    device = torch.device(args.device)

    print(f"Generating {args.samples} random initial conditions...")
    y_base, m_batch = generate_configs_torch(args.samples, device)

    # Initialize perturbation array for mathematical LLE computation
    # tangent: [B, 12]
    tangent = torch.randn((args.samples, 12), dtype=torch.float64, device=device)
    tangent_norm = torch.norm(tangent, dim=1, keepdim=True)
    tangent = tangent / tangent_norm

    epsilon = 1e-8
    y_pert = y_base + epsilon * tangent

    t_eval = np.arange(0, T_MAX, DT)
    num_steps = len(t_eval)
    B = args.samples

    # Pre-allocate arrays to hold metrics across time
    # For memory efficiency, we won't save the full 12D trajectories on GPU,
    # just store K_all, LLE, etc.
    K_all_buffer = torch.zeros((num_steps, B), dtype=torch.float64, device=device)
    lle_buffer = torch.zeros((num_steps, B), dtype=torch.float64, device=device)

    # Record starting kinetic energy
    vel_base = y_base[:, 6:].view(B, 3, 2)
    K_all_buffer[0, :] = 0.5 * torch.sum(m_batch * torch.sum(vel_base**2, dim=2), dim=1)

    ejected_buffer = torch.zeros((num_steps, B), dtype=torch.bool, device=device)
    ejected_buffer[0, :] = check_ejection(y_base)

    print(f"Running vectorized integration for {num_steps} steps...")
    start_time = time.time()

    # The integration Loop
    for i in range(1, num_steps):
        if i % 500 == 0:
            print(f"  Step {i}/{num_steps} ({(i/num_steps)*100:.1f}%)")

        # Standard RK4
        y_base_next = rk4_step_torch(y_base, m_batch, DT)
        y_pert_next = rk4_step_torch(y_pert, m_batch, DT)

        # Calculate LLE across the batch
        dist = torch.norm(y_pert_next - y_base_next, dim=1)  # [B]
        # Gamma calculation equivalent to CPU
        dist_clamp = torch.clamp(dist / epsilon, min=1e-12)
        gamma = torch.log(dist_clamp) / DT
        lle_buffer[i, :] = gamma

        # Renormalize tangent vector
        tangent = y_pert_next - y_base_next
        tangent_norm = torch.norm(tangent, dim=1, keepdim=True)
        # Avoid div by zero
        tangent = torch.where(tangent_norm > 0, tangent / tangent_norm, tangent)

        y_pert_next = y_base_next + epsilon * tangent

        # Calculate Kinetic Energy
        vel_base = y_base_next[:, 6:].view(B, 3, 2)
        K_all_buffer[i, :] = 0.5 * torch.sum(
            m_batch * torch.sum(vel_base**2, dim=2), dim=1
        )

        # Advance state
        y_base = y_base_next
        y_pert = y_pert_next

        # Track ejections (once ejected, stays ejected)
        current_ejection = check_ejection(y_base)
        ejected_buffer[i, :] = ejected_buffer[i - 1, :] | current_ejection

    integration_time = time.time() - start_time
    print(f"Vectorized integration complete in {integration_time:.2f} seconds.")

    # =============================================================================
    # Move to CPU for analysis
    # =============================================================================
    print("Moving data to CPU for metric extraction...")
    K_all_cpu = K_all_buffer.cpu().numpy()  # [T, B]
    lle_cpu = lle_buffer.cpu().numpy()  # [T, B]
    ejected_cpu = ejected_buffer.cpu().numpy()  # [T, B]

    print("Calculating Kinetic Variance...")
    # Calculate rolling variance over time (axis=0)
    K_var_log_all = np.zeros_like(K_all_cpu)

    for b in range(B):
        series = pd.Series(K_all_cpu[:, b])
        K_var = series.rolling(window=20, min_periods=1).var().fillna(0).values
        K_var_log_all[:, b] = np.log1p(K_var)

    print("Quantizing Signals...")
    bits_array = np.zeros_like(K_all_cpu, dtype=np.uint8)

    # Static minimum variation check bounds
    valid_batch_indices = []

    for b in range(B):
        klog = K_var_log_all[:, b]
        kmin, kmax = np.min(klog), np.max(klog)

        if (kmax - kmin) > 1e-6:
            valid_batch_indices.append(b)
            signal_norm = 0.1 + 0.8 * (klog - kmin) / (kmax - kmin + 1e-12)

            acc = 0.0
            for i in range(num_steps):
                acc += signal_norm[i]
                if acc >= 1.0:
                    bits_array[i, b] = 1
                    acc -= 1.0

    print(f"Valid active simulations: {len(valid_batch_indices)} / {B}")

    print("Running C++ Symbolic Chaos Kernel over batch...")
    # Pre-allocate
    chaos_score_all = np.zeros_like(K_all_cpu)
    window_size = 200

    # Pack the valid ones
    t_compile_start = time.time()

    # For every valid configuration `b`, we must slide the window over time `i`
    for b in valid_batch_indices:
        windows_to_process = []
        for i in range(num_steps):
            if i < window_size:
                window = bits_array[: i + 1, b]
            else:
                window = bits_array[i - window_size + 1 : i + 1, b]

            pad = (8 - len(window) % 8) % 8
            if pad:
                window = np.concatenate([window, np.zeros(pad, dtype=np.uint8)])
            windows_to_process.append(bytes(np.packbits(window).tolist()))

        results = chaos_proxy.analyze_window_batch(windows_to_process)
        for i, res in enumerate(results):
            chaos_score_all[i, b] = res.fluctuation_persistence

    print(f"C++ mapping took {time.time() - t_compile_start:.2f} sec")

    print("Aligning statistics and computing metrics...")

    pearsons = []
    spearmans = []
    mis = []
    survival_times = []
    mean_chaos_scores = []

    for b in valid_batch_indices:
        # LLE Smoothing
        l_smoothed = (
            pd.Series(lle_cpu[:, b])
            .rolling(window=window_size, min_periods=window_size)
            .mean()
            .values
        )

        # Chop burn-in and only keep pre-ejection steps
        start_idx = window_size
        valid_mask = ~np.isnan(l_smoothed[start_idx:])

        # Pre-ejection mask
        pre_ejection_mask = ~ejected_cpu[start_idx:, b]
        final_mask = valid_mask & pre_ejection_mask

        l_valid = l_smoothed[start_idx:][final_mask]
        c_valid = chaos_score_all[start_idx:, b][final_mask]

        if len(l_valid) < 50 or np.var(l_valid) < 1e-10 or np.var(c_valid) < 1e-10:
            continue

        p_val = np.corrcoef(l_valid, c_valid)[0, 1]
        if np.isnan(p_val):
            continue

        s_val, _ = spearmanr(l_valid, c_valid)
        if np.isnan(s_val):
            continue

        mi_val = compute_mutual_information(l_valid, c_valid)

        pearsons.append(float(p_val))
        spearmans.append(float(s_val))
        mis.append(float(mi_val))

        if np.any(ejected_cpu[:, b]):
            survival_step = np.argmax(ejected_cpu[:, b])
        else:
            survival_step = num_steps
        survival_times.append(survival_step * DT)
        mean_chaos_scores.append(float(np.mean(c_valid)))

    print("\n--- Massive GPU Validation Summary ---")
    valid_fraction = len(pearsons) / args.samples
    print(
        f"Total Valid Trajectories evaluated: {len(pearsons)} / {args.samples} ({valid_fraction*100:.1f}%)"
    )
    print(
        f"Pearson r   | Mean: {np.mean(pearsons):.3f} +/- {np.std(pearsons):.3f} | Median: {np.median(pearsons):.3f}"
    )
    print(
        f"Spearman rho| Mean: {np.mean(spearmans):.3f} +/- {np.std(spearmans):.3f} | Median: {np.median(spearmans):.3f}"
    )
    print(
        f"Mutual Info | Mean: {np.mean(mis):.3f} +/- {np.std(mis):.3f} | Median: {np.median(mis):.3f}"
    )

    # Generate distribution plot
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].hist(pearsons, bins=50, color="blue", alpha=0.7)
    axes[0].set_title(
        f"Pearson (r)\nMean: {np.mean(pearsons):.3f} | Median: {np.median(pearsons):.3f}"
    )

    axes[1].hist(spearmans, bins=50, color="orange", alpha=0.7)
    axes[1].set_title(
        f"Spearman (rho)\nMean: {np.mean(spearmans):.3f} | Median: {np.median(spearmans):.3f}"
    )

    axes[2].hist(mis, bins=50, color="green", alpha=0.7)
    axes[2].set_title(
        f"Mutual Information\nMean: {np.mean(mis):.3f} | Median: {np.median(mis):.3f}"
    )

    plt.tight_layout()
    plt.savefig("assets/validation_distributions.png", dpi=200)
    print("\nSaved statistical breakdown to assets/validation_distributions.png")

    # Optional plot: Chaos Score vs Survival Time
    plt.figure(figsize=(8, 6))
    plt.scatter(survival_times, mean_chaos_scores, alpha=0.3, s=5, c="cyan")
    plt.xscale("log")
    plt.xlabel("Survival Time (T)")
    plt.ylabel("Mean Chaos Score (F_p)")
    plt.title("Macroscopic Instability Correlation\nChaos Score vs Survival Time")
    plt.tight_layout()
    plt.savefig("assets/survival_correlation.png", dpi=200)
    print("Saved survival time correlation plot to assets/survival_correlation.png")
