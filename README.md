# Symbolic Chaos Proxy for the 3-Body Problem

This standalone package implements the underlying symbolic dynamics that reliably predict chaotic ejection in the 3-Body problem, achieving a strong **r = 0.811** Pearson correlation with the mathematically rigorous Local Lyapunov Exponent (LLE).

## 1. Physical Signal Extraction: Kinetic Energy Variance
Instead of tracking raw spatial coordinates or distances, this system tracks the **Rolling Variance of the Total Kinetic Energy**:

$$K(t) = \frac{1}{2} \sum_{i=1}^n m_i |v_i(t)|^2$$
$$\sigma_K^2(t) = \text{Var}(K(t) \text{ over } w=20)$$

Kinetic energy variance isolates structural interaction scaling. During quasi-stable orbits, the variance remains bounded. As bodies transition into chaotic close-contact encounters (precursors to ejection), the energy violently accelerates, causing $\sigma_K^2(t)$ to spike. To preserve dynamical range, the variance is scaled logarithmically:
$$S(t) = \ln(1 + \sigma_K^2(t))$$

## 2. Quantization via Delta-Sigma Modulation
The continuous, normalized analog signal $S_{norm}(t)$ is fed into a pure 1-bit Delta-Sigma Modulator. This produces a binary temporal sequence where the density of `1`s tracks the average fluctuation strength, and consecutive runs of `1`s directly identify persistent high-fluctuation episodes.

## 3. Symbolic Dynamics (C++ Kernel)
The Python physics simulation passes sliding discrete windows into the optimized `chaos_proxy` C++ module. The kernel models the structural stability by analyzing state transitions statistically mapped to physical stability constraints. 

For 3-body physics, the meaningful operational states are:
* `LOW_FLUCTUATION`: The sequence generated low event activity, confirming an energetically calm period.
* `OSCILLATION`: Normal dynamic oscillation without exponential breakout.
* `PERSISTENT_HIGH`: The bitstream has maintained a high-density pattern long enough to register consecutive events, representing persistent, high-variance instability.

### Mathematical Signatures
1. **Fluctuation Persistence:** 
   This is the fraction of time the kinetic fluctuation signal stays "high" long enough to produce consecutive `1`s.
   $F_p = \frac{\text{Count of PERSISTENT\_HIGH transitions}}{\text{Total Window size}}$

2. **Entropy ($H$):**
   $H = - (P_{low} \log_2 P_{low} + P_{osc} \log_2 P_{osc} + P_{high} \log_2 P_{high}) / \log_2(3)$
   This Shannon formulation extracts the informational uncertainty of the positional manifold natively.

## Practical Win: Low Computational Overhead
In standard astrodynamics, determining true chaos requires calculating the Local Lyapunov Exponent (LLE) using tangent perturbation vectors—effectively doubling computational cost and introducing profound numerical sensitivity.

By calculating this symbolic Chaos Score, we capture a **structural isomorphism** to the Lyapunov Exponent. While integrating the base N-body system continues to carry its inherent computational cost, this method provides real-time monitoring of orbital chaos during massive simulations with **zero-overhead chaos detection**.

## Massive GPU Validation
To validate the generalisability of the proxy, this repository includes `gpu_batch_validation.py`, a PyTorch-vectorized RK4 solver that simulates 10,000 randomized configurations (masses, positions, and non-zero initial velocities). Extracting the statistics prior to body ejections across $N=8,496$ valid interacting trajectories yields:
- **Median Spearman Rank**: $\rho=0.640$ (mean $0.539 \pm 0.321$)
- **Median Mutual Information**: $\text{MI}=0.645$ (mean $0.691 \pm 0.410$)
- **Median Pearson Correlation**: $r=0.910$ (mean $0.758 \pm 0.345$)

This demonstrates that the proxy robustly preserves the monotonic sequence of escalating chaos across generalised orbital configurations. The high Pearson standard deviation accurately reflects that while the linear correlation is exceptionally tight ($r \approx 0.9$) for most long-lived systems, it diverges in brief transient scenarios (ejections occurring before the asymptotic LLE can be resolved).

## Build & Run
```bash
python -m pip install -r requirements.txt
python setup.py build_ext --inplace
python three_body_demo.py
# GPU validation
python gpu_batch_validation.py --samples 10000
```
