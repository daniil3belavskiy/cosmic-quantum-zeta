"""
Cosmic Quantum-Zeta Simulator: Investigating Prime-Driven Quantum Spacings and Refined Zeta Zeros
- Simulates quantum eigenvalue spacings with prime-modulated potential.
- Generates refined untwisted zeta zeros with prime harmonic adjustments.
- Compares results to GUE using KS, CvM, entropy, and Wasserstein metrics.
- Implements a hypothesis linking prime cycles to quantum coherence.

Dependencies: numpy, scipy, matplotlib, tqdm, joblib
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.stats import kstest, cramervonmises, entropy, wasserstein_distance
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

# --- Core Functions ---

def is_prime(n):
    """Check if a number is prime."""
    if n < 2:
        return False
    return all(n % d != 0 for d in range(2, int(n ** 0.5) + 1))

def simulate_gue(N):
    """Generate GUE eigenvalue spacings."""
    gue_matrix = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    gue_matrix = (gue_matrix + gue_matrix.conj().T) / np.sqrt(2)
    eigenvalues = np.linalg.eigvalsh(gue_matrix)
    spacings = np.diff(np.sort(eigenvalues))
    return spacings / np.mean(spacings)  # Unfolded spacings

def prime_potential(x, A, L, alpha_p=48.0, max_prime=500):
    """Prime-modulated potential for quantum simulation."""
    primes = np.array([p for p in range(2, max_prime + 1) if is_prime(p)], dtype=np.float32)
    potential = np.zeros_like(x, dtype=np.float32)
    for p in primes:
        idx = int(p * (len(x) - 1) / L)
        potential[max(0, idx-2):idx+3] += alpha_p * (1 / np.log(p))  # Sharpened prime peaks
    return A * potential

def simulate_quantum_spacings(N=8192, L=100.0, A=15.3, t_steps=4000, dt=0.005, alpha_p=48.0):
    """Simulate quantum eigenvalue spacings with prime-driven potential."""
    x = np.linspace(0, L, N, dtype=np.float32)
    dx = x[1] - x[0]
    T = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(N, N)) / dx**2  # Kinetic term
    V = prime_potential(x, A, L, alpha_p)  # Potential with prime modulation
    H = T + sparse.diags(V)
    k = min(200, N-1)  # Number of eigenvalues to compute
    eigenvalues = eigsh(H, k=k, which='SA', return_eigenvectors=False)
    spacings = np.diff(np.sort(eigenvalues))
    return spacings / np.mean(spacings)  # Unfolded spacings

def generate_mock_zeta_zeros(n_zeros=2000):
    """Generate mock zeta zeros starting with known values."""
    known_zeros = np.array([14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
                            37.586177, 40.918719, 43.327073, 48.005151, 49.773832], dtype=np.float32)
    n = np.arange(10, n_zeros + 9)
    t_approx = 2 * np.pi * n / np.log(n / (2 * np.pi))
    base_spacings = simulate_gue(n_zeros)
    spacings = base_spacings * (2 * np.pi / np.log(t_approx / (2 * np.pi)))
    t_n = np.zeros(n_zeros, dtype=np.float32)
    t_n[:10] = known_zeros
    t_n[10:] = known_zeros[-1] + np.cumsum(spacings[9:])
    return t_n

def refine_untwisted_zeros(n_zeros=2000, quantum_spacings=None, eta=0.15):
    """Refine untwisted zeta zeros with prime harmonic modulation."""
    t_n = generate_mock_zeta_zeros(n_zeros)
    delta_n = np.diff(t_n)
    n = np.arange(len(delta_n))
    primes = np.array([p for p in range(2, 501) if is_prime(p)])

    # Prime harmonic influence
    prime_influence = 1 + eta * np.cos(2 * np.pi * np.array([np.sum(np.exp(-((n_val + 1 - primes)**2) / 2)) 
                                                            for n_val in n]))
    delta_weighted = delta_n * prime_influence

    # High-end scaling
    mean_spacing = np.mean(delta_weighted)
    delta_weighted[delta_weighted > mean_spacing] *= 0.10

    # Quantum-prime fusion
    if quantum_spacings is not None and len(quantum_spacings) == len(delta_weighted):
        delta_weighted = 0.005 * delta_weighted + 0.995 * quantum_spacings[:len(delta_weighted)]
        delta_weighted[delta_weighted > mean_spacing] *= (0.99 - 0.01 * np.sin(delta_weighted / mean_spacing)**4)

    # Lag-1 shift
    delta_shifted = np.roll(delta_weighted, 1)
    delta_shifted[0] = delta_weighted[0]

    t_n_refined = np.zeros_like(t_n)
    t_n_refined[0] = t_n[0]
    t_n_refined[1:] = t_n[0] + np.cumsum(delta_shifted)
    delta_n_refined = np.diff(t_n_refined)
    mean_density = 2 * np.pi / np.log(t_n_refined[:-1] / (2 * np.pi))
    return delta_n_refined * mean_density

# --- Statistical Analysis ---

def bootstrap_ks(spacings, gue_spacings, n_resamples=10):
    """Bootstrap KS statistic for confidence intervals."""
    ks_stats = [kstest(np.random.choice(spacings, len(spacings), replace=True), gue_spacings)[0]
                for _ in range(n_resamples)]
    return np.mean(ks_stats), np.percentile(ks_stats, [2.5, 97.5])

def analyze_spacings(spacings, gue_spacings, label):
    """Analyze spacings against GUE with multiple metrics."""
    ks_stat, p_val = kstest(spacings, gue_spacings)
    mean_ks, ci = bootstrap_ks(spacings, gue_spacings)
    cvm_result = cramervonmises(spacings, gue_spacings, method='auto')
    ent = entropy(np.histogram(spacings, bins=50, density=True)[0])
    wass = wasserstein_distance(spacings, gue_spacings)
    print(f"{label} vs GUE:")
    print(f"  KS: {ks_stat:.4f}, P: {p_val:.4f}, Mean KS: {mean_ks:.4f}, CI: {ci}")
    print(f"  CvM: {cvm_result.statistic:.4f}, P: {cvm_result.pvalue:.4f}")
    print(f"  Entropy: {ent:.4f}")
    print(f"  Wasserstein: {wass:.4f}")
    return ks_stat, p_val, cvm_result.pvalue

def plot_cdf(spacings, gue_spacings, label):
    """Plot CDF comparison."""
    sorted_spacings = np.sort(spacings)
    sorted_gue = np.sort(gue_spacings)
    cdf_spacings = np.arange(1, len(sorted_spacings) + 1) / len(sorted_spacings)
    cdf_gue = np.arange(1, len(sorted_gue) + 1) / len(sorted_gue)
    plt.plot(sorted_spacings, cdf_spacings, label=f'{label} CDF')
    plt.plot(sorted_gue, cdf_gue, label='GUE CDF', linestyle='--')
    plt.xlabel("Spacings")
    plt.ylabel("CDF")
    plt.legend()
    plt.title(f"{label} vs GUE CDF")
    plt.savefig(f"{label.lower()}_cdf.png")
    plt.close()

# --- Main Simulation ---

def main():
    """Run quantum and refined zeta simulations, compare to GUE."""
    # Parameters
    N = 8192
    n_zeros = 2000
    n_runs_quantum = 100
    n_runs_refined = 200

    # Quantum simulation
    print("Running quantum simulations...")
    quantum_spacings_list = Parallel(n_jobs=-1)(delayed(simulate_quantum_spacings)(N) 
                                                for _ in tqdm(range(n_runs_quantum)))
    quantum_spacings = np.mean(quantum_spacings_list, axis=0)
    gue_spacings_quantum = simulate_gue(n_zeros)
    analyze_spacings(quantum_spacings, gue_spacings_quantum, "Quantum")
    plot_cdf(quantum_spacings, gue_spacings_quantum, "Quantum")

    # Refined zeta simulation
    print("Running refined zeta simulations...")
    refined_spacings_list = Parallel(n_jobs=-1)(delayed(refine_untwisted_zeros)(n_zeros, quantum_spacings) 
                                                for _ in tqdm(range(n_runs_refined)))
    refined_spacings = np.mean(refined_spacings_list, axis=0)
    gue_spacings_refined = simulate_gue(n_zeros)
    analyze_spacings(refined_spacings, gue_spacings_refined, "Refined")
    plot_cdf(refined_spacings, gue_spacings_refined, "Refined")

if __name__ == "__main__":
    main()