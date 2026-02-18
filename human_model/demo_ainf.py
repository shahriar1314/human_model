import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def vfe_and_grads(mu, log_sigma, o, mu0, sigma0, sigma_o):
    """
    Variational Free Energy for:
      prior      p(s)=N(mu0, sigma0^2)
      likelihood p(o|s)=N(s, sigma_o^2)
      posterior  q(s)=N(mu, sigma^2), where sigma = exp(log_sigma)

    Returns: (F, dF/dmu, dF/dlog_sigma)
    """
    sigma = math.exp(log_sigma)
    s2 = sigma * sigma
    s02 = sigma0 * sigma0
    so2 = sigma_o * sigma_o

    # Complexity = KL(q||p) for Gaussians
    kl = 0.5 * (s2 / s02 + (mu - mu0) ** 2 / s02 - 1.0 + math.log(s02 / s2))

    # Negative accuracy (up to constants that do not affect gradients)
    neg_acc = 0.5 * (((o - mu) ** 2 + s2) / so2)

    F = kl + neg_acc  # constants omitted

    # Gradients:
    # d/dmu: from KL + neg_acc
    dF_dmu = (mu - mu0) / s02 + (mu - o) / so2

    # d/dsigma (then chain to log_sigma): sigma = exp(log_sigma)
    # KL part: 0.5*(2*sigma/s02 - 2/sigma) = sigma/s02 - 1/sigma
    # neg_acc part: 0.5*(2*sigma/so2) = sigma/so2
    dF_dsigma = (sigma / s02 - 1.0 / sigma) + (sigma / so2)

    # chain rule: dF/dlog_sigma = dF/dsigma * dsigma/dlog_sigma = dF/dsigma * sigma
    dF_dlog_sigma = dF_dsigma * sigma

    return F, dF_dmu, dF_dlog_sigma


def minimise_vfe(o=2.0, mu0=0.0, sigma0=2.0, sigma_o=1.0,
                 lr=0.1, steps=200):
    # initialise variational posterior parameters
    mu = 0.0
    log_sigma = math.log(1.0)

    # track history for plotting
    F_history = []
    mu_history = []
    sigma_history = []

    for t in range(steps):
        F, g_mu, g_logsig = vfe_and_grads(mu, log_sigma, o, mu0, sigma0, sigma_o)

        # gradient descent
        mu -= lr * g_mu
        log_sigma -= lr * g_logsig

        # record history
        F_history.append(F)
        mu_history.append(mu)
        sigma_history.append(math.exp(log_sigma))

        if t % 20 == 0 or t == steps - 1:
            print(f"step={t:3d}  F={F:.6f}  mu={mu:.6f}  sigma={math.exp(log_sigma):.6f}")

    return mu, math.exp(log_sigma), F_history, mu_history, sigma_history


if __name__ == "__main__":
    # Example: prior mean 0, prior std 2; observe o=2 with sensor std 1
    mu_star, sigma_star, F_hist, mu_hist, sigma_hist = minimise_vfe(o=2.0, mu0=0.0, sigma0=2.0, sigma_o=1.0)
    print("\nLearned posterior q(s)=N(mu, sigma^2):")
    print("mu   =", mu_star)
    print("sigma=", sigma_star)

    # Create visualization with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Free Energy Convergence
    axes[0].plot(F_hist, linewidth=2, color='#2E86AB')
    axes[0].set_xlabel('Iteration', fontsize=11)
    axes[0].set_ylabel('Variational Free Energy (F)', fontsize=11)
    axes[0].set_title('Free Energy Convergence', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Parameter Evolution (mu and sigma)
    ax2 = axes[1]
    ax2.plot(mu_hist, label='μ (mean)', linewidth=2, color='#A23B72')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(sigma_hist, label='σ (std dev)', linewidth=2, color='#F18F01')
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('μ', fontsize=11, color='#A23B72')
    ax2_twin.set_ylabel('σ', fontsize=11, color='#F18F01')
    ax2.set_title('Parameter Evolution', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#A23B72')
    ax2_twin.tick_params(axis='y', labelcolor='#F18F01')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Prior, Likelihood, and Posterior
    ax3 = axes[2]
    x = np.linspace(-4, 6, 200)
    
    # Prior p(s) = N(0, 2^2)
    prior = norm.pdf(x, loc=0.0, scale=2.0)
    ax3.plot(x, prior, 'b-', linewidth=2, label='Prior p(s)', alpha=0.7)
    
    # Likelihood p(o|s) = N(2, 1^2) (observation o=2 with noise std 1)
    likelihood = norm.pdf(x, loc=2.0, scale=1.0)
    ax3.plot(x, likelihood, 'g-', linewidth=2, label='Likelihood p(o|s)', alpha=0.7)
    
    # Posterior q(s) = N(mu_star, sigma_star^2)
    posterior = norm.pdf(x, loc=mu_star, scale=sigma_star)
    ax3.fill_between(x, posterior, alpha=0.3, color='r', label='Posterior q(s)')
    ax3.plot(x, posterior, 'r-', linewidth=2.5)
    
    ax3.axvline(x=mu_star, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('State s', fontsize=11)
    ax3.set_ylabel('Probability Density', fontsize=11)
    ax3.set_title('Prior, Likelihood & Posterior', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('vfe_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ Graphs saved to 'vfe_results.png'")
    plt.show()
