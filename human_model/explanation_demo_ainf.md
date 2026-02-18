# **Complete, small “from start to end” Active Inference / Variational Free Energy (VFE) minimisation example** with:

1. **Problem formulation**
2. **Math (generative model → VFE → complexity/accuracy)**
3. **A simple Python code** that minimises VFE by gradient descent on belief parameters.

This example is **1D** so it stays understandable.

---

## 1) Problem setup (what we are trying to do)

There is a hidden state (s) (unknown).
We observe a noisy measurement (o).

### Generative model (what the agent assumes)

**Prior (before seeing data):**
[
p(s) = \mathcal{N}(s; \mu_0,\sigma_0^2)
]

**Likelihood (sensor model):**
[
p(o\mid s) = \mathcal{N}(o; s,\sigma_o^2)
]

We observe one value (o).
Goal: infer the hidden state (s).

---

## 2) Variational approximation

We approximate the true posterior (p(s\mid o)) with:

[
q(s)=\mathcal{N}(s;\mu,\sigma^2)
]

We will find (\mu,\sigma) by **minimising variational free energy**.

---

## 3) Variational Free Energy

Definition:

[
F[q,o] = \mathbb{E}_q\big[\log q(s) - \log p(o,s)\big]
]

Using (p(o,s)=p(o\mid s)p(s)), you get the standard decomposition:

[
F[q,o] =
\underbrace{D_{KL}(q(s)|p(s))}_{\text{complexity}}
--------------------------------------------------

\underbrace{\mathbb{E}*q[\log p(o\mid s)]}*{\text{accuracy}}
]

### Complexity term (Gaussian KL)

For (q=\mathcal{N}(\mu,\sigma^2)) and (p=\mathcal{N}(\mu_0,\sigma_0^2)):

[
D_{KL}(q|p)
===========

\frac12\left(
\frac{\sigma^2}{\sigma_0^2}
+
\frac{(\mu-\mu_0)^2}{\sigma_0^2}
-1
+\log\frac{\sigma_0^2}{\sigma^2}
\right)
]

### Accuracy term (expected log-likelihood)

Since (p(o\mid s)=\mathcal{N}(o;s,\sigma_o^2)),

[
-\mathbb{E}_q[\log p(o\mid s)]
==============================

\frac12\left(
\frac{(o-\mu)^2+\sigma^2}{\sigma_o^2}
+\log(2\pi\sigma_o^2)
\right)
]

So the full free energy (up to constants) is:

[
F(\mu,\sigma)
=============

D_{KL}(q|p)
+
\frac12\left(
\frac{(o-\mu)^2+\sigma^2}{\sigma_o^2}
\right)
+\text{const}
]

Minimise (F) → get the best approximate posterior.

---

## 4) Simple Python code (gradient descent on VFE)

This code:

* defines (F(\mu,\log\sigma)) (we optimise (\log\sigma) to keep (\sigma>0))
* computes gradients analytically
* runs gradient descent
* prints the learned posterior parameters

```python
import math

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

    for t in range(steps):
        F, g_mu, g_logsig = vfe_and_grads(mu, log_sigma, o, mu0, sigma0, sigma_o)

        # gradient descent
        mu -= lr * g_mu
        log_sigma -= lr * g_logsig

        if t % 20 == 0 or t == steps - 1:
            print(f"step={t:3d}  F={F:.6f}  mu={mu:.6f}  sigma={math.exp(log_sigma):.6f}")

    return mu, math.exp(log_sigma)


if __name__ == "__main__":
    # Example: prior mean 0, prior std 2; observe o=2 with sensor std 1
    mu_star, sigma_star = minimise_vfe(o=2.0, mu0=0.0, sigma0=2.0, sigma_o=1.0)
    print("\nLearned posterior q(s)=N(mu, sigma^2):")
    print("mu   =", mu_star)
    print("sigma=", sigma_star)
```

### What you should see

* (\mu) moves from the prior mean toward the observation value (o)
* (\sigma) shrinks as the posterior becomes more confident than the prior

---

## 5) (Optional) The closed-form Bayesian posterior (for checking)

For Gaussian prior + Gaussian likelihood, the true posterior is also Gaussian:

[
\sigma_{\text{post}}^2 = \left(\frac{1}{\sigma_0^2}+\frac{1}{\sigma_o^2}\right)^{-1}
,\quad
\mu_{\text{post}} = \sigma_{\text{post}}^2\left(\frac{\mu_0}{\sigma_0^2}+\frac{o}{\sigma_o^2}\right)
]

Your gradient descent result should converge close to these values.

---
<br>



# **Easier Explanation**

### What you’re looking at (one sentence)

This example shows how a robot/agent can **infer a hidden state** by choosing a belief distribution (q(s)) that **minimises variational free energy**—which is a computable objective that balances **fit to data** and **not over-changing your prior belief**.

I’ll explain **Steps 1–3** in detail (with every variable), then explain the plots/results.

---

## Step 1 — Define the problem (what are (s) and (o)?)

### Hidden state (s)

* (s) is something real but **not directly observed**.
* Example: the true distance to an object, the true position of an obstacle, the true temperature, etc.

So (s) is called a **latent** or **hidden** variable.

### Observation (o)

* (o) is what you actually measure (sensor reading).
* Sensors are noisy, so (o) is an imperfect version of (s).

Example: if the true distance is (s=2), your sensor might output (o=2.1) or (1.9).

### The goal

Given one observed value (o), estimate what (s) probably is.

That is inference:
[
\text{infer } s \text{ from } o.
]

---

## Step 2 — Define the agent’s “generative model” (p(\cdot))

A generative model is just the agent’s assumptions about **how the world makes data**.

It has two parts:

### (A) Prior: (p(s))

This describes what you believed about (s) **before** seeing the current observation.

We choose a Gaussian prior:
[
p(s)=\mathcal{N}(s;\mu_0,\sigma_0^2)
]
Variables:

* (\mu_0): prior mean (“my best guess before data”)
* (\sigma_0^2): prior variance (“how uncertain I was”)
* (\sigma_0): prior standard deviation

Why Gaussian?

* Easy math
* Common in robotics (Kalman filters)
* Works well for “one main hypothesis” problems

**Interpretation:**
If (\sigma_0) is large → you were unsure.
If (\sigma_0) is small → you were very confident.

---

### (B) Likelihood: (p(o\mid s))

This describes how observations are generated from a state.

We assume:
[
p(o\mid s)=\mathcal{N}(o; s,\sigma_o^2)
]

Variables:

* (\sigma_o^2): observation noise variance
* (\sigma_o): observation noise std (sensor quality)

This says:

> if the true state is (s), the sensor reading (o) will be near (s), with noise level (\sigma_o).

Why this is here:

* Without a likelihood, you cannot connect the hidden state to what you observe.
* This is literally the “sensor model”.

---

## Step 3 — Define what we want but can’t compute easily: the posterior (p(s\mid o))

The “ideal Bayesian answer” is:
[
p(s\mid o)=\frac{p(o\mid s)p(s)}{p(o)}
]

Where:

* (p(s\mid o)): posterior = “belief about state after seeing observation”
* (p(o)): evidence = (\int p(o\mid s)p(s),ds)

Why is (p(o)) a problem?

* In many real systems, that integral is hard or impossible.
* (In this simple Gaussian case it’s easy, but the *general method* is designed for hard cases.)

So Active Inference / Variational Inference says:

> Instead of computing (p(s\mid o)) exactly, we approximate it.

---

## Step 3 (continued) — Variational approximation (q(s))

We introduce a simpler distribution:
[
q(s)=\mathcal{N}(s;\mu,\sigma^2)
]

Variables:

* (\mu): the current estimated mean of the state
* (\sigma^2): the current uncertainty about the state

**Goal now becomes:**

> Choose (\mu,\sigma) so that (q(s)) is close to the true posterior (p(s\mid o)).

But how do we measure “close”? That’s where free energy comes in.

---

## Why Free Energy is introduced

Free energy is defined as:
[
F[q,o]=\mathbb{E}_q[\log q(s)-\log p(o,s)]
]

Where:

* (\mathbb{E}_q[\cdot]) means “average under (q(s))”
* (p(o,s)=p(o\mid s)p(s)) is the joint probability from the generative model

This quantity is useful because:

1. It is **computable** (given your model and (q))
2. Minimising it makes (q(s)) move toward the true posterior

And it decomposes into:
[
F[q,o] = \underbrace{D_{KL}(q(s)|p(s))}_{\text{complexity}}
-\underbrace{\mathbb{E}*q[\log p(o\mid s)]}*{\text{accuracy}}
]

Meaning:

* **Complexity:** “how much you changed from the prior”
* **Accuracy:** “how well you explain the observation”

Minimising (F) means:

* fit data well (increase accuracy)
* don’t overreact (keep complexity low)

---

# Explaining the results in your plot

### Plot 1: Free Energy Convergence

Free energy drops quickly then flattens.

* Early iterations: beliefs are wrong → big improvement each step
* Later: belief is near optimum → small changes

**Interpretation:** the optimisation successfully found a stable belief.

---

### Plot 2: Parameter Evolution ((\mu) and (\sigma))

* (\mu) (posterior mean) moves from prior mean toward the observation.
* (\sigma) (posterior uncertainty) shrinks compared to the prior if the observation is informative.

**Interpretation:**

* the agent becomes more confident
* the mean becomes a compromise between prior and data

---

### Plot 3: Prior, Likelihood, Posterior

* **Blue**: prior (p(s)) (what you believed before)
* **Green**: likelihood (p(o\mid s)) viewed as a function of (s) (which states make the observed (o) likely)
* **Red**: posterior approximation (q(s))

You see the posterior (red) sits between the prior and the likelihood peak:

* If sensor is reliable (small (\sigma_o)), posterior moves closer to the observation.
* If prior is strong (small (\sigma_0)), posterior stays closer to the prior.

**That is the key result:**

> The final belief is a **principled compromise** between prior knowledge and sensory evidence.

---

## What this demonstrates (in plain words)

Active inference here is doing:

1. start with a prior belief
2. see an observation
3. adjust your belief distribution (q(s)) by minimising free energy
4. end up with a posterior belief that matches Bayesian inference (in this simple case)

---

If you tell me the numbers you used ((\mu_0,\sigma_0,o,\sigma_o)), I can interpret your exact curves precisely (e.g., why (\mu) converges near 1.6, why (\sigma) settles near ~0.89).
