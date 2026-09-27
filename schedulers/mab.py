"""Multi-Armed Bandit schedulers — Step 4 (+ Step 11 stretch: Thompson Sampling).
Each frequency band is an "arm". Reward = 1 if a transmission is caught there.

A limitation worth being upfront about: these treat each band as a
*stationary* bandit arm — they learn "this band tends to be busy" as a
single running average, not "this band tends to be busy around this
particular moment in time". A periodic or rotating-scan emitter that is
only active for a few timesteps out of every few dozen will therefore look,
to a MAB, like a band with a middling, roughly constant hit rate rather than
a band with a predictable rhythm. Modelling that rhythm is exactly the gap
the DQN scheduler (training/train_dqn.py) is meant to close, since it
conditions its choice on recent history rather than a single scalar
per band.
"""
import numpy as np
from env.spectrum_sim import SpectrumSimulator


def run_epsilon_greedy(sim: SpectrumSimulator, epsilon=0.1, seed=0, decay=True):
    """decay=True uses a shrinking exploration rate (epsilon_t = epsilon / (1 + visits/50)),
    so the scheduler explores heavily early on and exploits its estimates more as it
    gathers evidence, rather than exploring at a constant 10% rate for the whole episode."""
    sim.reset()
    rng = np.random.default_rng(seed)
    n = sim.n_bands
    counts = np.zeros(n)
    values = np.zeros(n)
    log = []
    done = False
    t = 0
    while not done:
        t += 1
        eff_epsilon = max(0.03, epsilon / (1 + t / 150)) if decay else epsilon
        if rng.random() < eff_epsilon or counts.sum() == 0:
            band = int(rng.integers(0, n))
        else:
            band = int(np.argmax(values))
        info, done = sim.step([band])
        log.append(info)
        r = info["results"][band]
        counts[band] += 1
        values[band] += (r - values[band]) / counts[band]
    return log


def run_ucb1(sim: SpectrumSimulator, c=2.0, seed=0):
    sim.reset()
    n = sim.n_bands
    counts = np.zeros(n)
    values = np.zeros(n)
    log = []
    t = 0
    done = False
    while not done:
        t += 1
        if (counts == 0).any():
            band = int(np.argmin(counts))  # ensure each arm tried once first
        else:
            ucb = values + c * np.sqrt(np.log(t) / counts)
            band = int(np.argmax(ucb))
        info, done = sim.step([band])
        log.append(info)
        r = info["results"][band]
        counts[band] += 1
        values[band] += (r - values[band]) / counts[band]
    return log


def run_thompson_sampling(sim: SpectrumSimulator, seed=0):
    """Stretch goal (Step 11.4): Beta-Bernoulli Thompson Sampling."""
    sim.reset()
    rng = np.random.default_rng(seed)
    n = sim.n_bands
    alpha = np.ones(n)
    beta = np.ones(n)
    log = []
    done = False
    while not done:
        samples = rng.beta(alpha, beta)
        band = int(np.argmax(samples))
        info, done = sim.step([band])
        log.append(info)
        r = info["results"][band]
        if r:
            alpha[band] += 1
        else:
            beta[band] += 1
    return log
