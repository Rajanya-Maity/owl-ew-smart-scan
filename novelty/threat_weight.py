"""
Novelty layer — Threat-weighted prioritization (Step 11.3).
Some emitters matter more than others (e.g. a fire-control radar vs. a beacon).
This assigns a threat weight per band and multiplies reward accordingly,
so the scheduler learns to prioritize high-threat bands.
"""
import numpy as np


def assign_threat_weights(n_bands, high_threat_bands=None, high_weight=3.0, seed=0):
    rng = np.random.default_rng(seed)
    weights = np.ones(n_bands)
    if high_threat_bands is None:
        # default: randomly mark ~20% of bands as high-threat
        n_high = max(1, n_bands // 5)
        high_threat_bands = rng.choice(n_bands, size=n_high, replace=False)
    for b in high_threat_bands:
        weights[b] = high_weight
    return weights, list(high_threat_bands)


def weighted_reward(base_reward, band, weights):
    return base_reward * weights[band]
