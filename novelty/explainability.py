"""
Novelty layer — Explainability: live "confidence per band" (Step 11.2).

Wraps a DQN model (or any scheduler exposing Q-values) and turns raw scores
into a normalized confidence vector over bands, for live dashboard display.
"""
import numpy as np


def dqn_band_confidence(model, obs):
    """Given a trained SB3 DQN model and an observation, return a normalized
    (softmax) confidence score per band from its Q-value estimates."""
    import torch
    obs_t = torch.as_tensor(obs).float().unsqueeze(0)
    with torch.no_grad():
        q_values = model.q_net(obs_t).numpy().flatten()
    # softmax for a readable 0-1 "confidence" display
    q = q_values - q_values.max()
    exp_q = np.exp(q)
    conf = exp_q / exp_q.sum()
    return conf


def mab_band_confidence(values, counts):
    """For UCB1/epsilon-greedy: turn empirical value estimates into a
    display-friendly confidence vector (higher value & higher count -> more confident)."""
    values = np.asarray(values, dtype=float)
    counts = np.asarray(counts, dtype=float)
    certainty = counts / (counts.max() + 1e-9)
    score = values * (0.5 + 0.5 * certainty)
    score = score - score.min() + 1e-9
    return score / score.sum()
