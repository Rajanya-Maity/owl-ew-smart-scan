"""
Whittle-Index-inspired scheduler, with a periodicity-detection layer.

Background
----------
The scan-scheduling problem this project simulates — one receiver, many
bands, each band's activity continuing to evolve whether or not it is being
watched — is a specific, named problem in the literature: the **Restless
Multi-Armed Bandit** (Whittle, 1988). Classic bandits (Epsilon-Greedy, UCB1,
Thompson Sampling, as implemented in schedulers/mab.py) assume a band's
value is a single fixed number, which is a mismatch for a band whose true
state keeps changing while unobserved. Liu & Zhao (IEEE Trans. Info. Theory,
2010) derived a closed-form Whittle Index for exactly the two-state
busy/idle channel model used here, and showed that — for positively
correlated channels — picking the band with the highest *current belief* of
being active is equivalent to the Whittle-optimal policy. That is the
approach implemented below: no training required, and it adapts immediately
to any spectrum, including ones it has never seen.

How it works
------------
1. Belief tracking: each band keeps a running estimate of the probability
   that it is active *right now*. When a band is observed, its belief is
   set exactly (0 or 1). While unobserved, its belief decays back toward
   the band's own long-run activity rate — reflecting that a band recently
   seen active is more likely to still be active soon after, but that this
   certainty fades the longer it goes unwatched.
2. Long-run activity rate: a simple running average of what has actually
   been observed for that band (the same statistic UCB1/Epsilon-Greedy use
   as their sole signal) — used here as the belief's resting point rather
   than the whole story.
3. Periodicity detection: the timestamps at which each band was observed
   *and found active* are kept. Once a handful of hits on a band show a
   consistent spacing, that spacing is treated as an estimated period, and
   the band's belief is boosted sharply just before its predicted next
   active window — directly targeting the periodic and rotating-scan
   emitter behaviours the brief specifically calls out.

This is a heuristic, practical implementation of the idea, not a from-the-
paper derivation of the exact Whittle index for arbitrary channel dynamics —
that closed form only holds cleanly for a strict two-state Markov channel,
and the periodic/agile/rotating-scan emitters in this simulator are not
exactly that. It is offered as a theoretically-motivated middle ground
between "no model of persistence at all" (the bandits) and "learn everything
from scratch" (the DQN scheduler).
"""
import numpy as np
from env.spectrum_sim import SpectrumSimulator


class BandBelief:
    def __init__(self, decay=0.85, smoothing=0.1):
        self.decay = decay
        self.smoothing = smoothing
        self.rate_estimate = 0.1   # long-run activity rate, EWMA of observations
        self.belief = 0.1          # current belief the band is active right now
        self.last_seen_t = None
        self.hit_times = []        # timestamps at which this band was observed active

    def observe(self, t, hit):
        self.belief = float(hit)
        self.rate_estimate = (1 - self.smoothing) * self.rate_estimate + self.smoothing * hit
        self.last_seen_t = t
        if hit:
            self.hit_times.append(t)
            if len(self.hit_times) > 20:
                self.hit_times.pop(0)

    def predict(self, t):
        """Belief that this band is active at time t, given everything
        observed about it so far (without actually looking)."""
        if self.last_seen_t is None:
            return self.rate_estimate
        gap = t - self.last_seen_t
        decayed = self.rate_estimate + (self.belief - self.rate_estimate) * (self.decay ** gap)

        # periodicity boost: if recent hits on this band show a consistent
        # spacing, boost belief sharply near the predicted next active time
        if len(self.hit_times) >= 3:
            diffs = np.diff(self.hit_times)
            mean_gap = float(np.mean(diffs))
            std_gap = float(np.std(diffs))
            if mean_gap > 0 and std_gap / mean_gap < 0.25:
                last_hit = self.hit_times[-1]
                since_last_hit = t - last_hit
                phase = since_last_hit % mean_gap
                # close to a predicted hit (within +/- 1 step of the cycle boundary)
                distance_to_cycle = min(phase, mean_gap - phase)
                if distance_to_cycle <= 1.5:
                    return max(decayed, 0.92)
        return decayed


def run_whittle_index(sim: SpectrumSimulator, decay=0.85, seed=0):
    sim.reset()
    n = sim.n_bands
    beliefs = [BandBelief(decay=decay) for _ in range(n)]
    log = []
    done = False
    while not done:
        t = sim.t
        scores = [beliefs[b].predict(t) for b in range(n)]
        band = int(np.argmax(scores))
        info, done = sim.step([band])
        r = info["results"][band]
        beliefs[band].observe(t, r)
        log.append(info)
    return log
