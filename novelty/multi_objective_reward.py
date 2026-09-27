"""
Novelty layer — Multi-objective reward (Step 11.1).
Balances Pd, Pfa and intercept-time-error together instead of a flat +1/-0.05.
"""
import numpy as np


class MultiObjectiveReward:
    """
    reward = w_pd * hit
             - w_pfa * miss
             - w_time * (steps_since_last_hit_on_this_band / horizon)   [encourages timely revisits]
    """
    def __init__(self, w_pd=1.0, w_pfa=0.3, w_time=0.2, horizon=200):
        self.w_pd = w_pd
        self.w_pfa = w_pfa
        self.w_time = w_time
        self.horizon = horizon

    def compute(self, hit, since_last_watch_this_band):
        timing_penalty = self.w_time * min(since_last_watch_this_band / self.horizon, 1.0)
        if hit:
            return self.w_pd * 1.0 - timing_penalty * 0  # no penalty on a hit
        return -self.w_pfa * 1.0 - timing_penalty
