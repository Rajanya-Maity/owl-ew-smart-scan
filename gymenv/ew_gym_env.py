"""
Gymnasium environment wrapper for the spectrum scanning problem — Step 6.

State: rolling window of the last K timesteps' hit/miss history per band
       (unwatched slots encoded as 0, since the scheduler has no info there).
Action: which band to scan next (Discrete(n_bands)); bands_per_step=1.
Reward: +1 for a hit, small negative for a miss.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.spectrum_sim import SpectrumSimulator


class SmartScanEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, n_bands=10, n_timesteps=200, history_window=10,
                 mode="hard", seed=42, reward_hit=1.0, reward_miss=-0.05,
                 randomize_each_episode=True, custom_emitters=None):
        super().__init__()
        self.n_bands = n_bands
        self.n_timesteps = n_timesteps
        self.history_window = history_window
        self.mode = mode
        self.seed_val = seed
        self.reward_hit = reward_hit
        self.reward_miss = reward_miss
        # If True (the default for training), every reset() builds a freshly
        # re-randomized spectrum instead of repeating the same one — this is
        # what stops the agent from simply memorizing one fixed layout of
        # emitters instead of learning a general scanning strategy.
        self.randomize_each_episode = randomize_each_episode
        self.custom_emitters = custom_emitters
        self._episode_rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(n_bands)
        # observation: for each band, K-length recent hit(1)/miss(0)/unseen(0) history
        # plus a "steps since last watched" channel, normalized
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_bands * (history_window + 1),), dtype=np.float32
        )

        self.sim = None
        self._history = None      # n_bands x history_window, most recent last
        self._since_watched = None

    def _build_obs(self):
        hist_flat = self._history.flatten()
        since = np.clip(self._since_watched / self.n_timesteps, 0, 1)
        return np.concatenate([hist_flat, since]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            actual_seed = seed
        elif self.randomize_each_episode and self.mode != "custom":
            actual_seed = int(self._episode_rng.integers(0, 2**31 - 1))
        else:
            actual_seed = self.seed_val
        self.sim = SpectrumSimulator(
            n_bands=self.n_bands, n_timesteps=self.n_timesteps,
            bands_per_step=1, mode=self.mode, seed=actual_seed,
            custom_emitters=self.custom_emitters,
        )
        self.sim.reset()
        self._history = np.zeros((self.n_bands, self.history_window), dtype=np.float32)
        self._since_watched = np.zeros(self.n_bands, dtype=np.float32)
        return self._build_obs(), {}

    def step(self, action):
        band = int(action)
        info, done = self.sim.step([band])
        hit = info["results"][band]
        reward = self.reward_hit if hit == 1 else self.reward_miss

        # slide history for the watched band; increment "since watched" for others
        self._history[band] = np.roll(self._history[band], -1)
        self._history[band, -1] = hit
        self._since_watched += 1
        self._since_watched[band] = 0

        truncated = False
        return self._build_obs(), reward, done, truncated, info

    def render(self):
        pass
