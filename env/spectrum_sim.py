"""
Spectrum Simulator — Smart Scan Strategy for Electronic Warfare
Author credit: R. Maity
SIH 2026 · PS 26055 · DRDO

Implements Steps 1, 2 and 5 of the build guide:
  - N frequency bands, T discrete timesteps per episode
  - Bernoulli on/off ground truth per band
  - Periodic emitters (ON for k of every p timesteps)
  - Frequency-agile emitters that hop between bands on a repeating/semi-random sequence
  - Receiver constraint: can only observe `k` bands per timestep
  - Ground truth is hidden from the scheduler; it only sees hit/miss on bands it looked at
"""
import numpy as np


class Emitter:
    """Base emitter: fixed-probability Bernoulli transmitter on one band."""
    def __init__(self, band, p_on=0.1, name=None):
        self.band = band
        self.p_on = p_on
        self.name = name or f"bernoulli_b{band}"

    def state_at(self, t, rng):
        return 1 if rng.random() < self.p_on else 0

    def active_band_at(self, t):
        return self.band


class PeriodicEmitter(Emitter):
    """ON for `on_len` steps out of every `period` steps, starting at `phase`."""
    def __init__(self, band, period=20, on_len=5, phase=0, name=None):
        super().__init__(band, p_on=on_len / period, name=name or f"periodic_b{band}")
        self.period = period
        self.on_len = on_len
        self.phase = phase

    def state_at(self, t, rng):
        pos = (t + self.phase) % self.period
        return 1 if pos < self.on_len else 0


class AgileEmitter(Emitter):
    """Hops between a set of bands following a repeating (or semi-random) sequence.
    At each timestep it transmits, but on a different band according to the hop schedule.
    """
    def __init__(self, bands, dwell=3, on_prob=0.9, random_hop=False, seed=0, name=None):
        super().__init__(bands[0], p_on=on_prob, name=name or "agile")
        self.bands = list(bands)
        self.dwell = dwell
        self.on_prob = on_prob
        self.random_hop = random_hop
        self._rng = np.random.default_rng(seed)
        if not random_hop:
            self._sequence = self._rng.permutation(self.bands)
            self._seq_idx = 0

    def _band_for_slot(self, slot):
        if self.random_hop:
            return self.bands[self._rng.integers(0, len(self.bands))]
        idx = slot % len(self._sequence)
        return int(self._sequence[idx])

    def active_band_at(self, t):
        slot = t // self.dwell
        return self._band_for_slot(slot)

    def state_at(self, t, rng):
        return 1 if rng.random() < self.on_prob else 0


class PeriodicScanEmitter(Emitter):
    """A hostile *periodic scanning* transmitter — e.g. a rotating-scan radar —
    that sweeps sequentially through a fixed, ordered list of bands and dwells
    on each one for a fixed number of timesteps before moving to the next.
    Unlike AgileEmitter (which hops in a shuffled/random order), this always
    visits the bands in the SAME order, which is what makes "predicting where
    it will be next" a meaningful, learnable problem — this is the emitter
    behaviour the interception problem is aimed at.
    """
    def __init__(self, bands, dwell=5, on_prob=0.95, name=None):
        super().__init__(bands[0], p_on=on_prob, name=name or "periodic_scan")
        self.bands = list(bands)
        self.dwell = dwell
        self.on_prob = on_prob

    def active_band_at(self, t):
        slot = (t // self.dwell) % len(self.bands)
        return self.bands[slot]

    def state_at(self, t, rng):
        return 1 if rng.random() < self.on_prob else 0


class SpectrumSimulator:
    """
    Ground-truth spectrum generator + receiver constraint model.

    Ground truth table: activity[band, t] in {0,1} — hidden from the scheduler.
    The scheduler calls `observe(bands_to_look_at, t)` and only receives hit/miss
    for the band(s) it chose to look at that timestep.
    """

    def __init__(self, n_bands=10, n_timesteps=200, bands_per_step=1,
                 mode="simple", seed=42, custom_emitters=None):
        """
        mode:
          'simple'  -> Step 1/3/4 world: fixed Bernoulli bands only
          'hard'    -> Step 5 world: adds periodic + frequency-agile emitters
          'custom'  -> emitters supplied explicitly via `custom_emitters`
                       (list of dicts), for the interactive web UI. Each dict:
                         {"type": "bernoulli", "band": 3, "p_on": 0.2}
                         {"type": "periodic", "band": 5, "period": 20, "on_len": 5, "phase": 0}
                         {"type": "agile", "bands": [1,4,7], "dwell": 4, "on_prob": 0.85, "random_hop": False}
        """
        self.n_bands = n_bands
        self.n_timesteps = n_timesteps
        self.bands_per_step = bands_per_step
        self.mode = mode
        self.seed = seed
        self.custom_emitters = custom_emitters
        self.rng = np.random.default_rng(seed)
        self.emitters = []
        self._build_emitters()
        self.ground_truth = self._generate_ground_truth()
        # Scheduler-visible observation log: band x t, values -1 (not looked), 0 (miss), 1 (hit)
        self.obs_log = -np.ones((n_bands, n_timesteps), dtype=int)
        self.t = 0

    # ---------- construction ----------
    def _build_emitters(self):
        rng = np.random.default_rng(self.seed)
        if self.mode == "simple":
            for b in range(self.n_bands):
                # heterogeneous but fixed activity rates -> makes "smart beats dumb" visible
                p = float(rng.uniform(0.05, 0.45))
                self.emitters.append(Emitter(b, p_on=p))
        elif self.mode == "hard":
            for b in range(self.n_bands):
                p = float(rng.uniform(0.03, 0.25))
                self.emitters.append(Emitter(b, p_on=p))
            # at least one clearly periodic band (Step 5 / Step 9 requirement)
            periodic_band = self.n_bands // 2
            self.emitters.append(
                PeriodicEmitter(periodic_band, period=20, on_len=5, phase=0,
                                 name="periodic_target"))
            # at least one frequency-agile emitter hopping among several bands
            agile_bands = list(rng.choice(self.n_bands, size=min(4, self.n_bands), replace=False))
            self.emitters.append(
                AgileEmitter(agile_bands, dwell=4, on_prob=0.85, random_hop=False,
                             seed=self.seed + 1, name="agile_target"))
            # a rotating-scan (periodic-scan) emitter: sweeps a fixed band order on a fixed dwell
            scan_bands = list(rng.choice(self.n_bands, size=min(5, self.n_bands), replace=False))
            self.emitters.append(
                PeriodicScanEmitter(scan_bands, dwell=5, on_prob=0.9, name="periodic_scan_target"))
        elif self.mode == "domain_random":
            # Used for DQN training only: the *composition* of the spectrum
            # itself is randomized every episode (not just the numeric
            # parameters within a fixed structure), so the policy is forced
            # to learn a general scanning strategy rather than memorizing a
            # single fixed layout of emitter types and counts.
            n_bernoulli = int(rng.integers(max(1, self.n_bands // 3), self.n_bands + 1))
            bernoulli_bands = rng.choice(self.n_bands, size=n_bernoulli, replace=False)
            for b in bernoulli_bands:
                p = float(rng.uniform(0.02, 0.4))
                self.emitters.append(Emitter(int(b), p_on=p))

            n_periodic = int(rng.integers(0, 3))
            for _ in range(n_periodic):
                band = int(rng.integers(0, self.n_bands))
                period = int(rng.integers(8, 60))
                on_len = int(rng.integers(1, max(2, period // 3)))
                phase = int(rng.integers(0, period))
                self.emitters.append(PeriodicEmitter(band, period=period, on_len=on_len, phase=phase))

            n_agile = int(rng.integers(0, 2))
            for _ in range(n_agile):
                hop_n = int(rng.integers(2, min(6, self.n_bands) + 1))
                bands = list(rng.choice(self.n_bands, size=hop_n, replace=False))
                dwell = int(rng.integers(2, 8))
                on_prob = float(rng.uniform(0.6, 0.95))
                self.emitters.append(AgileEmitter(bands, dwell=dwell, on_prob=on_prob,
                                                   random_hop=bool(rng.integers(0, 2)),
                                                   seed=int(rng.integers(0, 1_000_000))))

            n_scan = int(rng.integers(0, 2))
            for _ in range(n_scan):
                hop_n = int(rng.integers(2, min(6, self.n_bands) + 1))
                bands = list(rng.choice(self.n_bands, size=hop_n, replace=False))
                dwell = int(rng.integers(2, 10))
                on_prob = float(rng.uniform(0.7, 0.98))
                self.emitters.append(PeriodicScanEmitter(bands, dwell=dwell, on_prob=on_prob))
        elif self.mode == "custom":
            specs = self.custom_emitters or []
            for i, spec in enumerate(specs):
                etype = spec.get("type", "bernoulli")
                if etype == "bernoulli":
                    self.emitters.append(Emitter(spec["band"], p_on=spec.get("p_on", 0.1),
                                                  name=spec.get("name", f"custom_bernoulli_{i}")))
                elif etype == "periodic":
                    self.emitters.append(PeriodicEmitter(
                        spec["band"], period=spec.get("period", 20),
                        on_len=spec.get("on_len", 5), phase=spec.get("phase", 0),
                        name=spec.get("name", f"custom_periodic_{i}")))
                elif etype == "agile":
                    self.emitters.append(AgileEmitter(
                        spec["bands"], dwell=spec.get("dwell", 4),
                        on_prob=spec.get("on_prob", 0.85),
                        random_hop=spec.get("random_hop", False),
                        seed=self.seed + i + 1,
                        name=spec.get("name", f"custom_agile_{i}")))
                elif etype == "periodic_scan":
                    self.emitters.append(PeriodicScanEmitter(
                        spec["bands"], dwell=spec.get("dwell", 5),
                        on_prob=spec.get("on_prob", 0.9),
                        name=spec.get("name", f"custom_periodic_scan_{i}")))
                else:
                    raise ValueError(f"Unknown emitter type: {etype}")
        else:
            raise ValueError("mode must be 'simple', 'hard', 'domain_random' or 'custom'")

    def _generate_ground_truth(self):
        gt = np.zeros((self.n_bands, self.n_timesteps), dtype=int)
        rng = np.random.default_rng(self.seed + 100)
        for t in range(self.n_timesteps):
            for e in self.emitters:
                band = e.active_band_at(t)
                if 0 <= band < self.n_bands:
                    gt[band, t] = max(gt[band, t], e.state_at(t, rng))
        return gt

    # ---------- interaction ----------
    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.ground_truth = self._generate_ground_truth()
        self.obs_log = -np.ones((self.n_bands, self.n_timesteps), dtype=int)
        self.t = 0
        return self.t

    def step(self, bands_chosen):
        """bands_chosen: int or list[int] of length <= bands_per_step.
        Returns dict with per-band hit/miss results for this timestep."""
        if isinstance(bands_chosen, (int, np.integer)):
            bands_chosen = [int(bands_chosen)]
        bands_chosen = bands_chosen[: self.bands_per_step]

        results = {}
        for b in bands_chosen:
            hit = int(self.ground_truth[b, self.t])
            self.obs_log[b, self.t] = hit
            results[b] = hit

        any_active_unwatched = False
        total_active = int(self.ground_truth[:, self.t].sum())
        watched_hits = sum(results.values())
        if total_active > 0 and watched_hits == 0:
            any_active_unwatched = True

        info = {
            "t": self.t,
            "results": results,
            "total_active_bands": total_active,
            "missed_activity": any_active_unwatched,
        }
        self.t += 1
        done = self.t >= self.n_timesteps
        return info, done

    def total_activity_heatmap(self):
        """Ground truth heatmap — for plotting/analysis only, not for the scheduler."""
        return self.ground_truth.copy()
