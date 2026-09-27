"""
Turing Synthetic Radar Dataset (TSRD) integration — Step 10.

The PS points to:
  huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset
Companion repo: alan-turing-institute/turing-deinterleaving-challenge

TSRD is a *pulse-deinterleaving* dataset (a different task from ours). We are
NOT solving deinterleaving — we only borrow realistic emitter timing/frequency
statistics (dwell/period distributions, hop counts) to make our simulator's
emitters more true-to-life, as the study guide specifies.

This loader requires a Hugging Face login/network access, which is not
guaranteed in an offline / immediately-runnable deployment. To keep this
project runnable out of the box with zero setup, we:
  1. Try to load TSRD via `datasets` if the `huggingface_hub`/`datasets`
     packages are installed AND network + auth are available.
  2. If that fails for ANY reason (no internet, no login, package missing),
     we fall back to literature-informed synthetic statistics representative
     of typical radar dwell/period behaviour, so the pipeline still runs.

Use `get_emitter_stats()` everywhere else in the codebase.

Presentation note: this project does not train the scheduler directly on
TSRD. TSRD is used only as an optional source of realistic emitter
timing/frequency statistics for parameterizing the simulated RF environment
(dwell times, periods, hop counts). The scheduler itself is trained and
evaluated entirely within the task-specific simulated environment in
env/spectrum_sim.py. That distinction should be kept when this project is
described to others, rather than implying TSRD was used as training data.
"""
import numpy as np


_FALLBACK_STATS = {
    "dwell_times": [2, 3, 3, 4, 5, 6, 8, 10],          # timesteps
    "periods": [15, 20, 20, 25, 30, 40, 50],           # timesteps
    "on_fractions": [0.15, 0.2, 0.25, 0.3, 0.35],      # ON time / period
    "hop_counts": [2, 3, 3, 4, 5],                     # bands an agile emitter hops among
    "source": "fallback_synthetic (TSRD unavailable offline)",
}


def try_load_tsrd(max_records=200):
    """Attempt a real TSRD pull. Returns a stats dict, or None on any failure."""
    try:
        from datasets import load_dataset  # noqa: F401
        ds = load_dataset(
            "alan-turing-institute/turing-synthetic-radar-dataset", split="train"
        )
        dwell_times, periods, on_fracs, hop_counts = [], [], [], []
        for i, row in enumerate(ds):
            if i >= max_records:
                break
            # Field names vary by dataset version; be defensive.
            if "pulse_width" in row:
                dwell_times.append(row["pulse_width"])
            if "pri" in row:  # pulse repetition interval ~ period proxy
                periods.append(row["pri"])
        if not dwell_times and not periods:
            return None
        stats = {
            "dwell_times": dwell_times or _FALLBACK_STATS["dwell_times"],
            "periods": periods or _FALLBACK_STATS["periods"],
            "on_fractions": _FALLBACK_STATS["on_fractions"],
            "hop_counts": _FALLBACK_STATS["hop_counts"],
            "source": "TSRD (huggingface, live)",
        }
        return stats
    except Exception:
        return None


def get_emitter_stats(prefer_live=False):
    """Main entry point used by the simulator to ground emitter parameters.
    Set prefer_live=True to attempt a real Hugging Face pull first (requires
    `pip install datasets huggingface_hub` and `huggingface-cli login`).
    """
    if prefer_live:
        live = try_load_tsrd()
        if live is not None:
            return live
    return _FALLBACK_STATS


def sample_grounded_periodic_params(rng: np.random.Generator, stats=None):
    stats = stats or get_emitter_stats()
    period = int(rng.choice(stats["periods"]))
    on_frac = float(rng.choice(stats["on_fractions"]))
    on_len = max(1, int(round(period * on_frac)))
    return period, on_len


def sample_grounded_agile_params(rng: np.random.Generator, n_bands, stats=None):
    stats = stats or get_emitter_stats()
    hop_n = int(rng.choice(stats["hop_counts"]))
    hop_n = min(hop_n, n_bands)
    dwell = int(rng.choice(stats["dwell_times"]))
    bands = list(rng.choice(n_bands, size=hop_n, replace=False))
    return bands, dwell


if __name__ == "__main__":
    stats = get_emitter_stats(prefer_live=True)
    print("TSRD stats source:", stats["source"])
    print(stats)
