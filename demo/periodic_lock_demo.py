"""
Periodic-scan-receiver phase-lock demo — Step 9.

Explicitly named requirement in the PS: show that a scheduler (trained RL
agent, or a fresh MAB with more episodes) converges to visiting the periodic
emitter's band right when it turns ON, i.e. "phase-locks" onto its period.

Run:  python -m demo.periodic_lock_demo
Produces: results/periodic_lock.png  (heatmap of periodic band watch-times over training-time)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env.spectrum_sim import SpectrumSimulator, PeriodicEmitter


def build_periodic_focused_sim(n_bands=6, n_timesteps=400, period=20, on_len=5, seed=7):
    """A simulator with ONE dominant periodic emitter and quiet background bands,
    so phase-lock is easy to see."""
    sim = SpectrumSimulator(n_bands=n_bands, n_timesteps=n_timesteps, bands_per_step=1,
                             mode="simple", seed=seed)
    # override with a clean periodic-only setup
    target_band = n_bands // 2
    sim.emitters = [PeriodicEmitter(target_band, period=period, on_len=on_len, phase=0)]
    for b in range(n_bands):
        if b != target_band:
            sim.emitters.append(__import__("env.spectrum_sim", fromlist=["Emitter"]).Emitter(b, p_on=0.02))
    sim.ground_truth = sim._generate_ground_truth()
    return sim, target_band


def run_ucb1_repeatedly(n_repeats=30, n_bands=6, n_timesteps=400, period=20, on_len=5):
    from schedulers.mab import run_ucb1
    watch_history = []  # for each repeat, list of bands watched at each t
    target_band = None
    for r in range(n_repeats):
        sim, target_band = build_periodic_focused_sim(
            n_bands=n_bands, n_timesteps=n_timesteps, period=period, on_len=on_len, seed=7)
        log = run_ucb1(sim, seed=r)
        watched = [list(info["results"].keys())[0] for info in log]
        watch_history.append(watched)
    return np.array(watch_history), target_band, period, on_len


def phase_lock_plot(save_path="results/periodic_lock.png"):
    watch_history, target_band, period, on_len = run_ucb1_repeatedly()
    n_repeats, n_t = watch_history.shape
    watched_target = (watch_history == target_band).astype(int)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    axes[0].imshow(watched_target, aspect="auto", cmap="Greens", interpolation="nearest")
    axes[0].set_title(f"Scheduler watching the periodic band (period={period}, on_len={on_len})")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Repeat / episode #")

    # fraction of repeats correctly watching the target band, per timestep
    lock_fraction = watched_target.mean(axis=0)
    axes[1].plot(lock_fraction, color="green")
    axes[1].set_title("Fraction of episodes watching the periodic band per timestep\n"
                       "(rising & spiking in-phase = phase-lock achieved)")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Fraction watching target band")
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    print(f"Saved phase-lock demo figure to {save_path}")
    return save_path


if __name__ == "__main__":
    phase_lock_plot()
