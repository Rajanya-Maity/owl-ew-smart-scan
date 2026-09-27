"""
Smart Scan Strategy for Electronic Warfare — Main Pipeline
SIH 2026 · PS 26055 · DRDO
Author credit: R. Maity

Runs Steps 1-9 end to end and saves comparison outputs to ./results/:
  - open-loop baseline (Step 3)
  - epsilon-greedy & UCB1 MAB schedulers (Step 4)
  - harder simulator with periodic + agile emitters (Step 5)
  - (optional) trained DQN agent, if a model exists or --train is passed (Step 7)
  - all figures of merit for every scheduler (Step 8)
  - periodic phase-lock demo figure (Step 9)

Usage:
  python main.py                 # quick run: open-loop, MAB (simple + hard world)
  python main.py --train_dqn     # also trains a DQN agent (takes a few minutes)
  python main.py --demo          # also generates the periodic phase-lock demo figure
"""
import argparse
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env.spectrum_sim import SpectrumSimulator
from schedulers.open_loop import run_open_loop
from schedulers.mab import run_epsilon_greedy, run_ucb1, run_thompson_sampling
from schedulers.whittle import run_whittle_index
from metrics.metrics import compute_metrics, metrics_table

RESULTS_DIR = "results"


def heatmap_figure(sim: SpectrumSimulator, save_path):
    gt = sim.total_activity_heatmap()
    plt.figure(figsize=(10, 4))
    plt.imshow(gt, aspect="auto", cmap="hot", interpolation="nearest")
    plt.colorbar(label="Transmitting (1) / Idle (0)")
    plt.title(f"Ground-truth spectrum activity ({sim.mode} world, hidden from scheduler)")
    plt.xlabel("Timestep")
    plt.ylabel("Band")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()


def comparison_bar_chart(results_dict, save_path):
    metrics_to_plot = [
        "Pd (Probability of Detection)",
        "Miss Rate on Watched Bands",
        "Sensitivity",
        "Avg Intercept Rate (hits/timestep)",
    ]
    names = list(results_dict.keys())
    colors = ["#888888", "#6f9ceb", "#4fa38a", "#c17a91", "#d99a4e", "#9b7fd4", "#5aa5b8"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(4.2 * len(metrics_to_plot), 4.6))
    for ax, m in zip(axes, metrics_to_plot):
        vals = [results_dict[n][m] for n in names]
        ax.bar(names, vals, color=colors[: len(names)])
        ax.set_title(m, fontsize=9)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    fig.subplots_adjust(bottom=0.32, wspace=0.35)
    plt.savefig(save_path, dpi=130)
    plt.close()


def run_world(mode, n_bands, n_timesteps, seed, train_dqn=False, dqn_timesteps=200000):
    print(f"\n=== Running world: mode='{mode}', n_bands={n_bands}, n_timesteps={n_timesteps} ===")
    results = {}

    # --- Open loop baseline ---
    sim = SpectrumSimulator(n_bands=n_bands, n_timesteps=n_timesteps, bands_per_step=1,
                             mode=mode, seed=seed)
    log = run_open_loop(sim)
    results["Open-Loop"] = compute_metrics(log, sim)
    heatmap_figure(sim, os.path.join(RESULTS_DIR, f"heatmap_{mode}.png"))

    # --- Epsilon-Greedy ---
    sim = SpectrumSimulator(n_bands=n_bands, n_timesteps=n_timesteps, bands_per_step=1,
                             mode=mode, seed=seed)
    log = run_epsilon_greedy(sim, epsilon=0.1, seed=seed)
    results["Epsilon-Greedy"] = compute_metrics(log, sim)

    # --- UCB1 ---
    sim = SpectrumSimulator(n_bands=n_bands, n_timesteps=n_timesteps, bands_per_step=1,
                             mode=mode, seed=seed)
    log = run_ucb1(sim, c=2.0, seed=seed)
    results["UCB1"] = compute_metrics(log, sim)

    # --- Thompson Sampling (stretch, Step 11.4) ---
    sim = SpectrumSimulator(n_bands=n_bands, n_timesteps=n_timesteps, bands_per_step=1,
                             mode=mode, seed=seed)
    log = run_thompson_sampling(sim, seed=seed)
    results["Thompson Sampling"] = compute_metrics(log, sim)

    # --- Whittle-Index (belief-tracking, periodicity-aware) ---
    sim = SpectrumSimulator(n_bands=n_bands, n_timesteps=n_timesteps, bands_per_step=1,
                             mode=mode, seed=seed)
    log = run_whittle_index(sim, seed=seed)
    results["Whittle-Index"] = compute_metrics(log, sim)

    # --- DQN (Step 7), only for the 'hard' world ---
    # A trained model is required to fairly demonstrate the ML-based scheduler
    # this project is centered on, so one is trained automatically here if it
    # doesn't already exist, rather than being an opt-in extra step. Training
    # itself always uses 'domain_random' (see env/spectrum_sim.py) regardless
    # of which world it is later evaluated on, since that is what gives the
    # policy exposure to many different spectrum layouts instead of one.
    model_path = "models/dqn_smart_scan.zip"
    if mode == "hard":
        need_training = train_dqn or not os.path.exists(model_path)
        try:
            from training.train_dqn import train, run_trained_agent
            if need_training:
                print("Training DQN agent across randomized spectrum compositions "
                      "(this takes a couple of minutes)...")
                train(total_timesteps=dqn_timesteps, n_bands=n_bands,
                      n_timesteps=n_timesteps, mode="domain_random", model_path=model_path, seed=seed)
            log, dqn_sim = run_trained_agent(
                model_path,
                dict(n_bands=n_bands, n_timesteps=n_timesteps, mode=mode, seed=seed),
            )
            results["DQN (RL)"] = compute_metrics(log, dqn_sim)
        except Exception as e:
            print(f"[warn] DQN step skipped ({e}). Install stable-baselines3 & torch, "
                  f"or run with --train_dqn.")

    print(metrics_table(results))
    with open(os.path.join(RESULTS_DIR, f"metrics_{mode}.json"), "w") as f:
        json.dump(results, f, indent=2)
    comparison_bar_chart(results, os.path.join(RESULTS_DIR, f"comparison_{mode}.png"))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_bands", type=int, default=10)
    parser.add_argument("--n_timesteps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_dqn", action="store_true",
                         help="Train a fresh DQN agent (Step 7). Requires stable-baselines3 + torch.")
    parser.add_argument("--dqn_timesteps", type=int, default=200000)
    parser.add_argument("--demo", action="store_true",
                         help="Also generate the periodic phase-lock demo figure (Step 9).")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Smart Scan Strategy for Electronic Warfare — Full Pipeline")
    print("SIH 2026 | PS 26055 | DRDO | Credit: R. Maity")

    simple_results = run_world("simple", args.n_bands, args.n_timesteps, args.seed)
    hard_results = run_world("hard", args.n_bands, args.n_timesteps, args.seed,
                              train_dqn=args.train_dqn, dqn_timesteps=args.dqn_timesteps)

    if args.demo:
        from demo.periodic_lock_demo import phase_lock_plot
        phase_lock_plot(os.path.join(RESULTS_DIR, "periodic_lock.png"))

    print(f"\nAll results saved in ./{RESULTS_DIR}/")
    print("Run `streamlit run dashboard/app.py` for the live visual dashboard (Step 12).")


if __name__ == "__main__":
    main()
