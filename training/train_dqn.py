"""
Train the RL scheduler with DQN (Stable-Baselines3) — Step 7.

Run:  python -m training.train_dqn --timesteps 50000
Produces: models/dqn_smart_scan.zip
"""
import argparse
import os

from gymenv.ew_gym_env import SmartScanEnv


def train(total_timesteps=150000, n_bands=10, n_timesteps=200, mode="domain_random",
          model_path="models/dqn_smart_scan.zip", seed=42):
    from stable_baselines3 import DQN
    from stable_baselines3.common.env_util import make_vec_env

    # 'domain_random' (see env/spectrum_sim.py) re-rolls not just the numeric
    # parameters but the entire *composition* of the spectrum every episode —
    # how many emitters of each type, which bands, what periods/probabilities —
    # so the trained policy has to learn a general scanning strategy instead
    # of memorizing one fixed layout.
    env = SmartScanEnv(n_bands=n_bands, n_timesteps=n_timesteps, mode=mode, seed=seed,
                        randomize_each_episode=True)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=7e-4,
        buffer_size=100000,
        learning_starts=2000,
        batch_size=128,
        gamma=0.95,
        target_update_interval=750,
        train_freq=4,
        exploration_fraction=0.4,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=1,
        seed=seed,
    )
    model.learn(total_timesteps=total_timesteps)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Saved trained DQN model to {model_path}")
    return model_path


def run_trained_agent(model_path, sim_class_kwargs):
    """Run a saved DQN model against a fresh SpectrumSimulator, producing a log
    compatible with metrics.compute_metrics (same format as the other schedulers).

    sim_class_kwargs is passed straight to SmartScanEnv, so it can include
    mode='custom' with a custom_emitters list to evaluate the trained agent
    on a user-built spectrum, not just the fixed 'hard' world it was trained on.
    A fixed seed here means a fixed, reproducible evaluation episode —
    randomize_each_episode is turned off for evaluation on purpose.
    """
    from stable_baselines3 import DQN

    model = DQN.load(model_path)
    env = SmartScanEnv(randomize_each_episode=False, **sim_class_kwargs)
    obs, _ = env.reset()
    log = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        log.append(info)
    return log, env.sim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--n_bands", type=int, default=10)
    parser.add_argument("--episode_len", type=int, default=200)
    parser.add_argument("--mode", type=str, default="hard", choices=["simple", "hard"])
    args = parser.parse_args()
    train(total_timesteps=args.timesteps, n_bands=args.n_bands,
          n_timesteps=args.episode_len, mode=args.mode)
