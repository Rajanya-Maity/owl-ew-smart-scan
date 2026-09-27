"""
O.W.L. EW — Omni-Frequency Wideband Listener for Electronic Warfare
A machine-learning based scan scheduler for an electronic support receiver.

Run with:  streamlit run dashboard/app.py   (from the project root: smart_scan_ew/)

The page is organised as three tabs that mirror the actual data flow, so
nothing that happens is hidden from view:

  Tab 1 — Configure the Simulated Spectrum
          The hidden RF environment is built here: bands, and whichever mix
          of random, periodic, frequency-agile, or rotating-scan emitters is
          chosen. Nothing is precomputed — whatever is configured on this
          tab is exactly what gets scanned in the next two.

  Tab 2 — Live Demonstration of the Scheduler Processing the Simulated Spectrum
          A scheduler (or several, side by side) is run against the spectrum
          from Tab 1, one timestep at a time, so its behaviour can be
          watched rather than only read about in a table.

  Tab 3 — Performance Benchmark & Metrics Comparison
          Every scheduler is scored against the same spectrum on the figures
          of merit relevant to an electronic support receiver, and a
          rotating-scan phase-lock experiment can be generated on demand.
"""
import base64
import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.spectrum_sim import SpectrumSimulator
from schedulers.open_loop import run_open_loop
from schedulers.mab import run_epsilon_greedy, run_ucb1, run_thompson_sampling
from schedulers.whittle import run_whittle_index
from metrics.metrics import compute_metrics
from novelty.explainability import mab_band_confidence
from demo.periodic_lock_demo import run_ucb1_repeatedly

st.set_page_config(page_title="O.W.L. EW", layout="wide")

DQN_PATH = "models/dqn_smart_scan.zip"
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "owl_logo.png")

EMITTER_COLORS = {
    "bernoulli": "#6f9ceb",       # muted steel blue — background, mostly-random chatter
    "periodic": "#4fa38a",        # muted teal green — a clock-like, repeating source
    "agile": "#d99a4e",           # muted amber — jumps between bands
    "periodic_scan": "#c17a91",   # muted rose — a rotating scanner sweeping in a fixed order
}
EMITTER_LABELS = {
    "bernoulli": "Random / Bernoulli",
    "periodic": "Periodic",
    "agile": "Frequency-Agile",
    "periodic_scan": "Rotating Scan",
}

# ---------------------------------------------------------------------
# Page chrome: font, colour palette, and the O.W.L. EW banner
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Cascadia Code SemiBold is used where it is installed on the viewer's
       machine; on systems without it, the stack quietly falls back to a
       similar monospace face rather than breaking the layout. */
    .owl-title {
        font-family: "Cascadia Code SemiBold", "Cascadia Code", "Consolas", monospace;
        font-weight: 600;
        font-size: 2.6rem;
        color: #d8b93c;   /* canary / chrome yellow, kept on the muted side */
        margin-bottom: 0rem;
        letter-spacing: 0.5px;
    }
    .owl-subtitle {
        font-family: "Cascadia Code SemiBold", "Cascadia Code", "Consolas", monospace;
        font-style: italic;
        color: #b9b4c4;   /* greyish-white */
        font-size: 1.02rem;
        margin-top: -0.3rem;
    }
    .emitter-tag {
        display: inline-block;
        padding: 1px 9px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

header_col1, header_col2 = st.columns([1, 9])
with header_col1:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=88)
with header_col2:
    st.markdown('<div class="owl-title">O.W.L. EW</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="owl-subtitle">Omni-Frequency Wideband Listener for Electronic Warfare</div>',
        unsafe_allow_html=True,
    )

st.write("")

# =========================================================================
# SHARED STATE — the spectrum configuration built in Tab 1 is used by Tabs 2 & 3
# =========================================================================
if "emitters" not in st.session_state:
    st.session_state["emitters"] = [
        {"type": "bernoulli", "band": 0, "p_on": 0.10},
        {"type": "bernoulli", "band": 1, "p_on": 0.20},
        {"type": "bernoulli", "band": 2, "p_on": 0.05},
        {"type": "bernoulli", "band": 3, "p_on": 0.15},
        {"type": "periodic", "band": 5, "period": 20, "on_len": 5, "phase": 0},
        {"type": "agile", "bands": [1, 4, 7, 8], "dwell": 4, "on_prob": 0.85, "random_hop": False},
    ]
if "n_bands" not in st.session_state:
    st.session_state["n_bands"] = 10
if "n_timesteps" not in st.session_state:
    st.session_state["n_timesteps"] = 200
if "sim_seed" not in st.session_state:
    st.session_state["sim_seed"] = 42


def build_sim_from_config():
    return SpectrumSimulator(
        n_bands=st.session_state["n_bands"],
        n_timesteps=st.session_state["n_timesteps"],
        bands_per_step=1,
        mode="custom",
        seed=st.session_state["sim_seed"],
        custom_emitters=st.session_state["emitters"],
    )


def dqn_model_matches_current_bands():
    """The saved DQN policy was trained for a fixed observation size (based
    on the band count it was trained with). If the band count has since been
    changed on Tab 1, the policy cannot be applied directly — this checks
    for that before anything is run against it, rather than letting it fail
    partway through."""
    if not os.path.exists(DQN_PATH):
        return False, "no trained model file was found"
    try:
        from stable_baselines3 import DQN
        model = DQN.load(DQN_PATH)
        expected_bands = model.observation_space.shape[0] // 11  # 10 history + 1 recency channel
        if expected_bands != st.session_state["n_bands"]:
            return False, (f"the saved model was trained for {expected_bands} bands, "
                            f"but {st.session_state['n_bands']} are currently configured")
        return True, ""
    except Exception as ex:
        return False, f"the model could not be loaded ({ex})"


def heatmap_fig(sim, title="Simulated spectrum activity (not visible to the scheduler)"):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.imshow(sim.ground_truth, aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Band")
    fig.tight_layout()
    return fig


tab_input, tab_process, tab_output = st.tabs(
    ["Configure the Simulated Spectrum", "Live Scheduler Demonstration", "Performance Benchmark & Metrics"]
)

# =========================================================================
# TAB 1 — CONFIGURE THE SIMULATED SPECTRUM
# =========================================================================
with tab_input:
    st.subheader("Configure the Simulated Spectrum")
    st.caption(
        "A hidden radio-frequency environment is assembled on this tab. The scheduler that is "
        "run in the following tabs is never given access to this configuration directly — only "
        "a hit or a miss is reported back for whichever band it chooses to look at, which mirrors "
        "the situation an electronic support receiver is assumed to be in."
    )

    c1, c2, c3 = st.columns(3)
    st.session_state["n_bands"] = c1.number_input(
        "Number of frequency bands", min_value=2, max_value=32,
        value=st.session_state["n_bands"])
    st.session_state["n_timesteps"] = c2.number_input(
        "Episode length, in timesteps", min_value=20, max_value=1000,
        value=st.session_state["n_timesteps"], step=10)
    st.session_state["sim_seed"] = c3.number_input(
        "Random seed", value=st.session_state["sim_seed"], step=1)

    st.markdown("#### Emitters present in this spectrum")
    st.caption(
        "Each entry below corresponds to one emitter. A colour is assigned per emitter type so "
        "that the mix of behaviours in the list can be told apart at a glance."
    )

    remove_idx = None
    for i, e in enumerate(st.session_state["emitters"]):
        color = EMITTER_COLORS.get(e["type"], "#888888")
        label = EMITTER_LABELS.get(e["type"], e["type"])
        st.markdown(
            f'<span class="emitter-tag" style="background-color:{color}22; '
            f'color:{color}; border: 1px solid {color}66;">{label}</span>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            cols = st.columns([1.3, 1, 1, 1, 1, 0.6])
            etype = cols[0].selectbox(
                "Type", ["bernoulli", "periodic", "agile", "periodic_scan"],
                index=["bernoulli", "periodic", "agile", "periodic_scan"].index(e["type"]),
                key=f"type_{i}",
            )
            e["type"] = etype
            if etype == "bernoulli":
                e["band"] = cols[1].number_input("Band", 0, st.session_state["n_bands"] - 1,
                                                   value=e.get("band", 0), key=f"band_{i}")
                e["p_on"] = cols[2].slider("Transmit probability", 0.01, 0.9, value=e.get("p_on", 0.1), key=f"pon_{i}")
            elif etype == "periodic":
                e["band"] = cols[1].number_input("Band", 0, st.session_state["n_bands"] - 1,
                                                   value=e.get("band", 0), key=f"band_{i}")
                e["period"] = cols[2].number_input("Period", 2, 200, value=e.get("period", 20), key=f"per_{i}")
                e["on_len"] = cols[3].number_input("Active length", 1, e["period"], value=min(e.get("on_len", 5), e["period"]), key=f"onl_{i}")
                e["phase"] = cols[4].number_input("Phase offset", 0, e["period"] - 1, value=e.get("phase", 0), key=f"ph_{i}")
            elif etype == "agile":
                default_bands = ",".join(str(b) for b in e.get("bands", [0, 1, 2]))
                bands_str = cols[1].text_input("Bands visited (comma-separated)", value=default_bands, key=f"bands_{i}")
                try:
                    e["bands"] = [int(x.strip()) for x in bands_str.split(",") if x.strip() != ""]
                except ValueError:
                    st.warning("The band list could not be parsed, so the previous value was kept.")
                e["dwell"] = cols[2].number_input("Dwell (steps per band)", 1, 50, value=e.get("dwell", 4), key=f"dwell_{i}")
                e["on_prob"] = cols[3].slider("Transmit probability while present", 0.1, 1.0, value=e.get("on_prob", 0.85), key=f"onp_{i}")
                e["random_hop"] = cols[4].checkbox("Shuffled order", value=e.get("random_hop", False), key=f"rh_{i}")
            elif etype == "periodic_scan":
                default_bands = ",".join(str(b) for b in e.get("bands", [0, 1, 2, 3]))
                bands_str = cols[1].text_input("Scan order (comma-separated)", value=default_bands, key=f"sbands_{i}")
                try:
                    e["bands"] = [int(x.strip()) for x in bands_str.split(",") if x.strip() != ""]
                except ValueError:
                    st.warning("The scan order could not be parsed, so the previous value was kept.")
                e["dwell"] = cols[2].number_input("Dwell per band", 1, 50, value=e.get("dwell", 5), key=f"sdwell_{i}")
                e["on_prob"] = cols[3].slider("Transmit probability while dwelling", 0.1, 1.0, value=e.get("on_prob", 0.9), key=f"sonp_{i}")
                cols[4].caption("Bands are visited in this exact order, repeatedly — unlike the shuffled agile type.")
            if cols[5].button("Remove", key=f"rm_{i}"):
                remove_idx = i

    if remove_idx is not None:
        st.session_state["emitters"].pop(remove_idx)
        st.rerun()

    add_cols = st.columns(4)
    if add_cols[0].button("Add a random emitter"):
        st.session_state["emitters"].append({"type": "bernoulli", "band": 0, "p_on": 0.1})
        st.rerun()
    if add_cols[1].button("Add a periodic emitter"):
        st.session_state["emitters"].append({"type": "periodic", "band": 0, "period": 20, "on_len": 5, "phase": 0})
        st.rerun()
    if add_cols[2].button("Add a frequency-agile emitter"):
        st.session_state["emitters"].append({"type": "agile", "bands": [0, 1, 2], "dwell": 4, "on_prob": 0.85, "random_hop": False})
        st.rerun()
    if add_cols[3].button("Add a rotating-scan emitter"):
        st.session_state["emitters"].append({"type": "periodic_scan", "bands": [0, 1, 2, 3], "dwell": 5, "on_prob": 0.9})
        st.rerun()

    st.markdown("---")
    if st.button("Build the spectrum", type="primary"):
        try:
            sim_preview = build_sim_from_config()
            st.session_state["preview_sim"] = sim_preview
        except Exception as ex:
            st.error(f"The spectrum could not be built: {ex}")

    if "preview_sim" in st.session_state:
        st.pyplot(heatmap_fig(st.session_state["preview_sim"]))
        total_active = int(st.session_state["preview_sim"].ground_truth.sum())
        st.caption(
            f"A total of {total_active} band-timestep slots are active across this episode — "
            f"this is the activity a scheduler is being asked to find, without this picture ever "
            f"being shown to it."
        )
    else:
        st.info("A preview of the ground truth is shown here once the spectrum above has been built.")

# =========================================================================
# TAB 2 — LIVE SCHEDULER DEMONSTRATION
# =========================================================================
with tab_process:
    st.subheader("Live Demonstration of the Scheduler Processing the Simulated Spectrum")
    st.caption(
        "The spectrum configured on the previous tab is used here without modification. The grey "
        "background in the figures below is the true activity, which the scheduler is not shown; "
        "only a green mark (a signal was caught) or a red mark (the band looked at was idle) is "
        "available to it, for whichever band it chose to observe."
    )

    mode_choice = st.radio("Viewing mode", ["Several schedulers, side by side", "One scheduler, in detail"], horizontal=True)

    if mode_choice.startswith("Several"):
        rc1, rc2 = st.columns(2)
        race_speed = rc1.slider("Playback speed, in steps per frame", 1, 15, 3, key="race_speed2")
        start_race = rc2.button("Start", type="primary")

        def build_race_sims():
            cfg = dict(n_bands=st.session_state["n_bands"], n_timesteps=st.session_state["n_timesteps"],
                       bands_per_step=1, mode="custom", seed=st.session_state["sim_seed"],
                       custom_emitters=st.session_state["emitters"])
            sims, logs = {}, {}
            sims["Open-Loop sweep"] = SpectrumSimulator(**cfg)
            logs["Open-Loop sweep"] = run_open_loop(sims["Open-Loop sweep"])

            sims["UCB1"] = SpectrumSimulator(**cfg)
            logs["UCB1"] = run_ucb1(sims["UCB1"], c=2.0, seed=st.session_state["sim_seed"])

            sims["Whittle-Index"] = SpectrumSimulator(**cfg)
            logs["Whittle-Index"] = run_whittle_index(sims["Whittle-Index"], seed=st.session_state["sim_seed"])

            dqn_ok, dqn_reason = dqn_model_matches_current_bands()
            if dqn_ok:
                from training.train_dqn import run_trained_agent
                dqn_log, dqn_sim = run_trained_agent(
                    DQN_PATH,
                    dict(n_bands=st.session_state["n_bands"], n_timesteps=st.session_state["n_timesteps"],
                         mode="custom", seed=st.session_state["sim_seed"],
                         custom_emitters=st.session_state["emitters"]))
                sims["DQN scheduler"] = dqn_sim
                logs["DQN scheduler"] = dqn_log
            else:
                st.caption(f"The DQN scheduler was left out of this run because {dqn_reason}.")
            return sims, logs

        def race_frame(sim, name, t_cutoff):
            gt = sim.ground_truth[:, : t_cutoff + 1]
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.imshow(gt, aspect="auto", cmap="Greys", alpha=0.45,
                      extent=[0, t_cutoff + 1, sim.n_bands, 0])
            watched = sim.obs_log[:, : t_cutoff + 1]
            ys, xs = np.where(watched >= 0)
            vals = watched[ys, xs]
            ax.scatter(xs[vals == 1], ys[vals == 1], c="#5f9d7a", s=14)
            ax.scatter(xs[vals == 0], ys[vals == 0], c="#c05f5f", s=8)
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("t", fontsize=8)
            ax.set_ylabel("band", fontsize=8)
            fig.tight_layout()
            return fig

        if start_race:
            sims, logs = build_race_sims()
            names = list(sims.keys())
            cols = st.columns(len(names))
            plot_slots = [c.empty() for c in cols]
            score_slots = [c.empty() for c in cols]
            max_len = max(len(l) for l in logs.values())

            for t in range(0, max_len, race_speed):
                for idx, name in enumerate(names):
                    log, sim = logs[name], sims[name]
                    cutoff = min(t, len(log) - 1)
                    plot_slots[idx].pyplot(race_frame(sim, name, cutoff))
                    hits = sum(list(info["results"].values())[0] for info in log[: cutoff + 1])
                    score_slots[idx].metric(f"{name} — hits so far", hits, f"of {cutoff + 1} looks")
                time.sleep(0.03)

            for idx, name in enumerate(names):
                plot_slots[idx].pyplot(race_frame(sims[name], name, len(logs[name]) - 1))

            st.success("The comparison run has finished.")
            final_cols = st.columns(len(names))
            for idx, name in enumerate(names):
                m = compute_metrics(logs[name], sims[name])
                with final_cols[idx]:
                    st.markdown(f"**{name}**")
                    st.metric("Total hits", m["Total Hits"])
                    st.metric("Probability of detection", m["Pd (Probability of Detection)"])
                    st.metric("Average intercept time", m["Avg Intercept Time (steps)"])
        else:
            st.info("Once started, each scheduler listed is run against the identical spectrum from the previous tab.")

    else:
        ec1, ec2, ec3 = st.columns(3)
        scheduler_name = ec1.selectbox(
            "Scheduler", ["Open-Loop sweep", "Epsilon-Greedy", "UCB1",
                          "Thompson Sampling", "Whittle-Index", "DQN scheduler"])
        speed = ec2.slider("Playback speed, in steps per refresh", 1, 20, 5)
        run_button = ec3.button("Run this episode", type="primary")

        dqn_ok, dqn_reason = dqn_model_matches_current_bands()
        if scheduler_name.startswith("DQN") and not dqn_ok:
            st.warning(f"The DQN scheduler is not available for this run because {dqn_reason}. "
                        f"UCB1 is shown instead below.")

        def get_log_and_sim():
            if scheduler_name.startswith("DQN") and dqn_ok:
                from training.train_dqn import run_trained_agent
                log, sim = run_trained_agent(
                    DQN_PATH,
                    dict(n_bands=st.session_state["n_bands"], n_timesteps=st.session_state["n_timesteps"],
                         mode="custom", seed=st.session_state["sim_seed"],
                         custom_emitters=st.session_state["emitters"]))
                return log, sim
            sim = build_sim_from_config()
            if scheduler_name.startswith("Open-Loop"):
                log = run_open_loop(sim)
            elif scheduler_name.startswith("Epsilon"):
                log = run_epsilon_greedy(sim, epsilon=0.1, seed=st.session_state["sim_seed"])
            elif scheduler_name.startswith("UCB1"):
                log = run_ucb1(sim, c=2.0, seed=st.session_state["sim_seed"])
            elif scheduler_name.startswith("Thompson"):
                log = run_thompson_sampling(sim, seed=st.session_state["sim_seed"])
            elif scheduler_name.startswith("Whittle"):
                log = run_whittle_index(sim, seed=st.session_state["sim_seed"])
            else:
                log = run_ucb1(sim, c=2.0, seed=st.session_state["sim_seed"])
            return log, sim

        if "proc_log" not in st.session_state or run_button:
            log, sim = get_log_and_sim()
            st.session_state["proc_log"] = log
            st.session_state["proc_sim"] = sim
            st.session_state["proc_name"] = scheduler_name

        log = st.session_state["proc_log"]
        sim = st.session_state["proc_sim"]
        name_used = st.session_state["proc_name"]

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("**Simulated activity compared against what was observed**")
            heat_slot = st.empty()
        with col2:
            st.markdown("**Running metrics**")
            metrics_slot = st.empty()
            st.markdown("**Confidence held in each band**")
            conf_slot = st.empty()
        progress = st.progress(0)
        play = st.button("Play the episode step by step")

        def draw_frame(t_cutoff):
            gt = sim.ground_truth[:, : t_cutoff + 1]
            watched_mask = (sim.obs_log[:, : t_cutoff + 1] >= 0).astype(float)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(gt, aspect="auto", cmap="Greys", alpha=0.5,
                      extent=[0, t_cutoff + 1, sim.n_bands, 0])
            ys, xs = np.where(watched_mask > 0)
            hit_vals = sim.obs_log[:, : t_cutoff + 1][ys, xs]
            ax.scatter(xs[hit_vals == 1], ys[hit_vals == 1], c="#5f9d7a", s=10, label="Signal caught")
            ax.scatter(xs[hit_vals == 0], ys[hit_vals == 0], c="#c05f5f", s=6, label="Band was idle")
            ax.set_xlabel("Timestep"); ax.set_ylabel("Band")
            ax.set_title(f"{name_used}")
            ax.legend(loc="upper right", fontsize=8)
            return fig

        def draw_metrics(t_cutoff):
            partial = log[: t_cutoff + 1]
            if not partial:
                return
            m = compute_metrics(partial, sim)
            df = pd.DataFrame(list(m.items()), columns=["Metric", "Value"])
            st.dataframe(df, hide_index=True, use_container_width=True)

        def draw_confidence(t_cutoff):
            counts = np.zeros(sim.n_bands); values = np.zeros(sim.n_bands)
            for info in log[: t_cutoff + 1]:
                for b, r in info["results"].items():
                    counts[b] += 1
                    values[b] += (r - values[b]) / counts[b]
            conf = mab_band_confidence(values, counts)
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(range(sim.n_bands), conf, color="#6f9ceb")
            ax.set_xlabel("Band"); ax.set_ylabel("Confidence")
            return fig

        if play:
            for t in range(0, len(log), speed):
                heat_slot.pyplot(draw_frame(t))
                with metrics_slot.container():
                    draw_metrics(t)
                conf_slot.pyplot(draw_confidence(t))
                progress.progress(min((t + 1) / len(log), 1.0))
                time.sleep(0.05)
        heat_slot.pyplot(draw_frame(len(log) - 1))
        with metrics_slot.container():
            draw_metrics(len(log) - 1)
        conf_slot.pyplot(draw_confidence(len(log) - 1))
        progress.progress(1.0)

# =========================================================================
# TAB 3 — PERFORMANCE BENCHMARK & METRICS
# =========================================================================
with tab_output:
    st.subheader("Performance Benchmark & Metrics Comparison")
    st.caption(
        "Every scheduler covered in this project is evaluated against the same spectrum "
        "configured earlier, and the figures of merit that matter for an electronic support "
        "receiver are computed and laid out side by side below."
    )

    if st.button("Compute the comparison", type="primary"):
        cfg = dict(n_bands=st.session_state["n_bands"], n_timesteps=st.session_state["n_timesteps"],
                   bands_per_step=1, mode="custom", seed=st.session_state["sim_seed"],
                   custom_emitters=st.session_state["emitters"])
        results = {}

        sim = SpectrumSimulator(**cfg)
        results["Open-Loop"] = compute_metrics(run_open_loop(sim), sim)

        sim = SpectrumSimulator(**cfg)
        results["Epsilon-Greedy"] = compute_metrics(
            run_epsilon_greedy(sim, epsilon=0.1, seed=st.session_state["sim_seed"]), sim)

        sim = SpectrumSimulator(**cfg)
        results["UCB1"] = compute_metrics(
            run_ucb1(sim, c=2.0, seed=st.session_state["sim_seed"]), sim)

        sim = SpectrumSimulator(**cfg)
        results["Thompson Sampling"] = compute_metrics(
            run_thompson_sampling(sim, seed=st.session_state["sim_seed"]), sim)

        sim = SpectrumSimulator(**cfg)
        results["Whittle-Index"] = compute_metrics(
            run_whittle_index(sim, seed=st.session_state["sim_seed"]), sim)

        dqn_ok, dqn_reason = dqn_model_matches_current_bands()
        if dqn_ok:
            from training.train_dqn import run_trained_agent
            dqn_log, dqn_sim = run_trained_agent(
                DQN_PATH,
                dict(n_bands=st.session_state["n_bands"], n_timesteps=st.session_state["n_timesteps"],
                     mode="custom", seed=st.session_state["sim_seed"],
                     custom_emitters=st.session_state["emitters"]))
            results["DQN"] = compute_metrics(dqn_log, dqn_sim)
        else:
            st.caption(f"The DQN scheduler was left out of this comparison because {dqn_reason}.")

        st.session_state["output_results"] = results

    if "output_results" in st.session_state:
        results = st.session_state["output_results"]
        df = pd.DataFrame(results).T
        st.dataframe(df, use_container_width=True)

        metrics_to_plot = ["Pd (Probability of Detection)", "Miss Rate on Watched Bands",
                            "Sensitivity", "Avg Intercept Rate (hits/timestep)"]
        names = list(results.keys())
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(4.2 * len(metrics_to_plot), 4.6))
        colors = ["#888888", "#6f9ceb", "#4fa38a", "#c17a91", "#d99a4e", "#9b7fd4", "#5aa5b8"]
        for ax, m in zip(axes, metrics_to_plot):
            vals = [results[n][m] for n in names]
            ax.bar(names, vals, color=colors[: len(names)])
            ax.set_title(m, fontsize=9)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
        fig.subplots_adjust(bottom=0.32, wspace=0.35)
        st.pyplot(fig)

        csv = df.to_csv().encode("utf-8")
        st.download_button("Download these metrics as a CSV file", csv, "owl_ew_metrics.csv", "text/csv")
    else:
        st.info("A table and comparison chart are shown here once the comparison above has been computed.")

    st.markdown("---")
    st.subheader("Rotating-Scan Phase-Lock Experiment")
    st.caption(
        "A scheduler's tendency to look at a rotating emitter's band around the moment it "
        "switches on is examined here, based on a number of repeated episodes. This is presented "
        "as an illustration of phase-locking onto a predictable emitter, rather than a complete "
        "model of intercepting an arbitrary scanning receiver."
    )
    plc1, plc2 = st.columns(2)
    demo_period = plc1.number_input("Period of the emitter under study", 5, 100, 20)
    demo_on_len = plc2.number_input("Length of its active window", 1, demo_period, min(5, demo_period))
    if st.button("Generate the phase-lock figure"):
        watch_history, target_band, period, on_len = run_ucb1_repeatedly(
            n_repeats=25, n_bands=6, n_timesteps=300, period=demo_period, on_len=demo_on_len)
        watched_target = (watch_history == target_band).astype(int)
        fig, axes = plt.subplots(2, 1, figsize=(9, 5.5))
        axes[0].imshow(watched_target, aspect="auto", cmap="Greens")
        axes[0].set_title(f"Attention paid to the periodic band across episodes (period={period}, active length={on_len})")
        axes[0].set_xlabel("Timestep"); axes[0].set_ylabel("Episode number")
        axes[1].plot(watched_target.mean(axis=0), color="#4fa38a")
        axes[1].set_title("Share of episodes in which the periodic band was being watched, per timestep")
        axes[1].set_xlabel("Timestep"); axes[1].set_ylabel("Fraction"); axes[1].set_ylim(0, 1)
        fig.tight_layout()
        st.pyplot(fig)

with st.expander("A note on some of the modelling choices made here"):
    st.write(
        "The receiver is assumed to look at one band per timestep, which stands in for an "
        "instantaneous bandwidth that is narrower than the full surveillance range being watched. "
        "Detection is treated as binary — a look either lands on an active band or it doesn't — "
        "so there is no explicit noise floor or detection threshold in this version, and the "
        "figures reported here should be read with that simplification in mind. The bandit-style "
        "schedulers (Epsilon-Greedy, UCB1, Thompson Sampling) keep a single running estimate per "
        "band, which suits a band that is simply busy or quiet on average, but does not by itself "
        "capture a band that is only busy at particular moments. The Whittle-Index scheduler is "
        "included to address that gap directly: it tracks a belief, per band, of how likely that "
        "band is to be active right now, lets that belief decay back toward the band's long-run "
        "rate the longer it goes unwatched, and boosts the belief sharply when a band's own history "
        "of hits shows a consistent rhythm. This is a practical approximation of the Whittle Index "
        "policy described by Whittle (1988) and given closed form for busy/idle channels by Liu and "
        "Zhao (2010) for exactly this class of problem, usually referred to as a restless "
        "multi-armed bandit, since a band's true state keeps evolving whether or not it is being "
        "observed. The DQN scheduler is kept as a separate comparison point, trained across many "
        "randomly composed spectra so that it is not simply memorizing one fixed layout, on the "
        "premise that some emitter behaviours may not fit the two-state model the Whittle-Index "
        "scheduler assumes as neatly."
    )
