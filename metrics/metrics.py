"""
Figures of merit for interception performance, plus a one-step prediction
scoring module — Step 8 (and its follow-up correction).

`log` is the list of per-timestep info dicts produced by any scheduler run
against a SpectrumSimulator (see schedulers/*.py). `sim` is the simulator
instance itself (post-run), whose `ground_truth` and `obs_log` hold the
full truth / observation tables.

A note on two metrics that are easy to get wrong (and were corrected here
after review):

- "Probability of False Alarm" in the classic sense needs a detector with
  its own decision threshold, independent of whether a transmission was
  truly present — something like noise crossing a threshold. This
  simulation does not model receiver noise or a detection threshold; a
  band the scheduler looked at and found idle is a *missed opportunity*,
  not necessarily a textbook false alarm. That quantity is reported below
  as "Miss Rate on Watched Bands", with the earlier "Pfa" framing dropped
  so the number is not mistaken for something it is not.

- "Percentage of correct predictions" is reported here as an actual
  one-step-ahead prediction score (see `compute_prediction_metrics`),
  rather than being a re-statement of the detection probability. A simple
  online per-band frequency estimate is used to predict, at every
  timestep, whether each band will be active next — and that prediction
  is then checked against what the simulator's ground truth turned out to
  be. This is an evaluation-only comparison; the predictor never receives
  ground truth as an input, only as the answer key used afterwards.
"""
import numpy as np


def compute_metrics(log, sim, reward_hit=1.0, reward_miss=-0.05):
    n_bands, n_t = sim.ground_truth.shape

    total_watched = 0
    hits = 0                # watched band was actually active -> True Positive
    misses_on_watch = 0     # watched band was inactive -> a look that found nothing
    total_active_slots = int(sim.ground_truth.sum())
    detected_active_slots = 0
    reward_total = 0.0
    intercept_times = []    # time between successive hits, for intercept-rate / timing stats
    last_hit_t = None

    # per-band-per-timestep confusion, restricted to what the scheduler actually watched
    for info in log:
        t = info["t"]
        for band, r in info["results"].items():
            total_watched += 1
            reward_total += reward_hit if r == 1 else reward_miss
            if r == 1:
                hits += 1
                detected_active_slots += 1
                if last_hit_t is not None:
                    intercept_times.append(t - last_hit_t)
                last_hit_t = t
            else:
                misses_on_watch += 1

    pd = hits / total_watched if total_watched else 0.0          # detection rate given a look
    miss_rate_on_watch = misses_on_watch / total_watched if total_watched else 0.0
    sensitivity = detected_active_slots / total_active_slots if total_active_slots else 0.0
    avg_intercept_rate = hits / n_t                              # hits per timestep across episode
    avg_reward = reward_total / total_watched if total_watched else 0.0

    if len(intercept_times) > 0:
        avg_intercept_time = float(np.mean(intercept_times))
        # "error" vs the theoretically expected gap if activity were evenly spread
        expected_gap = n_t / max(total_active_slots, 1)
        avg_intercept_time_error = abs(avg_intercept_time - expected_gap)
    else:
        avg_intercept_time = float("nan")
        avg_intercept_time_error = float("nan")

    prediction_accuracy = compute_prediction_metrics(sim)["Prediction Accuracy"]

    return {
        "Pd (Probability of Detection)": round(pd, 4),
        "Miss Rate on Watched Bands": round(miss_rate_on_watch, 4),
        "Sensitivity": round(sensitivity, 4),
        "Avg Intercept Rate (hits/timestep)": round(avg_intercept_rate, 4),
        "Avg Reward": round(avg_reward, 4),
        "One-Step Prediction Accuracy": round(prediction_accuracy, 2),
        "Avg Intercept Time (steps)": round(avg_intercept_time, 3) if avg_intercept_time == avg_intercept_time else None,
        "Avg Intercept Time Error": round(avg_intercept_time_error, 3) if avg_intercept_time_error == avg_intercept_time_error else None,
        "Total Hits": hits,
        "Total Watched Slots": total_watched,
        "Total Active Slots (ground truth)": total_active_slots,
    }


def compute_prediction_metrics(sim, smoothing=0.15):
    """A genuine one-step-ahead prediction score, evaluated separately from
    detection performance.

    At every timestep t, a running per-band activity-frequency estimate is
    updated from what has actually been observed so far (only the bands the
    scheduler looked at contribute new information; unobserved bands keep
    their last estimate). That estimate is then used to predict whether
    each band will be active at t+1 (threshold 0.5), and the prediction is
    scored against the simulator's true ground truth at t+1. Ground truth
    is used only to grade the prediction after the fact, never to make it.
    """
    n_bands, n_t = sim.ground_truth.shape
    est = np.full(n_bands, 0.1)  # mild prior: most bands are usually quiet

    tp = fp = fn = tn = 0
    for t in range(n_t - 1):
        predicted_active = est > 0.5
        actual_next = sim.ground_truth[:, t + 1] == 1
        tp += int(np.sum(predicted_active & actual_next))
        fp += int(np.sum(predicted_active & ~actual_next))
        fn += int(np.sum(~predicted_active & actual_next))
        tn += int(np.sum(~predicted_active & ~actual_next))

        # update the running estimate using only what was actually observed at t
        observed_mask = sim.obs_log[:, t] >= 0
        observed_vals = sim.obs_log[:, t]
        est[observed_mask] = (
            (1 - smoothing) * est[observed_mask] + smoothing * observed_vals[observed_mask]
        )

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total * 100 if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "Prediction Accuracy": accuracy,
        "Prediction Precision": round(precision, 4),
        "Prediction Recall": round(recall, 4),
        "Prediction F1": round(f1, 4),
    }


def metrics_table(results_dict):
    """results_dict: {scheduler_name: metrics_dict}. Returns a pandas-free text table."""
    names = list(results_dict.keys())
    keys = list(next(iter(results_dict.values())).keys())
    col_w = max(len(k) for k in keys) + 2
    header = "Metric".ljust(col_w) + "".join(n.rjust(18) for n in names)
    lines = [header, "-" * len(header)]
    for k in keys:
        row = k.ljust(col_w)
        for n in names:
            v = results_dict[n][k]
            row += str(v).rjust(18)
        lines.append(row)
    return "\n".join(lines)

