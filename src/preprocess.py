"""Turn raw spikes and hand movement into training data for decoders.

The idea in one picture, for each reach (trial):

    time bins (20 ms each)  ->  t0  t1  t2  ...  t34
    INPUT  X[t]:  spike counts of all 182 neurons in bin t AND the 4 bins before it
    OUTPUT y[t]:  hand velocity (x, y) at bin t

We use a window from 250 ms before movement onset to 450 ms after it
(the same window the Neural Latents Benchmark uses), so each reach gives
35 time bins.

Why include the previous bins ("spike history")? Motor cortex fires ~100-200 ms
BEFORE the hand moves (you saw this in the PSTH), so recent activity carries
information about what the hand is doing now.
"""
import numpy as np

from data import bin_spikes, sample_at

BIN_SIZE = 0.02   # 20 ms bins
PRE = 0.25        # seconds before movement onset
POST = 0.45       # seconds after movement onset
N_HIST = 5        # current bin + 4 previous bins = 100 ms of history


def build_trials(session, bin_size=BIN_SIZE, pre=PRE, post=POST, n_hist=N_HIST):
    """Bin every trial and line it up with hand velocity.

    Returns a dict with:
      counts : (n_trials, n_bins + n_hist - 1, n_neurons) spike counts. The first
               n_hist - 1 bins are extra "history" bins before the window starts.
      vel    : (n_trials, n_bins, 2) hand velocity at each bin center
      pos    : (n_trials, n_bins, 2) hand position at each bin center
      t_rel  : (n_bins,) bin centers relative to movement onset (seconds)
    """
    pad = n_hist - 1
    counts, vel, pos = [], [], []
    for onset in session.trials["move_onset"]:
        start = onset - pre - pad * bin_size
        c, centers = bin_spikes(session.spike_times, start, onset + post, bin_size)
        t = centers[pad:]                       # bins inside the decoding window
        counts.append(c)
        vel.append(sample_at(t, session.hand_t, session.hand_vel))
        pos.append(sample_at(t, session.hand_t, session.hand_pos))
    counts, vel, pos = np.stack(counts), np.stack(vel), np.stack(pos)

    # Drop any trial whose hand data has gaps (NaNs), so they can't break training.
    ok = ~np.isnan(vel).any(axis=(1, 2)) & ~np.isnan(pos).any(axis=(1, 2))
    if (~ok).any():
        print(f"Dropping {(~ok).sum()} trials with missing hand data")

    return {
        "counts": counts[ok].astype(np.int16),
        "vel": vel[ok].astype(np.float32),
        "pos": pos[ok].astype(np.float32),
        "t_rel": (t - onset).astype(np.float32),
        "bin_size": bin_size,
        "n_hist": n_hist,
    }


def add_history(counts, n_hist):
    """Stack each bin with the n_hist - 1 bins before it.

    counts: (n_trials, n_bins + n_hist - 1, n_neurons)
    returns (n_trials, n_bins, n_hist * n_neurons)

    Feature order at bin t: [bin t, bin t-1, ..., bin t-(n_hist-1)], each a block
    of n_neurons values.
    """
    pad = n_hist - 1
    n_bins = counts.shape[1] - pad
    lagged = [counts[:, pad - k: pad - k + n_bins, :] for k in range(n_hist)]
    return np.concatenate(lagged, axis=2)


def split_trials(n_trials, frac_train=0.7, frac_val=0.15, seed=0):
    """Randomly split TRIAL indices (not time bins) into train/val/test.

    Every bin from a given reach lands in the same split. If we split bins
    randomly instead, a test bin's neighbors (20 ms away, nearly identical)
    would be in the training set, and the score would be inflated.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_trials)
    n_tr = int(frac_train * n_trials)
    n_va = int(frac_val * n_trials)
    return {
        "train": np.sort(idx[:n_tr]),
        "val": np.sort(idx[n_tr:n_tr + n_va]),
        "test": np.sort(idx[n_tr + n_va:]),
    }


def save(path, trials, splits):
    np.savez_compressed(path, **trials, **{f"{k}_idx": v for k, v in splits.items()})


def load(path):
    """Load processed data saved by notebooks/02_preprocess.py."""
    d = dict(np.load(path))
    d["bin_size"] = float(d["bin_size"])
    d["n_hist"] = int(d["n_hist"])
    return d
