"""Load and preprocess the MC_Maze dataset (Neural Latents Benchmark).

The data comes as an NWB file (a standard HDF5-based format for neuroscience).
We read it directly with pynwb instead of the benchmark's nlb_tools package,
because nlb_tools pins an old pandas version that won't install on modern Python.

A Session holds everything we need:
  spike_times : list of 1D arrays, one per neuron, spike times in seconds
  hand_t      : 1D array of behavior timestamps (seconds)
  hand_pos    : (T, 2) hand x/y position
  hand_vel    : (T, 2) hand x/y velocity
  trials      : DataFrame with one row per reach, including 'move_onset'
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Session:
    spike_times: list
    hand_t: np.ndarray
    hand_pos: np.ndarray
    hand_vel: np.ndarray
    trials: pd.DataFrame

    @property
    def n_neurons(self):
        return len(self.spike_times)


# ---------------------------------------------------------------- loading

def find_train_file(data_dir="data"):
    """Find the training NWB file (the one with behavior) under data_dir."""
    matches = sorted(Path(data_dir).rglob("*train*.nwb"))
    if not matches:
        raise FileNotFoundError(
            f"No *train*.nwb file under {data_dir}/. See README step 3 to download it."
        )
    return matches[0]


def _iter_series(nwb):
    """Yield (name, series) for every time series in acquisition and processing,
    including ones nested inside containers like Position."""
    containers = [nwb.acquisition]
    containers += [mod.data_interfaces for mod in nwb.processing.values()]
    for container in containers:
        for name, obj in container.items():
            yield name, obj
            for attr in ("spatial_series", "time_series"):
                nested = getattr(obj, attr, None)
                if isinstance(nested, dict):
                    yield from nested.items()


def _series_times(series):
    if series.timestamps is not None:
        return np.asarray(series.timestamps[:])
    n = series.data.shape[0]
    return series.starting_time + np.arange(n) / series.rate


def _get_series(nwb, name):
    for key, series in _iter_series(nwb):
        if key == name and hasattr(series, "data"):
            return np.asarray(series.data[:]), _series_times(series)
    return None, None


def load_session(path):
    """Read spikes, hand kinematics and trial info from an MC_Maze NWB file."""
    from pynwb import NWBHDF5IO  # imported here so the rest works without pynwb

    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()

        units = nwb.units
        spike_times = [np.sort(np.asarray(units["spike_times"][i])) for i in range(len(units))]

        hand_pos, hand_t = _get_series(nwb, "hand_pos")
        hand_vel, vel_t = _get_series(nwb, "hand_vel")
        if hand_pos is None:
            names = [k for k, _ in _iter_series(nwb)]
            raise KeyError(f"No 'hand_pos' series found. Available: {names}. "
                           "Run `python src/inspect_nwb.py` to see the file layout.")
        if hand_vel is None:
            # Not stored: compute it from position (units per second).
            hand_vel = np.gradient(hand_pos, hand_t, axis=0)
        elif len(vel_t) != len(hand_t) or not np.allclose(vel_t, hand_t):
            hand_vel = np.column_stack(
                [np.interp(hand_t, vel_t, hand_vel[:, d]) for d in range(hand_vel.shape[1])]
            )

        trials = nwb.trials.to_dataframe()

    trials = _add_move_onset(trials)
    return Session(spike_times, hand_t, hand_pos[:, :2], hand_vel[:, :2], trials)


def _add_move_onset(trials):
    if "success" in trials.columns:
        # Keep only reaches the monkey completed correctly.
        trials = trials[trials["success"].astype(bool)]
    for col in ("move_onset_time", "move_begins_time", "go_cue_time"):
        if col in trials.columns:
            if col == "go_cue_time":
                print("Note: no movement-onset column; aligning to go cue instead.")
            trials = trials.assign(move_onset=trials[col])
            break
    else:
        raise KeyError(f"No movement-onset or go-cue column in trials: {list(trials.columns)}")
    return trials.dropna(subset=["move_onset"]).reset_index(drop=True)


# ---------------------------------------------------------- preprocessing

def bin_spikes(spike_times, t_start, t_stop, bin_size=0.02):
    """Count spikes per neuron in fixed-width bins.

    Returns (counts, bin_centers) where counts has shape (n_bins, n_neurons).
    """
    # Compute the bin count first so every trial gets the same number of bins
    # (np.arange with float steps can be off by one from rounding).
    n_bins = int(round((t_stop - t_start) / bin_size))
    edges = t_start + np.arange(n_bins + 1) * bin_size
    # searchsorted on sorted spike times counts spikes per bin without scanning
    # the whole recording each time (the session is ~2 hours long).
    counts = np.stack([np.diff(np.searchsorted(st, edges)) for st in spike_times], axis=1)
    centers = edges[:-1] + bin_size / 2
    return counts, centers


def sample_at(t_query, t, values):
    """Linearly interpolate a (T, D) signal at the query times."""
    return np.column_stack([np.interp(t_query, t, values[:, d]) for d in range(values.shape[1])])


def reach_directions(session, window=0.5, n_dirs=8):
    """Label each trial by reach direction.

    Uses the target's position when available (otherwise hand displacement from
    movement onset to `window` seconds later), and sorts the angle into n_dirs equal sectors (0 = rightward, counting
    counter-clockwise). Returns (angles_rad, direction_labels).
    """
    trials = session.trials
    if {"target_pos", "active_target"} <= set(trials.columns):
        # Maze reaches curve around barriers, so use where the target was
        # (the task is center-out: reaches start near the screen center).
        d = np.array([np.asarray(tp)[int(at)][:2] for tp, at
                      in zip(trials["target_pos"], trials["active_target"])], dtype=float)
    else:
        onset = trials["move_onset"].to_numpy()
        start = sample_at(onset, session.hand_t, session.hand_pos)
        end = sample_at(onset + window, session.hand_t, session.hand_pos)
        d = end - start
    angles = np.arctan2(d[:, 1], d[:, 0])
    sector = 2 * np.pi / n_dirs
    labels = np.floor(((angles + sector / 2) % (2 * np.pi)) / sector).astype(int)
    return angles, labels


def aligned_rates(session, before=0.3, after=0.5, bin_size=0.02):
    """Firing rates (spikes/s) for every trial, aligned to movement onset.

    Returns (rates, t_rel) with rates shaped (n_trials, n_bins, n_neurons) and
    t_rel the bin centers relative to movement onset.
    """
    out = []
    for onset in session.trials["move_onset"]:
        counts, centers = bin_spikes(session.spike_times, onset - before, onset + after, bin_size)
        out.append(counts / bin_size)
    t_rel = centers - onset
    return np.stack(out), t_rel
