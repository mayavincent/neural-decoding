"""Plotting helpers for Day 1 exploration."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

DIRECTION_NAMES = ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"]


def _direction_colors(n):
    return plt.cm.hsv(np.linspace(0, 1, n, endpoint=False))


def plot_raster(spike_times, onset, before=0.3, after=0.5, ax=None):
    """Raster plot: one row per neuron, one tick per spike, for a single trial."""
    ax = ax or plt.gca()
    rows = [st[(st >= onset - before) & (st < onset + after)] - onset for st in spike_times]
    ax.eventplot(rows, colors="black", linelengths=0.8, linewidths=0.6)
    ax.axvline(0, color="tab:red", lw=1, label="movement onset")
    ax.set(xlabel="time from movement onset (s)", ylabel="neuron",
           title="Spike raster, single trial", xlim=(-before, after), ylim=(-1, len(rows)))
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_psth(rates, t_rel, labels, neuron, smooth_bins=2, ax=None):
    """Peri-stimulus time histogram for one neuron, averaged per reach direction.

    rates: (n_trials, n_bins, n_neurons) firing rates aligned to movement onset.
    """
    ax = ax or plt.gca()
    n_dirs = len(DIRECTION_NAMES)
    for d, color in zip(range(n_dirs), _direction_colors(n_dirs)):
        mask = labels == d
        if mask.sum() == 0:
            continue
        mean = gaussian_filter1d(rates[mask, :, neuron].mean(axis=0), smooth_bins)
        ax.plot(t_rel, mean, color=color, label=f"{DIRECTION_NAMES[d]} (n={mask.sum()})")
    ax.axvline(0, color="gray", lw=1, ls="--")
    ax.set(xlabel="time from movement onset (s)", ylabel="firing rate (spikes/s)",
           title=f"Neuron {neuron}: average rate by reach direction")
    ax.legend(fontsize=7, ncol=2)
    return ax


def plot_population_heatmap(rates, t_rel, smooth_bins=2, ax=None):
    """Trial-averaged, z-scored rate of every neuron, sorted by time of peak."""
    ax = ax or plt.gca()
    mean = gaussian_filter1d(rates.mean(axis=0), smooth_bins, axis=0)  # (bins, neurons)
    z = (mean - mean.mean(axis=0)) / (mean.std(axis=0) + 1e-9)
    order = np.argsort(np.argmax(z, axis=0))
    im = ax.imshow(z[:, order].T, aspect="auto", cmap="viridis",
                   extent=[t_rel[0], t_rel[-1], len(order), 0])
    ax.axvline(0, color="white", lw=1, ls="--")
    ax.set(xlabel="time from movement onset (s)", ylabel="neuron (sorted by peak time)",
           title="All neurons, trial-averaged (z-scored)")
    plt.colorbar(im, ax=ax, label="z-score")
    return ax


def plot_trajectories(session, labels, window=0.6, max_trials=200, ax=None):
    """Hand paths from movement onset, colored by reach direction."""
    ax = ax or plt.gca()
    colors = _direction_colors(len(DIRECTION_NAMES))
    onsets = session.trials["move_onset"].to_numpy()
    for i in range(min(len(onsets), max_trials)):
        m = (session.hand_t >= onsets[i]) & (session.hand_t <= onsets[i] + window)
        xy = session.hand_pos[m]
        if len(xy) == 0:
            continue
        xy = xy - xy[0]  # start every reach at the origin
        ax.plot(xy[:, 0], xy[:, 1], color=colors[labels[i]], lw=0.8, alpha=0.7)
    ax.set(xlabel="x (relative)", ylabel="y (relative)", aspect="equal",
           title="Hand trajectories, colored by direction")
    return ax
