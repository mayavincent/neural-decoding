# %% [markdown]
# # Day 1: Explore the MC_Maze dataset
#
# A monkey (Jenkins) made reaches through a maze while ~180 neurons in motor
# cortex were recorded. Today's goal: understand what one trial looks like.
#
# Run this file cell by cell in VS Code (each `# %%` is a cell, click
# "Run Cell") or all at once with `python notebooks/01_explore.py` from the
# project root. Figures are saved to figures/.

# %% Setup
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np

from data import aligned_rates, find_train_file, load_session, reach_directions
from plots import (plot_population_heatmap, plot_psth, plot_raster,
                   plot_trajectories)

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# %% Load the data (takes ~30 s the first time)
session = load_session(find_train_file(ROOT / "data"))

print(f"Neurons:       {session.n_neurons}")
print(f"Trials:        {len(session.trials)}")
print(f"Session length {session.hand_t[-1] - session.hand_t[0]:.0f} s")
print(f"Behavior rate  {1 / np.median(np.diff(session.hand_t)):.0f} Hz")
rates_overall = [len(st) / (st[-1] - st[0]) for st in session.spike_times if len(st) > 1]
print(f"Mean firing rate across neurons: {np.mean(rates_overall):.1f} spikes/s")

# %% Label reaches by direction and compute aligned firing rates
angles, labels = reach_directions(session)
rates, t_rel = aligned_rates(session, before=0.3, after=0.5, bin_size=0.02)
print("Trials per direction:", np.bincount(labels, minlength=8))
print("Aligned rates shape (trials, bins, neurons):", rates.shape)

# %% Figure 1: spike raster for one trial
fig, ax = plt.subplots(figsize=(8, 6))
plot_raster(session.spike_times, session.trials["move_onset"].iloc[0], ax=ax)
fig.tight_layout()
fig.savefig(FIG_DIR / "01_raster.png", dpi=150)

# %% Figure 2: PSTH for the most direction-tuned neuron
# "Tuned" = its average rate around movement differs most across directions.
window = (t_rel > -0.1) & (t_rel < 0.3)
per_dir = np.stack([rates[labels == d][:, window].mean(axis=(0, 1))
                    for d in range(8) if (labels == d).any()])
best = int(np.argmax(per_dir.std(axis=0)))
print(f"Most direction-tuned neuron: {best}")

fig, ax = plt.subplots(figsize=(8, 5))
plot_psth(rates, t_rel, labels, neuron=best, ax=ax)
fig.tight_layout()
fig.savefig(FIG_DIR / "02_psth.png", dpi=150)

# %% Figure 3: the whole population at once
fig, ax = plt.subplots(figsize=(8, 6))
plot_population_heatmap(rates, t_rel, ax=ax)
fig.tight_layout()
fig.savefig(FIG_DIR / "03_population.png", dpi=150)

# %% Figure 4: hand trajectories
fig, ax = plt.subplots(figsize=(6, 6))
plot_trajectories(session, labels, ax=ax)
fig.tight_layout()
fig.savefig(FIG_DIR / "04_trajectories.png", dpi=150)

print(f"Saved figures to {FIG_DIR}")

# %% [markdown]
# ## Questions to answer before Day 2 (write your answers in the README)
# 1. In the raster, does activity change before or after movement onset?
#    (Motor cortex should lead movement by ~100 ms. This sets your decoder's lag.)
# 2. In the PSTH, does the neuron fire more for some directions than others?
#    That's "directional tuning," the basis of every movement decoder.
# 3. In the heatmap, do neurons peak at different times? If the population
#    tiles time, a decoder can track the reach as it unfolds.
