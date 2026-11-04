# %% [markdown]
# # Preprocessing: turn spikes into training data
#
# Goal: for every reach, build
#   INPUT  = spike counts of all neurons in 20 ms bins (+ 100 ms of history)
#   OUTPUT = hand velocity at each bin
# then split reaches into train / validation / test sets and save the result.
#
# Run from the project root:  python notebooks/02_preprocess.py

# %% Setup
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np

from data import find_train_file, load_session
from preprocess import add_history, build_trials, save, split_trials

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
OUT = ROOT / "data" / "processed.npz"

# %% 1. Load and bin every reach
session = load_session(find_train_file(ROOT / "data"))
trials = build_trials(session)

counts, vel, t_rel = trials["counts"], trials["vel"], trials["t_rel"]
n_trials, _, n_neurons = counts.shape
n_bins = len(t_rel)
print(f"Reaches: {n_trials}   bins per reach: {n_bins} "
      f"(-250 to +450 ms around movement onset)   neurons: {n_neurons}")

# %% 2. How sparse are the spikes?
vals, freq = np.unique(counts, return_counts=True)
share = freq / freq.sum()
print("Spikes per neuron per 20 ms bin:",
      ", ".join(f"{v}: {s:.1%}" for v, s in zip(vals[:4], share[:4])))
# Most bins are 0: a single neuron rarely fires in any given 20 ms.

# %% 3. Add spike history -> the decoder's input features
X = add_history(counts, trials["n_hist"])
print(f"Input X shape (reaches, bins, features): {X.shape}  "
      f"= {trials['n_hist']} bins x {n_neurons} neurons per time step")
print(f"Output y shape (reaches, bins, x/y velocity): {vel.shape}")

# %% 4. Why we split by TRIAL, not by time bin
# Velocity barely changes from one 20 ms bin to the next:
r = np.corrcoef(vel[:, :-1, 0].ravel(), vel[:, 1:, 0].ravel())[0, 1]
print(f"Correlation of x-velocity between neighboring bins: {r:.3f}")
# If bins were split randomly, nearly every test bin would have a near-copy
# 20 ms away in the training set, and the score would look much better than it is.

splits = split_trials(n_trials)
print("Reaches per split:", {k: len(v) for k, v in splits.items()})
assert not set(splits["train"]) & set(splits["test"]), "train/test overlap!"

# %% 5. Figure: what ONE training example looks like
i = splits["train"][0]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True, layout="constrained",
                               gridspec_kw={"height_ratios": [2, 1]})
pad = trials["n_hist"] - 1
im = ax1.imshow(counts[i, pad:].T, aspect="auto", cmap="Greys", interpolation="nearest",
                extent=[t_rel[0] - 0.01, t_rel[-1] + 0.01, n_neurons, 0], vmin=0, vmax=3)
ax1.set(ylabel="neuron", title=f"One reach as training data (trial {i})\n"
        "INPUT: spike counts per 20 ms bin")
fig.colorbar(im, ax=[ax1, ax2], label="spikes in bin", pad=0.01, shrink=0.6, anchor=(0, 1))
ax2.plot(t_rel, vel[i, :, 0], label="x velocity")
ax2.plot(t_rel, vel[i, :, 1], label="y velocity")
ax2.axhline(0, color="gray", lw=0.5)
ax2.set(xlabel="time from movement onset (s)", ylabel="hand velocity",
        title="OUTPUT: what the decoder must predict")
ax2.legend(fontsize=8)
for ax in (ax1, ax2):
    ax.axvline(0, color="tab:red", lw=1, ls="--")
fig.savefig(FIG_DIR / "05_training_example.png", dpi=150)

# %% 6. Save for the decoders
save(OUT, trials, splits)
print(f"Saved processed data to {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")

# %% [markdown]
# ## Check your understanding
# 1. What is one "row" of training data? (One 20 ms moment of one reach: the
#    counts of 182 neurons over the last 100 ms -> the hand's velocity right now.)
# 2. Why include the previous 4 bins? (Motor cortex leads movement.)
# 3. Why split by reach instead of by bin? (Neighboring bins are near-copies.)
