# Decoding Reach Movements from Motor Cortex Neurons

Predicting a monkey's hand velocity from the activity of ~180 neurons in motor
cortex, and comparing the decoder used in early human BCI trials (Kalman
filter) against deep learning.

**Data:** MC_Maze from the [Neural Latents Benchmark](https://neurallatents.github.io/),
hosted on the DANDI Archive (dandiset 000128). Monkey J reached through a maze
to targets while neurons in primary motor and dorsal premotor cortex were recorded.

> This README is a working template. Replace it with your final write-up on Day 7.

---

## Setup

**1. Python environment** (Python 3.11 or newer)

```bash
cd neural-decoding
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Git**

```bash
git init
git add .
git commit -m "Project setup and Day 1 exploration code"
```

Then create an empty repo on GitHub (no README) and follow its instructions to
push. Use a branch per day so each step stays reviewable, for example
`git checkout -b day2-preprocessing`, then merge into main when the day's work runs.

**3. Download the data** (~700 MB, a few minutes)

```bash
dandi download DANDI:000128 -o data
```

You should end up with `data/000128/sub-Jenkins/...desc-train_behavior+ecephys.nwb`
(the file with behavior) and a much smaller test file we won't use.

**4. Check the file layout**

```bash
python src/inspect_nwb.py
```

This prints the neurons, time series and trial columns. If the loader in
`src/data.py` ever complains about a missing field, this tells you its real name.

**5. Run Day 1**

```bash
python notebooks/01_explore.py
```

Or open the file in VS Code and run it cell by cell. Figures land in `figures/`.

---

## Project structure

```
src/data.py            load NWB file, bin spikes, align trials, label directions
src/plots.py           raster, PSTH, population heatmap, trajectories
src/inspect_nwb.py     print what's inside the NWB file
notebooks/01_explore.py  Day 1 exploration
figures/               saved plots
data/                  raw data (gitignored)
```

## Plan

- [ ] **Day 1, explore:** raster, PSTH, population heatmap, hand trajectories
- [ ] **Day 2, preprocess:** bin spikes (20 ms), add spike history, split **by trial** into train/val/test
- [ ] **Day 3, classic decoders:** Wiener filter and Kalman filter, R² on hand velocity
- [ ] **Day 4, deep learning:** GRU/LSTM in PyTorch on the same splits
- [ ] **Day 5, analysis:** neuron-dropping curve, training-data curve, lag sweep
- [ ] **Day 6, visuals:** predicted vs. actual reaches for all decoders
- [ ] **Day 7, write-up:** results table, figures, limitations, demo video

## Notes and findings

*(Answer the questions at the end of `01_explore.py` here.)*

## Limitations

- One monkey, one task
- Offline decoding: the monkey never saw the decoder's output (no closed loop)
