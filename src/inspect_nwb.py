"""Print the layout of an NWB file: units, time series, and trial columns.

Usage:  python src/inspect_nwb.py [path/to/file.nwb]
Run this first if the loader complains that it can't find something.
"""
import sys

from pynwb import NWBHDF5IO

from data import _iter_series, find_train_file

path = sys.argv[1] if len(sys.argv) > 1 else find_train_file()
print(f"File: {path}\n")

with NWBHDF5IO(str(path), "r") as io:
    nwb = io.read()
    print(f"Session: {nwb.session_description}")
    print(f"Neurons (units): {len(nwb.units)}   columns: {list(nwb.units.colnames)}\n")

    print("Time series:")
    for name, obj in _iter_series(nwb):
        shape = getattr(getattr(obj, "data", None), "shape", None)
        print(f"  {name:<20} {type(obj).__name__:<16} shape={shape}")

    trials = nwb.trials.to_dataframe()
    print(f"\nTrials: {len(trials)}")
    print(f"Trial columns: {list(trials.columns)}")
    print(trials.head(3).T)
