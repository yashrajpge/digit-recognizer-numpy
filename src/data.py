"""Loading the Optdigits handwritten digits data, downloaded on first use.

The data is fetched directly from its owning source, the UCI Machine
Learning Repository, the first time it is needed, and cached locally in
data/ (which is git-ignored, so no dataset copy is redistributed with
this repo). See README for source, citation and terms.
"""

import os
import urllib.request

import numpy as np

BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits"
FILES = {"train": "optdigits.tra", "test": "optdigits.tes"}


def _ensure_cached(cache_dir="data"):
    os.makedirs(cache_dir, exist_ok=True)
    for name in FILES.values():
        path = os.path.join(cache_dir, name)
        if not os.path.exists(path):
            print(f"downloading {name} from UCI Machine Learning Repository...")
            urllib.request.urlretrieve(f"{BASE}/{name}", path)


def load_split(which="train", cache_dir="data"):
    """Return (X, y) for 'train' (3,823 samples) or 'test' (1,797)."""
    if which not in FILES:
        raise ValueError(f"unknown split {which!r}, expected one of {sorted(FILES)}")
    _ensure_cached(cache_dir)
    rows = np.loadtxt(os.path.join(cache_dir, FILES[which]), delimiter=",")
    X = rows[:, :64].astype(np.float64) / 16.0  # pixels are 0-16
    y = rows[:, 64].astype(int)
    return X, y
