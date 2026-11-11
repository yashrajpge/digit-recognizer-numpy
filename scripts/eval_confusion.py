"""Per-digit accuracy and confusion matrix for the trained digit model.

Loads models/digits_net.npz (run src/train.py first) and reports, on the
held-out test set: overall accuracy, per-digit accuracy, the confusion
matrix, and the digit pairs most often confused.

Run: python3 scripts/eval_confusion.py   (from the repo root)
"""
import os
import sys

sys.path.insert(0, "src")

import numpy as np

from data import load_split
from model import NeuralNetwork


def confusion_matrix(y_true, y_pred, num_classes=10):
    """m[t, p] = how often true digit t was predicted as p."""
    m = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def main():
    if not os.path.exists("models/digits_net.npz"):
        print("no trained weights found - run python src/train.py first")
        raise SystemExit(1)
    net = NeuralNetwork.load("models/digits_net.npz")
    X_test, y_test = load_split("test")
    y_pred = net.predict(X_test)
    correct = int((y_pred == y_test).sum())
    print(f"overall: {correct}/{len(y_test)} = {correct / len(y_test):.1%} correct")

    m = confusion_matrix(y_test, y_pred)
    print("\nper-digit accuracy:")
    for d in range(10):
        row = int(m[d].sum())
        print(f"  {d}: {m[d, d]}/{row} = {m[d, d] / row:.1%}")

    print("\nconfusion matrix (rows = true digit, columns = predicted):")
    print("    " + " ".join(f"{d:>4}" for d in range(10)))
    for d in range(10):
        print(f"  {d} " + " ".join(f"{v:>4}" for v in m[d]))

    pairs = sorted(
        (int(m[t, p]), t, p) for t in range(10) for p in range(10) if t != p and m[t, p]
    )
    print("\nmost confused pairs (true -> predicted: count):")
    for count, t, p in reversed(pairs[-5:]):
        print(f"  {t} -> {p}: {count}")


if __name__ == "__main__":
    main()
