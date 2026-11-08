"""Sanity tests: run with `python tests/test_model.py` from the repo root.

The first run downloads the dataset from UCI (needs internet once).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from data import load_split
from model import NeuralNetwork


def test_data_shape():
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    assert X_train.shape == (3823, 64), X_train.shape
    assert X_test.shape == (1797, 64), X_test.shape
    assert set(np.unique(y_train)) == set(range(10))
    assert 0.0 <= X_train.min() and X_train.max() <= 1.0
    print("ok: canonical split shapes and value range")


def test_training_reduces_loss():
    X_train, y_train = load_split("train")
    net = NeuralNetwork((64, 64, 32, 10), l2=1e-4)
    Xb, yb = X_train[:256], y_train[:256]
    first = net.loss(net.forward(Xb), yb)
    for _ in range(20):
        net.forward(Xb)
        net.backward(yb, 0.5)
    last = net.loss(net.forward(Xb), yb)
    assert last < first, (first, last)
    print(f"ok: loss falls during training ({first:.3f} -> {last:.3f})")


def test_model_is_deterministic():
    X_train, y_train = load_split("train")
    Xb, yb = X_train[:128], y_train[:128]
    losses = []
    for _ in range(2):
        net = NeuralNetwork((64, 32, 10), seed=42)
        for _ in range(5):
            net.forward(Xb)
            net.backward(yb, 0.5)
        losses.append(net.loss(net.forward(Xb), yb))
    assert losses[0] == losses[1], losses
    print(f"ok: same seed gives identical loss ({losses[0]:.6f})")


def test_save_load_roundtrip():
    import tempfile
    X_test, _ = load_split("test")
    net = NeuralNetwork((64, 64, 32, 10))
    before = net.predict(X_test[:32])
    with tempfile.TemporaryDirectory() as d:
        net.save(os.path.join(d, "w.npz"))
        net2 = NeuralNetwork.load(os.path.join(d, "w.npz"))
    assert (net2.predict(X_test[:32]) == before).all()
    assert net2.layer_sizes == [64, 64, 32, 10]
    print("ok: saved weights reload to identical predictions")


if __name__ == "__main__":
    test_data_shape()
    test_training_reduces_loss()
    test_model_is_deterministic()
    test_save_load_roundtrip()
    print("all tests passed")
