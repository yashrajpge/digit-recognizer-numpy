"""Train the network on the Optdigits handwritten digits data.

Uses the canonical split: 3,823 training and 1,797 testing images.
Usage: python src/train.py [--layers 64,64,32,10] [--epochs 80] [--lr 0.5] [--l2 0.0001]
"""

import argparse

import numpy as np

from data import load_split
from model import NeuralNetwork


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="64,128,64,10",
                        help="comma-separated layer sizes, input first, output last")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    layers = [int(s) for s in args.layers.split(",")]
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    assert layers[0] == X_train.shape[1] and layers[-1] == 10

    net = NeuralNetwork(layers, l2=args.l2)
    rng = np.random.default_rng(0)
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(X_train))
        for start in range(0, len(X_train), args.batch):
            idx = order[start:start + args.batch]
            net.forward(X_train[idx])
            net.backward(y_train[idx], args.lr)
        if epoch % 20 == 0 or epoch == 1:
            train_acc = (net.predict(X_train) == y_train).mean()
            print(f"epoch {epoch:>3}: train accuracy {train_acc:.3f}")

    test_acc = (net.predict(X_test) == y_test).mean()
    print(f"layers {layers}, l2 {args.l2}")
    print(f"test accuracy: {test_acc:.4f} ({len(X_test)} held-out images)")
    import os
    os.makedirs("models", exist_ok=True)
    net.save("models/digits_net.npz")
    net.export_json("models/digits_net.json")
    print("saved weights to models/digits_net.npz and .json")


if __name__ == "__main__":
    main()
