"""Classify a single 8x8 digit image given as 64 pixel values.

Usage: python src/predict.py 0 0 5 13 ... (64 integers in 0-16)
"""

import sys

import numpy as np

from model import NeuralNetwork


def main():
    if len(sys.argv) != 65:
        print(__doc__)
        raise SystemExit(1)
    pixels = np.array([int(v) for v in sys.argv[1:]], dtype=np.float64) / 16.0
    net = NeuralNetwork.load("models/digits_net.npz")
    probs = net.forward(pixels[None, :])[0]
    digit = int(probs.argmax())
    print(f"predicted digit: {digit} (confidence {probs[digit]:.1%})")


if __name__ == "__main__":
    main()
