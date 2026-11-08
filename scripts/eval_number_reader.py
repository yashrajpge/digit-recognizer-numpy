"""Evaluate the number-reading pipeline end to end on photo-like images.

Rebuilds each Optdigits test digit as a scattered-pixel 32x32 bitmap
(the original capture format), concatenates 2-4 digits into a "photo of
a form field" line, and measures whole-number reads, per-digit accuracy
and how often low-confidence flags catch real errors.

Run: python3 scripts/eval_number_reader.py   (from the repo root)
Requires: numpy, Pillow. Downloads Optdigits from UCI on first use.
"""
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "src/number_reader")

import numpy as np
from PIL import Image

from data import load_split
from model import NeuralNetwork
from pipeline import read_number


def digit_bitmap(counts8, rng):
    bm = np.zeros((32, 32), dtype=np.uint8)
    c16 = (counts8 * 16).round().astype(int)
    for r in range(8):
        for c in range(8):
            v = c16[r, c]
            if v > 0:
                cells = rng.choice(16, size=min(v, 16), replace=False)
                for cell in cells:
                    bm[r * 4 + cell // 4, c * 4 + cell % 4] = 1
    return bm


def make_number_image(idxs, X, rng, gap=4, pad=10, upscale=3):
    imgs = [digit_bitmap(X[i].reshape(8, 8), rng) for i in idxs]
    w = 32 * len(imgs) + gap * (len(imgs) - 1)
    canvas = np.zeros((32 + 2 * pad, w + 2 * pad), dtype=np.uint8)
    x = pad
    for im in imgs:
        canvas[pad:pad + 32, x:x + 32] = im
        x += 32 + gap
    gray = 255.0 * (1.0 - canvas)
    big = Image.fromarray(gray.astype(np.uint8)).resize(
        (gray.shape[1] * upscale, gray.shape[0] * upscale), Image.BILINEAR)
    return np.asarray(big, dtype=float)


def main():
    X_test, y_test = load_split("test")
    net = NeuralNetwork.load("models/digits_net.npz")
    rng = np.random.default_rng(7)

    # 1) single digits, one "photo" each (all 1,797 test digits)
    single_ok = 0
    for i in range(len(X_test)):
        res = read_number(make_number_image([i], X_test, rng), net)
        if len(res["digits"]) == 1 and res["digits"][0]["digit"] == int(y_test[i]):
            single_ok += 1
    print(f"single-digit photos: {single_ok}/{len(X_test)} = {single_ok/len(X_test):.1%} read correctly")

    # 2) multi-digit numbers (2-4 digits), 200 photos
    n_trials = 200
    exact = digit_correct = digit_total = flagged = seg_errors = 0
    wrong = flagged_wrong = 0
    for _ in range(n_trials):
        k = int(rng.integers(2, 5))
        idxs = rng.integers(0, len(X_test), size=k)
        res = read_number(make_number_image(idxs, X_test, rng), net)
        truth = "".join(str(int(y_test[i])) for i in idxs)
        exact += res["number"] == truth
        seg_errors += len(res["digits"]) != k
        for d, i in zip(res["digits"], idxs):
            digit_total += 1
            ok = d["digit"] == int(y_test[i])
            digit_correct += ok
            wrong += not ok
            if d["flag"]:
                flagged += 1
                flagged_wrong += not ok
    print(f"multi-digit numbers: {n_trials} photos of 2-4 digit numbers")
    print(f"whole-number exact reads: {exact}/{n_trials} = {exact/n_trials:.1%}")
    print(f"per-digit accuracy: {digit_correct}/{digit_total} = {digit_correct/digit_total:.1%}")
    print(f"segmentation surprises (wrong digit count): {seg_errors}")
    print(f"flagged for human check: {flagged}/{digit_total} = {flagged/digit_total:.1%}")
    print(f"flags caught {flagged_wrong}/{wrong} actual errors")


if __name__ == "__main__":
    main()
