# Handwritten Digit Recognizer (NumPy from scratch)

A small neural network that recognizes handwritten digits, built entirely
with NumPy: no TensorFlow, no PyTorch, no scikit-learn at runtime. Forward
pass, backpropagation and gradient updates are implemented by hand so every
step of the learning process is readable in a few dozen lines.

Built by Yashraj Pangave.

## What it does

- Trains a three-layer network (64 inputs -> 128 and 64 hidden ReLU
  units -> 10 softmax outputs, with L2 regularization) on the Optdigits
  handwritten digits data, using the canonical split: 3,823 training
  images and 1,797 held-out test images.
- With the default settings (`python src/train.py`, seed 42) the measured
  result is 97.4% accuracy on the 1,797 test images (1,750 correct). The
  run is deterministic, so the same command reproduces the same number. A
  simpler one-hidden-layer configuration (`--layers 64,32,10 --epochs 60
  --l2 0`) measures 95.7% on the same test set.
- `python scripts/eval_confusion.py` breaks the result down per digit:
  0 is never missed (178/178), 8 and 7 are the hardest (93.7% and 93.9%),
  and the most common slip is 7 read as 5 (7 times in 1,797).

## Quick start

```bash
pip install -r requirements.txt   # numpy; Pillow only for the photo eval script
python src/train.py               # downloads the data once, trains, saves weights
python tests/test_model.py        # sanity tests
python scripts/eval_confusion.py  # per-digit accuracy and confusion matrix
```

The dataset is not bundled with this repository. The first run downloads
it (about 800 KB) directly from its owning source, the UCI Machine
Learning Repository, and caches it in `data/`, which is git-ignored.
Internet access is needed for that first download only.

Classify a single 8x8 image (64 pixel values, each 0-16):

```bash
python src/predict.py 0 0 5 13 9 1 0 0 0 0 13 15 10 15 5 0 0 3 15 2 0 11 8 0 0 4 12 0 0 8 8 0 0 5 8 0 0 9 8 0 0 4 11 0 1 12 7 0 0 2 14 5 10 12 0 0 0 0 6 13 10 0 0 0
```

## Reading multi-digit numbers (from photos)

`src/number_reader/pipeline.py` turns a photo of a handwritten number
(like a field on a paper form) into digits: grayscale -> binarize (Otsu)
-> dilate so stroke fragments merge -> connected components -> size
filter -> normalize each digit crop -> classify with the trained network
-> flag low-confidence digits for a human check.

Measured end to end (`python scripts/eval_number_reader.py`, deterministic
seed 7, on the held-out test set rendered as photo-like images):

- Single digits, one photo each: 95.0% read correctly (1,707 of 1,797),
  vs 97.4% on the untouched test features. Going through a photo costs
  about two points.
- Photos of 2-4 digit numbers (200 photos): 88.5% whole-number exact
  reads, 96.0% per-digit accuracy, no segmentation surprises.
- The confidence flag (below 0.90) marked 8.4% of digits for a human
  glance and caught 17 of the 24 actual errors.

Same-test-set note: these numbers reuse the Optdigits test split that
also guided model selection, so treat them as a description of this
setup, not an independent final evaluation. Real phone photos (glare,
skew, paper texture, different pens) will be harder; touching digits and
broken digits are known failure modes, and full-page forms are out of
scope.

## Data source and terms

Optdigits - E. Alpaydin and C. Kaynak (1998), "Optical Recognition of
Handwritten Digits", UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits

The UCI page does not state an explicit license for this dataset, so this
project does not redistribute it: the code downloads it directly from UCI
on your machine. The MIT license in this repository covers the code only,
not the data. If you use the data, cite Alpaydin and Kaynak (1998) and
respect any terms UCI sets.

## Project layout

```
src/model.py     the network: forward, backward, save/load
src/data.py      download-on-first-use loader for the UCI files
src/train.py     training loop (canonical 3,823 / 1,797 split)
src/predict.py   classify one image from the command line
src/number_reader/  photo -> multi-digit number pipeline
scripts/eval_number_reader.py  end-to-end photo evaluation
scripts/eval_confusion.py      per-digit accuracy and confusion matrix
tests/           sanity tests (split shapes, loss decreases, determinism)
data/            local dataset cache (git-ignored, created on first run)
models/          trained weights (git-ignored, created by src/train.py)
```

## Why from scratch?

Frameworks hide the math. Writing the backpropagation by hand is the fastest
way to actually understand what a neural network computes, and the same
code extends naturally to larger problems.
