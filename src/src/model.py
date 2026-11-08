"""A configurable multi-layer perceptron written from scratch with NumPy.

No deep-learning framework is used: forward pass, backpropagation and
weight updates are implemented directly so every step is readable.
Supports any number of hidden layers and optional L2 regularization.
"""

import numpy as np


class NeuralNetwork:
    """Layers of ReLU hidden units feeding a softmax output."""

    def __init__(self, layer_sizes=(64, 32, 10), l2=0.0, seed=42):
        rng = np.random.default_rng(seed)
        self.layer_sizes = list(layer_sizes)
        self.l2 = l2
        # He initialisation keeps ReLU activations on a sane scale.
        self.weights = [
            rng.normal(0, np.sqrt(2 / fan_in), (fan_in, fan_out))
            for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.biases = [np.zeros(n) for n in layer_sizes[1:]]

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, X):
        self._activations = [X]
        a = X
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            a = np.maximum(0, a @ W + b)  # ReLU
            self._activations.append(a)
        probs = self._softmax(a @ self.weights[-1] + self.biases[-1])
        self._activations.append(probs)
        return probs

    def loss(self, probs, y):
        n = len(y)
        data_loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()
        reg = 0.5 * self.l2 * sum((W ** 2).sum() for W in self.weights)
        return data_loss + reg

    def backward(self, y, lr):
        n = len(y)
        dZ = self._activations[-1]
        dZ[np.arange(n), y] -= 1
        dZ /= n
        for i in range(len(self.weights) - 1, -1, -1):
            a_prev = self._activations[i]
            dW = a_prev.T @ dZ + self.l2 * self.weights[i]
            db = dZ.sum(axis=0)
            if i > 0:
                dZ = dZ @ self.weights[i].T
                dZ[self._activations[i] <= 0] = 0
            self.weights[i] -= lr * dW
            self.biases[i] -= lr * db

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    def save(self, path):
        arrays = {}
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"W{i}"], arrays[f"b{i}"] = W, b
        np.savez(path, layer_sizes=np.array(self.layer_sizes), l2=np.array(self.l2), **arrays)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        sizes = [int(s) for s in data["layer_sizes"]]
        net = cls(sizes, l2=float(data["l2"]))
        net.weights = [data[f"W{i}"] for i in range(len(sizes) - 1)]
        net.biases = [data[f"b{i}"] for i in range(len(sizes) - 1)]
        return net

    def export_json(self, path):
        import json
        payload = {
            "layer_sizes": self.layer_sizes,
            "weights": [[[round(v, 5) for v in row] for row in W.tolist()] for W in self.weights],
            "biases": [[round(v, 5) for v in b.tolist()] for b in self.biases],
        }
        with open(path, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
