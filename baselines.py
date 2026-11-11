"""Classical decoders.

All decoders share one interface:

    decode(det, state) -> predicted observable flips, shape (shots, n_obs)
    observe(det)       -> feed shots after they are decoded, so a decoder only learns from the past

`state` is the true NoiseState. Only Oracle reads it.

Static     PyMatching with the nominal noise model, never updated.
Oracle     PyMatching rebuilt from the true noise at every step.
WindowDEM  PyMatching reweighted from the last `window` shots. Each edge probability
           is estimated from how often its two detectors fire together.
"""

from collections import deque

import numpy as np
import pymatching
import stim

from circuits import matcher_from_circuit

P_MIN, P_MAX = 1e-6, 0.45


class Static:
    name = "static"

    def __init__(self, nominal_circuit):
        self.matching = matcher_from_circuit(nominal_circuit)

    def decode(self, det, state=None):
        return self.matching.decode_batch(det)

    def observe(self, det):
        pass


class Oracle:
    name = "oracle"

    def __init__(self, stream):
        self.stream = stream
        self._cache = {}

    def decode(self, det, state):
        key = state.quantized()
        if key not in self._cache:
            self._cache[key] = matcher_from_circuit(self.stream.circuit(state))
        return self._cache[key].decode_batch(det)

    def observe(self, det):
        pass


class ShotWindow:
    """Rolling buffer of the most recent `window` shots of detection events."""

    def __init__(self, window):
        self.window = window
        self.chunks = deque()
        self.count = 0

    def add(self, det):
        self.chunks.append(det)
        self.count += len(det)
        while self.chunks and self.count - len(self.chunks[0]) >= self.window:
            self.count -= len(self.chunks.popleft())

    def array(self):
        if not self.chunks:
            return None
        return np.concatenate(list(self.chunks))[-self.window :]


class WindowDEM:
    name = "window"

    def __init__(self, nominal_circuit, window=5000, min_shots=1000):
        self.n = nominal_circuit.num_detectors
        self.base = matcher_from_circuit(nominal_circuit)
        self.matching = self.base
        self.buf = ShotWindow(window)
        self.min_shots = min_shots
        edges = self.base.edges()
        self.eu = np.array([u for u, v, _ in edges if v is not None], dtype=int)
        self.ev = np.array([v for u, v, _ in edges if v is not None], dtype=int)
        self.efid = [tuple(a.get("fault_ids", ())) for u, v, a in edges if v is not None]
        self.bu = np.array([u for u, v, _ in edges if v is None], dtype=int)
        self.bfid = [tuple(a.get("fault_ids", ())) for u, v, a in edges if v is None]

    def decode(self, det, state=None):
        return self.matching.decode_batch(det)

    def observe(self, det):
        self.buf.add(det)
        if self.buf.count >= self.min_shots:
            self.matching = self._reweighted(self.buf.array())

    def estimate(self, X):
        """Edge and boundary error probabilities from detection events X, shape (shots, n)."""
        X = X.astype(np.float64)
        mu = X.mean(axis=0)
        xx = (X.T @ X) / len(X)
        mu_u, mu_v, xy = mu[self.eu], mu[self.ev], xx[self.eu, self.ev]
        num = 4.0 * (xy - mu_u * mu_v)
        den = 1.0 - 2.0 * mu_u - 2.0 * mu_v + 4.0 * xy
        inside = np.clip(1.0 - num / np.where(np.abs(den) < 1e-12, 1e-12, den), 0.0, 1.0)
        p_edge = np.clip(0.5 - 0.5 * np.sqrt(inside), P_MIN, P_MAX)
        # A detector fires if an odd number of its edges flipped, so
        # 1 - 2*mu = prod(1 - 2*p) over all edges touching it, boundary included.
        prod = np.ones(self.n)
        np.multiply.at(prod, self.eu, 1 - 2 * p_edge)
        np.multiply.at(prod, self.ev, 1 - 2 * p_edge)
        ratio = (1 - 2 * mu[self.bu]) / np.maximum(prod[self.bu], 1e-12)
        p_bound = np.clip(0.5 * (1 - ratio), P_MIN, P_MAX)
        return p_edge, p_bound

    def _reweighted(self, X):
        p_edge, p_bound = self.estimate(X)
        lines = []
        for u, v, p, f in zip(self.eu, self.ev, p_edge, self.efid):
            lines.append(f"error({p:.8g}) D{u} D{v}" + "".join(f" L{l}" for l in f))
        for u, p, f in zip(self.bu, p_bound, self.bfid):
            lines.append(f"error({p:.8g}) D{u}" + "".join(f" L{l}" for l in f))
        lines.append(f"detector D{self.n - 1}")
        return pymatching.Matching.from_detector_error_model(stim.DetectorErrorModel("\n".join(lines)))
