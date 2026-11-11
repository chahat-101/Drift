"""Drifting noise regimes and syndrome-stream sampling.

Time is split into steps. The noise is fixed within a step and changes between
steps according to a regime:

    static      no drift
    wander      mean-reverting random walk of the global noise scale (log space)
    telegraph   global noise scale jumps between two levels at random times
    hot_qubit   a few qubits get much worse partway through and stay bad
    bursts      short bursts on a cluster of neighbouring qubits
    bursts_big  like bursts, with 8-qubit clusters at hot_p 0.1 (outside the training range)

The noise at a step is a NoiseState: a multiplier on the nominal p plus an
optional set of hot qubits with extra noise.
"""

from dataclasses import dataclass

import numpy as np

from circuits import cached_circuit, nearest_qubits

SCALE_QUANTUM = 0.05  # scales are rounded to this so circuits can be cached


@dataclass(frozen=True)
class NoiseState:
    scale: float = 1.0  # physical error rate = scale * nominal p
    hot_qubits: tuple = ()
    hot_p: float = 0.0

    def quantized(self):
        s = round(round(self.scale / SCALE_QUANTUM) * SCALE_QUANTUM, 4)
        return NoiseState(max(s, SCALE_QUANTUM), tuple(sorted(self.hot_qubits)), round(self.hot_p, 4))


def static(steps):
    return [NoiseState() for _ in range(steps)]


def wander(steps, rng, sigma=0.08, theta=0.03, lo=0.5, hi=2.5):
    """Mean-reverting random walk of log(scale) around 0 (scale 1)."""
    out, x = [], 0.0
    for _ in range(steps):
        x += -theta * x + sigma * rng.standard_normal()
        x = float(np.clip(x, np.log(lo), np.log(hi)))
        out.append(NoiseState(scale=float(np.exp(x))))
    return out


def telegraph(steps, rng, high=2.0, switch_prob=0.04):
    """Two-level random telegraph noise on the global scale."""
    out, up = [], False
    for _ in range(steps):
        if rng.random() < switch_prob:
            up = not up
        out.append(NoiseState(scale=high if up else 1.0))
    return out


def hot_qubit(steps, nominal_circuit, n_hot=2, hot_p=0.03, start=None):
    """A few central qubits go bad at `start` (default: a third of the way in)."""
    start = steps // 3 if start is None else start
    hot = nearest_qubits(nominal_circuit, n_hot)
    return [NoiseState() if t < start else NoiseState(hot_qubits=hot, hot_p=hot_p) for t in range(steps)]


def bursts(steps, rng, nominal_circuit, burst_prob=0.04, burst_len=4, size=5, hot_p=0.05):
    """A cluster of neighbouring qubits is hit for burst_len steps; bursts start with probability burst_prob."""
    coords = list(nominal_circuit.get_final_qubit_coordinates().values())
    out, left, cluster = [], 0, ()
    for _ in range(steps):
        if left == 0 and rng.random() < burst_prob:
            center = coords[rng.integers(len(coords))][:2]
            cluster = nearest_qubits(nominal_circuit, size, center=center)
            left = burst_len
        if left > 0:
            out.append(NoiseState(hot_qubits=cluster, hot_p=hot_p))
            left -= 1
        else:
            out.append(NoiseState())
    return out


def make_schedule(regime, steps, rng, nominal_circuit):
    """The list of NoiseStates for a regime."""
    if regime == "static":
        return static(steps)
    if regime == "wander":
        return wander(steps, rng)
    if regime == "telegraph":
        return telegraph(steps, rng)
    if regime == "hot_qubit":
        return hot_qubit(steps, nominal_circuit)
    if regime == "bursts":
        return bursts(steps, rng, nominal_circuit)
    if regime == "bursts_big":
        return bursts(steps, rng, nominal_circuit, size=8, hot_p=0.1)
    raise ValueError(f"unknown regime {regime!r}")


REGIMES = ["static", "wander", "telegraph", "hot_qubit", "bursts", "bursts_big"]


def random_state(rng, nominal, max_scale=2.5, hot_prob=0.5):
    """Random training noise: a global scale, and with probability hot_prob a cluster of 1-6 hot qubits."""
    scale = float(np.exp(rng.uniform(np.log(0.5), np.log(max_scale))))
    if rng.random() < hot_prob:
        coords = list(nominal.get_final_qubit_coordinates().values())
        center = coords[rng.integers(len(coords))][:2]
        hot = nearest_qubits(nominal, int(rng.integers(1, 7)), center=center)
        return NoiseState(scale, hot, float(rng.uniform(0.005, 0.08)))
    return NoiseState(scale)


def test_states(nominal):
    """The fixed noise settings the trained decoder is scored on."""
    hot = nearest_qubits(nominal, 2)
    return {
        "nominal": NoiseState(),
        "scale x2": NoiseState(scale=2.0),
        "2 hot qubits": NoiseState(hot_qubits=hot, hot_p=0.03),
    }


class DriftStream:
    """Builds the circuit for a NoiseState and samples shots from it."""

    def __init__(self, distance, rounds, p, basis="z"):
        self.distance, self.rounds, self.p, self.basis = distance, rounds, p, basis

    def circuit(self, state=NoiseState()):
        s = state.quantized()
        return cached_circuit(self.distance, self.rounds, round(self.p * s.scale, 8), s.hot_qubits, s.hot_p, self.basis)

    def nominal(self):
        return self.circuit(NoiseState())

    def sample(self, state, shots, seed):
        det, obs = self.circuit(state).compile_detector_sampler(seed=int(seed)).sample(shots, separate_observables=True)
        return det, obs


def sample_chunk(stream, rng, batch, window, nominal=False):
    """One training chunk: (state, det, obs, window_det).

    The chunk has `batch` shots and `window` fingerprint shots from one random NoiseState,
    or from NoiseState() if nominal.
    """
    state = NoiseState() if nominal else random_state(rng, stream.nominal())
    det, obs = stream.sample(state, batch + window, rng.integers(2**31))
    return state, det[:batch], obs[:batch], det[batch:]


def sample_stream(stream, regime, steps, shots, rng):
    """Yield (state, det, obs) for each step of a drifting stream."""
    for state in make_schedule(regime, steps, rng, stream.nominal()):
        det, obs = stream.sample(state, shots, rng.integers(2**31))
        yield state, det, obs
