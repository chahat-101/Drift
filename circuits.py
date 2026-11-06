"""Surface-code circuits with controllable (drifting) noise.

Everything else in the repo builds circuits through `make_circuit`, so the
noise model lives in one place. Noise is SI-style uniform depolarizing noise
at strength `p`, plus optional extra depolarizing noise on a set of "hot"
qubits after every TICK (a qubit whose coherence suddenly got worse, e.g.
a TLS defect drifting into resonance, or a radiation burst).
"""
from functools import lru_cache

import numpy as np
import pymatching
import stim


def make_circuit(distance, rounds, p, hot_qubits=(), hot_p=0.0, basis="z"):
    """Rotated surface-code memory experiment.

    distance, rounds : code distance and number of syndrome rounds
    p                : uniform physical error rate
    hot_qubits       : qubit indices that get extra noise
    hot_p            : extra DEPOLARIZE1 strength on hot qubits, per TICK
    basis            : "z" or "x" memory experiment
    """
    c = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )
    if not hot_qubits or hot_p <= 0:
        return c
    hot = sorted(int(q) for q in hot_qubits)
    out = stim.Circuit()
    for inst in c.flattened():
        out.append(inst)
        if inst.name == "TICK":
            out.append("DEPOLARIZE1", hot, hot_p)
    return out


def qubit_coords(circuit):
    """{qubit index: (x, y)} for every qubit in the circuit."""
    return {q: tuple(v[:2]) for q, v in circuit.get_final_qubit_coordinates().items()}


def nearest_qubits(circuit, k, center=None):
    """The k qubits closest to `center` (default: the middle of the patch)."""
    coords = qubit_coords(circuit)
    pts = np.array(list(coords.values()), dtype=float)
    if center is None:
        center = pts.mean(axis=0)
    ids = list(coords)
    d2 = ((pts - np.asarray(center, dtype=float)) ** 2).sum(axis=1)
    return tuple(sorted(ids[i] for i in np.argsort(d2, kind="stable")[:k]))


def matcher_from_circuit(circuit):
    """PyMatching decoder whose weights come from the circuit's own noise."""
    dem = circuit.detector_error_model(decompose_errors=True)
    return pymatching.Matching.from_detector_error_model(dem)


@lru_cache(maxsize=512)
def cached_circuit(distance, rounds, p, hot_qubits=(), hot_p=0.0, basis="z"):
    return make_circuit(distance, rounds, p, hot_qubits, hot_p, basis)
