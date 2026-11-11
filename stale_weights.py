"""How much does matching suffer when noise drifts?

Three cases, all decoded with PyMatching on the same drifted syndromes:
  clean  no drift, nominal weights
  blind  drifted noise, weights from the nominal noise model (a static decoder)
  aware  drifted noise, weights from the true drifted noise

The ratio blind/aware is the headroom for an adaptive decoder.
"""

import numpy as np
import pymatching
import stim

D = 5  # code distance
ROUNDS = 5
P = 0.003  # nominal circuit-level depolarizing noise
HOT_P = 0.03  # extra depolarizing noise on the drifted qubits, per round
N_HOT = 2  # number of drifted qubits
SHOTS = 200_000
SEED = 1


def make_circuit(hot_qubits=()):
    c = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=D,
        rounds=ROUNDS,
        after_clifford_depolarization=P,
        after_reset_flip_probability=P,
        before_measure_flip_probability=P,
        before_round_data_depolarization=P,
    )
    if not hot_qubits:
        return c
    out = stim.Circuit()
    for inst in c.flattened():
        out.append(inst)
        if inst.name == "TICK":
            out.append("DEPOLARIZE1", list(hot_qubits), HOT_P)
    return out


def pick_hot_qubits(circuit):
    # the qubits nearest the patch centre
    coords = circuit.get_final_qubit_coordinates()
    cx = np.mean([v[0] for v in coords.values()])
    cy = np.mean([v[1] for v in coords.values()])
    ranked = sorted(coords, key=lambda q: (coords[q][0] - cx) ** 2 + (coords[q][1] - cy) ** 2)
    return ranked[:N_HOT]


def ler(circuit, decoder_circuit, shots):
    """Logical error rate on shots from `circuit`, decoded with a matching graph built from `decoder_circuit`."""
    dem = decoder_circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    det, obs = circuit.compile_detector_sampler(seed=SEED).sample(shots, separate_observables=True)
    pred = matching.decode_batch(det)
    return float(np.mean(np.any(pred != obs, axis=1)))


if __name__ == "__main__":
    nominal = make_circuit()
    hot = pick_hot_qubits(nominal)
    drifted = make_circuit(hot)
    print(f"d={D} rounds={ROUNDS} p={P} hot qubits={hot} hot_p={HOT_P} shots={SHOTS}")

    clean = ler(nominal, nominal, SHOTS)
    blind = ler(drifted, nominal, SHOTS)
    aware = ler(drifted, drifted, SHOTS)

    print(f"clean (no drift, nominal weights)   : {clean:.5f}")
    print(f"drifted, blind (stale nominal DEM)  : {blind:.5f}")
    print(f"drifted, aware (true drifted DEM)   : {aware:.5f}")
    print(f"headroom blind/aware                : {blind / aware:.2f}x")
