"""Physics-based, time-varying noise for Stim surface-code circuits.

One sub-step represents 200 shots, ten sub-steps one 2000-shot step, and the
default 1500 sub-steps one 150-step stream. SI1000 follows Tables 1 and 2 of
Gidney et al., arXiv:2108.10457. The TLS spectrum is 1/f-like between the
fluctuator corner frequencies (about 2e-5 to 2e-2 per sub-step), 1/f^2 above.
Two-qubit noise is applied after CX, the generated circuit's native gate,
whereas SI1000 was defined for CZ. Cosmic-ray impacts are deliberately
accelerated (0.02 per 2000-shot step) relative to physical rates of roughly
one per ten seconds.
"""

from dataclasses import dataclass, field
import math

import numpy as np
import stim


SUBSTEPS_PER_STEP = 10
DEFAULT_SUBSTEPS = 1500
MAX_PROBABILITY = 0.4
# 200,000-shot static-MWPM LERs at seed 0: 0.000590, 0.002255,
# 0.004895, and 0.015930 for p = 0.001, 0.0015, 0.002, and 0.003.
CHOSEN_SI1000_P = 0.0015  # Closest LER (0.002255) to the 0.0033 target.
PHYS_REGIMES = (
    "phys_static",
    "phys_tls",
    "phys_cosmic",
    "phys_slow",
    "phys_coupler",
    "phys_mixed",
)


@dataclass(frozen=True)
class NoiseModel:
    """Noise strengths and per-qubit state for one circuit."""

    kind: str
    p: float
    m: dict[int, float] = field(default_factory=dict)
    burst: dict[int, float] = field(default_factory=dict)
    coupler: dict[tuple[int, int], float] = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in ("uniform", "si1000"):
            raise ValueError(f"unknown noise model {self.kind!r}")

    def multiplier(self, qubit):
        return float(self.m.get(qubit, 1.0))

    def pair_multiplier(self, a, b):
        return math.sqrt(self.multiplier(a) * self.multiplier(b))


@dataclass
class PhysicsNoiseState:
    """Recorded process states and resulting noise over all sub-steps."""

    qubits: np.ndarray
    coords: np.ndarray
    data_spacing: float
    m: np.ndarray
    burst: np.ndarray
    coupler_on: np.ndarray
    coupler_pairs: tuple
    coupler_switch_on: tuple
    impact_steps: np.ndarray
    impact_points: np.ndarray
    heterogeneity: np.ndarray
    slow_x: np.ndarray
    tls_on: np.ndarray
    tls_timescales: np.ndarray
    tls_strengths: np.ndarray

    def model(self, p, substep, kind="si1000"):
        """Freeze this state at one sub-step into a circuit noise model."""
        m = {int(q): float(self.m[substep, i]) for i, q in enumerate(self.qubits)}
        burst = {
            int(q): float(self.burst[substep, i])
            for i, q in enumerate(self.qubits)
            if self.burst[substep, i] > 0
        }
        coupler = {
            pair: 0.02
            for i, pair in enumerate(self.coupler_pairs)
            if self.coupler_on[substep, i]
        }
        return NoiseModel(kind, p, m=m, burst=burst, coupler=coupler)


def _data_qubits(base):
    return sorted(
        target.qubit_value
        for inst in base
        if inst.name == "M"
        for target in inst.targets_copy()
    )


def _two_qubit_pairs(base):
    pairs = set()
    for inst in base:
        if inst.name == "CX":
            targets = inst.targets_copy()
            for i in range(0, len(targets), 2):
                a = targets[i].qubit_value
                b = targets[i + 1].qubit_value
                pairs.add(tuple(sorted((a, b))))
    return sorted(pairs)


def _data_spacing(base, coords_by_qubit):
    data = _data_qubits(base)
    points = np.asarray([coords_by_qubit[q] for q in data], dtype=float)
    distances = [
        np.linalg.norm(points[i] - points[j])
        for i in range(len(points))
        for j in range(i + 1, len(points))
    ]
    if not distances:
        raise ValueError("at least two measured data qubits are required")
    return float(min(distances))


def make_noise_state(base, regime, device_seed=0, stream_seed=0, n_substeps=DEFAULT_SUBSTEPS):
    """Generate fixed-device heterogeneity and reproducible stream noise."""
    if regime not in PHYS_REGIMES:
        raise ValueError(f"unknown physics regime {regime!r}")
    if n_substeps <= 0:
        raise ValueError("n_substeps must be positive")

    coords_by_qubit = {
        q: np.asarray(x[:2], dtype=float)
        for q, x in base.get_final_qubit_coordinates().items()
    }
    qubits = np.asarray(sorted(coords_by_qubit), dtype=int)
    coords = np.asarray([coords_by_qubit[q] for q in qubits])
    n_qubits = len(qubits)
    spacing = _data_spacing(base, coords_by_qubit)

    # Keep the device independent from every drifting process and stream seed.
    h_seed = np.random.SeedSequence(device_seed).spawn(1)[0]
    heterogeneity = np.exp(0.3 * np.random.default_rng(h_seed).standard_normal(n_qubits))
    # One reproducible child generator per drifting physical process.
    children = np.random.SeedSequence(stream_seed).spawn(4)
    slow_rng, tls_rng, burst_rng, coupler_rng = map(np.random.default_rng, children)

    slow_x = np.zeros((n_substeps, n_qubits))
    if regime in ("phys_slow", "phys_mixed"):
        a = math.exp(-1 / 1000)
        innovation_std = 0.2 * math.sqrt(1 - a * a)
        slow_x[0] = 0.2 * slow_rng.standard_normal(n_qubits)
        for t in range(1, n_substeps):
            slow_x[t] = a * slow_x[t - 1] + innovation_std * slow_rng.standard_normal(n_qubits)

    tls_on = np.zeros((n_substeps, n_qubits, 2), dtype=bool)
    tls_timescales = np.ones((n_qubits, 2))
    tls_strengths = np.ones((n_qubits, 2))
    tls_factor = np.ones((n_substeps, n_qubits))
    if regime in ("phys_tls", "phys_mixed"):
        tls_timescales = np.exp(tls_rng.uniform(math.log(10), math.log(10000), (n_qubits, 2)))
        tls_strengths = tls_rng.uniform(1.5, 3.0, (n_qubits, 2))
        on = tls_rng.random((n_qubits, 2)) < 0.1
        for t in range(n_substeps):
            if t:
                random = tls_rng.random((n_qubits, 2))
                on = np.where(
                    on,
                    random >= 0.9 / tls_timescales,
                    random < 0.1 / tls_timescales,
                )
            tls_on[t] = on
            tls_factor[t] = np.prod(np.where(on, tls_strengths, 1.0), axis=1)

    m = heterogeneity[None, :] * np.exp(slow_x) * tls_factor

    burst = np.zeros((n_substeps, n_qubits))
    impact_steps = []
    impact_points = []
    if regime in ("phys_cosmic", "phys_mixed"):
        lower = coords.min(axis=0)
        upper = coords.max(axis=0)
        for t in range(n_substeps):
            if burst_rng.random() < 0.002:
                impact_steps.append(t)
                impact_points.append(lower + burst_rng.random(2) * (upper - lower))
            for impact, point in zip(impact_steps, impact_points):
                age = t - impact
                amplitude = 0.2 * math.exp(-age / 25)
                if amplitude < 1e-4:
                    continue
                sigma = 1 + 3 * min(age / 10, 1)
                r2 = np.sum(((coords - point) / spacing) ** 2, axis=1)
                burst[t] += amplitude * np.exp(-r2 / (2 * sigma * sigma))
            np.minimum(burst[t], MAX_PROBABILITY, out=burst[t])

    pairs = _two_qubit_pairs(base)
    coupler_pairs = ()
    coupler_on = np.zeros((n_substeps, 0), dtype=bool)
    coupler_switch_on = ()
    if regime in ("phys_coupler", "phys_mixed"):
        if len(pairs) < 2:
            raise ValueError("coupler drift requires at least two distinct gate pairs")
        chosen = coupler_rng.choice(len(pairs), size=2, replace=False)
        coupler_pairs = tuple(pairs[i] for i in chosen)
        coupler_on = np.zeros((n_substeps, 2), dtype=bool)
        # Couplers start off and can switch only at 10-sub-step boundaries.
        on = np.zeros(2, dtype=bool)
        switch_on = [[], []]
        for t in range(n_substeps):
            if t % SUBSTEPS_PER_STEP == 0:
                was_on = on.copy()
                random = coupler_rng.random(2)
                on = np.where(on, random >= 0.2, random < 0.02)
                for i in range(2):
                    if on[i] and not was_on[i]:
                        switch_on[i].append(t)
            coupler_on[t] = on
        coupler_switch_on = tuple(np.asarray(x, dtype=int) for x in switch_on)

    return PhysicsNoiseState(
        qubits=qubits,
        coords=coords,
        data_spacing=spacing,
        m=m,
        burst=burst,
        coupler_on=coupler_on,
        coupler_pairs=coupler_pairs,
        coupler_switch_on=coupler_switch_on,
        impact_steps=np.asarray(impact_steps, dtype=int),
        impact_points=np.asarray(impact_points, dtype=float).reshape(-1, 2),
        heterogeneity=heterogeneity,
        slow_x=slow_x,
        tls_on=tls_on,
        tls_timescales=tls_timescales,
        tls_strengths=tls_strengths,
    )


def _scaled_probability(p, multiplier):
    return min(MAX_PROBABILITY, p * multiplier)


def _append_qubit_noise(out, name, qubits, p, model):
    groups = {}
    for q in qubits:
        probability = _scaled_probability(p, model.multiplier(q))
        if probability > 0:
            groups.setdefault(probability, []).append(q)
    for probability, targets in groups.items():
        out.append(name, targets, probability)


def _compress_uniform_rounds(circuit):
    """Preserve Stim's REPEAT coordinate convention for uniform memory circuits."""
    instructions = list(circuit)
    mr_indices = [i for i, inst in enumerate(instructions) if inst.name == "MR"]
    m_indices = [i for i, inst in enumerate(instructions) if inst.name == "M"]
    if len(mr_indices) < 2 or not m_indices:
        return circuit
    final_m = m_indices[-1]
    m_targets = instructions[final_m].targets_copy()
    suffix_start = final_m
    while (
        suffix_start > 0
        and instructions[suffix_start - 1].name == "X_ERROR"
        and instructions[suffix_start - 1].targets_copy() == m_targets
    ):
        suffix_start -= 1

    def round_end(mr_index):
        for i in range(mr_index + 1, suffix_start + 1):
            if i == suffix_start or instructions[i].name == "TICK":
                return i
        raise ValueError("unterminated measurement round")

    boundaries = [round_end(mr_indices[0])]
    boundaries += [round_end(i) for i in mr_indices[1:]]
    rounds = [
        instructions[start:end]
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    def signature(insts):
        return [
            (
                inst.name,
                inst.gate_args_copy()[:2] if inst.name == "DETECTOR" else inst.gate_args_copy(),
                inst.targets_copy(),
            )
            for inst in insts
        ]

    if any(signature(block) != signature(rounds[0]) for block in rounds[1:]):
        return circuit
    body = stim.Circuit()
    round_time = None
    for inst in rounds[0]:
        if inst.name == "DETECTOR":
            args = inst.gate_args_copy()
            if round_time is None:
                round_time = args[2]
                body.append("SHIFT_COORDS", [], [0, 0, round_time])
            body.append("DETECTOR", inst.targets_copy(), args[:2] + [0])
        else:
            body.append(inst)

    compressed = stim.Circuit()
    for inst in instructions[:boundaries[0]]:
        compressed.append(inst)
    compressed.append(stim.CircuitRepeatBlock(len(rounds), body))
    for inst in instructions[suffix_start:]:
        if inst.name == "DETECTOR" and round_time is not None:
            args = inst.gate_args_copy()
            args[2] -= len(rounds) * round_time
            compressed.append("DETECTOR", inst.targets_copy(), args)
        else:
            compressed.append(inst)
    return compressed


def build_noisy_circuit(base, model):
    """Insert noise into a flattened noiseless circuit, one TICK layer at a time."""
    out = stim.Circuit()
    all_qubits = sorted(base.get_final_qubit_coordinates())
    data_qubits = _data_qubits(base)
    layer = []
    after_tick = False

    def flush_layer():
        nonlocal layer
        operations = [inst for inst in layer if inst.name in ("R", "H", "CX", "MR", "M")]
        if model.kind == "uniform" and after_tick and any(x.name == "H" for x in operations):
            # Stim puts before_round_data_depolarization immediately before each round's first H.
            _append_qubit_noise(out, "DEPOLARIZE1", data_qubits, model.p, model)

        touched = set()
        has_measurement_or_reset = False
        for inst in layer:
            name = inst.name
            if name not in ("R", "H", "CX", "MR", "M"):
                out.append(inst)
                continue
            targets = [target.qubit_value for target in inst.targets_copy()]
            touched.update(targets)

            if name == "R":
                has_measurement_or_reset = True
                out.append(inst)
                p = model.p if model.kind == "uniform" else 2 * model.p
                _append_qubit_noise(out, "X_ERROR", targets, p, model)
            elif name == "H":
                out.append(inst)
                p = model.p if model.kind == "uniform" else model.p / 10
                _append_qubit_noise(out, "DEPOLARIZE1", targets, p, model)
            elif name == "CX":
                out.append(inst)
                groups = {}
                pairs = [targets[i:i + 2] for i in range(0, len(targets), 2)]
                for a, b in pairs:
                    probability = _scaled_probability(model.p, model.pair_multiplier(a, b))
                    if probability > 0:
                        groups.setdefault(probability, []).extend((a, b))
                for probability, pair_targets in groups.items():
                    out.append("DEPOLARIZE2", pair_targets, probability)
                for a, b in pairs:
                    coupler_p = model.coupler.get(tuple(sorted((a, b))))
                    if coupler_p:
                        out.append("DEPOLARIZE2", [a, b], min(MAX_PROBABILITY, coupler_p))
            elif name in ("MR", "M"):
                has_measurement_or_reset = True
                p = model.p if model.kind == "uniform" else 5 * model.p
                _append_qubit_noise(out, "X_ERROR", targets, p, model)
                out.append(inst)
                if name == "MR":
                    p = model.p if model.kind == "uniform" else 2 * model.p
                    _append_qubit_noise(out, "X_ERROR", targets, p, model)
            else:
                raise ValueError(f"unsupported operation {name!r}")

        if model.kind == "si1000":
            idle = [q for q in all_qubits if q not in touched]
            _append_qubit_noise(out, "DEPOLARIZE1", idle, model.p / 10, model)
            if has_measurement_or_reset:
                _append_qubit_noise(out, "DEPOLARIZE1", idle, 2 * model.p, model)
        layer = []

    for inst in base:
        if inst.name == "TICK":
            starts_round = any(x.name in ("R", "MR") for x in layer)
            flush_layer()
            out.append(inst)
            for q in all_qubits:
                p = model.burst.get(q, 0.0)
                if p > 0:
                    out.append("DEPOLARIZE1", [q], min(MAX_PROBABILITY, p))
            after_tick = starts_round
        else:
            layer.append(inst)
    flush_layer()
    if model.kind == "uniform":
        return _compress_uniform_rounds(out)
    return out
