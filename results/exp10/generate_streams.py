"""Generate the physics-based drift streams of round 10 into data/phys10/.

    python results/exp10/generate_streams.py                    # all 30 streams + calibration
    python results/exp10/generate_streams.py --regimes phys_tls --seeds 0 --substeps 50 --out /tmp/x

Per stream: make_noise_state(base, regime, device_seed=0, stream_seed=S, n_substeps=N),
one SI1000 circuit per sub-step (state.model(CHOSEN_SI1000_P, t)), 200 shots sampled with
seed = blake2b(f"{regime}|{stream_seed}|{substep}"). Files hold bit-packed det/obs (packed as in
dataset.py), the full PhysicsNoiseState arrays, the drift/burst-bin masks and the event windows.
The calibration set is 2000 shots of the static device (sampler seed blake2b("phys10|calibration")).
Resumable: a stream is skipped when its file exists, is listed in the manifest and its SHA-256
matches. The manifest is rewritten atomically after each file.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import stim

sys.path.insert(0, ".")
from dataset import pack_det, pack_obs, sha256_file  # noqa: E402
from physics_noise import (  # noqa: E402
    CHOSEN_SI1000_P,
    PHYS_REGIMES,
    build_noisy_circuit,
    make_noise_state,
)

FORMAT = 1
SHOTS = 200
CALIBRATION_SHOTS = 2000
DRIFT_BURST = 0.005  # max burst that still counts as drift


def noiseless_d5():
    return stim.Circuit.generated("surface_code:rotated_memory_z", distance=5, rounds=5).flattened()


def stim_seed(*parts):
    """64-bit sampler seed from the labels, as fixed in notes/10_definitions.md."""
    return int.from_bytes(hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=8).digest(), "big")


def sample(circuit, seed, shots):
    """(det, obs) bool arrays of `shots` shots from one circuit."""
    det, obs = circuit.compile_detector_sampler(seed=seed).sample(shots=shots, separate_observables=True)
    return det, obs


def burst_bins(state):
    """(drift, bin ids -1/0/1/2) per sub-step; bin = amplitude of the youngest active burst."""
    t_max = len(state.m)
    max_burst = state.burst.max(axis=1)
    drift = max_burst >= DRIFT_BURST
    bins = np.full(t_max, -1, dtype=np.int8)
    impacts = state.impact_steps
    if len(impacts):
        # active at t: age <= t and 0.2*exp(-age/25) >= 0.005, i.e. age <= 25*ln(40)
        max_age = 25 * np.log(0.2 / DRIFT_BURST)
        ages = np.arange(t_max)[:, None] - impacts[None, :]
        active = (ages >= 0) & (ages <= max_age)
        youngest = np.where(active.any(axis=1), np.max(np.where(active, impacts[None, :], -1), axis=1), -1)
        has = youngest >= 0
        amplitude = np.zeros(t_max)
        amplitude[has] = 0.2 * np.exp(-(np.arange(t_max)[has] - youngest[has]) / 25.0)
        bins[has & (amplitude >= 0.05)] = 0
        bins[has & (amplitude < 0.05) & (amplitude >= 0.0125)] = 1
        bins[has & (amplitude < 0.0125) & (amplitude >= 0.005)] = 2
    return drift, bins


def drift_mask(regime, state):
    """Drift sub-steps per notes/10_definitions.md."""
    t_max = len(state.m)
    if regime == "phys_static":
        return np.zeros(t_max, dtype=bool)
    if regime == "phys_slow":
        return np.ones(t_max, dtype=bool)
    tls = state.tls_on.reshape(t_max, -1).any(axis=1)
    coupler = state.coupler_on.reshape(t_max, -1).any(axis=1)
    cosmic, _ = burst_bins(state)
    if regime == "phys_cosmic":
        return cosmic
    if regime == "phys_tls":
        return tls
    if regime == "phys_coupler":
        return coupler
    return cosmic | tls | coupler  # phys_mixed; slow alone never marks drift


def runs_of(flags):
    """Maximal True runs of a 1-D bool array as (start, end) pairs."""
    padded = np.concatenate(([False], flags, [False])).astype(np.int8)
    change = np.diff(padded)
    return np.stack((np.flatnonzero(change == 1), np.flatnonzero(change == -1)), axis=1)


def coupler_switch_array(state):
    """The per-pair switch-on times as a (2, L) int array padded with -1, or (0, 0)."""
    if not state.coupler_switch_on:
        return np.empty((0, 0), dtype=np.int64)
    width = max(len(x) for x in state.coupler_switch_on)
    out = np.full((len(state.coupler_switch_on), width), -1, dtype=np.int64)
    for i, x in enumerate(state.coupler_switch_on):
        out[i, : len(x)] = x
    return out


def events(state):
    """Event windows: bursts (impact until max burst < 0.005), TLS and coupler on-periods."""
    max_burst = state.burst.max(axis=1)
    quiet = max_burst < DRIFT_BURST
    bursts = []
    for s in state.impact_steps:
        e = s
        while e < len(max_burst) and not quiet[e]:
            e += 1
        bursts.append((s, e))
    tls = [[q, k, s, e] for q in range(len(state.qubits)) for k in range(2)
           for s, e in runs_of(state.tls_on[:, q, k])]
    coupler = [[i, s, e] for i in range(state.coupler_on.shape[1]) for s, e in runs_of(state.coupler_on[:, i])]
    return {
        "burst_events": np.asarray(bursts, dtype=np.int64).reshape(-1, 2),
        "tls_events": np.asarray(tls, dtype=np.int64).reshape(-1, 4),
        "coupler_events": np.asarray(coupler, dtype=np.int64).reshape(-1, 3),
    }


def stream_file(regime, seed):
    return f"stream_{regime}_s{seed}.npz"


def save(path, arrays):
    np.savez_compressed(path, **arrays)


def generate_stream(base, regime, seed, n_substeps):
    """All arrays of one stream, deterministically."""
    state = make_noise_state(base, regime, device_seed=0, stream_seed=seed, n_substeps=n_substeps)
    det, obs = [], []
    for t in range(n_substeps):
        circuit = build_noisy_circuit(base, state.model(CHOSEN_SI1000_P, t))
        d, o = sample(circuit, stim_seed(regime, seed, t), SHOTS)
        det.append(pack_det(d))
        obs.append(pack_obs(o))
    _, bins = burst_bins(state)
    drift = drift_mask(regime, state)
    arrays = {
        "det": np.stack(det),
        "obs": np.stack(obs),
        "drift": drift,
        "burst_bin": bins,
        "qubits": state.qubits,
        "coords": state.coords,
        "data_spacing": np.float64(state.data_spacing),
        "m": state.m,
        "burst": state.burst,
        "coupler_on": state.coupler_on,
        "coupler_pairs": np.asarray(state.coupler_pairs, dtype=np.int64).reshape(-1, 2),
        "coupler_switch_on": coupler_switch_array(state),
        "impact_steps": state.impact_steps,
        "impact_points": state.impact_points,
        "heterogeneity": state.heterogeneity,
        "slow_x": state.slow_x,
        "tls_on": state.tls_on,
        "tls_timescales": state.tls_timescales,
        "tls_strengths": state.tls_strengths,
        **events(state),
    }
    return arrays


def generate_calibration(base):
    """2000 shots of the time-invariant static device."""
    state = make_noise_state(base, "phys_static", device_seed=0, stream_seed=0, n_substeps=1)
    circuit = build_noisy_circuit(base, state.model(CHOSEN_SI1000_P, 0))
    det, obs = sample(circuit, stim_seed("phys10", "calibration"), CALIBRATION_SHOTS)
    return {"det": pack_det(det), "obs": pack_obs(obs)}


def entry(path, **kw):
    return {"sha256": sha256_file(path), "bytes": os.path.getsize(path), **kw}


def manifest(out):
    return {
        "format": FORMAT,
        "kind": "phys10",
        "args": {
            "distance": 5, "rounds": 5, "p": CHOSEN_SI1000_P, "device_seed": 0,
            "n_substeps": 1500, "shots_per_substep": SHOTS,
            "regimes": list(PHYS_REGIMES), "stream_seeds": [0, 1, 2, 3, 4],
            "calibration_shots": CALIBRATION_SHOTS,
            "stim_seed_rule": 'blake2b(f"{regime}|{stream_seed}|{substep}", 8 bytes, big-endian); '
                               "calibration: blake2b('phys10|calibration')",
            "drift_rule": "cosmic: max burst >= 0.005; tls: any fluctuator on; coupler: any coupler on; "
                          "slow: all; mixed: cosmic or tls or coupler",
        },
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__, "stim": stim.__version__,
            "physics_noise_sha256": sha256_file("physics_noise.py"),
        },
        "files": {},
    }


def done(man, out, filename):
    """A file is finished when it exists, is listed and its hash matches."""
    full = os.path.join(out, filename)
    e = man["files"].get(filename)
    return e is not None and os.path.exists(full) and sha256_file(full) == e["sha256"]


def write_manifest(man, out):
    tmp = os.path.join(out, "manifest.json.tmp")
    with open(tmp, "w") as f:
        json.dump(man, f, indent=1)
    os.replace(tmp, os.path.join(out, "manifest.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/phys10")
    ap.add_argument("--regimes", default=",".join(PHYS_REGIMES))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--substeps", type=int, default=1500)
    ap.add_argument("--calibration", action="store_true", default=True)
    args = ap.parse_args()
    regimes = [r for r in args.regimes.split(",") if r]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    os.makedirs(args.out, exist_ok=True)
    man_path = os.path.join(args.out, "manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else manifest(args.out)
    base = noiseless_d5()

    if args.calibration and not done(man, args.out, "calibration.npz"):
        t0 = time.time()
        path = os.path.join(args.out, "calibration.npz")
        save(path, generate_calibration(base))
        man["files"]["calibration.npz"] = entry(path, kind="calibration")
        write_manifest(man, args.out)
        print(f"calibration.npz written in {time.time() - t0:.1f}s", flush=True)
    for regime in regimes:
        for seed in seeds:
            fn = stream_file(regime, seed)
            if done(man, args.out, fn):
                print(f"skip {fn} (exists)", flush=True)
                continue
            t0 = time.time()
            path = os.path.join(args.out, fn)
            save(path, generate_stream(base, regime, seed, args.substeps))
            man["files"][fn] = entry(path, kind="stream", regime=regime, stream_seed=seed,
                                     substeps=args.substeps)
            write_manifest(man, args.out)
            print(f"{fn} written in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
