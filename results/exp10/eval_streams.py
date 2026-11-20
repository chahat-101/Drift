"""Evaluate every round-10 decoder on the data/phys10 streams.

    python results/exp10/eval_streams.py                 # all 30 streams, all decoders
    python results/exp10/eval_streams.py --regimes phys_tls --seeds 0

Decoders per stream: calibrated static MWPM, nominal MWPM (reference only), oracle MWPM
(true DEM per sub-step, cached per distinct circuit), WindowDEM 500/1000/2000/5000/10000 on
the calibrated DEM's matching graph, and the four neural ensembles (+ single seeds) via
ensemble_eval's path: mean of the 3 seeds' logits > 0, causal 200-shot sub-batches, batch
16384, cuDNN TF32 on / matmul TF32 off. Adaptive models run live (own 2000-shot window)
and frozen (fingerprint of data/phys10/calibration.npz).

Writes results/exp10/errors/<regime>_s<seed>.npz (packed per-shot error bits, key per
decoder, shape sub-steps x 25) and results/exp10/eval_summary.json (LERs on all / drift /
nominal sub-steps and per burst bin). Causal everywhere: each 200-shot sub-batch is decoded
first, then observed. Resumable: finished streams are skipped. Never touches data/phys10.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

import ensemble_eval  # noqa: E402,F401  (imports set the TF32 flags of the exp6 path)
from ensemble_eval import Gpu, errors, logits, pack, window_fps  # noqa: E402
from baselines import WindowDEM  # noqa: E402
from circuits import make_circuit, matcher_from_circuit  # noqa: E402
from dataset import unpack_det  # noqa: E402
from decoder import DetectorGrid, Fingerprint, load_checkpoint  # noqa: E402
from generate_streams import noiseless_d5  # noqa: E402
from physics_noise import (  # noqa: E402
    CHOSEN_SI1000_P,
    PHYS_REGIMES,
    NoiseModel,
    build_noisy_circuit,
    make_noise_state,
)

N_DET = 120
SHOTS = 200
WINDOWS = (500, 1000, 2000, 5000, 10000)
BATCH = 16384
UPDATE = 200  # shots per causal sub-batch
TYPES = ("static_wide", "adaptive_wide", "adaptive_stale", "static_nominal")
DATA = "data/phys10"
OUT = "results/exp10"


def ckpt_path(t, s):
    return f"ckpt/decoder_{t}_d5_long_s{s}.pt"


def calibrated_circuit(base):
    """The time-invariant static device: the calibrated DEM's circuit."""
    state = make_noise_state(base, "phys_static", device_seed=0, stream_seed=0, n_substeps=1)
    return build_noisy_circuit(base, state.model(CHOSEN_SI1000_P, 0))


def substep_model(arrays, t):
    """The true NoiseModel of sub-step t, rebuilt from the saved stream arrays."""
    qubits = arrays["qubits"]
    m = {int(q): float(arrays["m"][t, i]) for i, q in enumerate(qubits)}
    burst = {int(q): float(arrays["burst"][t, i]) for i, q in enumerate(qubits) if arrays["burst"][t, i] > 0}
    coupler = {tuple(int(x) for x in pair): 0.02 for i, pair in enumerate(arrays["coupler_pairs"])
               if arrays["coupler_on"][t, i]}
    return NoiseModel("si1000", CHOSEN_SI1000_P, m=m, burst=burst, coupler=coupler)


def decode_flat(matching, det, chunk=50000):
    """(T, SHOTS) predicted flips from a pymatching Matching on unpacked det (T, SHOTS, n)."""
    flat = det.reshape(-1, det.shape[-1])
    out = np.empty(len(flat), dtype=bool)
    for a in range(0, len(flat), chunk):
        out[a : a + chunk] = np.asarray(matching.decode_batch(flat[a : a + chunk])).reshape(
            len(flat[a : a + chunk]), -1
        )[:, 0]
    return out.reshape(det.shape[:2])


def mwpm_errors(matching, det):
    return decode_flat(matching, det)


def oracle_errors(base, arrays, det):
    """True-DEM MWPM, one circuit per sub-step, cached per distinct circuit."""
    preds = np.empty(det.shape[:2], dtype=bool)
    cache = {}
    for t in range(len(det)):
        circuit = build_noisy_circuit(base, substep_model(arrays, t))
        key = hashlib.sha1(str(circuit).encode()).hexdigest()
        if key not in cache:
            cache[key] = matcher_from_circuit(circuit)
        preds[t] = np.asarray(cache[key].decode_batch(det[t])).reshape(len(det[t]), -1)[:, 0]
    return preds, len(cache)


def window_errors(circuit, det, window):
    """Causal WindowDEM: decode each sub-batch, then observe it."""
    wd = WindowDEM(circuit, window=window)
    preds = np.empty(det.shape[:2], dtype=bool)
    for t in range(len(det)):
        preds[t] = np.asarray(wd.decode(det[t])).reshape(len(det[t]), -1)[:, 0]
        wd.observe(det[t])
    return preds


def neural_errors(ckpts, gpu, grid, det_packed, obs_packed, frozen_bits, update=UPDATE, batch=BATCH):
    """{(kind, who): (T, shots) error bits} of one model type, on ensemble_eval's path."""
    dev = gpu.device
    steps, shots = det_packed.shape[:2]
    det = gpu.bits(torch.from_numpy(det_packed).to(dev), gpu.n_det).reshape(steps * shots, -1)
    obs = gpu.bits(torch.from_numpy(obs_packed).to(dev).transpose(1, 2), shots)[:, 0].reshape(-1).bool()
    starts = [t * shots + i for t in range(steps) for i in range(0, shots, update)]
    sub = torch.repeat_interleave(
        torch.arange(len(starts), device=dev),
        torch.diff(torch.tensor(starts + [steps * shots], device=dev)),
    )
    models, cfgs = [], []
    for ck in ckpts:
        m, cfg = cached_model(ck, dev)
        models.append(m)
        cfgs.append(cfg)
    fpr = Fingerprint(cfgs[0], grid, dev) if cfgs[0]["adaptive"] else None
    out = {}
    if fpr is None:
        for who, v in errors(logits(models, gpu, det, batch=batch), obs).items():
            out["live", who] = v.view(steps, shots)
        return out
    lg = torch.empty(len(models), len(det), device=dev)
    for wnd in sorted({c["window"] for c in cfgs}):
        ks = [k for k, c in enumerate(cfgs) if c["window"] == wnd]
        fpg = gpu.fp_grid(window_fps(det, starts, wnd, fpr))
        lg[ks] = logits([models[k] for k in ks], gpu, det, fpg, sub, batch)
    zero = torch.zeros(len(det), dtype=torch.long, device=dev)
    frozen_grid = gpu.fp_grid(fpr.of_windows(frozen_bits)[None])
    lgf = logits(models, gpu, det, frozen_grid, zero, batch)
    for who, v in errors(lg, obs).items():
        out["live", who] = v.view(steps, shots)
    for who, v in errors(lgf, obs).items():
        out["frozen", who] = v.view(steps, shots)
    return out


def lers(bits, drift, bins):
    """LER on all / drift / nominal sub-steps and the three burst bins."""
    out = {"all": float(bits.mean())}
    for name, mask in (("drift", drift), ("nominal", ~drift)):
        out[name] = float(bits[mask].mean()) if mask.any() else None
    for b in (0, 1, 2):
        m = bins == b
        out[f"bin{b}"] = float(bits[m].mean()) if m.any() else None
    return out


_MODEL_CACHE = {}


def cached_model(path, dev):
    if path not in _MODEL_CACHE:
        m, cfg = load_checkpoint(path, dev)
        _MODEL_CACHE[path] = (m.to(dev), cfg)
    return _MODEL_CACHE[path]


def eval_stream(stream, base, cal_circuit, mwpm_cal, mwpm_nom, gpu, grid, frozen_bits):
    """{decoder name: packed error bits} + LERs + timing for one stream's arrays."""
    det = unpack_det(stream["det"], N_DET)
    obs = np.unpackbits(stream["obs"], axis=-2, count=SHOTS)[..., 0]
    drift, bins = stream["drift"], stream["burst_bin"]
    bits, secs = {}, {}

    t0 = time.time()
    bits["mwpm_cal"] = np.packbits(mwpm_errors(mwpm_cal, det) != obs, axis=-1)
    secs["mwpm_cal"] = time.time() - t0
    t0 = time.time()
    bits["mwpm_nom"] = np.packbits(mwpm_errors(mwpm_nom, det) != obs, axis=-1)
    secs["mwpm_nom"] = time.time() - t0
    t0 = time.time()
    oracle, n_circuits = oracle_errors(base, stream, det)
    bits["mwpm_oracle"] = np.packbits(oracle != obs, axis=-1)
    secs["mwpm_oracle"] = time.time() - t0
    for w in WINDOWS:
        t0 = time.time()
        bits[f"window_{w}"] = np.packbits(window_errors(cal_circuit, det, w) != obs, axis=-1)
        secs[f"window_{w}"] = time.time() - t0

    for t in TYPES:
        cks = [ckpt_path(t, s) for s in (0, 1, 2)]
        t0 = time.time()
        neural = neural_errors(cks, gpu, grid, stream["det"], stream["obs"], frozen_bits)
        secs[t] = time.time() - t0
        for (kind, who), v in neural.items():
            bits[f"{t}_{kind}_{who}"] = pack(v.cpu())
    summary = {k: lers(np.unpackbits(v, axis=-1), drift, bins) for k, v in bits.items()}
    return bits, summary, secs, n_circuits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default=",".join(PHYS_REGIMES))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()
    regimes, seeds = args.regimes.split(","), [int(s) for s in args.seeds.split(",")]
    os.makedirs(os.path.join(args.out, "errors"), exist_ok=True)
    summary_path = os.path.join(args.out, "eval_summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {"streams": {}}

    base = noiseless_d5()
    cal_circuit = calibrated_circuit(base)
    mwpm_cal = matcher_from_circuit(cal_circuit)
    mwpm_nom = matcher_from_circuit(make_circuit(5, 5, 0.003))
    nominal = make_circuit(5, 5, 0.003)
    grid = DetectorGrid(nominal)
    gpu = Gpu(grid, N_DET, "cuda")
    with np.load(os.path.join(args.data, "calibration.npz")) as cal:
        frozen_bits = gpu.bits(torch.from_numpy(cal["det"]).to("cuda"), N_DET)

    for regime in regimes:
        for seed in seeds:
            name = f"{regime}|s{seed}"
            if name in summary["streams"]:
                print(f"skip {name} (finished)", flush=True)
                continue
            fn = os.path.join(args.data, f"stream_{regime}_s{seed}.npz")
            with np.load(fn) as z:
                stream = {k: z[k] for k in z.files}
            t0 = time.time()
            bits, lers_, secs, n_circuits = eval_stream(
                stream, base, cal_circuit, mwpm_cal, mwpm_nom, gpu, grid, frozen_bits
            )
            out = os.path.join(args.out, "errors", f"{regime}_s{seed}.npz")
            np.savez_compressed(out + ".tmp.npz", **bits)
            os.replace(out + ".tmp.npz", out)
            summary["streams"][name] = {"lers": lers_, "seconds": secs,
                                        "oracle_circuits": n_circuits, "total_seconds": round(time.time() - t0, 1)}
            with open(summary_path + ".tmp", "w") as f:
                json.dump(summary, f, indent=1)
            os.replace(summary_path + ".tmp", summary_path)
            print(f"{name}: {time.time() - t0:.0f}s (oracle {n_circuits} circuits, "
                  f"oracle {secs['mwpm_oracle']:.0f}s, windows {sum(v for k, v in secs.items() if k.startswith('window')):.0f}s, "
                  f"neural {sum(v for k, v in secs.items() if k in TYPES):.0f}s)", flush=True)


if __name__ == "__main__":
    main()
