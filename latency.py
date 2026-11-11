"""Decode latency of every decoder on this machine.

    python latency.py --data data/d3_wide --checkpoint decoder_static_d3_wide_s0.pt \
        --checkpoint decoder_adaptive_d3_wide_s0.pt --out results/exp3/latency_d3.json

For each decoder, batch size and device, decode() is called repeatedly on nominal
test-set shots (a different slice each call) and each call is timed with a wall clock.
Times are reported per call, per shot (call / batch) and per round (call / batch /
rounds), as mean and 99th percentile.

  MWPM       static PyMatching (CPU only)
  WindowDEM  decode with reweighted weights, plus the reweighting cost per update
             (estimate and new Matching), timed separately for each history
  neural     end to end: numpy grid layout, host to device copy, forward pass, copy
             back. Adaptive models include the fingerprint (mean over their 2000-shot
             buffer) on every call.

The numbers include Python overhead and depend on the machine.
"""

import argparse
import json
import os
import platform
import time

import numpy as np

from baselines import Static, WindowDEM
from dataset import Dataset
from drift import DriftStream

REPS = {1: 1000, 64: 300, 4096: 100}


def time_calls(fn, det, batch, reps, warmup=10):
    """Wall-clock seconds of `reps` calls of fn on slices of `batch` shots, after a warm-up."""
    n, out = len(det), []
    for i in range(warmup + reps):
        j = (i * batch) % (n - batch)
        x = det[j : j + batch]
        t = time.perf_counter()
        fn(x)
        if i >= warmup:
            out.append(time.perf_counter() - t)
    return np.array(out)


def summary(times, batch, rounds):
    """Mean and p99 in microseconds per call, per shot and per round."""
    us = times * 1e6
    out = {}
    for unit, div in (("call", 1), ("shot", batch), ("round", batch * rounds)):
        out[unit] = {"mean": float(us.mean() / div), "p99": float(np.percentile(us, 99) / div)}
    return out


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            return next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except (OSError, StopIteration):
        return platform.processor() or platform.machine()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset whose nominal test shots are decoded")
    ap.add_argument("--checkpoint", action="append", default=[], help="neural decoder .pt (repeatable)")
    ap.add_argument("--batches", default="1,64,4096")
    ap.add_argument("--devices", default="cpu,cuda")
    ap.add_argument("--windows", default="2000,10000", help="WindowDEM histories for the reweighting cost")
    ap.add_argument("--out", required=True, help="JSON file for the results")
    args = ap.parse_args()
    import torch

    ds = Dataset(args.data)
    a = ds.args
    stream = DriftStream(a["distance"], a["rounds"], a["p"])
    nominal = stream.nominal()
    det = next(t for t in ds.test_sets() if t[0] == "nominal")[2]
    batches = [int(b) for b in args.batches.split(",")]
    rows = []

    def record(decoder, device, batch, times, **extra):
        r = {
            "decoder": decoder,
            "device": device,
            "batch": batch,
            "reps": len(times),
            **summary(times, batch, a["rounds"]),
            **extra,
        }
        rows.append(r)
        print(
            f"{decoder:<34}{device:<6}{batch:>6}  call {r['call']['mean']:>10.1f} us (p99 {r['call']['p99']:>10.1f})"
            f"  shot {r['shot']['mean']:>9.2f} us (p99 {r['shot']['p99']:>9.2f})  round {r['round']['mean']:>8.2f} us",
            flush=True,
        )

    static = Static(nominal)
    for b in batches:
        record("MWPM (static)", "cpu", b, time_calls(static.decode, det, b, REPS[b]))
    for w in (int(x) for x in args.windows.split(",")):
        wd = WindowDEM(nominal, window=w, min_shots=min(w, 1000))
        wd.observe(det[:w])  # fill the history so the graph is reweighted
        for b in batches:
            record(f"WindowDEM-{w} decode", "cpu", b, time_calls(wd.decode, det, b, REPS[b]))
        # cost of one update: estimate from the history and build a new Matching
        hist = det[:w]
        times = time_calls(lambda x: wd._reweighted(hist), det, 1, 100)
        record(f"WindowDEM-{w} reweight", "cpu", 1, times, per_update=True)

    from decoder import NeuralDecoder

    for ck in args.checkpoint:
        name = os.path.splitext(os.path.basename(ck))[0]
        for device in args.devices.split(","):
            if device == "cuda" and not torch.cuda.is_available():
                continue
            nd = NeuralDecoder(ck, nominal, name=name, device=device)
            ds.check_model(nd.cfg, ck)
            nd.observe(det[: nd.cfg["window"]])
            for b in batches:
                record(name, device, b, time_calls(nd.decode, det, b, REPS[b]))

    info = {
        "distance": a["distance"],
        "rounds": a["rounds"],
        "p": a["p"],
        "data": args.data,
        "cpu": cpu_name(),
        "cpu_threads": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(info, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
