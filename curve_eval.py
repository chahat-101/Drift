"""Score neural checkpoints on the test sets and saved streams, for learning curves.

    python curve_eval.py --checkpoint decoder_adaptive_d5_long_s0_step30000.pt --out results/exp4/curves

Only the neural decoders run. Streams are replayed as eval.py does (decode a
sub-batch of --update-every shots, then observe it), live and, for adaptive models,
with the fingerprint frozen at the nominal one. Static MWPM, WindowDEM and oracle
MWPM are deterministic, so their per-shot errors are read from existing eval.py
output (--baselines). On the test sets, static and oracle MWPM are computed once and
cached in <out>/baselines.npz.

Writes <out>/<checkpoint>.json (LERs) and <checkpoint>_errors.npz (per-shot error bits,
packed along the shot axis). Checkpoints whose JSON exists are skipped.
"""

import argparse
import json
import os

import numpy as np

from baselines import Oracle, Static
from dataset import Dataset
from decoder import NeuralDecoder
from drift import DriftStream
from eval import PHASES


def test_baselines(ds, stream, path):
    """Per-shot static and oracle MWPM errors on each test set, cached in `path`."""
    if os.path.exists(path):
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    static, oracle, out = Static(stream.nominal()), Oracle(stream), {}
    for name, st, det, obs, _ in ds.test_sets():
        out[f"{name}|static"] = static.decode(det)[:, 0] != obs[:, 0]
        out[f"{name}|oracle"] = oracle.decode(det, st)[:, 0] != obs[:, 0]
    np.savez_compressed(path, **out)
    return out


def replay(dec, steps, update):
    """Per-shot errors of one decoder on a saved stream, shape (steps, shots), computed as in eval.py."""
    out = []
    for _, det, obs in steps:
        errs = []
        for i in range(0, len(det), update):
            errs.append(np.any(dec.decode(det[i : i + update]) != obs[i : i + update], axis=1))
            dec.observe(det[i : i + update])
        out.append(np.concatenate(errs))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", required=True)
    ap.add_argument("--data", default="data/d5_wide", help="dataset with the test sets and streams")
    ap.add_argument("--streams", default="bursts_big,bursts")
    ap.add_argument("--stream-seed", type=int, default=0)
    ap.add_argument("--update-every", type=int, default=200)
    ap.add_argument("--baselines", default="results/exp3/d5", help="folder of eval.py _errors.npz files")
    ap.add_argument("--baseline-tag", default="s0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp4/curves")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ds = Dataset(args.data)
    a = ds.args
    stream = DriftStream(a["distance"], a["rounds"], a["p"])
    nominal, frozen = stream.nominal(), ds.nominal_fingerprint()
    tests = ds.test_sets()
    tbase = test_baselines(ds, stream, os.path.join(args.out, "baselines.npz"))
    streams = {r: ds.stream(r, args.stream_seed) for r in args.streams.split(",")}

    for ck in args.checkpoint:
        name = os.path.splitext(os.path.basename(ck))[0]
        js = os.path.join(args.out, f"{name}.json")
        if os.path.exists(js):
            continue
        dec = NeuralDecoder(ck, nominal, name=name, device=args.device)
        ds.check_model(dec.cfg, ck)
        summary = {"checkpoint": ck, "step": dec.cfg.get("step", dec.cfg["steps"]), "tests": {}, "streams": {}}
        errs = {}
        for tname, _, det, obs, wdet in tests:
            dec.frozen = wdet.mean(axis=0).astype(np.float32) if dec.adaptive else None
            e = dec.decode(det)[:, 0] != obs[:, 0]
            errs[f"test|{tname}"] = e
            summary["tests"][tname] = {
                "neural": float(e.mean()),
                "static": float(tbase[f"{tname}|static"].mean()),
                "oracle": float(tbase[f"{tname}|oracle"].mean()),
            }
        for r, steps in streams.items():
            base = np.load(
                os.path.join(
                    args.baselines,
                    f"{r}_d{a['distance']}_r{a['rounds']}_u{args.update_every}"
                    f"_st{args.stream_seed}_{args.baseline_tag}_errors.npz",
                )
            )
            masks = {ph: base[f"phase_{ph}"] for ph in PHASES}
            runs = {"live": None} | ({"frozen": frozen} if dec.adaptive else {})
            res = {}
            for kind, fp in runs.items():
                d = NeuralDecoder(ck, nominal, name=name, device=args.device, frozen_fingerprint=fp)
                e = replay(d, steps, args.update_every)
                errs[f"{r}|{kind}"] = np.packbits(e, axis=-1)
                res[kind] = e
            for k in base.files:
                if not k.startswith("phase_") and not k.startswith("decoder_"):
                    res[k] = np.unpackbits(base[k], axis=-1, count=len(steps[0][1])).astype(bool)
            summary["streams"][r] = {k: {ph: float(e[m].mean()) for ph, m in masks.items()} for k, e in res.items()}
        np.savez_compressed(os.path.join(args.out, f"{name}_errors.npz"), **errs)
        with open(js, "w") as f:
            json.dump(summary, f, indent=1)
        t = summary["tests"]["nominal"]
        bb = summary["streams"].get("bursts_big", {}).get("live", {})
        print(
            f"{name}: nominal {t['neural']:.5f} (MWPM {t['static']:.5f})  bursts_big drift {bb.get('drift', 0):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
