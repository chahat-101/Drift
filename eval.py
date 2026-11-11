"""Run every decoder on the same drifting syndrome stream and compare.

    python eval.py --regime hot_qubit
    python eval.py --regime bursts --distance 5 --steps 200
    python eval.py --regime telegraph --checkpoint decoder_adaptive_d3.pt --distance 3 --rounds 3
    python eval.py --regime bursts --data data/d3_r3_p0.003_s0 --checkpoint decoder_adaptive_d3.pt
    python eval.py --regime bursts --data data/d3_r3_p0.003_s0 --update-every 200 --tag s0 ...
    python eval.py --regime bursts --data data/d3_r3_p0.003_s0 --stream-seed 2 --frozen-fingerprint \
        --window-sizes 500,1000,2000,5000,10000 --checkpoint decoder_adaptive_d3_wide_s0.pt

Each step is decoded in sub-batches of --update-every shots. Every decoder decodes a
sub-batch and then observes it, so decoders only learn from the past. The default is
one sub-batch per step.

With --data the stream is replayed from the dataset instead of sampled, so every run
sees the same shots. Distance, rounds, p, steps and shots per step come from the dataset.

Neural decoders are named after their checkpoint file, without .pt.

Options:
  --stream-seed N         replay extra stream N of the dataset (default: the main stream)
  --window-sizes a,b,...  one WindowDEM per history size, named window_<size>
                          (default: one WindowDEM of --window shots, named "window")
  --frozen-fingerprint    also run each adaptive checkpoint with the nominal test-set
                          fingerprint held fixed, named <checkpoint>_frozen

Metrics, printed and written to the summary JSON:
  LER          whole stream
  LER (drift)  steps where the noise is off-nominal
  onset        first off-nominal step after a nominal one
  later        the other off-nominal steps
  after        first nominal step after an off-nominal one
  nominal      all nominal steps

Writes <regime>_d<d>_r<rounds>_u<update-every>[_st<stream seed>][_<tag>].csv, .json
and .png to --out, plus _errors.npz with each decoder's per-shot error bits (packed
along the shot axis, shape steps x shots/8) and the phase masks, for paired tests.
"""

import argparse
import csv
import json
import os

import numpy as np

from baselines import Oracle, Static, WindowDEM
from dataset import Dataset, sha256_file
from drift import REGIMES, DriftStream, sample_stream

DEFAULTS = {"distance": 5, "p": 0.003, "steps": 150, "shots_per_step": 2000}
FROM_DATA = {
    "distance": "distance",
    "rounds": "rounds",
    "p": "p",
    "steps": "stream_steps",
    "shots_per_step": "shots_per_step",
}
PHASES = ["all", "drift", "onset", "later", "after", "nominal"]


def phase_masks(rows):
    """Boolean step masks for each phase in PHASES."""
    drift = np.array([(r["scale"] != 1.0) or r["n_hot"] > 0 for r in rows])
    before = np.concatenate([[False], drift[:-1]])  # previous step was drifted
    has_prev = np.arange(len(rows)) > 0
    onset = drift & ~before & has_prev
    return {
        "all": np.ones(len(rows), bool),
        "drift": drift,
        "onset": onset,
        "later": drift & ~onset,
        "after": ~drift & before,
        "nominal": ~drift,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=REGIMES, default="hot_qubit")
    ap.add_argument("--data", default=None, help="dataset folder from make_dataset.py: replay its stream")
    ap.add_argument("--distance", type=int, default=None, help="default 5, or from --data")
    ap.add_argument("--rounds", type=int, default=None, help="default: = distance, or from --data")
    ap.add_argument("--p", type=float, default=None, help="default 0.003, or from --data")
    ap.add_argument("--steps", type=int, default=None, help="default 150, or from --data")
    ap.add_argument("--shots-per-step", type=int, default=None, help="default 2000, or from --data")
    ap.add_argument(
        "--update-every",
        type=int,
        default=None,
        help="shots decoded between observe() calls (default: = shots per step)",
    )
    ap.add_argument("--window", type=int, default=10000, help="WindowDEM history, in shots")
    ap.add_argument("--window-sizes", default="", help="comma-separated WindowDEM histories, e.g. 500,2000")
    ap.add_argument("--stream-seed", type=int, default=None, help="extra stream to replay (--data only)")
    ap.add_argument(
        "--frozen-fingerprint",
        action="store_true",
        help="also run adaptive checkpoints with a fixed nominal fingerprint (--data only)",
    )
    ap.add_argument("--checkpoint", action="append", default=[], help="neural decoder .pt (repeatable)")
    ap.add_argument("--seed", type=int, default=0, help="ignored with --data")
    ap.add_argument("--device", default="cpu", help="torch device for neural decoders, e.g. cuda")
    ap.add_argument("--tag", default="", help="suffix for output filenames, e.g. s0")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    ds = None
    if args.data:
        ds = Dataset(args.data)
        ds.fill_args(args, FROM_DATA)
    for k, v in DEFAULTS.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    rounds = args.rounds or args.distance
    update = args.update_every or args.shots_per_step
    if (args.stream_seed is not None or args.frozen_fingerprint) and not ds:
        raise SystemExit("--stream-seed and --frozen-fingerprint need --data")

    stream = DriftStream(args.distance, rounds, args.p)
    nominal = stream.nominal()
    if ds:
        steps = ds.stream(args.regime, args.stream_seed or 0)
    else:
        rng = np.random.default_rng(args.seed)
        steps = sample_stream(stream, args.regime, args.steps, args.shots_per_step, rng)

    decoders = [Static(nominal)]
    if args.window_sizes:
        for w in (int(x) for x in args.window_sizes.split(",")):
            wd = WindowDEM(nominal, window=w, min_shots=min(w, 1000))
            wd.name = f"window_{w}"
            decoders.append(wd)
    else:
        decoders.append(WindowDEM(nominal, window=args.window))
    decoders.append(Oracle(stream))
    frozen = ds.nominal_fingerprint() if args.frozen_fingerprint else None
    for ck in args.checkpoint:
        from decoder import NeuralDecoder  # torch is needed only with --checkpoint

        name = os.path.splitext(os.path.basename(ck))[0]
        nd = NeuralDecoder(ck, nominal, name=name, device=args.device)
        if ds:
            ds.check_model(nd.cfg, ck)
        if (nd.cfg["distance"], nd.cfg["rounds"]) != (args.distance, rounds):
            raise SystemExit(f"{ck} was trained for d={nd.cfg['distance']} r={nd.cfg['rounds']}")
        decoders.append(nd)
        if frozen is not None and nd.adaptive:
            decoders.append(
                NeuralDecoder(ck, nominal, name=f"{name}_frozen", device=args.device, frozen_fingerprint=frozen)
            )
    names = [d.name for d in decoders]

    if len(set(names)) != len(names):
        raise SystemExit(f"decoder names must be unique: {names}")

    rows, errs, shot_err = [], {n: [] for n in names}, {n: [] for n in names}
    for t, (state, det, obs) in enumerate(steps):
        row = {"step": t, "scale": state.scale, "n_hot": len(state.hot_qubits), "hot_p": state.hot_p}
        step_err = {n: [] for n in names}
        for i in range(0, len(det), update):
            sub_det, sub_obs = det[i : i + update], obs[i : i + update]
            for d in decoders:
                step_err[d.name].append(np.any(d.decode(sub_det, state) != sub_obs, axis=1))
            for d in decoders:
                d.observe(sub_det)
        for n in names:
            e = np.concatenate(step_err[n])
            shot_err[n].append(e)
            errs[n].append(int(e.sum()))
            row[n] = errs[n][-1] / len(det)
        rows.append(row)

    os.makedirs(args.out, exist_ok=True)
    tag = (
        f"{args.regime}_d{args.distance}_r{rounds}_u{update}"
        + (f"_st{args.stream_seed}" if args.stream_seed is not None else "")
        + (f"_{args.tag}" if args.tag else "")
    )
    with open(os.path.join(args.out, f"{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    masks = phase_masks(rows)
    np.savez_compressed(
        os.path.join(args.out, f"{tag}_errors.npz"),
        **{n: np.packbits(np.stack(shot_err[n]), axis=-1) for n in names},
        **{f"phase_{ph}": m for ph, m in masks.items()},
    )
    shots = {ph: int(m.sum()) * args.shots_per_step for ph, m in masks.items()}
    summary = {
        "regime": args.regime,
        "distance": args.distance,
        "rounds": rounds,
        "p": args.p,
        "steps": args.steps,
        "shots_per_step": args.shots_per_step,
        "update_every": update,
        "window": args.window,
        "window_sizes": args.window_sizes,
        "data": args.data,
        "stream_seed": args.stream_seed,
        "frozen_fingerprint": args.frozen_fingerprint,
        "manifest_sha256": sha256_file(os.path.join(args.data, "manifest.json")) if ds else None,
        "seed": None if ds else args.seed,
        "checkpoints": args.checkpoint,
        "phase_steps": {ph: int(m.sum()) for ph, m in masks.items()},
        "phase_shots": shots,
        "decoders": {},
    }
    print(
        f"regime={args.regime} d={args.distance} rounds={rounds} p={args.p} steps={args.steps} "
        f"shots/step={args.shots_per_step} update-every={update} drifted steps={masks['drift'].sum()} "
        f"onset={masks['onset'].sum()} after={masks['after'].sum()}"
    )
    print(f"{'decoder':<38}{'LER':>10}{'LER (drift)':>13}{'onset':>10}{'later':>10}{'after':>10}{'nominal':>10}")
    for n in names:
        e = np.array(errs[n])
        err = {ph: int(e[m].sum()) for ph, m in masks.items()}
        ler = {ph: err[ph] / shots[ph] if shots[ph] else None for ph in PHASES}
        summary["decoders"][n] = {"errors": err, "ler": ler}
        cells = "".join(
            f"{'-' if ler[ph] is None else f'{ler[ph]:.5f}':>{w}}" for ph, w in zip(PHASES, (10, 13, 10, 10, 10, 10))
        )
        print(f"{n:<38}{cells}")
    with open(os.path.join(args.out, f"{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    smooth = lambda x, k=5: np.convolve(x, np.ones(k) / k, mode="same")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, height_ratios=[3, 1])
    for n in names:
        a1.plot(smooth(np.array([r[n] for r in rows])), label=n)
    a1.set_ylabel("logical error rate (smoothed)")
    a1.set_yscale("log")
    a1.legend()
    a1.set_title(tag)
    a2.plot([r["scale"] for r in rows], label="noise scale")
    a2.plot([1 + r["n_hot"] * r["hot_p"] * 10 for r in rows], label="hot-qubit load", alpha=0.7)
    a2.set_xlabel("step")
    a2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"{tag}.png"), dpi=120)
    print(f"wrote {args.out}/{tag}.csv, .json, .png and _errors.npz")


if __name__ == "__main__":
    main()
