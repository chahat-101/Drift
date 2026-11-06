"""Week-1 sanity check: reproduce the surface-code threshold plot.

Logical error rate per shot vs physical error rate, for several distances,
decoded with PyMatching. Below threshold (~0.6-0.7% for this noise model) the
larger codes win; above it they lose. If your curves cross, your setup works.

    python threshold.py                     # d = 3, 5, 7
    python threshold.py --max-shots 200000  # tighter error bars, slower
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sinter

from circuits import make_circuit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--ps", type=float, nargs="+", default=[0.002, 0.004, 0.006, 0.008, 0.010, 0.012])
    ap.add_argument("--max-shots", type=int, default=50_000)
    ap.add_argument("--max-errors", type=int, default=500)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    tasks = [
        sinter.Task(circuit=make_circuit(d, d, p), json_metadata={"d": d, "p": p})
        for d in args.distances for p in args.ps
    ]
    stats = sinter.collect(
        num_workers=os.cpu_count() or 2,
        tasks=tasks,
        decoders=["pymatching"],
        max_shots=args.max_shots,
        max_errors=args.max_errors,
    )
    for s in sorted(stats, key=lambda s: (s.json_metadata["d"], s.json_metadata["p"])):
        m = s.json_metadata
        print(f"d={m['d']} p={m['p']:.3f}  shots={s.shots:>7}  errors={s.errors:>4}  LER={s.errors / s.shots:.5f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    sinter.plot_error_rate(ax=ax, stats=stats, x_func=lambda s: s.json_metadata["p"],
                           group_func=lambda s: f"d={s.json_metadata['d']}")
    ax.set_xlabel("physical error rate p")
    ax.set_ylabel("logical error rate per shot (rounds = d)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(which="both", alpha=0.3)
    ax.legend()
    os.makedirs(args.out, exist_ok=True)
    fig.savefig(os.path.join(args.out, "threshold.png"), dpi=120)
    print(f"wrote {args.out}/threshold.png")


if __name__ == "__main__":
    main()
