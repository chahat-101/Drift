"""Seed-0 learning curves (nominal test, bursts_big drift st0) with reference lines, from existing curve JSONs."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

E4 = "results/exp4/curves"
MODELS = [
    ("static_wide", "static-wide", "#eda100"),
    ("adaptive_wide", "adaptive-wide", "#e87ba4"),
    ("adaptive_stale", "adaptive-stale", "#008300"),
    ("static_nominal", "static-nominal", "#6250d6"),
    ("adaptive_stale_h64l6", "adaptive-stale h64/l6", "#e34948"),
]
STEPS = range(30000, 300001, 30000)
J = {m: [json.load(open(f"{E4}/decoder_{m}_d5_long_s0_step{s}.json")) for s in STEPS] for m, _, _ in MODELS}
ref = J["static_wide"][0]
x = [s / 1000 for s in STEPS]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, title, get, refs in [
    (
        axs[0],
        "nominal test set",
        lambda j: j["tests"]["nominal"]["neural"],
        {"static MWPM = oracle": ref["tests"]["nominal"]["static"]},
    ),
    (
        axs[1],
        "bursts_big drift steps (stream seed 0, live)",
        lambda j: j["streams"]["bursts_big"]["live"]["drift"],
        {
            "static MWPM": ref["streams"]["bursts_big"]["static"]["drift"],
            "best WindowDEM": min(
                v["drift"] for k, v in ref["streams"]["bursts_big"].items() if k.startswith("window")
            ),
            "oracle MWPM": ref["streams"]["bursts_big"]["oracle"]["drift"],
        },
    ),
]:
    for (lab, y), c in zip(refs.items(), ["#2a78d6", "#eb6834", "#1baf7a"]):
        ax.axhline(y, color=c, ls="--", lw=1.5, label=lab)
    for m, lab, c in MODELS:
        ax.plot(x, [get(j) for j in J[m]], "o-", color=c, lw=2, ms=4, label=lab)
    ax.set(title=title, xlabel="training step (k)", ylabel="logical error rate", yscale="log")
    ax.grid(alpha=0.3)
axs[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig("results/exp4/summary_curves.png", dpi=130)
print("saved results/exp4/summary_curves.png")
