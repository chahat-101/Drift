"""Learning curves of the long d=5 runs, and tables for their final checkpoints.

    python analysis/learning_curves.py

Reads results/exp4/curves (curve_eval.py output) and results/exp4/full/st0-st4 (full evals of the final
checkpoints) and writes results/exp4/report4_tables.md, report4_data.json and PNG figures.
"""

import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from report import mcnemar_p, verdict  # noqa: E402

EXP, EXP3 = "results/exp4", "results/exp3/d5"
MODELS = {  # key -> display name
    "static_wide": "static-wide",
    "adaptive_wide": "adaptive-wide",
    "adaptive_stale": "adaptive-stale",
    "static_nominal": "static-nominal",
    "adaptive_stale_h64l6": "adaptive-stale h64/l6",
}
# 60k-step d=5 models (seeds 0-2, 1x dataset); their keys in the bits dicts are prefixed "r3:"
SHORT = {
    "static_wide": "decoder_static_d5_wide_s{s}",
    "adaptive_wide": "decoder_adaptive_d5_wide_s{s}",
    "adaptive_stale": "decoder_adaptive_d5_stale_s{s}",
    "static_nominal": "decoder_static_d5_nominal_fixed_s{s}",
}
COLORS = {
    "static MWPM": "#2a78d6",
    "best WindowDEM": "#eb6834",
    "oracle MWPM": "#1baf7a",
    "static-wide": "#eda100",
    "adaptive-wide": "#e87ba4",
    "adaptive-stale": "#008300",
    "static-nominal": "#6250d6",
    "adaptive-stale h64/l6": "#e34948",
}
TESTS = ["nominal", "scale x2", "2 hot qubits"]
PHASES = ["drift", "onset", "later", "after", "nominal"]
REGIMES = ["bursts", "bursts_big", "hot_qubit", "telegraph"]
WINDOWS = [500, 1000, 2000, 5000, 10000]


def name(m, s, step=None):
    return f"decoder_{m}_d5_long_s{s}" + (f"_step{step}" if step else "")


# ---------------------------------------------------------------- learning curves (seed 0, stream 0)


def curves():
    """{model: {step: curve_eval JSON}}"""
    out = {}
    for m in MODELS:
        for p in glob.glob(f"{EXP}/curves/{name(m, 0)}_step*.json"):
            j = json.load(open(p))
            out.setdefault(m, {})[int(re.search(r"_step(\d+)", p).group(1))] = j
    return {m: dict(sorted(v.items())) for m, v in out.items()}


def val_losses(m, s=0):
    steps, vals = [], []
    path = f"{EXP}/train_{m}_d5_long_s{s}.log"
    if os.path.exists(path):
        for line in open(path):
            r = re.match(r"step\s+(\d+)\s+loss \S+\s+val (\S+)", line)
            if r:
                steps.append(int(r.group(1)))
                vals.append(float(r.group(2)))
    return steps, vals


def style(ax, steps=True):
    from matplotlib.ticker import FuncFormatter

    if steps:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    ax.grid(True, color="#e6e5e1", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#52514e", labelsize=8)


def plot_curves(cv):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    any_j = next(iter(next(iter(cv.values())).values()))
    figs = []
    # stream phases
    for r in any_j["streams"]:
        fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
        for ax, ph in zip(axes, PHASES):
            b = any_j["streams"][r]
            refs = {
                "static MWPM": b["static"][ph],
                "oracle MWPM": b["oracle"][ph],
                "best WindowDEM": min(b[f"window_{w}"][ph] for w in WINDOWS),
            }
            for lab, v in refs.items():
                ax.axhline(v, color=COLORS[lab], lw=1.5, ls="--", label=lab)
            for m, c in cv.items():
                st = list(c)
                ax.plot(
                    st,
                    [c[s]["streams"][r]["live"][ph] for s in st],
                    color=COLORS[MODELS[m]],
                    lw=2,
                    marker="o",
                    ms=4,
                    label=MODELS[m],
                )
                if "frozen" in c[st[0]]["streams"][r]:
                    ax.plot(
                        st,
                        [c[s]["streams"][r]["frozen"][ph] for s in st],
                        color=COLORS[MODELS[m]],
                        lw=1.5,
                        ls=":",
                        marker="x",
                        ms=5,
                        label=f"{MODELS[m]} (frozen)",
                    )
            ax.set_title(f"{r}: {ph}", fontsize=10)
            ax.set_xlabel("training step", fontsize=8)
            ax.set_yscale("log")
            style(ax)
        axes[0].set_ylabel("LER (stream seed 0)")
        axes[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        fig.tight_layout()
        fn = f"{EXP}/report4_curve_{r}.png"
        fig.savefig(fn, dpi=110)
        plt.close(fig)
        figs.append(fn)
    # test sets + val loss
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, t in zip(axes, TESTS):
        tb = any_j["tests"][t]
        ax.axhline(tb["static"], color=COLORS["static MWPM"], lw=1.5, ls="--", label="static MWPM")
        ax.axhline(tb["oracle"], color=COLORS["oracle MWPM"], lw=1.5, ls="--", label="oracle MWPM")
        for m, c in cv.items():
            st = list(c)
            ax.plot(
                st,
                [c[s]["tests"][t]["neural"] for s in st],
                color=COLORS[MODELS[m]],
                lw=2,
                marker="o",
                ms=4,
                label=MODELS[m],
            )
        ax.set_title(f"test set: {t}", fontsize=10)
        ax.set_yscale("log")
        ax.set_xlabel("training step", fontsize=8)
        style(ax)
    axes[0].set_ylabel("LER (200000 shots)")
    for m in cv:
        s, v = val_losses(m)
        axes[3].plot(s, v, color=COLORS[MODELS[m]], lw=2, label=MODELS[m])
    axes[3].set_title("validation loss (own val set)", fontsize=10)
    axes[3].set_xlabel("training step", fontsize=8)
    style(axes[3])
    axes[3].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fn = f"{EXP}/report4_curve_tests.png"
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    return figs + [fn]


def curve_tables(cv):
    lines = []
    any_j = next(iter(next(iter(cv.values())).values()))
    lines.append("### Test-set LER vs step (seed 0)\n")
    for t in TESTS:
        tb = any_j["tests"][t]
        lines.append(f"**{t}** (static MWPM {tb['static']:.5f}, oracle MWPM {tb['oracle']:.5f})\n")
        steps = sorted({s for c in cv.values() for s in c})
        lines.append("| model | " + " | ".join(f"{s // 1000}k" for s in steps) + " |")
        lines.append("|---|" + "---|" * len(steps))
        for m, c in cv.items():
            lines.append(
                f"| {MODELS[m]} | "
                + " | ".join(f"{c[s]['tests'][t]['neural']:.5f}" if s in c else "–" for s in steps)
                + " |"
            )
        lines.append("")
    for r in any_j["streams"]:
        b = any_j["streams"][r]
        lines.append(f"### {r} (stream seed 0) LER vs step, live fingerprint\n")
        for ph in PHASES:
            bw = min(WINDOWS, key=lambda w: b[f"window_{w}"][ph])
            lines.append(
                f"**{ph}**: static MWPM {b['static'][ph]:.4f}, best WindowDEM ({bw}) "
                f"{b[f'window_{bw}'][ph]:.4f}, oracle {b['oracle'][ph]:.4f}\n"
            )
            steps = sorted({s for c in cv.values() for s in c})
            lines.append("| model | " + " | ".join(f"{s // 1000}k" for s in steps) + " |")
            lines.append("|---|" + "---|" * len(steps))
            for m, c in cv.items():
                lines.append(
                    f"| {MODELS[m]} | "
                    + " | ".join(f"{c[s]['streams'][r]['live'][ph]:.4f}" if s in c else "–" for s in steps)
                    + " |"
                )
                if "frozen" in next(iter(c.values()))["streams"][r]:
                    lines.append(
                        f"| {MODELS[m]} (frozen) | "
                        + " | ".join(f"{c[s]['streams'][r]['frozen'][ph]:.4f}" if s in c else "–" for s in steps)
                        + " |"
                    )
            lines.append("")
    return lines


def last_60k(cv):
    """Change from step 240000 to 300000 (stream 0 / test sets) with 2-SE verdicts."""
    z = np.load(f"{EXP3}/bursts_big_d5_r5_u200_st0_s0_errors.npz")
    nd = int(z["phase_drift"].sum()) * 2000
    rows = ["| model | nominal test 240k → 300k | | bursts_big drift (st0) 240k → 300k | |", "|---|---|---|---|---|"]
    for m, c in cv.items():
        if 240000 in c and 300000 in c:
            a, b = c[240000], c[300000]
            na, nb = a["tests"]["nominal"]["neural"], b["tests"]["nominal"]["neural"]
            da, db = a["streams"]["bursts_big"]["live"]["drift"], b["streams"]["bursts_big"]["live"]["drift"]
            rows.append(
                f"| {MODELS[m]} | {na:.5f} → {nb:.5f} | {verdict(nb, na, 200000)} | {da:.4f} → {db:.4f} | "
                f"{verdict(db, da, nd)} |"
            )
    return rows


# ---------------------------------------------------------------- final checkpoints (5 streams pooled)


def unpack(a):
    return np.unpackbits(a, axis=-1).astype(bool)


def final_bits():
    """{(label, seed): {(regime, phase): bits pooled over 5 streams}} for the long finals, the 60k-step models and the baselines."""
    seeds = {}
    for m in MODELS:
        for s in range(3):
            if all(os.path.exists(f"{EXP}/full/st{st}/{name(m, s)}.json") for st in range(5)):
                seeds.setdefault(m, []).append(s)
    bits = {}
    for r in REGIMES:
        base = [np.load(f"{EXP3}/{r}_d5_r5_u200_st{st}_s0_errors.npz") for st in range(5)]
        masks = [{ph: f[f"phase_{ph}"] for ph in PHASES} for f in base]

        def add(label, s, arrays):
            for ph in PHASES:
                bits.setdefault((label, s), {})[r, ph] = np.concatenate(
                    [a[mk[ph]].ravel() for a, mk in zip(arrays, masks)]
                )

        for k in ["static", "oracle"] + [f"window_{w}" for w in WINDOWS]:
            add(k, 0, [unpack(f[k]) for f in base])
        for m, ss in seeds.items():
            for s in ss:
                fs = [np.load(f"{EXP}/full/st{st}/{name(m, s)}_errors.npz") for st in range(5)]
                add(m, s, [unpack(f[f"{r}|live"]) for f in fs])
                if f"{r}|frozen" in fs[0]:
                    add(m + "|frozen", s, [unpack(f[f"{r}|frozen"]) for f in fs])
        for m, key in SHORT.items():
            for s in range(3):
                fs = [np.load(f"{EXP3}/{r}_d5_r5_u200_st{st}_s{s}_errors.npz") for st in range(5)]
                add("r3:" + m, s, [unpack(f[key.format(s=s)]) for f in fs])
                if "adaptive" in m:
                    add("r3:" + m + "|frozen", s, [unpack(f[key.format(s=s) + "_frozen"]) for f in fs])
    return bits, seeds


def best_w(bits, r, ph):
    if len(bits["static", 0][r, ph]) == 0:
        return WINDOWS[0]
    return min(WINDOWS, key=lambda w: bits[f"window_{w}", 0][r, ph].mean())


def cmp(bits, a, b, key, seeds):
    """Paired McNemar per seed of a vs b on one (regime, phase). Baselines have seed 0 only."""
    diffs, ps, low = [], [], 0
    if len(bits[a, seeds[0] if (a, seeds[0]) in bits else 0][key]) == 0:
        return "–", None
    for s in seeds:
        ea = bits[a, s if (a, s) in bits else 0][key]
        eb = bits[b, s if (b, s) in bits else 0][key]
        na, nb = int((ea & ~eb).sum()), int((~ea & eb).sum())
        diffs.append((na - nb) / len(ea))
        ps.append(mcnemar_p(na, nb))
        low += na < nb
    pa = np.mean([bits[a, s if (a, s) in bits else 0][key].mean() for s in seeds])
    pb = np.mean([bits[b, s if (b, s) in bits else 0][key].mean() for s in seeds])
    tie = verdict(pa, pb, len(ea)) == "tie"
    sig = "**" if max(ps) < 0.05 and low in (0, len(seeds)) else ""
    p = f"p={max(ps):.1g}" if max(ps) > 0 else "p<1e-300"
    return f"{sig}{np.mean(diffs):+.4f}{sig} {p} ({low}/{len(seeds)}){' tie' if tie else ''}", float(np.mean(diffs))


def m4(x):
    return "–" if np.isnan(x) else f"{x:.4f}"


def final_tables(bits, seeds):
    lines = []
    for r in REGIMES:
        lines.append(f"### {r}: final checkpoints, pooled over 5 streams\n")
        lines.append("| decoder | " + " | ".join(PHASES) + " |")
        lines.append("|---|" + "---|" * len(PHASES))
        rows = [("static MWPM", "static"), ("oracle MWPM", "oracle")]
        rows += [(f"WindowDEM-{w}", f"window_{w}") for w in WINDOWS]
        for lab, k in rows:
            lines.append(f"| {lab} | " + " | ".join(m4(bits[k, 0][r, ph].mean()) for ph in PHASES) + " |")
        for m, ss in seeds.items():
            for suf in ("", "|frozen"):
                if (m + suf, ss[0]) not in bits:
                    continue
                vals = [np.mean([bits[m + suf, s][r, ph].mean() for s in ss]) for ph in PHASES]
                lines.append(
                    f"| {MODELS[m]}{' (frozen)' if suf else ''} long ({len(ss)} seed{'s' * (len(ss) > 1)}) | "
                    + " | ".join(m4(v) for v in vals)
                    + " |"
                )
        for m in SHORT:
            for suf in ("", "|frozen"):
                if ("r3:" + m + suf, 0) in bits:
                    lines.append(
                        f"| {MODELS[m]}{' (frozen)' if suf else ''} report 3, seed 0 | "
                        + " | ".join(m4(bits["r3:" + m + suf, 0][r, ph].mean()) for ph in PHASES)
                        + " |"
                    )
        lines.append("")
        lines.append(
            f"Paired comparisons ({r}; diff = LER(a) − LER(b), negative = a better; best WindowDEM picked per phase):\n"
        )
        lines.append("| a vs b | " + " | ".join(PHASES) + " |")
        lines.append("|---|" + "---|" * len(PHASES))
        for m, ss in seeds.items():
            for lab, b in (("static MWPM", "static"), ("best WindowDEM", None), ("oracle MWPM", "oracle")):
                cells = [cmp(bits, m, b or f"window_{best_w(bits, r, ph)}", (r, ph), ss)[0] for ph in PHASES]
                lines.append(f"| {MODELS[m]} long vs {lab} | " + " | ".join(cells) + " |")
            if (m + "|frozen", ss[0]) in bits:
                lines.append(
                    f"| {MODELS[m]} long: live vs frozen | "
                    + " | ".join(cmp(bits, m, m + "|frozen", (r, ph), ss)[0] for ph in PHASES)
                    + " |"
                )
            if m in SHORT:
                lines.append(
                    f"| {MODELS[m]}: long vs report 3 (same seeds) | "
                    + " | ".join(cmp(bits, m, "r3:" + m, (r, ph), ss)[0] for ph in PHASES)
                    + " |"
                )
        if "adaptive_stale_h64l6" in seeds and "adaptive_stale" in seeds:
            ss = sorted(set(seeds["adaptive_stale_h64l6"]) & set(seeds["adaptive_stale"]))
            lines.append(
                "| h64/l6 vs h48/l4 (adaptive-stale, long) | "
                + " | ".join(cmp(bits, "adaptive_stale_h64l6", "adaptive_stale", (r, ph), ss)[0] for ph in PHASES)
                + " |"
            )
        lines.append("")
    return lines


def gap_table(bits, seeds):
    """Live minus frozen on drift steps, 60k-step models against long models."""
    lines = [
        "| model | regime | live − frozen drift, report 3 (60k) | live − frozen drift, long (300k) |",
        "|---|---|---|---|",
    ]
    for m in ("adaptive_wide", "adaptive_stale"):
        if m not in seeds:
            continue
        for r in REGIMES:
            a = cmp(bits, "r3:" + m, "r3:" + m + "|frozen", (r, "drift"), seeds[m])[0]
            b = cmp(bits, m, m + "|frozen", (r, "drift"), seeds[m])[0]
            lines.append(f"| {MODELS[m]} | {r} | {a} | {b} |")
    return lines


def test_final(seeds):
    """Final test-set LERs (from the full evals of stream seed 0) and paired tests against static MWPM."""
    tb = np.load(f"{EXP}/curves/baselines.npz")
    lines = ["| decoder | " + " | ".join(TESTS) + " |", "|---|" + "---|" * 3]
    lines.append("| static MWPM | " + " | ".join(f"{tb[f'{t}|static'].mean():.5f}" for t in TESTS) + " |")
    lines.append("| oracle MWPM | " + " | ".join(f"{tb[f'{t}|oracle'].mean():.5f}" for t in TESTS) + " |")
    cells = {}
    for m, ss in seeds.items():
        errs = [np.load(f"{EXP}/full/st0/{name(m, s)}_errors.npz") for s in ss]
        lines.append(
            f"| {MODELS[m]} long | "
            + " | ".join(f"{np.mean([e[f'test|{t}'].mean() for e in errs]):.5f}" for t in TESTS)
            + " |"
        )
        for t in TESTS:
            res = []
            for e in errs:
                a, b = e[f"test|{t}"], tb[f"{t}|static"]
                na, nb = int((a & ~b).sum()), int((~a & b).sum())
                res.append(((na - nb) / len(a), mcnemar_p(na, nb), na < nb))
            cells[m, t] = res
    lines.append("")
    lines.append("| long model vs static MWPM | " + " | ".join(TESTS) + " |")
    lines.append("|---|" + "---|" * 3)
    for m in seeds:
        row = []
        for t in TESTS:
            res = cells[m, t]
            d, p, k = np.mean([x[0] for x in res]), max(x[1] for x in res), sum(x[2] for x in res)
            sig = "**" if p < 0.05 and k in (0, len(res)) else ""
            row.append(f"{sig}{d:+.5f}{sig} p={p:.1g} ({k}/{len(res)})")
        lines.append(f"| {MODELS[m]} | " + " | ".join(row) + " |")
    pairs = [
        ("static_nominal", "static_wide"),
        ("adaptive_wide", "static_nominal"),
        ("adaptive_wide", "static_wide"),
        ("adaptive_stale_h64l6", "adaptive_stale"),
    ]
    lines += ["", "| a vs b (long models, common seeds) | " + " | ".join(TESTS) + " |", "|---|" + "---|" * 3]
    for a, b in pairs:
        ss = sorted(set(seeds.get(a, [])) & set(seeds.get(b, [])))
        if not ss:
            continue
        ea = [np.load(f"{EXP}/full/st0/{name(a, s)}_errors.npz") for s in ss]
        eb = [np.load(f"{EXP}/full/st0/{name(b, s)}_errors.npz") for s in ss]
        row = []
        for t in TESTS:
            res = []
            for x, y in zip(ea, eb):
                u, v = x[f"test|{t}"], y[f"test|{t}"]
                na, nb = int((u & ~v).sum()), int((~u & v).sum())
                res.append(((na - nb) / len(u), mcnemar_p(na, nb), na < nb))
            d, p, k = np.mean([r[0] for r in res]), max(r[1] for r in res), sum(r[2] for r in res)
            sig = "**" if p < 0.05 and k in (0, len(res)) else ""
            row.append(f"{sig}{d:+.5f}{sig} p={p:.1g} ({k}/{len(res)})")
        lines.append(f"| {MODELS[a]} vs {MODELS[b]} | " + " | ".join(row) + " |")
    return lines


def gpu_section():
    path = f"{EXP}/gpu_util.csv"
    if not os.path.exists(path):
        return []
    rows = [l.strip().split(",") for l in open(path).readlines()[1:]]
    h = np.array([float(r[1]) for r in rows])
    u = np.array([float(r[2]) for r in rows])
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.plot(h, u, color="#2a78d6", lw=1.5)
    ax.set_xlabel("hours since start of this round")
    ax.set_ylabel("GPU util % (1-min mean)")
    ax.set_ylim(0, 105)
    style(ax, steps=False)
    fig.tight_layout()
    fig.savefig(f"{EXP}/report4_gpu_util.png", dpi=110)
    plt.close(fig)
    hourly = [
        f"{int(k)}–{int(k) + 1} h: {u[(h >= k) & (h < k + 1)].mean():.0f}%"
        for k in range(int(h.max()) + 1)
        if ((h >= k) & (h < k + 1)).any()
    ]
    return [
        f"GPU utilisation, {len(u)} one-minute means from {h.min():.2f} h to {h.max():.2f} h: overall mean "
        f"**{u.mean():.0f}%**, {100 * (u >= 85).mean():.0f}% of minutes ≥ 85%, "
        f"{100 * (u < 70).mean():.0f}% of minutes < 70%. Hourly: " + "; ".join(hourly) + "\n",
        "![GPU utilisation](exp4/report4_gpu_util.png)\n",
    ]


def main():
    out = ["# Learning curves and final checkpoints, d=5 (generated by analysis/learning_curves.py)\n"]
    cv = curves()
    if cv:
        figs = plot_curves(cv)
        out += ["## Learning curves (seed 0, stream seed 0)\n"] + [
            f"![{f}](exp4/{os.path.basename(f)})\n" for f in figs
        ]
        out += curve_tables(cv)
        out += ["### Last 60k steps (2-SE rule; 'lower' = improved)\n"] + last_60k(cv) + [""]
    bits, seeds = final_bits()
    if seeds:
        out += ["## Final checkpoints\n", "### Test sets\n"] + test_final(seeds) + [""]
        out += final_tables(bits, seeds)
        out += ["### Live − frozen on drift steps: report 3 vs long\n"] + gap_table(bits, seeds) + [""]
    out += ["## GPU\n"] + gpu_section()
    open(f"{EXP}/report4_tables.md", "w").write("\n".join(out) + "\n")
    json.dump(
        {"seeds": seeds, "curve_steps": {m: list(c) for m, c in cv.items()}},
        open(f"{EXP}/report4_data.json", "w"),
        indent=1,
    )
    print(f"wrote {EXP}/report4_tables.md")


if __name__ == "__main__":
    main()
