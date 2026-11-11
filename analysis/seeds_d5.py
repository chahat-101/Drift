"""Tables for the d=5 long-training models over 3 model seeds (0, 1, 2).

    python analysis/seeds_d5.py     # writes results/exp5/report5_tables.md and report5_data.json

Reuses analysis/learning_curves.py (loading, pooling over 5 streams, McNemar per seed) and adds
mean ± std over model seeds (sample std, ddof=1) and per-seed tables. Reads only existing files.
"""

import json

import learning_curves as r4  # also puts the repo root on sys.path
import numpy as np

from report import mcnemar_p

OUT = "results/exp5"
MODELS, PHASES, REGIMES, TESTS = r4.MODELS, r4.PHASES, r4.REGIMES, r4.TESTS


def ms(vals, nd=4):
    """'mean ± std' over seeds ('mean' alone for one seed)."""
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return "–"
    if len(vals) == 1:
        return f"{vals[0]:.{nd}f}"
    return f"{np.mean(vals):.{nd}f} ± {np.std(vals, ddof=1):.{nd}f}"


def test_errs(m, s):
    return np.load(f"{r4.EXP}/full/st0/{r4.name(m, s)}_errors.npz")


def pair(ea, eb):
    """(diff = LER(a) − LER(b), exact McNemar p) on shared shots."""
    na, nb = int((ea & ~eb).sum()), int((~ea & eb).sum())
    return (na - nb) / len(ea), mcnemar_p(na, nb)


def fmt(d, p, nd=4):
    return f"{d:+.{nd}f} (p={p:.1g})" if p > 0 else f"{d:+.{nd}f} (p<1e-300)"


# ---------------------------------------------------------------- test sets


def test_section(seeds):
    tb = np.load(f"{r4.EXP}/curves/baselines.npz")
    lines = [
        "## Test sets (200000 shots each; mean ± std over model seeds)\n",
        "| decoder | seeds | " + " | ".join(TESTS) + " |",
        "|---|---|" + "---|" * 3,
        "| static MWPM | – | " + " | ".join(f"{tb[f'{t}|static'].mean():.5f}" for t in TESTS) + " |",
        "| oracle MWPM | – | " + " | ".join(f"{tb[f'{t}|oracle'].mean():.5f}" for t in TESTS) + " |",
    ]
    per = {}
    for m, ss in seeds.items():
        for s in ss:
            e = test_errs(m, s)
            per[m, s] = {t: float(e[f"test|{t}"].mean()) for t in TESTS}
        lines.append(
            f"| {MODELS[m]} | {','.join(map(str, ss))} | "
            + " | ".join(ms([per[m, s][t] for s in ss], 5) for t in TESTS)
            + " |"
        )
    lines += ["", "Per seed:\n", "| decoder | seed | " + " | ".join(TESTS) + " |", "|---|---|" + "---|" * 3]
    for (m, s), v in per.items():
        lines.append(f"| {MODELS[m]} | {s} | " + " | ".join(f"{v[t]:.5f}" for t in TESTS) + " |")
    # paired comparisons: mean diff, largest p over seeds, seeds where a is lower
    full = r4.test_final(seeds)
    lines += [
        "",
        "Paired McNemar per seed (diff = LER(a) − LER(b); p = largest over seeds; (k/n) = seeds where a "
        "is lower; ** = p < 0.05 in every seed and all seeds agree):\n",
    ] + full[full.index("") + 1 :]
    return lines, per


# ---------------------------------------------------------------- streams


def stream_section(bits, seeds):
    lines = [
        "## Streams: final checkpoints, pooled over 5 stream seeds (mean ± std over model seeds)\n",
        "Adaptive models use the live fingerprint unless marked (frozen). Report-3 rows are the 60k-step "
        "models, seeds 0–2. The best WindowDEM is picked per phase after the fact.\n",
    ]
    for r in REGIMES:
        lines += [f"### {r}\n", "| decoder | seeds | " + " | ".join(PHASES) + " |", "|---|---|" + "---|" * len(PHASES)]
        base = [("static MWPM", "static"), ("oracle MWPM", "oracle")] + [
            (f"WindowDEM-{w}", f"window_{w}") for w in r4.WINDOWS
        ]
        for lab, k in base:
            lines.append(f"| {lab} | – | " + " | ".join(r4.m4(bits[k, 0][r, ph].mean()) for ph in PHASES) + " |")
        rows = [
            (m, suf, ss, f"{MODELS[m]}{' (frozen)' if suf else ''}, 300k")
            for m, ss in seeds.items()
            for suf in ("", "|frozen")
        ]
        rows += [
            ("r3:" + m, suf, [0, 1, 2], f"{MODELS[m]}{' (frozen)' if suf else ''}, report 3 (60k)")
            for m in r4.SHORT
            for suf in ("", "|frozen")
        ]
        for key, suf, ss, lab in rows:
            if (key + suf, ss[0]) not in bits:
                continue
            lines.append(
                f"| {lab} | {','.join(map(str, ss))} | "
                + " | ".join(
                    ms([bits[key + suf, s][r, ph].mean() if len(bits[key + suf, s][r, ph]) else np.nan for s in ss])
                    for ph in PHASES
                )
                + " |"
            )
        lines += [
            "",
            f"Paired comparisons ({r}; McNemar per model seed, same format as report 4):\n",
            "| a vs b | " + " | ".join(PHASES) + " |",
            "|---|" + "---|" * len(PHASES),
        ]
        for m, ss in seeds.items():
            for lab, b in (("static MWPM", "static"), ("best WindowDEM", None), ("oracle MWPM", "oracle")):
                lines.append(
                    f"| {MODELS[m]} vs {lab} | "
                    + " | ".join(
                        r4.cmp(bits, m, b or f"window_{r4.best_w(bits, r, ph)}", (r, ph), ss)[0] for ph in PHASES
                    )
                    + " |"
                )
            if (m + "|frozen", ss[0]) in bits:
                lines.append(
                    f"| {MODELS[m]}: live vs frozen | "
                    + " | ".join(r4.cmp(bits, m, m + "|frozen", (r, ph), ss)[0] for ph in PHASES)
                    + " |"
                )
            if m in r4.SHORT:
                lines.append(
                    f"| {MODELS[m]}: 300k vs report 3 (same seed) | "
                    + " | ".join(r4.cmp(bits, m, "r3:" + m, (r, ph), ss)[0] for ph in PHASES)
                    + " |"
                )
        lines.append("")
    return lines


# ---------------------------------------------------------------- per-seed numbers behind the answers


def key_section(bits, seeds, per):
    lines = [
        "## Per-seed numbers behind the answers\n",
        "Pooled over 5 streams. Diffs are LER(model) − LER(reference) on the same shots, with the exact "
        "McNemar p of that seed.\n",
    ]

    def d(m, s, b, r, ph, bs=0):
        return pair(bits[m, s][r, ph], bits[b, bs][r, ph])

    # Q2
    lines += [
        "### Q2: bursts_big drift (live)\n",
        "| model | seed | LER | vs static MWPM | vs best WindowDEM | vs oracle MWPM |",
        "|---|---|---|---|---|---|",
    ]
    bw = f"window_{r4.best_w(bits, 'bursts_big', 'drift')}"
    for m, ss in seeds.items():
        for s in ss:
            lines.append(
                f"| {MODELS[m]} | {s} | {bits[m, s]['bursts_big', 'drift'].mean():.4f} | "
                + " | ".join(fmt(*d(m, s, b, "bursts_big", "drift")) for b in ("static", bw, "oracle"))
                + " |"
            )
    lines.append(f"\nBest WindowDEM on bursts_big drift: {bw} ({bits[bw, 0]['bursts_big', 'drift'].mean():.4f}).\n")
    # Q3
    lines += [
        "### Q3: adaptive vs oracle MWPM on later drift steps\n",
        "| model | seed | bursts later | bursts_big later | hot_qubit later | telegraph later |",
        "|---|---|---|---|---|---|",
    ]
    for m in ("adaptive_wide", "adaptive_stale", "adaptive_stale_h64l6"):
        for s in seeds.get(m, []):
            lines.append(
                f"| {MODELS[m]} | {s} | "
                + " | ".join(fmt(*d(m, s, "oracle", r, "later")) for r in REGIMES[:2] + ["hot_qubit", "telegraph"])
                + " |"
            )
    lines.append("")
    # gap to oracle on later steps, 60k vs 300k steps, per seed
    lines += [
        "Later-step gap to oracle, report 3 (60k) → 300k, same seed:\n",
        "| model | seed | bursts | bursts_big | hot_qubit |",
        "|---|---|---|---|---|",
    ]
    for m in ("adaptive_wide", "adaptive_stale"):
        for s in seeds.get(m, []):
            cells = []
            for r in ("bursts", "bursts_big", "hot_qubit"):
                o = bits["oracle", 0][r, "later"].mean()
                cells.append(
                    f"{bits['r3:' + m, s][r, 'later'].mean() - o:+.4f} → {bits[m, s][r, 'later'].mean() - o:+.4f}"
                )
            lines.append(f"| {MODELS[m]} | {s} | " + " | ".join(cells) + " |")
    for m in ("adaptive_wide", "adaptive_stale"):
        ss, cells = seeds.get(m, []), []
        for r in ("bursts", "bursts_big", "hot_qubit"):
            o = bits["oracle", 0][r, "later"].mean()
            g3 = np.mean([bits["r3:" + m, s][r, "later"].mean() - o for s in ss])
            g = np.mean([bits[m, s][r, "later"].mean() - o for s in ss])
            cells.append(f"{g3:+.4f} → {g:+.4f}" + (f" ({g3 / g:.1f}× smaller)" if g > 0 else ""))
        lines.append(f"| {MODELS[m]} | mean | " + " | ".join(cells) + " |")
    lines.append("")
    # model vs model on drift steps
    lines += [
        "### Model vs model on drift steps (McNemar per common seed, report-4 format)\n",
        "| a vs b | " + " | ".join(REGIMES) + " |",
        "|---|" + "---|" * 4,
    ]
    for a, b in (
        ("adaptive_wide", "static_wide"),
        ("adaptive_stale", "static_wide"),
        ("adaptive_stale", "adaptive_wide"),
        ("static_wide", "static_nominal"),
    ):
        ss = sorted(set(seeds.get(a, [])) & set(seeds.get(b, [])))
        lines.append(
            f"| {MODELS[a]} vs {MODELS[b]} | "
            + " | ".join(r4.cmp(bits, a, b, (r, "drift"), ss)[0] for r in REGIMES)
            + " |"
        )
    lines.append("")
    # Q4
    lines += [
        "### Q4: live − frozen on drift steps, report 3 (60k) → 300k, same seed\n",
        "| model | seed | " + " | ".join(REGIMES) + " |",
        "|---|---|" + "---|" * 4,
    ]
    gaps = {}
    for m in ("adaptive_wide", "adaptive_stale", "adaptive_stale_h64l6"):
        for s in seeds.get(m, []):
            cells = []
            for r in REGIMES:
                g = pair(bits[m, s][r, "drift"], bits[m + "|frozen", s][r, "drift"])
                gaps.setdefault((m, r), []).append(g[0])
                if ("r3:" + m, s) in bits:
                    g3 = pair(bits["r3:" + m, s][r, "drift"], bits["r3:" + m + "|frozen", s][r, "drift"])
                    cells.append(f"{g3[0]:+.4f} → {fmt(*g)}")
                else:
                    cells.append(fmt(*g))
            lines.append(f"| {MODELS[m]} | {s} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Mean ± std over seeds of live − frozen (300k):\n",
        "| model | " + " | ".join(REGIMES) + " |",
        "|---|" + "---|" * 4,
    ]
    for m in ("adaptive_wide", "adaptive_stale"):
        if m in seeds:
            lines.append(f"| {MODELS[m]} | " + " | ".join(ms(gaps[m, r]) for r in REGIMES) + " |")
    lines.append("")
    # after-burst regression
    lines += [
        "### After-burst steps (bursts_big 'after'): live vs static MWPM and vs frozen\n",
        "| model | seed | LER | vs static MWPM | vs frozen |",
        "|---|---|---|---|---|",
    ]
    for m in ("adaptive_wide", "adaptive_stale", "adaptive_stale_h64l6"):
        for s in seeds.get(m, []):
            lines.append(
                f"| {MODELS[m]} | {s} | {bits[m, s]['bursts_big', 'after'].mean():.4f} | "
                f"{fmt(*d(m, s, 'static', 'bursts_big', 'after'))} | "
                f"{fmt(*pair(bits[m, s]['bursts_big', 'after'], bits[m + '|frozen', s]['bursts_big', 'after']))} |"
            )
    lines.append("")
    # Q5
    if "adaptive_stale_h64l6" in seeds:
        h = "adaptive_stale_h64l6"
        lines += [
            "### Q5: h64/l6 (seed 0 only) vs each h48/l4 adaptive-stale seed\n",
            "Paired on the same shots, but against different h48/l4 seeds, so seed-to-seed variation is "
            "mixed into the diff.\n",
            "| h48/l4 seed | bursts_big drift | bursts drift | test scale x2 | test nominal |",
            "|---|---|---|---|---|",
        ]
        eh = test_errs(h, 0)
        for s in seeds["adaptive_stale"]:
            ea = test_errs("adaptive_stale", s)
            lines.append(
                f"| {s} | {fmt(*pair(bits[h, 0]['bursts_big', 'drift'], bits['adaptive_stale', s]['bursts_big', 'drift']))} | "
                f"{fmt(*pair(bits[h, 0]['bursts', 'drift'], bits['adaptive_stale', s]['bursts', 'drift']))} | "
                f"{fmt(*pair(eh['test|scale x2'], ea['test|scale x2']), 5)} | "
                f"{fmt(*pair(eh['test|nominal'], ea['test|nominal']), 5)} |"
            )
        hv = [
            bits[h, 0]["bursts_big", "drift"].mean(),
            bits[h, 0]["bursts", "drift"].mean(),
            per[h, 0]["scale x2"],
            per[h, 0]["nominal"],
        ]
        sv = [
            [bits["adaptive_stale", s]["bursts_big", "drift"].mean() for s in seeds["adaptive_stale"]],
            [bits["adaptive_stale", s]["bursts", "drift"].mean() for s in seeds["adaptive_stale"]],
            [per["adaptive_stale", s]["scale x2"] for s in seeds["adaptive_stale"]],
            [per["adaptive_stale", s]["nominal"] for s in seeds["adaptive_stale"]],
        ]
        lines += [
            "",
            "| | bursts_big drift | bursts drift | test scale x2 | test nominal |",
            "|---|---|---|---|---|",
            "| h64/l6 seed 0 | " + " | ".join(f"{v:.5f}" for v in hv) + " |",
            "| h48/l4 seeds 0–2 (min … max) | " + " | ".join(f"{min(v):.5f} … {max(v):.5f}" for v in sv) + " |",
            "| h48/l4 mean ± std | " + " | ".join(ms(v, 5) for v in sv) + " |",
            "",
        ]
    return lines


def main():
    bits, seeds = r4.final_bits()
    out = [
        "# d=5 long-training models over 3 seeds (generated by analysis/seeds_d5.py)\n",
        "Model seeds found (all 5 stream seeds present): "
        + ", ".join(f"{MODELS[m]} {ss}" for m, ss in seeds.items())
        + "\n",
    ]
    tl, per = test_section(seeds)
    out += tl + [""]
    out += stream_section(bits, seeds)
    out += key_section(bits, seeds, per)
    out += (
        ["## Live − frozen on drift: report 3 vs 300k (report-4 format, 3 seeds)\n"] + r4.gap_table(bits, seeds) + [""]
    )
    cv = r4.curves()
    out += (
        ["## Learning curves: last 60k steps (seed 0 only; seeds 1/2 have no curve evals)\n"] + r4.last_60k(cv) + [""]
    )
    open(f"{OUT}/report5_tables.md", "w").write("\n".join(out) + "\n")
    data = {
        "seeds": seeds,
        "test": {f"{m}|s{s}": v for (m, s), v in per.items()},
        "streams": {
            f"{k[0]}|s{k[1]}": {f"{r}|{ph}": float(b.mean()) if len(b) else None for (r, ph), b in v.items()}
            for k, v in bits.items()
        },
    }
    json.dump(data, open(f"{OUT}/report5_data.json", "w"), indent=1)
    print(f"wrote {OUT}/report5_tables.md")


if __name__ == "__main__":
    main()
