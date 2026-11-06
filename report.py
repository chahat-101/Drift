"""Collect the stale-fingerprint experiment into tables and plots.

    python report.py --data data/d3_wide          # experiment 1 -> results/report_tables.md
    python report.py --data data/d3_wide --exp2   # experiment 2 -> results/exp2/report2_tables.md

Experiment 2 (frozen fingerprints, WindowDEM window sweep, 5 streams per regime)
pools every phase over streams and model seeds and compares decoders with an exact
paired McNemar test on the per-shot error bits in results/exp2/*_errors.npz. Each
model seed is tested separately; a difference counts as significant when every
seed has p < 0.05 in the same direction. The 2-SE rule below is kept as a
secondary column.

Expects these files from train.py / eval.py (see results/report.md):
  decoder_{static,adaptive}_d3_wide_s{0,1,2}.pt, decoder_adaptive_d3_stale_s{0,1,2}.pt
  results/<regime>_d3_r3_u<update>_s<seed>.json and .csv

Scores every checkpoint on the saved test sets (cached in results/test_scores.json),
averages the eval summaries over seeds and writes results/report_tables.md,
results/report_data.json and the PNG plots. The prose in report.md is written by hand.

Ties: two decoders tie on a metric when their difference is within 2 standard
errors, SE = sqrt(SE_a^2 + SE_b^2) with binomial SE_x = sqrt(p_x (1 - p_x) / shots).
"""
import argparse
import json
import os

import numpy as np

RESULTS = "results"
SEEDS = [0, 1, 2]
REGIMES = ["hot_qubit", "bursts", "bursts_big", "telegraph", "wander"]
UPDATES = [2000, 200]
PHASES = ["all", "drift", "onset", "later", "after"]
PHASE_TITLES = {"all": "LER", "drift": "LER (drift)", "onset": "onset", "later": "later", "after": "after"}
# display name -> decoder name in eval output ({s} = seed)
DECODERS = {
    "static MWPM": "static",
    "WindowDEM": "window",
    "oracle MWPM": "oracle",
    "static-wide": "decoder_static_d3_wide_s{s}",
    "adaptive-wide": "decoder_adaptive_d3_wide_s{s}",
    "adaptive-stale": "decoder_adaptive_d3_stale_s{s}",
}
NEURAL = ["static-wide", "adaptive-wide", "adaptive-stale"]
# fixed categorical order (validated palette, light mode)
COLORS = dict(zip(DECODERS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
PHASE_COLORS = {"onset": "#2a78d6", "later": "#eb6834", "after": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def fmt(mean, std=None):
    if mean is None or np.isnan(mean):
        return "–"
    return f"{mean:.5f}" if std is None else f"{mean:.5f} ± {std:.5f}"


def se(p, n):
    return np.sqrt(p * (1 - p) / n)


def verdict(pa, pb, n):
    """'tie' if |pa - pb| < 2 SE, else which side is lower."""
    if n == 0 or pa is None or pb is None:
        return None
    if abs(pa - pb) < 2 * np.sqrt(se(pa, n) ** 2 + se(pb, n) ** 2):
        return "tie"
    return "lower" if pa < pb else "higher"


# ---------------------------------------------------------------- test sets

def test_scores(data):
    """{display name: {state: [LER per seed]}} plus MWPM baselines, cached on disk."""
    path = os.path.join(RESULTS, "test_scores.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    import torch
    from dataset import Dataset
    from decoder import DetectorGrid, load_checkpoint
    from drift import DriftStream
    from train import evaluate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = Dataset(data)
    a = ds.args
    stream = DriftStream(a["distance"], a["rounds"], a["p"])
    grid, tests = DetectorGrid(stream.nominal()), ds.test_sets()
    out = {name: {} for name in NEURAL}
    for name in NEURAL:
        for s in SEEDS:
            ck = DECODERS[name].format(s=s) + ".pt"
            model, cfg = load_checkpoint(ck, device)
            ds.check_model(cfg, ck)
            print(f"== {ck}")
            res = evaluate(model.to(device), grid, stream, tests, cfg["adaptive"], device)
            for state, r in res.items():
                out[name].setdefault(state, []).append(r["neural"])
                out.setdefault("static MWPM", {})[state] = [r["static"]]
                out.setdefault("oracle MWPM", {})[state] = [r["oracle"]]
    out["_shots"] = len(tests[0][2])
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def table1(scores):
    states = [k for k in scores["static MWPM"]]
    lines = ["| decoder | " + " | ".join(states) + " |", "|---" * (len(states) + 1) + "|"]
    for name in ["static MWPM", "oracle MWPM"] + NEURAL:
        cells = [fmt(np.mean(v), np.std(v, ddof=1) if len(v) > 1 else None)
                 for v in (scores[name][st] for st in states)]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table1_comparisons(scores, pairs):
    """Difference (a - b) of seed means on each test state, with the 2-SE tie rule."""
    states, n = list(scores["static MWPM"]), scores["_shots"]
    lines = ["| a vs b | " + " | ".join(states) + " |", "|---" * (len(states) + 1) + "|"]
    for a, b in pairs:
        cells = []
        for st in states:
            pa, pb = np.mean(scores[a][st]), np.mean(scores[b][st])
            v = verdict(pa, pb, n)
            cells.append(f"{pa - pb:+.4f} {'tie' if v == 'tie' else ('a better' if v == 'lower' else 'b better')}")
        lines.append(f"| {a} vs {b} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- streams

def load_runs():
    """{(regime, update): [summary per seed]}"""
    runs = {}
    for regime in REGIMES:
        for u in UPDATES:
            runs[regime, u] = []
            for s in SEEDS:
                path = os.path.join(RESULTS, f"{regime}_d3_r3_u{u}_s{s}.json")
                with open(path) as f:
                    runs[regime, u].append(json.load(f))
    return runs


def phase_stats(summaries):
    """{display name: {phase: (mean, std or None)}} over seeds."""
    out = {}
    for name, key in DECODERS.items():
        out[name] = {}
        for ph in PHASES:
            vals = [s["decoders"][key.format(s=i)]["ler"][ph] for i, s in enumerate(summaries)]
            if vals[0] is None:
                out[name][ph] = (None, None)
            else:
                out[name][ph] = (float(np.mean(vals)), float(np.std(vals, ddof=1)) if name in NEURAL else None)
    return out


def table2(regime, runs, stats):
    s0 = runs[regime, UPDATES[0]][0]
    counts = ", ".join(f"{ph} {s0['phase_steps'][ph]} steps" for ph in ["drift", "onset", "later", "after"])
    lines = [f"Shots per step {s0['shots_per_step']}; {counts}.", "",
             "| decoder | update | " + " | ".join(PHASE_TITLES[p] for p in PHASES) + " |",
             "|---" * (len(PHASES) + 2) + "|"]
    for name in DECODERS:
        for u in UPDATES:
            cells = [fmt(*stats[regime, u][name][ph]) for ph in PHASES]
            lines.append(f"| {name} | {u} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def comparisons(regime, runs, stats, pairs):
    """Verdict table: difference (a - b) and win/tie per phase for each pair and update."""
    s0 = runs[regime, UPDATES[0]][0]
    phases = ["drift", "onset", "later", "after"]
    lines = ["| a vs b | update | " + " | ".join(phases) + " |", "|---" * (len(phases) + 2) + "|"]
    for a, b in pairs:
        for u in UPDATES:
            cells = []
            for ph in phases:
                pa, pb, n = stats[regime, u][a][ph][0], stats[regime, u][b][ph][0], s0["phase_shots"][ph]
                v = verdict(pa, pb, n)
                cells.append("–" if v is None else f"{pa - pb:+.4f} {'tie' if v == 'tie' else ('a better' if v == 'lower' else 'b better')}")
            lines.append(f"| {a} vs {b} | {u} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def recoverable(regime, pairs, cutoff=0.4):
    """Phase table and comparisons without bursts that no decoder survives (oracle LER > cutoff).

    A dropped burst also drops its after step. Reads the per-step CSVs.
    """
    import csv
    stats, shots, dropped = {}, {}, None
    for u in UPDATES:
        per_seed = []
        for s in SEEDS:
            rows = list(csv.DictReader(open(os.path.join(RESULTS, f"{regime}_d3_r3_u{u}_s{s}.csv"))))
            drift = [float(r["scale"]) != 1.0 or int(r["n_hot"]) > 0 for r in rows]
            keep, t = {"onset": [], "later": [], "after": []}, 1
            dropped = []
            while t < len(rows):
                if drift[t] and not drift[t - 1]:
                    end = t
                    while end < len(rows) and drift[end]:
                        end += 1
                    if np.mean([float(rows[k]["oracle"]) for k in range(t, end)]) > cutoff:
                        dropped.append(t)
                    else:
                        keep["onset"].append(t)
                        keep["later"] += list(range(t + 1, end))
                        keep["after"] += [end] if end < len(rows) else []
                    t = end
                t += 1
            per_seed.append({name: {ph: np.mean([float(rows[k][key.format(s=s)]) for k in idx])
                                    for ph, idx in keep.items()} for name, key in DECODERS.items()})
            shots = {ph: len(idx) * 2000 for ph, idx in keep.items()}
        stats[u] = {name: {ph: (float(np.mean([p[name][ph] for p in per_seed])),
                                float(np.std([p[name][ph] for p in per_seed], ddof=1)) if name in NEURAL else None)
                           for ph in keep} for name in DECODERS}
    phases = ["onset", "later", "after"]
    lines = [f"Dropped bursts starting at steps {dropped} (oracle LER > {cutoff}); kept "
             + ", ".join(f"{ph} {shots[ph] // 2000} steps" for ph in phases) + ".", "",
             "| decoder | update | " + " | ".join(phases) + " |", "|---" * (len(phases) + 2) + "|"]
    for name in DECODERS:
        for u in UPDATES:
            lines.append(f"| {name} | {u} | " + " | ".join(fmt(*stats[u][name][ph]) for ph in phases) + " |")
    lines += ["", "| a vs b | update | " + " | ".join(phases) + " |", "|---" * (len(phases) + 2) + "|"]
    for a, b in pairs:
        for u in UPDATES:
            cells = []
            for ph in phases:
                pa, pb = stats[u][a][ph][0], stats[u][b][ph][0]
                v = verdict(pa, pb, shots[ph])
                cells.append(f"{pa - pb:+.4f} {'tie' if v == 'tie' else ('a better' if v == 'lower' else 'b better')}")
            lines.append(f"| {a} vs {b} | {u} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- plots

def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_time(regime, u=200, seed=0):
    """Per-step LER of every decoder on one stream, drifted steps shaded."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(os.path.join(RESULTS, f"{regime}_d3_r3_u{u}_s{seed}.csv"))))
    t = np.array([int(r["step"]) for r in rows])
    drift = np.array([float(r["scale"]) != 1.0 or int(r["n_hot"]) > 0 for r in rows])
    shots = 2000
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i in np.flatnonzero(drift):
        ax.axvspan(i - 0.5, i + 0.5, color=GRID, linewidth=0)
    for name, key in DECODERS.items():
        y = np.array([float(r[key.format(s=seed)]) for r in rows])
        ax.plot(t, np.maximum(y, 0.5 / shots), color=COLORS[name], linewidth=2 if name in NEURAL else 1.2,
                label=name, linestyle="-" if name in NEURAL else "--")
    ax.set_yscale("log")
    ax.set_xlim(t[0] - 0.5, t[-1] + 0.5)
    ax.set_xlabel("step (2000 shots each; shaded = burst)", color=MUTED)
    ax.set_ylabel("logical error rate per step", color=MUTED)
    ax.set_title(f"{regime}: LER per step (seed {seed}, update every {u} shots)", color=INK, loc="left")
    style(ax)
    ax.legend(frameon=False, fontsize=9, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    path = os.path.join(RESULTS, f"report_{regime}_time.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_phases(regime, stats, u=200):
    """Grouped bars: onset / later / after LER per decoder, error bars = std over seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, phases, w = list(DECODERS), ["onset", "later", "after"], 0.26
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for j, ph in enumerate(phases):
        means = [stats[regime, u][n][ph][0] for n in names]
        stds = [stats[regime, u][n][ph][1] if n in NEURAL else np.nan for n in names]  # MWPM: deterministic
        ax.bar(x + (j - 1) * w, means, w - 0.02, yerr=stds, color=PHASE_COLORS[ph], label=ph,
               error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 2})
    ax.set_xticks(x, names)
    ax.set_ylabel("logical error rate", color=MUTED)
    ax.set_title(f"{regime}: LER by phase (mean ± std over 3 seeds, update every {u} shots)",
                 color=INK, loc="left")
    style(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = os.path.join(RESULTS, f"report_{regime}_phases.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ================================================================ experiment 2
# Adaptation vs better decoder (frozen fingerprints), fair WindowDEM baselines, and
# 5 independent streams per regime. Pools phases over streams and uses paired
# McNemar tests on per-shot error bits (eval.py writes them to *_errors.npz).

EXP2 = os.path.join(RESULTS, "exp2")
REGIMES2 = ["bursts", "bursts_big", "hot_qubit", "telegraph"]
STREAMS = [0, 1, 2, 3, 4]
WINDOWS = [500, 1000, 2000, 5000, 10000]
PHASES2 = ["all", "drift", "onset", "later", "after", "nominal"]
DECODERS2 = {
    "static MWPM": "static",
    **{f"WindowDEM-{w}": f"window_{w}" for w in WINDOWS},
    "oracle MWPM": "oracle",
    "static-nominal": "decoder_static_d3_nominal_s{s}",
    "static-wide": "decoder_static_d3_wide_s{s}",
    "adaptive-wide": "decoder_adaptive_d3_wide_s{s}",
    "adaptive-wide frozen": "decoder_adaptive_d3_wide_s{s}_frozen",
    "adaptive-stale": "decoder_adaptive_d3_stale_s{s}",
    "adaptive-stale frozen": "decoder_adaptive_d3_stale_s{s}_frozen",
}
NEURAL2 = [n for n, k in DECODERS2.items() if "{s}" in k]
TEST_MODELS = ["static-nominal", "static-wide", "adaptive-wide", "adaptive-stale"]


def mcnemar_p(b, c):
    """Exact two-sided McNemar p-value from the two discordant counts b and c."""
    from math import exp, lgamma, log
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    logs = [lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) - n * log(2) for i in range(k + 1)]
    m = max(logs)
    return min(1.0, 2 * exp(m) * sum(exp(x - m) for x in logs))


def paired(bits, a, b, ph, neural=None):
    """Compare decoders a and b on one phase, per model seed (same shots, so paired).

    neural: names that differ per model seed (default NEURAL2); MWPM decoders are the
    same for every seed, so a pair of them is tested once. Returns dict: diff (LER a -
    LER b, mean over seeds), p_max (largest McNemar p over seeds), agree (seeds where
    a is lower, out of how many), tie (2-SE rule), n (shots).
    """
    neural = NEURAL2 if neural is None else neural
    seeds = SEEDS if (a in neural or b in neural) else SEEDS[:1]
    diffs, ps, lower = [], [], 0
    for s in seeds:
        ea, eb = bits[a, s][ph], bits[b, s][ph]
        if len(ea) == 0:
            return None
        n_a, n_b = int((ea & ~eb).sum()), int((~ea & eb).sum())
        diffs.append((n_a - n_b) / len(ea))
        ps.append(mcnemar_p(n_a, n_b))
        lower += n_a < n_b
    pa = np.mean([bits[a, s][ph].mean() for s in seeds])
    pb = np.mean([bits[b, s][ph].mean() for s in seeds])
    return {"diff": float(np.mean(diffs)), "p_max": max(ps), "agree": f"{lower}/{len(seeds)}",
            "tie": verdict(pa, pb, len(bits[a, seeds[0]][ph])) == "tie", "n": len(bits[a, seeds[0]][ph])}


def fmt_pair(r):
    if r is None:
        return "–"
    sig = "**" if r["p_max"] < 0.05 and r["agree"] in ("0/3", "3/3", "0/1", "1/1") else ""
    p = f"p={r['p_max']:.1g}" if r["p_max"] > 0 else "p<1e-300"
    return f"{sig}{r['diff']:+.4f}{sig} {p} ({r['agree']}){' tie' if r['tie'] else ''}"


def load_bits(regime):
    """{(display name, model seed): {phase: per-shot error bits pooled over the 5 streams}}"""
    out = {}
    for s in SEEDS:
        files = [np.load(os.path.join(EXP2, f"{regime}_d3_r3_u200_st{st}_s{s}_errors.npz")) for st in STREAMS]
        for name, key in DECODERS2.items():
            k = key.format(s=s)
            errs = [(np.unpackbits(f[k], axis=-1).astype(bool), f) for f in files]
            out[name, s] = {ph: np.concatenate([e[f[f"phase_{ph}"]].ravel() for e, f in errs]) for ph in PHASES2}
    return out


def pooled(bits):
    """{display name: {phase: (LER pooled over streams and seeds, std over seeds or None)}}"""
    out = {}
    for name in DECODERS2:
        out[name] = {}
        for ph in PHASES2:
            vals = [bits[name, s][ph].mean() if len(bits[name, s][ph]) else np.nan for s in SEEDS]
            out[name][ph] = (float(np.mean(vals)), float(np.std(vals, ddof=1)) if name in NEURAL2 else None)
    return out


def best_window(stats, ph):
    """The WindowDEM history with the lowest pooled LER on phase ph."""
    cands = [(stats[f"WindowDEM-{w}"][ph][0], w) for w in WINDOWS if not np.isnan(stats[f"WindowDEM-{w}"][ph][0])]
    return min(cands)[1] if cands else None


def test_errors(data):
    """Per-shot test-set error bits {(name, seed, state): bool array}, cached in results/exp2."""
    path = os.path.join(EXP2, "test_errors.npz")
    if not os.path.exists(path):
        import torch
        from baselines import Oracle, Static
        from dataset import Dataset
        from decoder import DetectorGrid, build_input, load_checkpoint
        from drift import DriftStream

        device = "cuda" if torch.cuda.is_available() else "cpu"
        ds = Dataset(data)
        stream = DriftStream(ds.args["distance"], ds.args["rounds"], ds.args["p"])
        grid, tests = DetectorGrid(stream.nominal()), ds.test_sets()
        static, oracle, out = Static(stream.nominal()), Oracle(stream), {}
        for state, st, det, obs, wdet in tests:
            out[f"static MWPM|0|{state}"] = static.decode(det)[:, 0] != obs[:, 0]
            out[f"oracle MWPM|0|{state}"] = oracle.decode(det, st)[:, 0] != obs[:, 0]
        for name in TEST_MODELS:
            for s in SEEDS:
                ck = DECODERS2[name].format(s=s) + ".pt"
                model, cfg = load_checkpoint(ck, device)
                model.to(device)
                ds.check_model(cfg, ck)
                for state, st, det, obs, wdet in tests:
                    fp = wdet.mean(axis=0).astype(np.float32) if cfg["adaptive"] else None
                    with torch.no_grad():
                        pred = np.concatenate([(model(build_input(grid, det[j:j + 4096], fp).to(device)) > 0).cpu().numpy()
                                               for j in range(0, len(det), 4096)])
                    out[f"{name}|{s}|{state}"] = pred != obs[:, 0]
        np.savez_compressed(path, **{k: np.packbits(v) for k, v in out.items()}, _shots=len(tests[0][2]))
    z = np.load(path)
    n = int(z["_shots"])
    return {tuple(k.split("|")): np.unpackbits(z[k], count=n).astype(bool) for k in z.files if k != "_shots"}


def exp2_table1(terr):
    """Test-set LER table plus paired comparisons on each test state."""
    states = list(dict.fromkeys(k[2] for k in terr))
    get = lambda name, s, st: terr[name, str(s), st]
    lines = ["| decoder | " + " | ".join(states) + " |", "|---" * (len(states) + 1) + "|"]
    for name in ["static MWPM", "oracle MWPM"] + TEST_MODELS:
        if name in TEST_MODELS:
            cells = [fmt(np.mean([get(name, s, st).mean() for s in SEEDS]),
                         np.std([get(name, s, st).mean() for s in SEEDS], ddof=1)) for st in states]
        else:
            cells = [fmt(get(name, 0, st).mean()) for st in states]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    # reuse paired() by shaping the test sets like stream phases
    bits = {(name, s): {st: get(name, s if name in TEST_MODELS else 0, st) for st in states}
            for name in ["static MWPM", "oracle MWPM"] + TEST_MODELS for s in SEEDS}
    pairs = [("adaptive-wide", "static-nominal"), ("adaptive-stale", "static-nominal"),
             ("static-wide", "static-nominal"), ("static-nominal", "static MWPM"), ("adaptive-wide", "static-wide")]
    lines += ["", "| a vs b | " + " | ".join(states) + " |", "|---" * (len(states) + 1) + "|"]
    for a, b in pairs:
        lines.append(f"| {a} vs {b} | " + " | ".join(fmt_pair(paired(bits, a, b, st, TEST_MODELS))
                                                     for st in states) + " |")
    return "\n".join(lines)


def exp2_regime_table(stats):
    lines = ["| decoder | " + " | ".join(PHASE_TITLES.get(p, p) for p in PHASES2) + " |",
             "|---" * (len(PHASES2) + 1) + "|"]
    for name in DECODERS2:
        lines.append(f"| {name} | " + " | ".join(fmt(*stats[name][ph]) for ph in PHASES2) + " |")
    return "\n".join(lines)


def exp2_pairs_table(bits, stats, pairs, phases):
    lines = ["| a vs b | " + " | ".join(phases) + " |", "|---" * (len(phases) + 1) + "|"]
    for a, b in pairs:
        cells = []
        for ph in phases:
            best = f"WindowDEM-{best_window(stats, ph)}"
            aa, bb = (best if x == "best WindowDEM" else x for x in (a, b))
            cells.append(fmt_pair(paired(bits, aa, bb, ph)))
        lines.append(f"| {a} vs {b} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def exp2_adaptation(bits, stats, phases):
    """Split each adaptive model's gain over static-wide into a decoder part and an adaptation part.

    gain = static-wide - live = (static-wide - frozen) + (frozen - live)
         = decoder part (same model, nominal fingerprint) + adaptation part (live fingerprint).
    """
    lines = ["| model | phase | static-wide | frozen | live | gain over static-wide | decoder part | "
             "adaptation part | live vs frozen (McNemar) |", "|---" * 9 + "|"]
    for model in ["adaptive-wide", "adaptive-stale"]:
        for ph in phases:
            sw, fr, lv = stats["static-wide"][ph][0], stats[f"{model} frozen"][ph][0], stats[model][ph][0]
            if np.isnan(lv):
                continue
            lines.append(f"| {model} | {ph} | {sw:.5f} | {fr:.5f} | {lv:.5f} | {sw - lv:+.5f} | {sw - fr:+.5f} | "
                         f"{fr - lv:+.5f} | {fmt_pair(paired(bits, model, f'{model} frozen', ph))} |")
    return "\n".join(lines)


def exp2_windows(stats):
    """Best WindowDEM history per phase, against the 10000-shot default."""
    phases = ["drift", "onset", "later", "after", "nominal"]
    lines = ["| phase | " + " | ".join(f"w={w}" for w in WINDOWS) + " | best |", "|---" * (len(WINDOWS) + 2) + "|"]
    for ph in phases:
        cells = [fmt(stats[f"WindowDEM-{w}"][ph][0]) for w in WINDOWS]
        lines.append(f"| {ph} | " + " | ".join(cells) + f" | {best_window(stats, ph) or '–'} |")
    return "\n".join(lines)


def plot_phases2(regime, stats):
    """Grouped bars of onset / later / after per decoder; WindowDEM shown at its best history per phase."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["static MWPM", "best WindowDEM", "oracle MWPM", "static-nominal", "static-wide",
             "adaptive-wide", "adaptive-wide frozen", "adaptive-stale", "adaptive-stale frozen"]
    phases, w = ["onset", "later", "after"], 0.27
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for j, ph in enumerate(phases):
        bw = best_window(stats, ph)
        rows = [stats[f"WindowDEM-{bw}"][ph] if n == "best WindowDEM" and bw else
                stats[n][ph] if n != "best WindowDEM" else (np.nan, None) for n in names]
        means = [r[0] for r in rows]
        stds = [r[1] if r[1] is not None else np.nan for r in rows]
        ax.bar(x + (j - 1) * w, means, w - 0.02, yerr=stds, color=PHASE_COLORS[ph], label=ph,
               error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 2})
        if bw and not np.isnan(means[1]):
            ax.text(x[1] + (j - 1) * w, means[1], f"{bw}", ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.set_xticks(x, [n.replace(" frozen", "\nfrozen").replace("best ", "best\n") for n in names], fontsize=9)
    ax.set_ylabel("logical error rate", color=MUTED)
    ax.set_title(f"{regime}: LER by phase, pooled over 5 streams x 3 model seeds (update every 200 shots; "
                 f"error bars = std over model seeds; number = best WindowDEM history)", color=INK, loc="left", fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = os.path.join(EXP2, f"report2_{regime}_phases.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def exp2_main(data):
    terr = test_errors(data)
    parts = ["## Table 1: test sets", "", exp2_table1(terr), ""]
    data_out = {}
    pairs = [("adaptive-wide", "adaptive-wide frozen"), ("adaptive-stale", "adaptive-stale frozen"),
             ("adaptive-wide", "static-wide"), ("adaptive-stale", "static-wide"),
             ("adaptive-wide frozen", "static-wide"), ("adaptive-stale frozen", "static-wide"),
             ("adaptive-stale", "adaptive-wide"), ("adaptive-wide", "static-nominal"),
             ("adaptive-stale", "static-nominal"), ("static-wide", "static-nominal"),
             ("adaptive-wide", "best WindowDEM"), ("adaptive-stale", "best WindowDEM"),
             ("adaptive-wide", "oracle MWPM"), ("adaptive-stale", "oracle MWPM"),
             ("best WindowDEM", "static MWPM")]
    for regime in REGIMES2:
        bits = load_bits(regime)
        stats = pooled(bits)
        data_out[regime] = stats
        n = {ph: len(bits["static MWPM", 0][ph]) // 2000 for ph in PHASES2}
        phases = [ph for ph in ["drift", "onset", "later", "after", "nominal"] if n[ph]]
        parts += [f"## {regime}", "",
                  f"Pooled over 5 streams: " + ", ".join(f"{ph} {n[ph]} steps" for ph in PHASES2[1:]) + ".", "",
                  exp2_regime_table(stats), "", f"### Paired comparisons: {regime}", "",
                  exp2_pairs_table(bits, stats, pairs, phases), "", f"### Decoder vs adaptation: {regime}", "",
                  exp2_adaptation(bits, stats, phases), "", f"### WindowDEM history: {regime}", "",
                  exp2_windows(stats), ""]
        print("wrote", plot_phases2(regime, stats), flush=True)
    with open(os.path.join(EXP2, "report2_tables.md"), "w") as f:
        f.write("\n".join(parts))
    with open(os.path.join(EXP2, "report2_data.json"), "w") as f:
        json.dump(data_out, f, indent=2)
    print(f"wrote {EXP2}/report2_tables.md and report2_data.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/d3_wide", help="dataset whose test sets score Table 1")
    ap.add_argument("--exp2", action="store_true", help="experiment 2 (frozen fingerprints, window sweep, 5 streams)")
    args = ap.parse_args()
    if args.exp2:
        return exp2_main(args.data)

    scores = test_scores(args.data)
    runs = load_runs()
    stats = {k: phase_stats(v) for k, v in runs.items()}
    pairs = [("adaptive-stale", "static-wide"), ("adaptive-stale", "adaptive-wide"),
             ("adaptive-wide", "static-wide"), ("adaptive-stale", "oracle MWPM"),
             ("adaptive-stale", "static MWPM"), ("adaptive-stale", "WindowDEM")]
    parts = ["## Table 1: test sets", "", table1(scores), "",
             table1_comparisons(scores, [("adaptive-stale", "adaptive-wide"), ("adaptive-stale", "static-wide"),
                                         ("adaptive-stale", "oracle MWPM"), ("adaptive-wide", "oracle MWPM")]), ""]
    for regime in REGIMES:
        parts += [f"## Table 2: {regime}", "", table2(regime, runs, stats), "",
                  f"### Comparisons: {regime}", "", comparisons(regime, runs, stats, pairs), ""]
    parts += ["## Supplementary: bursts_big without saturated bursts", "",
              recoverable("bursts_big", pairs[:4]), ""]
    with open(os.path.join(RESULTS, "report_tables.md"), "w") as f:
        f.write("\n".join(parts))
    with open(os.path.join(RESULTS, "report_data.json"), "w") as f:
        json.dump({f"{r}_u{u}": v for (r, u), v in stats.items()}, f, indent=2)
    for regime in ["bursts", "bursts_big"]:
        print("wrote", plot_time(regime), plot_phases(regime, stats))
    print(f"wrote {RESULTS}/report_tables.md and report_data.json")


if __name__ == "__main__":
    main()
