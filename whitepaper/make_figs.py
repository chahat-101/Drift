"""Figures for the Drift whitepaper.

The result figures are drawn from result files that already exist in the repository; nothing is
trained or evaluated here. The three illustration figures (drift regimes, example shots, fingerprint
maps) are drawn from the simulator itself (drift.py / circuits.py, Stim 1.16.0, PyMatching 2.4.0),
with the same seeds and qubit choices as tech_doc_src/figs.py, which made the earlier versions.
Run from anywhere:

    python whitepaper/make_figs.py

Sources
-------
- results/exp3/d5/*_u200_st{0..4}_s0.csv   per-step LERs of static MWPM, WindowDEM-w and oracle
                                            MWPM at d=5 (deterministic, identical for every seed file)
- results/exp6/report6_data.json            d=3 and d=5 three-seed ensembles and their members
- results/exp4/curves/*.json                d=5 seed-0 learning-curve evals (30k ... 300k steps)
- results/exp9/localisation_baseline.json   NF noise-map head vs the training-free fingerprint score

Numbers recomputed here are checked (assert) against the values printed in the notes / tables,
so a figure cannot silently disagree with the text.
"""
from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "whitepaper", "fig")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- style
# Okabe-Ito palette (colour-blind safe); one fixed colour per decoder in every figure.
COL = {
    "static": "#7f7f7f",  # static MWPM (grey)
    "window": "#E69F00",  # best WindowDEM (orange)
    "oracle": "#000000",  # oracle MWPM (black)
    "static_wide": "#56B4E9",  # static-wide ensemble (sky blue)
    "adaptive_wide": "#009E73",  # adaptive-wide ensemble (green)
    "adaptive_stale": "#0072B2",  # adaptive-stale ensemble (blue)
    "static_nominal": "#CC79A7",  # static-nominal (pink)
    "h64": "#D55E00",  # adaptive-stale h64/l6 (vermillion)
    "nf": "#0072B2",
    "fp_baseline": "#E69F00",
}
LABEL = {
    "static": "static MWPM",
    "window": "best WindowDEM",
    "oracle": "oracle MWPM",
    "static_wide": "static-wide",
    "adaptive_wide": "adaptive-wide",
    "adaptive_stale": "adaptive-stale",
    "static_nominal": "static-nominal",
    "h64": "adaptive-stale h64/l6",
}
HATCH = {"oracle": "////"}
plt.rcParams.update({
    "font.size": 8.5,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
REGIMES = ["bursts", "bursts_big", "hot_qubit", "telegraph"]
RLAB = {"bursts": "bursts", "bursts_big": "bursts_big (out of distr.)", "hot_qubit": "hot_qubit",
        "telegraph": "telegraph"}
WINDOWS = [500, 1000, 2000, 5000, 10000]
TEXTW = 6.4  # inches, ~ text width of a 21 cm page with 2.3 cm margins


def close(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


# ---------------------------------------------------------------- data: classical baselines at d=5
def phase_masks(rows):
    """Same definition as eval.phase_masks."""
    drift = np.array([(float(r["scale"]) != 1.0) or int(r["n_hot"]) > 0 for r in rows])
    before = np.concatenate([[False], drift[:-1]])
    has_prev = np.arange(len(rows)) > 0
    onset = drift & ~before & has_prev
    return {"drift": drift, "onset": onset, "later": drift & ~onset, "after": ~drift & before,
            "nominal": ~drift}


def mwpm_d5():
    """Pooled (5 streams) LER per regime/phase for static, oracle and WindowDEM-w at d=5."""
    out = {}
    for r in REGIMES:
        err = {}
        n = {}
        for st in range(5):
            path = os.path.join(RES, "exp3", "d5", f"{r}_d5_r5_u200_st{st}_s0.csv")
            rows = list(csv.DictReader(open(path)))
            m = phase_masks(rows)
            for col in ["static", "oracle"] + [f"window_{w}" for w in WINDOWS]:
                e = np.array([round(float(x[col]) * 2000) for x in rows])
                for ph, mk in m.items():
                    err[(col, ph)] = err.get((col, ph), 0) + e[mk].sum()
                    n[(col, ph)] = n.get((col, ph), 0) + 2000 * mk.sum()
        ler = {k: err[k] / n[k] if n[k] else float("nan") for k in err}
        for ph in ["drift", "onset", "later", "after", "nominal"]:
            ws = [ler[(f"window_{w}", ph)] for w in WINDOWS]
            ler[("window", ph)] = min(ws) if not np.isnan(ws[0]) else float("nan")
        out[r] = ler
    # checks against notes/07_ensembles.md (reference LERs on drift) and notes/06 tables
    ref = {"bursts": (0.0439, 0.0356, 0.0197), "bursts_big": (0.0980, 0.0896, 0.0426),
           "hot_qubit": (0.0181, 0.0108, 0.0103), "telegraph": (0.0228, 0.0229, 0.0231)}
    for r, (s, w, o) in ref.items():
        close(out[r][("static", "drift")], s, 5e-5)
        close(out[r][("window", "drift")], w, 5e-5)
        close(out[r][("oracle", "drift")], o, 5e-5)
    close(out["bursts"][("window", "onset")], 0.0455, 5e-5)
    close(out["bursts_big"][("window", "after")], 0.0206, 5e-5)
    return out


# ---------------------------------------------------------------- data: ensembles
R6 = json.load(open(os.path.join(RES, "exp6", "report6_data.json")))
KEY5 = {"static_wide": "static_wide_d5_long", "adaptive_wide": "adaptive_wide_d5_long",
        "adaptive_stale": "adaptive_stale_d5_long"}
KEY3 = {"static_wide": "static_d3_wide", "adaptive_wide": "adaptive_d3_wide",
        "adaptive_stale": "adaptive_d3_stale"}


def ens(d, model, regime, ph, fp="live"):
    keys = KEY5 if d == 5 else KEY3
    c = R6[f"d{d}"][keys[model]][f"{regime}|{fp}|{ph}"]
    return c["ens"], np.array(c["seeds"])


# checks against notes/07_ensembles.md
close(ens(5, "adaptive_stale", "bursts_big", "drift")[0], 0.0457, 5e-5)
close(ens(5, "static_wide", "bursts_big", "drift")[0], 0.0564, 5e-5)
close(ens(5, "adaptive_wide", "hot_qubit", "drift")[0], 0.0071, 5e-5)
close(ens(3, "adaptive_stale", "bursts", "drift")[0], 0.0720, 5e-5)
close(ens(3, "static_wide", "bursts_big", "drift")[0], 0.2427, 5e-5)
close(np.std(ens(5, "static_wide", "bursts_big", "drift")[1], ddof=1), 0.0153, 5e-5)


# ---------------------------------------------------------------- Fig: main bar chart (d=5)
def fig_main(base):
    fig, axes = plt.subplots(1, 4, figsize=(TEXTW, 2.1))
    order = ["static", "window", "oracle", "static_wide", "adaptive_wide", "adaptive_stale"]
    for ax, r in zip(axes, REGIMES):
        for i, k in enumerate(order):
            if k in ("static", "window", "oracle"):
                v = base[r][(k, "drift")]
                ax.bar(i, v, color="white" if k == "oracle" else COL[k], edgecolor=COL[k],
                       hatch=HATCH.get(k), lw=0.8, width=0.78)
            else:
                v, seeds = ens(5, k, r, "drift")
                ax.bar(i, v, color=COL[k], edgecolor=COL[k], lw=0.8, width=0.78)
                ax.errorbar(i, seeds.mean(), yerr=seeds.std(ddof=1), fmt="D", ms=2.6, color="#222222",
                            mfc="white", mew=0.7, elinewidth=0.8, capsize=2.0, zorder=4)
        ax.set_xticks([])
        ax.set_title(RLAB[r], fontsize=8.5)
        ax.set_ylim(0, None)
        ax.ticklabel_format(axis="y", style="plain")
        ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel("drift-step LER")
    handles = [Patch(facecolor="white" if k == "oracle" else COL[k], edgecolor=COL[k], hatch=HATCH.get(k),
                     label=LABEL[k] + (" ensemble" if k not in ("static", "window", "oracle") else ""))
               for k in order]
    handles.append(Line2D([], [], marker="D", ms=3, color="#222222", mfc="white", lw=0.8,
                          label="single seeds: mean $\\pm$ s.d."))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.17),
               columnspacing=1.2, handlelength=1.6)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(os.path.join(OUT, "main_d5.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig: live vs frozen
def fig_live_frozen():
    fig, axes = plt.subplots(2, 4, figsize=(TEXTW, 2.75))
    for row, d in enumerate([3, 5]):
        for ax, r in zip(axes[row], REGIMES):
            sw = ens(d, "static_wide", r, "drift")[0]
            fr = ens(d, "adaptive_stale", r, "drift", "frozen")[0]
            lv = ens(d, "adaptive_stale", r, "drift", "live")[0]
            vals = [sw, fr, lv]
            cols = [COL["static_wide"], "white", COL["adaptive_stale"]]
            ax.bar([0, 1, 2], vals, color=cols, edgecolor=[COL["static_wide"], COL["adaptive_stale"],
                                                          COL["adaptive_stale"]],
                   hatch=[None, "....", None], lw=0.9, width=0.75)
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(["static", "frozen", "live"], fontsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.set_ylim(0, max(vals) * 1.15)
            ax.text(2, lv + 0.03 * max(vals), f"{lv - fr:+.4f}", ha="center", va="bottom", fontsize=6.6)
            if row == 0:
                ax.set_title(RLAB[r], fontsize=8.5)
        axes[row, 0].set_ylabel(f"$d={d}$\ndrift-step LER")
    fig.tight_layout(h_pad=0.8, w_pad=1.0)
    fig.savefig(os.path.join(OUT, "live_frozen.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig: learning curves (d=5, seed 0)
def fig_curves():
    models = ["static_wide", "adaptive_wide", "adaptive_stale", "static_nominal", "h64"]
    stem = {"static_wide": "static_wide", "adaptive_wide": "adaptive_wide", "adaptive_stale": "adaptive_stale",
            "static_nominal": "static_nominal", "h64": "adaptive_stale_h64l6"}
    steps = list(range(30000, 300001, 30000))
    data = {}
    for m in models:
        rows = []
        for s in steps:
            f = os.path.join(RES, "exp4", "curves", f"decoder_{stem[m]}_d5_long_s0_step{s}.json")
            rows.append(json.load(open(f)))
        data[m] = rows
    # checks against notes/05_d5_long_training.md (learning-curve tables)
    close(data["adaptive_wide"][-1]["tests"]["nominal"]["neural"], 0.00210, 5e-6)
    close(data["static_nominal"][0]["tests"]["nominal"]["neural"], 0.01976, 5e-6)
    close(data["adaptive_wide"][-1]["streams"]["bursts_big"]["live"]["drift"], 0.0618, 5e-5)
    close(data["adaptive_wide"][-1]["streams"]["bursts_big"]["frozen"]["drift"], 0.0782, 5e-5)
    close(data["h64"][-1]["streams"]["bursts_big"]["live"]["drift"], 0.0590, 5e-5)
    sb = data["static_wide"][-1]["streams"]["bursts_big"]
    close(sb["static"]["drift"], 0.0904, 5e-5)
    close(sb["oracle"]["drift"], 0.0480, 5e-5)
    close(sb["window_1000"]["drift"], 0.0912, 5e-5)
    x = np.array(steps) / 1000
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.25))
    for m in models:
        y = [row["tests"]["nominal"]["neural"] for row in data[m]]
        axes[0].plot(x, y, "-o", ms=2.5, lw=1.1, color=COL[m], label=LABEL[m])
        y = [row["streams"]["bursts_big"]["live"]["drift"] for row in data[m]]
        axes[1].plot(x, y, "-o", ms=2.5, lw=1.1, color=COL[m])
        if m in ("adaptive_wide", "adaptive_stale"):
            y = [row["streams"]["bursts_big"]["frozen"]["drift"] for row in data[m]]
            axes[1].plot(x, y, ":", lw=1.1, color=COL[m])
    nom = data["static_wide"][0]["tests"]["nominal"]["static"]
    axes[0].axhline(nom, color=COL["static"], ls="--", lw=0.9)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("nominal test-set LER")
    axes[0].set_title("nominal test set (200 000 shots)", fontsize=8.5)
    for k, lab in [("static", "static MWPM"), ("window_1000", "best WindowDEM (w=1000)"),
                   ("oracle", "oracle MWPM")]:
        c = COL["window"] if k.startswith("window") else COL[k]
        axes[1].axhline(sb[k]["drift"], color=c, ls="--", lw=0.9)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("bursts_big drift-step LER")
    axes[1].set_title("bursts_big, stream seed 0", fontsize=8.5)
    from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter
    axes[0].yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    axes[1].yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    for ax in axes:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("training steps (thousands)")
        ax.set_xticks([30, 90, 150, 210, 270])
        ax.tick_params(labelsize=7.5)
    hs, ls = axes[0].get_legend_handles_labels()
    hs += [Line2D([], [], color="#555555", ls=":", lw=1.1),
           Line2D([], [], color=COL["static"], ls="--", lw=0.9),
           Line2D([], [], color=COL["window"], ls="--", lw=0.9),
           Line2D([], [], color=COL["oracle"], ls="--", lw=0.9)]
    ls += ["frozen fingerprint", "static MWPM (= oracle on nominal)", "best WindowDEM (w=1000)", "oracle MWPM"]
    fig.legend(hs, ls, loc="lower center", ncol=3, fontsize=7.3, bbox_to_anchor=(0.5, -0.17),
               columnspacing=1.2, handlelength=1.8)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(OUT, "curves_d5.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig: phases on bursts and bursts_big
def fig_phases(base):
    phases = ["onset", "later", "after"]
    order = ["static", "window", "oracle", "static_wide", "adaptive_wide", "adaptive_stale"]
    fig, axes = plt.subplots(2, 3, figsize=(TEXTW, 2.9))
    for row, r in enumerate(["bursts", "bursts_big"]):
        for col, ph in enumerate(phases):
            ax = axes[row, col]
            for i, k in enumerate(order):
                if k in ("static", "window", "oracle"):
                    v = base[r][(k, ph)]
                    ax.bar(i, v, color="white" if k == "oracle" else COL[k], edgecolor=COL[k],
                           hatch=HATCH.get(k), lw=0.8, width=0.78)
                else:
                    v, seeds = ens(5, k, r, ph)
                    ax.bar(i, v, color=COL[k], edgecolor=COL[k], lw=0.8, width=0.78)
                    ax.errorbar(i, seeds.mean(), yerr=seeds.std(ddof=1), fmt="D", ms=2.4, color="#222222",
                                mfc="white", mew=0.7, elinewidth=0.8, capsize=1.8, zorder=4)
            ax.set_xticks([])
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(ph, fontsize=8.5)
            if col == 0:
                ax.set_ylabel(f"{RLAB[r]}\nLER", fontsize=8)
    handles = [Patch(facecolor="white" if k == "oracle" else COL[k], edgecolor=COL[k], hatch=HATCH.get(k),
                     label=LABEL[k] + (" ens." if k not in ("static", "window", "oracle") else ""))
               for k in order]
    handles.append(Line2D([], [], marker="D", ms=3, color="#222222", mfc="white", lw=0.8,
                          label="seeds: mean $\\pm$ s.d."))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.3, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(h_pad=0.6, w_pad=1.0)
    fig.savefig(os.path.join(OUT, "phases_d5.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig: localisation
def fig_localisation():
    L = json.load(open(os.path.join(RES, "exp9", "localisation_baseline.json")))["regimes"]
    # checks against results/exp9/localisation_baseline.md
    close(L["bursts"]["drift"]["baseline"]["test-set window"], 0.829, 5e-4)
    close(L["bursts_big"]["later"]["models"]["NF s1"]["topk"], 0.958, 5e-4)
    regs = ["bursts", "bursts_big", "hot_qubit"]
    phases = ["onset", "later"]
    fig, ax = plt.subplots(figsize=(TEXTW * 0.8, 2.4))
    xt, xl = [], []
    x = 0.0
    for r in regs:
        for ph in phases:
            c = L[r][ph]
            b = c["baseline"]["test-set window"]
            nf = np.array([c["models"][f"NF s{s}"]["topk"] for s in range(3)])
            ax.bar(x - 0.2, b, width=0.38, color=COL["fp_baseline"], edgecolor=COL["fp_baseline"])
            ax.bar(x + 0.2, nf.mean(), width=0.38, color=COL["nf"], edgecolor=COL["nf"])
            ax.plot([x + 0.2] * 3, nf, "o", ms=2.6, mfc="white", mec="#222222", mew=0.7, zorder=4)
            xt.append(x)
            xl.append(f"{ph}\n({c['events']} ev.)")
            x += 1
        x += 0.5
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=7)
    ax.set_ylim(0.6, 1.04)
    ax.set_ylabel("top-$k$ hot-qubit hit rate")
    ax.tick_params(axis="y", labelsize=7.5)
    for i, r in enumerate(regs):
        ax.text(i * 2.5 + 0.5, 1.045, RLAB[r], ha="center", va="bottom", fontsize=8)
    ax.legend(handles=[Patch(color=COL["fp_baseline"], label="training-free fingerprint score"),
                       Patch(color=COL["nf"], label="NF noise-map head (mean of 3 seeds)"),
                       Line2D([], [], marker="o", ls="", ms=3, mfc="white", mec="#222222",
                              label="NF single seeds")],
              loc="upper center", fontsize=6.8, bbox_to_anchor=(0.5, -0.22), ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "localisation.pdf"))
    plt.close(fig)

# ---------------------------------------------------------------- illustration figures (simulator, not results)
def fig_simulator():
    import sys
    from matplotlib.colors import LinearSegmentedColormap

    sys.path.insert(0, ROOT)
    from circuits import nearest_qubits  # noqa: E402
    from drift import DriftStream, NoiseState, make_schedule  # noqa: E402

    stream = DriftStream(5, 5, 0.003)
    nom = stream.nominal()

    # same mapping as decoder.DetectorGrid (re-implemented here so torch is not needed)
    coords_det = nom.get_detector_coordinates()
    c = np.array([coords_det[i][:3] for i in range(nom.num_detectors)], dtype=float)
    axes_v = [np.unique(c[:, k]) for k in range(3)]
    idx = [np.searchsorted(axes_v[k], c[:, k]) for k in range(3)]
    gx, gy, gt = (len(a) for a in axes_v)
    shape = (gt, gx, gy)
    flat = (idx[2] * gx + idx[0]) * gy + idx[1]

    def to_grid(v):
        out = np.zeros(int(np.prod(shape)), dtype=float)
        out[flat] = v
        return out.reshape(shape)

    occ = to_grid(np.ones(nom.num_detectors)) > 0
    assert shape == (6, 6, 6) and occ.sum() == 120

    # qubits, exactly as tech_doc_src/figs.py
    meas_q = set()
    for inst in nom.flattened():
        if inst.name == "MR":
            meas_q |= {t.value for t in inst.targets_copy()}
    qc = {q: v[:2] for q, v in nom.get_final_qubit_coordinates().items()}
    hot = nearest_qubits(nom, 2)
    burst = nearest_qubits(nom, 5, center=qc[sorted(meas_q)[3]])

    # --- drift regimes: the four evaluated regimes, one schedule each (rng seed 3, as before)
    regs = ["telegraph", "hot_qubit", "bursts", "bursts_big"]
    fig, axs = plt.subplots(2, 2, figsize=(TEXTW, 2.2), sharex=True)
    for ax, r in zip(axs.flat, regs):
        sch = make_schedule(r, 150, np.random.default_rng(3), nom)
        sc = np.array([x.scale for x in sch])
        load = np.array([len(x.hot_qubits) * x.hot_p for x in sch])
        ax2 = ax.twinx()
        ax2.bar(np.arange(150), load, width=1.0, color=COL["window"], alpha=0.75, lw=0, zorder=1)
        ax2.set_ylim(0, 0.9)
        ax2.set_yticks([0, 0.4, 0.8])
        ax2.tick_params(labelsize=7)
        ax2.spines["top"].set_visible(False)
        ax.plot(np.arange(150), sc, color=COL["adaptive_stale"], lw=1.3, zorder=3)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.set_ylim(0, 2.6)
        ax.set_yticks([0, 1, 2])
        ax.tick_params(labelsize=7)
        ax.set_title(r, fontsize=8, family="monospace", pad=2)
        if r in ("hot_qubit", "bursts_big"):
            ax2.set_ylabel("hot load", fontsize=7.5)
    for ax in axs[1]:
        ax.set_xlabel("step (2000 shots each)", fontsize=7.5)
    for ax in axs[:, 0]:
        ax.set_ylabel("scale $s_t$", fontsize=7.5)
    fig.legend(handles=[Line2D([], [], color=COL["adaptive_stale"], lw=1.3, label="global scale $s_t$ (left axis)"),
                        Patch(color=COL["window"], alpha=0.75, label=r"hot load $|H|\,p_{\mathrm{hot}}$ (right axis)")],
               loc="upper center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.89), h_pad=0.4, w_pad=1.5)
    fig.savefig(os.path.join(OUT, "regimes.pdf"))
    plt.close(fig)

    # --- two shots (sample seed 7, as before)
    det_n, _ = stream.sample(NoiseState(), 4000, 7)
    det_h, _ = stream.sample(NoiseState(hot_qubits=hot, hot_p=0.03), 4000, 7)

    def pick(det):
        k = det.sum(1)
        return int(np.argmin(np.abs(k - np.median(k[k > 0]) - 1)))

    fig, axs = plt.subplots(2, 6, figsize=(TEXTW * 0.9, 2.15))
    for row, (det, lab) in enumerate([(det_n, "nominal"), (det_h, "2 hot qubits")]):
        i = pick(det)
        g = to_grid(det[i].astype(float))
        for t in range(6):
            ax = axs[row, t]
            ax.imshow(np.where(occ[t], 0.15, np.nan), cmap="Greys", vmin=0, vmax=1, interpolation="none")
            yy, xx = np.nonzero(g[t] > 0)
            ax.scatter(xx, yy, s=38, color=COL["h64"], marker="s", lw=0, zorder=3)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if row == 0:
                ax.set_title(f"$t={t}$", fontsize=7.5, pad=2)
        axs[row, 0].set_ylabel(f"{lab}\n{int(det[i].sum())} events", fontsize=7.5)
    fig.tight_layout(w_pad=0.3, h_pad=0.3)
    fig.savefig(os.path.join(OUT, "shot.pdf"))
    plt.close(fig)

    # --- fingerprints (sample seed 11, as before): mean over the four middle time layers
    states = [("nominal", NoiseState()), ("scale x2", NoiseState(scale=2.0)),
              ("2 hot qubits", NoiseState(hot_qubits=hot, hot_p=0.03)),
              ("5-qubit burst", NoiseState(hot_qubits=burst, hot_p=0.05))]
    maps = []
    for _, st in states:
        w, _ = stream.sample(st, 2000, 11)
        g = to_grid(w.mean(0))
        with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)  # grid cells with no detector are NaN
            maps.append(np.nanmean(np.where(occ[1:-1], g[1:-1], np.nan), axis=0))
    vmax = max(np.nanmax(m) for m in maps)
    cmap = LinearSegmentedColormap.from_list("blues_oi", ["#ffffff", "#56B4E9", "#0072B2", "#002b45"])
    fig, axs = plt.subplots(1, 4, figsize=(TEXTW * 0.84, 1.75))
    for ax, (lab, _), m in zip(axs, states, maps):
        im = ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax, interpolation="none")
        ax.set_title(lab, fontsize=7.5, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
    cb.set_label("firing rate", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(os.path.join(OUT, "fingerprint.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    base = mwpm_d5()
    fig_main(base)
    fig_live_frozen()
    fig_curves()
    fig_phases(base)
    fig_localisation()
    fig_simulator()
    print("figures written to", OUT)
    for r in REGIMES:
        print(r, {k: round(v, 5) for k, v in base[r].items() if k[0] in ("static", "window", "oracle")})
