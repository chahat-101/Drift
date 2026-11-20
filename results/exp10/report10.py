"""Round 10 report: physics-based drift, d=5, existing models only.

    python results/exp10/report10.py     # reads errors/ and data/phys10/, writes
        results/exp10/report10_tables.md, report10_data.json, ler_per_regime.png, ler_vs_burst_bin.png

Reads only what generate_streams.py and eval_streams.py wrote. Streams are pooled over the
5 stream seeds. McNemar exact two-sided on shared shots (report.mcnemar_p, as report6);
event-level exact two-sided sign tests, ties dropped, over the events each regime has
(bursts: phys_cosmic; TLS: phys_tls; couplers: phys_coupler; all three: phys_mixed).
p < 0.001 significant, 0.001-0.05 weak, no correction. The best WindowDEM is chosen after
the fact (per regime and slice) and reported as such. H1-H3 as in
notes/10_physics_drift_hypotheses.md.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from physics_noise import PHYS_REGIMES  # noqa: E402
from report import mcnemar_p  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
WINDOWS = (500, 1000, 2000, 5000, 10000)
SHOTS = 200
TYPES = ("static_wide", "adaptive_wide", "adaptive_stale", "static_nominal")
LABELS = {
    "mwpm_cal": "MWPM (calibrated)", "mwpm_nom": "MWPM (nominal, ref)", "mwpm_oracle": "MWPM (oracle)",
    "static_wide_live_ens": "static-wide ens", "static_nominal_live_ens": "static-nominal ens",
    "adaptive_wide_live_ens": "adaptive-wide ens (live)", "adaptive_wide_frozen_ens": "adaptive-wide ens (frozen)",
    "adaptive_stale_live_ens": "adaptive-stale ens (live)", "adaptive_stale_frozen_ens": "adaptive-stale ens (frozen)",
}
NEURAL = {  # ensemble keys shown in every table
    "static_wide": ("static_wide_live_ens", ("static_wide_live_m0", "static_wide_live_m1", "static_wide_live_m2")),
    "static_nominal": ("static_nominal_live_ens", ("static_nominal_live_m0", "static_nominal_live_m1", "static_nominal_live_m2")),
    "adaptive_wide": ("adaptive_wide_live_ens", ("adaptive_wide_live_m0", "adaptive_wide_live_m1", "adaptive_wide_live_m2")),
    "adaptive_stale": ("adaptive_stale_live_ens", ("adaptive_stale_live_m0", "adaptive_stale_live_m1", "adaptive_stale_live_m2")),
}
EVENT_KINDS = {"phys_cosmic": ("burst",), "phys_tls": ("tls",), "phys_coupler": ("coupler",),
               "phys_mixed": ("burst", "tls", "coupler"), "phys_static": (), "phys_slow": ()}
EVENT_COLUMNS = {"burst": (0, 1), "tls": (2, 3), "coupler": (1, 2)}
SLICES = ("all", "drift", "nominal", "bin0", "bin1", "bin2")


def unpack(a):
    return np.unpackbits(a, axis=-1).astype(bool)


def load_regime(regime, data="data/phys10", out="results/exp10"):
    """(bits, masks, events): per-seed error-bit arrays, (drift, burst_bin, max_burst) and events."""
    bits, masks, events = {}, [], []
    for seed in SEEDS:
        with np.load(f"{out}/errors/{regime}_s{seed}.npz") as z:
            for k in z.files:
                bits.setdefault(k, []).append(unpack(z[k]))
        with np.load(f"{data}/stream_{regime}_s{seed}.npz") as z:
            masks.append((z["drift"], z["burst_bin"], z["burst"].max(axis=1)))
            events.append({kind: z[f"{kind}_events"] for kind in ("burst", "tls", "coupler")})
    return bits, masks, events


def pooled(bits, masks, regime_slice):
    """Concatenated shots of one decoder over seeds, on one slice."""
    out = []
    for per_seed, (drift, bins, _) in zip(bits, masks):
        m = {"all": np.ones(len(drift), bool), "drift": drift, "nominal": ~drift,
             "bin0": bins == 0, "bin1": bins == 1, "bin2": bins == 2}[regime_slice]
        out.append(per_seed[m])
    return np.concatenate(out)


def ler(bits, masks, regime_slice):
    x = pooled(bits, masks, regime_slice)
    return None if len(x) == 0 else float(x.mean())


def pair(a, b):
    """(LER diff, exact two-sided McNemar p) on shared shots (arrays of any shape)."""
    na, nb = int((a & ~b).sum()), int((~a & b).sum())
    return (na - nb) / a.size, mcnemar_p(na, nb)


def sign_test(a_bits, b_bits, events):
    """Exact two-sided sign test over events, ties dropped; per-seed bits + (s, e) windows."""
    wa = wb = ties = n = 0
    for a, b, ev in zip(a_bits, b_bits, events):
        for s, e in ev:
            ca, cb = int(a[s:e].sum()), int(b[s:e].sum())
            n += 1
            wa, wb, ties = wa + (ca < cb), wb + (ca > cb), ties + (ca == cb)
    return wa, wb, ties, n, mcnemar_p(wa, wb)


def event_windows(events, kinds):
    """[(s, e)] sub-step windows of the given event kinds in one stream's events dict."""
    if not kinds:
        return np.empty((0, 2), dtype=int)
    return np.concatenate([events[k][:, EVENT_COLUMNS[k]] for k in kinds])


def custom_pooled(bits_list, masks_list):
    """Concatenated shots of one decoder over seeds, on custom per-seed sub-step masks."""
    return np.concatenate([b[m] for b, m in zip(bits_list, masks_list)])


def tls_only_masks(data="data/phys10"):
    """Per-seed masks of phys_mixed sub-steps whose only drift cause is a fluctuator on."""
    out = []
    for seed in SEEDS:
        with np.load(f"{data}/stream_phys_mixed_s{seed}.npz") as z:
            t = len(z["drift"])
            tls = z["tls_on"].reshape(t, -1).any(axis=1)
            coupler = z["coupler_on"].reshape(t, -1).any(axis=1)
            cosmic = z["burst"].max(axis=1) >= 0.005
            out.append(tls & ~cosmic & ~coupler)
    return out


def best_window(bits, masks, regime_slice):
    """(key, LER) of the best WindowDEM on one slice, chosen after the fact."""
    return min(((f"window_{w}", ler(bits[f"window_{w}"], masks, regime_slice)) for w in WINDOWS),
               key=lambda kv: kv[1])


def fmt(diff, p, nd=4):
    s = f"{diff:+.{nd}f} (p={p:.1g})" if p > 0 else f"{diff:+.{nd}f} (p<1e-300)"
    return f"**{s}**" if p < 0.001 else s


def f4(x):
    return "–" if x is None else f"{x:.4f}"


def main():
    data, header, md = {}, [], []
    md += ["# Report 10 tables (generated by results/exp10/report10.py)\n",
           "Physics-based drift (SI1000 base device, p=0.0015, device seed 0), 6 regimes x stream "
           f"seeds {SEEDS}, 1500 sub-steps x 200 shots each, pooled over stream seeds. Existing d=5 "
           "models only (seeds 0-1-2 ensembles, mean logit > 0), evaluated causally in 200-shot "
           "sub-batches on ensemble_eval.py's path. Frozen = fingerprint of the 2000-shot static-device "
           "calibration set. McNemar exact two-sided on shared shots; event-level exact two-sided sign "
           "tests (ties dropped) over the events each regime has (phys_mixed: bursts + TLS + couplers; "
           "phys_slow and phys_static have no events); p < 0.001 significant (bold), 0.001-0.05 weak, "
           "no correction. The best WindowDEM window is picked per regime and slice after the fact.\n"]

    R = {r: load_regime(r) for r in PHYS_REGIMES}

    # ---- sub-step census (honest reporting of the drift masks)
    md += ["## Sub-step census (pooled over stream seeds)\n",
           "| regime | drift sub-steps | nominal sub-steps | bin 0 | bin 1 | bin 2 | drift, no active "
           "burst | binned, not drift | events (burst/TLS/coupler) |", "|---|" + "---|" * 8]
    for r in PHYS_REGIMES:
        _, masks, events = R[r]
        drift = sum(int(m[0].sum()) for m in masks)
        total = sum(len(m[0]) for m in masks)
        edge = sum(int(((m[2] >= 0.005) & (m[1] < 0)).sum()) for m in masks)
        binned_not_drift = sum(int(((m[1] >= 0) & ~m[0]).sum()) for m in masks)
        counts = {k: sum(len(e[k]) for e in events) for k in ("burst", "tls", "coupler")}
        empty = sum(1 for e in events for s, f in e["burst"] if s == f)
        row = [f"{drift}", f"{total - drift}",
               *(f"{sum(int((m[1] == b).sum()) for m in masks)}" for b in range(3))]
        md.append(f"| {r} | " + " | ".join(row) + f" | {edge} | {binned_not_drift} | "
                  f"{counts['burst']}/{counts['tls']}/{counts['coupler']} |")
        data["census"] = data.get("census", {})
        data["census"][r] = {"drift": drift, "nominal": total - drift, "edge_case": edge,
                             "binned_not_drift": binned_not_drift, "events": counts,
                             "empty_burst_events": empty}
    md.append("")
    md.append("Bins follow the youngest active burst's amplitude; they can include sub-steps whose max "
              "burst is < 0.005 (impact far from every qubit), and drift by max burst can occur with no "
              "active burst (summed older bursts) — the edge case allowed for in the definitions.\n")
    header_pos = len(md)

    # ---- H1: adaptive-stale live vs frozen on drift steps of tls/cosmic/mixed
    h1 = {}
    md += ["## H1: adaptive-stale ensemble, live vs frozen, drift sub-steps\n",
           "| regime | ΔLER (live−frozen) | McNemar p | sign test (live better/total, p) | verdict |",
           "|---|---|---|---|---|"]
    h1_pass = True
    for r in ("phys_tls", "phys_cosmic", "phys_mixed"):
        bits, masks, events = R[r]
        live = pooled(bits["adaptive_stale_live_ens"], masks, "drift")
        frozen = pooled(bits["adaptive_stale_frozen_ens"], masks, "drift")
        diff, p = pair(live, frozen)
        wa, wb, ties, n, sp = sign_test(bits["adaptive_stale_live_ens"],
                                        bits["adaptive_stale_frozen_ens"],
                                        [event_windows(e, EVENT_KINDS[r]) for e in events])
        ok = diff < 0 and p < 0.001 and wa > wb and sp < 0.01
        h1_pass &= ok
        h1[r] = {"diff": diff, "mcnemar_p": p, "sign_wins": wa, "sign_losses": wb,
                 "sign_ties": ties, "sign_n": n, "sign_p": sp, "pass": ok}
        md.append(f"| {r} | {diff:+.5f} | {p:.1g} | {wa}/{n - ties}, p={sp:.1g} | "
                  f"{'PASS' if ok else 'FAIL'} |")
    md.append("\nOverlapping events share shots, so events are not fully independent; event-level "
              "p-values for phys_tls and phys_mixed are approximate.\n")
    md.append(f"\n**H1: {'PASS' if h1_pass else 'FAIL'}** (all three regimes: live better, McNemar "
              "p < 0.001, sign test p < 0.01).\n")
    data["H1"] = {"pass": h1_pass, "regimes": h1}

    # ---- H2: adaptive-stale live vs calibrated MWPM and best WindowDEM, drift steps of cosmic/mixed
    h2 = {}
    md += ["## H2: adaptive-stale ensemble (live) vs baselines, drift sub-steps\n",
           "| regime | vs | ΔLER | McNemar p | best window | verdict |", "|---|---|---|---|---|---|"]
    h2_pass = True
    for r in ("phys_cosmic", "phys_mixed"):
        bits, masks, _ = R[r]
        live = pooled(bits["adaptive_stale_live_ens"], masks, "drift")
        for ref, label in (("mwpm_cal", "calibrated MWPM"), (None, "best WindowDEM")):
            if ref is None:
                key, wler = best_window(bits, masks, "drift")
                ref_bits = pooled(bits[key], masks, "drift")
                label += f" ({key.split('_')[1]})"
                window_cell = f"{wler:.4f}"
            else:
                ref_bits = pooled(bits[ref], masks, "drift")
                window_cell = "–"
            diff, p = pair(live, ref_bits)
            ok = diff < 0 and p < 0.001
            h2_pass &= ok
            h2[f"{r}|{label}"] = {"diff": diff, "mcnemar_p": p, "pass": ok}
            md.append(f"| {r} | {label} | {diff:+.5f} | {p:.1g} | {window_cell} | "
                      f"{'PASS' if ok else 'FAIL'} |")
    md.append(f"\n**H2: {'PASS' if h2_pass else 'FAIL'}** (both regimes, both baselines: ΔLER < 0, "
              "p < 0.001).\n")
    data["H2"] = {"pass": h2_pass, "tests": h2}

    # ---- H3: neural models vs calibrated MWPM on phys_static (all sub-steps nominal)
    h3 = {}
    md += ["## H3 (risk): neural ensembles vs calibrated MWPM on phys_static (all sub-steps)\n",
           "| model | ΔLER (model−MWPM) | McNemar p | worse? |", "|---|---|---|---|"]
    bits, masks, _ = R["phys_static"]
    cal = pooled(bits["mwpm_cal"], masks, "all")
    h3_pass = True
    for t, (ens, singles) in NEURAL.items():
        m = pooled(bits[ens], masks, "all")
        diff, p = pair(m, cal)
        worse = diff > 0 and p < 0.001
        h3_pass &= worse
        h3[t] = {"diff": diff, "mcnemar_p": p, "worse": worse}
        singles_ler = [ler(bits[s], masks, "all") for s in singles]
        md.append(f"| {LABELS[ens]} | {diff:+.5f} | {p:.1g} | {'yes' if worse else 'no'}"
                  f" (seeds: {'/'.join(f'{x:.5f}' for x in singles_ler)}) |")
    md.append(f"\n**H3: {'PASS (all four ensembles significantly worse)' if h3_pass else 'FAIL'}** "
              "(marked PASS iff every ensemble is significantly worse than calibrated MWPM, p < 0.001).\n")
    data["H3"] = {"pass": h3_pass, "models": h3}

    # ---- header table with the verdicts and deciding numbers
    md[header_pos:header_pos] = [
        "## Hypotheses (notes/10_physics_drift_hypotheses.md)\n",
        "| H | statement | verdict | deciding numbers (drift sub-steps, pooled seeds) |",
        "|---|---|---|---|",
        f"| H1 | adaptive-stale ensemble (live) beats its frozen version on drift steps of phys_tls, "
        f"phys_cosmic, phys_mixed (McNemar p < 0.001 and event sign p < 0.01) | "
        f"**{'PASS' if h1_pass else 'FAIL'}** | " + "; ".join(
            f"{r[5:]}: ΔLER {h1[r]['diff']:+.2e}, p={h1[r]['mcnemar_p']:.1g}, sign "
            f"{h1[r]['sign_wins']}/{h1[r]['sign_wins'] + h1[r]['sign_losses']}, p={h1[r]['sign_p']:.1g}"
            for r in h1) + " |",
        f"| H2 | adaptive-stale ensemble (live) beats calibrated MWPM and the best WindowDEM on drift "
        f"steps of phys_cosmic and phys_mixed (p < 0.001) | **{'PASS' if h2_pass else 'FAIL'}** | "
        + "; ".join(f"{k.replace('|', ' ')}: ΔLER {v['diff']:+.2e}, p={v['mcnemar_p']:.1g}"
                    for k, v in h2.items()) + " |",
        f"| H3 (risk) | the neural models are worse than calibrated MWPM on the physics base device "
        f"with no drift | **{'PASS' if h3_pass else 'FAIL'}** | " + "; ".join(
            f"{t}: ΔLER {v['diff']:+.2e}, p={v['mcnemar_p']:.1g}, {'worse' if v['worse'] else 'better'}"
            for t, v in h3.items()) + " |",
        ""]

    # ---- LER tables per regime
    md += ["## LER per regime (pooled over stream seeds; – = no such sub-steps)\n"]
    for r in PHYS_REGIMES:
        bits, masks, _ = R[r]
        md += [f"### {r}\n", "| decoder | all | drift | nominal | bin [0.05,0.2] | bin [0.0125,0.05) | "
               "bin [0.005,0.0125) |", "|---|" + "---|" * 6]
        rows = [("mwpm_cal",), ("mwpm_nom",), ("mwpm_oracle",)] + [(f"window_{w}",) for w in WINDOWS]
        for (k,) in rows:
            md.append(f"| {LABELS.get(k, 'WindowDEM w=' + k.split('_')[1])} | " +
                      " | ".join(f4(ler(bits[k], masks, s)) for s in SLICES) + " |")
        for t, (ens, singles) in NEURAL.items():
            kinds = ("live",) if t.startswith("static") else ("live", "frozen")
            for kind in kinds:
                k = f"{t}_{kind}_ens"
                md.append(f"| {LABELS[k]} | " + " | ".join(f4(ler(bits[k], masks, s)) for s in SLICES) + " |")
                sm = {s2: [ler(bits[s], masks, s2) for s in singles] for s2 in SLICES}
                md.append(f"|   seeds (mean ± std) | " + " | ".join(
                    "–" if any(sm[s2][i] is None for i in range(3)) else
                    f"{np.mean(sm[s2]):.4f} ± {np.std(sm[s2], ddof=1):.4f}" for s2 in SLICES) + " |")
        for k in bits:
            for s2 in SLICES:
                data.setdefault("lers", {}).setdefault(r, {}).setdefault(k, {})[s2] = \
                    ler(bits[k], masks, s2)
        md.append("")

    # ---- McNemar + sign tests per regime (the comparisons fixed in the definitions)
    md += ["## Paired comparisons (ΔLER, exact McNemar p; bold = p < 0.001)\n",
           "Comparison list fixed in notes/10_definitions.md; drift sub-steps (all sub-steps for "
           "phys_static, which has none). Sign test: events of the regime, ties dropped, exact "
           "two-sided; regimes without events show no sign test.\n",
           "Overlapping events share shots, so events are not fully independent; event-level "
           "p-values for phys_tls and phys_mixed are approximate.\n",
           "| regime | comparison | slice ΔLER | McNemar p | sign (A better/B better/ties, p) |",
           "|---|---|---|---|---|"]
    pairs = [("adaptive_stale_live_ens", "adaptive_stale_frozen_ens", "adaptive-stale live vs frozen"),
             ("adaptive_stale_live_ens", "mwpm_cal", "adaptive-stale live vs calibrated MWPM"),
             ("adaptive_stale_live_ens", None, "adaptive-stale live vs best WindowDEM"),
             ("adaptive_stale_live_ens", "mwpm_oracle", "adaptive-stale live vs oracle MWPM"),
             ("adaptive_stale_live_ens", "static_wide_live_ens", "adaptive-stale live vs static-wide ens")]
    for r in PHYS_REGIMES:
        bits, masks, events = R[r]
        slice_name = "drift" if masks[0][0].any() else "all"
        evs = ([event_windows(e, EVENT_KINDS[r]) for e in events]
               if EVENT_KINDS[r] else None)
        for a_key, b_key, label in pairs:
            a = pooled(bits[a_key], masks, slice_name)
            if b_key is None:
                key, _ = best_window(bits, masks, slice_name)
                b_key, label2 = key, f"{label} ({key.split('_')[1]})"
            else:
                label2 = label
            b = pooled(bits[b_key], masks, slice_name)
            diff, p = pair(a, b)
            if evs is None:
                sign = "no events"
                wa = wb = ties = n = sp = None
            else:
                wa, wb, ties, n, sp = sign_test(bits[a_key], bits[b_key], evs)
                sign = f"{wa}/{wb}/{ties}, p={sp:.1g}"
            md.append(f"| {r} | {label2} | {diff:+.5f} | {p:.1g} | {sign} |")
            data.setdefault("paired", {})[f"{r}|{label2}"] = {
                "diff": diff, "mcnemar_p": p, "sign_a_better": wa, "sign_b_better": wb,
                "sign_ties": ties, "sign_n": n, "sign_p": sp}
    md.append("")

    # ---- exploratory (post hoc, not hypothesis tests)
    md += ["## Exploratory (post hoc)\n",
           "adaptive-stale ensemble, live vs frozen (the H1 comparison), sliced after the fact. "
           "Not hypothesis tests.\n",
           "| regime | slice | shots | ΔLER (live−frozen) | McNemar p | LER live | LER frozen |",
           "|---|---|---|---|---|---|---|"]
    expl = {}
    for r in ("phys_cosmic", "phys_mixed"):
        bits, masks, _ = R[r]
        for b, lo, hi in ((0, "0.05–0.2", "0.05"), (1, "0.0125–0.05", "0.0125"), (2, "0.005–0.0125", "0.005")):
            sel = [m[1] == b for m in masks]
            live = custom_pooled(bits["adaptive_stale_live_ens"], sel)
            frozen = custom_pooled(bits["adaptive_stale_frozen_ens"], sel)
            diff, p = pair(live, frozen)
            md.append(f"| {r} | burst bin {lo} | {live.size} | {diff:+.5f} | {p:.1g} | "
                      f"{live.mean():.4f} | {frozen.mean():.4f} |")
            expl[f"{r}|bin{b}"] = {"shots": int(live.size), "diff": diff, "mcnemar_p": p,
                                   "ler_live": float(live.mean()), "ler_frozen": float(frozen.mean())}
    bits, masks, events = R["phys_mixed"]
    sel = tls_only_masks()
    live = custom_pooled(bits["adaptive_stale_live_ens"], sel)
    frozen = custom_pooled(bits["adaptive_stale_frozen_ens"], sel)
    diff, p = pair(live, frozen)
    md.append(f"| phys_mixed | TLS-only drift sub-steps | {live.size} | {diff:+.5f} | {p:.1g} | "
              f"{live.mean():.5f} | {frozen.mean():.5f} |")
    expl["phys_mixed|tls_only"] = {"shots": int(live.size), "diff": diff, "mcnemar_p": p,
                                   "ler_live": float(live.mean()), "ler_frozen": float(frozen.mean())}
    md += ["",
           "Composition of the phys_mixed disagreement between the pooled McNemar (favours frozen) "
           "and the event-level sign test (favours live): the pooled McNemar is dominated by burst "
           "errors — burst sub-steps are ~10% of drift sub-steps but carry LER ~0.49, favouring "
           "frozen — while the sign test is dominated by the numerous TLS events (992 of 1029 events), "
           "which favour live. Per-kind event sign counts (live better / frozen better / ties):\n"]
    kinds = {}
    for kind in ("burst", "tls", "coupler"):
        evs = [event_windows(e, (kind,)) for e in events]
        wa, wb, ties, n, sp = sign_test(bits["adaptive_stale_live_ens"], bits["adaptive_stale_frozen_ens"], evs)
        kinds[kind] = {"live_better": wa, "frozen_better": wb, "ties": ties, "n": n, "p": sp}
        md.append(f"- phys_mixed {kind} events: {wa}/{wb}/{ties} (n={n}), p={sp:.1g}")
    md.append("")
    expl["phys_mixed|sign_by_kind"] = kinds
    data["exploratory"] = expl
    md.append("")

    # ---- plots
    plot_regimes(R)
    plot_bins(R)
    json.dump(data, open("results/exp10/report10_data.json", "w"), indent=1)
    open("results/exp10/report10_tables.md", "w").write("\n".join(md) + "\n")
    print("wrote results/exp10/report10_tables.md, report10_data.json, and 2 plots")


PLOT_KEYS = ["mwpm_cal", "mwpm_oracle", "static_wide_live_ens", "static_nominal_live_ens",
             "adaptive_wide_live_ens", "adaptive_stale_live_ens", "adaptive_stale_frozen_ens"]


def plot_regimes(R):
    """Drift LER per regime for all decoders (phys_static has no drift: shows all)."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(PHYS_REGIMES))
    for k in PLOT_KEYS:
        ys = []
        for r in PHYS_REGIMES:
            bits, masks, _ = R[r]
            v = ler(bits[k], masks, "drift")
            if v is None:
                v = ler(bits[k], masks, "all")
            ys.append(v)
        ax.plot(x, ys, marker="o", label=LABELS[k])
    for r, xx in zip(PHYS_REGIMES, x):
        bits, masks, _ = R[r]
        if not masks[0][0].any():
            ax.text(xx, ax.get_ylim()[0] * 1.3, "all shots\n(no drift)", fontsize=7, ha="center")
    ax.set(xticks=x, xticklabels=PHYS_REGIMES, ylabel="LER (drift sub-steps)", yscale="log",
           title="Round 10: LER on drift sub-steps per regime (pooled stream seeds 0-4)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/exp10/ler_per_regime.png", dpi=160)
    plt.close(fig)


def plot_bins(R):
    """LER vs burst bin, pooled over phys_cosmic + phys_mixed."""
    bits_all = {k: [] for k in PLOT_KEYS}
    masks_all = []
    for r in ("phys_cosmic", "phys_mixed"):
        bits, masks, _ = R[r]
        masks_all += masks
        for k in PLOT_KEYS:
            bits_all[k] += bits[k]
    names = ["bin [0.05, 0.2]", "bin [0.0125, 0.05)", "bin [0.005, 0.0125)", "nominal"]
    slices = ["bin0", "bin1", "bin2", "nominal"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in PLOT_KEYS:
        ys = [ler(bits_all[k], masks_all, s) for s in slices]
        ax.plot(names, ys, marker="o", label=LABELS[k])
    ax.set(ylabel="LER", yscale="log", title="LER vs burst amplitude bin (phys_cosmic + phys_mixed, pooled)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/exp10/ler_vs_burst_bin.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
