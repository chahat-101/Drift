"""Tables for report 9: noise-map head (N) and noise-map head + spatial FiLM (NF) at d=5.

    python results/exp9/report9.py     # writes results/exp9/report9_tables.md and report9_data.json

Reads only existing files. Variants (3 model seeds each unless the deadline cut training; ensemble = mean logit > 0):
  base   adaptive-stale d=5 300k, single fingerprint: results/exp6/d5/st<N>/ens_adaptive_stale_d5_long (reused)
  sw     static-wide d=5 300k: results/exp6/d5/st<N>/ens_static_wide_d5_long (reused)
  N      --aux-noise: <exp>/d5/st<N>/ens_adaptive_d5_aux_long
  NF     --aux-noise --film: <exp>/d5/st<N>/ens_adaptive_d5_auxfilm_long
Static MWPM, WindowDEM, oracle MWPM and the phase masks come from results/exp3/d5, test-set MWPM bits from
results/exp4/curves/baselines.npz (as report 6). Single-seed numbers are the ensemble files' members.
Noise-map quality comes from <exp>/noise/noise_<aux|auxfilm>_s<seed>.json (noise_eval.py).
Streams are pooled over stream seeds. Paired tests are exact McNemar on shared shots.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, ".")
from report import mcnemar_p  # noqa: E402

PHASES = ["drift", "onset", "later", "after", "nominal"]
WINDOWS = [500, 1000, 2000, 5000, 10000]
TESTS = ["nominal", "scale x2", "2 hot qubits"]
MEMBERS = ["m0", "m1", "m2"]
VARIANTS = {"base": ("adaptive-stale (baseline)", "results/exp6/d5/st{st}/ens_adaptive_stale_d5_long"),
            "sw": ("static-wide", "results/exp6/d5/st{st}/ens_static_wide_d5_long"),
            "N": ("N noise-map head", "{exp}/d5/st{st}/ens_adaptive_d5_aux_long"),
            "NF": ("NF noise-map head + FiLM", "{exp}/d5/st{st}/ens_adaptive_d5_auxfilm_long")}
NEW = ("N", "NF")
NOISE = {"N": "aux", "NF": "auxfilm"}
METRICS = [("topk", "top-k hot hit rate", 3), ("mae_hot", "MAE p_hot, hot qubits", 4),
           ("mae_cold", "MAE p_hot, non-hot qubits", 5), ("scale_mae", "MAE scale", 3),
           ("mean_p", "mean predicted p_hot (all qubits)", 5)]
BASE = "results/exp3/d5/{r}_d5_r5_u200_st{st}_s0_errors.npz"
P_RULE = 0.001


def unpack(a):
    return np.unpackbits(a, axis=-1).astype(bool)


def load(exp, seeds, regimes):
    """bits[key][sub]: key = (baseline,) or (variant, live|frozen|test, ens|m0|m1|m2);
    sub = (regime, phase) or ("test", name). Variants without a file for every stream seed are left out."""
    found = [v for v, (_, p) in VARIANTS.items()
             if all(os.path.exists(p.format(exp=exp, st=st) + ".json") for st in seeds)]
    bits = {}
    for r in regimes:
        base = [np.load(BASE.format(r=r, st=st)) for st in seeds]
        masks = [{ph: b[f"phase_{ph}"] for ph in PHASES} for b in base]

        def add(key, arrays):
            for ph in PHASES:
                bits.setdefault(key, {})[r, ph] = np.concatenate([a[m[ph]].ravel() for a, m in zip(arrays, masks)])

        for k in ["static", "oracle"] + [f"window_{w}" for w in WINDOWS]:
            add((k,), [unpack(b[k]) for b in base])
        for v in found:
            zs = [np.load(VARIANTS[v][1].format(exp=exp, st=st) + "_errors.npz") for st in seeds]
            for k in zs[0].files:
                rr, kind, who = k.split("|")
                if rr == r:
                    add((v, kind, who), [unpack(z[k]) for z in zs])
    tb = np.load("results/exp4/curves/baselines.npz")
    for t in TESTS:
        for lab in ("static", "oracle"):
            bits.setdefault((lab,), {})["test", t] = tb[f"{t}|{lab}"].astype(bool)
    if 0 in seeds:
        for v in found:
            z = np.load(VARIANTS[v][1].format(exp=exp, st=0) + "_errors.npz")
            for k in z.files:
                if k.startswith("test|"):
                    _, tn, who = k.split("|")
                    n = len(bits["static",]["test", tn])
                    bits.setdefault((v, "test", who), {})["test", tn] = np.unpackbits(z[k])[:n].astype(bool)
    return bits, found


def pair(a, b):
    """(LER(a) − LER(b), exact McNemar p) on shared shots."""
    na, nb = int((a & ~b).sum()), int((~a & b).sum())
    return (na - nb) / len(a), mcnemar_p(na, nb)


def fmt(d, p, nd=4):
    s = f"{d:+.{nd}f} (p={p:.1g})" if p > 0 else f"{d:+.{nd}f} (p<1e-300)"
    return f"**{s}**" if p < 0.05 else s


def best_w(bits, sub):
    return min(WINDOWS, key=lambda w: bits[f"window_{w}",][sub].mean())


def ler(bits, key, sub):
    b = bits.get(key, {}).get(sub)
    return None if b is None or len(b) == 0 else float(b.mean())


def name(v, kind):
    return VARIANTS[v][0] + (" (frozen)" if kind == "frozen" else "")


def decision(bits, found, regimes):
    """The preset rule, per variant: (passes, win, losses)."""
    out = {}
    for v in [x for x in NEW if x in found and "base" in found]:
        wins, losses = [], []
        if "bursts_big" in regimes:
            d, p = pair(bits[v, "live", "ens"]["bursts_big", "drift"], bits["base", "live", "ens"]["bursts_big", "drift"])
            if d < 0 and p < P_RULE:
                wins.append(f"bursts_big drift {fmt(d, p).strip('*')}")
        for r in regimes:
            for ph in ("drift", "later"):
                a, b = bits[v, "live", "ens"][r, ph], bits["base", "live", "ens"][r, ph]
                if len(a) == 0:
                    continue
                d, p = pair(a, b)
                if d > 0 and p < P_RULE:
                    losses.append(f"{r} {ph} {fmt(d, p).strip('*')}")
        tk = ("test", "nominal")
        if (v, "test", "ens") in bits and tk in bits[v, "test", "ens"]:
            d, p = pair(bits[v, "test", "ens"][tk], bits["base", "test", "ens"][tk])
            if d > 0 and p < P_RULE:
                losses.append(f"nominal test set {fmt(d, p, 5).strip('*')}")
        else:
            losses.append("nominal test set not evaluated")
        out[v] = {"passes": bool(wins) and not losses, "wins": wins, "losses": losses}
    return out


def regressions(bits, found, regimes, data):
    """Every regime x phase (and test set) where an N / NF ensemble is worse than the baseline ensemble (live),
    any p; for information only, the verdict above is not changed by it."""
    L = ["", "### Size of every regression vs the adaptive-stale ensemble (information only; does not change "
         "the verdict)\n", "Live ensembles, pooled streams (test sets: own fingerprint window), every phase where the "
         "variant's LER is higher, any p. Relative = (LER(variant) − LER(base)) / LER(base).\n",
         "| variant | regime | phase | LER base | LER variant | diff | relative | p | counts for the rule |",
         "|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for v in [x for x in NEW if x in found and "base" in found]:
        subs = [(r, ph) for r in regimes for ph in PHASES] + [("test", t) for t in TESTS]
        for sub in subs:
            kind = "test" if sub[0] == "test" else "live"
            if (v, kind, "ens") not in bits or sub not in bits[v, kind, "ens"]:
                continue
            a, b = bits[v, kind, "ens"][sub], bits["base", kind, "ens"][sub]
            if len(a) == 0:
                continue
            d, p = pair(a, b)
            if d > 0:
                rel = d / b.mean()
                counts = "yes" if p < P_RULE and (sub[1] in ("later", "drift") or sub == ("test", "nominal")) else "no"
                rows.append(f"| {v} | {sub[0]} | {sub[1]} | {b.mean():.5f} | {a.mean():.5f} | {d:+.5f} | {100 * rel:+.1f}% | "
                            f"{p:.1g} | {counts} |".replace("| 0 |", "| <1e-300 |"))
                data.setdefault("regressions", []).append({"variant": v, "regime": sub[0], "phase": sub[1],
                                                           "base": float(b.mean()), "variant_ler": float(a.mean()),
                                                           "diff": d, "rel": rel, "p": p})
    return L + (rows or ["| none | – | – | – | – | – | – | – | – |"])


def noise_section(exp, seeds, regimes, data):
    """Noise-map quality tables (noise_eval.py), per variant x seed, per regime x phase."""
    L = ["## Noise-map quality (the diagnostic claim; live fingerprint, pooled over stream seeds)\n",
         "Per step: predicted p_hot of each qubit = mean over the step's shots of the noise head / 10; predicted "
         "scale = exp(mean predicted log scale). top-k: fraction of truly hot qubits among the k highest predicted "
         "(k = number of hot qubits; steps with hot qubits only). MAE p_hot on hot qubits / non-hot qubits; MAE of "
         "the predicted scale; mean predicted p_hot over all qubits (on nominal and after steps = false-alarm "
         "level). bursts_big is held out (8 qubits, p_hot 0.1; training had 1-6 qubits, p_hot 0.005-0.08). Cells: "
         "seed mean ± std, then the seeds' values.\n"]
    nd = data.setdefault("noise_map", {})
    for r in regimes:
        L += [f"### {r}\n", "| variant | phase | " + " | ".join(m[1] for m in METRICS) + " |",
              "|---|---|" + "---|" * len(METRICS)]
        for v in NEW:
            js = {s: f"{exp}/noise/noise_{NOISE[v]}_s{s}.json" for s in seeds}
            js = {s: json.load(open(f))["metrics"] for s, f in js.items() if os.path.exists(f)}
            if not js:
                L.append(f"| {VARIANTS[v][0]} | – | " + " | ".join("missing" for _ in METRICS) + " |")
                continue
            for ph in PHASES:
                cells = []
                for key, _, nd_ in METRICS:
                    vals = [(js[s].get(r) or {}).get(ph) or {} for s in js]
                    vals = [x.get(key) for x in vals]
                    nd.setdefault(f"{v}|{r}|{ph}|{key}", dict(zip([f"s{s}" for s in js], vals)))
                    if any(x is None for x in vals):
                        cells.append("–")
                        continue
                    sd = f" ± {np.std(vals, ddof=1):.{nd_}f}" if len(vals) > 1 else ""
                    cells.append(f"{np.mean(vals):.{nd_}f}{sd} ({', '.join(f'{x:.{nd_}f}' for x in vals)})")
                L.append(f"| {VARIANTS[v][0]} (seeds {','.join(map(str, js))}) | {ph} | " + " | ".join(cells) + " |")
        L.append("")
    return L


def test_section(bits, found, data):
    L = ["## Test sets (200000 shots each; adaptive models use the test set's own fingerprint window)\n",
         "| decoder | " + " | ".join(TESTS) + " |", "|---|" + "---|" * 3]
    for lab in ("static", "oracle"):
        L.append(f"| {lab} MWPM | " + " | ".join(f"{ler(bits, (lab,), ('test', t)):.5f}" for t in TESTS) + " |")
    for v in found:
        if (v, "test", "ens") not in bits:
            continue
        ens = [ler(bits, (v, "test", "ens"), ("test", t)) for t in TESTS]
        seeds = [[x for x in (ler(bits, (v, "test", m), ("test", t)) for m in MEMBERS) if x is not None] for t in TESTS]
        L.append(f"| {name(v, 'live')}: ensemble | " + " | ".join(f"{e:.5f}" for e in ens) + " |")
        L.append(f"| {name(v, 'live')}: seed mean ± std | " + " | ".join(
            f"{np.mean(s):.5f} ± {np.std(s, ddof=1):.5f}" for s in seeds) + " |")
        for t, e, s in zip(TESTS, ens, seeds):
            data["ler"][f"{v}|test|{t}"] = {"ens": e, "seeds": s}
    L += ["", "Paired, ensembles (diff = LER(a) − LER(b), negative = a better; bold = p < 0.05):\n",
          "| a vs b | " + " | ".join(TESTS) + " |", "|---|" + "---|" * 3]
    for a, b in comparisons(found):
        ka = (a, "test", "ens")
        kb = (b,) if b in ("static", "oracle") else (b, "test", "ens")
        if ka not in bits or kb not in bits:
            continue
        cells = []
        for t in TESTS:
            d, p = pair(bits[ka]["test", t], bits[kb]["test", t])
            data["paired"][f"test|{t}|{a} vs {b}"] = [d, p]
            cells.append(fmt(d, p, 5))
        L.append(f"| {label(a)} vs {label(b)} | " + " | ".join(cells) + " |")
    return L + [""]


def label(x):
    return {"static": "static MWPM", "oracle": "oracle MWPM", "window": "best WindowDEM"}.get(x, VARIANTS.get(x, (x,))[0])


def comparisons(found, window=False):
    out = [(a, b) for a, b in (("N", "base"), ("NF", "base"), ("NF", "N"), ("N", "sw"), ("NF", "sw"), ("base", "sw"))
           if a in found and b in found]
    for v in [x for x in ("base", "N", "NF", "sw") if x in found]:
        out += [(v, "static")] + ([(v, "window")] if window else []) + [(v, "oracle")]
    return out


def stream_section(bits, found, r, data):
    L = [f"## {r} (pooled over stream seeds)\n", "| decoder | " + " | ".join(PHASES) + " |",
         "|---|" + "---|" * len(PHASES)]
    for lab in ("static", "oracle"):
        L.append(f"| {lab} MWPM | " + " | ".join(
            "–" if ler(bits, (lab,), (r, ph)) is None else f"{ler(bits, (lab,), (r, ph)):.4f}" for ph in PHASES) + " |")
    L.append("| best WindowDEM (picked per phase) | " + " | ".join(
        f"{ler(bits, (f'window_{best_w(bits, (r, ph))}',), (r, ph)):.4f} (w={best_w(bits, (r, ph))})"
        if ler(bits, ("static",), (r, ph)) is not None else "–" for ph in PHASES) + " |")
    for kind in ("live", "frozen"):
        for v in found:
            if (v, kind, "ens") not in bits:
                continue
            ens, ms = [], []
            for ph in PHASES:
                e = ler(bits, (v, kind, "ens"), (r, ph))
                s = [x for x in (ler(bits, (v, kind, m), (r, ph)) for m in MEMBERS) if x is not None]
                if e is None:
                    ens.append("–")
                    ms.append("–")
                    continue
                ens.append(f"{e:.4f}")
                ms.append(f"{np.mean(s):.4f} ± {np.std(s, ddof=1):.4f}")
                data["ler"][f"{v}|{kind}|{r}|{ph}"] = {"ens": e, "seeds": s}
            L.append(f"| {name(v, kind)}: ensemble | " + " | ".join(ens) + " |")
            L.append(f"| {name(v, kind)}: seed mean ± std | " + " | ".join(ms) + " |")
    L += ["", f"Paired ({r}, live ensembles; diff = LER(a) − LER(b), negative = a better; bold = p < 0.05):\n",
          "| a vs b | " + " | ".join(PHASES) + " |", "|---|" + "---|" * len(PHASES)]
    for a, b in comparisons(found, window=True):
        cells = []
        for ph in PHASES:
            ea = bits[a, "live", "ens"][r, ph]
            if len(ea) == 0:
                cells.append("–")
                continue
            kb = (b,) if b in ("static", "oracle") else (f"window_{best_w(bits, (r, ph))}",) if b == "window" \
                else (b, "live", "ens")
            d, p = pair(ea, bits[kb][r, ph])
            data["paired"][f"{r}|{ph}|{a} vs {b}"] = [d, p]
            cells.append(fmt(d, p))
        L.append(f"| {label(a)} vs {label(b)} | " + " | ".join(cells) + " |")
    for v in found:
        if (v, "frozen", "ens") in bits:
            cells = []
            for ph in PHASES:
                a, b = bits[v, "live", "ens"][r, ph], bits[v, "frozen", "ens"][r, ph]
                if len(a) == 0:
                    cells.append("–")
                    continue
                d, p = pair(a, b)
                data["paired"][f"{r}|{ph}|{v} live vs frozen"] = [d, p]
                cells.append(fmt(d, p))
            L.append(f"| {label(v)}: ensemble live vs ensemble frozen | " + " | ".join(cells) + " |")
    return L + [""]


def runtime(exp):
    L = ["## Run time\n"]
    for f in sorted(glob.glob(f"{exp}/train_*.log")):
        t = re.findall(r"^step +300000 .* (\d+)s$", open(f).read(), re.M)
        if t:
            L.append(f"- {os.path.basename(f)}: 300000 steps in {int(t[-1])} s (training time, summed over resumes)")
    secs = [json.load(open(p))["seconds"] for p in glob.glob(f"{exp}/noise/noise_*.json")]
    if secs:
        L.append(f"- {len(secs)} noise-map evals, {sum(secs) / 60:.1f} min in total")
    secs = [json.load(open(p))["seconds"] for p in glob.glob(f"{exp}/d5/st*/ens_*.json")]
    if secs:
        L.append(f"- {len(secs)} ensemble evals, {sum(secs) / 60:.1f} min in total ({min(secs):.0f}–{max(secs):.0f} s each)")
    return L + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="results/exp9")
    ap.add_argument("--seeds", default="0,1,2", help="model seeds for the noise-map tables")
    ap.add_argument("--stream-seeds", default="0,1,2,3,4")
    ap.add_argument("--regimes", default="bursts,bursts_big,hot_qubit,telegraph")
    args = ap.parse_args()
    seeds = [int(s) for s in args.stream_seeds.split(",")]
    regimes = args.regimes.split(",")
    bits, found = load(args.exp, seeds, regimes)
    data = {"stream_seeds": seeds, "regimes": regimes, "found": found, "ler": {}, "paired": {}}
    dec = decision(bits, found, regimes)
    data["decision"] = dec
    members = {v: json.load(open(VARIANTS[v][1].format(exp=args.exp, st=seeds[0]) + ".json"))["checkpoints"]
               for v in NEW if v in found}
    data["members"] = members
    out = ["# Report 9 tables (generated by results/exp9/report9.py)\n",
           "Variants, all d=5, 5 rounds, p=0.003, data/d5_wide_x5, h48/l4, fp32, batch 512, 300000 steps, --compile, "
           "--stale-prob 0.3, single fingerprint. Baseline = the existing adaptive-stale models (seeds 0, 1, 2). "
           "N = the same command plus --aux-noise (noise-map head on the last feature map + log(scale) head on the "
           "pooled vector; loss = BCE + 0.1 MSE(p_hot x 10) + 0.1 MSE(log scale), auxiliary losses masked on stale "
           "steps; the logit path is unchanged). NF = N plus --film (fingerprint encoder 2 x Conv3d 16 ch, "
           "zero-initialised per-layer gamma/beta, h <- h (1 + gamma) + beta after each GELU). Ensemble members: "
           + "; ".join(f"{VARIANTS[v][0]}: {', '.join(os.path.basename(c) for c in cs)}" for v, cs in members.items())
           + ". static-wide = the existing static models trained on data/d5_wide_x5. Baseline and "
           "static-wide results are reused from results/exp6. Ensemble = mean of the 3 seeds' logits > 0 (fixed). "
           "Live: each model's own causal buffer; frozen: every channel from the nominal test set's fingerprint "
           f"shots. Update every 200 shots. Streams pooled over stream seeds {args.stream_seeds}. McNemar exact, "
           "two-sided; p in 0.001–0.05 = weak evidence; no multiple-testing correction. The best WindowDEM is "
           "picked per phase after the fact.\n",
           "## Decision rule (fixed in advance, computed here)\n",
           f"A variant passes if its ensemble beats the adaptive-stale ensemble on bursts_big drift steps with "
           f"p < {P_RULE}, and is not worse than it on drift or later steps in any regime ({', '.join(regimes)}) at "
           f"p < {P_RULE}, and is not worse on the nominal test set at p < {P_RULE}. Live ensembles, pooled streams.\n",
           "| variant | verdict | win (bursts_big drift, p < 0.001) | losses (drift/later any regime, nominal test set; p < 0.001) |",
           "|---|---|---|---|"]
    for v in NEW:
        if v in dec:
            d = dec[v]
            out.append(f"| {VARIANTS[v][0]} | **{'PASSES' if d['passes'] else 'FAILS'}** | "
                       f"{'; '.join(d['wins']) or 'none'} | {'; '.join(d['losses']) or 'none'} |")
        else:
            out.append(f"| {VARIANTS[v][0]} | not evaluated (missing results) | – | – |")
    out += regressions(bits, found, regimes, data)
    out.append("")
    if 0 in seeds:
        out += test_section(bits, found, data)
    for r in regimes:
        out += stream_section(bits, found, r, data)
    out += noise_section(args.exp, [int(s) for s in args.seeds.split(",")], regimes, data)
    out += runtime(args.exp)
    open(f"{args.exp}/report9_tables.md", "w").write("\n".join(out) + "\n")
    json.dump(data, open(f"{args.exp}/report9_data.json", "w"), indent=1)
    print(f"wrote {args.exp}/report9_tables.md; decision: " +
          ", ".join(f"{v} {'passes' if d['passes'] else 'fails'}" for v, d in dec.items()))


if __name__ == "__main__":
    main()
