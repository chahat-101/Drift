"""Tables for report 8: the multiscale fingerprint (variant B of round 7) at d=5.

    python results/exp8/report8.py     # writes results/exp8/report8_tables.md and report8_data.json

Reads only existing files. Variants (3 model seeds each, ensemble = mean logit > 0):
  base   adaptive-stale d=5 300k, single fingerprint: results/exp6/d5/st<N>/ens_adaptive_stale_d5_long (reused)
  sw     static-wide d=5 300k: results/exp6/d5/st<N>/ens_static_wide_d5_long (reused)
  B      multiscale d=5 300k: <exp>/d5/st<N>/ens_adaptive_d5_ms_long
Static MWPM, WindowDEM, oracle MWPM and the phase masks come from results/exp3/d5, test-set MWPM bits from
results/exp4/curves/baselines.npz (as report 6). Single-seed numbers are the ensemble files' members m0-m2.
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
            "B": ("B multiscale", "{exp}/d5/st{st}/ens_adaptive_d5_ms_long")}
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
    """The preset rule, per variant: (passes, list of reasons)."""
    out = {}
    for v in [x for x in ("B",) if x in found and "base" in found]:
        wins, losses = [], []
        for r in regimes:
            for ph in PHASES:
                a, b = bits[v, "live", "ens"][r, ph], bits["base", "live", "ens"][r, ph]
                if len(a) == 0:
                    continue
                d, p = pair(a, b)
                if r in ("bursts", "bursts_big") and ph in ("onset", "after") and d < 0 and p < P_RULE:
                    wins.append(f"{r} {ph} {fmt(d, p).strip('*')}")
                if ph in ("later", "drift") and d > 0 and p < P_RULE:
                    losses.append(f"{r} {ph} {fmt(d, p).strip('*')}")
        out[v] = {"passes": bool(wins) and not losses, "wins": wins, "losses": losses}
    return out


def regressions(bits, found, regimes, data):
    """Every regime x phase where the B ensemble is worse than the baseline ensemble (live), any p; for
    information only, the verdict above is not changed by it."""
    L = ["", "### Size of every regression of B vs the adaptive-stale ensemble (information only; does not change "
         "the verdict)\n", "Live ensembles, pooled streams, every phase where B's LER is higher, any p. "
         "Relative = (LER(B) − LER(base)) / LER(base).\n",
         "| regime | phase | LER base | LER B | diff | relative | p | counts for the rule |", "|---|---|---|---|---|---|---|---|"]
    if "B" not in found or "base" not in found:
        return L + ["| – | – | – | – | – | – | – | – |"]
    rows = []
    for r in regimes:
        for ph in PHASES:
            a, b = bits["B", "live", "ens"][r, ph], bits["base", "live", "ens"][r, ph]
            if len(a) == 0:
                continue
            d, p = pair(a, b)
            if d > 0:
                rel = d / b.mean()
                counts = "yes" if ph in ("later", "drift") and p < P_RULE else "no"
                rows.append(f"| {r} | {ph} | {b.mean():.4f} | {a.mean():.4f} | {d:+.4f} | {100 * rel:+.1f}% | "
                            f"{p:.1g} | {counts} |".replace("| 0 |", "| <1e-300 |"))
                data.setdefault("regressions", []).append({"regime": r, "phase": ph, "base": float(b.mean()),
                                                           "B": float(a.mean()), "diff": d, "rel": rel, "p": p})
    return L + (rows or ["| none | – | – | – | – | – | – | – |"])


def test_section(bits, found, data):
    L = ["## Test sets (200000 shots each; adaptive models use the test set's own fingerprint window)\n",
         "| decoder | " + " | ".join(TESTS) + " |", "|---|" + "---|" * 3]
    for lab in ("static", "oracle"):
        L.append(f"| {lab} MWPM | " + " | ".join(f"{ler(bits, (lab,), ('test', t)):.5f}" for t in TESTS) + " |")
    for v in found:
        if (v, "test", "ens") not in bits:
            continue
        ens = [ler(bits, (v, "test", "ens"), ("test", t)) for t in TESTS]
        seeds = [[ler(bits, (v, "test", m), ("test", t)) for m in MEMBERS] for t in TESTS]
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
    out = [(a, b) for a, b in (("B", "base"), ("B", "sw"), ("base", "sw")) if a in found and b in found]
    for v in [x for x in ("base", "B", "sw") if x in found]:
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
                s = [ler(bits, (v, kind, m), (r, ph)) for m in MEMBERS]
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
    secs = [json.load(open(p))["seconds"] for p in glob.glob(f"{exp}/d5/st*/ens_*.json")]
    if secs:
        L.append(f"- {len(secs)} ensemble evals, {sum(secs) / 60:.1f} min in total ({min(secs):.0f}–{max(secs):.0f} s each)")
    return L + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="results/exp8")
    ap.add_argument("--stream-seeds", default="0,1,2,3,4")
    ap.add_argument("--regimes", default="bursts,bursts_big,hot_qubit,telegraph")
    args = ap.parse_args()
    seeds = [int(s) for s in args.stream_seeds.split(",")]
    regimes = args.regimes.split(",")
    bits, found = load(args.exp, seeds, regimes)
    data = {"stream_seeds": seeds, "regimes": regimes, "found": found, "ler": {}, "paired": {}}
    dec = decision(bits, found, regimes)
    data["decision"] = dec
    out = ["# Report 8 tables (generated by results/exp8/report8.py)\n",
           "Variants, all d=5, 5 rounds, p=0.003, data/d5_wide_x5, h48/l4, fp32, batch 512, 300000 steps, --compile, "
           "seeds 0, 1, 2. Baseline = the existing adaptive-stale models (--stale-prob 0.3, single fingerprint); "
           "B = the same command with --fingerprint multiscale (long = whole 2000-shot window, short = last 200 "
           "shots, short − long). static-wide = the existing static models trained on data/d5_wide_x5. Baseline and "
           "static-wide results are reused from results/exp6. Ensemble = mean of the 3 seeds' logits > 0 (fixed). "
           "Live: each model's own causal buffer; frozen: every channel from the nominal test set's fingerprint "
           f"shots. Update every 200 shots. Streams pooled over stream seeds {args.stream_seeds}. McNemar exact, "
           "two-sided; p in 0.001–0.05 = weak evidence; no multiple-testing correction. The best WindowDEM is "
           "picked per phase after the fact.\n",
           "## Decision rule (fixed in advance, same as round 7, computed here)\n",
           f"B passes if its ensemble beats the adaptive-stale ensemble at onset or after on bursts or bursts_big "
           f"with p < {P_RULE}, and is not worse than the adaptive-stale ensemble on later or drift steps in any "
           f"regime ({', '.join(regimes)}) at p < {P_RULE}. Live ensembles, pooled streams.\n",
           "| variant | verdict | wins (onset/after, bursts/bursts_big, p < 0.001) | losses (later/drift, any regime, p < 0.001) |",
           "|---|---|---|---|"]
    for v in ("B",):
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
    out += runtime(args.exp)
    open(f"{args.exp}/report8_tables.md", "w").write("\n".join(out) + "\n")
    json.dump(data, open(f"{args.exp}/report8_data.json", "w"), indent=1)
    print(f"wrote {args.exp}/report8_tables.md; decision: " +
          ", ".join(f"{v} {'passes' if d['passes'] else 'fails'}" for v, d in dec.items()))


if __name__ == "__main__":
    main()
