"""Tables for report 6: seed ensembles (mean logit of model seeds 0, 1, 2 > 0), d=5 and d=3.

    python results/exp6/report6.py      # writes results/exp6/report6_tables.md and report6_data.json

Reads only existing files. Ensembles: results/exp6/d{5,3}/st<N>/ens_<type>_errors.npz from ensemble_eval.py.
Their members m0, m1, m2 are the single seeds 0, 1, 2 from the same run (checked below against the saved
single-model bits of results/exp4 and results/exp2). Static MWPM, WindowDEM, oracle MWPM and the phase masks are reused:
d=5 from results/exp3/d5 and results/exp4/curves/baselines.npz, d=3 from results/exp2.
Streams are pooled over stream seeds. Paired tests are exact McNemar on shared shots.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from report import mcnemar_p  # noqa: E402

PHASES = ["drift", "onset", "later", "after", "nominal"]
WINDOWS = [500, 1000, 2000, 5000, 10000]
TESTS = ["nominal", "scale x2", "2 hot qubits"]
TYPES = {
    5: {"static_wide_d5_long": "static-wide", "adaptive_wide_d5_long": "adaptive-wide",
        "adaptive_stale_d5_long": "adaptive-stale", "static_nominal_d5_long": "static-nominal"},
    3: {"static_d3_wide": "static-wide", "adaptive_d3_wide": "adaptive-wide",
        "adaptive_d3_stale": "adaptive-stale", "static_d3_nominal": "static-nominal"},
}
BASE = {5: "results/exp3/d5/{r}_d5_r5_u200_st{st}_s0_errors.npz", 3: "results/exp2/{r}_d3_r3_u200_st{st}_s0_errors.npz"}
MEMBERS = ["m0", "m1", "m2"]


def unpack(a):
    return np.unpackbits(a, axis=-1).astype(bool)


def load(exp, d, seeds, regimes):
    """bits[key][sub]: key = (baseline,) or (type, live|frozen|test, ens|m0|m1|m2); sub = (regime, phase) or ("test", name).

    Stream phases are pooled over the stream seeds. Only types with a file for every stream seed are used.
    """
    found = [t for t in TYPES[d] if all(os.path.exists(f"{exp}/d{d}/st{st}/ens_{t}.json") for st in seeds)]
    bits = {}
    for r in regimes:
        base = [np.load(BASE[d].format(r=r, st=st)) for st in seeds]
        masks = [{ph: b[f"phase_{ph}"] for ph in PHASES} for b in base]

        def add(key, arrays):
            for ph in PHASES:
                bits.setdefault(key, {})[r, ph] = np.concatenate([a[m[ph]].ravel() for a, m in zip(arrays, masks)])

        for k in ["static", "oracle"] + [f"window_{w}" for w in WINDOWS]:
            add((k,), [unpack(b[k]) for b in base])
        zs = {t: [np.load(f"{exp}/d{d}/st{st}/ens_{t}_errors.npz") for st in seeds] for t in found}
        for t, z in zs.items():
            for k in z[0].files:
                rr, kind, who = k.split("|")
                if rr == r:
                    add((t, kind, who), [unpack(f[k]) for f in z])
    # test sets (from the stream-seed-0 file, if it has them)
    if d == 5:
        tb = np.load("results/exp4/curves/baselines.npz")
        for t in TESTS:
            for lab in ("static", "oracle"):
                bits.setdefault((lab,), {})["test", t] = tb[f"{t}|{lab}"]
    else:
        z = np.load("results/exp2/test_errors.npz")
        for t in TESTS:
            for lab, k in (("static", "static MWPM"), ("oracle", "oracle MWPM")):
                bits.setdefault((lab,), {})["test", t] = np.unpackbits(z[f"{k}|0|{t}"], count=int(z["_shots"])).astype(bool)
    for t in found:
        p = f"{exp}/d{d}/st0/ens_{t}_errors.npz"
        if 0 in seeds and os.path.exists(p):
            z = np.load(p)
            for k in z.files:
                if k.startswith("test|"):
                    _, tn, who = k.split("|")
                    n = len(bits["static",]["test", tn])
                    bits.setdefault((t, "test", who), {})["test", tn] = np.unpackbits(z[k])[:n].astype(bool)
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


def summary_cells(bits, t, kind, sub, nd):
    """ensemble, seed mean ± std, best seed, worst seed (as strings) and the raw numbers."""
    e = ler(bits, (t, kind, "ens"), sub)
    s = [ler(bits, (t, kind, m), sub) for m in MEMBERS]
    if e is None or None in s:
        return ["–"] * 4, None
    f = f"{{:.{nd}f}}"
    return ([f.format(e), f"{f.format(np.mean(s))} ± {f.format(np.std(s, ddof=1))}", f.format(min(s)), f.format(max(s))],
            {"ens": e, "seeds": s, "mean": float(np.mean(s)), "best": min(s), "worst": max(s)})


def kinds(bits, t):
    return [k for k in ("live", "frozen") if (t, k, "ens") in bits]


def section(bits, d, found, regimes, data):
    names = TYPES[d]
    L = [f"## d={d}\n"]
    if not found:
        return L + ["No ensemble results found.\n"]
    # ---- test sets
    if any((t, "test", "ens") in bits for t in found):
        L += [f"### d={d} test sets (200000 shots each)\n",
              "| decoder | " + " | ".join(TESTS) + " |", "|---|" + "---|" * 3]
        for lab in ("static", "oracle"):
            L.append(f"| {lab} MWPM | " + " | ".join(f"{ler(bits, (lab,), ('test', t)):.5f}" for t in TESTS) + " |")
        for t in found:
            cells = [summary_cells(bits, t, "test", ("test", tn), 5) for tn in TESTS]
            for i, row in enumerate(("ensemble", "seed mean ± std", "best seed", "worst seed")):
                L.append(f"| {names[t]}: {row} | " + " | ".join(c[0][i] for c in cells) + " |")
            for tn, c in zip(TESTS, cells):
                data.setdefault(f"d{d}", {}).setdefault(t, {})[f"test|{tn}"] = c[1]
        L += ["", "Paired (diff = LER(ensemble) − LER(b), negative = ensemble better):\n",
              "| ensemble vs b | " + " | ".join(TESTS) + " |", "|---|" + "---|" * 3]
        for t in found:
            ens = bits[t, "test", "ens"]
            for b, lab in [((t, "test", m), f"seed {m[1]}") for m in MEMBERS] + [(("static",), "static MWPM"),
                                                                                 (("oracle",), "oracle MWPM")]:
                L.append(f"| {names[t]} vs {lab} | " + " | ".join(fmt(*pair(ens["test", tn], bits[b]["test", tn]), 5)
                                                                  for tn in TESTS) + " |")
        a, b = [k for k in found if k.startswith("adaptive_d3_stale") or k.startswith("adaptive_stale")], \
               [k for k in found if k.startswith("static_d3_wide") or k.startswith("static_wide")]
        if a and b and (a[0], "test", "ens") in bits and (b[0], "test", "ens") in bits:
            L.append("| ensemble adaptive-stale vs ensemble static-wide | " + " | ".join(
                fmt(*pair(bits[a[0], "test", "ens"]["test", tn], bits[b[0], "test", "ens"]["test", tn]), 5)
                for tn in TESTS) + " |")
        L.append("")
    # ---- streams
    for r in regimes:
        L += [f"### d={d} {r} (pooled over stream seeds)\n", "| decoder | " + " | ".join(PHASES) + " |",
              "|---|" + "---|" * len(PHASES)]
        for lab in ("static", "oracle"):
            L.append(f"| {lab} MWPM | " + " | ".join(f"{ler(bits, (lab,), (r, ph)) or float('nan'):.4f}" for ph in PHASES) + " |")
        L.append("| best WindowDEM (picked per phase) | " + " | ".join(
            f"{ler(bits, (f'window_{best_w(bits, (r, ph))}',), (r, ph)):.4f} (w={best_w(bits, (r, ph))})"
            if ler(bits, ("static",), (r, ph)) is not None else "–" for ph in PHASES) + " |")
        for t in found:
            for kind in kinds(bits, t):
                cells = [summary_cells(bits, t, kind, (r, ph), 4) for ph in PHASES]
                tag = f"{names[t]}{' (frozen)' if kind == 'frozen' else ''}"
                for i, row in enumerate(("ensemble", "seed mean ± std", "best seed", "worst seed")):
                    L.append(f"| {tag}: {row} | " + " | ".join(c[0][i] for c in cells) + " |")
                for ph, c in zip(PHASES, cells):
                    data.setdefault(f"d{d}", {}).setdefault(t, {})[f"{r}|{kind}|{ph}"] = c[1]
        L += ["", f"Paired ({r}; diff = LER(ensemble) − LER(b), negative = ensemble better; bold = p < 0.05):\n",
              "| ensemble vs b | " + " | ".join(PHASES) + " |", "|---|" + "---|" * len(PHASES)]
        for t in found:
            for kind in kinds(bits, t):
                ens, tag = bits[t, kind, "ens"], f"{names[t]}{' (frozen)' if kind == 'frozen' else ''}"
                refs = [((t, kind, m), f"seed {m[1]}") for m in MEMBERS]
                refs += [(("static",), "static MWPM"), (None, "best WindowDEM"), (("oracle",), "oracle MWPM")]
                for b, lab in refs:
                    cells = []
                    for ph in PHASES:
                        if len(ens[r, ph]) == 0:
                            cells.append("–")
                            continue
                        bb = b or (f"window_{best_w(bits, (r, ph))}",)
                        dd, p = pair(ens[r, ph], bits[bb][r, ph])
                        cells.append(fmt(dd, p))
                        data.setdefault("paired", {})[f"d{d}|{t}|{kind}|{r}|{ph}|vs {lab}"] = [dd, p]
                    L.append(f"| {tag} vs {lab} | " + " | ".join(cells) + " |")
        for t in found:
            if "frozen" in kinds(bits, t):
                L.append(f"| {names[t]}: ensemble live vs ensemble frozen | " + " | ".join(
                    fmt(*pair(bits[t, "live", "ens"][r, ph], bits[t, "frozen", "ens"][r, ph]))
                    if len(bits[t, "live", "ens"][r, ph]) else "–" for ph in PHASES) + " |")
        st = [t for t in found if "stale" in t]
        sw = [t for t in found if t in ("static_wide_d5_long", "static_d3_wide")]
        if st and sw:
            L.append("| ensemble adaptive-stale vs ensemble static-wide | " + " | ".join(
                fmt(*pair(bits[st[0], "live", "ens"][r, ph], bits[sw[0], "live", "ens"][r, ph]))
                if len(bits[st[0], "live", "ens"][r, ph]) else "–" for ph in PHASES) + " |")
        L.append("")
    # ---- compact: ensemble vs seeds on drift steps
    L += [f"### d={d} summary: drift steps, ensemble vs single seeds\n",
          "(ens − seed mean; (k/3) = seeds the ensemble beats; p = largest McNemar p over the 3 seeds)\n",
          "| model | " + " | ".join(regimes) + " |", "|---|" + "---|" * len(regimes)]
    for t in found:
        for kind in kinds(bits, t):
            cells = []
            for r in regimes:
                e = bits[t, kind, "ens"][r, "drift"]
                res = [pair(e, bits[t, kind, m][r, "drift"]) for m in MEMBERS]
                mean = np.mean([bits[t, kind, m][r, "drift"].mean() for m in MEMBERS])
                k, p = sum(x[0] < 0 for x in res), max(x[1] for x in res)
                cells.append(f"{e.mean() - mean:+.4f} ({k}/3, p≤{p:.1g})")
            L.append(f"| {names[t]}{' (frozen)' if kind == 'frozen' else ''} | " + " | ".join(cells) + " |")
    L.append("")
    return L


def checks(exp, seeds):
    """Members m0-m2 vs saved single-model bits: d=5 results/exp4/full, d=3 results/exp2."""
    L = ["## Members vs saved single-seed bits\n", "| d | model | differing shots / compared |", "|---|---|---|"]
    lab3 = {"static_d3_wide": "static-wide", "adaptive_d3_wide": "adaptive-wide", "adaptive_d3_stale": "adaptive-stale",
            "static_d3_nominal": "static-nominal"}
    for d in (5, 3):
        for t in TYPES[d]:
            diff = n = 0
            for st in seeds:
                p = f"{exp}/d{d}/st{st}/ens_{t}_errors.npz"
                if not os.path.exists(p):
                    continue
                z = np.load(p)
                for k in z.files:
                    a, kind, who = k.split("|")
                    if who == "ens":
                        continue
                    s = int(who[1:])
                    if d == 5:
                        ref = np.load(f"results/exp4/full/st{st}/decoder_{t.replace('_d5_long', '')}_d5_long_s{s}_errors.npz")
                        rb = ref[f"test|{kind}"] if a == "test" else unpack(ref[f"{a}|{kind}"])
                    elif a == "test":
                        tz = np.load("results/exp2/test_errors.npz")
                        rb = np.unpackbits(tz[f"{lab3[t]}|{s}|{kind}"], count=int(tz["_shots"])).astype(bool)
                    else:
                        rz = np.load(f"results/exp2/{a}_d3_r3_u200_st{st}_s{s}_errors.npz")
                        rb = unpack(rz[f"decoder_{t}_s{s}" + ("_frozen" if kind == "frozen" else "")])
                    nb = unpack(z[k])[..., : rb.shape[-1]]
                    diff += int((nb != rb).sum())
                    n += rb.size
            if n:
                L.append(f"| {d} | {TYPES[d][t]} | {diff} / {n} |")
    return L + ["", "Expected 0: d=5 references are results/exp4/full (this VM), d=3 references are results/exp2 "
                "(report 2). All comparisons in this report use the members from this run.\n"]


def runtime(exp):
    L = ["## Run time\n"]
    secs = [json.load(open(p))["seconds"] for p in glob.glob(f"{exp}/d*/st*/ens_*.json")]
    if secs:
        L.append(f"{len(secs)} ensemble evals, {sum(secs) / 60:.1f} min of eval time in total "
                 f"({min(secs):.0f}–{max(secs):.0f} s each).\n")
    path = f"{exp}/gpu_util.csv"
    if os.path.exists(path):
        u = [float(x.split(",")[1].split()[0]) for x in open(path).readlines()[1:] if "run6" in x]
        if u:
            L.append(f"GPU utilisation during run6.sh ({len(u)} samples every 30 s): mean {np.mean(u):.0f}%, "
                     f"{100 * np.mean(np.array(u) >= 85):.0f}% of samples ≥ 85%.\n")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="results/exp6")
    ap.add_argument("--stream-seeds", default="0,1,2,3,4")
    ap.add_argument("--regimes", default="bursts,bursts_big,hot_qubit,telegraph")
    args = ap.parse_args()
    seeds = [int(s) for s in args.stream_seeds.split(",")]
    regimes = args.regimes.split(",")
    out = ["# Report 6 tables (generated by results/exp6/report6.py)\n",
           "Ensemble = mean of the logits of model seeds 0, 1, 2; flip if > 0 (rule fixed in advance). "
           "Single seeds are the ensemble members from the same run. Adaptive members each keep their own "
           "fingerprint buffer (live) or all get the frozen nominal fingerprint (frozen). Update every 200 shots. "
           f"Streams pooled over stream seeds {args.stream_seeds}. McNemar exact, two-sided; p in 0.001–0.05 = "
           "weak evidence; no multiple-testing correction. The best WindowDEM is picked per phase after the fact.\n"]
    data = {"stream_seeds": seeds, "regimes": regimes}
    for d in (5, 3):
        bits, found = load(args.exp, d, seeds, regimes)
        data[f"found_d{d}"] = found
        out += section(bits, d, found, regimes, data)
    out += checks(args.exp, seeds) + runtime(args.exp)
    open(f"{args.exp}/report6_tables.md", "w").write("\n".join(out) + "\n")
    json.dump(data, open(f"{args.exp}/report6_data.json", "w"), indent=1)
    print(f"wrote {args.exp}/report6_tables.md")


if __name__ == "__main__":
    main()
