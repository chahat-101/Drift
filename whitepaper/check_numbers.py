"""Map every number in drift_whitepaper.tex to the file and line it came from, and check it.

Each entry below names (a) a fragment of the paper that holds the numbers, (b) the numbers, and (c) one or more
source rows: file, optional section header (the row is searched after it), and a regex for the row. The script
finds the line numbers in both files and checks that every listed number occurs on the listed source row
(as a whole number token, after removing Markdown/LaTeX markup and signs). It writes number_sources.md and exits
non-zero if any check fails.

    python whitepaper/check_numbers.py
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPER = os.path.join(ROOT, "whitepaper", "drift_whitepaper.tex")
F = {
    "N02": "notes/02_d3_stale_fingerprint.md",
    "N03": "notes/03_d3_controls.md",
    "N04": "notes/04_d5_initial.md",
    "N05": "notes/05_d5_long_training.md",
    "N06": "notes/06_d5_three_seeds.md",
    "N07": "notes/07_ensembles.md",
    "N08": "notes/08_d3_fingerprints.md",
    "N09": "notes/09_d5_multiscale.md",
    "R6": "results/exp6/report6_tables.md",
    "R9": "results/exp9/report9_tables.md",
    "LOC": "results/exp9/localisation_baseline.md",
    "TD": "whitepaper/tech_doc_src/drift.tex",
    "R6PY": "results/exp6/report6.py",
    "PT": "whitepaper/paired_tests.md",
    "EXT": "whitepaper/external_sources.md",
    "R4": "results/report4.md",
    "N10H": "notes/10_physics_drift_hypotheses.md",
    "N10D": "notes/10_definitions.md",
    "PN": "physics_noise.py",
    "R10J": "results/exp10/report10_data.json",
    **{f"A{i}": f"whitepaper/audit/out/t{i}.txt" for i in range(1, 11)},  # audit recomputations (audit/README.md)
}

# Round-10 numbers are checked directly against report10_data.json values: each R10J check names the
# JSON path (and printf format) of every number, or 'argmin:path' for the best WindowDEM window (the
# arg-min over the window LERs, {w} iterating 500..10000).
R10 = json.load(open(os.path.join(ROOT, F["R10J"]), encoding="utf-8"))


def r10_value(path):
    v = R10
    for k in path.split("."):
        v = v[k]
    return v


def r10_get(spec):
    """The displayed string for one R10J spec: 'path|fmt' or 'argmin:path'."""
    if spec.startswith("argmin:"):
        pat = spec[len("argmin:"):]
        windows = (500, 1000, 2000, 5000, 10000)
        vals = {w: r10_value(pat.format(w=w)) for w in windows}
        return str(min(vals, key=vals.get))
    path, fmt = spec.rsplit("|", 1)
    return f"{r10_value(path):{fmt}}"

# (paper fragment, [ (numbers, file, section-or-None, row regex), ... ])
E = []


def e(paper, *checks):
    E.append((paper, checks))


# ------------------------------------------------------------------ abstract
e("over the last 2000 shots it has decoded", ("2000", "TD", None, r"seen \(\$W=2000\$\)"))
e("In circuit-level simulation ($p=0.003$)", ("0.003", "TD", None, r"strength \$p_0 = 0.003\$"))
e("reaches a drift-step logical error rate of 0.0457",
  ("0.0457", "N07", None, r"^\| adaptive-stale \| 0.0170 \| 0.0023 \|"),
  ("0.0980 0.0426", "N07", None, r"^Reference LERs on drift"),
  ("0.0896", "N07", None, r"^Reference LERs on drift"))
e("frozen nominal fingerprint the rate is 0.0576",
  ("0.0576", "R9", "## bursts_big", r"^\| adaptive-stale \(baseline\) \(frozen\): ensemble"),
  ("3e-247", "N07", ("## 3. Ensemble live", "### d=5$"), r"^\| adaptive-stale \| bursts_big \| -0.0119 \(3e-247\)"),
  ("0.0564", "R6", "### d=5 bursts_big", r"^\| static-wide: ensemble"))

# ------------------------------------------------------------------ background
e("$N=24$ at $d=3$ and $N=120$", ("24 120", "TD", None, r"this is \$N=24\$"))

# ------------------------------------------------------------------ method
e("noise of strength $p_0=0.003$", ("0.003", "TD", None, r"strength \$p_0 = 0.003\$"))
e("A stream consists of 150 \\emph{steps} of 2000 shots",
  ("2000", "TD", None, r"fixed and \$2000\$ shots are sampled"), ("150", "TD", None, r"Streams are \$150\$ steps"))
e("switching with probability $0.04$ per step", ("0.04", "TD", None, r"^  probability \$0.04\$\."))
e("from step 50 on, the two qubits nearest the centre",
  ("50", "N02", None, r"hot_qubit \(one persistent change at step 50\)"),
  ("0.03", "TD", None, r"centre are hot with \$p_\{\\mathrm\{hot\}\} = 0.03\$"))
e("a burst starts with probability $0.04$",
  ("0.04", "TD", None, r"a burst starts with probability \$0.04\$"),
  ("5 0.05", "TD", None, r"it picks a random centre qubit, makes the \$5\$"), ("4", "TD", None, r"lasts \$4\$ steps"))
e("as \\reg{bursts} with 8-qubit clusters and $p_{\\mathrm{hot}}=0.1$",
  ("8", "TD", None, r"as \\code\{bursts\} but with \$8\$-qubit"), ("0.1", "TD", None, r"^  \$p_\{\\mathrm\{hot\}\}=0.1\$, outside"))
e("was evaluated only at $d=3$ on one stream", ("150 0", "N02", "### wander", r"^Shots per step 2000; drift 150 steps"))
e("$\\ln s\\sim\\mathcal U(\\ln0.5,\\ln2.5)$", ("0.5 2.5", "TD", None, r"\\ln s &\\sim \\mathcal U"),
  ("1 6", "TD", None, r"k &\\sim \\mathcal U\\\{1,\\dots,6"), ("0.005 0.08", "TD", None, r"\\mathcal U\(0.005,\\ 0.08\)"))
e("A chunk holds $B=512$ training shots", ("512 2000", "TD", None, r"plus \$B=512\$ training shots"))
e("that is\n$6\\times6\\times6$ at $d=5$", ("6", "TD", None, r"that is \$\(4,4,4\)\$ for \$d=3\$ and \$\(6,6,6\)\$"))
e("laid out on the same grid and multiplied by 10", ("10", "TD", None, r"multiplied by \$10\$"))
e("nominal mean firing rate of $4.28\\%$", ("4.279 4.28", "A10", None, r"^nominal mean firing rate"),
  ("10", "TD", None, r"about \$10\\%\$ of the rate"))
e("default $H_c=48$, $L=4$ has 195\\,681", ("194385 195681", "TD", None, r"\$H=48\$, \$L=4\$ \(default\)"),
  ("9", "TD", None, r"For \$L=4\$ that is \$9\^3\$"))
e("With probability $q=0.3$ per training step", ("0.3", "TD", None, r"With probability \$q=0.3\$"))
e("seeds (0, 1, 2). An \\emph{ensemble} averages", ("0 1 2", "N07", None, r"averages the logits of model seeds 0, 1 and 2"))
e("for each of the 49 qubits at $d=5$", ("49", "LOC", None, r"^- 49 qubits are scored"))
e("+0.1\\,\\mathrm{MSE}\\bigl(10", ("0.1 10 0.1", "R9", None, r"loss = BCE \+ 0.1 MSE\(p_hot x 10\) \+ 0.1 MSE\(log scale\)"))
e("encoder of two 16-channel 3D convolutions", ("2 16", "R9", None, r"fingerprint encoder 2 x Conv3d 16 ch"))

# ------------------------------------------------------------------ setup
e("All data are simulated with Stim~1.16.0", ("1.16.0 2.4.0", "N02", None, r"^\| Versions \|"))
e("At $d=3$ models train for 60\\,000 steps", ("60000", "N02", None, r"^\| Training \| 2 epochs = 60000 steps"),
  ("30000", "N02", None, r"^\| Training data \(W\) \|"))
e("at $d=5$ for 300\\,000 steps (two passes over 150\\,000", ("150000 300000", "N05", None, r"^\| Data size \|"))
e("Three fixed test sets of 200\\,000", ("200000 2000", "TD", None, r"Each has \$200\\,000\$ shots plus \$2000\$"),
  ("0.03", "TD", None, r"with \$p_\{\\mathrm\{hot\}\}=0.03\$. Each has"))
e("(stream seeds 0--4)", ("0 4", "TD", None, r"Stream seeds \$0\$--\$4\$"))
e("$w\\in\\{500,1000,2000,5000,10000\\}$ and", ("500 1000 2000 5000 10000", "TD", None, r"Window lengths \$w\\in"))
e("processed in sub-batches of 200", ("2000 200", "TD", None, r"processed in sub-batches of \$200\$"))
e("has 29 onset steps (33 bursts", ("29 29 21 5", "N03", None, r"^\| Onsets pooled over 5 streams"),
  ("33 29", "LOC", None, r"bursts: 33 events but 29 onset steps"))
e("nominal test\nset's 2000 fingerprint shots", ("2000", "N03", None, r"frozen fingerprint = mean of D's nominal"))
e("$0.001<p<0.05$ as weak evidence", ("0.001 0.05", "N07", None, r"0.001 < p < 0.05 is weak evidence"))
e("at onset or\nafter on \\reg{bursts} or \\reg{bursts\\_big} with $p<0.001$",
  ("0.001", "N08", "## Decision rule", r"^- \*\*Win:\*\*"), ("0.001", "N08", "## Decision rule", r"^- \*\*No loss:\*\*"))
e("drift steps with $p<0.001$, and", ("0.001", "R9", "## Decision rule", r"^A variant passes if"))

# ------------------------------------------------------------------ results: main
e("paired tests have $p\\le\\Eb{2}{302}$, the largest being",
  ("1e-300", "R6PY", None, r'else f"\{d:\+\.\{nd\}f\} \(p<1e-300\)"'),
  ("1e-300", "N07", "### d=5, drift steps: ensemble", r"^\| adaptive-stale \| bursts \|"),
  ("1e-300", "N07", "### d=5, drift steps: ensemble", r"^\| adaptive-stale \| bursts_big \|"),
  ("1e-300 2e-302", "N07", "### d=5, drift steps: ensemble", r"^\| adaptive-stale \| hot_qubit \|"),
  ("1e-300", "N07", "### d=5, drift steps: ensemble", r"^\| adaptive-stale \| telegraph \|"))
e("oracle MWPM remains better by $+0.0031$ (\\E{3}{14})",
  ("0.0031 3e-14", "N07", "### d=5, drift steps: ensemble", r"^\| adaptive-stale \| bursts_big \|"))
for reg, s5, w5, o5 in [("bursts", "0.0439", "0.0356", "0.0197"), ("bursts_big", "0.0980", "0.0896", "0.0426"),
                        ("hot_qubit", "0.0181", "0.0108", "0.0103"), ("telegraph", "0.0228", "0.0229", "0.0231")]:
    pass
e("static MWPM & 0.0439 & 0.0980 & 0.0181 & 0.0228",
  ("0.0439 0.0980 0.0181 0.0228", "N07", None, r"^Reference LERs on drift"))
e("best WindowDEM & 0.0356 & 0.0896 & 0.0108 & 0.0229",
  ("0.0356 0.0896 0.0108 0.0229", "N07", None, r"^Reference LERs on drift"))
e("oracle MWPM & 0.0197 & 0.0426 & 0.0103 & 0.0231",
  ("0.0197 0.0426 0.0103 0.0231", "N07", None, r"^Reference LERs on drift"))
for model, rows in [("static-wide", [("bursts", "0.0224 0.0258 0.0021"), ("bursts_big", "0.0564 0.0757 0.0153"),
                                     ("hot_qubit", "0.0098 0.0106 0.0005"), ("telegraph", "0.0175 0.0186 0.0004")]),
                    ("adaptive-wide", [("bursts", "0.0171 0.0190 0.0011"), ("bursts_big", "0.0484 0.0634 0.0125"),
                                       ("hot_qubit", "0.0071 0.0076 0.0002"), ("telegraph", "0.0164 0.0171 0.0003")]),
                    ("adaptive-stale", [("bursts", "0.0170 0.0184 0.0008"), ("bursts_big", "0.0457 0.0548 0.0059"),
                                        ("hot_qubit", "0.0073 0.0077 0.0004"), ("telegraph", "0.0164 0.0171 0.0002")])]:
    anchor = {"static-wide": "static-wide ensemble & 0.0224", "adaptive-wide": "adaptive-wide ensemble & 0.0171",
              "adaptive-stale": "adaptive-stale ensemble & 0.0170"}[model]
    e(anchor + " (and its single-seed row)",
      *[(nums, "N07", "### d=5, drift steps", rf"^\| {model} \| {reg} \| ") for reg, nums in rows])
e("$\\Delta$ vs static-wide ens. & $-0.0053$",
  ("0.0053 4e-142", "N07", "### d=5$", r"^\| bursts \| -0.0053"), ("0.0107 7e-158", "N07", "### d=5$", r"^\| bursts_big \| -0.0107"),
  ("0.0025 1e-298", "N07", "### d=5$", r"^\| hot_qubit \| -0.0025"), ("0.0010 3e-29", "N07", "### d=5$", r"^\| telegraph \| -0.0010"))
e("$\\Delta$ vs oracle MWPM & $-0.0027$",
  ("0.0027 5e-27", "N07", "### d=5, drift steps: ensemble", r"^\| adaptive-stale \| bursts \|"),
  ("0.0031 3e-14", "N07", "### d=5, drift steps: ensemble", r"^\| adaptive-stale \| bursts_big \|"),
  ("0.0030 1e-224", "N07", "### d=5, drift steps: ensemble", r"^\| adaptive-stale \| hot_qubit \|"),
  ("0.0066 1e-300", "N07", "### d=5, drift steps: ensemble", r"^\| adaptive-stale \| telegraph \|"))
e("static MWPM & 0.1742 & 0.3778 & 0.0586 & 0.0239",
  ("0.1742 0.3778", "N07", "### Baseline numbers", r"^\| static MWPM \| 0.1742"),
  ("0.0586 0.0239", "N07", "### Baseline numbers", r"^\| static MWPM \| 0.0586"))
e("best WindowDEM & 0.1069 & 0.2848 & 0.0250 & 0.0237",
  ("0.1069 0.2848", "N07", "### Baseline numbers", r"^\| best WindowDEM \(per phase\) \| 0.1069"),
  ("0.0250 0.0237", "N07", "### Baseline numbers", r"^\| best WindowDEM \(per phase\) \| 0.0250"))
e("oracle MWPM & 0.0667 & 0.1937 & 0.0228 & 0.0239",
  ("0.0667 0.1937", "N07", "### Baseline numbers", r"^\| oracle MWPM \| 0.0667"),
  ("0.0228 0.0239", "N07", "### Baseline numbers", r"^\| oracle MWPM \| 0.0228"))
for model, nums in [("static-wide", ["0.1049 0.1058 0.0001", "0.2427 0.2496 0.0055", "0.0276 0.0268 0.0008",
                                     "0.0235 0.0237 0.0001"]),
                    ("adaptive-wide", ["0.0713 0.0725 0.0006", "0.2014 0.2071 0.0042", "0.0196 0.0202 0.0001",
                                       "0.0208 0.0209 0.0000"]),
                    ("adaptive-stale", ["0.0720 0.0730 0.0002", "0.2011 0.2037 0.0026", "0.0208 0.0213 0.0004",
                                        "0.0214 0.0216 0.0001"])]:
    e(f"{model} ensemble & {nums[0].split()[0]} (and its single-seed row)",
      *[(n, "N07", "### d=3, drift steps", rf"^\| {model} \| {reg} \| ") for n, reg in
        zip(nums, ["bursts", "bursts_big", "hot_qubit", "telegraph"])])
e("$\\Delta$ vs static-wide ens. & $-0.0329$",
  ("0.0329 1e-300", "N07", "### d=3$", r"^\| bursts \| -0.0329"), ("0.0416 1e-300", "N07", "### d=3$", r"^\| bursts_big \| -0.0416"),
  ("0.0068 1e-300", "N07", "### d=3$", r"^\| hot_qubit \| -0.0068"), ("0.0021 7e-102", "N07", "### d=3$", r"^\| telegraph \| -0.0021"))
e("$\\Delta$ vs oracle MWPM & $+0.0053$",
  ("0.0053 7e-35", "N07", "### d=3, drift steps: ensemble", r"^\| adaptive-stale \| bursts \|"),
  ("0.0073 3e-30", "N07", "### d=3, drift steps: ensemble", r"^\| adaptive-stale \| bursts_big \|"),
  ("0.0020 3e-66", "N07", "### d=3, drift steps: ensemble", r"^\| adaptive-stale \| hot_qubit \|"),
  ("0.0025 5e-76", "N07", "### d=3, drift steps: ensemble", r"^\| adaptive-stale \| telegraph \|"))
e("$+0.0219$ on \\reg{bursts}, \\E{2}{90}",
  ("0.0219 2e-90 0.0224 7e-59 0.0006 0.2", "N07", "### d=3, the phases that matter", r"^\| adaptive-stale \|"))

# ------------------------------------------------------------------ results: adaptation
e("frozen is $+0.0253$ above static-wide",
  ("0.0253 1e-300", "N03", "#### Paired comparisons: bursts$", r"^\| adaptive-wide frozen vs static-wide \|"))
e("drift 0.0576 vs 0.0564; paired",
  ("0.0576", "R6", "### d=5 bursts_big", r"^\| adaptive-stale \(frozen\): ensemble"),
  ("0.0564", "R6", "### d=5 bursts_big", r"^\| static-wide: ensemble"),
  ("0.0576 0.0564 0.0012 0.0015", "PT", None, r"^\| 1\. adaptive-stale ens\. frozen"))
e("\\reg{bursts} ($+0.0001$, $p=0.6$) and",
  ("0.0171 0.0170 0.0001 0.6", "PT", None, r"^\| 2\. .* \| bursts \|"),
  ("0.0164 0.0164 0.0001 0.37", "PT", None, r"^\| 2\. .* \| telegraph \|"),
  ("0.0484 0.0457 0.0027 3.1e-19", "PT", None, r"^\| 2\. .* \| bursts_big \|"),
  ("0.0071 0.0073 0.0002 6.3e-06", "PT", None, r"^\| 2\. .* \| hot_qubit \|"))
e("cause error bursts that start at a localised impact, spread across a 26-qubit array",
  ("26", "EXT", None, r"We use a 26 qubit subset"), ("25", "EXT", None, r"exponential time constant of ∼ 25 ms"),
  ("10", "EXT", None, r"occurring on average every 10 s"))
e("at roughly 1\\,\\textmu s per",
  ("1 5 2000 10", "EXT", None, r"^Arithmetic: 2000 shots"), ("25", "EXT", None, r"exponential time constant of ∼ 25 ms"))
e("ensemble minus seed mean: $-0.0193$",
  ("0.0193 0.0149 0.0091", "N07", "## Short answers", r"Vs the seed mean it is"))
e("the live fingerprint contributes $0.0002$ or less", ("0.0002", "N07", None, r"^On telegraph the effect is 0.0002 or less"))
for d, model, regs in [(5, "adaptive-wide", ["-0.0092 2e-315", "-0.0196 1e-300", "-0.0033 1e-300", "-0.0002 0.002"]),
                       (5, "adaptive-stale", ["-0.0063 1e-209", "-0.0119 3e-247", "-0.0025 1e-300", "-0.0002 0.0002"]),
                       (3, "adaptive-wide", ["-0.0568 1e-300", "-0.0891 1e-300", "-0.0194 1e-300", "-0.0000 0.5"]),
                       (3, "adaptive-stale", ["-0.0410 1e-300", "-0.0519 1e-300", "-0.0098 1e-300", "-0.0002 2e-05"])]:
    e(f"$d={d}$ {model} & ${regs[0].split()[0]}$",
      *[(n, "N07", ("## 3. Ensemble live", f"### d={d}$"), rf"^\| {model} \| {reg} \| ") for n, reg in
        zip(regs, ["bursts", "bursts_big", "hot_qubit", "telegraph"])])
e("the adaptive-wide ensemble is $+0.0035$ worse live",
  ("0.0035 5e-41", "N07", "## 3. Ensemble live", r"^\| adaptive-wide \| bursts_big \|"),
  ("0.0021 3e-11", "N07", "### d=5, the phases that matter", r"^\| adaptive-wide \|"))
e("reduces the live cost to\n$+0.0006$ ($p=0.0001$)",
  ("0.0006 0.0001", "N07", "## 3. Ensemble live", r"^\| adaptive-stale \| bursts_big \|"),
  ("0.0007 0.006", "N07", "### d=5, the phases that matter", r"^\| adaptive-stale \|"))
e("$+0.0026$ above static MWPM after \\reg{bursts\\_big} events",
  ("0.0026 3e-13", "N07", "### d=3, the phases that matter", r"^\| adaptive-stale \|"))
e("trained on nominal noise only scores 0.00536",
  ("0.00532 0.00536 0.4", "N03", "### b\\) Adaptive vs static-nominal", r"^\| adaptive-wide vs static-nominal \|"),
  ("0.0007 0.00605 0.00536 1e-08", "N03", "### b\\) Adaptive vs static-nominal", r"^\| static-wide vs static-nominal \|"),
  ("13", "N03", None, r"^- Static-wide pays about 13%"))
e("there is no such cost: static-wide is lower than static-nominal in all three seeds",
  ("0.2", "N06", None, r"static-wide vs static-nominal is −0.00028"),
  ("0.038", "A6", None, r"^d5 seed0 nominal"), ("3.2e-09", "A6", None, r"^d5 seed1 nominal"),
  ("0.18", "A6", None, r"^d5 seed2 nominal"))

# ------------------------------------------------------------------ results: phases
e("ties oracle\nMWPM at onset ($-0.0001$, $p=0.9$)",
  ("0.0001 0.9", "N09", None, r"^\| bursts onset vs oracle MWPM \|"),
  ("0.0034 2e-34", "N09", None, r"^\| bursts later vs oracle MWPM \|"),
  ("0.0065 3e-14", "N09", None, r"^\| bursts_big onset vs oracle MWPM \|"),
  ("0.0019 2e-05", "N09", None, r"^\| bursts_big later vs oracle MWPM \|"))
e("even the best window is at 0.0116",
  ("0.0116", "R6", "### d=5 bursts \\(", r"^\| best WindowDEM"), ("0.0033", "R6", "### d=5 bursts \\(", r"^\| static MWPM"))
e("pooled over 5 streams (29 onset steps each)", ("29 29", "N03", None, r"^\| Onsets pooled over 5 streams"))

# ------------------------------------------------------------------ results: training length
e("(three-seed means 0.118 to 0.150, against 0.098;",
  ("0.098 0.128 0.150 0.118", "N04", None, r"^- The classical baselines win back two regimes"),
  ("0.29441", "N04", None, r"^\| static-nominal \| 0.04879"))
e("it still loses on \\reg{bursts\\_big}: 0.216", ("0.2159", "N06", None, r"^\| static-nominal \| 0.2159 ± 0.0604 \| 0/3"),
  ("0.216 0.0980", "A6", None, r"^d5 static-nominal single seeds"))
e("static-nominal was worse than MWPM on nominal noise (0.00400 vs 0.00329)",
  ("0.00400 0.00329", "N04", None, r"^- The d=5 static-nominal model is broken"))
e("(static-nominal $0.00253\\pm0.00026$", ("0.00253 0.00026 1e-4", "N06", "## Summary", r"^1\. Q1"))
e("Between 240\\,000 and 300\\,000", ("240k 300k 5", "N05", None, r"^From 240k to 300k steps, all 5 models"))
e("checkpoints every 30\\,000 steps", ("30000", "N05", None, r"Checkpoints every 30000 steps"))

# ------------------------------------------------------------------ results: localisation
e("(9 to 24 detectors per qubit at $d=5$)", ("9 24", "LOC", None, r"^- Each qubit has 9–24 adjacent"))
e("Both methods use the same\ncausal 2000-shot buffer", ("2000", "LOC", None, r"`window` = 2000"))
e("the baseline scores 0.825 to 0.841", ("0.825 0.841 0.975 1.000", "LOC", None, r"^   - Baseline: 0.825–0.841"))
e("NF wins 20 bursts, loses 1 and ties 12", ("20 1 12 2e-05 17 18 0 2e-05", "LOC", None, r"^   - Per event, NF wins 20 bursts"))
e("including the training-free score, scores 1.000", ("1.000", "LOC", None, r"^1\. \*\*hot_qubit: the baseline is perfect"))
e("averages 0.056 to 0.114 above the baseline (0.845 to 0.849)",
  ("0.845 0.849 0.905 0.958 0.056 0.114", "LOC", None, r"^   - Baseline: 0.845–0.849"))
e("seed 1 on later steps ($p=0.013$, weak)", ("0.001 0.013", "LOC", None, r"^   - Per event, no seed reaches p < 0.001"))
e("96\\% (\\reg{bursts}) and 100\\% (\\reg{bursts\\_big})", ("96 100", "LOC", None, r"^4\. \*\*The baseline's misses"))
e("``ev.'' is the number of events",
  ("29 29", "LOC", "^## Results", r"^\| bursts \| onset \| 29 \(29\)"),
  ("103 33", "LOC", "^## Results", r"^\| bursts \| later \| 103 \(33\)"),
  ("29 29", "LOC", "^## Results", r"^\| bursts_big \| onset \| 29 \(29\)"),
  ("87 29", "LOC", "^## Results", r"^\| bursts_big \| later \| 87 \(29\)"),
  ("5 5", "LOC", "^## Results", r"^\| hot_qubit \| onset \| 5 \(5\)"),
  ("495 5", "LOC", "^## Results", r"^\| hot_qubit \| later \| 495 \(5\)"))

# ------------------------------------------------------------------ ablations
e("B: multiscale fingerprint & 3 & fails",
  ("0.0171 3e-112 0.0002 0.0005", "N08", "## Decision rule", r"^\| B multiscale \| \*\*FAILS\*\*"))
e("B: multiscale fingerprint & 5 & fails",
  ("0.0019 2e-07", "N09", "## Decision rule", r"^\| Win \(onset"),
  ("0.0019 4e-09 0.0024 2e-11", "N09", "## Decision rule", r"^\| No loss"))
e("C: B + 8 pair covariances & 3 & fails",
  ("0.0180 2e-110 0.0050 8e-21 0.0082 2e-43", "N08", "## Decision rule", r"^\| C multiscale\+pairs \| \*\*FAILS\*\*"),
  ("8", "N08", None, r"^\| C multiscale\+pairs \| B's channels \+ 8"))
e("N: noise-map head & 5 & fails",
  ("0.0007 0.03", "R9", "## bursts_big", r"^\| N noise-map head vs adaptive-stale \(baseline\) \|"))
e("NF: noise-map head + FiLM & 5 & fails",
  ("0.0007 0.04", "R9", "## bursts_big", r"^\| NF noise-map head \+ FiLM vs adaptive-stale \(baseline\) \|"))
e("the 2000-shot rates (long), the rates over the last 200 shots",
  ("2000", "N09", None, r"^  - long = the whole 2000-shot window"), ("200", "N09", None, r"^  - short = the last 200 shots"))
e("\\reg{bursts\\_big}\n($-0.0175$, \\E{4}{67})",
  ("0.0175 4e-67", "N08", "\\*\\*B multiscale vs baseline\\*\\*", r"^\| bursts_big \|"),
  ("0.0014 8e-09", "N08", "\\*\\*B multiscale vs baseline\\*\\*", r"^\| bursts \|"),
  ("0.17", "A5", None, r"^  d3 B after bursts - static"), ("0.15", "A5", None, r"^  d3 B after bursts_big - static"))
e("by about 1\\% relative", ("0.0002 0.0005", "N08", None, r"^Outcome, stated exactly"),
  ("1", "N08", None, r"B's hot_qubit loss is tiny but significant"))
e("The onset gain shrinks on \\reg{bursts}",
  ("0.0019", "N09", "## Summary", r"^   - bursts onset improves by"),
  ("0.0019 2e-07", "N09", "## Decision rule", r"^\| Win \(onset"),
  ("0.0003 0.7", "N09", "## Summary", r"^   - bursts_big onset does not change"),
  ("0.0001 1", "N09", "## Summary", r"^   - hot_qubit onset does not change"))
e("($+4.1\\%$ and $+5.4\\%$ relative)",
  ("0.0019 4e-09 4.1 0.0024 2e-11 5.4", "N09", "## Summary", r"^1\. \*\*Decision rule: B FAILS"))
e("$+6.4\\%$ to $+17.7\\%$ relative on nominal stream steps",
  ("6.4 17.7", "N09", None, r"^15\. \*\*At d=5 the live multiscale"),
  ("6.4 5.5e-10", "A5", "== d5 B", r"^   bursts     nominal"), ("7.4 2.8e-12", "A5", "== d5 B", r"^   bursts_big nominal"),
  ("17.7 5e-21", "A5", "== d5 B", r"^   hot_qubit  nominal"), ("8.3 1.2e-10", "A5", "== d5 B", r"^   telegraph  nominal"))
e("beats oracle MWPM\nat \\reg{bursts} onset ($-0.0020$, $p=0.0002$)", ("0.0020 0.0002", "N09", None, r"^6\. \*\*One new positive result"))
e("(60\\,000 vs 300\\,000 steps) and data", ("60k 300k", "N09", None, r"^- \*\*Confounds\.\*\*"))
e("drift\nvs baseline $-0.0064$, \\E{1}{114}",
  ("0.0064 1e-114", "N08", "\\*\\*C multiscale\\+pairs vs baseline\\*\\*", r"^\| bursts \|"),
  ("0.0011 2e-54", "N08", "\\*\\*C multiscale\\+pairs vs baseline\\*\\*", r"^\| hot_qubit \|"),
  ("0.00112 1e-11", "N08", None, r"^- C vs baseline:"),
  ("0.0050 8e-21", "N08", "\\*\\*C multiscale\\+pairs vs baseline\\*\\*", r"^\| bursts_big \|"),
  ("0.0112 1e-113", "N08", "\\*\\*C vs B\\*\\*", r"^\| bursts_big \|"))
e("(0.2714 vs\n0.2539, unpaired)", ("0.2714 0.2539", "N08", None, r"Frozen ensembles on bursts_big drift score"))
e("On the test sets NF vs the baseline is $-0.00006$",
  ("0.00006 0.3 0.00016 0.3 0.00012 0.3", "R9", None, r"^\| NF noise-map head \+ FiLM vs adaptive-stale \(baseline\) \| -0.00006"))
e("(565\\,121 parameters)", ("565121", "TD", None, r"\$H=64\$, \$L=6\$ \(capacity check\)"))
e("(0.0484 vs 0.0514, $-0.0031$, \\E{8}{14})", ("0.0484 0.0514 0.0031 8e-14", "N05", "## Summary", r"^5\. h64/l6 beats h48/l4"),
  ("0.00009 0.1", "N05", None, r"^Q5\. Does h64/l6 beat"), ("2.2 2 22 65", "R4", None, r"^5\. \*\*h64/l6 beats h48/l4"))
e("with 0.3 points (0.0590 vs 0.0620)", ("0.0590 0.0620 0.3 1", "N05", None, r"^\| \"h64/l6 clearly won\" \|"))
e("ensembles (0.0457 and 0.0484) are at or below it", ("0.0457 0.0484 0.0484", "N07", None, r"^14\. \*\*A 3-seed h48/l4 ensemble"))
e("(largest\n$p=\\Eb{1}{7}$)", ("1e-07", "N07", "## Short answers", r"^     - The gain is largest|^   - \*\*d=5: yes, clearly"))
e("the gain over the best seed is $-0.0065$",
  ("0.0065 0.0032 0.0056", "N07", "## Short answers", r"^     - The gain is largest where seeds disagree"))
e("falls from $-0.0209$ (single-seed mean) to $-0.0107$",
  ("0.0107 0.0209", "N07", "## Short answers", r"^   - At d=5 the margin is smaller than with single seeds"))
e("($-0.0119$ vs a single-seed mean of $-0.0125$)",
  ("0.0119 0.0125", "N07", ("## 3. Ensemble live", "### d=5$"), r"^\| adaptive-stale \| bursts_big \| -0.0119"))
e("on \\reg{hot\\_qubit} drift ($+0.0015$, \\E{4}{137})", ("0.0015 4e-137", "N07", None, r"^     - Elsewhere it ties the best seed"))

# ------------------------------------------------------------------ limitations, conclusion
e("it takes 1.85\\,\\textmu s per shot on CPU",
  ("1.85", "N04", "Per-shot latency", r"^\| 5 \| MWPM \| CPU \|"),
  ("10.2", "N04", "Per-shot latency", r"^\| 5 \| static NN \| L4 \|"),
  ("10.8", "N04", "Per-shot latency", r"^\| 5 \| adaptive NN \| L4 \|"),
  ("4096", "N04", None, r"^- MWPM is the fastest decoder in every configuration"))
e("curves were still falling at 300\\,000 steps", ("300k", "N05", None, r"^Training is not finished"))
e("and trails it by $0.0031$ on out-of-distribution large", ("0.0031 3e-14", "N07", "### d=5, drift steps: ensemble", r"^\| adaptive-stale \| bursts_big \|"))

e("default $H_c=48$, $L=4$ (also Appendix A, and h64/l6 $H_c=64$, $L=6$)",
  ("48 4", "TD", None, r"\$H=48\$, \$L=4\$ \(default\)"), ("64 6", "TD", None, r"\$H=64\$, \$L=6\$ \(capacity check\)"))
e("lowers the drift-step LER in every case at $p\\le\\Eb{2}{209}$ (the largest localised-regime p in Table 2)",
  ("1.2e-209", "A2", None, r"^  d5 ast bursts:"))

# ------------------------------------------------------------------ added after the claim audit (whitepaper/audit)
e("On nominal noise the network is already about 36\\,\\%", ("36", "A8", None, r"^  bursts: 0.0021 vs 0.0033"))
e("since the network is already 35--39\\,\\% below MWPM on nominal steps",
  ("36", "A8", None, r"^  bursts: 0.0021"), ("38", "A8", None, r"^  bursts_big: 0.0021"),
  ("39", "A8", None, r"^  hot_qubit: 0.0021"), ("35", "A8", None, r"^  telegraph: 0.0021"))
e("while frozen adaptive-wide is well above it (0.0680, \\E{2}{178})",
  ("0.0680 0.0564 2e-178", "A8", None, r"^  d5 aw frozen - sw ens bursts_big"))
e("(29 runs per burst regime, 21 for", ("29 29 21 5", "N03", None, r"^\| Onsets pooled over 5 streams"))
e("keeps $p\\le\\Eb{1.5}{5}$ except",
  ("26 3 1.5e-05", "A9", None, r"^d5 bursts     ast vs oracle"),
  ("28 0 7.5e-09", "A9", None, r"^d5 bursts_big ast vs static-wide"),
  ("27 1 2.2e-07", "A9", None, r"^d5 bursts_big ast live-frozen"),
  ("9 20 0.061", "A9", None, r"^d5 bursts_big ast vs oracle"),
  ("5 0 0.062", "A9", None, r"^d5 hot_qubit  ast vs oracle"))
e("all five seed-0 models\n(static-wide, static-nominal", ("5", "N05", None, r"^From 240k to 300k steps, all 5 models"))

# ------------------------------------------------------------------ appendix A, B
e("Detectors per shot; grid $(T,X,Y)$ & 24", ("24 120", "TD", None, r"this is \$N=24\$"))
e("194\\,385 (static), 195\\,681 (adaptive); h64/l6: 563\\,393 / 565\\,121",
  ("194385 195681", "TD", None, r"\$H=48\$, \$L=4\$ \(default\)"), ("563393 565121", "TD", None, r"\$H=64\$, \$L=6\$"))
e("Fingerprint & $W=2000$ shots, scaled by 10; updated every 200 shots",
  ("2000", "TD", None, r"seen \(\$W=2000\$\)"), ("200", "TD", None, r"sub-batches of \$200\$"))
e("AdamW, peak learning rate $10^{-3}$", ("10\\^\\{-3\\}", "TD", None, r"AdamW with peak learning rate"), )
e("512 shots (one chunk) per step, fp32", ("512", "TD", None, r"its \$B=512\$ shots"))
e("$w\\in\\{500,1000,2000,5000,10000\\}$; estimates clipped", ("10\\^\\{-6\\} 0.45", "TD", None, r"clipped to"))
e("Stim 1.16.0, PyMatching 2.4.0, NumPy 2.4.6, PyTorch 2.15.0",
  ("1.16.0 2.4.0 2.4.6 2.15.0", "N02", None, r"^\| Versions \|"))
for name, row in [("d3\\_wide", r"\\code\{d3\\_wide\} & wide"), ("d5\\_wide", r"\\code\{d5\\_wide\} & wide"),
                  ("d5\\_nominal", r"\\code\{d5\\_nominal\} & nominal"), ("d5\\_wide\\_x5", r"\\code\{d5\\_wide\\_x5\}"),
                  ("d5\\_nominal\\_x5", r"\\code\{d5\\_nominal\\_x5\}")]:
    nums = {"d3\\_wide": "30000 15.4", "d5\\_wide": "30000 15.4", "d5\\_nominal": "30000 15.4",
            "d5\\_wide\\_x5": "150000 76.8", "d5\\_nominal\\_x5": "150000 76.8"}[name]
    size = {"d3\\_wide": ("107", "d3_wide"), "d5\\_wide": ("510", "d5_wide"), "d5\\_nominal": ("85", "d5_nominal"),
            "d5\\_wide\\_x5": ("2.28", "d5_wide_x5"), "d5\\_nominal\\_x5": ("392", "d5_nominal_x5")}[name]
    e("\\code{" + name + "} & ", (nums, "TD", None, row), (size[0], "A10", None, rf"^dataset {size[1]}:"))
e("\\code{d3\\_r3\\_p0.003\\_s0}$^{*}$ & 3", ("101", "A10", None, r"^dataset d3_r3_p0.003_s0:"))
e("\\code{d3\\_nominal} & 3 & nominal only & 30\\,000 & 15.4\\,M & 16", ("16", "A10", None, r"^dataset d3_nominal:"))
e("\\code{d3\\_nominal} & 3 & nominal only & 30\\,000", ("30000 512", "N04", None, r"is a nominal-only dataset of the same size as"))
e("All have 1000 validation chunks", ("1000", "TD", None, r"All have \$1000\$ validation chunks"))


# ------------------------------------------------------------------ appendix C (row by row)
def rows_r6(d, reg, items):
    for lab, src, nums in items:
        e(f"[Table {'6' if d == 5 else '7'}, {reg}] {lab}",
          (nums, "R6", rf"### d={d} {reg} \(", rf"^\| {re.escape(src)} \|"))


T5 = {
    "bursts": [("static MWPM", "static MWPM", "0.0439 0.0455 0.0435 0.0033 0.0033"),
               ("best WindowDEM", "best WindowDEM (picked per phase)", "0.0356 2000 0.0455 1000 0.0327 2000 0.0116 1000 0.0046 5000"),
               ("oracle MWPM", "oracle MWPM", "0.0197 0.0195 0.0198 0.0033 0.0033"),
               ("static-wide ens.", "static-wide: ensemble", "0.0224 0.0232 0.0221 0.0021 0.0021"),
               ("adaptive-wide ens.", "adaptive-wide: ensemble", "0.0171 0.0204 0.0162 0.0033 0.0021"),
               ("adaptive-stale ens.", "adaptive-stale: ensemble", "0.0170 0.0194 0.0164 0.0023 0.0021"),
               ("adaptive-stale ens., frozen", "adaptive-stale (frozen): ensemble", "0.0233 0.0244 0.0230 0.0019 0.0020")],
    "bursts_big": [("static MWPM", "static MWPM", "0.0980 0.0993 0.0975 0.0034 0.0034"),
                   ("best WindowDEM", "best WindowDEM (picked per phase)", "0.0896 2000 0.0992 500 0.0822 2000 0.0206 500 0.0056 2000"),
                   ("oracle MWPM", "oracle MWPM", "0.0426 0.0422 0.0428 0.0034 0.0034"),
                   ("static-wide ens.", "static-wide: ensemble", "0.0564 0.0559 0.0566 0.0022 0.0022"),
                   ("adaptive-wide ens.", "adaptive-wide: ensemble", "0.0484 0.0539 0.0466 0.0056 0.0022"),
                   ("adaptive-stale ens.", "adaptive-stale: ensemble", "0.0457 0.0487 0.0447 0.0028 0.0021"),
                   ("adaptive-stale ens., frozen", "adaptive-stale (frozen): ensemble", "0.0576 0.0583 0.0574 0.0022 0.0020")],
    "hot_qubit": [("static MWPM", "static MWPM", "0.0181 0.0187 0.0180 0.0034"),
                  ("best WindowDEM", "best WindowDEM (picked per phase)", "0.0108 10000 0.0135 5000 0.0108 10000 0.0033 10000"),
                  ("oracle MWPM", "oracle MWPM", "0.0103 0.0110 0.0103 0.0034"),
                  ("static-wide ens.", "static-wide: ensemble", "0.0098 0.0089 0.0098 0.0020"),
                  ("adaptive-wide ens.", "adaptive-wide: ensemble", "0.0071 0.0073 0.0071 0.0020"),
                  ("adaptive-stale ens.", "adaptive-stale: ensemble", "0.0073 0.0071 0.0073 0.0021"),
                  ("adaptive-stale ens., frozen", "adaptive-stale (frozen): ensemble", "0.0098 0.0084 0.0098 0.0019")],
    "telegraph": [("static MWPM", "static MWPM", "0.0228 0.0217 0.0229 0.0035 0.0033"),
                  ("best WindowDEM", "best WindowDEM (picked per phase)", "0.0229 10000 0.0217 10000 0.0230 10000 0.0038 5000 0.0033 10000"),
                  ("oracle MWPM", "oracle MWPM", "0.0231 0.0217 0.0232 0.0035 0.0033"),
                  ("static-wide ens.", "static-wide: ensemble", "0.0175 0.0167 0.0175 0.0022 0.0021"),
                  ("adaptive-wide ens.", "adaptive-wide: ensemble", "0.0164 0.0154 0.0164 0.0021 0.0021"),
                  ("adaptive-stale ens.", "adaptive-stale: ensemble", "0.0164 0.0160 0.0165 0.0021 0.0021"),
                  ("adaptive-stale ens., frozen", "adaptive-stale (frozen): ensemble", "0.0166 0.0160 0.0167 0.0022 0.0021")],
}
T3 = {
    "bursts": [("static MWPM", "static MWPM", "0.1742 0.1792 0.1728 0.0065 0.0066"),
               ("best WindowDEM", "best WindowDEM (picked per phase)", "0.1069 2000 0.1143 2000 0.1027 5000 0.0242 500 0.0093 2000"),
               ("oracle MWPM", "oracle MWPM", "0.0667 0.0681 0.0663 0.0065 0.0066"),
               ("static-wide ens.", "static-wide: ensemble", "0.1049 0.1074 0.1042 0.0062 0.0062"),
               ("adaptive-wide ens.", "adaptive-wide: ensemble", "0.0713 0.0954 0.0645 0.0133 0.0059"),
               ("adaptive-stale ens.", "adaptive-stale: ensemble", "0.0720 0.0901 0.0669 0.0075 0.0058"),
               ("adaptive-stale ens., frozen", "adaptive-stale (frozen): ensemble", "0.1130 0.1149 0.1125 0.0057 0.0057")],
    "bursts_big": [("static MWPM", "static MWPM", "0.3778 0.3804 0.3769 0.0066 0.0065"),
                   ("best WindowDEM", "best WindowDEM (picked per phase)", "0.2848 2000 0.2989 1000 0.2760 2000 0.0288 500 0.0101 2000"),
                   ("oracle MWPM", "oracle MWPM", "0.1937 0.1926 0.1941 0.0066 0.0065"),
                   ("static-wide ens.", "static-wide: ensemble", "0.2427 0.2416 0.2430 0.0059 0.0061"),
                   ("adaptive-wide ens.", "adaptive-wide: ensemble", "0.2014 0.2248 0.1935 0.0228 0.0062"),
                   ("adaptive-stale ens.", "adaptive-stale: ensemble", "0.2011 0.2151 0.1964 0.0092 0.0058"),
                   ("adaptive-stale ens., frozen", "adaptive-stale (frozen): ensemble", "0.2530 0.2514 0.2535 0.0054 0.0056")],
    "hot_qubit": [("static MWPM", "static MWPM", "0.0586 0.0575 0.0586 0.0067"),
                  ("best WindowDEM", "best WindowDEM (picked per phase)", "0.0250 10000 0.0299 2000 0.0249 10000 0.0066 10000"),
                  ("oracle MWPM", "oracle MWPM", "0.0228 0.0214 0.0229 0.0067"),
                  ("static-wide ens.", "static-wide: ensemble", "0.0276 0.0268 0.0276 0.0061"),
                  ("adaptive-wide ens.", "adaptive-wide: ensemble", "0.0196 0.0255 0.0195 0.0055"),
                  ("adaptive-stale ens.", "adaptive-stale: ensemble", "0.0208 0.0241 0.0208 0.0056"),
                  ("adaptive-stale ens., frozen", "adaptive-stale (frozen): ensemble", "0.0307 0.0311 0.0307 0.0057")],
    "telegraph": [("static MWPM", "static MWPM", "0.0239 0.0245 0.0239 0.0068 0.0064"),
                  ("best WindowDEM", "best WindowDEM (picked per phase)", "0.0237 10000 0.0244 10000 0.0237 10000 0.0066 10000 0.0064 10000"),
                  ("oracle MWPM", "oracle MWPM", "0.0239 0.0245 0.0239 0.0068 0.0064"),
                  ("static-wide ens.", "static-wide: ensemble", "0.0235 0.0232 0.0236 0.0061 0.0060"),
                  ("adaptive-wide ens.", "adaptive-wide: ensemble", "0.0208 0.0208 0.0208 0.0055 0.0054"),
                  ("adaptive-stale ens.", "adaptive-stale: ensemble", "0.0214 0.0216 0.0214 0.0056 0.0056"),
                  ("adaptive-stale ens., frozen", "adaptive-stale (frozen): ensemble", "0.0217 0.0216 0.0217 0.0056 0.0056")],
}
for reg, items in T5.items():
    rows_r6(5, reg, items)
for reg, items in T3.items():
    rows_r6(3, reg, items)

# Table 8 (variants)
V3 = {"bursts": ["0.0720 0.0901 0.0669 0.0075 0.0058", "0.0672 0.0730 0.0655 0.0061 0.0057", "0.0655 0.0720 0.0637 0.0061 0.0057"],
      "bursts_big": ["0.2011 0.2151 0.1964 0.0092 0.0058", "0.1950 0.1976 0.1941 0.0062 0.0056", "0.2061 0.2105 0.2047 0.0068 0.0056"],
      "hot_qubit": ["0.0208 0.0241 0.0208 0.0056", "0.0210 0.0208 0.0210 0.0056", "0.0197 0.0196 0.0197 0.0056"],
      "telegraph": ["0.0214 0.0216 0.0214 0.0056 0.0056", "0.0214 0.0212 0.0214 0.0058 0.0056", "0.0212 0.0212 0.0212 0.0056 0.0056"]}
for reg, rows in V3.items():
    e(f"[Table 8, d=3, {reg}] adaptive-stale / B / C",
      (" ".join(rows), "N08", "### Per regime and phase", rf"^\| {reg} \| "))
V5 = {"bursts": ["0.0170 0.0194 0.0164 0.0023 0.0021", "0.0167 0.0175 0.0165 0.0020 0.0022"],
      "bursts_big": ["0.0457 0.0487 0.0447 0.0028 0.0021", "0.0476 0.0490 0.0472 0.0023 0.0022"],
      "hot_qubit": ["0.0073 0.0071 0.0073 0.0021", "0.0074 0.0072 0.0074 0.0024"],
      "telegraph": ["0.0164 0.0160 0.0165 0.0021 0.0021", "0.0164 0.0158 0.0165 0.0020 0.0023"]}
for reg, rows in V5.items():
    e(f"[Table 8, d=5, {reg}] adaptive-stale / B",
      (" ".join(rows), "N09", "## Results per regime and phase", rf"^\| {reg} \| "))
VN = {"bursts": ["0.0169 0.0196 0.0161 0.0024 0.0021", "0.0168 0.0195 0.0161 0.0024 0.0021"],
      "bursts_big": ["0.0464 0.0500 0.0452 0.0027 0.0021", "0.0464 0.0501 0.0451 0.0029 0.0021"],
      "hot_qubit": ["0.0073 0.0072 0.0073 0.0021", "0.0073 0.0070 0.0073 0.0021"],
      "telegraph": ["0.0163 0.0153 0.0164 0.0020 0.0021", "0.0163 0.0153 0.0164 0.0019 0.0021"]}
for reg, (n, nf) in VN.items():
    e(f"[Table 8, d=5, {reg}] N / NF",
      (n, "R9", rf"^## {reg} \(", r"^\| N noise-map head: ensemble \|"),
      (nf, "R9", rf"^## {reg} \(", r"^\| NF noise-map head \+ FiLM: ensemble \|"))


# ------------------------------------------------------------------ round 10 (physics-based drift)
# Abstract
e("Trained once on uniform circuit-level noise ($p=0.003$), the same frozen weights transfer",
  ("0.003", "TD", None, r"strength \$p_0 = 0.003\$"),
  ("0.0015", "N10D", None, r"SI1000 strength p = CHOSEN_SI1000_P"))
e("a device-level simulation with SI1000 circuit noise ($p=0.0015$) under per-qubit heterogeneity",
  ("0.0015", "N10D", None, r"SI1000 strength p = CHOSEN_SI1000_P = 0.0015"))
e("trained ensembles beat MWPM calibrated to that device on its static noise (0.0016 to 0.0017 against 0.0022,",
  ("0.0016 0.0017 0.0022", "R10J", None,
   "lers.phys_static.static_wide_live_ens.all|.4f;lers.phys_static.static_nominal_live_ens.all|.4f;lers.phys_static.mwpm_cal.all|.4f"))
e("largest paired McNemar $p=\\Eb{4}{56}$), and are below or statistically indistinguishable from",
  ("4e-56", "R10J", None, "H3.models.static_nominal.mcnemar_p|.1g"))
e("(\\reg{phys\\_mixed}: 0.0463 against 0.0476 and 0.0472). The live fingerprint, however, does not beat its",
  ("0.0463 0.0476 0.0472", "R10J", None,
   "lers.phys_mixed.adaptive_stale_live_ens.drift|.4f;lers.phys_mixed.mwpm_cal.drift|.4f;lers.phys_mixed.mwpm_oracle.drift|.4f"))

# Physics setup
e("Gidney et al.~\\cite{gidney2021honeycomb} at strength $p=0.0015$",
  ("0.0015", "N10D", None, r"SI1000 strength p = CHOSEN_SI1000_P = 0.0015"))
e("strength was chosen before the run: the static device's 200\\,000-shot matching LER is 0.002255, the closest",
  ("200 0.002255", "PN", None, r"^# 200,000-shot static-MWPM LERs"),
  ("0.002255 0.0015", "PN", None, r"^CHOSEN_SI1000_P = 0.0015"))
e("of the candidate strengths to the 0.0033 nominal LER of the version-1 circuit.",
  ("0.0033", "PN", None, r"^CHOSEN_SI1000_P = 0.0015"))
e("one simulated device (device seed 0) carries a fixed per-qubit heterogeneity",
  ("0", "N10D", None, r"^- Streams: 6 regimes x stream seeds 0-4, device_seed 0"))
e("error bursts from point impacts at rate 0.02 per 2000-shot step, deliberately",
  ("0.02 2000", "PN", None, r"accelerated \(0\.02 per 2000-shot step\)"))
e("single-qubit depolarising noise with amplitude $0.2\\,e^{-\\mathrm{age}/25}$, age in sub-steps",
  ("0.2 25", "N10D", None, r"burst amplitude 0\.2\*exp\(-age/25\)"))
e("Each of the six regimes runs on five",
  ("1500 200", "N10D", None, r"1500 sub-steps x 200 shots each"))
e("$w\\in\\{500, 1000, 2000, 5000, 10000\\}$ shots",
  ("500 1000 2000 5000 10000", "N10D", None, r"WindowDEM windows 500, 1000, 2000, 5000, 10000"))
e("after the fact. The four ensembles of \\cref{sec:variants} and their single seeds are evaluated causally in",
  ("200", "N10D", None, r"^ensemble = mean of the 3 seeds"))
e("and frozen (the fingerprint of a 2000-shot static-device calibration set, shared by all members). A",
  ("2000", "N10D", None, r"2000 shots of the static device"))
e("sub-step is \\emph{drift} by pre-registered rule: max burst $\\ge0.005$ in \\reg{phys\\_cosmic}, any fluctuator",
  ("0.005", "N10D", None, r"phys_cosmic: max burst >= 0.005"))
e("two-sided McNemar on shared shots, $p<0.001$ significant, $0.001<p<0.05$ weak, no correction; exact",
  ("0.001 0.05", "N10D", None, r"p < 0.001 significant, 0.001-0.05 weak"))

# Pre-registered hypotheses (quoted verbatim)
e("drift steps of \\reg{phys\\_tls}, \\reg{phys\\_cosmic} and \\reg{phys\\_mixed} (McNemar $p<0.001$, and",
  ("0.001 0.01", "N10H", None, r"^H1:"))
e("\\reg{phys\\_cosmic} and \\reg{phys\\_mixed} ($p<0.001$).",
  ("0.001", "N10H", None, r"^H2:"))

# Results table (Table tab:phys)
def phys_row(label, key):
    regs = ["phys_static", "phys_tls", "phys_cosmic", "phys_slow", "phys_coupler", "phys_mixed"]
    vals = [f"{J_LERS[r][key]['all' if r == 'phys_static' else 'drift']:.4f}" for r in regs]
    specs = ";".join(f"lers.{r}.{key}.{'all' if r == 'phys_static' else 'drift'}|.4f" for r in regs)
    e(f"{label} & " + " & ".join(vals), (" ".join(vals), "R10J", None, specs))

J_LERS = R10["lers"]
phys_row("MWPM (calibrated)", "mwpm_cal")
phys_row("MWPM (nominal, ref.)", "mwpm_nom")
phys_row("MWPM (oracle)", "mwpm_oracle")
phys_row("static-wide ens.", "static_wide_live_ens")
phys_row("static-nominal ens.", "static_nominal_live_ens")
phys_row("adaptive-wide ens. (live)", "adaptive_wide_live_ens")
phys_row("adaptive-stale ens. (live)", "adaptive_stale_live_ens")
phys_row("adaptive-stale ens. (frozen)", "adaptive_stale_frozen_ens")
BW = {r: min((500, 1000, 2000, 5000, 10000), key=lambda w: J_LERS[r][f"window_{w}"]["all" if r == "phys_static" else "drift"])
      for r in ("phys_static", "phys_tls", "phys_cosmic", "phys_slow", "phys_coupler", "phys_mixed")}
BW_NUMS = " ".join(f"{J_LERS[r][f'window_{BW[r]}']['all' if r == 'phys_static' else 'drift']:.4f} {BW[r]}" for r in BW)
BW_SPECS = ";".join(f"lers.{r}.window_{BW[r]}.{'all' if r == 'phys_static' else 'drift'}|.4f;argmin:lers.{r}.window_{{w}}.{'all' if r == 'phys_static' else 'drift'}" for r in BW)
e("best WindowDEM (window) & " + " & ".join(f"{J_LERS[r][f'window_{BW[r]}']['all' if r == 'phys_static' else 'drift']:.4f} ({BW[r]})" for r in BW),
  (BW_NUMS, "R10J", None, BW_SPECS))

# Results text
e("Live $-$ frozen on drift sub-steps is $+0.00003$ ($p=0.06$) on \\reg{phys\\_tls}, $+0.00309$",
  ("0.00003 0.06", "R10J", None, "H1.regimes.phys_tls.diff|+.5f;H1.regimes.phys_tls.mcnemar_p|.1g"),
  ("0.00309 0.008", "R10J", None, "H1.regimes.phys_cosmic.diff|+.5f;H1.regimes.phys_cosmic.mcnemar_p|.1g"))
e("($p=0.008$\\weak) on \\reg{phys\\_cosmic} and $+0.00014$ ($p=0.2$) on \\reg{phys\\_mixed}. The event-level sign",
  ("0.00014 0.2", "R10J", None, "H1.regimes.phys_mixed.diff|+.5f;H1.regimes.phys_mixed.mcnemar_p|.1g"))
e("frozen is better in all 10 burst events ($p=0.002$); on \\reg{phys\\_tls}",
  ("10 0.002", "R10J", None, "H1.regimes.phys_cosmic.sign_losses|d;H1.regimes.phys_cosmic.sign_p|.1g"))
e("it is a tie (299 against 282 events, $p=0.5$). On \\reg{phys\\_mixed} the sign test favours live ($384/291$,",
  ("299 282 0.5 384 291", "R10J", None,
   "H1.regimes.phys_tls.sign_wins|d;H1.regimes.phys_tls.sign_losses|d;H1.regimes.phys_tls.sign_p|.1g;H1.regimes.phys_mixed.sign_wins|d;H1.regimes.phys_mixed.sign_losses|d"))
e("$p=\\Eb{4}{4}$) while the pooled comparison does not ($p=0.2$); \\cref{sec:phys-explor} traces this to",
  ("0.0004 0.2", "R10J", None, "H1.regimes.phys_mixed.sign_p|.1g;H1.regimes.phys_mixed.mcnemar_p|.1g"))
e("$+0.00003$ above frozen (\\E{7}{6}).",
  ("0.00003 7e-06", "R10J", None, "paired.phys_static|adaptive-stale live vs frozen.diff|+.5f;paired.phys_static|adaptive-stale live vs frozen.mcnemar_p|.1g"))
e("pre-registered: $-0.00126$, \\E{9}{14} against both. On \\reg{phys\\_cosmic} it is lower than both",
  ("0.00126 9e-14 0.00269", "R10J", None,
   "H2.tests.phys_mixed|calibrated MWPM.diff|+.5f;H2.tests.phys_mixed|calibrated MWPM.mcnemar_p|.1g;H2.tests.phys_cosmic|calibrated MWPM.diff|+.5f"))
e("($-0.00269$) but the regime contributes only 747 drift sub-steps pooled over five seeds, and at drift LER",
  ("747", "R10J", None, "census.phys_cosmic.drift|d"))
e("0.42 nearly every shot fails for every decoder, so the comparison does not reach significance ($p=0.1$; the",
  ("0.42 0.1", "R10J", None, "lers.phys_cosmic.adaptive_stale_live_ens.drift|.2f;H2.tests.phys_cosmic|calibrated MWPM.mcnemar_p|.1g"))
e("best window is 500, whose weights barely leave the calibrated ones). Oracle MWPM is 0.4158 against",
  ("500 0.4158", "R10J", None, "argmin:lers.phys_cosmic.window_{w}.drift;lers.phys_cosmic.mwpm_oracle.drift|.4f"))
e("calibrated 0.4199: under these bursts the matching decoder itself is the bottleneck, not its weights.",
  ("0.4199", "R10J", None, "lers.phys_cosmic.mwpm_cal.drift|.4f"))
e("On the static device every ensemble is significantly better than calibrated MWPM (0.0022): static-wide",
  ("0.0022", "R10J", None, "lers.phys_static.mwpm_cal.all|.4f"))
e("0.0016 ($-0.00060$, \\E{4}{79}), static-nominal 0.0017 ($-0.00051$, \\E{4}{56}), adaptive-wide",
  ("0.0016 0.00060 4e-79 0.0017 0.00051 4e-56", "R10J", None,
   "lers.phys_static.static_wide_live_ens.all|.4f;H3.models.static_wide.diff|+.5f;H3.models.static_wide.mcnemar_p|.1g;lers.phys_static.static_nominal_live_ens.all|.4f;H3.models.static_nominal.diff|+.5f;H3.models.static_nominal.mcnemar_p|.1g"))
e("0.0016 ($-0.00063$, \\E{2}{87}) and adaptive-stale 0.0016 ($-0.00064$, \\E{9}{91}). The $p=0.003$-trained",
  ("0.0016 0.00063 2e-87 0.0016 0.00064 9e-91", "R10J", None,
   "lers.phys_static.adaptive_wide_live_ens.all|.4f;H3.models.adaptive_wide.diff|+.5f;H3.models.adaptive_wide.mcnemar_p|.1g;lers.phys_static.adaptive_stale_live_ens.all|.4f;H3.models.adaptive_stale.diff|+.5f;H3.models.adaptive_stale.mcnemar_p|.1g"))
e("(\\reg{phys\\_static} on all sub-steps; largest $p=\\Eb{9}{14}$: \\reg{phys\\_tls} \\E{8}{129}, \\reg{phys\\_slow}",
  ("8e-129 7e-69", "R10J", None,
   "paired.phys_tls|adaptive-stale live vs calibrated MWPM.mcnemar_p|.1g;paired.phys_slow|adaptive-stale live vs calibrated MWPM.mcnemar_p|.1g"))
e("\\E{7}{69}, \\reg{phys\\_coupler} \\E{4}{24}, \\reg{phys\\_mixed} \\E{9}{14}). They beat oracle MWPM on",
  ("4e-24 9e-14", "R10J", None,
   "paired.phys_coupler|adaptive-stale live vs calibrated MWPM.mcnemar_p|.1g;paired.phys_mixed|adaptive-stale live vs calibrated MWPM.mcnemar_p|.1g"))
e("\\reg{phys\\_tls} ($-0.00085$, \\E{5}{78}) and \\reg{phys\\_mixed} ($-0.00089$, \\E{2}{7}) and are not separable",
  ("0.00085 5e-78 0.00089 2e-07", "R10J", None,
   "paired.phys_tls|adaptive-stale live vs oracle MWPM.diff|+.5f;paired.phys_tls|adaptive-stale live vs oracle MWPM.mcnemar_p|.1g;paired.phys_mixed|adaptive-stale live vs oracle MWPM.diff|+.5f;paired.phys_mixed|adaptive-stale live vs oracle MWPM.mcnemar_p|.1g"))
e("from it on \\reg{phys\\_cosmic} ($+0.00139$, $p=0.4$). The best WindowDEM is $w=500$ in every regime but",
  ("0.00139 0.4 500", "R10J", None,
   "paired.phys_cosmic|adaptive-stale live vs oracle MWPM.diff|+.5f;paired.phys_cosmic|adaptive-stale live vs oracle MWPM.mcnemar_p|.1g;argmin:lers.phys_tls.window_{w}.drift"))
e("\\reg{phys\\_coupler} ($w=10000$) --- after 500 shots its weights barely leave the calibrated ones --- and",
  ("10000", "R10J", None, "argmin:lers.phys_coupler.window_{w}.drift"))
e("long windows hurt under bursts: on \\reg{phys\\_cosmic} $w=10000$ has drift LER 0.4631 against 0.4199 for",
  ("0.4631 0.4199", "R10J", None, "lers.phys_cosmic.window_10000.drift|.4f;lers.phys_cosmic.window_500.drift|.4f"))

# Exploratory (post hoc)
e("locates the live fingerprint's cost in \\reg{phys\\_cosmic}: live $-$ frozen is $+0.00050$",
  ("0.00050", "R10J", None, "exploratory.phys_cosmic|bin0.diff|+.5f"))
e("($p=0.8$) in the strongest bin [0.05, 0.2], where both decoders fail about half the shots (0.4919 against",
  ("0.8 0.4919", "R10J", None, "exploratory.phys_cosmic|bin0.mcnemar_p|.1g;exploratory.phys_cosmic|bin0.ler_live|.4f"),
  ("0.05 0.2", "N10D", None, r"youngest active burst: \[0.05, 0.2\]"))
e("0.4914) and nothing separates, $+0.00420$ ($p=0.06$) in [0.0125, 0.05) and $+0.00686$ (\\E{5}{7})",
  ("0.4914 0.00420 0.06 0.00686 5e-07", "R10J", None,
   "exploratory.phys_cosmic|bin0.ler_frozen|.4f;exploratory.phys_cosmic|bin1.diff|+.5f;exploratory.phys_cosmic|bin1.mcnemar_p|.1g;exploratory.phys_cosmic|bin2.diff|+.5f;exploratory.phys_cosmic|bin2.mcnemar_p|.1g"),
  ("0.0125 0.05", "N10D", None, r"youngest active burst"))
e("in the weakest bin [0.005, 0.0125) --- the one bin where decoding still works (0.2028 against 0.1959) ---",
  ("0.005 0.0125", "N10D", None, r"youngest active burst"),
  ("0.2028 0.1959", "R10J", None, "exploratory.phys_cosmic|bin2.ler_live|.4f;exploratory.phys_cosmic|bin2.ler_frozen|.4f"))
e("but weaker: $-0.00104$ ($p=0.6$), $+0.00349$ ($p=0.1$) and $+0.00339$",
  ("0.00104 0.6 0.00349 0.1 0.00339", "R10J", None,
   "exploratory.phys_mixed|bin0.diff|+.5f;exploratory.phys_mixed|bin0.mcnemar_p|.1g;exploratory.phys_mixed|bin1.diff|+.5f;exploratory.phys_mixed|bin1.mcnemar_p|.1g;exploratory.phys_mixed|bin2.diff|+.5f"))
e("(0.02\\weak). On the TLS-only drift sub-steps of \\reg{phys\\_mixed} (1\\,017\\,000 shots) live and frozen are",
  ("0.02 1017000", "R10J", None, "exploratory.phys_mixed|bin2.mcnemar_p|.1g;exploratory.phys_mixed|tls_only.shots|d"))
e("indistinguishable ($+0.00002$, $p=0.5$), and the apparent conflict on \\reg{phys\\_mixed} --- pooled McNemar",
  ("0.00002 0.5", "R10J", None, "exploratory.phys_mixed|tls_only.diff|+.5f;exploratory.phys_mixed|tls_only.mcnemar_p|.1g"))
e("the errors (bin 0 LER 0.4928 against 0.0049 on TLS-only sub-steps) and favour frozen, while the sign test",
  ("0.4928 0.0049", "R10J", None, "exploratory.phys_mixed|bin0.ler_live|.4f;exploratory.phys_mixed|tls_only.ler_live|.4f"))
e("is dominated by the 992 TLS events, which",
  ("992", "R10J", None, "census.phys_mixed.events.tls|d"))
e("favour live ($366/276$, $p=\\Eb{4}{4}$); the 10 burst events slightly favour frozen ($4/6$) and the 27",
  ("366 276 0.0004 10 4 6 27", "R10J", None,
   "exploratory.phys_mixed|sign_by_kind.tls.live_better|d;exploratory.phys_mixed|sign_by_kind.tls.frozen_better|d;exploratory.phys_mixed|sign_by_kind.tls.p|.1g;exploratory.phys_mixed|sign_by_kind.burst.n|d;exploratory.phys_mixed|sign_by_kind.burst.live_better|d;exploratory.phys_mixed|sign_by_kind.burst.frozen_better|d;exploratory.phys_mixed|sign_by_kind.coupler.n|d"))
e("coupler events are undecided ($14/9$, $p=0.4$).",
  ("14 9 0.4", "R10J", None, "exploratory.phys_mixed|sign_by_kind.coupler.live_better|d;exploratory.phys_mixed|sign_by_kind.coupler.frozen_better|d;exploratory.phys_mixed|sign_by_kind.coupler.p|.1g"))

e("consistent with the 2000-shot live window lagging the decaying burst",
  ("2000", "TD", None, r"seen \(\$W=2000\$\)"))

# Limitations of the physics evaluation
e("The cosmic-ray impact rate is accelerated (0.02 per 2000-shot step against roughly one per ten",
  ("0.02 2000", "PN", None, r"accelerated \(0\.02 per 2000-shot step\)"))
e("0.42 in \\reg{phys\\_cosmic} every decoder is near failure, which hides differences.",
  ("0.42", "R10J", None, "lers.phys_cosmic.adaptive_stale_live_ens.drift|.2f"))
e("pairs; \\reg{phys\\_cosmic} contributes only 747 drift sub-steps pooled.",
  ("747", "R10J", None, "census.phys_cosmic.drift|d"))

# Conclusion
e("$p=0.0015$ with per-qubit heterogeneity, slow drift, TLS fluctuators, accelerated cosmic-ray bursts and",
  ("0.0015", "N10D", None, r"SI1000 strength p = CHOSEN_SI1000_P = 0.0015"))
e("calibrated to the static device (0.0016 to 0.0017 against 0.0022) and is below or statistically",
  ("0.0016 0.0017 0.0022", "R10J", None,
   "lers.phys_static.static_wide_live_ens.all|.4f;lers.phys_static.static_nominal_live_ens.all|.4f;lers.phys_static.mwpm_cal.all|.4f"))

# ------------------------------------------------------------------ engine
def norm(line):
    line = line.replace("−", "-").replace("**", "").replace("\\,", "").replace("{,}", "")
    line = re.sub(r"(?<=\d)\s(?=\d{3}\b)", "", line)  # 300 000 -> 300000
    return line


def has(line, tok):
    if any(c in tok for c in "\\^{"):  # raw regex token (LaTeX source)
        return re.search(tok, line) is not None
    t = tok.lstrip("+-")
    return re.search(r"(?<![\d.])" + re.escape(t) + r"(?![\d])", line) is not None


def find_row(path, section, row):
    lines = open(os.path.join(ROOT, path), encoding="utf-8").read().split("\n")
    start = 0
    for sec in ((section,) if isinstance(section, str) else (section or ())):
        hits = [i for i, l in enumerate(lines) if i >= start and re.search(sec, l)]
        if not hits:
            return None, None, f"section /{sec}/ not found"
        start = hits[0]
    for i in range(start, len(lines)):
        if re.search(row, lines[i]):
            return i + 1, lines[i], None
    return None, None, f"row /{row}/ not found"


def paper_line(fragment, paper):
    frag = re.split(r" \(and its| \(also Appendix| \(the largest", fragment)[0]
    if frag.startswith("[Table"):
        return None
    idx = paper.find(frag)
    if idx < 0:
        return -1
    return paper.count("\n", 0, idx) + 1


def main():
    paper = open(PAPER, encoding="utf-8").read()
    out = ["# Number sources for `drift_whitepaper.tex`", "",
           "Generated by `whitepaper/check_numbers.py`. Each row: where the numbers are in the paper "
           "(`drift_whitepaper.tex` line, or the appendix table), the numbers, and the source file:line they were "
           "copied from. The script re-reads every source line and checks that each number occurs on it; the "
           "`check` column is the result of that re-check. Signs and Markdown bold are ignored when matching; "
           "p-values such as 3e-247 are written $3\\times10^{-247}$ in the paper.", "",
           "Figures are not listed here. `whitepaper/make_figs.py` draws the result figures from "
           "`results/exp6/report6_data.json`, `results/exp4/curves/*.json`, `results/exp9/localisation_baseline.json` "
           "and the per-step CSVs in `results/exp3/d5/`, and asserts that the values it recomputes equal the values "
           "in the notes. The live$-$frozen labels in the live-vs-frozen figure are differences of two ensemble LERs "
           "from `report6_data.json` and equal the corresponding cells of `notes/07_ensembles.md` section 3. "
           "The regimes, example-shot and fingerprint figures are illustrations drawn from the simulator "
           "(`drift.py`, `circuits.py`; Stim 1.16.0, PyMatching 2.4.0) with the seeds of "
           "`tech_doc_src/figs.py`; they show no results. Round-10 numbers (the physics-based drift "
           "evaluation) are checked directly against `results/exp10/report10_data.json`: the source cell "
           "names the JSON path and printf format each number was read from, and `argmin:` cells verify the "
           "best WindowDEM window as the arg-min over the JSON window LERs.", "",
           "| # | paper | numbers | source (file:line) | check |", "|---|---|---|---|---|"]
    bad = 0
    for k, (frag, checks) in enumerate(E, 1):
        pl = paper_line(frag, paper)
        where = (f"L{pl}: " if pl and pl > 0 else ("NOT FOUND: " if pl == -1 else "")) + \
            frag.replace("\n", " ").replace("|", "\\|")[:70]
        if pl == -1:
            bad += 1
        for nums, f, sec, row in checks:
            if f == "R10J":
                toks, specs = nums.split(), row.split(";")
                try:
                    missing = [t for t, sp in zip(toks, specs) if r10_get(sp).lstrip("+-") != t.lstrip("+-")]
                    if len(toks) != len(specs):
                        missing = toks
                except (KeyError, ValueError, IndexError):
                    missing = toks
                ok = "ok" if not missing else "FAIL: " + ", ".join(missing)
                if ok != "ok":
                    bad += 1
                shown = nums.replace("\\", "").replace("^", "")
                out.append(f"| {k} | {where} | {shown} | `{F[f]}:{row}` | {ok} |")
                where = "″"
                continue
            ln, text, err = find_row(F[f], sec, row)
            if err:
                ok = "FAIL (" + err + ")"
            else:
                missing = [t for t in nums.split() if not has(norm(text), t)]
                ok = "ok" if not missing else "FAIL: missing " + ", ".join(missing)
            if ok != "ok":
                bad += 1
            shown = nums.replace("\\", "").replace("^", "")
            out.append(f"| {k} | {where} | {shown} | `{F[f]}:{ln}` | {ok} |")
            where = "″"
    out += ["", f"{len(E)} paper locations, {sum(len(c) for _, c in E)} source rows, {bad} failures."]
    open(os.path.join(ROOT, "whitepaper", "number_sources.md"), "w").write("\n".join(out) + "\n")
    print(f"{len(E)} entries, {bad} failures")
    for l in out:
        if "FAIL" in l or "NOT FOUND" in l:
            print(l)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
