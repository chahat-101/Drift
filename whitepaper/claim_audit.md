# Claim audit: `whitepaper/drift_whitepaper.tex` / `.pdf` (18 pp., built 2026-11-15)

Independent audit, 2026-11-18. The paper was not edited. Everything below was re-derived from primary
sources rather than from `whitepaper/number_sources.md`.

**How the numbers were checked.** The MWPM baseline per-shot bits (static, oracle, WindowDEM ×5) are not in
`results/`. They were extracted read-only from `drift_data.tgz` into a scratch directory, together with the
single-seed and h64/l6 bits under `results/exp2,3,4/**/_errors.npz`. Phase masks were rebuilt from the noise-state
columns (`scale`, `n_hot`) of `results/exp{2,3}/…_st{0..4}_s0.csv`, independently of the saved `phase_*` masks.
Every LER, Δ and exact two-sided McNemar *p* in Tables 1, 2, 3 and 6–8 and in the text of §5–§6 was then
recomputed from those bits. The McNemar *p* was computed in log space, using the formula of §4 Statistics.
The ensembles used `results/exp6–9/d*/st*/ens_*_errors.npz` (members m0–m2 equal the saved single-seed bits;
checked for adaptive-stale seed 0). NF hit rates were recomputed from `results/exp9/noise/noise_auxfilm_s*.npz`.
The scratch scripts are in `/tmp/claude-1000/-home-chahat-projects-drift/d9aea361-3978-4202-ad28-dc2c4b8ea26a/scratchpad/audit/`
(`lib.py`, `t1.py`…`t8.py`). A subagent checked all 17 references against the arXiv export API, Crossref and full
texts. A second subagent checked the method descriptions against the code, logs and manifests (§M below).

Verdicts: **verified** = matches the source to the printed precision; **wrong** = contradicted by the source;
**overstated** = the number is right but the wording claims more than the evidence; **unsupported** = no
source in the repo or literature; **unverifiable** = the source could not be checked.

---

## Table of claims

### Abstract (p. 1)

| # | claim (quote) | page/§ | source checked | verdict | fix |
|---|---|---|---|---|---|
| 1 | "a fingerprint: the firing rate of every detector over the last 2000 shots it has decoded. It receives no calibration data." | p1 Abstract | code (§M) | verified | — |
| 2 | "circuit-level simulation (p=0.003) with four drift regimes" | p1 | `circuits.py`, manifests (§M) | verified | — |
| 3 | "three-seed ensemble … d=5 … drift-step LER of 0.0457 on held-out large bursts" | p1 | recomputed from `exp6/d5/st*/ens_adaptive_stale_d5_long_errors.npz` | verified (0.04569) | — |
| 4 | "against 0.0980 for static matching, 0.0896 for the best sliding-window reweighted matching and 0.0426 for matching with the true noise" | p1 | recomputed from `exp3/d5/*_errors.npz` (static, window_2000, oracle) | verified | — |
| 5 | "held-out large bursts" | p1 | notes/report3, §5.4, §6.3–6.4 | **overstated** | `bursts_big` is outside the *training distribution*. It was not held out from *development*: the 60k→300k decision (report 3: "classical baselines win back … bursts_big"), the N/NF decision rule, the B/C rules and the h64/l6 adoption bar were all defined on `bursts_big`. Say "out-of-distribution large bursts" and disclose its use in model selection (summary item S4). |
| 6 | "With the same weights and a frozen nominal fingerprint the rate is 0.0576 (paired McNemar p=3×10⁻²⁴⁷)" | p1 | recomputed: 0.05763, p = 3.1e-247 | verified | — |
| 7 | "so the gain comes from adaptation" | p1 | recomputed | **overstated** | The frozen model (0.0576) already beats static MWPM (0.0980) and best WindowDEM (0.0896). Of the 0.0523 gain over static MWPM, 0.0404 (77 %) remains with the fingerprint frozen. Only 0.0119 (23 %) is adaptation. Adaptation explains the gain over a *non-adaptive neural decoder* (static-wide ensemble 0.0564), not the gain over matching. |
| 8 | "An auxiliary head … beats a training-free fingerprint score on in-distribution bursts but not, at the event level, on held-out bursts." | p1 | `exp9/localisation_baseline.{md,json}`, NF hits recomputed | verified | ("held-out": see #5) |
| 9 | "Three extensions with decision rules fixed in advance (multiscale and pair-covariance fingerprints, a noise-map head with FiLM conditioning) failed those rules." | p1 | Table 3, recomputed | **wrong (count)** | Four variants were judged (B, C, N, NF), and N has no FiLM. Say "Four variants … (multiscale and pair-covariance fingerprints, a noise-map head with and without FiLM conditioning)". Whether the rules were truly fixed in advance: see §M. |
| 10 | "All results are from simulation at d≤5." | p1 | — | verified | — |

### 1 Introduction (pp. 1–2)

| # | claim | page/§ | source checked | verdict | fix |
|---|---|---|---|---|---|
| 11 | "Energy-relaxation times … fluctuate in frequency and time, and TLS defects are responsible for the largest fluctuations [Klimov]" | p1 §1 | arXiv:1809.01043 abstract | verified | — |
| 12 | "bursts that start at a localised impact, spread across a 26-qubit array and decay with a time constant of about 25 ms, occurring about every 10 s [McEwen]" | p1 §1 | arXiv:2104.05219 full text ("26 qubit subset", "∼25 ms", "on average every 10 s", "initial localised impact") | verified | — |
| 13 | "recalibration cannot follow changes this fast" | p1 §1 | none cited | unsupported (plausible) | Add a citation, or soften: "recalibration, which takes [minutes/hours], cannot follow millisecond-scale changes". |
| 14 | Contribution (a): "its ensemble beats static and tuned sliding-window MWPM on all four regimes, and MWPM with the true noise on moderate bursts and hot-qubit drift" | p1 §1 | Table 1 recomputed | numbers verified; **overstated framing** | "Tuned" means a window chosen *post hoc on the test streams* (disclosed in §4). The win over oracle MWPM is not evidence of better adaptation: on *nominal* steps the same network is already 35–39 % below MWPM (0.0021 vs 0.0033–0.0034), while on drift it is only 14 % below oracle on `bursts`, 29 % on `hot_qubit` and +7 % above it on `bursts_big`. Its advantage over MWPM *shrinks* under drift. Add one sentence saying so (summary item S2). |
| 15 | Contribution (b): "the gain on localised drift is adaptation, and … uniform scale drift gets almost none" | p2 §1 | Table 2 recomputed | overstated (as #7) | "the gain over a non-adaptive network trained on the same data is adaptation" |
| 16 | Contributions (c), (d) | p2 §1 | §5.5, §6 | verified | — |

### 2 Background (pp. 2–3)

| # | claim | page/§ | source checked | verdict | fix |
|---|---|---|---|---|---|
| 17 | "rotated surface code … d² data qubits and d²−1 measure qubits" | p2 §2 | standard; 49 qubits at d=5 matches `aux_qubits` | verified | — |
| 18 | "D_{a,t}=m_{a,t}⊕m_{a,t−1}" | p2 §2 | Stim generated circuits | verified (simplified: the first layer compares with the reset value and the last with data-qubit parities; acceptable) | — |
| 19 | "N=r(d²−1) … N=24 at d=3 and N=120 at d=5" | p2 §2 | (d²−1)/2 + (r−1)(d²−1) + (d²−1)/2 = r(d²−1); see §M for the Stim count | verified | — |
| 20 | "w_e=ln((1−p_e)/p_e)" | p2 §2 | standard; PyMatching docs | verified | — |
| 21 | "PyMatching 2 with sparse blossom [Higgott & Gidney]; Stim [Gidney]" | p2 §2 | arXiv 2303.15933, 2103.02202 | verified | — |
| 22 | Spitz et al.: "edge weights can be estimated from the correlations of detection events, without an error model, and … works when the noise varies slowly compared with the time needed to collect those statistics" | p2 §2 | arXiv:1712.02360 abstract + text ("in the absence of an error model"; condition τ_env > δt/p̄) | verified (paraphrase) | Optionally quote the condition τ_env > δt/p̄. Fix the bib metadata (#R2). |
| 23 | Varbanov et al. showed NN beats matching "on simulated and experimental small surface codes" | p2 §2 | arXiv:2307.03280 abstract | verified | — |
| 24 | Lange et al.: GNN under circuit-level noise | p2 §2 | arXiv:2307.01241 | verified | — |
| 25 | AlphaQubit: recurrent transformer, pre-trained on simulated data, fine-tuned on Sycamore data, beats SOTA at d=3,5 | p2 §2 | arXiv:2310.05900 + Nature | verified | Fix the bib title (#R1). |
| 26 | AlphaQubit 2: larger surface and colour codes, real-time throughput | p2 §2 | arXiv:2512.07737 ("surface code to distance 11 … color code to distance 9 … faster than 1 µs per cycle") | verified | — |
| 27 | Yan et al.: "data scale matters more than architectural complexity at near-term distances" | p2 §2 | arXiv:2605.12046 ("driven more by data scale than architectural complexity") | verified | — |
| 28 | Stein et al.: graph encoder of calibration data, FiLM-conditioned CNN, IBM repetition codes, generalises to new chains and calibrations days later | p2 §2 | arXiv:2601.16123 | verified | — |
| 29 | QAdapt: continual training with EWC across noise environments | p2 §2 | arXiv:2607.28422 | verified | — |
| 30 | Peled et al.: per-syndrome MWPM weights with a GNN and a transformer | p3 §2 | arXiv:2601.00242 | verified | — |
| 31 | Chamberland et al.: local pre-decoder + global matching; infer weights from syndrome statistics | p3 §2 | arXiv:2604.12841 | verified | — |
| 32 | "Hoyt et al. show on IBM **and Google** data that syndrome statistics drift away from the stationary DEM" | p3 §2 | arXiv:2609.29018 abstract: drift shown "On IBM quantum hardware"; for Google only "a similar hardware-model mismatch" | **overstated** | "show on IBM hardware that syndrome statistics drift across acquisition windows away from the stationary DEM, report a similar mismatch in Google surface-code data, and use the mismatch for post-selection" |
| 33 | "below-threshold surface-code memories on Google's Willow processor [Google 2025]" | p3 §2 | Nature abstract names Willow (arXiv v1 does not) | verified | — |

### 3 Method (pp. 3–6) — see §M for the code-level checks

| # | claim | page/§ | source checked | verdict | fix |
|---|---|---|---|---|---|
| 34 | Window estimate: "2000×5 µs = 10 ms plus reset and readout overhead between shots, so of order 10 ms, comparable to the 25 ms decay" | p4 §3.1 | `external_sources.md` (author-supplied inputs) | **unsupported** | The per-shot overhead (reset, readout, any between-shot wait) is not bounded. If it is ≥5 µs/shot the window is ≥20 ms; tens of µs would put it well above the 25 ms decay. Either give a source for the per-shot repetition time or say "≥10 ms, depending on the per-shot overhead". It is labelled a rough estimate, so this is minor. |
| 35 | "nominal mean firing rate of 4.27% (d=5)" | p4 §3.3 | Stim simulation of 400k nominal d=5 shots (methods agent): 4.275 %. Analytic DEM (`localisation_baseline.json`): 4.279 %. Nominal test fingerprint window: 4.347 %. | verified (rounding borderline) | 4.28 % is the better rounding of both estimates; trivial. The only written source is the old `tech_doc_src/drift.tex`, so cite a result file. |
| 36 | "binomial standard error of one rate over 2000 shots is about 10% of the rate" | p4 §3.3 | √(0.0428·0.957/2000)/0.0428 = 10.6 % | verified | — |
| 37 | "after four layers the receptive field (9³) covers the whole d=5 grid" | p5 §3.4 | 1+4·2=9 ≥ 6 | verified | — |
| 38 | "The noise map is read out per step as the mean over the step's shots" | p6 §3.7 | `localisation_baseline.md`, `noise_eval.py` | verified | — |

### 4 Experimental setup (pp. 6–7)

| # | claim | page/§ | source checked | verdict | fix |
|---|---|---|---|---|---|
| 39 | "report, for each regime and phase, the best window chosen after the fact. This favours WindowDEM." | p6 §4 | recomputed best windows; they match Table 6/7 window labels | verified (disclosed) | — |
| 40 | "bursts has 29 onset steps (33 bursts; four start directly after another), bursts_big 29, telegraph 21 and hot_qubit 5" | p6 §4 | rebuilt masks from CSVs, d=5 *and* d=3 streams: 29/29/5/21. Events: 33 (localisation json). | verified | — |
| 41 | Exact two-sided McNemar formula | p7 §4 | standard binomial form | verified | — |
| 42 | "For single models we test each seed and report the largest p" | p7 §4 | report6 tables | verified | — |
| 43 | "McNemar measures shot noise for fixed models" | p7 §4 | — | verified, but **incomplete** | McNemar also treats all shots as independent across *events*. The regime-level unit of replication is the burst/defect event (29 runs per burst regime, 21 telegraph runs, **5** hot-qubit runs, all at the *same* two central qubits). Event-level sign tests (recomputed, d=5): adaptive-stale ens. vs oracle on `bursts` 26/3 runs (p=1.5e-5); vs static-wide on `bursts_big` 28/0 (p=7.5e-9); live vs frozen `bursts_big` 27/1 (p=2.2e-7); but on `hot_qubit` every comparison is 5/0 (p=0.06), and `bursts_big` vs oracle is 9 better / 20 worse (p=0.06). §5.5 already argues "the event-level test is the honest one"; apply the same caveat to §5. |
| 44 | Pre-set decision rules: "judged by rules written before each run … We applied the rules as written" | p7 §4 | rule text in `results/exp7,8,9/report*_tables.md:7`, `report9.py:110-132`; git history | wording verified; **timing unverifiable** | The paper's wording matches the recorded rules exactly, and the verdicts follow from them (re-scanned in #84–87). But the rule text exists only in post-run report generators and notes, all first committed in bulk commit a3eaef5 (2026-11-14 23:45), after round 9 finished (`run9.out` 23:03 the same day). Nothing in the repo predates the runs. Either point to dated evidence (e.g. a pre-run commit, email or timestamped note) or write "rules recorded with each round's launch scripts; the repository history cannot show they predate the runs". The same applies to the ensemble rule "fixed before the ensemble runs" (§3.6). |

### 5.1 Main comparison and Table 1 (pp. 7–8)

| # | claim | page/§ | source checked | verdict | fix |
|---|---|---|---|---|---|
| 45 | Table 1, all 64 LERs (d=5 and d=3: static, best WindowDEM, oracle, 3 ensembles) | p8 T1 | recomputed | verified (all 64) | — |
| 46 | Table 1, all 24 "single seeds mean ± s.d." | p8 T1 | recomputed from m0–m2 (ddof=1) | verified (all 24) | — |
| 47 | Table 1 Δ vs static-wide ens., d=5: −0.0053 (4e-142), −0.0107 (7e-158), −0.0025 (1e-298), −0.0010 (3e-29) | p8 T1 | recomputed: 3.6e-142, 7.49e-158, 9.7e-299, 2.9e-29 | verified | — |
| 48 | Table 1 Δ vs oracle, d=5: −0.0027 (5e-27), +0.0031 (3e-14), −0.0030 (1e-224), −0.0066 (<1e-300) | p8 T1 | recomputed: 5.1e-27, 2.6e-14, 1.2e-224, 1.2e-388 | verified | — |
| 49 | Table 1 Δ, d=3 (8 cells) | p8 T1 | recomputed: <1e-300 ×3, 7.3e-102; 7.4e-35, 3.2e-30, 2.6e-66, 4.8e-76 | verified | — |
| 50 | "beats static MWPM and the best WindowDEM in every regime: all eight paired tests have p≤2×10⁻³⁰²; the largest being hot_qubit vs best WindowDEM; the other seven are too small to represent in double precision" | p7 §5.1 | recomputed: hot_qubit vs W(10000) 1.6e-302; the others 1e-368 … 1e-1847 | verified | — |
| 51 | "It beats oracle MWPM on bursts, hot_qubit and telegraph (on telegraph through decoder quality)" | p7 §5.1 | recomputed | numbers verified; **overstated by omission** | Decoder quality also explains much of the `bursts`/`hot_qubit` wins (see #14). The static-wide ensemble, with no fingerprint, also beats oracle on `hot_qubit` (0.0098 vs 0.0103) and `telegraph`. Add: "On nominal steps the network is already ≈36 % below MWPM, so beating oracle MWPM reflects decoder quality as well as adaptation." |
| 52 | "On held-out bursts_big, oracle MWPM remains better by +0.0031 (3e-14)" | p7 §5.1 | recomputed | verified | ("held-out": #5) |
| 53 | "the adaptive-stale ensemble is lower [than static-wide] on drift steps of every regime at both distances" | p7 §5.1 | recomputed (8/8, all p ≤ 3e-29) | verified | — |
| 54 | adaptive-wide − adaptive-stale: bursts +0.0001 (p=0.6), telegraph −0.0001 (0.37), bursts_big +0.0027 (3.1e-19), hot_qubit −0.0002 (6.3e-6) | p7 §5.1 | recomputed | verified | — |
| 55 | "At d=3 the same ordering holds against static MWPM, the best WindowDEM and static-wide, but the adaptive ensembles trail oracle MWPM on both burst regimes" | p8 §5.1 | recomputed (adaptive-wide vs oracle d=3 bursts 0.0713 vs 0.0667, bursts_big 0.2014 vs 0.1937) | verified | — |
| 56 | d=3 onset vs oracle: +0.0219 (2e-90) bursts; +0.0224 (7e-59) bursts_big; later ties on bursts +0.0006 (p=0.2) | p8 §5.1 | recomputed: +0.02193 (2.3e-90); +0.02243 (7.3e-59); +0.00065 (0.16) | verified | (On `bursts_big` later the gap is +0.0023, p=0.0014, weak; it is not mentioned but not claimed either.) |
| 57 | "Most of that remaining gap is at burst onset" | p8 §5.1 | onset share of the drift-step gap: bursts 0.0219·29/132 = 0.0048 of 0.0053; bursts_big 0.0224·29/116 = 0.0056 of 0.0073 | verified | — |
| 58 | Table 1 caption: "d=3 models were trained for 5× fewer steps" | p8 | 60k vs 300k | verified | — |

### 5.2 The gain is adaptation, Table 2 (pp. 8–9)

| # | claim | page/§ | source checked | verdict | fix |
|---|---|---|---|---|---|
| 59 | Section title "The gain is adaptation" | p8 §5.2 | — | overstated (as #7) | "The gain over a non-adaptive network is adaptation" or "What the fingerprint adds" |
| 60 | Table 2, all 16 cells (Δ and p) | p8 T2 | recomputed: e.g. d5 adaptive-wide telegraph −0.0002 (p=0.0018 → "0.002 w"), adaptive-stale telegraph −0.0002 (1.8e-4) | verified (all 16) | — |
| 61 | "On localised drift the live fingerprint lowers the drift-step LER in every case at p≤1×10⁻²⁰⁹" | p8 §5.2 | largest p = 1.2e-209 (d=5 adaptive-stale, bursts) | **wrong (bound)** | 1.2e-209 > 1e-209. Write "p ≤ 2×10⁻²⁰⁹". |
| 62 | "averaged over single seeds on bursts, adaptive-wide frozen is +0.0253 above static-wide (p<10⁻³⁰⁰ in each of the 3 seeds)" | p8 §5.2 | recomputed seed-paired: +0.0248/+0.0247/+0.0266 (mean +0.0253), p ≤ 8e-618 | verified | — |
| 63 | "At d=5 the frozen ensembles are close to the static-wide ensemble (bursts_big drift 0.0576 vs 0.0564; paired +0.0012, p=0.0015, weak)" | p8 §5.2 | recomputed | **wrong for adaptive-wide** | True for adaptive-stale only. The frozen *adaptive-wide* ensemble is far above static-wide: bursts_big 0.0680 vs 0.0564 (+0.0116, p=2e-178), bursts +0.0039 (7e-77). Write "At d=5 the frozen adaptive-stale ensemble is close to …; frozen adaptive-wide is not (0.0680 vs 0.0564), so, as at d=3, adaptive-wide relies on the fingerprint." The adaptive-stale number is also only close on bursts_big/hot_qubit: on bursts frozen is +0.0010 worse (p=2e-7). |
| 64 | "On telegraph the live fingerprint contributes 0.0002 or less at both distances; … uniform scale drift needs no adaptation from this decoder" | p9 §5.2 | Table 2 recomputed | verified (magnitude); slight overstatement | At d=5 the live−frozen difference is significant (p=1.8e-4, adaptive-stale), so "needs little" is more accurate than "needs no". At the event level it is 15/5 (p=0.04). |
| 65 | After step, d=5 bursts_big: adaptive-wide live−frozen +0.0035 (5e-41); vs static MWPM +0.0021 (3e-11); adaptive-stale live−frozen +0.0006 (p=0.0001); vs static MWPM −0.0007 (p=0.006, weak) | p9 §5.2 | recomputed: +0.00353 (5.3e-41); +0.00212 (2.8e-11); +0.00060 (1.3e-4); −0.00067 (0.0056) | verified | — |
| 66 | "At d=3 … the adaptive-stale ensemble is +0.0026 above static MWPM after bursts_big events (3e-13)" | p9 §5.2 | recomputed +0.00264 (2.6e-13) | verified | — |
| 67 | d=3 nominal test: static-nominal 0.00536, adaptive-wide ties (0.00532, p=0.4), static-wide 0.00605 (1e-8), "about 13%" | p9 §5.2 | single-seed bits in `exp6/d3/st0/ens_*` (test keys): means 0.00536/0.00532/0.00605; per-seed largest p 0.39 and 1.0e-8; ratio 1.129 | verified | Say these are three-seed means of single models (the reader may assume ensembles). |
| 68 | "At d=5 after 300 000 steps this cost is not significant (largest p=0.2)" | p9 §5.2 | per seed: static-wide *below* static-nominal in all 3 seeds (0.00216/0.00226/0.00233 vs 0.00232/0.00281/0.00245; p = 0.04, 3e-9, 0.18) | **misleading** | There is no cost at d=5: static-wide is lower in every seed. Write "At d=5 … static-wide is in fact lower than static-nominal in all three seeds (largest p=0.2)". |

### 5.3 Phases (p. 9)

| # | claim | page/§ | source checked | verdict | fix |
|---|---|---|---|---|---|
| 69 | bursts: ties oracle at onset (−0.0001, p=0.9), beats it on later steps (−0.0034, 2e-34); bursts_big: trails at onset (+0.0065, 3e-14) and later (+0.0019, 2e-5) | p9 §5.3 | recomputed: −0.00010 (0.88); −0.00340 (1.8e-34); +0.00650 (2.6e-14); +0.00195 (2.3e-5) | verified | — |
| 70 | "even the best window is at 0.0116 on bursts against 0.0033 for static MWPM" (after step) | p9 §5.3 | recomputed W(1000) 0.0116, static 0.0033 | verified | — |
| 71 | Fig. 7 caption "29 onset steps each" | p10 | #40 | verified | — |

### 5.4 Training length (p. 9)

| # | claim | page/§ | source checked | verdict | fix |
|---|---|---|---|---|---|
| 72 | "After 60 000 steps every neural model lost to static MWPM on bursts_big drift (three-seed means 0.118 to 0.150, against 0.098)" | p9 §5.4 | `results/report3.md:10,384` (adaptive-wide 0.128, static-wide 0.150, adaptive-stale 0.118) | verified for the three wide-trained models | "every wide-trained model" (the range excludes static-nominal) |
| 73 | "static-nominal was worse than MWPM on nominal noise (0.00400 vs 0.00329)" | p9 §5.4 | `report3.md:9,94` | verified | — |
| 74 | "After 300 000 steps both failures are gone in every seed (static-nominal 0.00253±0.00026, largest p=1e-4)" | p9 §5.4 | `notes/06_d5_three_seeds.md:11,31,40`; `notes/04:187-191` | **wrong** | Failure 1 ("every neural model lost to static MWPM on bursts_big") is gone only for the wide-trained models. static-nominal is still far worse than static MWPM on bursts_big drift at 300k: 0.2159±0.0604 vs 0.0980, 0/3 seeds win (notes/06 l.40). At 60k it was 0.294, outside the stated "0.118 to 0.150" range. Write "every wide-trained model … both failures are gone for these models in every seed". |
| 75 | "Between 240 000 and 300 000 steps no seed-0 model improved on the nominal test set by more than 2 standard errors, but all five improved on bursts_big drift" | p9 §5.4 | `results/exp4/partC_decision.json` | verified | The five are adaptive-wide, static-nominal, adaptive-stale, static-wide and h64/l6, and drift is on **stream seed 0 only**. Name them and add "(stream seed 0)". |
| 76 | "so the d=5 numbers are a snapshot of an unconverged run" | p9 §5.4 | — | verified (honest) | — |

### 5.5 Localising the drift (pp. 9–10)

| # | claim | page/§ | source checked | verdict | fix |
|---|---|---|---|---|---|
| 77 | "A(q) … 9 to 24 detectors per qubit at d=5" | p10 §5.5 | `localisation_baseline.json` adjacency: min 9, max 24 | verified | — |
| 78 | "baseline scores 0.825 to 0.841 by phase and NF 0.975 to 1.000 by seed and phase" (bursts) | p10 §5.5 | NF recomputed from `noise_auxfilm_s*.npz` (0.975–1.000); baseline from json (0.825–0.841) | verified | — |
| 79 | "NF wins 20 bursts, loses 1 and ties 12 for every seed (p=2e-5); at onset it wins 17 or 18 events and loses none (p≤2e-5)" | p10 §5.5 | json counts; sign test 2·22/2²¹ = 2.1e-5 recomputed | verified | — |
| 80 | "On hot_qubit every method … scores 1.000" | p10 §5.5 | recomputed NF (500 steps); baseline json | verified | — |
| 81 | bursts_big: "NF averages 0.056 to 0.114 above the baseline (0.845 to 0.849) … no seed reaches p<0.001; the best cell is seed 1 on later steps (p=0.013, weak)" | p10 §5.5 | json; NF hits recomputed (0.905–0.958); 3/14 sign test = 0.0127 | verified | "averages" → "is 0.056 to 0.114 above, by seed and phase" |
| 82 | "96% (bursts) and 100% (bursts_big) of its wrong picks share a detector with a hot qubit" | p10 §5.5 | json false_picks: 108/113 = 95.6 %, 143/143 | verified | — |
| 83 | "its hit rate at onset is about the same as on later steps: it loses on spatial resolution, not reaction time" | p10 §5.5 | 0.841 vs 0.825 (bursts); 0.849 vs 0.845 (bursts_big) | verified (inference reasonable) | — |

### 6 Ablations, Table 3 (pp. 10–12)

| # | claim | page/§ | source checked | verdict | fix |
|---|---|---|---|---|---|
| 84 | Table 3 row B d=3: win bursts onset −0.0171 (3e-112); loss hot_qubit later +0.0002 (p=0.0005) | p12 T3 | recomputed: −0.0171 (3.2e-112); +0.0002 (4.8e-4) | verified | — |
| 85 | Table 3 row B d=5: win −0.0019 (2e-7); losses bursts_big drift +0.0019 (4e-9), later +0.0024 (2e-11) | p12 T3 | recomputed: 2.5e-7, 4.3e-9, 2.3e-11 | verified | — |
| 86 | Table 3 row C d=3: win −0.0180 (2e-110); losses +0.0050 (8e-21), +0.0082 (2e-43) | p12 T3 | recomputed: 2.1e-110, 7.6e-21, 2.0e-43 | verified | — |
| 87 | Table 3 rows N, NF: bursts_big drift +0.0007 (p=0.03 / 0.04); "no losses" | p12 T3 | recomputed 0.031 / 0.043; full scan of 19 regime×phase cells + 3 test sets: no cell worse at p<0.001 | verified | — |
| 88 | §6.1 d=3: B beats baseline at onset on bursts_big (−0.0175, 4e-67) "and on both after steps" | p10 §6.1 | recomputed −0.0175 (3.8e-67); after −0.0014 (8e-9), −0.0030 (3e-26) | verified | — |
| 89 | "it ties static MWPM after bursts (p≥0.2)" | p11 §6.1 | recomputed p = 0.17 (bursts), 0.15 (bursts_big) | **wrong (bound)** | "p ≥ 0.15" |
| 90 | "fails the rule on a single cell … +0.0002 (p=0.0005), about 1% relative" | p11 §6.1 | recomputed +1.0 % | verified | — |
| 91 | d=5 B: absent on bursts_big onset (+0.0003, p=0.7) and hot_qubit (+0.0001, p=1); +4.1 %, +5.4 % | p11 §6.1 | recomputed | verified | — |
| 92 | "costs +6.4% to +17.7% relative on nominal stream steps in all four regimes (p≤1×10⁻¹⁰)" | p11 §6.1 | recomputed: +6.4 % (5.5e-10), +7.4 % (2.8e-12), +17.7 % (5e-21), +8.3 % (1.2e-10) | **wrong (bound)** | Largest p = 5.5e-10. Write "p ≤ 6×10⁻¹⁰". |
| 93 | "With a frozen fingerprint B and the baseline have the same nominal-step LERs" | p11 §6.1 | recomputed frozen nominal-step: 0.00199/0.00201, 0.00202/0.00202, 0.00195/0.00193, 0.00205/0.00208 | verified | — |
| 94 | "the B ensemble beats oracle MWPM at bursts onset (−0.0020, p=0.0002), where the baseline tied it" | p11 §6.1 | recomputed −0.00202 (2.0e-4); baseline −0.0001 (0.88) | verified | One positive cell out of many, and not a pre-registered endpoint. Label it "post hoc". |
| 95 | §6.2 C: "best variant on bursts and hot_qubit (drift vs baseline −0.0064, 1e-114, and −0.0011, 2e-54) and on the two-hot-qubit test set (−0.00112, 1e-11)" | p11 §6.2 | recomputed −0.0064 (1.5e-114), −0.0011 (2.3e-54), −0.00112 (1.1e-11) | numbers verified; **minor overstatement** | On hot_qubit drift (0.0197) and the two-hot test set (0.01953) C is *not* better than the adaptive-wide ensemble (0.0196; 0.01920). Write "best of the fingerprint variants (B, C, adaptive-stale)". |
| 96 | "much worse than B there (+0.0112, 1e-113)"; frozen "0.2714 vs 0.2539, unpaired" | p11 §6.2 | recomputed +0.01115 (1.1e-113); frozen 0.2714 / 0.2539 | verified | The frozen comparison is on identical shots, so it *can* be paired; "unpaired" is unnecessary. |
| 97 | §6.3: NF vs baseline on the test sets −0.00006 (p=0.3), −0.00016 (0.3), −0.00012 (0.3) | p12 §6.3 | recomputed from `exp9/d5/st0` test keys: p = 0.27, 0.27, 0.28 | verified | — |
| 98 | §6.4 h64/l6: "565 121 parameters"; bursts_big drift 0.0484 vs 0.0514 (−0.0031, 8e-14); nominal −0.00009 (p=0.1) | p12 §6.4 | recomputed from `exp4/full/st*/decoder_adaptive_stale_h64l6_…_errors.npz`: −0.00306 (8.0e-14); nominal −0.00009 (p=0.15). Parameters: §M | verified | p=0.15 rounds to 0.2 under round-half-up; harmless. |
| 99 | "2.2 times the GPU time (8540 s vs 3890 s)"; "missed the pre-set adoption bar of one percentage point … with 0.3 points (0.0590 vs 0.0620)" | p12 §6.4 | stream-0 LERs recomputed: 0.0590 vs 0.0620 ✓. Timing: only `notes/05:299`; `report4.md:13` says "2 h 22 min vs 65 min" (8520 s / 3900 s), `report4.md:71` implies ≈8570 s. No raw training logs. | LERs verified; **timing unsupported** (sources disagree slightly) | "about 2.2× the GPU time (≈2 h 22 min vs ≈65 min)", or add the logs. |
| 100 | "The three-seed H_c=48 ensembles (0.0457 and 0.0484) are at or below it (unpaired; one h64/l6 seed)" | p12 §6.4 | — | verified | It is on identical shots, so a paired test is possible. |
| 101 | §6.5: "every wide-trained ensemble beats all three of its members on drift steps of all four regimes (largest p=1e-7)" (d=5) | p12 §6.5 | recomputed 36 tests: largest 1.4e-7 | verified | — |
| 102 | "gain over the best seed −0.0065 (static-wide), −0.0032 (adaptive-wide), −0.0056 (adaptive-stale)"; "margin falls from −0.0209 … to −0.0107"; "live minus frozen … −0.0119 vs a single-seed mean of −0.0125" | p12 §6.5 | recomputed | verified | — |
| 103 | d=3: "static-wide ensemble is worse than its best member on hot_qubit drift (+0.0015, 4e-137)" | p12 §6.5 | recomputed +0.00146 (4.2e-137); also worse than seed 2 (+0.0011, 1.5e-87) | verified | — |

### 7 Discussion, 8 Limitations, 9 Conclusion (pp. 12–13)

| # | claim | page/§ | source checked | verdict | fix |
|---|---|---|---|---|---|
| 104 | live−frozen −0.0119 / −0.0063 / −0.0025; static MWPM 0.0980 vs 0.0439 | p12 §7 | recomputed | verified | — |
| 105 | ensemble − seed mean −0.0193 / −0.0149 / −0.0091; seed s.d. 0.0153 vs 0.0059 | p12 §7 | recomputed | verified | Both explanations are explicitly labelled hypotheses. Good. |
| 106 | "Only bursts_big extrapolates, and only in cluster size and strength" | p13 §8 | §3.2 ranges | verified | Add: hot_qubit (k=2, p_hot=0.03) and the two-hot test set are squarely inside the training range, at one fixed central location. |
| 107 | Latency: MWPM 1.85 µs/shot CPU; neural 10.2–10.8 µs on an NVIDIA L4 (d=5, batch 4096) | p13 §8 | `results/report3.md:448–453` (MWPM 1.85; static NN 10.2; adaptive NN 10.8) | verified | Note that the latency was measured on the 60k-step models (same architecture, so the same cost). |
| 108 | "the learning curves and h64/l6 are seed 0 only"; "no multiple-testing correction"; "best WindowDEM chosen after the fact" | p13 §8 | — | verified | Add: the pseudo-replication caveat (#43) and the use of bursts_big in development (#5). |
| 109 | Conclusion: "beats static and tuned sliding-window matching on every regime" | p13 §9 | Table 1 | verified ("tuned" = post hoc) | — |
| 110 | Conclusion: "beats matching with the true noise on moderate bursts and a hot-qubit defect, and trails it by 0.0031 on held-out large bursts" | p13 §9 | Table 1 | numbers verified; framing as #14, #5 | — |
| 111 | Conclusion: "The frozen control shows the gain is adaptation." | p13 §9 | — | **overstated** (as #7) | "The frozen control shows that the gain over a non-adaptive network trained on the same data is adaptation." |
| 112 | Data availability: "Code, experiment notes, result tables … and dataset manifests … are at github.com/chahat-101/drift. … per-shot error bits are not in the repository" | p13 | `git ls-files`: 0 `.npz` tracked, manifests tracked; remote `chahat-101/drift` | verified (repo contents); public visibility not checked | Resolve the TODO (#Q1). Note that the paired *p*-values cannot be reproduced from the public repo without the error bits. |

### Appendix tables (pp. 15–18)

| # | claim | page/§ | source checked | verdict | fix |
|---|---|---|---|---|---|
| 113 | Table 6 (d=5 per phase), all 7 decoders × 4 regimes × 5 phases incl. best-window labels | p16 | recomputed | verified (every cell) | — |
| 114 | Table 7 (d=3 per phase), every cell | p17 | recomputed | verified (every cell) | — |
| 115 | Table 8 (variants B, C, N, NF), every cell | p18 | recomputed | verified (every cell) | — |
| 116 | Tables 4, 5 (hyperparameters, datasets) | p15 | §M | mostly verified; see M1, M7–M9 | — |

### References (`refs.bib`, 17 entries; checked against the arXiv export API, Crossref and full texts on 2026-11-18)

All 17 arXiv IDs exist. All titles, author lists and years match the arXiv records, and every entry is cited.
No reference looks fabricated.

| # | entry | verdict | detail / fix |
|---|---|---|---|
| R1 | `bausch2024alphaqubit` | **wrong (title/venue mix)** | The DOI 10.1038/s41586-024-08148-8 is the Nature article titled "Learning high-accuracy error decoding for quantum processors". The bib gives the arXiv title under the Nature venue. Use the Nature title, or cite it as the arXiv preprint. |
| R2 | `spitz2018adaptive` | **wrong (title, missing volume)** | The published title is "Adaptive Weight Estimator for Quantum Error Correction in a Time-Dependent Environment", Adv. Quantum Technol. **1**(1), 1800012. Add the full title and `volume = {1}`. |
| R3 | `hoyt2026compat` | entry verified; text overstated | See #32. |
| R4 | `google2025below` | verified (minor) | The bib uses the arXiv author order. The Nature version has Abanin second, with the group author "Google Quantum AI and Collaborators". Optional: match the Nature author list. |
| R5 | `perez2018film` | verified | AAAI 2018 is confirmed by Crossref (vol. 32, DOI 10.1609/aaai.v32i1.11671). Add the DOI and drop the "supplied by the author" comment. |
| R6 | the other 12 entries | verified | senior2025alphaqubit2, gidney2021stim, higgott2025sparse, stein2026film, miao2026qadapt, yan2026rethink, peled2026nmwpm, chamberland2026predecoder, lange2025gnn, varbanov2025nn, klimov2018fluctuations, mcewen2022cosmic: ID, title, authors, year, venue and DOI all match. |
| R7 | line 688: hybrid "that sets matching weights from the fingerprint [Peled, Chamberland] could keep matching's speed" | minor | Peled et al. run a GNN plus a transformer on every syndrome, so their method does not keep matching's speed. Chamberland's weight-learning module is the closer analogue. Cite Chamberland alone, or say "in the spirit of". |

### arXiv readiness (arXiv publishes the `.tex` / `.bib` source, comments included)

| # | location | content | action |
|---|---|---|---|
| Q1 | `drift_whitepaper.tex:711` | `% TODO: decide where to host checkpoints and per-shot error bits (or state "available on request"), then update this paragraph.` | **Must fix.** Resolve it (Zenodo DOI or "available on request"), then delete the comment. |
| Q2 | `refs.bib:1–2` | `% All entries checked against the arXiv API (export.arxiv.org) on 2026-11-15: arXiv identifier, title, author list and (where present) journal reference.` | **Remove.** It is also inaccurate: R1 and R2 have wrong journal titles. |
| Q3 | `refs.bib:144` | `% Venue (AAAI 2018) supplied by the author; the arXiv record has no journal reference.` | Remove, and add the DOI (R5). |
| Q4 | `drift_whitepaper.tex:20,35–37` | macro comments ("float-only pages start at the top", "p-value: \E{3}{247} -> 3x10^-247", "same, for use inside math", "0.001 < p < 0.05") | Harmless. Optionally strip them. |
| Q5 | `drift_whitepaper.tex` (≈30 lines) | `%=====` section separator lines | Harmless. |
| Q6 | `whitepaper/` directory | `tech_doc_src/` (an older document whose `drift.tex` is the only cited source for the 4.27 % figure), `number_sources.md`, `check_numbers.py`, `paired_tests.*`, `make_figs.py`, `external_sources.md`, `claim_audit.md` | Upload only `drift_whitepaper.tex`, `refs.bib` (or the generated `.bbl`; arXiv needs the `.bbl` when biblatex/biber is used, and its biblatex version must match arXiv's) and `fig/*.pdf`. Do not upload the whole folder. |
| Q7 | `drift_whitepaper.tex:24` | `biblatex` + `biber` | arXiv does not run biber. Include the `.bbl` produced with a biblatex version that matches arXiv's TeX Live, or switch to BibTeX/natbib. |
| Q8 | `drift_whitepaper.tex:44` | `\date{\small Whitepaper, November 2026}` | Fine; "Whitepaper" may read oddly on arXiv. |

### §M Methods, setup and appendix vs code (methods subagent; file:line evidence)

| # | claim (quote) | page/§ | source checked | verdict | fix |
|---|---|---|---|---|---|
| M1 | "two-qubit depolarising noise after every two-qubit gate, single-qubit depolarising noise on every data qubit before every round, and bit-flip errors after resets and before measurements" | p3 §3.1 | `circuits.py:23-30` uses Stim `after_clifford_depolarization=p`; the generated circuit has `H …` followed by `DEPOLARIZE1(0.003) …` | **wrong (omission)** | Add "and single-qubit depolarising noise after every single-qubit gate". |
| M2 | "scales every location to p=s·p0 and adds … p_hot after every circuit layer to each qubit in H"; "k qubits nearest to some centre" | p3 §3.1 | `drift.py:150-152`, `circuits.py:34-38,46-54` (DEPOLARIZE1(p_hot) after every TICK, not scaled by s; s rounded to multiples of 0.05; hot set ranked over data *and* measure qubits) | verified, underspecified | Say "data or measure qubits"; `hot_qubit` = one data and one measure qubit (27 and 26). |
| M3 | Regimes (telegraph 0.04; hot_qubit from step 50, p_hot 0.03; bursts 0.04/5/0.05/4 steps; bursts_big 8/0.1; wander OU; static) | p4 §3.1 | `drift.py:47-94` | verified | — |
| M3b | Fig. 2 caption: "drawn with the same random seed, so bursts and bursts_big share burst times" | p4 | `make_figs.py:400` (illustration rng 3); evaluated streams use independent seeds per regime (`make_dataset.py:118`): 33 vs 29 bursts | verified for the figure; may mislead | Add "(the evaluated streams use independent seeds)". |
| M4 | Training distribution: ln s ~ U(ln0.5, ln2.5); prob ½ cluster k~U{1..6}, uniform centre, p_hot ~ U(0.005,0.08); 512 + 2000 shots; "Training never uses the regimes" | p4 §3.2 | `drift.py:100-108`, `make_dataset.py` | verified | "uniformly random centre" → "centred on a uniformly chosen qubit". |
| M5 | Grid (r+1,d+1,d+1); N=24/120; fingerprint ×10, W=2000 | p4 §3.3 | `DetectorGrid`; Stim `num_detectors`; `decoder.py:153,201` | verified | Optionally mention that the buffer starts empty (zeros) and is partial for the first 2000 shots of a stream. |
| M6 | Parameters 195 681 / 194 385; h64/l6 563 393 / 565 121 | p5 §3.4, p12, T4 | models instantiated | verified | — |
| M7 | Stale augmentation (q=0.3; ½ other chunk, ½ mixed window, k=⌊UW⌋; own RNG) | p5 §3.5 | `train.py:228-235,322,384-386` | verified | (j can equal i; negligible) |
| M8 | static-wide "trained on the same wide chunks in the same order as adaptive-wide" | p5 §3.6 | `train.py:322` (same `Order` rng; adaptive-wide makes no stale draws) | verified | — |
| M9 | Ensemble rule "was fixed before the ensemble runs" | p5 §3.6 | `ensemble_eval.py:6`, `run6.sh` header; both first committed in bulk commit a3eaef5 | **unverifiable from git** | Same remedy as #44. |
| M10 | "In a live ensemble each member keeps its own fingerprint buffer" | p5 §3.6 | `ensemble_eval.py:226-231` ("members with equal windows share one forward input"); the fingerprint depends only on detection events | **misleading** | "All members see the same fingerprint (same shots, same W)." |
| M11 | N/NF heads, loss weights, masking on stale steps, FiLM (two 16-ch convs, zero-init γ/β, after GELU), per-step mean readout | p6 §3.7 | `decoder.py:72-97,114-122`; `train.py:265-267,391`; `noise_eval.py:52-56` | verified | "The logit path is unchanged" holds for the architecture only: the shared trunk receives aux-loss gradients. Add "(architecturally)". |
| M12 | Stim 1.16.0, PyMatching 2.4.0, NumPy 2.4.6, PyTorch 2.15.0 | p6 §4, T4 | dataset manifests; `results/exp3/latency_d5.json`; `notes/03:25` | verified | `requirements.txt` is unpinned. Pin it for reproducibility. (The local `.venv` has torch 2.14.1 and no stim, so it is not the run environment.) |
| M13 | "Batch, precision: 512 shots per step, fp32" | p15 T4 | PyTorch default cuDNN TF32 for convs; `ensemble_eval.py:47-48` sets `allow_tf32=True` | **overstated** | "fp32 (convolutions in TF32, the PyTorch default)" |
| M14 | 60 000 / 300 000 steps = two passes over 30 000 / 150 000 chunks; AdamW, OneCycle, lr 1e-3 | p6 §4, T4 | training logs ("= 2.00 epochs"); `train.py:303-304` | verified | — |
| M15 | "test sets and streams are byte-identical across all datasets of one distance"; Table 5 dataset list | p6 §4, p15 T5 | manifest SHA-256 identical for the test sets and main streams. But `d3_wide` has no bursts_big stream, stream seeds 1–4 exist only in `d3_r3_p0.003_s0` / `d5_wide`, and **all d=3 evaluations used `d3_r3_p0.003_s0`** (`run6.sh:35`, `run7.sh:85`), a dataset not in Table 5. | imprecise | Add `d3_r3_p0.003_s0` (streams and test sets) to Table 5, or write "all streams of one distance come from a single dataset". |
| M16 | Table 5 "On disk" column | p15 T5 | bytes: d5_wide 534.65 MB (= 510 MiB), d5_nominal 89.5 MB (85.4 MiB), x5 sets 2.28 GiB / 392 MiB; d3_nominal 16.6 MB shown as "--" | mixed units, one gap | Say MiB/GiB; fill in d3_nominal ≈16 MiB; 86 → 85 MiB. |
| M17 | Static / oracle MWPM | p6 §4 | `baselines.py:27-51` | verified | — |
| M18 | WindowDEM "estimating each edge probability from pairwise detector correlations in the manner of [Spitz]"; clipped [1e-6, 0.45] | p6 §4, T4 | `baselines.py:89-120`: two-point estimator on the edges of the *nominal* decomposed graph only; boundary edges solved from single-detector rates; no hyperedge correction and no new edges; no reweighting until min(w,1000) shots; re-fit after every 200-shot batch | verified, under-described | Add one sentence with these details. A referee will ask, because the baseline's quality bounds the "beats WindowDEM" claim. Without a hyperedge (Y-error) correction, the estimator is biased under depolarising hot qubits. |
| M19 | Causal streaming (decode, then observe); frozen fingerprint from the nominal test set's 2000 fingerprint shots | p6 §4 | `eval.py:166-171`; `ensemble_eval.py:83-93,216-218`; `dataset.py:169` | verified | — |
| M20 | Phase definitions | p6 §4 | `eval.py:59-72` | verified | — |
| M21 | Localisation baseline A(q), score, sign tests | p10 §5.5 | `results/exp9/localisation_baseline.py:44-62` | verified | — |
| M22 | Latency 1.85 µs; 10.2–10.8 µs (L4, batch 4096) | p13 §8 | `results/exp3/latency_d5.json`: 1.853 / 10.16 / 10.76 µs (60k-step models, same architecture) | verified | Optionally say "end-to-end incl. host↔device copies, 60k-step checkpoints". |

---

## Summary

**Scope.** 116 numbered paper claims, 22 method rows (§M), 7 reference rows and 8 source-hygiene rows: 153 rows in
all. About 500 individual numbers were recomputed from raw per-shot error bits: all of Tables 1, 2, 3, 6, 7 and 8
and every number in §5–§6. All 17 references were checked against arXiv, Crossref and full texts.

**Bottom line.** The data are sound. Every LER, Δ and McNemar *p* in the headline results (abstract, Table 1,
Table 2, §5.5) reproduces from the saved bits to the printed precision, and so do the appendix tables. No
reference is fabricated. The problems are in **framing and inference**, plus about ten factual slips in the text.

### Problems, ranked by severity

**Major: a QEC referee would raise these**

- **S1. "The gain comes from adaptation" is overstated** (abstract, contribution (b), §5.2 title, conclusion; #7, #15, #59, #111).
  With the fingerprint frozen, the model is at 0.0576, already far below static MWPM (0.0980) and best WindowDEM
  (0.0896). About 77 % of the gain over matching on `bursts_big` is decoder quality; 23 % is adaptation. The frozen
  control shows that the gain *over a non-adaptive network trained on the same data* is adaptation.
- **S2. "Beats MWPM with the true noise" is framed as an adaptation result but is mostly decoder quality** (#14, #51, #110).
  On nominal steps the network is already 35–39 % below MWPM. On drift it is only 14 % below oracle (`bursts`),
  29 % (`hot_qubit`) and +7 % above (`bursts_big`). Its relative advantage over model-aware matching *shrinks* under
  drift. The no-fingerprint static-wide ensemble also beats oracle on `hot_qubit` and `telegraph`. The paper
  concedes the decoder-quality point only for `telegraph`.
- **S3. Pseudo-replication: shot-level p-values are presented as regime-level evidence** (#43).
  McNemar treats ~10⁵–10⁶ shots as independent, but they come from 29 burst runs per burst regime and **5**
  `hot_qubit` runs, all at the same two central qubits (as is the two-hot test set). At the event level the
  burst-regime headline results still hold: 26/3 to 29/0 runs, sign-test p ≤ 1.5e-5. But every `hot_qubit` claim
  rests on 5/0 runs (p = 0.06), and `bursts_big` vs oracle is 9/20 (p = 0.06). §5.5 already calls the event-level test
  "the honest one"; §4 and §5 should say the same.
- **S4. `bursts_big` is called "held out" but steered development** (#5).
  It drove the move from 60k to 300k steps (report 3), the N/NF decision rule, part of the B/C rules and the h64/l6
  adoption bar. It is out-of-distribution for *training*, not a held-out *test* set.
- **S5. "Decision rules fixed in advance" cannot be verified** (#44, M9).
  The rule text matches the paper exactly, but it lives only in post-run report scripts and notes. All of them were
  first committed in one bulk commit after round 9 had finished.

**Moderate: factually wrong statements**

- **S6.** §5.2: "At d=5 the frozen ensembles are close to the static-wide ensemble". This is true for adaptive-stale
  only. Frozen adaptive-wide is 0.0680 vs 0.0564 (+0.0116, p = 2e-178) (#63).
- **S7.** §5.4: "every neural model lost … (0.118 to 0.150)" excludes static-nominal (0.294). "Both failures are gone
  in every seed" is false for static-nominal, which still loses on `bursts_big` at 0.2159 vs 0.0980 (#72, #74).
- **S8.** §3.1 noise model omits the single-qubit depolarising noise after every single-qubit gate (M1).
- **S9.** §3.6: "each member keeps its own fingerprint buffer". All members get the identical fingerprint (M10).
- **S10.** §2: Hoyt et al. showed drift on IBM data only; on Google data they report only a "similar mismatch" (#32).
- **S11.** Bibliography: the AlphaQubit entry pairs the arXiv title with the Nature DOI (R1). The Spitz entry has the
  wrong title and is missing volume 1 (R2). The bib header comment claiming all entries were checked is therefore false.
- **S12.** Abstract: "Three extensions … (… a noise-map head with FiLM conditioning)". Four variants were judged, and
  N has no FiLM (#9).
- **S13.** §5.2: "At d=5 … this cost is not significant". There is no cost: static-wide is *below* static-nominal in
  all three seeds (#68).
- **S14.** arXiv blockers: the TODO at tex:711 (Q1); biblatex/biber needs a matching `.bbl` (Q7); don't upload the
  helper files (Q6).

**Minor: precision, wording, missing detail**

- **S15.** Wrong p-value bounds: "p ≤ 1×10⁻²⁰⁹" (actual 1.2e-209, #61); "p ≥ 0.2" (actual 0.15, #89);
  "p ≤ 1×10⁻¹⁰" (actual 5.5e-10, #92).
- **S16.** "fp32" while convolutions run in TF32 (M13).
- **S17.** h64/l6 timing "8540 s vs 3890 s" has no log behind it, and the two report sources disagree slightly (#99).
- **S18.** Window-length estimate ignores the per-shot overhead, which could dominate (#34).
- **S19.** `telegraph` "needs no adaptation". The live−frozen difference is small but significant at d=5 (p = 1.8e-4) (#64).
- **S20.** C "best variant on hot_qubit / two-hot test set": the adaptive-wide ensemble is lower (#95).
- **S21.** Table 5: units mixed; d3_nominal size missing; the d=3 evaluation dataset `d3_r3_p0.003_s0` is not listed (M15, M16).
- **S22.** WindowDEM estimator under-described: nominal-graph edges only, no hyperedge correction (M18).
- **S23.** Smaller items:
  - §1 "recalibration cannot follow" is uncited (#13).
  - Peled is cited for a speed-preserving hybrid (R7).
  - The Fig. 2 burst-time remark applies only to the illustration (M3b).
  - The B-vs-oracle onset win should be labelled post hoc (#94).
  - The C/B frozen comparison and the h64/l6 comparison are called "unpaired" although they are on identical shots (#96, #100).
  - Say "4.28 %" (#35).
  - Name the five models and "stream seed 0" in §5.4 (#75).
  - The d=3 nominal-test numbers are single-model means (#67).

### Recommended text changes (exact; not applied)

1. **Abstract, l.57–58.**
   - Old: "With the same weights and a frozen nominal fingerprint the rate is 0.0576 (paired McNemar $p=\Eb{3}{247}$), so the gain comes from adaptation."
   - New: "With the same weights and a frozen nominal fingerprint the rate is 0.0576 (paired McNemar $p=\Eb{3}{247}$), close to a network trained without a fingerprint (0.0564): the gain over that network comes from adaptation, while most of the gain over matching comes from the network itself."
2. **Abstract, l.55.** "on held-out large bursts" → "on large bursts outside the training distribution".
3. **Abstract, l.60–61.**
   - Old: "Three extensions with decision rules fixed in advance (multiscale and pair-covariance fingerprints, a noise-map head with FiLM conditioning) failed those rules."
   - New: "Four variants judged by pre-set decision rules (multiscale and pair-covariance fingerprints, a noise-map head with and without FiLM conditioning) failed those rules."
4. **Contribution (a), l.87–88.** Append: "On nominal noise the network is already about 36\,\% below MWPM, so part of this margin is decoder quality rather than adaptation."
5. **Contribution (b), l.89–90.** "show that the gain on localised drift is adaptation" → "show that the gain over a non-adaptive network trained on the same data is adaptation".
6. **§2, l.146–148.**
   - Old: "Hoyt et al.~\cite{hoyt2026compat} show on IBM and Google data that syndrome statistics drift away from the stationary detector error model used for decoding, and use the mismatch for post-selection"
   - New: "Hoyt et al.~\cite{hoyt2026compat} show on IBM hardware that syndrome statistics drift across acquisition windows away from the stationary detector error model used for decoding, report a similar mismatch in Google surface-code data, and use the mismatch for post-selection"
7. **§3.1, l.188–189.** After "two-qubit depolarising noise after every two-qubit gate," insert "single-qubit depolarising noise after every single-qubit gate,". After "the $k$ qubits nearest to some centre" insert "(data or measure qubits)".
8. **§3.3, l.241.** "$4.27\%$" → "$4.28\%$".
9. **§3.6, l.325–327.**
   - "this rule was fixed before the ensemble runs" → "this rule is stated in the ensemble launch script".
   - "In a live ensemble each member keeps its own fingerprint buffer." → "All members receive the same fingerprint."
10. **§4 Data, l.348–352, and Table 5.** Add the `d3_r3_p0.003_s0` dataset (or "all streams and test sets of one distance come from a single dataset"). Fix the units to MiB/GiB and fill in d3_nominal.
11. **§4 Baselines, l.357–359.** Append: "The estimator is the two-point formula on the edges of the nominal matching graph, with boundary edges solved from single-detector rates and no hyperedge correction; reweighting starts after $\min(w,1000)$ shots and is refreshed every 200 shots."
12. **§4 Statistics, after l.384.** Add: "Shots within a drift event share the same noise state, so for regime-level conclusions the unit of replication is the event (29 runs per burst regime, 21 for \reg{telegraph}, 5 for \reg{hot\_qubit}, all at the same location). At the event level the main \reg{bursts} and \reg{bursts\_big} comparisons in \cref{tab:main,tab:liveFrozen} hold with sign-test $p\le\Eb{1.5}{5}$; every \reg{hot\_qubit} comparison has $p=0.06$, the smallest possible with five events."
13. **§4 Pre-set decision rules, l.387 and l.392.**
    - "were judged by rules written before each run" → "were judged by rules recorded with each round's launch and report scripts".
    - Add: "The repository history does not show that the rules predate the runs."
14. **§4 or §3.1.** Add: "\reg{bursts\_big} was not used for training, but it did inform development decisions (training length, the noise-map decision rule and the h64/l6 adoption bar), so it is out-of-distribution rather than a sealed test set."
15. **§5.1, l.402–403.** After "It beats oracle MWPM on \reg{bursts}, \reg{hot\_qubit} and \reg{telegraph}" insert: "; since the network is already 35–39\,\% below MWPM on nominal steps, these wins reflect decoder quality as well as adaptation".
16. **§5.2 title.** "The gain is adaptation" → "What the fingerprint adds".
17. **§5.2, l.470.** "$p\le\Eb{1}{209}$" → "$p\le\Eb{2}{209}$".
18. **§5.2, l.473–475.**
    - Old: "At $d=5$ the frozen ensembles are close to the static-wide ensemble (\reg{bursts\_big} drift 0.0576 vs 0.0564; paired $+0.0012$, $p=0.0015$, weak), and the live fingerprint again supplies the gain."
    - New: "At $d=5$ the frozen adaptive-stale ensemble is close to the static-wide ensemble (\reg{bursts\_big} drift 0.0576 vs 0.0564; paired $+0.0012$, $p=0.0015$, weak), while frozen adaptive-wide is well above it (0.0680, \E{2}{178}); in both, the live fingerprint supplies the gain over static-wide."
19. **§5.2, l.478.** "so uniform scale drift needs no adaptation from this decoder" → "so uniform scale drift needs little adaptation from this decoder".
20. **§5.2, l.516.** "At $d=5$ after 300\,000 steps this cost is not significant (largest $p=0.2$)." → "At $d=5$ after 300\,000 steps there is no such cost: static-wide is lower than static-nominal in all three seeds (largest $p=0.2$)."
21. **§5.4, l.535–539.**
    - Old: "every neural model lost to static MWPM on \reg{bursts\_big} drift (three-seed means 0.118 to 0.150, against 0.098) … After 300\,000 steps both failures are gone in every seed"
    - New: "every wide-trained model lost to static MWPM on \reg{bursts\_big} drift (three-seed means 0.118 to 0.150, against 0.098; static-nominal 0.294) … After 300\,000 steps both failures are gone in every seed for the wide-trained models (static-nominal still loses on \reg{bursts\_big}: 0.216)"
    - Also change "all five improved" → "all five seed-0 models (static-wide, static-nominal, adaptive-wide, adaptive-stale, h64/l6) improved on \reg{bursts\_big} drift of stream seed 0".
22. **§6.1, l.614.** "($p\ge0.2$)" → "($p\ge0.15$)".
23. **§6.1, l.621.** "($p\le\Eb{1}{10}$)" → "($p\le\Eb{6}{10}$)".
24. **§6.1, l.623–624.** "One positive result:" → "One positive, post hoc result:".
25. **§6.2, l.628.** "it is the best variant on" → "it is the best of the fingerprint variants (adaptive-stale, B, C) on".
26. **§6.4, l.644.** "at 2.2 times the GPU time (8540\,s vs 3890\,s)" → "at about 2.2 times the GPU time (about 2\,h\,22\,min vs 65\,min)".
27. **Conclusion, l.703.** "The frozen control shows the gain is adaptation." → "The frozen control shows that the gain over a non-adaptive network trained on the same data is adaptation."
28. **Limitations, last bullet.** Append: "$p$-values are shot-level; regime-level conclusions rest on 29 burst events per burst regime and five \reg{hot\_qubit} events. \reg{bursts\_big} informed development decisions."
29. **Table 4.** "512 shots (one chunk) per step, fp32" → "…, fp32 (TF32 convolutions)".
30. **Fig. 2 caption.** Append: "The evaluated streams use independent seeds per regime."
31. **refs.bib.**
    - `bausch2024alphaqubit`: title → "Learning High-Accuracy Error Decoding for Quantum Processors".
    - `spitz2018adaptive`: title → "Adaptive Weight Estimator for Quantum Error Correction in a Time-Dependent Environment"; add `volume = {1}`, `number = {1}`.
    - `perez2018film`: add `doi = {10.1609/aaai.v32i1.11671}`, `volume = {32}`.
    - Delete the comments at lines 1–2 and 144.
32. **tex:711.** Resolve the TODO and delete the comment. Ship a `.bbl` built with a biblatex version matching arXiv's.

---

## Erratum to recommended change 12 (found while applying it, 2026-11-18)

The suggested sentence "the main bursts and bursts_big comparisons in Tables 1 and 2 hold with sign-test
p ≤ 1.5×10⁻⁵" was too broad. `audit/t9.py` runs event-level sign tests for every Table 1 and Table 2 cell
(`audit/out/t9.txt`). Two comparisons do not hold at that level:
- d=5 `bursts_big` adaptive-stale vs oracle: 9 events better, 20 worse, p=0.06.
- d=3 `bursts` vs oracle: 14 better, 15 worse, p=1.

The other d=3 `bursts_big` cells range from p=0.008 to 1e-7. The applied text says instead: "every d=5 comparison in
Tables 1 and 2 on bursts and bursts_big keeps p ≤ 1.5×10⁻⁵ except bursts_big against oracle MWPM (9 events better,
20 worse, p=0.06)".
