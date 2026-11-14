# Training-free localisation baseline for the report-9 noise map

Report 9 scores the noise-map heads of N and NF by their top-k hot-qubit hit rate. This file adds a baseline that needs no training: a hot-qubit score computed directly from the same live fingerprint the models receive. It compares that baseline with NF seed by seed, on identical steps.

Nothing was trained, and no model was run. The N and NF hit rates are recomputed from the saved per-step predictions in `results/exp9/noise/noise_{aux,auxfilm}_s{0,1,2}.npz`, and they reproduce report 9's values exactly (checked in all 54 cells). The baseline was computed by `results/exp9/localisation_baseline.py`, and its raw numbers are in `results/exp9/localisation_baseline.json`.

## Summary

1. **hot_qubit: the baseline is perfect.** It scores 1.000 in every phase, the same as every N and NF seed (500 drift steps; 5 events, one per stream). On this regime, report 9's 1.000 for the noise maps shows nothing that the raw fingerprint does not already give.
2. **bursts: NF is clearly better.**
   - Baseline: 0.825–0.841. NF: 0.975–1.000.
   - NF wins by 0.150–0.159 in every phase, for every seed.
   - Per event, NF wins 20 bursts, loses 1 and ties 12 (p=2e-05, all three seeds). At onset it wins 17–18 and loses 0 (p ≤ 2e-05).
3. **bursts_big: the NF advantage is not established once events are counted.** bursts_big is held out of NF's training.
   - Baseline: 0.845–0.849. NF: 0.905–0.958. NF is ahead by 0.056–0.114.
   - Per step, the gap is significant at p < 0.001 only for seed 1 (drift and later). It is weak for seed 0 (p=0.0014, 0.0015) and not significant for seed 2.
   - Per event, no seed reaches p < 0.001. The best is seed 1 on later steps, p=0.013 (weak).
4. **The baseline's misses are near misses.** 96% (bursts) and 100% (bursts_big) of its wrong picks share a detector with a hot qubit. Its hit rate at onset is the same as on later steps (bursts 0.841 vs 0.825). So the baseline loses on spatial resolution, not on reaction time.
5. **The choice of nominal reference hardly matters.** Using the exact nominal detection probabilities instead of the nominal test-set window moves the baseline by at most 0.002 on bursts and by −0.008 to −0.010 on bursts_big. No event-level count or p-value changes (Appendix).

## Method

### Live fingerprint

- The fingerprint is the same causal buffer the models use (`decoder.Fingerprint` and `ensemble_eval.window_fps`, `window` = 2000 from the checkpoint configs, all single-fingerprint).
- Before each 200-shot sub-batch, it holds each detector's firing rate over the last 2000 shots.
- A step's fingerprint `f` is the mean over its 10 sub-batches. This matches how `noise_eval.step_predictions` averages the noise head over a step's shots, and since the score is linear in `f`, scoring each sub-batch and averaging gives the same result.

### Nominal reference and excess

- `f(nominal)` is the frozen fingerprint: the nominal test set's 2000 fingerprint shots, through `Fingerprint.of_windows`. This is the same input as the frozen runs in `ensemble_eval`.
- For each detector a: excess_a = f_a − f_a(nominal).
- Sensitivity check: the exact nominal firing probabilities, 0.5 · (1 − ∏(1 − 2p_e)) over the nominal DEM errors that touch each detector.

### Adjacency and score

- A qubit's adjacent detectors are the endpoints of the matching-graph edges that a single-qubit Pauli fault on that qubit produces.
- They come from the decomposed DEM of a circuit whose only noise is DEPOLARIZE1 on that qubit after every TICK. This is exactly the hot-qubit noise model in `circuits.make_circuit`.
- Every such edge was checked to be an edge of the nominal matching graph (`circuits.matcher_from_circuit`).
- 49 qubits are scored, the checkpoints' `aux_qubits`: 25 data and 24 measure qubits.
- Each qubit has 9–24 adjacent detectors (mean 16.0 for data qubits, 17.0 for measure qubits).
- score_q = Σ excess_a over q's adjacent detectors. The sum is not normalised or weighted.

### Top-k hit rate

- This is the metric from `noise_eval.metrics`, unchanged.
- k = the number of truly hot qubits in the step. Qubits are ranked with a stable argsort. Only steps with hot qubits are counted.
- The phase masks are the same `eval.phase_masks`, and the five stream seeds are pooled.
- after and nominal steps have no hot qubits, so they have no top-k.

### Paired comparison with NF

- For each NF seed, the per-step difference is hit(baseline) − hit(NF), on identical steps.
- **p (steps):** exact two-sided sign test over the steps where the two differ.
- **p (events):** the same test over burst events, where each event's value is the mean of its step differences.
  - An event is a run of consecutive steps with the same hot set.
  - Steps within one event share the same hot qubits, and their 2000-shot windows overlap, so step-level p treats dependent steps as independent. Event-level p is the honest test.
  - At onset there is one step per event, so the two tests coincide.
- The baseline is deterministic, so all three NF seeds are compared with the same baseline values.
- **Bold** = p < 0.001. p from 0.001 to 0.05 is weak evidence. p ≥ 0.001 is printed to 2 significant figures. There is no multiple-testing correction.
- Event counts:
  - bursts: 33 events but 29 onset steps. Four bursts start on the step right after another burst, so their first step is not an onset.
  - bursts_big: 29 events.
  - hot_qubit: 5 events.

## Data: regenerated and verified

The stream files are not on this machine: `data/d5_wide/` holds only its manifest, and the project `.venv` is missing. I regenerated them with `results/exp9/regen_streams.py`, using the manifest's seeds in a scratch environment with the manifest's exact versions: Python 3.11.17, stim 1.16.0, pymatching 2.4.0, numpy 2.4.6. torch 2.15.0+cpu was used only to read the checkpoint configs and to run `Fingerprint` and `window_fps`.

File hashes cannot be compared, because `np.savez_compressed` writes timestamps into the zip. Instead I checked the content:

- **Streams.** For all 15 streams (bursts, bursts_big, hot_qubit × stream seeds 0–4), the static-MWPM error count on regenerated shots equals the saved count in `results/exp3/d5/*_s0.csv` on every one of the 150 steps.
- **Hot-qubit truth.** The truth per step equals the truth saved in NF's `.npz`.
- **Nominal test set.** Static MWPM makes 657 errors on the 200000 test shots, equal to `results/exp4/curves` (0.003285).
  - The 2000 fingerprint shots are the next draw of the same rng. They have no saved result of their own, so they are not checked directly.
  - The analytic-reference check does not use them and gives the same conclusions.

## Results

Top-k hot hit rate, live fingerprint, pooled over stream seeds 0–4. The N columns are for context only.

| regime | phase | steps (events) | baseline: test-set ref | baseline: analytic ref | NF s0 | NF s1 | NF s2 | NF mean ± std | N s0 / s1 / s2 (context) |
|---|---|---|---|---|---|---|---|---|---|
| bursts | drift | 132 (33) | 0.829 | 0.827 | 0.982 | 0.979 | 0.980 | 0.980 ± 0.002 | 0.965 / 0.958 / 0.976 |
| bursts | onset | 29 (29) | 0.841 | 0.841 | 1.000 | 0.993 | 0.993 | 0.995 ± 0.004 | 0.966 / 0.952 / 0.972 |
| bursts | later | 103 (33) | 0.825 | 0.823 | 0.977 | 0.975 | 0.977 | 0.976 ± 0.001 | 0.965 / 0.959 / 0.977 |
| bursts_big | drift | 116 (29) | 0.846 | 0.836 | 0.935 | 0.953 | 0.913 | 0.934 ± 0.020 | 0.943 / 0.929 / 0.922 |
| bursts_big | onset | 29 (29) | 0.849 | 0.841 | 0.931 | 0.935 | 0.905 | 0.924 ± 0.016 | 0.966 / 0.944 / 0.953 |
| bursts_big | later | 87 (29) | 0.845 | 0.835 | 0.937 | 0.958 | 0.915 | 0.937 ± 0.022 | 0.935 / 0.924 / 0.912 |
| hot_qubit | drift | 500 (5) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 | 1.000 / 1.000 / 1.000 |
| hot_qubit | onset | 5 (5) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 | 1.000 / 1.000 / 1.000 |
| hot_qubit | later | 495 (5) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 | 1.000 / 1.000 / 1.000 |

### Baseline vs NF, seed by seed

Reference: the nominal test-set window.

| regime | phase | NF seed | NF hit rate | diff (baseline − NF) | steps: baseline better / NF better / tie | p (steps) | events: baseline better / NF better / tie | p (events) |
|---|---|---|---|---|---|---|---|---|
| bursts | drift | s0 | 0.982 | -0.153 | 1 / 79 / 52 | **1e-22** | 1 / 20 / 12 | **2e-05** |
| bursts | drift | s1 | 0.979 | -0.150 | 1 / 78 / 53 | **3e-22** | 1 / 20 / 12 | **2e-05** |
| bursts | drift | s2 | 0.980 | -0.152 | 1 / 78 / 53 | **3e-22** | 1 / 20 / 12 | **2e-05** |
| bursts | onset | s0 | 1.000 | -0.159 | 0 / 18 / 11 | **8e-06** | 0 / 18 / 11 | **8e-06** |
| bursts | onset | s1 | 0.993 | -0.152 | 0 / 17 / 12 | **2e-05** | 0 / 17 / 12 | **2e-05** |
| bursts | onset | s2 | 0.993 | -0.152 | 0 / 17 / 12 | **2e-05** | 0 / 17 / 12 | **2e-05** |
| bursts | later | s0 | 0.977 | -0.151 | 1 / 61 / 41 | **3e-17** | 1 / 20 / 12 | **2e-05** |
| bursts | later | s1 | 0.975 | -0.150 | 1 / 61 / 41 | **3e-17** | 1 / 20 / 12 | **2e-05** |
| bursts | later | s2 | 0.977 | -0.151 | 1 / 61 / 41 | **3e-17** | 1 / 20 / 12 | **2e-05** |
| bursts_big | drift | s0 | 0.935 | -0.089 | 27 / 57 / 32 | 0.0014 | 10 / 15 / 4 | 0.42 |
| bursts_big | drift | s1 | 0.953 | -0.107 | 15 / 56 / 45 | **1e-06** | 6 / 14 / 9 | 0.12 |
| bursts_big | drift | s2 | 0.913 | -0.067 | 36 / 52 / 28 | 0.11 | 9 / 13 / 7 | 0.52 |
| bursts_big | onset | s0 | 0.931 | -0.082 | 10 / 15 / 4 | 0.42 | 10 / 15 / 4 | 0.42 |
| bursts_big | onset | s1 | 0.935 | -0.086 | 6 / 14 / 9 | 0.12 | 6 / 14 / 9 | 0.12 |
| bursts_big | onset | s2 | 0.905 | -0.056 | 9 / 13 / 7 | 0.52 | 9 / 13 / 7 | 0.52 |
| bursts_big | later | s0 | 0.937 | -0.092 | 17 / 42 / 28 | 0.0015 | 6 / 14 / 9 | 0.12 |
| bursts_big | later | s1 | 0.958 | -0.114 | 9 / 42 / 36 | **3e-06** | 3 / 14 / 12 | 0.013 |
| bursts_big | later | s2 | 0.915 | -0.070 | 27 / 39 / 21 | 0.18 | 9 / 13 / 7 | 0.52 |
| hot_qubit | drift | s0 | 1.000 | +0.000 | 0 / 0 / 500 | 1 | 0 / 0 / 5 | 1 |
| hot_qubit | drift | s1 | 1.000 | +0.000 | 0 / 0 / 500 | 1 | 0 / 0 / 5 | 1 |
| hot_qubit | drift | s2 | 1.000 | +0.000 | 0 / 0 / 500 | 1 | 0 / 0 / 5 | 1 |
| hot_qubit | onset | s0 | 1.000 | +0.000 | 0 / 0 / 5 | 1 | 0 / 0 / 5 | 1 |
| hot_qubit | onset | s1 | 1.000 | +0.000 | 0 / 0 / 5 | 1 | 0 / 0 / 5 | 1 |
| hot_qubit | onset | s2 | 1.000 | +0.000 | 0 / 0 / 5 | 1 | 0 / 0 / 5 | 1 |
| hot_qubit | later | s0 | 1.000 | +0.000 | 0 / 0 / 495 | 1 | 0 / 0 / 5 | 1 |
| hot_qubit | later | s1 | 1.000 | +0.000 | 0 / 0 / 495 | 1 | 0 / 0 / 5 | 1 |
| hot_qubit | later | s2 | 1.000 | +0.000 | 0 / 0 / 495 | 1 | 0 / 0 / 5 | 1 |

## Where the baseline goes wrong

Wrong picks on drift steps, pooled over stream seeds. A "wrong pick" is a non-hot qubit in the top k. 24 of the 49 qubits (49%) are measure qubits.

| regime | decoder | false picks (drift steps) | of which measure qubits | of which share a detector with a hot qubit |
|---|---|---|---|---|
| bursts | baseline | 113 | 39 (35%) | 108 (96%) |
| bursts | NF s0 | 12 | 7 (58%) | 5 (42%) |
| bursts | NF s1 | 14 | 6 (43%) | 7 (50%) |
| bursts | NF s2 | 13 | 7 (54%) | 7 (54%) |
| bursts_big | baseline | 143 | 86 (60%) | 143 (100%) |
| bursts_big | NF s0 | 60 | 49 (82%) | 60 (100%) |
| bursts_big | NF s1 | 44 | 30 (68%) | 44 (100%) |
| bursts_big | NF s2 | 81 | 67 (83%) | 81 (100%) |

- **Baseline.** Almost every wrong pick is next to the hot cluster: 108/113 on bursts and 143/143 on bursts_big.
  - Summing excess over adjacent detectors spreads a hot qubit's signal onto every qubit that shares those detectors.
  - Inside a compact cluster of 5 or 8 hot qubits, a cold neighbour can collect as much excess as a hot qubit on the cluster's edge.
- **NF on bursts.** It makes about a ninth as many wrong picks (12–14), and only 42–54% of them are next to the cluster.
- **NF on bursts_big.** Its wrong picks are all next to the cluster, and 68–83% are measure qubits.

## Limitations

- **One fixed rule, untuned.** The baseline is exactly as specified: an unnormalised sum over adjacent detectors. Other scores were not tried, for example the mean instead of the sum, per-edge excess, or a deconvolution over the cluster. A better training-free score could narrow the bursts gap.
- **NF sees more current data.** NF's per-step map averages the noise head over the step's 2000 shots, and each shot's input contains that shot's own detection events as well as the fingerprint. The baseline uses only the causal fingerprint, as specified. On bursts the baseline's onset hit rate equals its later-step rate, which suggests this asymmetry is not what drives the gap there. That has not been tested.
- **The sign test ignores the size of differences, and there are few events.** There are 29–33 burst events, and 5 on hot_qubit. With 5 events, the hot_qubit test could never reach p < 0.05.
- **hot_qubit is at the ceiling.** Every method scores 1.000, so the comparison cannot separate them there.
- **The seed-by-seed tests are not independent.** They share the same baseline. There is no multiple-testing correction.
- **One nominal window.** The reference is one 2000-shot nominal window. The analytic reference is the check on this.

## What this means for report 9's noise-map claim

- **bursts (training range).** NF localises hot qubits better than a training-free fingerprint score: +0.15 hit rate for every seed and phase, event-level p=2e-05 (onset p ≤ 2e-05).
- **hot_qubit.** The fingerprint alone localises perfectly, so the 1.000 hit rates of N and NF are not evidence of learned localisation in this regime.
- **bursts_big (held out).** NF averages 0.06–0.11 above the baseline. At the event level this is not significant for any seed; one cell is weak (s1 later, p=0.013). The statement "the noise head localises held-out large bursts better than the raw fingerprint" is not supported by these data.

## Files

- `results/exp9/localisation_baseline.py`: computes the baseline, the paired tests and the false-pick diagnostic. It also asserts that every N and NF top-k value equals report 9's JSON.
- `results/exp9/localisation_baseline.json`: all numbers in this file, including the per-qubit adjacency sizes.
- `results/exp9/regen_streams.py`: regenerates the streams and the nominal fingerprint window, and runs the checks above.

## Reproduce

```
# environment with the manifest's versions: python 3.11, stim==1.16.0, pymatching==2.4.0, numpy==2.4.6, torch (CPU is enough)
python results/exp9/regen_streams.py <dir>                        # ~20 s; prints ALL OK
python results/exp9/localisation_baseline.py --streams <dir>      # ~10 s
```

## Appendix: reference = exact nominal detection probabilities

| regime | phase | NF seed | NF hit rate | diff (baseline − NF) | steps: baseline better / NF better / tie | p (steps) | events: baseline better / NF better / tie | p (events) |
|---|---|---|---|---|---|---|---|---|
| bursts | drift | s0 | 0.982 | -0.155 | 1 / 80 / 51 | **7e-23** | 1 / 20 / 12 | **2e-05** |
| bursts | drift | s1 | 0.979 | -0.152 | 1 / 79 / 52 | **1e-22** | 1 / 20 / 12 | **2e-05** |
| bursts | drift | s2 | 0.980 | -0.153 | 1 / 79 / 52 | **1e-22** | 1 / 20 / 12 | **2e-05** |
| bursts | onset | s0 | 1.000 | -0.159 | 0 / 18 / 11 | **8e-06** | 0 / 18 / 11 | **8e-06** |
| bursts | onset | s1 | 0.993 | -0.152 | 0 / 17 / 12 | **2e-05** | 0 / 17 / 12 | **2e-05** |
| bursts | onset | s2 | 0.993 | -0.152 | 0 / 17 / 12 | **2e-05** | 0 / 17 / 12 | **2e-05** |
| bursts | later | s0 | 0.977 | -0.153 | 1 / 62 / 40 | **1e-17** | 1 / 20 / 12 | **2e-05** |
| bursts | later | s1 | 0.975 | -0.151 | 1 / 62 / 40 | **1e-17** | 1 / 20 / 12 | **2e-05** |
| bursts | later | s2 | 0.977 | -0.153 | 1 / 62 / 40 | **1e-17** | 1 / 20 / 12 | **2e-05** |
| bursts_big | drift | s0 | 0.935 | -0.099 | 27 / 57 / 32 | 0.0014 | 10 / 15 / 4 | 0.42 |
| bursts_big | drift | s1 | 0.953 | -0.116 | 15 / 56 / 45 | **1e-06** | 6 / 14 / 9 | 0.12 |
| bursts_big | drift | s2 | 0.913 | -0.077 | 36 / 52 / 28 | 0.11 | 9 / 13 / 7 | 0.52 |
| bursts_big | onset | s0 | 0.931 | -0.091 | 10 / 15 / 4 | 0.42 | 10 / 15 / 4 | 0.42 |
| bursts_big | onset | s1 | 0.935 | -0.095 | 6 / 14 / 9 | 0.12 | 6 / 14 / 9 | 0.12 |
| bursts_big | onset | s2 | 0.905 | -0.065 | 9 / 13 / 7 | 0.52 | 9 / 13 / 7 | 0.52 |
| bursts_big | later | s0 | 0.937 | -0.102 | 17 / 42 / 28 | 0.0015 | 6 / 14 / 9 | 0.12 |
| bursts_big | later | s1 | 0.958 | -0.124 | 9 / 42 / 36 | **3e-06** | 3 / 14 / 12 | 0.013 |
| bursts_big | later | s2 | 0.915 | -0.080 | 27 / 39 / 21 | 0.18 | 9 / 13 / 7 | 0.52 |
| hot_qubit | drift | s0 | 1.000 | +0.000 | 0 / 0 / 500 | 1 | 0 / 0 / 5 | 1 |
| hot_qubit | drift | s1 | 1.000 | +0.000 | 0 / 0 / 500 | 1 | 0 / 0 / 5 | 1 |
| hot_qubit | drift | s2 | 1.000 | +0.000 | 0 / 0 / 500 | 1 | 0 / 0 / 5 | 1 |
| hot_qubit | onset | s0 | 1.000 | +0.000 | 0 / 0 / 5 | 1 | 0 / 0 / 5 | 1 |
| hot_qubit | onset | s1 | 1.000 | +0.000 | 0 / 0 / 5 | 1 | 0 / 0 / 5 | 1 |
| hot_qubit | onset | s2 | 1.000 | +0.000 | 0 / 0 / 5 | 1 | 0 / 0 / 5 | 1 |
| hot_qubit | later | s0 | 1.000 | +0.000 | 0 / 0 / 495 | 1 | 0 / 0 / 5 | 1 |
| hot_qubit | later | s1 | 1.000 | +0.000 | 0 / 0 / 495 | 1 | 0 / 0 / 5 | 1 |
| hot_qubit | later | s2 | 1.000 | +0.000 | 0 / 0 / 495 | 1 | 0 / 0 / 5 | 1 |
