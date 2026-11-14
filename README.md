# Drift

A neural surface-code decoder that adapts to drifting noise from its own recent syndromes.

Noise on real hardware moves. Qubits wander, defects switch on and off, and radiation events hit clusters of qubits, so a decoder tuned to a fixed noise model gets worse. Drift decodes with a 3D CNN whose second input channel is a fingerprint: the firing rate of each detector over the last 2000 shots. It needs no calibration input and no retraining.

At d=5, a 3-seed ensemble of adaptive models beats static MWPM, the best WindowDEM and a static network trained on the same data, in every drift regime we test. It beats oracle MWPM (MWPM with the true noise) on bursts and hot-qubit drift and still trails it on large held-out bursts. Freezing the fingerprint removes most of the gain, so the gain comes from adaptation and not from a better decoder.

## Install

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

```
python make_dataset.py --distance 3 --rounds 3 --name d3_small --train-chunks 2000 --val-chunks 100 --test-shots 20000 --stream-steps 30
python train.py --data data/d3_small --epochs 1 --adaptive
python eval.py --data data/d3_small --regime bursts --update-every 200 --checkpoint decoder_adaptive_d3.pt
```

## Reproducing the results

d=3 (`notes/02_d3_stale_fingerprint.md` and `notes/03_d3_controls.md` list every command):

```
python make_dataset.py --distance 3 --rounds 3 --name d3_wide
python make_dataset.py --distance 3 --rounds 3
python make_dataset.py --streams-only --name d3_r3_p0.003_s0 --stream-seeds 1,2,3,4 --regimes hot_qubit,bursts,bursts_big,telegraph
python train.py --data data/d3_wide --epochs 2 --adaptive --stale-prob 0.3 --seed 0 --out decoder_adaptive_d3_stale_s0.pt   # seeds 0-2
python eval.py --data data/d3_r3_p0.003_s0 --regime bursts --stream-seed 0 --update-every 200 --tag s0 --frozen-fingerprint \
    --window-sizes 500,1000,2000,5000,10000 --checkpoint decoder_adaptive_d3_stale_s0.pt --out results/exp2
python report.py --data data/d3_wide --exp2
```

d=5 (`notes/04_d5_initial.md` to `notes/06_d5_three_seeds.md`):

```
scripts/make_data_d5_x5.sh        # d5_wide_x5 and d5_nominal_x5, 150000 train chunks each
python train.py --data data/d5_wide_x5 --adaptive --stale-prob 0.3 --epochs 2 --val-every 5000 --save-every 30000 \
    --device cuda --compile --resume --seed 0 --out decoder_adaptive_stale_d5_long_s0.pt
bash scripts/run_seeds.sh         # seeds 1 and 2: trains and evaluates, then shuts the machine down
python analysis/learning_curves.py && python analysis/seeds_d5.py
```

Later rounds each have a run script and a report script. The run scripts are meant for a cloud GPU: they skip finished jobs, write `STATUS.txt`, and shut the machine down when they finish or fail.

| round | what | run | tables | notes |
|---|---|---|---|---|
| 6 | seed ensembles, d=3 and d=5 | `run6.sh` (`ensemble_eval.py`) | `results/exp6/report6.py` | `notes/07_ensembles.md` |
| 7 | richer fingerprints at d=3 | `run7.sh` | `results/exp7/report7.py` | `notes/08_d3_fingerprints.md` |
| 8 | multiscale fingerprint at d=5 | `run8.sh` | `results/exp8/report8.py` | `notes/09_d5_multiscale.md` |
| 9 | noise-map head and FiLM at d=5 | `run9.sh` | `results/exp9/report9.py` | – |

Every round from 7 on has a decision rule fixed before the run and computed by the report script.

## Results

d=5, p=0.003, 300k training steps. LER on drifted steps, pooled over 5 streams. Ensembles average the logits of 3 seeds. The best WindowDEM history is picked per column after the fact, which flatters it. From `results/exp8/report8_tables.md`.

| decoder | nominal test | bursts | bursts_big | hot_qubit |
|---|---|---|---|---|
| static MWPM | 0.00329 | 0.0439 | 0.0980 | 0.0181 |
| best WindowDEM | – | 0.0356 | 0.0896 | 0.0108 |
| oracle MWPM | 0.00329 | 0.0197 | 0.0426 | 0.0103 |
| static-wide, ensemble | 0.00210 | 0.0224 | 0.0564 | 0.0098 |
| adaptive-stale, ensemble, frozen fingerprint | – | 0.0233 | 0.0576 | 0.0098 |
| adaptive-stale, ensemble | 0.00204 | 0.0170 | 0.0457 | 0.0073 |

bursts_big is held out: 8-qubit bursts at strength 0.1, larger than anything in training. Single seeds vary a lot on it (adaptive-stale 0.0548 ± 0.0059), which is why the table uses ensembles. Paired McNemar tests for every comparison are in the report tables.

Known weak spot: in the first step after a burst ends the fingerprint is out of date, and the adaptive ensemble is slightly worse than the static-wide one (+0.0003 on bursts, +0.0006 on bursts_big), though still better than static MWPM.

## What didn't help

- **Multiscale fingerprint** (200- and 2000-shot windows and their difference). At d=3 it cut onset and after-burst errors by 8-33%. At d=5 the baseline already matches the oracle at onset on bursts (it still trails on bursts_big, where multiscale didn't help either), and it costs 6-18% on nominal steps (cause untested). Fails its decision rule at both distances (`results/exp7`, `results/exp8`).
- **Pairwise co-firing channels.** Best on small, local drift, worse on large bursts (`results/exp7`).
- **Noise-map head and FiLM.** No significant change in decoding accuracy at p < 0.001 (weak losses on bursts_big, p 0.03-0.05) (`results/exp9`).

## Drift localisation

The noise-map head predicts, from syndromes alone, which qubits have extra noise. With FiLM it finds 98% of the hot qubits on burst steps, against 83% for a training-free rule that ranks qubits by the excess firing of nearby detectors (better on 20 of 33 bursts, worse on 1). On the held-out large bursts it scores 93%, but its advantage over the simple rule is not significant. It also underestimates noise stronger than anything seen in training. See `results/exp9/localisation_baseline.md`.

## Repository layout

```
circuits.py  drift.py            Stim circuits, drift regimes, stream sampling
baselines.py                     static MWPM, oracle MWPM, WindowDEM
decoder.py                       3D CNN decoder: static, adaptive, fingerprint variants, noise head, FiLM
dataset.py  make_dataset.py      fixed datasets with hash-checked manifests
train.py  eval.py                training and streaming evaluation
ensemble_eval.py  noise_eval.py  GPU evaluation of seed ensembles and of the noise-map head
curve_eval.py  latency.py        checkpoint scoring for learning curves, decode latency
report.py                        d=3 and d=5 comparison tables
stale_weights.py  threshold.py   demos: cost of stale weights, threshold plot
run*.sh  watchdog.sh             cloud run scripts with automatic shutdown
analysis/                        tables and figures for the d=5 runs
scripts/                         data generation and seed runs for d=5
data/                            dataset manifests (the data itself is not in the repo)
notes/                           experiment write-ups
results/                         eval outputs, logs, tables, figures
```

Datasets and checkpoints are not in the repo. Every dataset can be regenerated from its `manifest.json`, which records the seeds and library versions. Stim only promises identical samples on the same version, so use the versions listed there.
