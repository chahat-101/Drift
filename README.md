# Drift

A neural surface-code decoder that adapts to drifting noise from its own recent syndromes.

Noise on real hardware moves. Qubits wander, defects switch on and off, and radiation events hit clusters of qubits, so a decoder tuned to a fixed noise model gets worse. Drift decodes with a 3D CNN whose second input channel is a fingerprint, the firing rate of each detector over the last 2000 shots. It needs no calibration input and no retraining. At d=5, adaptive models trained with stale fingerprints beat static MWPM and the best WindowDEM on burst drift in every seed, and still trail oracle MWPM.

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

d=3 (`notes/02` and `notes/03` list every command):

```
python make_dataset.py --distance 3 --rounds 3 --name d3_wide
python make_dataset.py --distance 3 --rounds 3
python make_dataset.py --streams-only --name d3_r3_p0.003_s0 --stream-seeds 1,2,3,4 --regimes hot_qubit,bursts,bursts_big,telegraph
python train.py --data data/d3_wide --epochs 2 --adaptive --stale-prob 0.3 --seed 0 --out decoder_adaptive_d3_stale_s0.pt   # seeds 0-2
python eval.py --data data/d3_r3_p0.003_s0 --regime bursts --stream-seed 0 --update-every 200 --tag s0 --frozen-fingerprint \
    --window-sizes 500,1000,2000,5000,10000 --checkpoint decoder_adaptive_d3_stale_s0.pt --out results/exp2
python report.py --data data/d3_wide --exp2
```

d=5 (`notes/04` to `notes/06`):

```
scripts/make_data_d5_x5.sh        # d5_wide_x5 and d5_nominal_x5, 150000 train chunks each
python train.py --data data/d5_wide_x5 --adaptive --stale-prob 0.3 --epochs 2 --val-every 5000 --save-every 30000 \
    --device cuda --compile --resume --seed 0 --out decoder_adaptive_stale_d5_long_s0.pt
bash scripts/run_seeds.sh         # seeds 1 and 2: trains and evaluates, then shuts the machine down
python analysis/learning_curves.py && python analysis/seeds_d5.py
```

## Results

d=5, p=0.003, 300k training steps, mean ± std over 3 model seeds (LER on drifted steps, pooled over 5 streams; adaptive-stale uses the live fingerprint). The best WindowDEM history is picked per column after the fact. From `notes/06_d5_three_seeds.md`.

| decoder | nominal test | bursts drift | bursts_big drift | hot_qubit drift |
|---|---|---|---|---|
| static MWPM | 0.00329 | 0.0439 | 0.0980 | 0.0181 |
| best WindowDEM | – | 0.0356 | 0.0896 | 0.0108 |
| oracle MWPM | 0.00329 | 0.0197 | 0.0426 | 0.0103 |
| static-wide | 0.00225 ± 0.00009 | 0.0258 ± 0.0021 | 0.0757 ± 0.0153 | 0.0106 ± 0.0005 |
| adaptive-stale | 0.00209 ± 0.00005 | 0.0184 ± 0.0008 | 0.0548 ± 0.0059 | 0.0077 ± 0.0004 |

## Repository layout

```
circuits.py  drift.py            Stim circuits, drift regimes, stream sampling
baselines.py                     static MWPM, oracle MWPM, WindowDEM
decoder.py                       3D CNN decoder, static and adaptive
dataset.py  make_dataset.py      fixed datasets with hash-checked manifests
train.py  eval.py                training and streaming evaluation
curve_eval.py  latency.py        checkpoint scoring for learning curves, decode latency
report.py                        d=3 and d=5 comparison tables
stale_weights.py  threshold.py   demos: cost of stale weights, threshold plot
analysis/                        tables and figures for the d=5 runs
scripts/                         run scripts for the d=5 experiments
notes/                           experiment write-ups, 01 to 06
results/                         eval outputs, logs, tables, figures
```


