# Drift

A neural surface-code decoder that adapts to drifting noise using only its own recent syndromes. It gets no calibration input and needs no retraining.

Qubit noise on real hardware moves around. Qubits wander slowly, defects switch on and off, and radiation events hit clusters of qubits. A decoder tuned to yesterday's noise model gets worse. Drift asks whether a small neural decoder can read the drift from the statistics of its own syndrome stream and do better than classical decoders that try the same thing.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Everything except `decoder.py` and `train.py` runs without torch.

## Files

| File | What it does |
|---|---|
| `stale_weights.py` | Single-file demo: how much worse matching gets when two qubits go bad and the decoder doesn't know. Start here. |
| `threshold.py` | Reproduces the surface-code threshold plot with Sinter. Use it to check your setup. |
| `circuits.py` | Builds Stim surface-code circuits with uniform noise plus optional "hot" qubits. |
| `drift.py` | Drift simulator with six regimes: static, wander, telegraph, hot_qubit, bursts, and the held-out bursts_big (8-qubit clusters at hot_p 0.1, outside every training range). |
| `baselines.py` | Classical decoders: static PyMatching, oracle PyMatching, and the sliding-window correlation estimator (WindowDEM). |
| `decoder.py` | 3D-CNN neural decoder. Its adaptive mode takes a syndrome fingerprint as a second input channel. |
| `train.py` | Trains the neural decoder on randomly drifting Stim data, sampled fresh or read from a saved dataset. |
| `eval.py` | Runs all decoders on the same drifting stream. Reports LER overall, during drift, at drift onset, on later drifted steps and right after drift. Writes a CSV, a summary JSON and a plot to `results/`. |
| `make_dataset.py` | Generates a fixed dataset (train/val chunks, test sets, drift streams) in `data/`. |
| `dataset.py` | Loads saved datasets, checks file hashes, and checks that a model matches the dataset. |
| `report.py` | Collects the stale-fingerprint experiment into tables and plots for `results/report.md`. |

## Commands to run first

```
python stale_weights.py
python threshold.py
python eval.py --regime hot_qubit
python eval.py --regime bursts
python train.py --distance 3 --rounds 3 --steps 3000
python train.py --distance 3 --rounds 3 --steps 3000 --adaptive
python eval.py --regime bursts --distance 3 --rounds 3 \
    --checkpoint decoder_static_d3.pt --checkpoint decoder_adaptive_d3.pt
```

## Datasets

Without `--data`, `train.py` and `eval.py` sample new shots on every run, so the numbers change from run to run. To compare models fairly, generate a fixed dataset once and score every model and baseline on the same shots:

```
python make_dataset.py --distance 3 --rounds 3     # -> data/d3_r3_p0.003_s0/, ~0.1 GB, under a minute
python make_dataset.py --distance 5                # -> data/d5_r5_p0.003_s0/, ~0.5 GB, a few minutes
```

A dataset folder contains:

- `train.npz`, `val.npz`: training chunks (default 30000 / 1000). Each chunk has one random `NoiseState`, 512 training shots and 2000 fingerprint shots, all from that same noise.
- `test_<state>.npz`: the fixed test states (nominal, scale x2, 2 hot qubits), 200000 shots plus 2000 fingerprint shots each.
- `stream_<regime>.npz`: one drifting stream per regime (150 steps x 2000 shots), with the noise state of every step.
- `manifest.json`: generation args, per-split seeds, Stim/PyMatching versions, array shapes and a SHA-256 of every file.

Detection events and observables are bit-packed and compressed. Each split gets its own seed derived from `--seed`, so train, val, test and the streams are independent samples. Run `python make_dataset.py --help` to see the sizes. Stim only guarantees the same samples for a given seed on the same Stim version and machine, so share the folder itself rather than the seed.

Train and evaluate against it:

```
python train.py --data data/d3_r3_p0.003_s0 --epochs 2 --adaptive    # val loss every --val-every steps
python eval.py --data data/d3_r3_p0.003_s0 --regime bursts \
    --checkpoint decoder_static_d3.pt --checkpoint decoder_adaptive_d3.pt
```

To add streams for regimes that an existing dataset lacks (e.g. `bursts_big`), without touching its other files:

```
python make_dataset.py --streams-only --name d3_r3_p0.003_s0
```

Useful flags:

- `train.py --stale-prob 0.3` (adaptive only): with that probability, a batch gets a fingerprint that doesn't match its shots. Half of these come from another chunk; the other half are part another chunk, part this one. This teaches the model not to over-trust its fingerprint right after the noise changes.
- `eval.py --update-every 200`: decode each step in sub-batches of 200 shots and let decoders observe after each one, so fingerprints and WindowDEM update faster. The default (= shots per step) is the original behaviour.
- `eval.py --tag s0`: suffix for output files, which are named `<regime>_d<d>_r<r>_u<update-every>[_st<stream seed>][_<tag>]`. Every run also writes `_errors.npz` with per-shot error bits for paired tests.
- `make_dataset.py --streams-only --stream-seeds 1,2,3,4 --regimes bursts,telegraph`: add extra independent streams per regime. `eval.py --stream-seed N` replays one; the default is the main stream.
- `eval.py --frozen-fingerprint`: also run each adaptive checkpoint with a fixed fingerprint from the nominal test set (`<checkpoint>_frozen`). Comparing it with the live model separates adaptation from "is just a better decoder".
- `eval.py --window-sizes 500,2000,10000`: one WindowDEM per history size (`window_<size>`), for a fair baseline.
- `eval.py --device cuda`: run the neural decoders on the GPU (default CPU; the two can disagree on the rare shot whose logit is at the threshold).
- `train.py --nominal-only`: train on nominal noise only (on the fly), a specialist baseline for undrifted noise.
- `report.py --exp2`: pools experiment 2 over streams and seeds with paired McNemar tests (`results/exp2/report2_tables.md`).

With `--data`, distance, rounds and p (plus batch/window for training, steps/shots for eval) come from the manifest. The scripts stop if you pass a conflicting value, if a file's hash no longer matches, or if a checkpoint was trained for a different distance, rounds or p. `train.py --data` reshuffles the chunk order every epoch and scores on the saved test sets. `eval.py --data` replays the saved stream, so repeated runs print identical numbers.

## Results so far (d=5, p=0.003, 120 steps x 2000 shots)

| Regime | Static | WindowDEM | Oracle |
|---|---|---|---|
| hot_qubit (LER during drift) | 0.0183 | 0.0112 | 0.0105 |
| bursts (LER during drift) | 0.0404 | 0.0456 | 0.0151 |
| telegraph (LER during drift) | 0.0247 | 0.0250 | 0.0249 |

What these show:

- **Persistent local drift (hot_qubit):** the window estimator almost matches the oracle. Classical tracking already solves slow, lasting changes, so the neural decoder has to at least tie here.
- **Short bursts:** the window estimator is worse than doing nothing. Its history still holds the burst after the burst ends, so it decodes with stale weights in both directions. The oracle is 2.7x better than static, and that gap is the opening for this project.
- **Uniform scale changes (telegraph, wander):** static, window and oracle all tie. MWPM barely cares when every error rate rises together, so a decoder only wins on drift that changes the shape of the noise. Keep this in mind when designing experiments.

The neural decoder in this repo has only been smoke-tested for a few hundred steps and does not beat MWPM yet. Expect to need tens of thousands of steps, more data and probably a better architecture. A GPU (Colab or Kaggle) helps from d=5 up.

## Roadmap

1. **Week 1–3:** run everything above. Read the AlphaQubit papers and the FiLM-decoder paper (arXiv 2601.16123). Sweep `--window` for WindowDEM, because the best window size is part of a fair comparison.
2. **Week 4–6:** train the static neural decoder until it matches MWPM on nominal noise at d=3, then at d=5.
3. **Week 7–10:** make adaptation work, especially on bursts. Ideas:
   - a shorter fingerprint window
   - several windows at different timescales
   - FiLM conditioning in place of the extra input channel
   - a transformer or GNN in place of the CNN
   Add a "wide-range static" model trained on the same noise distribution without a fingerprint. It is the baseline that tests whether adaptation helps at all.
4. **Week 11–13:** run the full experiments: d=3, 5, 7, held-out drift patterns, recovery time after a jump, and decode latency. Optionally validate on Google's Willow data (Zenodo DOI 10.5281/zenodo.13273331, ~11.6 GB). Keep downloaded data in `data/`.
5. **Week 14–16:** write the paper.

## Notes

- `eval.py` is causal: each decoder decodes a step before it observes that step's shots. Keep it that way, or the comparison is not fair.
- Noise scales are rounded to 0.05 in `drift.py` so circuits can be cached.
