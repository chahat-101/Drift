# d=3 baseline: classical decoders and the first neural models

## Classical decoders under drift

Static MWPM, WindowDEM and oracle MWPM at d=5, p=0.003, 120 steps × 2000 shots. LER is over the drifted steps.

| Regime | Static | WindowDEM | Oracle |
|---|---|---|---|
| hot_qubit | 0.0183 | 0.0112 | 0.0105 |
| bursts | 0.0404 | 0.0456 | 0.0151 |
| telegraph | 0.0247 | 0.0250 | 0.0249 |

- hot_qubit (persistent local drift). WindowDEM almost matches the oracle. Classical tracking handles slow, lasting changes, so a neural decoder has to at least tie here.
- bursts. WindowDEM is worse than doing nothing. Its history still holds the burst after the burst ends, so it decodes with stale weights in both directions. The oracle is 2.7x better than static, which is the headroom for an adaptive decoder.
- telegraph and wander (uniform scale changes). Static, WindowDEM and oracle tie. MWPM barely changes when every error rate rises together, so a decoder can only win on drift that changes the shape of the noise.

`stale_weights.py` measures the same effect for two hot qubits with PyMatching alone (clean, blind and aware weights).

## First neural models at d=3

The first adaptive-vs-static comparison on `bursts` (onset 0.117 adaptive vs 0.089 static; later 0.053 adaptive vs 0.058 oracle) used the wide seed-0 models, `decoder_static_d3_wide_s0` and `decoder_adaptive_d3_wide_s0`. `results/bursts_d3_r3.csv` is exactly what these two produce on the `bursts` stream of `data/d3_r3_p0.003_s0`. The phase definitions in `02_d3_stale_fingerprint.md` reproduce those numbers from it.

The models trained on `data/d3_r3_p0.003_s0` itself (seed 0), on the same stream with update every 2000 shots (`results/bursts_d3_r3_u2000_D_s0.json`):

| | LER | drift | onset | later | after |
|---|---|---|---|---|---|
| decoder_static_d3_s0 | 0.0230 | 0.0985 | 0.0988 | 0.0984 | 0.0059 |
| decoder_adaptive_d3_s0 | 0.0225 | 0.0938 | 0.1156 | 0.0865 | 0.0172 |
| oracle MWPM | 0.0165 | 0.0590 | 0.0615 | 0.0581 | 0.0069 |

- The onset regression is the same (0.116 vs 0.099).
- The adaptive model trained on this narrower data is far from the oracle on later steps (0.0865 vs 0.0581). The wider training range is what lets the adaptive model beat oracle MWPM on later burst steps.
- The original `eval.py` gives per-step numbers identical to `eval.py --update-every 2000`.
