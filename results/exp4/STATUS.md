# Round 4 status audit (2026-11-08 06:46 UTC)

Audit only: nothing was started, resumed or edited apart from this file.

## 1. Running now
| pid | command | started (UTC) | CPU % |
|---|---|---|---|
| 60493 | `python results/exp4/run_queue.py` | 2026-11-07 19:00:16 | 0.0 (idle loop) |

No train.py, eval.py or curve_eval.py is running. GPU: **0 % util, 0 MiB / 23034 MiB**. Load average 0.00.

## 2. Training (all h48/l4 unless noted; step checkpoints every 30k)
| model | seed | step ckpts present | last step | status |
|---|---|---|---|---|
| adaptive-wide | 0 | 30k–300k (10/10) + final | 300000 | finished 13:19 |
| static-nominal | 0 | 30k–300k + final | 300000 | finished 14:23 |
| adaptive-stale | 0 | 30k–300k + final | 300000 | finished 15:29 |
| static-wide | 0 | 30k–300k + final | 300000 | finished 16:34 |
| adaptive-stale h64/l6 | 0 | 30k–300k + final | 300000 | finished 18:58 |
| adaptive-wide | 1 | 30k–300k + final | 300000 | finished 20:13 |
| adaptive-wide | 2 | 30k–300k + final | 300000 | finished 21:16 |
| static-nominal | 1 | 30k–300k + final | 300000 | finished 22:19 |
| static-nominal | 2 | 30k–300k + final | 300000 | finished 23:22 |
| adaptive-stale | 1, 2 | none | – | never started (12 h cap) |
| static-wide | 1, 2 | none | – | never started (12 h cap) |

No crashes: no Traceback, OOM, NaN or "killed" in any train log. Every log ends with its final test table.

## 3. Evals
- Learning-curve evals (`results/exp4/curves/`): **50/50** seed-0 step checkpoints have a JSON.
- Full evals of the seed-0 finals (`results/exp4/full/st0..st4`): **25/25** present.
- **Missing:** full evals of the seeds-1/2 finals, 4 checkpoints × 5 stream seeds = 20 JSONs:
  `decoder_adaptive_wide_d5_long_s{1,2}`, `decoder_static_nominal_d5_long_s{1,2}` in st0–st4.
  (The plan never had curve evals for seeds 1/2 step checkpoints.)

## 4. Data
| dataset | manifest | size | check |
|---|---|---|---|
| data/d5_wide_x5 | present | 2.3 GB | test sets and streams byte-identical to d5_wide (24 files). `val.npz` and `stream_wander.npz` differ only in float64 `scale` labels, by ≤ 1 ulp (laptop vs VM libm). Shots are identical. |
| data/d5_nominal_x5 | present | 393 MB | Same story: only `stream_wander.npz` differs, by the 1-ulp float label. Extra stream seed files are expected. |

## 5. Queue
- `results/exp4/queue_done`: **does not exist**.
- Part C branch: **STILL FALLING** (`partC_decision.json`). All 5 models improved on bursts_big drift by >2 SE from 240k to 300k steps; 0/5 improved on nominal. h64/l6 did not "clearly win", so the h48/l4 seeds were queued.
- Last queue log lines (`queue.out`):
```
19:27:17 eval ... decoder_adaptive_stale_h64l6_d5_long_s0.pt ... --stream-seed 4 ... --out results/exp4/full/st4
20:13:57 finished adaptive_wide s1 rc=None
20:13:57 start ... --seed 2 --out decoder_adaptive_wide_d5_long_s2.pt
21:17:17 finished adaptive_wide s2 rc=0
21:17:17 start ... d5_nominal_x5 ... --seed 1 --out decoder_static_nominal_d5_long_s1.pt
22:19:58 finished static_nominal s1 rc=0
22:19:58 start ... d5_nominal_x5 ... --seed 2 --out decoder_static_nominal_d5_long_s2.pt
23:22:18 finished static_nominal s2 rc=0
```
(Nothing has been logged since then. The earlier lines are seed-0 full-eval launches.)

## 6. Wasted time
- Round start: 2026-11-07 11:37 UTC (`t_start`). The GPU logger began at 12:10, when the queue started.
- Last useful job finished: **2026-11-07 23:22 UTC** (static-nominal s2), 11.76 h in.
- GPU below 10 % util: **7.4 h** (443 one-minute samples). Nearly all of it is 23:23 → 06:46, with the VM idle.
- **Why the VM did not shut down:** a deadlock bug in `run_queue.py`. After static-nominal s2, the 12 h cap correctly blocked adaptive-stale s1, but that job and three more stayed in `q.train`. The `seeds12 → final` step only fires when `not q.train`, so it never fired. The seeds-1/2 full evals were never queued and `queue_done` was never written. My shutdown step waited on `queue_done`, with no timeout and no idle-GPU check, so the report was never finished and `sudo shutdown` never ran. My fault: the cap should have cleared the training list, and my wait should have had a time limit.

## 7. Report
`results/report4.md` exists (last edited 19:31 UTC) but is **incomplete**. Seven placeholders are unfilled: SEEDS, GPU-UTIL, CURVE-TABLES, ANSWERS, SEEDS12, SEEDS-LIMIT, FILES. The Part A, decisions, curves, Part C, limitations and next-steps sections are drafted using seed-0 numbers.

## (a) Usable now for the 5 questions
- Q1 (static-nominal vs MWPM on nominal): seed-0 curves plus final test sets for seeds 0/1/2, already in the train logs' final test tables.
- Q2 (neural vs static MWPM / best WindowDEM on bursts_big drift): seed-0 finals, all 5 models, 4 regimes × 5 streams.
- Q3 (adaptive vs oracle on later bursts): seed-0 adaptive-wide and adaptive-stale, 5 streams.
- Q4 (live vs frozen with training): seed-0 curves, 10 checkpoints per model.
- Q5 (h64/l6 vs h48/l4): seed-0, same budget, full eval. This was always seed-0 only.
- GPU-efficiency section: all data present (`gpu_util.csv`, speed/concurrency logs).

## (b) Missing
1. Full evals of adaptive-wide s1/s2 and static-nominal s1/s2: 20 curve_eval runs. Needed for 3-seed answers to Q1–Q3 for those two models.
2. Seeds 1/2 for adaptive-stale and static-wide. Never trained because of the 12 h cap, so those stay seed-0 only.
3. Rerun `report4.py`, fill the 7 placeholders and the file list with sizes.
4. Shutdown.

## (c) Time to finish only the missing parts
- 20 full evals: about 1.3 min each serially (seed-0 timings), so **~25–30 min**. About 15 min if run 2 at a time on the 4 cores.
- report4.py plus filling the placeholders: **~20–30 min**.
- **Total ≈ 45–60 min** of VM time. No GPU training is needed.
- Optional (over the 12 h cap): seeds 1/2 for adaptive-stale and static-wide, 4 × ~1.07 h ≈ 4.3 h of GPU, plus ~15 min of evals.
