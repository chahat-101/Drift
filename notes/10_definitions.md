# Round 10 definitions (fixed before any results exist)

Scope: evaluate EXISTING trained d=5 models on physics-based drift (physics_noise.py).
No training. Existing checkpoints, datasets, results and
notes/10_physics_drift_hypotheses.md are never modified.

## Definitions (fixed before running)

- Streams: 6 regimes x stream seeds 0-4, device_seed 0, 1500 sub-steps x 200 shots each. Per sub-step, build the SI1000 circuit from the noise state and sample 200 shots with a Stim seed derived from (regime, stream_seed, sub-step). Save bit-packed det/obs plus the full noise-state arrays and events (bursts, TLS switch-ons, coupler switch-ons) in data/phys10/ with a manifest (seeds, versions, SHA-256 per file). Also save a calibration set: 2000 shots of the static device (separate seed), used for the frozen fingerprint.
- Decoders: calibrated static MWPM (DEM of the static device); nominal MWPM (DEM of make_circuit(5, 5, 0.003), reference only); oracle MWPM (true DEM per sub-step); WindowDEM windows 500, 1000, 2000, 5000, 10000 built on the calibrated DEM's matching graph (best chosen after the fact, reported as such); and the four neural ensembles plus their single seeds, adaptive ones live and frozen (frozen = fingerprint of the calibration set).
- Drift sub-steps per regime: phys_cosmic: max burst >= 0.005; phys_tls: any fluctuator on; phys_coupler: any coupler on; phys_slow: all sub-steps; phys_mixed: any of the above. Other sub-steps are "nominal". phys_static: all sub-steps nominal.
- Events (unit for sign tests): one burst (impact until max burst < 0.005); one TLS on-period of one fluctuator; one coupler on-period. Per event compare two decoders by error count; exact two-sided sign test, ties dropped.
- Burst bins (phys_cosmic and phys_mixed): sub-steps grouped by burst amplitude 0.2*exp(-age/25) of the youngest active burst: [0.05, 0.2], [0.0125, 0.05), [0.005, 0.0125).
- Hypothesis tests exactly as in notes/10_physics_drift_hypotheses.md (H1, H2, H3); McNemar exact two-sided on shared shots; p < 0.001 significant, 0.001-0.05 weak; no correction.

## Clarifications

Fixed now so nothing is chosen after seeing results. Flagged items (C1, C2) are
restated for confirmation before the run; everything else is a mechanical
consequence of the definitions above and the existing code.

### Models (identified by saved config, not filename)

All 12 are in ckpt/ (git-ignored via the existing `*.pt` rule), extracted from
~/drift_zenodo_data.tgz, SHA-256-verified against the tarball's SHA256SUMS.
Every one has distance=5, rounds=5, p=0.003, steps=300000, hidden=48, layers=4
and no `fingerprint` key, i.e. fingerprint "single" (decoder.py's default):

| type           | seeds 0, 1, 2                              | config that identifies it                          |
|----------------|--------------------------------------------|----------------------------------------------------|
| static-wide    | ckpt/decoder_static_wide_d5_long_s<N>.pt   | adaptive=False, window=2000, stale_prob=0.0        |
| adaptive-wide  | ckpt/decoder_adaptive_wide_d5_long_s<N>.pt  | adaptive=True, window=2000, stale_prob=0.0        |
| adaptive-stale | ckpt/decoder_adaptive_stale_d5_long_s<N>.pt | adaptive=True, window=2000, stale_prob=0.3        |
| static-nominal | ckpt/decoder_static_nominal_d5_long_s<N>.pt | adaptive=False, window=0, stale_prob=0.0           |

(Not used: decoder_adaptive_stale_h64l6_d5_long_s0.pt — hidden=64, layers=6,
seed 0 only; and the d5_60k / d5_curves_seed0 / d3 checkpoint sets.)

### Circuits and sampling

- Base circuit: stim.Circuit.generated("surface_code:rotated_memory_z",
  distance=5, rounds=5).flattened() (as test_physics_noise.noiseless_d5).
- Noise state per stream: make_noise_state(base, regime, device_seed=0,
  stream_seed=S, n_substeps=1500); SI1000 strength p = CHOSEN_SI1000_P = 0.0015.
- Sub-step t circuit: build_noisy_circuit(base, state.model(0.0015, t));
  test T8 (passing) guarantees the same detectors, coordinates, observable and
  DetectorGrid mapping as make_circuit(5, 5, 0.003), so trained models decode
  them unchanged.
- Stim sampler seed per sub-step:
  seed = int.from_bytes(hashlib.blake2b(
      f"{regime}|{stream_seed}|{substep}".encode(), digest_size=8), "big");
  compile_detector_sampler(seed=seed).sample(shots=200); det and obs recorded.
- Calibration set: the static device is time-invariant (m = device
  heterogeneity only; no slow/TLS/burst/coupler terms), circuit =
  build_noisy_circuit(base, make_noise_state(base, "phys_static",
  device_seed=0).model(0.0015, 0)); 2000 shots with sampler seed
  int.from_bytes(blake2b(b"phys10|calibration", digest_size=8), "big").

### Frozen fingerprint

Each adaptive model has window=2000 and fingerprint "single", so the frozen
fingerprint is exactly Fingerprint(cfg, grid).of_windows(calibration bits):
the per-detector firing rate over the 2000 calibration shots (x10, as in
decoder.Fingerprint), one shared fingerprint for all three ensemble members,
built the same way ensemble_eval.py builds its frozen fingerprint.

### Neural evaluation path

ensemble_eval.py's path, imported and reused (Gpu, window_fps, logits, errors):
ensemble = mean of the 3 seeds' logits > 0; causal — each 200-shot sub-batch is
decoded with the fingerprint of the last `window` shots before it (the member's
own window, live), then those shots enter the buffer; batch 16384; cuDNN TF32
on, matmul TF32 off. Static ensembles run once (live); adaptive ensembles run
live and frozen. Single seeds m0, m1, m2 come from the same forward passes.

### MWPM decoders

- Calibrated static MWPM: matcher_from_circuit of the static-device circuit
  defined above (the "calibrated DEM").
- Nominal MWPM: matcher_from_circuit(make_circuit(5, 5, 0.003)); reference
  only — it appears in the LER tables but in no hypothesis test.
- Oracle MWPM: matcher_from_circuit(build_noisy_circuit(base,
  state.model(0.0015, t))) per sub-step, cached per distinct circuit; on
  phys_static it is by construction the calibrated static MWPM.
- WindowDEM: baselines.WindowDEM(static-device circuit, window=W) for
  W in 500, 1000, 2000, 5000, 10000 (its matching graph and initial weights
  are the calibrated DEM's), min_shots=1000 (the existing default), observe
  after every 200-shot sub-batch. The best window is chosen after the fact
  and reported as such.

### Drift sub-steps (C1 — confirm)

- phys_cosmic: max over qubits of burst[t] >= 0.005.
- phys_tls: any (qubit, k) with tls_on[t, q, k].
- phys_coupler: any coupler_on[t, i].
- phys_slow: all sub-steps.
- phys_mixed: max burst >= 0.005 OR any fluctuator on OR any coupler on.
  The slow process is always present in phys_mixed but, exactly as in
  phys_slow alone, it does not by itself mark a sub-step as drift; otherwise
  phys_mixed would have no nominal sub-steps and the "nominal" column would
  be empty. (This is the one place "any of the above" needed a choice.)
- phys_static: all sub-steps nominal.

### Burst bins (phys_cosmic and phys_mixed)

A(t) = 0.2*exp(-age/25) of the youngest active burst, where an impact is
active at t if its age <= t and its amplitude is >= 0.005. Bins:
[0.05, 0.2], [0.0125, 0.05), [0.005, 0.0125). Sub-steps with no active burst
are not binned. If a sub-step is drift by the max-burst rule but has no active
burst (possible only through summed older bursts), it stays in the drift LER
and is counted and reported separately; if that count is zero everywhere, as
expected, nothing changes.

### Events (C2 definitions made explicit)

- Burst event: sub-steps [impact, end), end = first sub-step after the impact
  with max burst < 0.005 (or stream end). Two overlapping impacts give two
  overlapping events; the shots are counted in both.
- TLS event: one maximal run of consecutive sub-steps with tls_on[t, q, k]
  True, for one fluctuator (qubit, k).
- Coupler event: one maximal run of consecutive sub-steps with coupler_on[t, i]
  True, for one of the two chosen coupler pairs.
- Per event, two decoders are compared by their error counts over the event's
  shots; the exact two-sided sign test on n non-tied events with w wins is
  the exact binomial(0.5) two-sided tail (numerically the same formula as
  report.mcnemar_p, which is reused). Ties dropped. phys_static has no events;
  phys_slow has no sign-test events (all sub-steps are drift).

### Statistics

McNemar exact two-sided on shared shots (report.mcnemar_p, the same function
results/exp6/report6.py uses); p < 0.001 significant, 0.001-0.05 weak, no
multiple-testing correction. Hypothesis tests H1, H2, H3 exactly as in
notes/10_physics_drift_hypotheses.md.

### Outputs

- data/phys10/: one .npz per stream (30 streams) with bit-packed det/obs
  (packed as in dataset.py: det along the detector axis, obs along the shot
  axis), the full PhysicsNoiseState arrays, and the derived event windows;
  plus the calibration set and a manifest.json (seeds, versions, SHA-256 per
  file). Regeneration is byte-identical (checked in STEP 1).
- results/exp10/: per-shot error bits per decoder per stream, summaries,
  report10.py -> report10_tables.md and report10_data.json (per regime: LER on
  all / drift / nominal sub-steps and per burst bin for every decoder;
  McNemar and event-level sign tests for adaptive-stale ens. live vs frozen,
  adaptive-stale ens. vs calibrated static MWPM / best WindowDEM / oracle
  MWPM / static-wide ens., and each neural ensemble vs calibrated MWPM on
  phys_static; a header table marking H1, H2, H3 PASS/FAIL with the deciding
  numbers) and two plots: drift LER per regime for all decoders, LER vs burst
  bin.

### Pre-run checks (STEP 1, before the full run)

1. Reproduce an existing exp6 ensemble result bit for bit: ensemble_eval.py
   with ckpt/decoder_adaptive_stale_d5_long_s{0,1,2}.pt, data/d5_wide,
   --streams bursts --stream-seed 0 (the saved run configuration), compared
   against results/exp6/d5/st0/ens_adaptive_stale_d5_long_errors.npz.
2. Generate one 50-sub-step physics stream twice: byte-identical files.
3. Oracle MWPM on a phys_static sub-step equals calibrated static MWPM's
   predictions.
4. On 1000 shots, the frozen fingerprint is identical for all three ensemble
   members.
Then a time estimate for the full run; if it exceeds 4 hours, stop and report.
