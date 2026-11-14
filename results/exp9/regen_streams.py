"""Regenerate the data/d5_wide streams (bursts, bursts_big, hot_qubit; stream seeds 0-4) and the nominal
test set's fingerprint window from the manifest seeds, for results/exp9/localisation_baseline.py:

    python results/exp9/regen_streams.py <out dir>

Needs the manifest's library versions (stim 1.16.0, numpy 2.4.6; Stim samples are only reproducible on
the same version). Verifies the regenerated data against saved results:
  streams: per-step static-MWPM error counts vs results/exp3/d5/<regime>_d5_r5_u200_st<N>_s0.csv
           hot-qubit truth per step vs results/exp9/noise/noise_auxfilm_s0.npz
  test nominal: static-MWPM error count on the 200000 test shots vs results/exp4/curves (the
           fingerprint window is the next draw of the same rng; it has no saved result of its own)
Writes <out>/stream_<regime>_st<N>.npz (det packed, scale, hot_p, hot_qubits) and <out>/test_nominal_window.npz.
"""
import csv
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from circuits import matcher_from_circuit  # noqa: E402
from dataset import states_to_arrays  # noqa: E402
from drift import DriftStream, sample_stream, test_states  # noqa: E402

OUT = sys.argv[1]
m = json.load(open("data/d5_wide/manifest.json"))
a, seeds = m["args"], m["seeds"]
stream = DriftStream(a["distance"], a["rounds"], a["p"])
match = matcher_from_circuit(stream.nominal())
nf = np.load("results/exp9/noise/noise_auxfilm_s0.npz")
ok = True

for regime in ["bursts", "bursts_big", "hot_qubit"]:
    for st in range(5):
        stem = f"stream_{regime}" if st == 0 else f"stream_{regime}_seed{st}"
        rng = np.random.default_rng(seeds[stem])
        states, dets, errs = [], [], []
        for state, det, obs in sample_stream(stream, regime, a["stream_steps"], a["shots_per_step"], rng):
            states.append(state)
            dets.append(det)
            errs.append(int((match.decode_batch(det)[:, 0] != obs[:, 0]).sum()))
        rows = list(csv.DictReader(open(f"results/exp3/d5/{regime}_d5_r5_u200_st{st}_s0.csv")))
        saved = [round(float(r["static"]) * a["shots_per_step"]) for r in rows]
        arr = states_to_arrays(states)
        hot_truth = (nf[f"{regime}|st{st}|tmap"] > 0).sum(1)
        n_hot = (arr["hot_qubits"] >= 0).sum(1)
        same_mwpm = saved == errs
        same_hot = np.array_equal(hot_truth, n_hot)
        ok &= same_mwpm and same_hot
        print(f"{regime} st{st}: steps {len(errs)}, MWPM errors {sum(errs)} (saved {sum(saved)}), "
              f"per-step match {same_mwpm}, hot-qubit truth match {same_hot}", flush=True)
        np.savez_compressed(f"{OUT}/stream_{regime}_st{st}.npz", det=np.packbits(np.stack(dets), axis=-1), **arr)

# nominal test set: first draw = the 200000 test shots, second draw = the 2000 fingerprint shots
state = test_states(stream.nominal())["nominal"]
rng = np.random.default_rng(seeds["test_nominal"])
det, obs = stream.sample(state, a["test_shots"], rng.integers(2**31))
wdet, _ = stream.sample(state, a["test_window"], rng.integers(2**31))
err = match.decode_batch(det)[:, 0] != obs[:, 0]
c = json.load(open("results/exp4/curves/decoder_adaptive_stale_d5_long_s0_step300000.json"))
saved = round(c["tests"]["nominal"]["static"] * a["test_shots"])
same = int(err.sum()) == saved
ok &= same
print(f"test nominal: static MWPM errors {int(err.sum())} (saved {saved}, exp4 curves json), count match {same}")
np.savez_compressed(f"{OUT}/test_nominal_window.npz", window_det=wdet)
print("ALL OK" if ok else "MISMATCH")
