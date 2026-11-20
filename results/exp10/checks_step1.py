"""STEP 1 checks for round 10, run before the full run. Writes only under results/exp10/checks/.

1b. My eval path (eval_streams.neural_errors, ensemble_eval's functions) on the saved
    data/d5_wide bursts stream, stream seed 0, adaptive-stale ensemble, frozen fingerprint
    from the dataset's nominal window: bit-identical to results/exp6/d5/st0.
3.  Oracle MWPM on phys_static sub-steps equals calibrated static MWPM's predictions.
4.  The frozen fingerprint is identical for all three ensemble members (1000 shots).
+.  substep_model rebuilt from saved arrays equals make_noise_state's own state.model.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from circuits import make_circuit  # noqa: E402
from dataset import Dataset, unpack_det  # noqa: E402
from decoder import DetectorGrid, Fingerprint  # noqa: E402
from ensemble_eval import Gpu  # noqa: E402
from eval_streams import N_DET, calibrated_circuit, ckpt_path, substep_model  # noqa: E402
from eval_streams import neural_errors  # noqa: E402
from generate_streams import noiseless_d5  # noqa: E402
from physics_noise import CHOSEN_SI1000_P, make_noise_state  # noqa: E402

OUT = "results/exp10/checks"


def check_1b(gpu, grid):
    ds = Dataset("data/d5_wide")
    s = ds.load(ds.manifest["streams"]["bursts"])
    w = ds.load(ds.manifest["test_sets"]["nominal"])["window_det"]
    frozen = gpu.bits(torch.from_numpy(w).to("cuda"), N_DET)
    cks = [ckpt_path("adaptive_stale", i) for i in (0, 1, 2)]
    got = neural_errors(cks, gpu, grid, s["det"], s["obs"], frozen)
    old = np.load("results/exp6/d5/st0/ens_adaptive_stale_d5_long_errors.npz")
    diff = total = 0
    for kind in ("live", "frozen"):
        for who in ("ens", "m0", "m1", "m2"):
            mine = np.packbits(got[kind, who].cpu().numpy(), axis=-1)
            ref = old[f"bursts|{kind}|{who}"]
            assert mine.shape == ref.shape, (kind, who, mine.shape, ref.shape)
            diff += int((mine != ref).sum())
            total += ref.size
    print(f"CHECK 1b: my path vs exp6 bits: {diff} differing bits of {total} "
          f"-> {'BIT-IDENTICAL' if diff == 0 else 'MISMATCH'}")
    return diff == 0


def check_3():
    base = noiseless_d5()
    cal_circuit = calibrated_circuit(base)
    from circuits import matcher_from_circuit
    from eval_streams import oracle_errors

    mwpm_cal = matcher_from_circuit(cal_circuit)
    state = make_noise_state(base, "phys_static", device_seed=0, stream_seed=0, n_substeps=1500)
    arrays = {"qubits": state.qubits, "m": state.m, "burst": state.burst,
              "coupler_pairs": np.empty((0, 2), dtype=np.int64), "coupler_on": state.coupler_on}
    # 2000 fresh shots of the static device; the oracle sees them as one pseudo sub-step list
    det, _ = cal_circuit.compile_detector_sampler(seed=7).sample(shots=2000, separate_observables=True)
    det = det[None, :, :]
    oracle, n_circuits = oracle_errors(base, arrays, det)
    cal = np.asarray(mwpm_cal.decode_batch(det[0])).reshape(2000, -1)[:, 0]
    same = int((oracle[0] == cal).sum())
    print(f"CHECK 3: oracle vs calibrated MWPM on 2000 static-device shots: {same}/2000 identical; "
          f"oracle saw {n_circuits} distinct circuit(s) over 1500 sub-steps -> "
          f"{'PASS' if same == 2000 and n_circuits == 1 else 'FAIL'}")
    return same == 2000 and n_circuits == 1


def check_model_rebuild():
    base = noiseless_d5()
    state = make_noise_state(base, "phys_mixed", device_seed=0, stream_seed=3, n_substeps=1500)
    arrays = {"qubits": state.qubits, "m": state.m, "burst": state.burst,
              "coupler_pairs": np.asarray(state.coupler_pairs, dtype=np.int64).reshape(-1, 2),
              "coupler_on": state.coupler_on}
    ok = True
    for t in (0, 1, 42, 749, 1499):
        a, b = substep_model(arrays, t), state.model(CHOSEN_SI1000_P, t)
        ok &= (a.m == b.m and a.burst == b.burst and a.coupler == b.coupler and a.p == b.p)
    print(f"CHECK +: substep_model(saved arrays) == state.model(t) for t in 0,1,42,749,1499: "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_4(gpu, grid):
    from decoder import load_checkpoint

    with np.load("data/phys10/calibration.npz") as cal:
        bits1000 = gpu.bits(torch.from_numpy(cal["det"][:1000]).to("cuda"), N_DET)
    fps = []
    for s in (0, 1, 2):
        _, cfg = load_checkpoint(ckpt_path("adaptive_stale", s), "cpu")
        fpr = Fingerprint(cfg, grid, "cuda")
        fps.append(fpr.of_windows(bits1000).cpu().numpy())
    same = all((fps[0] == fps[i]).all() for i in (1, 2))
    print(f"CHECK 4: frozen fingerprint of 1000 calibration shots identical across the 3 members: "
          f"{'PASS' if same else 'FAIL'} (channels shape {fps[0].shape})")
    return same


def main():
    os.makedirs(OUT, exist_ok=True)
    nominal = make_circuit(5, 5, 0.003)
    grid = DetectorGrid(nominal)
    gpu = Gpu(grid, N_DET, "cuda")
    ok = check_1b(gpu, grid)
    ok &= check_3()
    ok &= check_model_rebuild()
    ok &= check_4(gpu, grid)
    print(f"\nSTEP 1 checks: {'ALL PASS' if ok else 'FAILURE'}")


if __name__ == "__main__":
    main()
