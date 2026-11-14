"""Check 2 of round 7: each new fingerprint channel by hand (numpy, CPU) against the GPU code.

    python results/exp7/check7.py     # prints max |GPU - numpy| per channel group; exits 1 on failure

Covers: one training chunk (GpuChunks.inputs, laid out on the grid), one stale mixed window
(GpuChunks.stale_fp), the eval causal buffer (ensemble_eval.window_fps at a sub-batch start)
and the frozen fingerprint (nominal test-set window).
"""
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from dataset import Dataset, unpack_det  # noqa: E402
from decoder import DetectorGrid, Fingerprint, pair_offsets  # noqa: E402
from drift import DriftStream  # noqa: E402
from ensemble_eval import Gpu, window_fps  # noqa: E402
from train import GpuChunks  # noqa: E402

TOL = 1e-5
nominal = DriftStream(3, 3, 0.003).nominal()
grid = DetectorGrid(nominal)
offsets = pair_offsets(grid, nominal)
cfg = {"adaptive": True, "fingerprint": "mspair", "short": 200, "pair_offsets": offsets, "pair_scale": 200.0}
# partners by hand from the detector coordinates (not decoder.pair_partners)
coords = nominal.get_detector_coordinates()
c = np.array([coords[i][:3] for i in range(grid.n)])
axes = [np.unique(c[:, k]) for k in range(3)]
pos = [(int(np.searchsorted(axes[2], v[2])), int(np.searchsorted(axes[0], v[0])), int(np.searchsorted(axes[1], v[1])))
       for v in c]
at = {p: i for i, p in enumerate(pos)}


def by_hand(W):
    """(11, n) channels of a 0/1 window (rows, n), plain numpy float64."""
    W = W.astype(np.float64)
    lo, sh = W.mean(0), W[-200:].mean(0)
    ch = [10 * lo, 10 * sh, 10 * (sh - lo)]
    for o in offsets:
        cov = np.zeros(grid.n)
        for i in range(grid.n):
            j = at.get(tuple(a + b for a, b in zip(pos[i], o)))
            if j is not None:
                cov[i] = (W[:, i] * W[:, j]).mean() - W[:, i].mean() * W[:, j].mean()
        ch.append(200.0 * cov)
    return np.array(ch)


def report(name, gpu, ref):
    g = gpu.detach().cpu().double().numpy()
    groups = {"long": [0], "short": [1], "short-long": [2], "covariances": list(range(3, 3 + len(offsets)))}
    errs = {k: float(np.abs(g[v] - ref[v]).max()) for k, v in groups.items()}
    ok = all(e < TOL for e in errs.values())
    print(f"{name:<34}" + "  ".join(f"{k} {e:.1e}" for k, e in errs.items()) + f"  {'OK' if ok else 'FAIL'}")
    return ok


dev = "cuda"
fpr = Fingerprint(cfg, grid, dev)
ds = Dataset("data/d3_wide")
chunks = ds.chunks("val")
data = GpuChunks(chunks, grid, dev, fpr, keep_windows=True)
ok = True
i, other = 17, 403
W = unpack_det(chunks.a["window_det"][i], ds.n_det)
ok &= report(f"train chunk {i} (fp)", data.fp[i], by_hand(W))
x, _ = data.inputs(i)  # (B, 12, T, X, Y): channel 1 + c at each detector's grid cell
lay = torch.stack([x[0, 1 + k].reshape(-1)[data.flat] for k in range(fpr.channels)])
lay_ok = bool((x[:, 1:] == x[:1, 1:]).all())  # same fingerprint for every shot of the batch
ok &= report(f"train chunk {i} (input grid)", lay, by_hand(W)) and lay_ok
print(f"{'':<34}fingerprint identical over the batch: {lay_ok}")
Wo = unpack_det(chunks.a["window_det"][other], ds.n_det)
rng = np.random.default_rng(5)
while True:  # draw until we get a mixed window (not the plain other window)
    state = rng.bit_generator.state
    if rng.random() >= 0.5:
        k = int(rng.uniform() * data.window)
        break
rng.bit_generator.state = state
f = data.stale_fp(i, other, rng)
ok &= report(f"stale mix k={k} (other[:k] + own[k:])", f, by_hand(np.concatenate([Wo[:k], W[k:]])))
# eval causal buffer
dsx = Dataset("data/d3_r3_p0.003_s0")
s = dsx.load(dsx.manifest["streams"]["bursts_big"])
steps, shots = s["det"].shape[:2]
det = unpack_det(s["det"], dsx.n_det).reshape(steps * shots, -1)
gpu = Gpu(grid, dsx.n_det, dev)
dg = torch.from_numpy(det.astype(np.uint8)).to(dev)
starts = [0, 200, 1800, 2200, 51 * 2000 + 400, 149 * 2000 + 1800]
F = window_fps(dg, starts, 2000, fpr)
for g, f in zip(starts, F):
    ref = by_hand(det[max(0, g - 2000):g]) if g >= 200 else np.zeros((fpr.channels, grid.n))
    if 0 < g < 2000:  # short window = last 200 shots, long window = all g shots so far
        ref = by_hand(det[:g])
    ok &= report(f"eval buffer, start g={g}", f, ref)
fz = dsx.load(dsx.manifest["test_sets"]["nominal"])["window_det"]
Wf = unpack_det(fz, dsx.n_det)
ok &= report("frozen (nominal test window)", fpr.of_windows(torch.from_numpy(Wf.astype(np.uint8)).to(dev)), by_hand(Wf))
print("ALL OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
