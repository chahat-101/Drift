"""Pre-run checks for round 6 (seed ensembles).

    python results/exp6/check6.py inventory          # checkpoints, dataset hashes, reused baseline files
    python results/exp6/check6.py repro DIR          # 1-member bits (m0, m1, m2) in DIR vs results/exp4 (d=5) / exp2 (d=3)
    python results/exp6/check6.py same A.npz B.npz   # two runs give identical arrays
    python results/exp6/check6.py logits             # ensemble logit = mean of member logits, 1000 shots
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
REGIMES = ["bursts", "bursts_big", "hot_qubit", "telegraph"]
# model type -> (member checkpoint pattern, dataset); d=5 from rounds 4-5, d=3 from reports 1-2
D5 = {f"{m}_d5_long": (f"decoder_{m}_d5_long_s{{s}}", "data/d5_wide")
      for m in ("static_wide", "adaptive_wide", "adaptive_stale", "static_nominal")}
D3 = {k: (f"decoder_{k}_s{{s}}", "data/d3_r3_p0.003_s0")
      for k in ("static_d3_wide", "adaptive_d3_wide", "adaptive_d3_stale", "static_d3_nominal")}
TYPES = {**D5, **D3}


def inventory():
    from dataset import sha256_file
    bad = []
    print("== checkpoints")
    for t, (pat, _) in TYPES.items():
        for s in range(3):
            p = pat.format(s=s) + ".pt"
            ok = os.path.exists(p)
            bad += [] if ok else [p]
            print(f"  {'ok ' if ok else 'MISSING'} {p}")
    print("== datasets (test sets + streams used, SHA-256 vs manifest)")
    for d in ("data/d5_wide", "data/d3_r3_p0.003_s0", "data/d3_wide"):
        m = json.load(open(f"{d}/manifest.json"))
        files = list(m["test_sets"].values())
        if d != "data/d3_wide":
            files += [m["streams"][r] for r in REGIMES]
            for r in REGIMES:
                for st in "1234":
                    f = m.get("stream_seeds", {}).get(r, {}).get(st)
                    files.append(f or f"MISSING stream {r} seed {st}")
        for f in files:
            p = f"{d}/{f}"
            ok = os.path.exists(p) and sha256_file(p) == m["files"][f]["sha256"]
            bad += [] if ok else [p]
            print(f"  {'ok ' if ok else 'BAD'} {p}")
    a = json.load(open("data/d3_wide/manifest.json"))["files"]
    b = json.load(open("data/d3_r3_p0.003_s0/manifest.json"))["files"]
    same = all(a[f]["sha256"] == b[f]["sha256"] for f in a if f.startswith("test_"))
    print(f"  d3_wide test sets identical to d3_r3_p0.003_s0's (report 2 scored tests on d3_wide): {same}")
    bad += [] if same else ["d3 test sets differ"]
    print("== reused results (MWPM, oracle, WindowDEM, single-seed bits)")
    need = ["results/exp4/curves/baselines.npz", "results/exp2/test_errors.npz"]
    need += [f"results/exp3/d5/{r}_d5_r5_u200_st{st}_s0_errors.npz" for r in REGIMES for st in range(5)]
    need += [f"results/exp2/{r}_d3_r3_u200_st{st}_s{s}_errors.npz" for r in REGIMES for st in range(5) for s in range(3)]
    need += [f"results/exp4/full/st{st}/decoder_{m}_d5_long_s{s}_errors.npz" for m in
             ("static_wide", "adaptive_wide", "adaptive_stale", "static_nominal") for st in range(5) for s in range(3)]
    miss = [p for p in need if not os.path.exists(p)]
    bad += miss
    print(f"  {len(need) - len(miss)}/{len(need)} present" + (": missing " + ", ".join(miss) if miss else ""))
    print("INVENTORY", "OK" if not bad else f"PROBLEMS: {bad}")


def ref_bits(t, s, st, key):
    """Saved single-model bits: d=5 from results/exp4/full, d=3 from results/exp2 (another machine)."""
    name = TYPES[t][0].format(s=s)
    if t in D5:
        z = np.load(f"results/exp4/full/st{st}/{name}_errors.npz")
        if key[0] == "test":
            return z[f"test|{key[1]}"]
        return np.unpackbits(z[f"{key[0]}|{key[1]}"], axis=-1).astype(bool)
    if key[0] == "test":
        lab = {"static_d3_wide": "static-wide", "adaptive_d3_wide": "adaptive-wide",
               "adaptive_d3_stale": "adaptive-stale", "static_d3_nominal": "static-nominal"}[t]
        z = np.load("results/exp2/test_errors.npz")
        return np.unpackbits(z[f"{lab}|{s}|{key[1]}"], count=int(z["_shots"])).astype(bool)
    z = np.load(f"results/exp2/{key[0]}_d3_r3_u200_st{st}_s{s}_errors.npz")
    return np.unpackbits(z[name + ("_frozen" if key[1] == "frozen" else "")], axis=-1).astype(bool)


def repro(folder):
    tot = {}
    for p in sorted(glob.glob(f"{folder}/**/ens_*_errors.npz", recursive=True)):
        t = os.path.basename(p)[4:-11]
        st = int(json.load(open(p[:-11] + ".json"))["stream_seed"])
        z = np.load(p)
        for k in z.files:
            parts = k.split("|")
            if parts[-1] == "ens":
                continue
            s = int(parts[-1][1:])
            new = np.unpackbits(z[k], axis=-1).astype(bool)
            ref = ref_bits(t, s, st, parts[:2])
            new = new[: ref.shape[-1]] if new.ndim == 1 else new
            d = int((new != ref).sum())
            tot[t] = tot.get(t, 0) + d
            print(f"{t} st{st} s{s} {parts[0]}|{parts[1]}: {d} / {ref.size} differ")
    print("REPRO", {t: v for t, v in tot.items()})


def same(a, b):
    za, zb = np.load(a), np.load(b)
    diff = [k for k in za.files if k not in zb.files or not np.array_equal(za[k], zb[k])]
    diff += [k for k in zb.files if k not in za.files]
    print(f"{len(za.files)} arrays; differing: {diff or 'none'}")
    print("SAME", not diff)


def logits_check():
    """Members' logits computed one model at a time, each with its own input build, vs the ensemble code."""
    import torch
    import ensemble_eval as ee
    from dataset import Dataset
    from decoder import DetectorGrid, build_input, load_checkpoint
    from drift import DriftStream
    ds = Dataset("data/d5_wide")
    grid = DetectorGrid(DriftStream(5, 5, 0.003).nominal())
    gpu = ee.Gpu(grid, ds.n_det, "cuda")
    s = ee.load_stream(ds, "bursts_big", 0)
    det = gpu.bits(torch.from_numpy(s["det"]).cuda(), ds.n_det).reshape(-1, ds.n_det)[:1000]
    starts = list(range(0, 1000, 200))
    sub = torch.arange(1000, device="cuda") // 200
    cks = [f"decoder_adaptive_wide_d5_long_s{i}.pt" for i in range(3)]
    models = [load_checkpoint(c, "cuda")[0].cuda() for c in cks]
    lg = ee.logits(models, gpu, det, gpu.to_grid(ee.window_fps(det, starts, 2000)), sub, batch=4096)
    ens = lg.mean(0)
    # by hand: CPU numpy fingerprints and build_input, one model at a time
    d = det.cpu().numpy().astype(bool)
    hand = []
    for m in models:
        out = []
        with torch.no_grad():
            for i in range(0, 1000, 200):
                fp = d[max(0, i - 2000):i].mean(axis=0).astype(np.float32) if i else np.zeros(ds.n_det, np.float32)
                out.append(m(build_input(grid, d[i:i + 200], fp).cuda()).cpu().numpy())
        hand.append(np.concatenate(out))
    hand = np.stack(hand)
    mean_hand = (hand[0] + hand[1] + hand[2]) / 3
    print("max |member logit (ensemble code) - member logit (by hand)|:", float(np.abs(lg.cpu().numpy() - hand).max()))
    print("max |ensemble logit - (l0+l1+l2)/3 by hand|:", float(np.abs(ens.cpu().numpy() - mean_hand).max()))
    print("decisions equal:", bool(((ens.cpu().numpy() > 0) == (mean_hand > 0)).all()),
          "| members disagree on", int(((hand > 0).sum(0) % 3 != 0).sum()), "of 1000 shots")
    for i in range(3):
        print(f"  shot {i}: members {hand[:, i].round(4).tolist()} mean {mean_hand[i]:.4f} ensemble {float(ens[i]):.4f}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"inventory": inventory, "repro": lambda: repro(sys.argv[2]), "same": lambda: same(*sys.argv[2:4]),
     "logits": logits_check}[cmd]()
