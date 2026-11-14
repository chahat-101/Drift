"""Seed ensembles of neural decoders on the saved test sets and drift streams, on the GPU.

    python ensemble_eval.py --checkpoints decoder_adaptive_wide_d5_long_s0.pt,decoder_adaptive_wide_d5_long_s1.pt,\
decoder_adaptive_wide_d5_long_s2.pt --data data/d5_wide --stream-seed 0 --tests --out results/exp6/d5/st0 \
        --name ens_adaptive_wide_d5_long

Combination rule (fixed in advance): average the members' logits, flip if the mean logit > 0.
A 1-checkpoint "ensemble" is just that model.

Streams follow the causal protocol of eval.py / curve_eval.py: shots are decoded in
sub-batches of --update-every, and each sub-batch uses the fingerprint of the last
`window` shots before it (each member's own window from its cfg; multiscale/mspair: the short
window is the last cfg["short"] shots before it, see decoder.Fingerprint). With the data fixed,
those fingerprints are computed up front, so all shots go through the model in large batches.
Adaptive models also run with the frozen fingerprint, shared by all members: every channel computed
from the nominal test set's fingerprint shots.

Packed shots are loaded onto the GPU once; unpacking, fingerprints and the grid layout
happen there. Members run one after another on the same input batch. fp32 (cuDNN TF32 convs, as before).

Writes <out>/<name>_errors.npz (packed error bits: "<regime>|<live|frozen>|<ens|m0|m1|...>",
shape steps x shots/8, and "test|<state>|<ens|m0|...>", shape shots/8) and <out>/<name>.json
(LERs per phase). Skips the job if the JSON exists and lists every requested regime.
"""

import argparse
import json
import os
import time

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

from dataset import Dataset  # noqa: E402
from decoder import DetectorGrid, Fingerprint, load_checkpoint  # noqa: E402
from drift import DriftStream  # noqa: E402
from eval import phase_masks  # noqa: E402

# the PyTorch defaults, as in every earlier eval and in training: fp32 matmuls, TF32 cuDNN convs.
# With cuDNN TF32 off, ~5 of 1.1M shots differ from the saved results/exp4 bits.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = True


class Gpu:
    """Bit unpacking and grid layout on the GPU (as train.GpuChunks)."""

    def __init__(self, grid, n_det, device):
        self.flat = torch.from_numpy(grid.flat).to(device)
        self.shape, self.n_det, self.device = grid.shape, n_det, device
        self.shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=device)

    def bits(self, packed, count):
        """np.unpackbits along the last axis."""
        b = (packed.unsqueeze(-1) >> self.shifts) & 1
        return b.flatten(-2)[..., :count]

    def to_grid(self, v):
        out = torch.zeros(len(v), self.shape[0] * self.shape[1] * self.shape[2], device=self.device)
        out[:, self.flat] = v.float()
        return out.view(len(v), *self.shape)

    def fp_grid(self, f):
        """(K, C, n_det) fingerprint channels -> (K, C, T, X, Y)."""
        return self.to_grid(f.flatten(0, 1)).view(*f.shape[:2], *self.shape)


def cumsum0(x):
    """Cumulative sums along the shot axis with a leading row of zeros (int64)."""
    cs = torch.zeros(len(x) + 1, x.shape[1], dtype=torch.int64, device=x.device)
    cs[1:] = torch.cumsum(x, 0, dtype=torch.int64)
    return cs


def window_fps(det, starts, window, fpr):
    """Fingerprint channels (K, C, n_det) before each sub-batch start g, from det[max(0, g - window):g]
    (short window: det[max(0, g - short):g]); empty windows give zeros."""
    cs = cumsum0(det)
    g = torch.as_tensor(starts, device=det.device)
    lo = torch.clamp(g - window, min=0)
    if fpr.mode == "single":
        return fpr.features(g - lo, cs[g] - cs[lo])
    los = torch.clamp(g - fpr.short, min=0)
    sp = None
    if fpr.partners is not None:
        cp = cumsum0(fpr.products(det))
        sp = (cp[g] - cp[lo]).unflatten(-1, fpr.partners.shape)
    return fpr.features(g - lo, cs[g] - cs[lo], g - los, cs[g] - cs[los], sp)


@torch.no_grad()
def logits(models, gpu, det, fp_grid=None, fp_idx=None, batch=8192):
    """(members, shots) logits. det: (shots, n_det) 0/1 uint8; fp_grid (K, C, T, X, Y) scaled channels,
    indexed per shot by fp_idx."""
    out = torch.empty(len(models), len(det), device=det.device)
    for a in range(0, len(det), batch):
        x = gpu.to_grid(det[a : a + batch]).unsqueeze(1)
        if fp_grid is not None:
            x = torch.cat([x, fp_grid[fp_idx[a : a + batch]]], dim=1)
        for k, m in enumerate(models):
            out[k, a : a + batch] = m(x)
    return out


def errors(lg, obs):
    """{ens, m0, m1, ...}: per-shot error bits (bool tensors). Ensemble = mean logit > 0."""
    e = {"ens": (lg.mean(0) > 0) != obs}
    e.update({f"m{k}": (lg[k] > 0) != obs for k in range(len(lg))})
    return e


def find_batch(models, gpu, det, fp_grid, fp_idx, start=1024):
    """Largest power-of-two batch that runs on real shots without OOM (doubling), times 0.75."""
    b, ok = start, None
    while b <= len(det):
        try:
            logits(models, gpu, det[:b], fp_grid, fp_idx, batch=b)
            torch.cuda.synchronize()
            ok, b = b, 2 * b
        except torch.OutOfMemoryError:
            break
        finally:
            torch.cuda.empty_cache()
    print(f"largest batch {ok}, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    return int(ok * 0.75)


def load_stream(ds, regime, seed):
    """Packed arrays of a saved stream (hash-checked), as Dataset.stream finds them."""
    fn = ds.manifest["streams"][regime] if seed == 0 else ds.manifest["stream_seeds"][regime][str(seed)]
    return ds.load(fn)


def pack(e):
    return np.packbits(e.cpu().numpy(), axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True, help="comma-separated member checkpoints")
    ap.add_argument("--data", required=True)
    ap.add_argument("--streams", default="bursts,bursts_big,hot_qubit,telegraph")
    ap.add_argument("--stream-seed", type=int, default=0)
    ap.add_argument("--update-every", type=int, default=200)
    ap.add_argument("--tests", action="store_true", help="also score the test sets")
    ap.add_argument("--batch", type=int, default=16384, help="shots per forward pass")
    ap.add_argument("--find-batch", action="store_true",
                    help="print the largest batch that fits (doubling until OOM, on the first stream) x 0.75, and exit")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--force", action="store_true", help="recompute even if the JSON exists")
    args = ap.parse_args()

    regimes = args.streams.split(",") if args.streams else []
    js = os.path.join(args.out, f"{args.name}.json")
    if args.find_batch:
        args.tests = False
    elif not args.force and os.path.exists(js):
        j = json.load(open(js))
        if set(regimes) <= set(j["streams"]) and (not args.tests or j["tests"]):
            print(f"skip {js} (exists)")
            return
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    cks = args.checkpoints.split(",")
    ds = Dataset(args.data)
    a = ds.args
    nominal = DriftStream(a["distance"], a["rounds"], a["p"]).nominal()
    grid = DetectorGrid(nominal)
    gpu = Gpu(grid, ds.n_det, dev)
    models, cfgs = [], []
    for ck in cks:
        m, cfg = load_checkpoint(ck, dev)
        ds.check_model(cfg, ck)
        models.append(m.to(dev))
        cfgs.append(cfg)
    adaptive = cfgs[0]["adaptive"]
    if any(c["adaptive"] != adaptive for c in cfgs):
        raise SystemExit("members must be all static or all adaptive")
    fpkey = lambda c: json.dumps([c.get(k) for k in ("fingerprint", "short", "pair_offsets", "pair_scale")])
    if any(fpkey(c) != fpkey(cfgs[0]) for c in cfgs):
        raise SystemExit("members must share one fingerprint config")
    fpr = Fingerprint(cfgs[0], grid, dev) if adaptive else None

    summary = {"checkpoints": cks, "data": args.data, "stream_seed": args.stream_seed,
               "update_every": args.update_every, "rule": "mean logit > 0", "tests": {}, "streams": {}}
    errs = {}
    batch = args.batch
    # test sets: adaptive models use the test set's own fingerprint window (as curve_eval.py)
    if args.tests:
        for tname, fn in ds.manifest["test_sets"].items():
            t = ds.load(fn)
            det = gpu.bits(torch.from_numpy(t["det"]).to(dev), ds.n_det)
            obs = gpu.bits(torch.from_numpy(t["obs"]).to(dev).T, len(det))[0].bool()
            fpg = idx = None
            if adaptive:
                w = gpu.bits(torch.from_numpy(t["window_det"]).to(dev), ds.n_det)
                fpg, idx = gpu.fp_grid(fpr.of_windows(w)[None]), torch.zeros(len(det), dtype=torch.long, device=dev)
            e = errors(logits(models, gpu, det, fpg, idx, batch), obs)
            for k, v in e.items():
                errs[f"test|{tname}|{k}"] = pack(v)
            summary["tests"][tname] = {k: float(v.float().mean()) for k, v in e.items()}
    frozen = None
    if adaptive:  # every channel from the nominal test set's fingerprint shots
        w = ds.load(ds.manifest["test_sets"]["nominal"])["window_det"]
        frozen = gpu.fp_grid(fpr.of_windows(gpu.bits(torch.from_numpy(w).to(dev), ds.n_det))[None])
    for r in regimes:
        s = load_stream(ds, r, args.stream_seed)
        steps, shots = s["det"].shape[:2]
        det = gpu.bits(torch.from_numpy(s["det"]).to(dev), ds.n_det).reshape(steps * shots, -1)
        obs = gpu.bits(torch.from_numpy(s["obs"]).to(dev).transpose(1, 2), shots)[:, 0].reshape(-1).bool()
        starts = [t * shots + i for t in range(steps) for i in range(0, shots, args.update_every)]
        sub = torch.repeat_interleave(torch.arange(len(starts), device=dev),
                                      torch.diff(torch.tensor(starts + [steps * shots], device=dev)))
        if args.find_batch:
            fpg = gpu.fp_grid(window_fps(det, starts, cfgs[0]["window"], fpr)) if adaptive else None
            print("BATCH", find_batch(models, gpu, det, fpg, sub if adaptive else None))
            return
        runs = {}
        if not adaptive:
            runs["live"] = logits(models, gpu, det, batch=batch)
        else:
            # live: each member its own window; members with equal windows share one forward input
            lg = torch.empty(len(models), len(det), device=dev)
            for wnd in sorted({c["window"] for c in cfgs}):
                ks = [k for k, c in enumerate(cfgs) if c["window"] == wnd]
                fpg = gpu.fp_grid(window_fps(det, starts, wnd, fpr))
                lg[ks] = logits([models[k] for k in ks], gpu, det, fpg, sub, batch)
            runs["live"] = lg
            zero = torch.zeros(len(det), dtype=torch.long, device=dev)
            runs["frozen"] = logits(models, gpu, det, frozen, zero, batch)
        rows = [{"scale": float(sc), "n_hot": int((h >= 0).sum())} for sc, h in zip(s["scale"], s["hot_qubits"])]
        masks = {ph: m for ph, m in phase_masks(rows).items()}
        summary["streams"][r] = {}
        for kind, lg in runs.items():
            for k, v in errors(lg, obs).items():
                v = v.view(steps, shots)
                errs[f"{r}|{kind}|{k}"] = pack(v)
                summary["streams"][r][f"{kind}|{k}"] = {
                    ph: float(v[torch.from_numpy(m).to(dev)].float().mean()) if m.any() else None
                    for ph, m in masks.items()}
    np.savez_compressed(os.path.join(args.out, f"{args.name}_errors.npz"), **errs)
    summary["seconds"] = round(time.time() - t0, 1)
    with open(js + ".tmp", "w") as f:
        json.dump(summary, f, indent=1)
    os.replace(js + ".tmp", js)
    bb = summary["streams"].get("bursts_big", {}).get("live|ens", {}).get("drift")
    print(f"{args.name} st{args.stream_seed}: {summary['seconds']}s" + (f", bursts_big drift (ens) {bb:.4f}" if bb else ""),
          flush=True)


if __name__ == "__main__":
    main()
