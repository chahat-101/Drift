"""Noise-map quality of an --aux-noise model on the saved drift streams (live fingerprint), on the GPU.

    python noise_eval.py --checkpoint decoder_adaptive_d5_aux_long_s0.pt --data data/d5_wide \
        --stream-seeds 0,1,2,3,4 --out results/exp9/noise --name aux_s0

Causal protocol as ensemble_eval.py: sub-batches of --update-every shots, each with the fingerprint of the
last `window` shots before it. Per step, the predicted p_hot of a qubit is the mean over the step's shots of
the noise head / 10, and the predicted scale is exp(mean predicted log scale). Truth = the step's NoiseState.

Metrics per regime x phase (eval.phase_masks), pooled over stream seeds (means over steps):
  topk       fraction of truly hot qubits among the k highest predicted (k = number of hot qubits);
             steps with hot qubits only
  mae_hot    mean |predicted - true p_hot| over the hot qubits of steps with hot qubits
  mae_cold   mean |predicted p_hot| over the non-hot qubits
  scale_mae  mean |predicted - true scale|
  mean_p     mean predicted p_hot over all qubits (on nominal steps: the false-alarm level)
Writes <out>/<name>.json (metrics) and <out>/<name>.npz (per step: pred, pred_scale and the truth).
Skips if the JSON exists and lists every regime.
"""
import argparse
import json
import os
import time

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

from dataset import Dataset  # noqa: E402
from decoder import DetectorGrid, Fingerprint, load_checkpoint, noise_targets  # noqa: E402
from drift import DriftStream  # noqa: E402
from ensemble_eval import Gpu, load_stream, window_fps  # noqa: E402
from eval import phase_masks  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = True
PHASES = ["drift", "onset", "later", "after", "nominal"]


@torch.no_grad()
def step_predictions(model, gpu, det, fpg, sub, steps, shots, batch):
    """Per-step mean of the noise map (steps, n_qubits) and of log(scale) (steps,)."""
    nq = model.qubit_pos.shape[0]
    pm = torch.zeros(steps, nq, dtype=torch.float64, device=det.device)
    ps = torch.zeros(steps, dtype=torch.float64, device=det.device)
    for a in range(0, len(det), batch):
        x = torch.cat([gpu.to_grid(det[a : a + batch]).unsqueeze(1), fpg[sub[a : a + batch]]], dim=1)
        _, noise, ls = model(x, aux=True)
        t = torch.arange(a, a + len(x), device=det.device) // shots
        pm.index_add_(0, t, noise.double())
        ps.index_add_(0, t, ls.double())
    return (pm / shots / 10).cpu().numpy(), np.exp((ps / shots).cpu().numpy())


def metrics(pred, pscale, tmap, tscale, mask):
    """The metrics above over the steps in mask (pred, tmap: p_hot per qubit; scales: plain)."""
    if not mask.any():
        return None
    p, t = pred[mask], tmap[mask]
    hot = t > 0
    out = {"steps": int(mask.sum()), "mean_p": float(p.mean()), "scale_mae": float(np.abs(pscale[mask] - tscale[mask]).mean()),
           "mae_cold": float(np.abs(p[~hot]).mean())}
    rows = hot.any(1)
    if rows.any():
        k = hot.sum(1)
        top = [len(set(np.argsort(-p[i], kind="stable")[: k[i]]) & set(np.flatnonzero(hot[i]))) / k[i]
               for i in np.flatnonzero(rows)]
        out.update(topk=float(np.mean(top)), mae_hot=float(np.abs(p[hot] - t[hot]).mean()), hot_steps=int(rows.sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--streams", default="bursts,bursts_big,hot_qubit,telegraph")
    ap.add_argument("--stream-seeds", default="0,1,2,3,4")
    ap.add_argument("--update-every", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    regimes, seeds = args.streams.split(","), [int(s) for s in args.stream_seeds.split(",")]
    js = os.path.join(args.out, f"{args.name}.json")
    if not args.force and os.path.exists(js) and set(regimes) <= set(json.load(open(js))["metrics"]):
        print(f"skip {js} (exists)")
        return
    t0, dev = time.time(), args.device
    os.makedirs(args.out, exist_ok=True)
    ds = Dataset(args.data)
    a = ds.args
    grid = DetectorGrid(DriftStream(a["distance"], a["rounds"], a["p"]).nominal())
    model, cfg = load_checkpoint(args.checkpoint, dev)
    ds.check_model(cfg, args.checkpoint)
    if not cfg.get("aux_noise"):
        raise SystemExit(f"{args.checkpoint} has no noise-map head (--aux-noise)")
    model.to(dev)
    gpu, fpr = Gpu(grid, ds.n_det, dev), Fingerprint(cfg, grid, dev)
    summary = {"checkpoint": args.checkpoint, "data": args.data, "stream_seeds": seeds,
               "update_every": args.update_every, "metrics": {}}
    arrays = {}
    for r in regimes:
        pooled = {k: [] for k in ("pred", "pscale", "tmap", "tscale")}
        masks = {ph: [] for ph in PHASES}
        for st in seeds:
            s = load_stream(ds, r, st)
            steps, shots = s["det"].shape[:2]
            det = gpu.bits(torch.from_numpy(s["det"]).to(dev), ds.n_det).reshape(steps * shots, -1)
            starts = [t * shots + i for t in range(steps) for i in range(0, shots, args.update_every)]
            sub = torch.repeat_interleave(torch.arange(len(starts), device=dev),
                                          torch.diff(torch.tensor(starts + [steps * shots], device=dev)))
            fpg = gpu.fp_grid(window_fps(det, starts, cfg["window"], fpr))
            pred, pscale = step_predictions(model, gpu, det, fpg, sub, steps, shots, args.batch)
            tmap, _ = noise_targets(s["scale"], s["hot_p"], s["hot_qubits"], cfg["aux_qubits"])
            rows = [{"scale": float(sc), "n_hot": int((h >= 0).sum())} for sc, h in zip(s["scale"], s["hot_qubits"])]
            for ph, m in phase_masks(rows).items():
                if ph in masks:
                    masks[ph].append(m)
            for k, v in (("pred", pred), ("pscale", pscale), ("tmap", tmap / 10), ("tscale", s["scale"])):
                pooled[k].append(v)
                arrays[f"{r}|st{st}|{k}"] = v
        P = {k: np.concatenate(v) for k, v in pooled.items()}
        summary["metrics"][r] = {ph: metrics(P["pred"], P["pscale"], P["tmap"], P["tscale"], np.concatenate(m))
                                 for ph, m in masks.items()}
        d = summary["metrics"][r]["drift"]
        print(f"{args.name} {r}: drift top-k {d.get('topk', float('nan')):.3f}, mae_hot {d.get('mae_hot', float('nan')):.4f}, "
              f"scale_mae {d['scale_mae']:.3f}; nominal mean p_hot {summary['metrics'][r]['nominal']['mean_p']:.5f}",
              flush=True)
    np.savez_compressed(os.path.join(args.out, f"{args.name}.npz"), **arrays)
    summary["seconds"] = round(time.time() - t0, 1)
    with open(js + ".tmp", "w") as f:
        json.dump(summary, f, indent=1)
    os.replace(js + ".tmp", js)
    print(f"{args.name}: {summary['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
