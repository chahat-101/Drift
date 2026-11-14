"""Train the neural decoder on Stim data with randomly drawn noise.

Each training step uses one chunk: a random NoiseState (a global scale change and,
with some probability, a cluster of hot qubits), a batch of shots from it and, for
the adaptive model, a separate window of shots from the same noise for the
fingerprint. The model never sees the NoiseState.

    python train.py --distance 3 --rounds 3 --steps 3000                 # static
    python train.py --distance 3 --rounds 3 --steps 3000 --adaptive      # adaptive
    python train.py --data data/d3_r3_p0.003_s0 --epochs 2               # fixed dataset
    python train.py --data data/d3_wide --epochs 2 --adaptive --stale-prob 0.3
    python train.py --data data/d3_wide --epochs 2 --adaptive --stale-prob 0.3 --fingerprint mspair
    python train.py --nominal-only --steps 60000 --out decoder_static_d3_nominal_s0.pt
    python train.py --data data/d3_nominal --epochs 2 --out decoder_static_d3_nominal_fixed_s0.pt

Without --data every chunk is sampled fresh. With --data, chunks come from a dataset
made by make_dataset.py and are reshuffled every epoch. Validation loss is printed
every --val-every steps and the final scores use the saved test sets. Distance,
rounds, p, batch and window come from the dataset.

--stale-prob (adaptive only): with this probability a batch gets a fingerprint that
does not match its shots. Half the time it is another random chunk's window, half the
time a window that is part another chunk and part this one. It has its own RNG, so
the chunk order is unchanged.

--nominal-only (no --data): every chunk uses NoiseState().

--fingerprint (adaptive only; see decoder.py): single (default, as all earlier models), multiscale
or mspair. Every channel comes from the same window rows, also for stale windows. mspair stores its
grid offsets and covariance scale (1 / RMS nominal covariance, 2 significant digits) in the cfg.
multiscale and mspair need the GPU data path.

--aux-noise (adaptive, GPU data path): noise-map and log(scale) heads (decoder.ConvDecoder) trained on the
chunk's NoiseState: loss = BCE + 0.1 MSE(p_hot x 10 per qubit) + 0.1 MSE(log scale), weights fixed in
advance. Steps with a stale (mismatched) fingerprint use BCE only. The printed loss and val are BCE;
"aux" is the running weighted auxiliary loss (unmasked steps only).
--film (adaptive, GPU data path): spatial FiLM from the fingerprint (decoder.ConvDecoder).

With --data the chunks are kept bit-packed on the GPU and unpacked there. --cpu-data
selects the older path that unpacks on the CPU and gives the same inputs.
--save-every k also saves <out>_step<k>.pt with optimiser and RNG state for --resume.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from baselines import Oracle, Static
from dataset import Dataset
from decoder import (
    FINGERPRINTS,
    SHORT,
    ConvDecoder,
    DetectorGrid,
    Fingerprint,
    build_input,
    in_channels,
    noise_targets,
    pair_offsets,
    pair_partners,
    qubit_positions,
    save_checkpoint,
)
from drift import DriftStream, NoiseState, sample_chunk, test_states

AUX_W = 0.1  # weight of each auxiliary MSE (--aux-noise), fixed in advance
DEFAULTS = {"distance": 3, "rounds": 3, "p": 0.003, "window": 2000, "batch": 512, "steps": 3000}
FROM_DATA = {"distance": "distance", "rounds": "rounds", "p": "p", "window": "window", "batch": "batch"}


def sampled_tests(stream, adaptive, window, shots, seed=12345):
    """Fresh test shots for each test state, as [(name, state, det, obs, window_det)]."""
    tests = []
    for i, (name, st) in enumerate(test_states(stream.nominal()).items()):
        det, obs = stream.sample(st, shots, seed + i)
        wdet = stream.sample(st, window, seed + 100 + i)[0] if adaptive else None
        tests.append((name, st, det, obs, wdet))
    return tests


def evaluate(model, grid, stream, tests, adaptive, device, fp_fn=None):
    """Print and return test-set LERs as {state name: {"static", "oracle", "neural"}}.

    fp_fn(window_det) gives the fingerprint (default: the single-rate fingerprint).
    """
    model.eval()
    static, oracle = Static(stream.nominal()), Oracle(stream)
    print(f"{'state':<14}{'static MWPM':>12}{'oracle MWPM':>12}{'neural':>10}")
    out = {}
    fp_fn = fp_fn or (lambda w: w.mean(axis=0).astype(np.float32))
    for name, st, det, obs, wdet in tests:
        fp = fp_fn(wdet) if adaptive else None
        with torch.no_grad():
            pred = np.concatenate(
                [
                    (model(build_input(grid, det[j : j + 4096], fp).to(device)) > 0).cpu().numpy()
                    for j in range(0, len(det), 4096)
                ]
            )
        ler = lambda p: float(np.mean(p.reshape(-1) != obs[:, 0]))
        out[name] = {"static": ler(static.decode(det)), "oracle": ler(oracle.decode(det, st)), "neural": ler(pred)}
        print(f"{name:<14}{out[name]['static']:>12.5f}{out[name]['oracle']:>12.5f}{out[name]['neural']:>10.5f}")
    model.train()
    return out


def chunk_loss(model, grid, det, obs, wdet, adaptive, device):
    """Mean BCE loss of the model on one chunk."""
    fp = wdet.mean(axis=0).astype(np.float32) if adaptive else None
    x = build_input(grid, det, fp).to(device)
    y = torch.from_numpy(obs[:, 0].astype(np.float32)).to(device)
    return F.binary_cross_entropy_with_logits(model(x), y)


@torch.no_grad()
def val_loss(model, grid, chunks, adaptive, device):
    """Mean loss over the validation chunks."""
    model.eval()
    loss = np.mean([chunk_loss(model, grid, det, obs, wdet, adaptive, device).item() for _, det, obs, wdet in chunks])
    model.train()
    return float(loss)


class GpuChunks:
    """Training chunks kept bit-packed on the GPU, with unpacking, fingerprint and grid layout done there.

    Gives the same inputs as Chunks and build_input. fp[i] is the fingerprint of chunk i, (C, n)
    channels from fpr (a decoder.Fingerprint), computed once. Packed windows are kept only if
    keep_windows (needed for --stale-prob). With qubit_ids, tmap[i] / tscale[i] are chunk i's noise-map and
    log(scale) targets (decoder.noise_targets).
    """

    def __init__(self, chunks, grid, device, fpr, keep_windows, qubit_ids=None):
        a, self.n_det, self.batch, self.device = chunks.a, chunks.n_det, chunks.batch, device
        self.n = len(chunks)
        self.det = torch.from_numpy(a["det"]).to(device)
        self.obs = torch.from_numpy(a["obs"]).to(device)
        self.flat = torch.from_numpy(grid.flat).to(device)
        self.shape = grid.shape
        self.shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=device)
        self.win = self.fp = None
        self.fpr = fpr
        if qubit_ids is not None:
            m, ls = noise_targets(a["scale"], a["hot_p"], a["hot_qubits"], qubit_ids)
            self.tmap, self.tscale = torch.from_numpy(m).to(device), torch.from_numpy(ls).to(device)
        if fpr is not None:
            win = torch.from_numpy(a["window_det"]).to(device)
            self.window = win.shape[1]
            self.fp = torch.cat([self.feats(win[i : i + 256]) for i in range(0, self.n, 256)])
            self.win = win if keep_windows else None
            del win

    def __len__(self):
        return self.n

    def bits(self, packed, count):
        """np.unpackbits along the last axis, on the GPU."""
        b = (packed.unsqueeze(-1) >> self.shifts) & 1
        return b.flatten(-2)[..., :count]

    def feats(self, packed):
        """Fingerprint channels (..., C, n_det) of packed windows with shape (..., window, bytes)."""
        return self.fpr.of_windows(self.bits(packed, self.n_det))

    def to_grid(self, v):
        out = torch.zeros(len(v), self.shape[0] * self.shape[1] * self.shape[2], device=self.device)
        out[:, self.flat] = v.float()
        return out.view(len(v), *self.shape)

    def inputs(self, i, fp=None):
        """(x, y) for chunk i. fp overrides the chunk's own fingerprint."""
        x = self.to_grid(self.bits(self.det[i], self.n_det)).unsqueeze(1)
        y = self.bits(self.obs[i].transpose(0, 1), self.batch)[0].float()
        if self.fp is not None:
            fp = self.fp[i] if fp is None else fp
            f = self.to_grid(fp).unsqueeze(0).expand(len(x), -1, -1, -1, -1)
            x = torch.cat([x, f], dim=1)
        return x.contiguous(memory_format=torch.channels_last_3d), y

    def stale_fp(self, i, other, rng):
        """Fingerprint of stale_window(window i, window other, rng), drawing from rng in the same order."""
        if rng.random() < 0.5:
            return self.fp[other]
        k = int(rng.uniform() * self.window)
        w = torch.cat([self.win[other, :k], self.win[i, k:]])
        return self.feats(w)


def gpu_loss(model, data, i, fp=None, aux=False, shared_fp=False):
    """BCE on chunk i. With aux: (BCE + weighted aux loss, BCE, weighted aux loss)."""
    x, y = data.inputs(i, fp)
    if not aux:
        return F.binary_cross_entropy_with_logits(model(x, shared_fp=shared_fp) if shared_fp else model(x), y)
    logit, noise, log_scale = model(x, aux=True, shared_fp=shared_fp)
    bce = F.binary_cross_entropy_with_logits(logit, y)
    a = AUX_W * F.mse_loss(noise, data.tmap[i].expand_as(noise)) + AUX_W * F.mse_loss(
        log_scale, data.tscale[i].expand_as(log_scale))
    return bce + a, bce, a


@torch.no_grad()
def gpu_val_loss(model, data, aux=False, shared_fp=False):
    """Mean val BCE, or with aux (mean BCE, mean weighted aux loss)."""
    model.eval()
    out = [gpu_loss(model, data, i, aux=aux, shared_fp=shared_fp) for i in range(len(data))]
    model.train()
    if not aux:
        return float(np.mean([v.item() for v in out]))
    return float(np.mean([v[1].item() for v in out])), float(np.mean([v[2].item() for v in out]))


def stale_window(wdet, other, rng):
    """A fingerprint window that does not match the shots.

    Half the time it is another chunk's window. Otherwise it is the first k rows of the
    other window followed by the remaining rows of this chunk's, with k uniform in [0, window).
    """
    if rng.random() < 0.5:
        return other
    k = int(rng.uniform() * len(wdet))
    return np.concatenate([other[:k], wdet[k:]])


def fresh_chunks(stream, rng, batch, window, nominal=False):
    """Endless stream of freshly sampled chunks."""
    while True:
        yield sample_chunk(stream, rng, batch, window, nominal)


class Order:
    """Chunk indices in a new random order every epoch. perm and pos are saved for --resume."""

    def __init__(self, n, rng):
        self.n, self.rng, self.perm, self.pos = n, rng, None, n

    def __next__(self):
        if self.pos == self.n:
            self.perm, self.pos = self.rng.permutation(self.n), 0
        self.pos += 1
        return int(self.perm[self.pos - 1])


def nominal_pair_scale(stream, grid, offsets, shots=200000, seed=0):
    """1 / RMS of the nominal covariances cov(x_i, x_{i+o}) over all valid pairs, rounded to 2 digits."""
    det = stream.sample(NoiseState(), shots, seed)[0].astype(np.float64)
    m, cov = det.mean(axis=0), []
    for row in pair_partners(grid, offsets):
        for i, j in enumerate(row):
            if j >= 0:
                cov.append((det[:, i] * det[:, j]).mean() - m[i] * m[j])
    return float(f"{1 / np.sqrt(np.mean(np.square(cov))):.2g}")


def step_path(out, step):
    return out[:-3] + f"_step{step}.pt" if out.endswith(".pt") else f"{out}_step{step}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="dataset folder from make_dataset.py")
    ap.add_argument("--distance", type=int, default=None, help="default 3, or from --data")
    ap.add_argument("--rounds", type=int, default=None, help="default 3, or from --data")
    ap.add_argument("--p", type=float, default=None, help="default 0.003, or from --data")
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument(
        "--window", type=int, default=None, help="shots used for the fingerprint (default 2000, or from --data)"
    )
    ap.add_argument("--steps", type=int, default=None, help="default 3000; with --data, epochs x train chunks")
    ap.add_argument("--epochs", type=int, default=1, help="passes over the train chunks (--data only)")
    ap.add_argument("--val-every", type=int, default=500, help="steps between val losses (--data only)")
    ap.add_argument("--batch", type=int, default=None, help="default 512, or from --data")
    ap.add_argument(
        "--stale-prob",
        type=float,
        default=0.0,
        help="probability of a mismatched fingerprint per batch (adaptive only)",
    )
    ap.add_argument("--fingerprint", choices=FINGERPRINTS, default="single", help="adaptive fingerprint channels")
    ap.add_argument("--aux-noise", action="store_true", help="noise-map + log(scale) auxiliary heads (see above)")
    ap.add_argument("--film", action="store_true", help="spatial FiLM from the fingerprint (see above)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument(
        "--nominal-only", action="store_true", help="train on nominal noise only (no drift); not with --data"
    )
    ap.add_argument("--eval-shots", type=int, default=50000, help="ignored with --data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-every", type=int, default=0, help="also save <out>_step<k>.pt every k steps")
    ap.add_argument("--resume", action="store_true", help="continue from the latest <out>_step<k>.pt")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (GPU data path)")
    ap.add_argument(
        "--cpu-data",
        action="store_true",
        help="unpack and lay out each chunk on the CPU instead of the GPU (--data only)",
    )
    args = ap.parse_args()

    if args.nominal_only and args.data:
        raise SystemExit("--nominal-only samples on the fly; it cannot be combined with --data")
    rng = np.random.default_rng(args.seed)
    ds = train_set = val_set = None
    if args.data:
        ds = Dataset(args.data)
        ds.fill_args(args, FROM_DATA)
        train_set, val_set = ds.chunks("train"), ds.chunks("val")
        if not args.adaptive:
            train_set.drop_windows()
            val_set.drop_windows()
        args.steps = args.steps or args.epochs * len(train_set)
    for k, v in DEFAULTS.items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    stream = DriftStream(args.distance, args.rounds, args.p)
    nominal = stream.nominal()
    grid = DetectorGrid(nominal)
    if args.fingerprint != "single" and (not args.adaptive or not args.data or args.cpu_data):
        raise SystemExit("--fingerprint multiscale/mspair needs --adaptive, --data and the GPU data path")
    fpcfg = {"fingerprint": args.fingerprint}
    if args.fingerprint != "single":
        fpcfg["short"] = SHORT
    if args.fingerprint == "mspair":
        fpcfg["pair_offsets"] = pair_offsets(grid, nominal)
        fpcfg["pair_scale"] = nominal_pair_scale(stream, grid, fpcfg["pair_offsets"])
        print(f"pair offsets {fpcfg['pair_offsets']}, scale {fpcfg['pair_scale']}")
    if (args.aux_noise or args.film) and (not args.adaptive or not args.data or args.cpu_data):
        raise SystemExit("--aux-noise/--film need --adaptive, --data and the GPU data path")
    fpcfg.update(aux_noise=args.aux_noise, film=args.film)
    qubit_ids = None
    if args.aux_noise:
        qubit_ids, fpcfg["aux_pos"] = qubit_positions(grid, nominal)
        fpcfg["aux_qubits"] = qubit_ids
    model = ConvDecoder(in_ch=in_channels(dict(fpcfg, adaptive=args.adaptive)), hidden=args.hidden,
                        layers=args.layers, aux_noise=args.aux_noise, film=args.film,
                        qubit_pos=fpcfg.get("aux_pos")).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps)
    mode = "adaptive" if args.adaptive else "static"
    print(
        f"training {mode} d={args.distance} r={args.rounds} grid={grid.shape} device={device} "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )
    gpu = ds is not None and not args.cpu_data
    if ds:
        print(
            f"data {args.data}: {len(train_set)} train chunks, {len(val_set)} val chunks, "
            f"{args.steps} steps = {args.steps / len(train_set):.2f} epochs"
        )
        order = Order(len(train_set), rng)
        source = (train_set[next(order)] for _ in iter(int, 1))
    else:
        source = fresh_chunks(stream, rng, args.batch, args.window if args.adaptive else 0, args.nominal_only)
    # source of the mismatched window for --stale-prob
    stale_rng = np.random.default_rng([args.seed, 1])
    if ds:
        other_window = lambda: train_set[stale_rng.integers(len(train_set))][3]
    else:
        other_window = lambda: sample_chunk(stream, stale_rng, 0, args.window)[3]
    stale = args.adaptive and args.stale_prob > 0
    if gpu:
        torch.backends.cudnn.benchmark = True
        model.to(memory_format=torch.channels_last_3d)
        fpr = Fingerprint(fpcfg, grid, device) if args.adaptive else None
        gtrain = GpuChunks(train_set, grid, device, fpr, stale, qubit_ids)
        gval = GpuChunks(val_set, grid, device, fpr, False, qubit_ids)
        del train_set, val_set, source, other_window  # free the CPU copies
        print(f"data on {device}: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    fmodel = torch.compile(model) if args.compile else model

    out = args.out or f"decoder_{mode}_d{args.distance}.pt"
    cfg = dict(vars(args), adaptive=args.adaptive, **fpcfg)
    start, run_loss, t_done, run_aux = 1, None, 0.0, None
    if args.resume and ds:
        done = (
            [
                k
                for k in range(args.save_every, args.steps + 1, args.save_every or args.steps)
                if os.path.exists(step_path(out, k))
            ]
            if args.save_every
            else []
        )
        if done:
            ck = torch.load(step_path(out, done[-1]), map_location=device, weights_only=False)
            r = ck["resume"]
            model.load_state_dict(ck["state_dict"])
            opt.load_state_dict(r["opt"])
            sched.load_state_dict(r["sched"])
            rng.bit_generator.state, stale_rng.bit_generator.state = r["rng"], r["stale_rng"]
            order.perm, order.pos = r["perm"], r["pos"]
            start, run_loss, t_done, run_aux = done[-1] + 1, r["run_loss"], r["time"], r.get("run_aux")
            print(f"resumed from {step_path(out, done[-1])}")

    t0, losses, auxes = time.time() - t_done, [], []
    for step in range(start, args.steps + 1):
        if gpu:
            i = next(order)
            fp = None
            if stale and stale_rng.random() < args.stale_prob:
                fp = gtrain.stale_fp(i, int(stale_rng.integers(len(gtrain))), stale_rng)
            if args.aux_noise or args.film:
                # all shots of a chunk share one fingerprint; aux losses are masked when it is stale
                res = gpu_loss(fmodel, gtrain, i, fp, aux=args.aux_noise and fp is None, shared_fp=True)
                if isinstance(res, tuple):
                    loss, bce, a = res
                    auxes.append(a.detach())
                else:
                    loss = bce = res
            else:
                loss = bce = gpu_loss(fmodel, gtrain, i, fp)
        else:
            _, det, obs, wdet = next(source)
            if stale and stale_rng.random() < args.stale_prob:
                wdet = stale_window(wdet, other_window(), stale_rng)
            loss = bce = chunk_loss(model, grid, det, obs, wdet, args.adaptive, device)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(bce.detach())
        do_val = ds is not None and (step % args.val_every == 0 or step == args.steps)
        save = args.save_every and step % args.save_every == 0
        if step % 250 == 0 or step == args.steps or do_val or save:
            for v in torch.stack(losses).tolist():  # losses are synced only here, not every step
                run_loss = v if run_loss is None else 0.98 * run_loss + 0.02 * v
            for v in torch.stack(auxes).tolist() if auxes else []:
                run_aux = v if run_aux is None else 0.98 * run_aux + 0.02 * v
            losses, auxes = [], []
            msg = f"step {step:>6}  loss {run_loss:.4f}"
            if run_aux is not None:
                msg += f"  aux {run_aux:.4f}"
            if do_val:
                if args.aux_noise or args.film:
                    v = gpu_val_loss(fmodel, gval, aux=args.aux_noise, shared_fp=True)
                    v, va = v if args.aux_noise else (v, None)
                else:
                    v, va = gpu_val_loss(fmodel, gval) if gpu else val_loss(model, grid, val_set, args.adaptive, device), None
                msg += f"  val {v:.4f}" + (f"  val_aux {va:.4f}" if va is not None else "")
            print(f"{msg}  {time.time() - t0:.0f}s")
        if save:
            resume = {
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "rng": rng.bit_generator.state,
                "stale_rng": stale_rng.bit_generator.state,
                "perm": order.perm,
                "pos": order.pos,
                "run_loss": run_loss,
                "run_aux": run_aux,
                "time": time.time() - t0,
            }
            torch.save(
                {"cfg": dict(cfg, step=step), "state_dict": model.state_dict(), "resume": resume}, step_path(out, step)
            )
            print(f"saved {step_path(out, step)}")

    model.to(memory_format=torch.contiguous_format)
    save_checkpoint(out, model, cfg)
    print(f"saved {out}\n")
    tests = ds.test_sets() if ds else sampled_tests(stream, args.adaptive, args.window, args.eval_shots)
    fp_fn = None
    if args.fingerprint != "single":
        fpr_cpu = Fingerprint(fpcfg, grid)
        fp_fn = lambda w: fpr_cpu.of_windows(torch.from_numpy(w.astype(np.uint8))).numpy()
    evaluate(model, grid, stream, tests, args.adaptive, device, fp_fn)


if __name__ == "__main__":
    main()
