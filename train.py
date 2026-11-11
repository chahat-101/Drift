"""Train the neural decoder on Stim data with randomly drawn noise.

Each training step uses one chunk: a random NoiseState (a global scale change and,
with some probability, a cluster of hot qubits), a batch of shots from it and, for
the adaptive model, a separate window of shots from the same noise for the
fingerprint. The model never sees the NoiseState.

    python train.py --distance 3 --rounds 3 --steps 3000                 # static
    python train.py --distance 3 --rounds 3 --steps 3000 --adaptive      # adaptive
    python train.py --data data/d3_r3_p0.003_s0 --epochs 2               # fixed dataset
    python train.py --data data/d3_wide --epochs 2 --adaptive --stale-prob 0.3
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
from decoder import ConvDecoder, DetectorGrid, build_input, save_checkpoint
from drift import DriftStream, sample_chunk, test_states

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


def evaluate(model, grid, stream, tests, adaptive, device):
    """Print and return test-set LERs as {state name: {"static", "oracle", "neural"}}."""
    model.eval()
    static, oracle = Static(stream.nominal()), Oracle(stream)
    print(f"{'state':<14}{'static MWPM':>12}{'oracle MWPM':>12}{'neural':>10}")
    out = {}
    for name, st, det, obs, wdet in tests:
        fp = wdet.mean(axis=0).astype(np.float32) if adaptive else None
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

    Gives the same inputs as Chunks and build_input. fp[i] is the fingerprint of chunk i,
    computed once. Packed windows are kept only if keep_windows (needed for --stale-prob).
    """

    def __init__(self, chunks, grid, device, adaptive, keep_windows):
        a, self.n_det, self.batch, self.device = chunks.a, chunks.n_det, chunks.batch, device
        self.n = len(chunks)
        self.det = torch.from_numpy(a["det"]).to(device)
        self.obs = torch.from_numpy(a["obs"]).to(device)
        self.flat = torch.from_numpy(grid.flat).to(device)
        self.shape = grid.shape
        self.shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=device)
        self.win = self.fp = None
        if adaptive:
            win = torch.from_numpy(a["window_det"]).to(device)
            self.window = win.shape[1]
            self.fp = torch.cat([self.rates(win[i : i + 256]) for i in range(0, self.n, 256)])
            self.win = win if keep_windows else None
            del win

    def __len__(self):
        return self.n

    def bits(self, packed, count):
        """np.unpackbits along the last axis, on the GPU."""
        b = (packed.unsqueeze(-1) >> self.shifts) & 1
        return b.flatten(-2)[..., :count]

    def rates(self, packed):
        """Per-detector firing rate of packed windows with shape (..., window, bytes), as float32."""
        return (self.bits(packed, self.n_det).sum(-2, dtype=torch.float64) / packed.shape[-2]).float()

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
            f = self.to_grid(fp[None]).unsqueeze(1).expand(len(x), -1, -1, -1, -1)
            x = torch.cat([x, f * 10.0], dim=1)
        return x.contiguous(memory_format=torch.channels_last_3d), y

    def stale_fp(self, i, other, rng):
        """Fingerprint of stale_window(window i, window other, rng), drawing from rng in the same order."""
        if rng.random() < 0.5:
            return self.fp[other]
        k = int(rng.uniform() * self.window)
        w = torch.cat([self.win[other, :k], self.win[i, k:]])
        return self.rates(w)


def gpu_loss(model, data, i, fp=None):
    x, y = data.inputs(i, fp)
    return F.binary_cross_entropy_with_logits(model(x), y)


@torch.no_grad()
def gpu_val_loss(model, data):
    model.eval()
    loss = np.mean([gpu_loss(model, data, i).item() for i in range(len(data))])
    model.train()
    return float(loss)


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
    model = ConvDecoder(in_ch=2 if args.adaptive else 1, hidden=args.hidden, layers=args.layers).to(device)
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
        gtrain = GpuChunks(train_set, grid, device, args.adaptive, stale)
        gval = GpuChunks(val_set, grid, device, args.adaptive, False)
        del train_set, val_set, source, other_window  # free the CPU copies
        print(f"data on {device}: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    fmodel = torch.compile(model) if args.compile else model

    out = args.out or f"decoder_{mode}_d{args.distance}.pt"
    cfg = dict(vars(args), adaptive=args.adaptive)
    start, run_loss, t_done = 1, None, 0.0
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
            start, run_loss, t_done = done[-1] + 1, r["run_loss"], r["time"]
            print(f"resumed from {step_path(out, done[-1])}")

    t0, losses = time.time() - t_done, []
    for step in range(start, args.steps + 1):
        if gpu:
            i = next(order)
            fp = None
            if stale and stale_rng.random() < args.stale_prob:
                fp = gtrain.stale_fp(i, int(stale_rng.integers(len(gtrain))), stale_rng)
            loss = gpu_loss(fmodel, gtrain, i, fp)
        else:
            _, det, obs, wdet = next(source)
            if stale and stale_rng.random() < args.stale_prob:
                wdet = stale_window(wdet, other_window(), stale_rng)
            loss = chunk_loss(model, grid, det, obs, wdet, args.adaptive, device)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.detach())
        do_val = ds is not None and (step % args.val_every == 0 or step == args.steps)
        save = args.save_every and step % args.save_every == 0
        if step % 250 == 0 or step == args.steps or do_val or save:
            for v in torch.stack(losses).tolist():  # losses are synced only here, not every step
                run_loss = v if run_loss is None else 0.98 * run_loss + 0.02 * v
            losses = []
            msg = f"step {step:>6}  loss {run_loss:.4f}"
            if do_val:
                v = gpu_val_loss(fmodel, gval) if gpu else val_loss(model, grid, val_set, args.adaptive, device)
                msg += f"  val {v:.4f}"
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
    evaluate(model, grid, stream, tests, args.adaptive, device)


if __name__ == "__main__":
    main()
