"""Train the neural decoder on Stim data with randomly drawn noise.

Each training step uses one chunk: a random NoiseState (uniform scale change,
and with some probability a cluster of hot qubits), a batch of shots from it,
and, for the adaptive model, a separate window of shots from the same noise
to compute the fingerprint. The model never sees the NoiseState itself.

    python train.py --distance 3 --rounds 3 --steps 3000            # static
    python train.py --distance 3 --rounds 3 --steps 3000 --adaptive # adaptive
    python train.py --data data/d3_r3_p0.003_s0 --epochs 2          # fixed dataset
    python train.py --data data/d3_wide --epochs 2 --adaptive --stale-prob 0.3
    python train.py --nominal-only --steps 60000 --out decoder_static_d3_nominal_s0.pt  # no drift

Without --data, every chunk is sampled fresh. With --data, chunks come from a
dataset made by make_dataset.py, reshuffled every epoch. Validation loss on the
val chunks is printed every --val-every steps, and the final scores use the
saved test sets. Distance, rounds, p, batch and window then come from the dataset.

--stale-prob (adaptive only) teaches the model not to over-trust its
fingerprint. With that probability a batch gets a fingerprint that does not
match its shots: half the time another random chunk's window (fully stale),
half the time a window that is part another chunk, part this chunk (caught
mid-transition). It draws from its own RNG, so the chunk order is unchanged.

--nominal-only (no --data) trains on nominal noise only: every chunk uses
NoiseState(), a specialist baseline for undrifted noise.

Start at d=3 on a laptop. d=5 wants a GPU (Colab/Kaggle) and many more steps.
"""
import argparse
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
    """Fresh test shots for each test state: [(name, state, det, obs, window_det)]."""
    tests = []
    for i, (name, st) in enumerate(test_states(stream.nominal()).items()):
        det, obs = stream.sample(st, shots, seed + i)
        wdet = stream.sample(st, window, seed + 100 + i)[0] if adaptive else None
        tests.append((name, st, det, obs, wdet))
    return tests


def evaluate(model, grid, stream, tests, adaptive, device):
    """Print and return test-set LERs: {state name: {"static", "oracle", "neural"}}."""
    model.eval()
    static, oracle = Static(stream.nominal()), Oracle(stream)
    print(f"{'state':<14}{'static MWPM':>12}{'oracle MWPM':>12}{'neural':>10}")
    out = {}
    for name, st, det, obs, wdet in tests:
        fp = wdet.mean(axis=0).astype(np.float32) if adaptive else None
        with torch.no_grad():
            pred = np.concatenate([
                (model(build_input(grid, det[j:j + 4096], fp).to(device)) > 0).cpu().numpy()
                for j in range(0, len(det), 4096)])
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
    """Mean loss over all validation chunks (they all have the same batch size)."""
    model.eval()
    loss = np.mean([chunk_loss(model, grid, det, obs, wdet, adaptive, device).item()
                    for _, det, obs, wdet in chunks])
    model.train()
    return float(loss)


def stale_window(wdet, other, rng):
    """A fingerprint window that does not match the shots.

    Half the time another chunk's window; otherwise its first k rows followed by
    this chunk's remaining rows, with k uniform in [0, window).
    """
    if rng.random() < 0.5:
        return other
    k = int(rng.uniform() * len(wdet))
    return np.concatenate([other[:k], wdet[k:]])


def fresh_chunks(stream, rng, batch, window, nominal=False):
    """Endless freshly sampled chunks (no --data)."""
    while True:
        yield sample_chunk(stream, rng, batch, window, nominal)


def saved_chunks(chunks, rng):
    """Endless saved chunks, in a new random order every epoch."""
    while True:
        for i in rng.permutation(len(chunks)):
            yield chunks[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="dataset folder from make_dataset.py")
    ap.add_argument("--distance", type=int, default=None, help="default 3, or from --data")
    ap.add_argument("--rounds", type=int, default=None, help="default 3, or from --data")
    ap.add_argument("--p", type=float, default=None, help="default 0.003, or from --data")
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--window", type=int, default=None,
                    help="shots used for the fingerprint (default 2000, or from --data)")
    ap.add_argument("--steps", type=int, default=None,
                    help="default 3000; with --data, epochs x train chunks")
    ap.add_argument("--epochs", type=int, default=1, help="passes over the train chunks (--data only)")
    ap.add_argument("--val-every", type=int, default=500, help="steps between val losses (--data only)")
    ap.add_argument("--batch", type=int, default=None, help="default 512, or from --data")
    ap.add_argument("--stale-prob", type=float, default=0.0,
                    help="probability of a mismatched fingerprint per batch (adaptive only)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--nominal-only", action="store_true",
                    help="train on nominal noise only (no drift); not with --data")
    ap.add_argument("--eval-shots", type=int, default=50000, help="ignored with --data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.nominal_only and args.data:
        raise SystemExit("--nominal-only samples on the fly; it cannot be combined with --data")
    rng = np.random.default_rng(args.seed)
    ds = train_set = val_set = None
    if args.data:
        ds = Dataset(args.data)
        ds.fill_args(args, FROM_DATA)
        train_set, val_set = ds.chunks("train"), ds.chunks("val")
        args.steps = args.steps or args.epochs * len(train_set)
    for k, v in DEFAULTS.items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    stream = DriftStream(args.distance, args.rounds, args.p)
    nominal = stream.nominal()
    grid = DetectorGrid(nominal)
    model = ConvDecoder(in_ch=2 if args.adaptive else 1, hidden=args.hidden, layers=args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps)
    mode = "adaptive" if args.adaptive else "static"
    print(f"training {mode} d={args.distance} r={args.rounds} grid={grid.shape} device={device} "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    if ds:
        print(f"data {args.data}: {len(train_set)} train chunks, {len(val_set)} val chunks, "
              f"{args.steps} steps = {args.steps / len(train_set):.2f} epochs")
        source = saved_chunks(train_set, rng)
    else:
        source = fresh_chunks(stream, rng, args.batch, args.window if args.adaptive else 0, args.nominal_only)
    # another random chunk's fingerprint window, for --stale-prob
    stale_rng = np.random.default_rng([args.seed, 1])
    if ds:
        other_window = lambda: train_set[stale_rng.integers(len(train_set))][3]
    else:
        other_window = lambda: sample_chunk(stream, stale_rng, 0, args.window)[3]

    t0, run_loss = time.time(), None
    for step in range(1, args.steps + 1):
        _, det, obs, wdet = next(source)
        if args.adaptive and args.stale_prob > 0 and stale_rng.random() < args.stale_prob:
            wdet = stale_window(wdet, other_window(), stale_rng)
        loss = chunk_loss(model, grid, det, obs, wdet, args.adaptive, device)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run_loss = loss.item() if run_loss is None else 0.98 * run_loss + 0.02 * loss.item()
        do_val = ds is not None and (step % args.val_every == 0 or step == args.steps)
        if step % 250 == 0 or step == args.steps or do_val:
            msg = f"step {step:>6}  loss {run_loss:.4f}"
            if do_val:
                msg += f"  val {val_loss(model, grid, val_set, args.adaptive, device):.4f}"
            print(f"{msg}  {time.time() - t0:.0f}s")

    out = args.out or f"decoder_{mode}_d{args.distance}.pt"
    cfg = dict(vars(args), adaptive=args.adaptive)
    save_checkpoint(out, model, cfg)
    print(f"saved {out}\n")
    tests = ds.test_sets() if ds else sampled_tests(stream, args.adaptive, args.window, args.eval_shots)
    evaluate(model, grid, stream, tests, args.adaptive, device)


if __name__ == "__main__":
    main()
