"""Generate a fixed dataset so every model and baseline is scored on the same shots.

    python make_dataset.py --distance 3 --rounds 3
    python make_dataset.py --distance 5 --name d5_big --train-chunks 60000
    python make_dataset.py --streams-only --name d3_r3_p0.003_s0   # add new regimes' streams
    python make_dataset.py --streams-only --name d3_r3_p0.003_s0 --stream-seeds 1,2,3,4 \
        --regimes hot_qubit,bursts,bursts_big,telegraph              # extra independent streams

Writes data/<name>/ (layout in dataset.py) and a manifest.json with the
generation args, library versions, array shapes and a SHA-256 of every file.
Every split (train, val, each test state, each drift stream) gets its own seed
derived from --seed, so the splits are independent samples.

--streams-only adds a stream for every regime in drift.REGIMES that an existing
dataset lacks, using the settings and seed in its manifest. Existing files are
not touched; the manifest gains the new files, their hashes and an `updates` entry.
With --stream-seeds it also adds extra independent streams stream_<regime>_seed<N>.npz
for each regime in --regimes (listed under `stream_seeds` in the manifest). Stream
seed 0 is the original stream_<regime>.npz.

Stim only promises identical samples for a seed on the same Stim version and
machine. That is why the data is saved instead of being regenerated.
"""
import argparse
import json
import os
import sys
import time
import zlib
from datetime import datetime, timezone

import numpy as np
import pymatching
import stim

from dataset import FORMAT, pack_det, pack_obs, sha256_file, states_to_arrays
from drift import REGIMES, DriftStream, sample_chunk, sample_stream, test_states


def split_seed(seed, name):
    """Seed for one split, derived from the main seed and the split's name."""
    return int(np.random.SeedSequence([seed, zlib.crc32(name.encode())]).generate_state(1)[0])


def slug(name):
    return "".join(c if c.isalnum() else "_" for c in name.lower())


def make_chunks(stream, rng, n, batch, window, label):
    """n training chunks as packed arrays: det, obs, window_det, plus their NoiseStates."""
    out, states, t0 = {}, [], time.time()
    for i in range(n):
        state, det, obs, wdet = sample_chunk(stream, rng, batch, window)
        parts = {"det": pack_det(det), "obs": pack_obs(obs), "window_det": pack_det(wdet)}
        if not out:
            out = {k: np.empty((n, *v.shape), v.dtype) for k, v in parts.items()}
        for k, v in parts.items():
            out[k][i] = v
        states.append(state)
        if (i + 1) % 1000 == 0 or i + 1 == n:
            print(f"  {label}: {i + 1}/{n} chunks  {time.time() - t0:.0f}s", flush=True)
    return {**out, **states_to_arrays(states)}


def save_npz(folder, stem, files, **arrays):
    """Write one compressed .npz and record its hash, size and array shapes in `files`."""
    filename = f"{stem}.npz"
    path = os.path.join(folder, filename)
    np.savez_compressed(path, **arrays)
    files[filename] = {"sha256": sha256_file(path), "bytes": os.path.getsize(path),
                       "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                                  for k, v in arrays.items()}}
    print(f"wrote {path}  {files[filename]['bytes'] / 1e6:.1f} MB", flush=True)
    return filename


def save_streams(folder, stream, regimes, seed, steps, shots, files, stream_seed=0):
    """Sample and save one drift stream per regime. Returns ({regime: filename}, {stem: seed}).

    stream_seed 0 is the dataset's main stream; other values give independent extra streams.
    """
    out, seeds = {}, {}
    for regime in regimes:
        stem = f"stream_{regime}" if stream_seed == 0 else f"stream_{regime}_seed{stream_seed}"
        seeds[stem] = split_seed(seed, stem)
        rng = np.random.default_rng(seeds[stem])
        states, det, obs = zip(*sample_stream(stream, regime, steps, shots, rng))
        out[regime] = save_npz(folder, stem, files, det=pack_det(np.stack(det)),
                               obs=pack_obs(np.stack(obs)), **states_to_arrays(states))
    return out, seeds


def versions():
    return {"stim": stim.__version__, "pymatching": pymatching.__version__,
            "numpy": np.__version__, "python": sys.version.split()[0]}


def write_manifest(path, manifest):
    """Write via a temporary file so an interrupted write never leaves a half manifest."""
    with open(path + ".tmp", "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(path + ".tmp", path)


def add_streams(folder, stream_seeds=(), regimes=REGIMES):
    """Add the streams the dataset in `folder` lacks; leave existing files untouched.

    That is: the main stream of any regime in drift.REGIMES, plus stream seed N of
    each regime in `regimes` for each N in `stream_seeds`.
    """
    manifest_path = os.path.join(folder, "manifest.json")
    if not os.path.exists(manifest_path):
        raise SystemExit(f"no manifest.json in {folder}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    extra = manifest.setdefault("stream_seeds", {})
    todo = [(r, 0) for r in REGIMES if r not in manifest["streams"]]
    todo += [(r, n) for n in stream_seeds for r in regimes if str(n) not in extra.get(r, {})]
    if not todo:
        print(f"{folder} already has every requested stream")
        return
    if manifest["versions"] != versions():
        print(f"warning: dataset made with {manifest['versions']}, now running {versions()}")
    a = manifest["args"]
    stream = DriftStream(a["distance"], a["rounds"], a["p"])
    files, seeds = manifest["files"], manifest["seeds"]
    added = []
    for regime, n in todo:
        new_files, new_seeds = save_streams(folder, stream, [regime], a["seed"], a["stream_steps"],
                                            a["shots_per_step"], files, stream_seed=n)
        if n == 0:
            manifest["streams"].update(new_files)
        else:
            extra.setdefault(regime, {})[str(n)] = new_files[regime]
        seeds.update(new_seeds)
        added.append(new_files[regime])
    manifest.setdefault("updates", []).append({
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv), "added": added, "versions": versions()})
    write_manifest(manifest_path, manifest)
    print(f"updated {manifest_path}: added {', '.join(added)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=None, help="default: = distance")
    ap.add_argument("--p", type=float, default=0.003)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data")
    ap.add_argument("--name", default=None, help="default: d<distance>_r<rounds>_p<p>_s<seed>")
    ap.add_argument("--train-chunks", type=int, default=30000)
    ap.add_argument("--val-chunks", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=512, help="training shots per chunk")
    ap.add_argument("--window", type=int, default=2000, help="fingerprint shots per chunk")
    ap.add_argument("--test-shots", type=int, default=200000, help="shots per test state")
    ap.add_argument("--test-window", type=int, default=2000, help="fingerprint shots per test state")
    ap.add_argument("--stream-steps", type=int, default=150)
    ap.add_argument("--shots-per-step", type=int, default=2000)
    ap.add_argument("--force", action="store_true", help="overwrite an existing dataset")
    ap.add_argument("--streams-only", action="store_true",
                    help="only add missing streams to the existing dataset --name "
                         "(all other settings come from its manifest)")
    ap.add_argument("--stream-seeds", default="",
                    help="with --streams-only: extra stream seeds to add, e.g. 1,2,3,4 (0 = main stream)")
    ap.add_argument("--regimes", default=",".join(REGIMES),
                    help="regimes that get the extra --stream-seeds (comma-separated)")
    args = ap.parse_args()
    stream_seeds = [int(s) for s in args.stream_seeds.split(",") if s]
    regimes = [r for r in args.regimes.split(",") if r]
    if any(r not in REGIMES for r in regimes):
        raise SystemExit(f"unknown regime in --regimes; choose from {REGIMES}")
    if any(n <= 0 for n in stream_seeds):
        raise SystemExit("--stream-seeds must be positive (seed 0 is the main stream)")
    if stream_seeds and not args.streams_only:
        raise SystemExit("--stream-seeds needs --streams-only (generate the dataset first)")
    if args.streams_only:
        if not args.name:
            raise SystemExit("--streams-only needs --name of an existing dataset")
        return add_streams(os.path.join(args.out, args.name), stream_seeds, regimes)
    args.rounds = args.rounds or args.distance
    args.name = args.name or f"d{args.distance}_r{args.rounds}_p{args.p:g}_s{args.seed}"

    folder = os.path.join(args.out, args.name)
    manifest_path = os.path.join(folder, "manifest.json")
    if os.path.exists(manifest_path):
        if not args.force:
            raise SystemExit(f"{folder} already exists; pass --force to overwrite it")
        os.remove(manifest_path)  # an interrupted rewrite must not look complete
    os.makedirs(folder, exist_ok=True)

    stream = DriftStream(args.distance, args.rounds, args.p)
    nominal = stream.nominal()
    tests = test_states(nominal)
    names = ["train", "val"] + [f"test_{slug(n)}" for n in tests]
    seeds = {n: split_seed(args.seed, n) for n in names}
    files = {}
    save = lambda stem, **arrays: save_npz(folder, stem, files, **arrays)

    print(f"generating {folder}: d={args.distance} r={args.rounds} p={args.p} "
          f"detectors={nominal.num_detectors}")
    for split, n in (("train", args.train_chunks), ("val", args.val_chunks)):
        rng = np.random.default_rng(seeds[split])
        save(split, **make_chunks(stream, rng, n, args.batch, args.window, split))

    test_files = {}
    for name, state in tests.items():
        stem = f"test_{slug(name)}"
        rng = np.random.default_rng(seeds[stem])
        det, obs = stream.sample(state, args.test_shots, rng.integers(2**31))
        wdet, _ = stream.sample(state, args.test_window, rng.integers(2**31))
        test_files[name] = save(stem, det=pack_det(det), obs=pack_obs(obs),
                                window_det=pack_det(wdet), **states_to_arrays([state]))

    stream_files, stream_seeds = save_streams(folder, stream, REGIMES, args.seed,
                                              args.stream_steps, args.shots_per_step, files)
    seeds.update(stream_seeds)

    manifest = {
        "format": FORMAT,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "args": vars(args),
        "seeds": seeds,
        "versions": versions(),
        "num_detectors": nominal.num_detectors,
        "num_observables": nominal.num_observables,
        "test_sets": test_files,
        "streams": stream_files,
        "files": files,
    }
    write_manifest(manifest_path, manifest)
    total = sum(f["bytes"] for f in files.values())
    print(f"wrote {manifest_path}  total {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
