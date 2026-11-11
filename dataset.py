"""Fixed datasets written by make_dataset.py.

A dataset is a folder data/<name>/ with a manifest.json and these files:

    train.npz, val.npz            training chunks: one NoiseState, `batch` shots with
                                  observables and `window` fingerprint shots from that noise
    test_<state>.npz              one fixed test set per state in drift.test_states()
    stream_<regime>.npz           one drifting stream per regime in drift.REGIMES
    stream_<regime>_seed<N>.npz   extra independent streams (stream seed N)

Detection events are bit-packed along the detector axis and observables along the
shot axis. NoiseStates are stored as arrays `scale`, `hot_p` and `hot_qubits`
(padded with -1). Each file is checked against the SHA-256 in the manifest before use.
"""

import hashlib
import json
import math
import os

import numpy as np

from drift import NoiseState

FORMAT = 1


def pack_det(det):
    """Bit-pack detection events (..., shots, detectors) along the detector axis."""
    return np.packbits(det, axis=-1)


def unpack_det(packed, num_detectors):
    return np.unpackbits(packed, axis=-1, count=num_detectors).astype(bool)


def pack_obs(obs):
    """Bit-pack observables (..., shots, observables) along the shot axis."""
    return np.packbits(obs, axis=-2)


def unpack_obs(packed, shots):
    return np.unpackbits(packed, axis=-2, count=shots).astype(bool)


def states_to_arrays(states):
    """A list of NoiseStates as arrays: scale, hot_p, and hot_qubits padded with -1."""
    width = max([len(s.hot_qubits) for s in states] + [1])
    hot = np.full((len(states), width), -1, dtype=np.int32)
    for i, s in enumerate(states):
        hot[i, : len(s.hot_qubits)] = s.hot_qubits
    return {
        "scale": np.array([s.scale for s in states], dtype=np.float64),
        "hot_p": np.array([s.hot_p for s in states], dtype=np.float64),
        "hot_qubits": hot,
    }


def state_at(arrays, i):
    """The i-th NoiseState stored by states_to_arrays."""
    hot = tuple(int(q) for q in arrays["hot_qubits"][i] if q >= 0)
    return NoiseState(float(arrays["scale"][i]), hot, float(arrays["hot_p"][i]))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Chunks:
    """Training chunks, bit-packed in memory and unpacked one at a time. chunks[i] is (state, det, obs, window_det)."""

    def __init__(self, arrays, num_detectors):
        self.a, self.n_det = arrays, num_detectors
        self.batch = arrays["det"].shape[1]

    def __len__(self):
        return len(self.a["scale"])

    def __getitem__(self, i):
        return (
            state_at(self.a, i),
            unpack_det(self.a["det"][i], self.n_det),
            unpack_obs(self.a["obs"][i], self.batch),
            unpack_det(self.a["window_det"][i], self.n_det),
        )

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def drop_windows(self):
        """Free the fingerprint windows (static models do not read them)."""
        w = self.a["window_det"]
        self.a["window_det"] = np.empty((w.shape[0], 0, w.shape[2]), w.dtype)


class Dataset:
    """A dataset folder. Every file is hash-checked when it is loaded."""

    def __init__(self, path):
        self.path = path
        manifest = os.path.join(path, "manifest.json")
        if not os.path.exists(manifest):
            raise SystemExit(f"no manifest.json in {path}; generate it with make_dataset.py")
        with open(manifest) as f:
            self.manifest = json.load(f)
        if self.manifest.get("format") != FORMAT:
            raise SystemExit(f"{path} has dataset format {self.manifest.get('format')}, expected {FORMAT}")
        self.args = self.manifest["args"]
        self.n_det = self.manifest["num_detectors"]

    def load(self, filename):
        """All arrays in one file, after checking its SHA-256 against the manifest."""
        entry = self.manifest["files"].get(filename)
        if entry is None:
            raise SystemExit(f"{filename} is not listed in {self.path}/manifest.json")
        full = os.path.join(self.path, filename)
        if not os.path.exists(full):
            raise SystemExit(f"{full} is missing")
        if sha256_file(full) != entry["sha256"]:
            raise SystemExit(f"{full} does not match its SHA-256 in the manifest; regenerate the dataset")
        with np.load(full) as f:
            return {k: f[k] for k in f.files}

    def chunks(self, split):
        """Training or validation chunks ("train" or "val")."""
        return Chunks(self.load(f"{split}.npz"), self.n_det)

    def test_sets(self):
        """[(name, state, det, obs, window_det)] for each fixed test state."""
        out = []
        for name, filename in self.manifest["test_sets"].items():
            a = self.load(filename)
            out.append(
                (
                    name,
                    state_at(a, 0),
                    unpack_det(a["det"], self.n_det),
                    unpack_obs(a["obs"], len(a["det"])),
                    unpack_det(a["window_det"], self.n_det),
                )
            )
        return out

    def stream(self, regime, stream_seed=0):
        """[(state, det, obs)] for each step of the saved stream for `regime`.

        stream_seed 0 is the main stream; N > 0 is stream_<regime>_seed<N>.
        """
        if stream_seed == 0:
            filename = self.manifest["streams"].get(regime)
        else:
            filename = self.manifest.get("stream_seeds", {}).get(regime, {}).get(str(stream_seed))
        if filename is None:
            raise SystemExit(f"{self.path} has no stream for regime {regime!r}, stream seed {stream_seed}")
        a = self.load(filename)
        shots = a["det"].shape[1]
        det, obs = unpack_det(a["det"], self.n_det), unpack_obs(a["obs"], shots)
        return [(state_at(a, t), det[t], obs[t]) for t in range(len(det))]

    def nominal_fingerprint(self):
        """Per-detector firing rate of the nominal test set's fingerprint shots."""
        a = self.load(self.manifest["test_sets"]["nominal"])
        return unpack_det(a["window_det"], self.n_det).mean(axis=0).astype(np.float32)

    def fill_args(self, args, keys):
        """Fill argparse `args` from the manifest.

        keys maps argument name to manifest arg name. Arguments left at None take the
        dataset's value; an explicit value that disagrees is an error.
        """
        for arg, key in keys.items():
            given, value = getattr(args, arg), self.args[key]
            if given is not None and not math.isclose(given, value):
                flag = "--" + arg.replace("_", "-")
                raise SystemExit(
                    f"{flag} {given} does not match {self.path} ({key}={value}); "
                    f"leave it out to use the dataset's value"
                )
            setattr(args, arg, value)

    def check_model(self, cfg, source):
        """Exit if a model's distance, rounds or p differ from this dataset's."""
        for k in ("distance", "rounds", "p"):
            if not math.isclose(cfg[k], self.args[k]):
                raise SystemExit(f"{source} was trained with {k}={cfg[k]}, but {self.path} has {k}={self.args[k]}")
