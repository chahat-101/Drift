"""3D CNN decoder.

Detection events are laid out on a (time, X, Y) grid from the detector coordinates.
The network outputs one logit: whether the logical observable flipped.

Two modes:
  static    one input channel, the shot's detection events.
  adaptive  extra channels hold the fingerprint, computed from the last `window` shots.
            The model never sees the true noise.

Fingerprints of adaptive models (cfg["fingerprint"]; checkpoints without the key are "single"):
  single      firing rate of each detector over the window (x10).
  multiscale  long rate (whole window), short rate (last `short` shots) and short - long (all x10).
  mspair      multiscale plus one channel per grid offset o in cfg["pair_offsets"]: at detector i,
              cov(x_i, x_{i+o}) over the whole window, times cfg["pair_scale"]; 0 if i+o has no detector.

Model options (cfg["aux_noise"], cfg["film"]; checkpoints without the keys have both off): see ConvDecoder.

NeuralDecoder wraps a trained model in the interface of baselines.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines import ShotWindow


class DetectorGrid:
    """Maps a flat detection-event vector onto a (T, X, Y) grid."""

    def __init__(self, circuit):
        coords = circuit.get_detector_coordinates()
        n = circuit.num_detectors
        c = np.array([coords[i][:3] for i in range(n)], dtype=float)
        axes = [np.unique(c[:, k]) for k in range(3)]
        self.axes = axes  # detector coordinate values along x, y, t
        self.idx = [np.searchsorted(axes[k], c[:, k]) for k in range(3)]
        x, y, t = (len(a) for a in axes)
        self.shape = (t, x, y)
        self.flat = (self.idx[2] * x + self.idx[0]) * y + self.idx[1]
        self.n = n

    def to_grid(self, det):
        """(B, n) bool or float array -> (B, T, X, Y) float32 tensor."""
        det = np.asarray(det, dtype=np.float32)
        out = np.zeros((len(det), int(np.prod(self.shape))), dtype=np.float32)
        out[:, self.flat] = det
        return torch.from_numpy(out.reshape(len(det), *self.shape))


class ConvDecoder(nn.Module):
    """3D CNN: 4 conv layers, then mean and max pooling and a small head -> one logit.

    aux_noise (adaptive only): extra heads that do not change the logit. The last feature map, averaged over
    time, is sampled bilinearly at each physical qubit's (x, y) position (qubit_pos: (n_qubits, 2) grid_sample
    coordinates in [-1, 1], (y, x) order) and a shared MLP (H -> 32 -> 1) predicts that qubit's extra noise
    p_hot x 10; a head on the pooled vector predicts log(scale).
    film (adaptive only): an encoder (2 x Conv3d 3^3, 16 channels, GELU) of the fingerprint channels gives, per
    layer, gamma and beta (zero-initialised 1x1x1 convs) and h <- h * (1 + gamma) + beta after the layer's GELU.
    The fingerprint stays an input channel too.
    """

    def __init__(self, in_ch=1, hidden=48, layers=4, aux_noise=False, film=False, qubit_pos=None):
        super().__init__()
        blocks, ch = [], in_ch
        for _ in range(layers):
            blocks += [nn.Conv3d(ch, hidden, 3, padding=1), nn.GELU()]
            ch = hidden
        self.body = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(2 * hidden, 64), nn.GELU(), nn.Linear(64, 1))
        # created after the decoding path, so the same seed gives the same decoding weights with or without them
        self.aux_noise, self.film = aux_noise, film
        if aux_noise:
            self.register_buffer("qubit_pos", torch.as_tensor(qubit_pos, dtype=torch.float32), persistent=False)
            self.noise_head = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1))
            self.scale_head = nn.Sequential(nn.Linear(2 * hidden, 32), nn.GELU(), nn.Linear(32, 1))
        if film:
            self.film_enc = nn.Sequential(nn.Conv3d(in_ch - 1, 16, 3, padding=1), nn.GELU(),
                                          nn.Conv3d(16, 16, 3, padding=1), nn.GELU())
            self.film_out = nn.ModuleList(nn.Conv3d(16, 2 * hidden, 1) for _ in range(layers))
            for m in self.film_out:
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def features(self, x, shared_fp=False):
        """Last feature map (B, H, T, X, Y). shared_fp: every shot in x has the same fingerprint, so the FiLM
        encoder runs on the first shot only."""
        if not self.film:
            return self.body(x)
        e = self.film_enc(x[:1, 1:] if shared_fp else x[:, 1:])
        h = x
        for k, film in enumerate(self.film_out):
            h = self.body[2 * k + 1](self.body[2 * k](h))
            gamma, beta = film(e).chunk(2, dim=1)
            h = h * (1 + gamma) + beta
        return h

    def forward(self, x, aux=False, shared_fp=False):  # x: (B, C, T, X, Y)
        """The logit; with aux also the noise map (B, n_qubits) (p_hot x 10) and log(scale) (B,)."""
        f = self.features(x, shared_fp)
        h = f.flatten(2)
        pooled = torch.cat([h.mean(-1), h.amax(-1)], dim=1)
        logit = self.head(pooled).squeeze(-1)
        if not aux:
            return logit
        m = f.mean(2)  # (B, H, X, Y): grid_sample's (H_in, W_in) = (X, Y), so coordinates are (y, x)
        grid = self.qubit_pos.view(1, 1, -1, 2).expand(len(m), -1, -1, -1)
        q = F.grid_sample(m, grid, mode="bilinear", align_corners=True).squeeze(2)  # (B, H, n_qubits)
        noise = self.noise_head(q.transpose(1, 2)).squeeze(-1)
        return logit, noise, self.scale_head(pooled).squeeze(-1)


def qubit_positions(grid, circuit):
    """(qubit ids, (n_qubits, 2) grid_sample coordinates (y, x)) of every physical qubit, ids sorted.

    A qubit's x (y) becomes a fractional index on the detector grid's x (y) axis, mapped to [-1, 1]
    (align_corners=True); positions outside the grid are clamped to its edge.
    """
    ids = sorted(circuit.get_final_qubit_coordinates())
    q = circuit.get_final_qubit_coordinates()
    pos = []
    for i in ids:
        u = [np.interp(q[i][k], grid.axes[k], np.arange(len(grid.axes[k]))) / (len(grid.axes[k]) - 1) * 2 - 1
             for k in (1, 0)]
        pos.append([float(v) for v in u])
    return ids, pos


def noise_targets(scale, hot_p, hot_qubits, qubit_ids):
    """NoiseState arrays (as dataset.states_to_arrays) -> (p_hot x 10 per qubit (N, n_qubits), log(scale) (N,))."""
    col = {q: k for k, q in enumerate(qubit_ids)}
    m = np.zeros((len(scale), len(qubit_ids)), dtype=np.float32)
    for i, row in enumerate(np.asarray(hot_qubits)):
        for q in row[row >= 0]:
            m[i, col[int(q)]] = 10.0 * hot_p[i]
    return m, np.log(np.asarray(scale)).astype(np.float32)


def build_input(grid, det, fingerprint=None):
    """Stack detection events and, if given, the fingerprint into model input.

    fingerprint: (n,) rates (single, scaled here) or (C, n) channels from Fingerprint (already scaled).
    """
    x = grid.to_grid(det).unsqueeze(1)
    if fingerprint is None:
        return x
    if np.ndim(fingerprint) == 2:
        f = grid.to_grid(fingerprint).unsqueeze(0).expand(len(det), -1, -1, -1, -1)
        return torch.cat([x, f], dim=1)
    f = grid.to_grid(fingerprint[None, :]).unsqueeze(1).expand(len(det), -1, -1, -1, -1)
    # rates are scaled up to the range of the 0/1 events
    return torch.cat([x, f * 10.0], dim=1)


FINGERPRINTS = ("single", "multiscale", "mspair")
SHORT = 200  # shots in the short window (multiscale, mspair)


def grid_pos(grid):
    """(n, 3) grid index (t, x, y) of each detector."""
    return np.stack([grid.idx[2], grid.idx[0], grid.idx[1]], axis=1)


def pair_offsets(grid, circuit, k=8):
    """The k most frequent grid offsets v - u over matching-graph edges (u, v) between two detectors.

    Each offset gets the sign that makes it lexicographically positive. Ties: smaller offset first.
    """
    from collections import Counter

    from circuits import matcher_from_circuit

    pos, count = grid_pos(grid), Counter()
    for u, v, _ in matcher_from_circuit(circuit).edges():
        if v is None or max(u, v) >= grid.n:
            continue
        o = tuple(int(a) for a in pos[v] - pos[u])
        count[max(o, tuple(-a for a in o))] += 1
    return [list(o) for o, _ in sorted(count.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def pair_partners(grid, offsets):
    """(k, n) int array: the detector at grid position pos(i) + offset, or -1."""
    pos = grid_pos(grid)
    where = {tuple(p): i for i, p in enumerate(pos)}
    return np.array([[where.get(tuple(p + np.array(o)), -1) for p in pos] for o in offsets], dtype=np.int64)


class Fingerprint:
    """Fingerprint channels of one model cfg, computed from integer sums over shot windows (any device).

    The same sums give the same channels in training (whole windows) and in eval (cumulative sums).
    """

    def __init__(self, cfg, grid, device="cpu"):
        self.mode = cfg.get("fingerprint", "single")
        self.short = cfg.get("short", SHORT)
        self.pair_scale = cfg.get("pair_scale", 1.0)
        self.n = grid.n
        self.partners = None
        if self.mode == "mspair":
            self.partners = torch.as_tensor(pair_partners(grid, cfg["pair_offsets"]), device=device)
            k = len(self.partners)
            self.pidx = self.partners.clamp(min=0).flatten()
            self.sidx = torch.arange(grid.n, device=device).repeat(k)
        self.channels = 1 if self.mode == "single" else 3 + (0 if self.partners is None else len(self.partners))

    def products(self, bits):
        """x_i * x_{i+o} for every offset: (..., k * n) from 0/1 bits (..., n)."""
        return bits[..., self.sidx] & bits[..., self.pidx]

    def features(self, n, s, n_short=None, s_short=None, s_pair=None):
        """Channels (..., C, n) from window lengths n (...,), detector sums s (..., n), short-window
        length and sums, and pair sums s_pair (..., k, n). Empty windows give 0."""
        n = n.clamp(min=1).double()[..., None]
        long = (s.double() / n).float()
        if self.mode == "single":
            return (long * 10.0).unsqueeze(-2)
        short = (s_short.double() / n_short.clamp(min=1).double()[..., None]).float()
        ch = [long * 10.0, short * 10.0, (short - long) * 10.0]
        if self.partners is not None:
            m = s.double() / n
            cov = s_pair.double() / n[..., None] - m[..., None, :] * m[..., self.partners.clamp(min=0)]
            cov = torch.where(self.partners >= 0, cov, torch.zeros_like(cov))
            ch += list((cov * self.pair_scale).float().unbind(-2))
        return torch.stack(ch, dim=-2)

    def of_windows(self, bits):
        """Channels (..., C, n) of whole windows of 0/1 uint8 bits (..., rows, n); short = the last rows."""
        rows = bits.shape[-2]
        n = torch.full(bits.shape[:-2], rows, device=bits.device)
        s = bits.sum(-2, dtype=torch.int64)
        if self.mode == "single":
            return self.features(n, s)
        ns = torch.full_like(n, min(rows, self.short))
        ss = bits[..., rows - min(rows, self.short) :, :].sum(-2, dtype=torch.int64)
        sp = None
        if self.partners is not None:
            sp = self.products(bits).sum(-2, dtype=torch.int64).unflatten(-1, self.partners.shape)
        return self.features(n, s, ns, ss, sp)


def in_channels(cfg):
    """Model input channels: 1 for static; 1 + fingerprint channels for adaptive."""
    if not cfg["adaptive"]:
        return 1
    mode = cfg.get("fingerprint", "single")
    return 2 if mode == "single" else 4 + (len(cfg["pair_offsets"]) if mode == "mspair" else 0)


def save_checkpoint(path, model, cfg):
    torch.save({"cfg": cfg, "state_dict": model.state_dict()}, path)


def load_checkpoint(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ck["cfg"]
    model = ConvDecoder(in_ch=in_channels(cfg), hidden=cfg["hidden"], layers=cfg["layers"],
                        aux_noise=cfg.get("aux_noise", False), film=cfg.get("film", False), qubit_pos=cfg.get("aux_pos"))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg


class NeuralDecoder:
    """A trained model behind the decode/observe interface.

    If frozen_fingerprint is given, an adaptive model uses it for every shot and observe() does nothing.
    """

    def __init__(self, path, nominal_circuit, name=None, device="cpu", batch=4096, frozen_fingerprint=None):
        self.model, self.cfg = load_checkpoint(path, device)
        if self.cfg.get("fingerprint", "single") != "single":
            raise SystemExit(f"{path}: NeuralDecoder supports single-fingerprint models only; use ensemble_eval.py")
        self.model.to(device)
        self.device, self.batch = device, batch
        self.grid = DetectorGrid(nominal_circuit)
        self.adaptive = self.cfg["adaptive"]
        self.buf = ShotWindow(self.cfg["window"])
        self.name = name or ("neural_adaptive" if self.adaptive else "neural_static")
        self.frozen = frozen_fingerprint

    def fingerprint(self):
        if self.frozen is not None:
            return self.frozen
        X = self.buf.array()
        if X is None:
            return np.zeros(self.grid.n, dtype=np.float32)
        return X.mean(axis=0).astype(np.float32)

    @torch.no_grad()
    def decode(self, det, state=None):
        fp = self.fingerprint() if self.adaptive else None
        preds = []
        for i in range(0, len(det), self.batch):
            x = build_input(self.grid, det[i : i + self.batch], fp).to(self.device)
            preds.append((self.model(x) > 0).cpu().numpy())
        return np.concatenate(preds)[:, None]

    def observe(self, det):
        if self.frozen is None:
            self.buf.add(det)
