"""Neural decoder.

Detection events are laid out on a 3D grid (time x X x Y) using the detector
coordinates Stim provides, then read by a small 3D CNN that outputs one logit:
"did the logical observable flip?".

Two modes:
  static   : input channel 0 = this shot's detection events
  adaptive : channel 1 = the fingerprint, i.e. each detector's firing rate over
             the last `window` shots. The decoder never sees the true noise; it
             only sees statistics of its own recent syndromes. This is the core
             idea of the project: calibration-free adaptation.

This is a deliberately simple starting point. Swap the CNN for a transformer
or GNN later; keep the interface (decode / observe) so eval.py still works.
"""
import numpy as np
import torch
import torch.nn as nn

from baselines import ShotWindow


class DetectorGrid:
    """Maps a flat detection-event vector onto a (T, X, Y) grid."""

    def __init__(self, circuit):
        coords = circuit.get_detector_coordinates()
        n = circuit.num_detectors
        c = np.array([coords[i][:3] for i in range(n)], dtype=float)
        axes = [np.unique(c[:, k]) for k in range(3)]
        self.idx = [np.searchsorted(axes[k], c[:, k]) for k in range(3)]
        x, y, t = (len(a) for a in axes)
        self.shape = (t, x, y)
        self.flat = (self.idx[2] * x + self.idx[0]) * y + self.idx[1]
        self.n = n

    def to_grid(self, det):
        """det: (B, n) bool/float array -> (B, T, X, Y) float32 tensor."""
        det = np.asarray(det, dtype=np.float32)
        out = np.zeros((len(det), int(np.prod(self.shape))), dtype=np.float32)
        out[:, self.flat] = det
        return torch.from_numpy(out.reshape(len(det), *self.shape))


class ConvDecoder(nn.Module):
    def __init__(self, in_ch=1, hidden=48, layers=4):
        super().__init__()
        blocks, ch = [], in_ch
        for _ in range(layers):
            blocks += [nn.Conv3d(ch, hidden, 3, padding=1), nn.GELU()]
            ch = hidden
        self.body = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(2 * hidden, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):  # x: (B, C, T, X, Y)
        h = self.body(x).flatten(2)
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], dim=1)).squeeze(-1)


def build_input(grid, det, fingerprint=None):
    """Stack detection events (and optionally the fingerprint) into model input."""
    x = grid.to_grid(det).unsqueeze(1)
    if fingerprint is None:
        return x
    f = grid.to_grid(fingerprint[None, :]).unsqueeze(1).expand(len(det), -1, -1, -1, -1)
    # scale rates up so they sit on a similar range to the 0/1 events
    return torch.cat([x, f * 10.0], dim=1)


def save_checkpoint(path, model, cfg):
    torch.save({"cfg": cfg, "state_dict": model.state_dict()}, path)


def load_checkpoint(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ck["cfg"]
    model = ConvDecoder(in_ch=2 if cfg["adaptive"] else 1, hidden=cfg["hidden"], layers=cfg["layers"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg


class NeuralDecoder:
    """Wraps a trained model in the same interface as the baselines.

    frozen_fingerprint: for adaptive models, a fixed fingerprint used for every
    shot; observe() then does nothing (a control for "how much is adaptation").
    """

    def __init__(self, path, nominal_circuit, name=None, device="cpu", batch=4096, frozen_fingerprint=None):
        self.model, self.cfg = load_checkpoint(path, device)
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
            x = build_input(self.grid, det[i:i + self.batch], fp).to(self.device)
            preds.append((self.model(x) > 0).cpu().numpy())
        return np.concatenate(preds)[:, None]

    def observe(self, det):
        if self.frozen is None:
            self.buf.add(det)
