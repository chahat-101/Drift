"""Round 9 checks (before launch). Each prints its result and exits 1 on failure.

    python results/exp9/check9.py init       # check 2: param counts; N = baseline logits, NF = N at step 0
    python results/exp9/check9.py byhand     # check 3: targets, qubit sampling positions, stale-step masking
    python results/exp9/check9.py losses DIR OUT [train.py args]   # check 1: first 500 training BCEs of the
                                             # train.py in DIR, written to OUT (compare two runs with `cmp`)
    python results/exp9/check9.py bits TMP   # check 1: options off reproduce exp6 eval bits of s0 (stream 0)
"""
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA = os.path.join(ROOT, "data/d5_wide_x5")
BASE_CK = os.path.join(ROOT, "decoder_adaptive_stale_d5_long_s0.pt")


def fail(msg):
    print("FAIL:", msg)
    sys.exit(1)


def losses(code_dir, out, argv):
    """Run train.main() from code_dir and record the first 500 training BCE values (val chunks excluded)."""
    sys.path.insert(0, code_dir)
    os.chdir(code_dir)
    import torch.nn.functional as F

    import train
    orig, rec = F.binary_cross_entropy_with_logits, []

    def bce(logit, y, *a, **k):
        v = orig(logit, y, *a, **k)
        if logit.requires_grad:  # training steps only (val runs under no_grad)
            rec.append(v.detach())
            if len(rec) == 500:
                np.save(out, torch.stack(rec).cpu().numpy())
                print(f"wrote {out}: first 500 BCEs, last {rec[-1].item():.6f}")
                os._exit(0)
        return v

    F.binary_cross_entropy_with_logits = bce
    sys.argv = ["train.py"] + argv
    train.main()


def init():
    sys.path.insert(0, ROOT)
    from dataset import Dataset
    from decoder import ConvDecoder, DetectorGrid, Fingerprint, qubit_positions
    from drift import DriftStream
    from train import GpuChunks
    torch.backends.cudnn.allow_tf32 = True
    nominal = DriftStream(5, 5, 0.003).nominal()
    grid = DetectorGrid(nominal)
    ids, pos = qubit_positions(grid, nominal)
    ds = Dataset(DATA)
    val = GpuChunks(ds.chunks("val"), grid, "cuda", Fingerprint({"fingerprint": "single"}, grid, "cuda"), False, ids)
    ok = True
    for seed in (0, 1, 2):
        ms = {}
        for name, kw in (("base", {}), ("N", dict(aux_noise=True)), ("NF", dict(aux_noise=True, film=True))):
            torch.manual_seed(seed)
            ms[name] = ConvDecoder(in_ch=2, hidden=48, layers=4, qubit_pos=pos, **kw).cuda().eval()
            if seed == 0:
                n = sum(p.numel() for p in ms[name].parameters())
                print(f"params {name}: {n:,}" + (f" (+{n - n0:,} vs base)" if name != "base" else ""))
                n0 = n if name == "base" else n0
        same = all(torch.equal(ms["base"].state_dict()[k], ms["N"].state_dict()[k]) for k in ms["base"].state_dict())
        same &= all(torch.equal(ms["N"].state_dict()[k], ms["NF"].state_dict()[k]) for k in ms["N"].state_dict())
        with torch.no_grad():
            for i in (0, 7, 123):
                x, _ = val.inputs(i)
                lb = ms["base"](x)
                ln = ms["N"](x)
                ln_aux, noise, ls = ms["N"](x, aux=True)
                lnf_shared = ms["NF"](x, shared_fp=True)
                lnf = ms["NF"](x)
                lnf_aux, noise_f, ls_f = ms["NF"](x, aux=True, shared_fp=True)
                eq = [torch.equal(lb, ln), torch.equal(ln, ln_aux), torch.equal(ln, lnf_shared), torch.equal(ln, lnf),
                      torch.equal(ln, lnf_aux), torch.equal(noise, noise_f), torch.equal(ls, ls_f)]
                print(f"seed {seed} chunk {i}: shared weights {same}; base=N {eq[0]}, N aux on/off {eq[1]}, "
                      f"NF(shared)=N {eq[2]}, NF(per-shot)=N {eq[3]}, NF aux=N {eq[4]}, noise/scale heads NF=N "
                      f"{eq[5] and eq[6]}; noise map {tuple(noise.shape)}")
                ok &= same and all(eq)
    # mixed fingerprints in one batch: per-shot FiLM must equal running each shot's group separately
    torch.manual_seed(0)
    nf = ConvDecoder(in_ch=2, hidden=48, layers=4, aux_noise=True, film=True, qubit_pos=pos).cuda().eval()
    for m in nf.film_out:  # non-zero FiLM for this test only
        torch.nn.init.normal_(m.weight, std=0.1)
    with torch.no_grad():
        xa, xb = val.inputs(3)[0][:64], val.inputs(4)[0][:64]
        mix = nf(torch.cat([xa, xb]))
        sep = torch.cat([nf(xa, shared_fp=True), nf(xb, shared_fp=True)])
        d = float((mix - sep).abs().max())
        print(f"NF with random FiLM: mixed-fingerprint batch vs per-group shared_fp, max |diff| {d:.1e}")
        ok &= d < 1e-4 and float((nf(xa) - ms["NF"](xa)).abs().max()) > 0
    print("ALL OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def byhand():
    sys.path.insert(0, ROOT)
    from dataset import Dataset
    from decoder import ConvDecoder, DetectorGrid, Fingerprint, qubit_positions
    from drift import DriftStream
    from train import GpuChunks
    nominal = DriftStream(5, 5, 0.003).nominal()
    grid = DetectorGrid(nominal)
    ids, pos = qubit_positions(grid, nominal)
    ok = True
    # targets by hand on chunks with and without hot qubits
    ds = Dataset(DATA)
    a = ds.load("val.npz")
    val = GpuChunks(ds.chunks("val"), grid, "cuda", Fingerprint({"fingerprint": "single"}, grid, "cuda"), False, ids)
    coords = nominal.get_final_qubit_coordinates()
    for i in [int(np.flatnonzero(a["hot_qubits"][:, 0] >= 0)[0]), int(np.flatnonzero(a["hot_qubits"][:, 0] < 0)[0])]:
        hot = [int(q) for q in a["hot_qubits"][i] if q >= 0]
        want = np.array([10 * a["hot_p"][i] if q in hot else 0.0 for q in sorted(coords)])
        got = val.tmap[i].cpu().numpy()
        e1 = np.abs(got - want).max()
        e2 = abs(val.tscale[i].item() - np.log(a["scale"][i]))
        print(f"val chunk {i}: hot qubits {hot}, hot_p {a['hot_p'][i]:.4f}, scale {a['scale'][i]:.4f}; target nonzero at "
              f"qubits {[q for q, v in zip(sorted(coords), got) if v > 0]} = {got[got > 0].round(4).tolist()}; "
              f"max |map - hand| {e1:.1e}, |log scale - hand| {e2:.1e}")
        ok &= e1 < 1e-6 and e2 < 1e-6 and len(want) == 49
    # sampling positions: a feature map whose channel 0 at grid index (ix, iy) is 10 ix + iy, sampled bilinearly,
    # must give 10 * (x / 2) + y / 2 at a qubit at (x, y) (detector x, y axes are 0, 2, ..., 10)
    print("detector axes x", grid.axes[0].tolist(), "y", grid.axes[1].tolist())
    m = ConvDecoder(in_ch=2, aux_noise=True, qubit_pos=pos)
    T, X, Y = grid.shape
    f = torch.zeros(2, 48, T, X, Y)
    f[:, 0] = (10 * torch.arange(X)[:, None] + torch.arange(Y)[None, :]).float()
    m.features = lambda x, shared_fp=False: f
    del m.noise_head  # replaced by "take channel 0"
    m.noise_head = lambda q: q[..., :1]
    _, got, _ = m(torch.zeros(2, 2, T, X, Y), aux=True)
    want = np.array([10 * coords[q][0] / 2 + coords[q][1] / 2 for q in sorted(coords)])
    e = np.abs(got[0].detach().numpy() - want).max()
    print(f"qubit sampling: {len(want)} qubits, e.g. qubit {ids[0]} at {coords[ids[0]]} -> {got[0, 0]:.3f} "
          f"(hand {want[0]:.3f}), qubit {ids[-1]} at {coords[ids[-1]]} -> {got[0, -1]:.3f} (hand {want[-1]:.3f}); "
          f"max |diff| {e:.1e}")
    ok &= e < 1e-4
    # stale masking: replay train.py's stale_rng draws by hand; aux must be applied exactly on the other steps
    import train
    rec, orig = [], train.gpu_loss

    def wrapped(model, data, i, fp=None, aux=False, shared_fp=False):
        out = orig(model, data, i, fp, aux, shared_fp)
        if len(data) == 150000:
            rec.append((fp is not None, aux, isinstance(out, tuple),
                        None if not isinstance(out, tuple) else float(out[0] - out[1] - out[2])))
            if len(rec) == 400:
                raise KeyboardInterrupt
        return out

    train.gpu_loss = wrapped
    sys.argv = ["train.py", "--data", DATA, "--adaptive", "--stale-prob", "0.3", "--epochs", "2", "--val-every",
                "100000", "--device", "cuda", "--aux-noise", "--seed", "1", "--out", "/tmp/unused_check9.pt"]
    try:
        train.main()
    except KeyboardInterrupt:
        pass
    r = np.random.default_rng([1, 1])
    exp = []
    for _ in range(400):
        st = r.random() < 0.3
        if st:
            r.integers(150000)
            if not r.random() < 0.5:
                r.uniform()
        exp.append(st)
    stale_seen = [s for s, _, _, _ in rec]
    aux_on = [a and t for _, a, t, _ in rec]
    resid = max(abs(v) for *_, v in rec if v is not None)
    print(f"stale masking: {sum(exp)} of 400 steps stale by hand; train.py stale steps match: {stale_seen == exp}; "
          f"aux applied exactly on the non-stale steps: {aux_on == [not s for s in exp]}; "
          f"|loss - bce - aux| max {resid:.1e}")
    ok &= stale_seen == exp and aux_on == [not s for s in exp] and resid < 1e-6
    print("ALL OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def bits(tmp):
    """ensemble_eval.py (new code) on the seed-0 baseline alone vs its member bits in exp6 (member m0)."""
    os.chdir(ROOT)
    subprocess.run([sys.executable, "ensemble_eval.py", "--checkpoints", BASE_CK, "--data", "data/d5_wide",
                    "--stream-seed", "0", "--tests", "--batch", "16384", "--out", tmp, "--name", "base_s0", "--force"],
                   check=True)
    new = np.load(os.path.join(tmp, "base_s0_errors.npz"))
    old = np.load("results/exp6/d5/st0/ens_adaptive_stale_d5_long_errors.npz")
    ok, n = True, 0
    for k in new.files:
        if not k.endswith("|m0"):
            continue
        same = np.array_equal(new[k], old[k])
        n += np.unpackbits(new[k]).size
        print(f"{k:<28} identical: {same}")
        ok &= same
    print(f"{n} decisions compared; " + ("ALL OK" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "losses":
        losses(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif cmd == "init":
        init()
    elif cmd == "byhand":
        byhand()
    elif cmd == "bits":
        bits(sys.argv[2])
