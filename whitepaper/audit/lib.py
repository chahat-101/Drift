"""Shared helpers for the whitepaper audit: load pooled per-shot error bits, rebuild phase masks from the
per-step noise-state CSVs, exact McNemar in log space."""
import csv, os
from math import lgamma, log, exp
import numpy as np
# R: repository root. X: folder holding the per-shot error bits extracted from drift_data.tgz (see README.md);
# files under results/ that are not in the repo are looked up as X/results/...
R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
X = os.path.abspath(os.environ.get("DRIFT_BITS", os.path.join(R, "audit_bits")))
REG = ["bursts", "bursts_big", "hot_qubit", "telegraph"]
PH = ["drift", "onset", "later", "after", "nominal"]
W = [500, 1000, 2000, 5000, 10000]

def masks_from_csv(path):
    rows = list(csv.DictReader(open(path)))
    drift = np.array([float(r["scale"]) != 1.0 or int(r["n_hot"]) > 0 for r in rows])
    before = np.r_[False, drift[:-1]]
    onset = drift & ~before & (np.arange(len(rows)) > 0)
    return dict(drift=drift, onset=onset, later=drift & ~onset, after=~drift & before, nominal=~drift)

def csvpath(d, r, st):
    if d == 5: return f"{R}/results/exp3/d5/{r}_d5_r5_u200_st{st}_s0.csv"
    return f"{R}/results/exp2/{r}_d3_r3_u200_st{st}_s0.csv"

def basefile(d, r, st):
    if d == 5: return f"{X}/results/exp3/d5/{r}_d5_r5_u200_st{st}_s0_errors.npz"
    return f"{X}/results/exp2/{r}_d3_r3_u200_st{st}_s0_errors.npz"

ENS = {5: dict(sw="exp6/d5/st{st}/ens_static_wide_d5_long", aw="exp6/d5/st{st}/ens_adaptive_wide_d5_long",
               ast="exp6/d5/st{st}/ens_adaptive_stale_d5_long", sn="exp6/d5/st{st}/ens_static_nominal_d5_long",
               B="exp8/d5/st{st}/ens_adaptive_d5_ms_long", N="exp9/d5/st{st}/ens_adaptive_d5_aux_long",
               NF="exp9/d5/st{st}/ens_adaptive_d5_auxfilm_long"),
       3: dict(sw="exp6/d3/st{st}/ens_static_d3_wide", aw="exp6/d3/st{st}/ens_adaptive_d3_wide",
               ast="exp6/d3/st{st}/ens_adaptive_d3_stale", sn="exp6/d3/st{st}/ens_static_d3_nominal",
               B="exp7/d3/st{st}/ens_adaptive_d3_ms", C="exp7/d3/st{st}/ens_adaptive_d3_mspair")}

_cache = {}
def _load(p):
    if p not in _cache: _cache[p] = np.load(p)
    return _cache[p]

def bits(d, r, who, ph, kind="live", member="ens"):
    """pooled bool error bits over 5 streams for phase ph. who: 'static','oracle','window_N' or ENS key."""
    out = []
    for st in range(5):
        m = masks_from_csv(csvpath(d, r, st))[ph]
        if who in ENS[d]:
            p = f"{R}/results/" + ENS[d][who].format(st=st) + "_errors.npz"
            if not os.path.exists(p): p = f"{X}/results/" + ENS[d][who].format(st=st) + "_errors.npz"
            a = _load(p)[f"{r}|{kind}|{member}"]
        else:
            a = _load(basefile(d, r, st))[who]
        b = np.unpackbits(a, axis=-1).astype(bool)
        out.append(b[m].ravel())
    return np.concatenate(out)

def mcn_log10(b, c):
    n, k = b + c, min(b, c)
    if n == 0: return 0.0
    logs = [lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) - n * log(2) for i in range(k + 1)]
    m = max(logs)
    return min(0.0, (log(2) + m + log(sum(exp(x - m) for x in logs))) / log(10))

def pstr(lp):
    if lp > -300: return f"{10**lp:.2g}"
    e = int(np.floor(lp)); return f"{10**(lp-e):.2g}e{e}"

def paired(a, b):
    nb, nc = int((a & ~b).sum()), int((~a & b).sum())
    lp = mcn_log10(nb, nc)
    return a.mean(), b.mean(), (nb - nc) / len(a), nb, nc, lp

def best_window(d, r, ph):
    v = {w: bits(d, r, f"window_{w}", ph).mean() for w in W}
    w = min(v, key=v.get); return w, v[w]
