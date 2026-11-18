"""Event-level sign tests for every Table 1 (adaptive-stale ens. vs static-wide ens. / oracle / static / best WindowDEM)
and Table 2 (live vs frozen) cell. An event is a run of consecutive drift steps; its value is the error-count
difference over the run; two-sided exact sign test over runs with nonzero difference."""
from math import comb
from lib import *

def stepcounts(d, r, st, who, kind):
    if who in ENS[d]:
        a = np.load(f"{R}/results/" + ENS[d][who].format(st=st) + "_errors.npz")[f"{r}|{kind}|ens"]
    else:
        a = np.load(basefile(d, r, st))[who]
    return np.unpackbits(a, axis=-1).astype(bool).sum(1)

def events(d, r, A, B, ka="live", kb="live"):
    w = l = t = 0
    for st in range(5):
        dr = masks_from_csv(csvpath(d, r, st))["drift"]
        ea, eb = stepcounts(d, r, st, A, ka), stepcounts(d, r, st, B, kb)
        i = 0
        while i < len(dr):
            if dr[i]:
                j = i
                while j < len(dr) and dr[j]: j += 1
                x = int(ea[i:j].sum() - eb[i:j].sum()); w += x < 0; l += x > 0; t += x == 0; i = j
            else:
                i += 1
    n, k = w + l, min(w, l)
    return w, l, t, min(1.0, 2 * sum(comb(n, x) for x in range(k + 1)) / 2 ** n)

if __name__ == "__main__":
    for d in (5, 3):
        for r in REG:
            bw, _ = best_window(d, r, "drift")
            for B, kb, lab in (("sw", "live", "vs static-wide ens"), ("oracle", "live", "vs oracle"),
                               ("static", "live", "vs static MWPM"), (f"window_{bw}", "live", f"vs best W({bw})"),
                               ("ast", "frozen", "live-frozen (ast)")):
                w, l, t, p = events(d, r, "ast", B, "live", kb)
                print(f"d{d} {r:10s} ast {lab:20s}: better {w:2d} worse {l:2d} tie {t} p={p:.2g}")
            w, l, t, p = events(d, r, "aw", "aw", "live", "frozen")
            print(f"d{d} {r:10s} aw  {'live-frozen (aw)':20s}: better {w:2d} worse {l:2d} tie {t} p={p:.2g}")
