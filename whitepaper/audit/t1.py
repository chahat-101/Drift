"""Table 1: drift-step LERs, single-seed mean +- s.d., and adaptive-stale ens. vs static-wide ens. / oracle / static MWPM / best WindowDEM (exact McNemar), d=5 and d=3."""
from lib import *
import sys
for d in (5, 3):
    print(f"===== d={d} Table1 drift")
    for r in REG:
        a = bits(d, r, "ast", "drift")
        st = bits(d, r, "static", "drift"); orc = bits(d, r, "oracle", "drift")
        w, wl = best_window(d, r, "drift"); wb = bits(d, r, f"window_{w}", "drift")
        print(f"{r:11s} n={len(a)} static {st.mean():.4f} bestW {wl:.4f}({w}) oracle {orc.mean():.4f}")
        for k in ("sw", "aw", "ast"):
            e = bits(d, r, k, "drift").mean()
            ms = [bits(d, r, k, "drift", member=f"m{i}").mean() for i in range(3)]
            print(f"   {k:4s} ens {e:.4f} seeds {np.mean(ms):.4f}±{np.std(ms, ddof=1):.4f}  best seed {min(ms):.4f}  ens-mean {e-np.mean(ms):+.4f} ens-best {e-min(ms):+.4f}")
        for name, o in (("static-wide ens", bits(d, r, "sw", "drift")), ("oracle", orc), ("static MWPM", st), (f"bestW", wb)):
            A, B, dd, nb, nc, lp = paired(a, o)
            print(f"   ast - {name}: {dd:+.4f} b={nb} c={nc} p={pstr(lp)} (log10 {lp:.1f})")
        A, B, dd, nb, nc, lp = paired(bits(d, r, "aw", "drift"), a)
        print(f"   aw - ast: {dd:+.4f} p={pstr(lp)}")
