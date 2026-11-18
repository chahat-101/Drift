"""Table 2 (live - frozen, drift) and the per-phase appendix tables (Tables 6-8), recomputed cell by cell."""
from lib import *
def P(a,b,label):
    A,B,dd,nb,nc,lp = paired(a,b); print(f"  {label}: A={A:.4f} B={B:.4f} {dd:+.4f} b={nb} c={nc} p={pstr(lp)}")
print("== Table2 live-frozen (drift)")
for d in (5,3):
    for k in ("aw","ast"):
        for r in REG:
            P(bits(d,r,k,"drift"), bits(d,r,k,"drift",kind="frozen"), f"d{d} {k} {r}")
        # single seed means live-frozen
        for r in ("bursts_big",):
            v=[bits(d,r,k,"drift",member=f"m{i}").mean()-bits(d,r,k,"drift",kind="frozen",member=f"m{i}").mean() for i in range(3)]
            print(f"  d{d} {k} {r} single-seed live-frozen mean {np.mean(v):+.4f}")
print("== phase tables")
for d in (5,3):
    for r in REG:
        print(f"-- d{d} {r}")
        for who,kind in (("static","live"),("oracle","live"),("sw","live"),("aw","live"),("ast","live"),("ast","frozen"),("B","live"),("C","live"),("N","live"),("NF","live")):
            if who not in ENS[d] and who not in ("static","oracle"): continue
            row=[]
            for ph in PH:
                b=bits(d,r,who,ph,kind=kind)
                row.append(f"{b.mean():.4f}" if len(b) else "--")
            print(f"   {who}{'(fr)' if kind=='frozen' else ''}: "+" ".join(row))
        row=[]
        for ph in PH:
            if not len(bits(d,r,"static",ph)): row.append("--"); continue
            w,v=best_window(d,r,ph); row.append(f"{v:.4f}({w})")
        print("   bestW: "+" ".join(row))
