"""Section 5.1-5.3 text: d=3 onset/later vs oracle, after-step costs, d=5 phases vs oracle; Section 6.5 ensemble vs members."""
from lib import *
def P(label,a,b):
    A,B,dd,nb,nc,lp = paired(a,b); print(f"  {label}: A={A:.5f} B={B:.5f} d={dd:+.5f} b={nb} c={nc} p={pstr(lp)}")
print("5.1 d3 onset/later ast vs oracle")
for r in ("bursts","bursts_big"):
    for ph in ("onset","later"): P(f"d3 {r} {ph}", bits(3,r,"ast",ph), bits(3,r,"oracle",ph))
print("5.2 after step")
P("d5 bb after aw live-frozen", bits(5,"bursts_big","aw","after"), bits(5,"bursts_big","aw","after",kind="frozen"))
P("d5 bb after aw - static", bits(5,"bursts_big","aw","after"), bits(5,"bursts_big","static","after"))
P("d5 bb after ast live-frozen", bits(5,"bursts_big","ast","after"), bits(5,"bursts_big","ast","after",kind="frozen"))
P("d5 bb after ast - static", bits(5,"bursts_big","ast","after"), bits(5,"bursts_big","static","after"))
P("d3 bb after ast - static", bits(3,"bursts_big","ast","after"), bits(3,"bursts_big","static","after"))
P("d3 b after ast - static", bits(3,"bursts","ast","after"), bits(3,"bursts","static","after"))
P("d3 bb after ast live-frozen", bits(3,"bursts_big","ast","after"), bits(3,"bursts_big","ast","after",kind="frozen"))
print("5.3 d5 phases ast vs oracle")
for r in ("bursts","bursts_big"):
    for ph in ("onset","later"): P(f"d5 {r} {ph}", bits(5,r,"ast",ph), bits(5,r,"oracle",ph))
print("6 ensembles vs members (drift), largest p")
for d in (5,3):
  for k in ("sw","aw","ast"):
    worst=-1e9
    for r in REG:
        e=bits(d,r,k,"drift")
        for i in range(3):
            A,B,dd,nb,nc,lp=paired(e,bits(d,r,k,"drift",member=f"m{i}"))
            if dd>=0: print(f"  d{d} {k} {r} m{i}: ens NOT better {dd:+.4f} p={pstr(lp)}")
            worst=max(worst,lp)
    print(f"  d{d} {k}: largest p ens vs member = {pstr(worst)}")
e=bits(3,"hot_qubit","sw","drift"); ms=[bits(3,"hot_qubit","sw","drift",member=f"m{i}") for i in range(3)]
i=int(np.argmin([m.mean() for m in ms])); P("d3 sw hot ens - best member", e, ms[i])
