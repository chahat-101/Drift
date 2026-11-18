"""Section 5.2: frozen ensembles vs static-wide; nominal-step advantage over MWPM; event-level sign tests (d=5)."""
from lib import *
import csv
def P(label,a,b):
    A,B,dd,nb,nc,lp = paired(a,b); print(f"  {label}: A={A:.4f} B={B:.4f} d={dd:+.4f} p={pstr(lp)}")
print("frozen vs static-wide ensembles d5/d3")
for d in (5,3):
    for k in ("aw","ast"):
        for r in REG: P(f"d{d} {k} frozen - sw ens {r}", bits(d,r,k,"drift",kind="frozen"), bits(d,r,"sw","drift"))
print("d3 aw frozen vs sw per seed, bursts drift")
v=[]
for i in range(3):
    a=bits(3,"bursts","aw","drift",kind="frozen",member=f"m{i}"); b=bits(3,"bursts","sw","drift",member=f"m{i}")
    A,B,dd,nb,nc,lp=paired(a,b); v.append(A-B); print(f"  seed{i} {dd:+.4f} p={pstr(lp)}")
print(f"  mean {np.mean(v):+.4f}")
P("static vs oracle telegraph d5", bits(5,"telegraph","static","drift"), bits(5,"telegraph","oracle","drift"))
print("nominal-step relative advantage of ast over static MWPM, d5")
for r in REG:
    a=bits(5,r,"ast","nominal").mean(); s=bits(5,r,"static","nominal").mean(); print(f"  {r}: {a:.4f} vs {s:.4f} rel {(a-s)/s*100:+.0f}%;  drift ast vs oracle rel {(bits(5,r,'ast','drift').mean()/bits(5,r,'oracle','drift').mean()-1)*100:+.0f}%")
# event-level analysis: bursts_big drift, per event (run of consecutive drift steps), sign of ast-vs-X error count diff
def events(d,r,A,B,kindA="live",kindB="live"):
    w=l=t=0
    for st in range(5):
        m=masks_from_csv(csvpath(d,r,st)); dr=m["drift"]
        def stepbits(who,kind):
            if who in ENS[d]:
                p=f"{R}/results/"+ENS[d][who].format(st=st)+"_errors.npz"; a=np.load(p)[f"{r}|{kind}|ens"]
            else: a=np.load(basefile(d,r,st))[who]
            return np.unpackbits(a,axis=-1).astype(bool).sum(1)
        ea,eb=stepbits(A,kindA),stepbits(B,kindB)
        i=0
        while i<len(dr):
            if dr[i]:
                j=i
                while j<len(dr) and dr[j]: j+=1
                dd=ea[i:j].sum()-eb[i:j].sum(); w+=dd<0; l+=dd>0; t+=dd==0; i=j
            else: i+=1
    n=w+l; k=min(w,l); from math import comb
    p=min(1,2*sum(comb(n,x) for x in range(k+1))/2**n)
    return w,l,t,p
for d in (5,):
  for r in ("bursts","bursts_big","telegraph","hot_qubit"):
    for (A,B,ka,kb) in (("ast","oracle","live","live"),("ast","sw","live","live"),("ast","ast","live","frozen"),("ast","static","live","live")):
        w,l,t,p=events(d,r,A,B,ka,kb); print(f"  d{d} {r} {A}({ka}) vs {B}({kb}) drift runs: A better {w}, worse {l}, tie {t}, sign p={p:.2g}")
