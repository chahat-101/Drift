"""Test sets: ensembles and variants vs adaptive-stale ens.; static-wide vs static-nominal per seed (Section 5.2)."""
from lib import *
import glob
def tb(d,who,t,member="ens"):
    p=f"{R}/results/"+ENS[d][who].format(st=0)+"_errors.npz"
    z=np.load(p); k=f"test|{t}|{member}"
    if k not in z.files: return None
    return np.unpackbits(z[k]).astype(bool)[:200000]
for d,vs in ((5,("sw","aw","sn","B","N","NF")),(3,("sw","aw","sn","B","C"))):
    for t in ("nominal","scale x2","2 hot qubits"):
        a=tb(d,"ast",t)
        if a is None: print("no test bits ast",d); continue
        s=f"d{d} {t:12s} ast={a.mean():.5f}"
        for v in vs:
            b=tb(d,v,t)
            if b is None: s+=f" {v}:NA"; continue
            A,B,dd,nb,nc,lp=paired(b,a); s+=f" | {v}={b.mean():.5f} d={dd:+.5f} p={pstr(lp)}"
        print(s)
# d5 sw vs sn per seed on nominal
for d in (5,3):
  for i in range(3):
    a=tb(d,"sw","nominal",f"m{i}"); b=tb(d,"sn","nominal",f"m{i}"); c=tb(d,"aw","nominal",f"m{i}")
    if a is None or b is None: continue
    print(f"d{d} seed{i} nominal: sw={a.mean():.5f} sn={b.mean():.5f} aw={c.mean():.5f} sw-sn p={pstr(paired(a,b)[5])} aw-sn p={pstr(paired(c,b)[5])}")
# MWPM nominal test d5
z=np.load(f"{X}/results/exp4/curves/baselines.npz"); print(z.files[:10])
v = [np.mean(bits(5, "bursts_big", "sn", "drift", member=f"m{i}")) for i in range(3)]
print(f"d5 static-nominal single seeds, bursts_big drift: mean {np.mean(v):.4f} (rounded {np.mean(v):.3f}), "
      f"seeds {' '.join(f'{x:.4f}' for x in v)}; static MWPM {bits(5, 'bursts_big', 'static', 'drift').mean():.4f}")
