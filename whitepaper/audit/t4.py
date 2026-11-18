"""Section 5.5: NF top-k hot-qubit hit rates from results/exp9/noise/noise_auxfilm_s*.npz."""
from lib import *
for s in range(3):
    z=np.load(f"{R}/results/exp9/noise/noise_auxfilm_s{s}.npz")
    out=[]
    for r in ("bursts","bursts_big","hot_qubit"):
        for ph in ("drift","onset","later"):
            hits=[]
            for st in range(5):
                m=masks_from_csv(csvpath(5,r,st))[ph]
                pred=z[f"{r}|st{st}|pred"]; t=z[f"{r}|st{st}|tmap"]>0
                for i in np.where(m)[0]:
                    k=t[i].sum()
                    if k==0: continue
                    top=np.argsort(-pred[i],kind="stable")[:k]
                    hits.append(t[i][top].mean())
            out.append(f"{r[:6]}/{ph}:{np.mean(hits):.3f}(n={len(hits)})")
    print(f"NF s{s}: "+"  ".join(out))
