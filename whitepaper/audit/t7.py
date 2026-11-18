"""Section 6.4: h64/l6 vs H_c=48 seed 0 (pooled and stream 0, nominal test set)."""
from lib import *
def sb(name,r,st,ph,kind="live"):
    z=np.load(f"{X}/results/exp4/full/st{st}/decoder_{name}_errors.npz")
    b=np.unpackbits(z[f"{r}|{kind}"],axis=-1).astype(bool)
    return b[masks_from_csv(csvpath(5,r,st))[ph]].ravel()
def pooled(name,r,ph): return np.concatenate([sb(name,r,st,ph) for st in range(5)])
h=pooled("adaptive_stale_h64l6_d5_long_s0","bursts_big","drift"); s=pooled("adaptive_stale_d5_long_s0","bursts_big","drift")
A,B,dd,nb,nc,lp=paired(h,s); print(f"h64 vs h48 s0 bb drift pooled: {A:.4f} vs {B:.4f} d={dd:+.5f} p={pstr(lp)}")
print(f"stream0: h64 {sb('adaptive_stale_h64l6_d5_long_s0','bursts_big',0,'drift').mean():.4f} h48 {sb('adaptive_stale_d5_long_s0','bursts_big',0,'drift').mean():.4f}")
z1=np.load(f"{X}/results/exp4/full/st0/decoder_adaptive_stale_h64l6_d5_long_s0_errors.npz"); z2=np.load(f"{X}/results/exp4/full/st0/decoder_adaptive_stale_d5_long_s0_errors.npz")
a=np.unpackbits(z1["test|nominal"]).astype(bool)[:200000] if z1["test|nominal"].dtype==np.uint8 and len(z1["test|nominal"])<200000 else z1["test|nominal"].astype(bool)
b=z2["test|nominal"].astype(bool)
A,B,dd,nb,nc,lp=paired(a,b); print(f"nominal test h64 {A:.5f} h48 {B:.5f} d={dd:+.5f} p={pstr(lp)}")
# member check: ens m0 equals exp4 single s0?
e=bits(5,"bursts_big","ast","drift",member="m0"); print("m0 vs exp4 s0 identical:", np.array_equal(e,s))
# learning curves: 60k step? 
