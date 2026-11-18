"""Section 6 / Table 3: every variant (B, C, N, NF) minus adaptive-stale ens., all regimes and phases; B/C extras."""
from lib import *
def P(label,a,b,rel=False):
    A,B,dd,nb,nc,lp = paired(a,b); s=f"  {label}: A={A:.4f} B={B:.4f} d={dd:+.5f} p={pstr(lp)}"
    if rel: s+=f" rel={dd/B*100:+.1f}%"
    print(s)
def rule_scan(d,v):
    print(f"== d{d} {v} - ast, all regime×phase (bold <1e-3)")
    for r in REG:
        for ph in PH:
            for kind in ("live",):
                a=bits(d,r,v,ph); b=bits(d,r,"ast",ph)
                if not len(a): continue
                A,B,dd,nb,nc,lp=paired(a,b)
                flag="**" if lp< -3 else ("w" if lp< np.log10(0.05) else "")
                print(f"   {r:10s} {ph:7s} {dd:+.4f} ({dd/B*100:+.1f}%) p={pstr(lp)} {flag}")
for d,vs in ((3,("B","C")),(5,("B","N","NF"))):
    for v in vs: rule_scan(d,v)
print("== extra")
for r in ("bursts","bursts_big"):
    P(f"d3 B after {r} - static", bits(3,r,"B","after"), bits(3,r,"static","after"))
P("d3 C - B bb drift", bits(3,"bursts_big","C","drift"), bits(3,"bursts_big","B","drift"))
for v in ("B","C"):
    print(f"  d3 {v} frozen bb drift {bits(3,'bursts_big',v,'drift',kind='frozen').mean():.4f}")
P("d5 B ens - oracle bursts onset", bits(5,"bursts","B","onset"), bits(5,"bursts","oracle","onset"))
P("d5 ast ens - oracle bursts onset", bits(5,"bursts","ast","onset"), bits(5,"bursts","oracle","onset"))
for r in REG:
    print(f"  d5 nominal steps {r}: B frozen {bits(5,r,'B','nominal',kind='frozen').mean():.5f} ast frozen {bits(5,r,'ast','nominal',kind='frozen').mean():.5f}")
