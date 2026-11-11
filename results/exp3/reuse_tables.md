| decoder | nominal | scale x2 | 2 hot qubits |
|---|---|---|---|
| static MWPM | 0.00648 | 0.02433 | 0.05803 |
| static-nominal (fresh) | 0.00536 ± 0.00009 | 0.02150 ± 0.00027 | 0.04195 ± 0.00267 |
| static-nominal (fixed) | 0.00544 ± 0.00002 | 0.02151 ± 0.00010 | 0.03990 ± 0.00210 |
| static-wide | 0.00605 ± 0.00003 | 0.02390 ± 0.00011 | 0.02661 ± 0.00078 |
| adaptive-wide | 0.00532 ± 0.00001 | 0.02127 ± 0.00013 | 0.01985 ± 0.00016 |

| a vs b | nominal | scale x2 | 2 hot qubits |
|---|---|---|---|
| static-nominal (fixed) vs static-nominal (fresh) | +0.0001 p=0.9 (1/3) tie | +0.0000 p=0.5 (1/3) tie | **-0.0020** p=6e-08 (3/3) |
| adaptive-wide vs static-nominal (fixed) | -0.0001 p=0.2 (3/3) tie | -0.0002 p=0.2 (3/3) tie | **-0.0200** p<1e-300 (3/3) |
| static-wide vs static-nominal (fixed) | **+0.0006** p=7e-08 (0/3) | **+0.0024** p=1e-25 (0/3) | **-0.0133** p=1e-305 (3/3) |
| adaptive-wide vs static-nominal (fresh) | -0.0000 p=0.4 (2/3) tie | -0.0002 p=0.4 (3/3) tie | **-0.0221** p<1e-300 (3/3) |

Nominal stream steps (pooled over 5 streams; LER mean over model seeds):

| regime | steps | static-nominal (fresh) | static-nominal (fixed) | static-wide | adaptive-wide | static-nominal (fixed) vs static-nominal (fresh) | adaptive-wide vs static-nominal (fixed) | static-wide vs static-nominal (fixed) | adaptive-wide vs static-nominal (fresh) |
|---|---|---|---|---|---|---|---|---|---|
| bursts | 618 | 0.00557 | 0.00558 | 0.00622 | 0.00590 | +0.0000 p=1 (1/3) tie | **+0.0003** p=5e-20 (0/3) | **+0.0006** p=4e-48 (0/3) | **+0.0003** p=5e-21 (0/3) |
| bursts_big | 634 | 0.00549 | 0.00547 | 0.00614 | 0.00630 | -0.0000 p=0.7 (3/3) tie | **+0.0008** p=2e-98 (0/3) | **+0.0007** p=8e-52 (0/3) | **+0.0008** p=2e-96 (0/3) |
| hot_qubit | 250 | 0.00556 | 0.00555 | 0.00620 | 0.00555 | -0.0000 p=0.9 (2/3) tie | +0.0000 p=0.7 (1/3) tie | **+0.0007** p=2e-22 (0/3) | -0.0000 p=0.9 (2/3) tie |
| telegraph | 439 | 0.00549 | 0.00550 | 0.00607 | 0.00547 | +0.0000 p=1 (2/3) tie | -0.0000 p=0.7 (3/3) tie | **+0.0006** p=3e-29 (0/3) | -0.0000 p=0.8 (3/3) tie |
| all 4 regimes | 1941 | 0.00552 | 0.00552 | 0.00616 | 0.00589 | -0.0000 p=0.5 (2/3) tie | **+0.0004** p=3e-74 (0/3) | **+0.0006** p=2e-148 (0/3) | **+0.0004** p=2e-85 (0/3) |
