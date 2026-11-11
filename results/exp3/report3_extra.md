## Data-reuse control (d=3)

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

## Headline claims, d=3 vs d=5

| claim / cell | d=3 | d=5 |
|---|---|---|
| 1. live vs frozen (adaptive-wide), drift, bursts | **-0.0587** p<1e-300 (3/3) | **-0.0107** p=2e-263 (3/3) |
| 1. live vs frozen (adaptive-wide), drift, bursts_big | **-0.0963** p<1e-300 (3/3) | **-0.0277** p<1e-300 (3/3) |
| 1. live vs frozen (adaptive-wide), drift, hot_qubit | **-0.0193** p<1e-300 (3/3) | **-0.0031** p=2e-291 (3/3) |
| 1b. frozen adaptive-wide vs static-wide, drift, bursts | **+0.0253** p<1e-300 (0/3) | +0.0026 p=0.005 (1/3) |
| 1b. frozen adaptive-wide vs static-wide, drift, bursts_big | **+0.0539** p<1e-300 (0/3) | +0.0062 p=4e-71 (1/3) |
| 1b. frozen adaptive-wide vs static-wide, drift, hot_qubit | **+0.0127** p<1e-300 (0/3) | **+0.0004** p=0.0007 (0/3) |
| 2. adaptive-wide vs static-nominal, nominal test set | -0.0000 p=0.4 (2/3) tie | **-0.0014** p=5e-24 (3/3) |
| 2b. static-wide vs static-nominal, nominal test set (hedging cost) | **+0.0007** p=1e-08 (0/3) | **-0.0012** p=6e-16 (3/3) |
| 2c. adaptive-wide vs static-nominal, scale x2 test set | -0.0002 p=0.4 (3/3) tie | **-0.0107** p=8e-154 (3/3) |
| 2d. adaptive-wide vs static-nominal, nominal steps, bursts | **+0.0003** p=5e-21 (0/3) | **-0.0011** p=1e-83 (3/3) |
| 2d. adaptive-wide vs static-nominal, nominal steps, bursts_big | **+0.0008** p=2e-96 (0/3) | **-0.0010** p=4e-66 (3/3) |
| 2d. adaptive-wide vs static-nominal, nominal steps, hot_qubit | -0.0000 p=0.9 (2/3) tie | **-0.0010** p=7e-20 (3/3) |
| 2d. adaptive-wide vs static-nominal, nominal steps, telegraph | -0.0000 p=0.8 (3/3) tie | **-0.0011** p=2e-64 (3/3) |
| 3. adaptive-wide vs best WindowDEM, all regimes x phases | 19 better, 0 tie, 0 worse; weakest: telegraph nominal (w=10000): **-0.0009** p=3e-56 (3/3) | 12 better, 2 tie, 5 worse; weakest: bursts_big later (w=2000): **+0.0437** p<1e-300 (0/3) |
| 4. adaptive-wide vs oracle MWPM, later, bursts | -0.0006 p=0.8 (3/3) tie | **+0.0109** p=1e-166 (0/3) |
| 4. adaptive-wide vs oracle MWPM, later, bursts_big | +0.0043 p=0.4 (0/3) | **+0.0831** p<1e-300 (0/3) |
| 4. adaptive-wide vs oracle MWPM, later, hot_qubit | **-0.0027** p=1e-114 (3/3) | **+0.0013** p=2e-21 (0/3) |
| 4. adaptive-wide vs oracle MWPM, later, telegraph | **-0.0030** p=2e-110 (3/3) | **-0.0020** p=4e-24 (3/3) |
| 5. live vs frozen (adaptive-wide), telegraph, after | +0.0000 p=1 (0/3) tie | -0.0001 p=1 (2/3) tie |
| 5. live vs frozen (adaptive-wide), telegraph, drift | -0.0001 p=0.6 (3/3) tie | **-0.0003** p=0.002 (3/3) tie |
| 5. live vs frozen (adaptive-wide), telegraph, later | -0.0001 p=0.8 (3/3) tie | **-0.0003** p=0.003 (3/3) tie |
| 5. live vs frozen (adaptive-wide), telegraph, nominal | +0.0000 p=0.9 (1/3) tie | **+0.0001** p=3e-07 (0/3) tie |
| 5. live vs frozen (adaptive-wide), telegraph, onset | -0.0001 p=0.6 (2/3) tie | -0.0002 p=0.8 (2/3) tie |

## Decode latency

| d | decoder | device | batch | per call mean (us) | per shot mean (us) | per shot p99 (us) | per round mean (us) | per round p99 (us) |
|---|---|---|---|---|---|---|---|---|
| 3 | MWPM (static) | cpu | 1 | 4.4 | 4.45 | 8.69 | 1.482 | 2.897 |
| 3 | MWPM (static) | cpu | 64 | 21.9 | 0.34 | 0.52 | 0.114 | 0.175 |
| 3 | MWPM (static) | cpu | 4096 | 998.1 | 0.24 | 0.26 | 0.081 | 0.087 |
| 3 | WindowDEM-500 decode | cpu | 1 | 4.4 | 4.38 | 8.05 | 1.460 | 2.683 |
| 3 | WindowDEM-500 decode | cpu | 64 | 23.4 | 0.36 | 0.58 | 0.122 | 0.194 |
| 3 | WindowDEM-500 decode | cpu | 4096 | 1105.1 | 0.27 | 0.29 | 0.090 | 0.095 |
| 3 | WindowDEM-1000 decode | cpu | 1 | 4.5 | 4.46 | 8.91 | 1.485 | 2.970 |
| 3 | WindowDEM-1000 decode | cpu | 64 | 22.6 | 0.35 | 0.56 | 0.118 | 0.187 |
| 3 | WindowDEM-1000 decode | cpu | 4096 | 1066.2 | 0.26 | 0.28 | 0.087 | 0.092 |
| 3 | WindowDEM-2000 decode | cpu | 1 | 4.4 | 4.39 | 8.39 | 1.463 | 2.795 |
| 3 | WindowDEM-2000 decode | cpu | 64 | 23.7 | 0.37 | 0.60 | 0.123 | 0.199 |
| 3 | WindowDEM-2000 decode | cpu | 4096 | 1121.9 | 0.27 | 0.29 | 0.091 | 0.097 |
| 3 | WindowDEM-5000 decode | cpu | 1 | 4.5 | 4.47 | 8.69 | 1.491 | 2.898 |
| 3 | WindowDEM-5000 decode | cpu | 64 | 23.0 | 0.36 | 0.55 | 0.120 | 0.182 |
| 3 | WindowDEM-5000 decode | cpu | 4096 | 1057.1 | 0.26 | 0.28 | 0.086 | 0.094 |
| 3 | WindowDEM-10000 decode | cpu | 1 | 4.5 | 4.49 | 9.13 | 1.498 | 3.045 |
| 3 | WindowDEM-10000 decode | cpu | 64 | 23.3 | 0.36 | 0.54 | 0.121 | 0.181 |
| 3 | WindowDEM-10000 decode | cpu | 4096 | 1076.1 | 0.26 | 0.28 | 0.088 | 0.094 |
| 3 | static-wide | cpu | 1 | 1026.4 | 1026.40 | 1221.10 | 342.134 | 407.032 |
| 3 | static-wide | cpu | 64 | 4142.8 | 64.73 | 74.01 | 21.577 | 24.670 |
| 3 | static-wide | cpu | 4096 | 349147.2 | 85.24 | 87.86 | 28.414 | 29.286 |
| 3 | static-wide | cuda | 1 | 789.1 | 789.07 | 950.29 | 263.022 | 316.764 |
| 3 | static-wide | cuda | 64 | 808.7 | 12.64 | 14.69 | 4.212 | 4.895 |
| 3 | static-wide | cuda | 4096 | 12232.9 | 2.99 | 3.08 | 0.996 | 1.027 |
| 3 | adaptive-wide | cpu | 1 | 1257.4 | 1257.37 | 1500.48 | 419.125 | 500.160 |
| 3 | adaptive-wide | cpu | 64 | 4398.0 | 68.72 | 95.58 | 22.906 | 31.861 |
| 3 | adaptive-wide | cpu | 4096 | 346771.3 | 84.66 | 87.27 | 28.220 | 29.089 |
| 3 | adaptive-wide | cuda | 1 | 976.3 | 976.28 | 1041.11 | 325.426 | 347.036 |
| 3 | adaptive-wide | cuda | 64 | 1074.1 | 16.78 | 20.50 | 5.594 | 6.832 |
| 3 | adaptive-wide | cuda | 4096 | 12888.6 | 3.15 | 3.26 | 1.049 | 1.088 |
| 5 | MWPM (static) | cpu | 1 | 6.3 | 6.27 | 15.23 | 1.255 | 3.046 |
| 5 | MWPM (static) | cpu | 64 | 134.1 | 2.09 | 2.71 | 0.419 | 0.543 |
| 5 | MWPM (static) | cpu | 4096 | 7588.2 | 1.85 | 1.92 | 0.371 | 0.384 |
| 5 | WindowDEM-500 decode | cpu | 1 | 7.1 | 7.13 | 18.38 | 1.425 | 3.677 |
| 5 | WindowDEM-500 decode | cpu | 64 | 204.7 | 3.20 | 4.21 | 0.640 | 0.843 |
| 5 | WindowDEM-500 decode | cpu | 4096 | 11836.5 | 2.89 | 3.00 | 0.578 | 0.600 |
| 5 | WindowDEM-1000 decode | cpu | 1 | 6.7 | 6.68 | 16.61 | 1.337 | 3.322 |
| 5 | WindowDEM-1000 decode | cpu | 64 | 175.3 | 2.74 | 3.56 | 0.548 | 0.712 |
| 5 | WindowDEM-1000 decode | cpu | 4096 | 10031.4 | 2.45 | 2.54 | 0.490 | 0.509 |
| 5 | WindowDEM-2000 decode | cpu | 1 | 6.7 | 6.65 | 17.45 | 1.330 | 3.491 |
| 5 | WindowDEM-2000 decode | cpu | 64 | 164.4 | 2.57 | 3.35 | 0.514 | 0.671 |
| 5 | WindowDEM-2000 decode | cpu | 4096 | 9451.1 | 2.31 | 2.38 | 0.461 | 0.476 |
| 5 | WindowDEM-5000 decode | cpu | 1 | 6.5 | 6.54 | 15.72 | 1.308 | 3.144 |
| 5 | WindowDEM-5000 decode | cpu | 64 | 149.4 | 2.33 | 2.95 | 0.467 | 0.590 |
| 5 | WindowDEM-5000 decode | cpu | 4096 | 8565.5 | 2.09 | 2.17 | 0.418 | 0.434 |
| 5 | WindowDEM-10000 decode | cpu | 1 | 6.5 | 6.52 | 16.45 | 1.304 | 3.289 |
| 5 | WindowDEM-10000 decode | cpu | 64 | 145.5 | 2.27 | 2.93 | 0.455 | 0.587 |
| 5 | WindowDEM-10000 decode | cpu | 4096 | 8278.2 | 2.02 | 2.09 | 0.404 | 0.419 |
| 5 | static-wide | cpu | 1 | 1823.2 | 1823.19 | 1931.70 | 364.638 | 386.340 |
| 5 | static-wide | cpu | 64 | 11376.4 | 177.76 | 218.51 | 35.551 | 43.703 |
| 5 | static-wide | cpu | 4096 | 1199731.2 | 292.90 | 296.01 | 58.581 | 59.201 |
| 5 | static-wide | cuda | 1 | 777.3 | 777.32 | 871.87 | 155.464 | 174.373 |
| 5 | static-wide | cuda | 64 | 803.6 | 12.56 | 13.34 | 2.511 | 2.667 |
| 5 | static-wide | cuda | 4096 | 41630.2 | 10.16 | 10.58 | 2.033 | 2.117 |
| 5 | adaptive-wide | cpu | 1 | 2203.1 | 2203.06 | 2630.07 | 440.611 | 526.014 |
| 5 | adaptive-wide | cpu | 64 | 11893.7 | 185.84 | 193.80 | 37.168 | 38.759 |
| 5 | adaptive-wide | cpu | 4096 | 1188842.3 | 290.24 | 292.72 | 58.049 | 58.544 |
| 5 | adaptive-wide | cuda | 1 | 1224.9 | 1224.89 | 1336.34 | 244.978 | 267.269 |
| 5 | adaptive-wide | cuda | 64 | 1287.7 | 20.12 | 21.28 | 4.024 | 4.255 |
| 5 | adaptive-wide | cuda | 4096 | 44058.3 | 10.76 | 11.02 | 2.151 | 2.205 |

| d | WindowDEM history | reweight per update, mean (ms) | p99 (ms) | amortized per shot at update every 200 (us) |
|---|---|---|---|---|
| 3 | 500 | 0.56 | 0.57 | 2.79 |
| 3 | 1000 | 0.61 | 0.63 | 3.03 |
| 3 | 2000 | 0.72 | 0.79 | 3.60 |
| 3 | 5000 | 1.08 | 1.13 | 5.38 |
| 3 | 10000 | 1.60 | 1.67 | 8.01 |
| 5 | 500 | 2.37 | 2.41 | 11.86 |
| 5 | 1000 | 2.68 | 2.87 | 13.38 |
| 5 | 2000 | 3.34 | 3.38 | 16.72 |
| 5 | 5000 | 5.34 | 5.45 | 26.71 |
| 5 | 10000 | 8.58 | 8.68 | 42.92 |
