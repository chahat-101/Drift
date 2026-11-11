## Table 1: test sets

| decoder | nominal | scale x2 | 2 hot qubits |
|---|---|---|---|
| static MWPM | 0.00648 | 0.02433 | 0.05803 |
| oracle MWPM | 0.00648 | 0.02433 | 0.02269 |
| static-wide | 0.00605 ± 0.00003 | 0.02390 ± 0.00011 | 0.02661 ± 0.00078 |
| adaptive-wide | 0.00532 ± 0.00001 | 0.02127 ± 0.00013 | 0.01985 ± 0.00016 |
| adaptive-stale | 0.00554 ± 0.00003 | 0.02164 ± 0.00014 | 0.02106 ± 0.00049 |

| a vs b | nominal | scale x2 | 2 hot qubits |
|---|---|---|---|
| adaptive-stale vs adaptive-wide | +0.0002 tie | +0.0004 tie | +0.0012 b better |
| adaptive-stale vs static-wide | -0.0005 a better | -0.0023 a better | -0.0056 a better |
| adaptive-stale vs oracle MWPM | -0.0009 a better | -0.0027 a better | -0.0016 a better |
| adaptive-wide vs oracle MWPM | -0.0012 a better | -0.0031 a better | -0.0028 a better |

## Table 2: hot_qubit

Shots per step 2000; drift 100 steps, onset 1 steps, later 99 steps, after 0 steps.

| decoder | update | LER | LER (drift) | onset | later | after |
|---|---|---|---|---|---|---|
| static MWPM | 2000 | 0.04106 | 0.05824 | 0.05800 | 0.05824 | – |
| static MWPM | 200 | 0.04106 | 0.05824 | 0.05800 | 0.05824 | – |
| WindowDEM | 2000 | 0.01872 | 0.02474 | 0.06100 | 0.02438 | – |
| WindowDEM | 200 | 0.01849 | 0.02442 | 0.04000 | 0.02426 | – |
| oracle MWPM | 2000 | 0.01714 | 0.02236 | 0.02250 | 0.02236 | – |
| oracle MWPM | 200 | 0.01714 | 0.02236 | 0.02250 | 0.02236 | – |
| static-wide | 2000 | 0.01987 ± 0.00049 | 0.02665 ± 0.00077 | 0.02750 ± 0.00100 | 0.02664 ± 0.00077 | – |
| static-wide | 200 | 0.01987 ± 0.00049 | 0.02665 ± 0.00077 | 0.02750 ± 0.00100 | 0.02664 ± 0.00077 | – |
| adaptive-wide | 2000 | 0.01527 ± 0.00016 | 0.02010 ± 0.00025 | 0.04367 ± 0.00208 | 0.01986 ± 0.00023 | – |
| adaptive-wide | 200 | 0.01517 ± 0.00016 | 0.01995 ± 0.00024 | 0.02633 ± 0.00058 | 0.01989 ± 0.00023 | – |
| adaptive-stale | 2000 | 0.01601 ± 0.00029 | 0.02113 ± 0.00047 | 0.03300 ± 0.00132 | 0.02101 ± 0.00046 | – |
| adaptive-stale | 200 | 0.01598 ± 0.00028 | 0.02107 ± 0.00046 | 0.02467 ± 0.00115 | 0.02103 ± 0.00046 | – |

### Comparisons: hot_qubit

| a vs b | update | drift | onset | later | after |
|---|---|---|---|---|---|
| adaptive-stale vs static-wide | 2000 | -0.0055 a better | +0.0055 tie | -0.0056 a better | – |
| adaptive-stale vs static-wide | 200 | -0.0056 a better | -0.0028 tie | -0.0056 a better | – |
| adaptive-stale vs adaptive-wide | 2000 | +0.0010 b better | -0.0107 tie | +0.0011 b better | – |
| adaptive-stale vs adaptive-wide | 200 | +0.0011 b better | -0.0017 tie | +0.0011 b better | – |
| adaptive-wide vs static-wide | 2000 | -0.0066 a better | +0.0162 b better | -0.0068 a better | – |
| adaptive-wide vs static-wide | 200 | -0.0067 a better | -0.0012 tie | -0.0068 a better | – |
| adaptive-stale vs oracle MWPM | 2000 | -0.0012 a better | +0.0105 b better | -0.0014 a better | – |
| adaptive-stale vs oracle MWPM | 200 | -0.0013 a better | +0.0022 tie | -0.0013 a better | – |
| adaptive-stale vs static MWPM | 2000 | -0.0371 a better | -0.0250 a better | -0.0372 a better | – |
| adaptive-stale vs static MWPM | 200 | -0.0372 a better | -0.0333 a better | -0.0372 a better | – |
| adaptive-stale vs WindowDEM | 2000 | -0.0036 a better | -0.0280 a better | -0.0034 a better | – |
| adaptive-stale vs WindowDEM | 200 | -0.0033 a better | -0.0153 a better | -0.0032 a better | – |

## Table 2: bursts

Shots per step 2000; drift 28 steps, onset 7 steps, later 21 steps, after 7 steps.

| decoder | update | LER | LER (drift) | onset | later | after |
|---|---|---|---|---|---|---|
| static MWPM | 2000 | 0.03487 | 0.15741 | 0.16043 | 0.15640 | 0.00686 |
| static MWPM | 200 | 0.03487 | 0.15741 | 0.16043 | 0.15640 | 0.00686 |
| WindowDEM | 2000 | 0.03012 | 0.10704 | 0.14614 | 0.09400 | 0.03643 |
| WindowDEM | 200 | 0.02853 | 0.10038 | 0.13671 | 0.08826 | 0.03629 |
| oracle MWPM | 2000 | 0.01649 | 0.05898 | 0.06150 | 0.05814 | 0.00686 |
| oracle MWPM | 200 | 0.01649 | 0.05898 | 0.06150 | 0.05814 | 0.00686 |
| static-wide | 2000 | 0.02174 ± 0.00017 | 0.08968 ± 0.00089 | 0.08990 ± 0.00089 | 0.08961 ± 0.00091 | 0.00648 ± 0.00011 |
| static-wide | 200 | 0.02174 ± 0.00017 | 0.08968 ± 0.00089 | 0.08990 ± 0.00089 | 0.08961 ± 0.00091 | 0.00648 ± 0.00011 |
| adaptive-wide | 2000 | 0.01806 ± 0.00023 | 0.06813 ± 0.00120 | 0.11450 ± 0.00223 | 0.05267 ± 0.00099 | 0.02517 ± 0.00048 |
| adaptive-wide | 200 | 0.01616 ± 0.00022 | 0.06121 ± 0.00096 | 0.08683 ± 0.00113 | 0.05267 ± 0.00102 | 0.01267 ± 0.00016 |
| adaptive-stale | 2000 | 0.01709 ± 0.00005 | 0.06614 ± 0.00028 | 0.09750 ± 0.00217 | 0.05569 ± 0.00056 | 0.00914 ± 0.00052 |
| adaptive-stale | 200 | 0.01615 ± 0.00004 | 0.06163 ± 0.00022 | 0.07914 ± 0.00119 | 0.05579 ± 0.00041 | 0.00729 ± 0.00019 |

### Comparisons: bursts

| a vs b | update | drift | onset | later | after |
|---|---|---|---|---|---|
| adaptive-stale vs static-wide | 2000 | -0.0235 a better | +0.0076 b better | -0.0339 a better | +0.0027 b better |
| adaptive-stale vs static-wide | 200 | -0.0281 a better | -0.0108 a better | -0.0338 a better | +0.0008 tie |
| adaptive-stale vs adaptive-wide | 2000 | -0.0020 tie | -0.0170 a better | +0.0030 tie | -0.0160 a better |
| adaptive-stale vs adaptive-wide | 200 | +0.0004 tie | -0.0077 a better | +0.0031 b better | -0.0054 a better |
| adaptive-wide vs static-wide | 2000 | -0.0216 a better | +0.0246 b better | -0.0369 a better | +0.0187 b better |
| adaptive-wide vs static-wide | 200 | -0.0285 a better | -0.0031 tie | -0.0369 a better | +0.0062 b better |
| adaptive-stale vs oracle MWPM | 2000 | +0.0072 b better | +0.0360 b better | -0.0025 tie | +0.0023 b better |
| adaptive-stale vs oracle MWPM | 200 | +0.0026 tie | +0.0176 b better | -0.0023 tie | +0.0004 tie |
| adaptive-stale vs static MWPM | 2000 | -0.0913 a better | -0.0629 a better | -0.1007 a better | +0.0023 b better |
| adaptive-stale vs static MWPM | 200 | -0.0958 a better | -0.0813 a better | -0.1006 a better | +0.0004 tie |
| adaptive-stale vs WindowDEM | 2000 | -0.0409 a better | -0.0486 a better | -0.0383 a better | -0.0273 a better |
| adaptive-stale vs WindowDEM | 200 | -0.0387 a better | -0.0576 a better | -0.0325 a better | -0.0290 a better |

## Table 2: bursts_big

Shots per step 2000; drift 24 steps, onset 6 steps, later 18 steps, after 6 steps.

| decoder | update | LER | LER (drift) | onset | later | after |
|---|---|---|---|---|---|---|
| static MWPM | 2000 | 0.06687 | 0.38329 | 0.38817 | 0.38167 | 0.00692 |
| static MWPM | 200 | 0.06687 | 0.38329 | 0.38817 | 0.38167 | 0.00692 |
| WindowDEM | 2000 | 0.06563 | 0.32625 | 0.35258 | 0.31747 | 0.05258 |
| WindowDEM | 200 | 0.06418 | 0.31838 | 0.30792 | 0.32186 | 0.05375 |
| oracle MWPM | 2000 | 0.03855 | 0.20629 | 0.20525 | 0.20664 | 0.00692 |
| oracle MWPM | 200 | 0.03855 | 0.20629 | 0.20525 | 0.20664 | 0.00692 |
| static-wide | 2000 | 0.04598 ± 0.00120 | 0.25376 ± 0.00754 | 0.25208 ± 0.00869 | 0.25432 ± 0.00719 | 0.00617 ± 0.00008 |
| static-wide | 200 | 0.04598 ± 0.00120 | 0.25376 ± 0.00754 | 0.25208 ± 0.00869 | 0.25432 ± 0.00719 | 0.00617 ± 0.00008 |
| adaptive-wide | 2000 | 0.04483 ± 0.00092 | 0.23978 ± 0.00668 | 0.31439 ± 0.01133 | 0.21491 ± 0.00550 | 0.04761 ± 0.00318 |
| adaptive-wide | 200 | 0.04093 ± 0.00088 | 0.22167 ± 0.00563 | 0.24147 ± 0.00757 | 0.21507 ± 0.00551 | 0.02294 ± 0.00104 |
| adaptive-stale | 2000 | 0.04124 ± 0.00027 | 0.22519 ± 0.00165 | 0.25889 ± 0.00228 | 0.21396 ± 0.00264 | 0.01272 ± 0.00073 |
| adaptive-stale | 200 | 0.03978 ± 0.00044 | 0.21700 ± 0.00277 | 0.22586 ± 0.00213 | 0.21405 ± 0.00301 | 0.00928 ± 0.00013 |

### Comparisons: bursts_big

| a vs b | update | drift | onset | later | after |
|---|---|---|---|---|---|
| adaptive-stale vs static-wide | 2000 | -0.0286 a better | +0.0068 tie | -0.0404 a better | +0.0066 b better |
| adaptive-stale vs static-wide | 200 | -0.0368 a better | -0.0262 a better | -0.0403 a better | +0.0031 b better |
| adaptive-stale vs adaptive-wide | 2000 | -0.0146 a better | -0.0555 a better | -0.0009 tie | -0.0349 a better |
| adaptive-stale vs adaptive-wide | 200 | -0.0047 tie | -0.0156 a better | -0.0010 tie | -0.0137 a better |
| adaptive-wide vs static-wide | 2000 | -0.0140 a better | +0.0623 b better | -0.0394 a better | +0.0414 b better |
| adaptive-wide vs static-wide | 200 | -0.0321 a better | -0.0106 tie | -0.0393 a better | +0.0168 b better |
| adaptive-stale vs oracle MWPM | 2000 | +0.0189 b better | +0.0536 b better | +0.0073 b better | +0.0058 b better |
| adaptive-stale vs oracle MWPM | 200 | +0.0107 b better | +0.0206 b better | +0.0074 b better | +0.0024 b better |
| adaptive-stale vs static MWPM | 2000 | -0.1581 a better | -0.1293 a better | -0.1677 a better | +0.0058 b better |
| adaptive-stale vs static MWPM | 200 | -0.1663 a better | -0.1623 a better | -0.1676 a better | +0.0024 b better |
| adaptive-stale vs WindowDEM | 2000 | -0.1011 a better | -0.0937 a better | -0.1035 a better | -0.0399 a better |
| adaptive-stale vs WindowDEM | 200 | -0.1014 a better | -0.0821 a better | -0.1078 a better | -0.0445 a better |

## Table 2: telegraph

Shots per step 2000; drift 58 steps, onset 4 steps, later 54 steps, after 4 steps.

| decoder | update | LER | LER (drift) | onset | later | after |
|---|---|---|---|---|---|---|
| static MWPM | 2000 | 0.01294 | 0.02397 | 0.02625 | 0.02381 | 0.00563 |
| static MWPM | 200 | 0.01294 | 0.02397 | 0.02625 | 0.02381 | 0.00563 |
| WindowDEM | 2000 | 0.01289 | 0.02401 | 0.02775 | 0.02373 | 0.00600 |
| WindowDEM | 200 | 0.01291 | 0.02397 | 0.02687 | 0.02376 | 0.00563 |
| oracle MWPM | 2000 | 0.01295 | 0.02399 | 0.02625 | 0.02382 | 0.00563 |
| oracle MWPM | 200 | 0.01295 | 0.02399 | 0.02625 | 0.02382 | 0.00563 |
| static-wide | 2000 | 0.01270 ± 0.00006 | 0.02378 ± 0.00011 | 0.02425 ± 0.00070 | 0.02375 ± 0.00013 | 0.00513 ± 0.00033 |
| static-wide | 200 | 0.01270 ± 0.00006 | 0.02378 ± 0.00011 | 0.02425 ± 0.00070 | 0.02375 ± 0.00013 | 0.00513 ± 0.00033 |
| adaptive-wide | 2000 | 0.01125 ± 0.00003 | 0.02093 ± 0.00007 | 0.02200 ± 0.00078 | 0.02085 ± 0.00002 | 0.00417 ± 0.00014 |
| adaptive-wide | 200 | 0.01123 ± 0.00001 | 0.02094 ± 0.00004 | 0.02183 ± 0.00038 | 0.02087 ± 0.00002 | 0.00413 ± 0.00013 |
| adaptive-stale | 2000 | 0.01159 ± 0.00010 | 0.02159 ± 0.00018 | 0.02242 ± 0.00029 | 0.02152 ± 0.00019 | 0.00442 ± 0.00007 |
| adaptive-stale | 200 | 0.01160 ± 0.00008 | 0.02161 ± 0.00012 | 0.02287 ± 0.00045 | 0.02151 ± 0.00013 | 0.00433 ± 0.00019 |

### Comparisons: telegraph

| a vs b | update | drift | onset | later | after |
|---|---|---|---|---|---|
| adaptive-stale vs static-wide | 2000 | -0.0022 a better | -0.0018 tie | -0.0022 a better | -0.0007 tie |
| adaptive-stale vs static-wide | 200 | -0.0022 a better | -0.0014 tie | -0.0022 a better | -0.0008 tie |
| adaptive-stale vs adaptive-wide | 2000 | +0.0007 tie | +0.0004 tie | +0.0007 tie | +0.0003 tie |
| adaptive-stale vs adaptive-wide | 200 | +0.0007 tie | +0.0010 tie | +0.0006 tie | +0.0002 tie |
| adaptive-wide vs static-wide | 2000 | -0.0029 a better | -0.0023 tie | -0.0029 a better | -0.0010 tie |
| adaptive-wide vs static-wide | 200 | -0.0028 a better | -0.0024 tie | -0.0029 a better | -0.0010 tie |
| adaptive-stale vs oracle MWPM | 2000 | -0.0024 a better | -0.0038 tie | -0.0023 a better | -0.0012 tie |
| adaptive-stale vs oracle MWPM | 200 | -0.0024 a better | -0.0034 tie | -0.0023 a better | -0.0013 tie |
| adaptive-stale vs static MWPM | 2000 | -0.0024 a better | -0.0038 tie | -0.0023 a better | -0.0012 tie |
| adaptive-stale vs static MWPM | 200 | -0.0024 a better | -0.0034 tie | -0.0023 a better | -0.0013 tie |
| adaptive-stale vs WindowDEM | 2000 | -0.0024 a better | -0.0053 a better | -0.0022 a better | -0.0016 tie |
| adaptive-stale vs WindowDEM | 200 | -0.0024 a better | -0.0040 tie | -0.0022 a better | -0.0013 tie |

## Table 2: wander

Shots per step 2000; drift 150 steps, onset 0 steps, later 150 steps, after 0 steps.

| decoder | update | LER | LER (drift) | onset | later | after |
|---|---|---|---|---|---|---|
| static MWPM | 2000 | 0.00968 | 0.00968 | – | 0.00968 | – |
| static MWPM | 200 | 0.00968 | 0.00968 | – | 0.00968 | – |
| WindowDEM | 2000 | 0.00962 | 0.00962 | – | 0.00962 | – |
| WindowDEM | 200 | 0.00958 | 0.00958 | – | 0.00958 | – |
| oracle MWPM | 2000 | 0.00968 | 0.00968 | – | 0.00968 | – |
| oracle MWPM | 200 | 0.00968 | 0.00968 | – | 0.00968 | – |
| static-wide | 2000 | 0.00955 ± 0.00006 | 0.00955 ± 0.00006 | – | 0.00955 ± 0.00006 | – |
| static-wide | 200 | 0.00955 ± 0.00006 | 0.00955 ± 0.00006 | – | 0.00955 ± 0.00006 | – |
| adaptive-wide | 2000 | 0.00851 ± 0.00010 | 0.00851 ± 0.00010 | – | 0.00851 ± 0.00010 | – |
| adaptive-wide | 200 | 0.00851 ± 0.00009 | 0.00851 ± 0.00009 | – | 0.00851 ± 0.00009 | – |
| adaptive-stale | 2000 | 0.00872 ± 0.00003 | 0.00872 ± 0.00003 | – | 0.00872 ± 0.00003 | – |
| adaptive-stale | 200 | 0.00871 ± 0.00004 | 0.00871 ± 0.00004 | – | 0.00871 ± 0.00004 | – |

### Comparisons: wander

| a vs b | update | drift | onset | later | after |
|---|---|---|---|---|---|
| adaptive-stale vs static-wide | 2000 | -0.0008 a better | – | -0.0008 a better | – |
| adaptive-stale vs static-wide | 200 | -0.0008 a better | – | -0.0008 a better | – |
| adaptive-stale vs adaptive-wide | 2000 | +0.0002 tie | – | +0.0002 tie | – |
| adaptive-stale vs adaptive-wide | 200 | +0.0002 tie | – | +0.0002 tie | – |
| adaptive-wide vs static-wide | 2000 | -0.0010 a better | – | -0.0010 a better | – |
| adaptive-wide vs static-wide | 200 | -0.0010 a better | – | -0.0010 a better | – |
| adaptive-stale vs oracle MWPM | 2000 | -0.0010 a better | – | -0.0010 a better | – |
| adaptive-stale vs oracle MWPM | 200 | -0.0010 a better | – | -0.0010 a better | – |
| adaptive-stale vs static MWPM | 2000 | -0.0010 a better | – | -0.0010 a better | – |
| adaptive-stale vs static MWPM | 200 | -0.0010 a better | – | -0.0010 a better | – |
| adaptive-stale vs WindowDEM | 2000 | -0.0009 a better | – | -0.0009 a better | – |
| adaptive-stale vs WindowDEM | 200 | -0.0009 a better | – | -0.0009 a better | – |

## Supplementary: bursts_big without saturated bursts

Dropped bursts starting at steps [42, 77] (oracle LER > 0.4); kept onset 4 steps, later 12 steps, after 4 steps.

| decoder | update | onset | later | after |
|---|---|---|---|---|
| static MWPM | 2000 | 0.33512 | 0.32592 | 0.00725 |
| static MWPM | 200 | 0.33512 | 0.32592 | 0.00725 |
| WindowDEM | 2000 | 0.28088 | 0.23463 | 0.05287 |
| WindowDEM | 200 | 0.21913 | 0.24113 | 0.05500 |
| oracle MWPM | 2000 | 0.06588 | 0.06742 | 0.00725 |
| oracle MWPM | 200 | 0.06588 | 0.06742 | 0.00725 |
| static-wide | 2000 | 0.13762 ± 0.01328 | 0.14017 ± 0.01135 | 0.00704 ± 0.00007 |
| static-wide | 200 | 0.13762 ± 0.01328 | 0.14017 ± 0.01135 | 0.00704 ± 0.00007 |
| adaptive-wide | 2000 | 0.22758 ± 0.01898 | 0.07856 ± 0.00780 | 0.06187 ± 0.00082 |
| adaptive-wide | 200 | 0.11988 ± 0.00963 | 0.07856 ± 0.00776 | 0.03067 ± 0.00213 |
| adaptive-stale | 2000 | 0.14625 ± 0.00258 | 0.07731 ± 0.00310 | 0.01483 ± 0.00128 |
| adaptive-stale | 200 | 0.09883 ± 0.00199 | 0.07742 ± 0.00315 | 0.01100 ± 0.00033 |

| a vs b | update | onset | later | after |
|---|---|---|---|---|
| adaptive-stale vs static-wide | 2000 | +0.0086 tie | -0.0629 a better | +0.0078 b better |
| adaptive-stale vs static-wide | 200 | -0.0388 a better | -0.0627 a better | +0.0040 b better |
| adaptive-stale vs adaptive-wide | 2000 | -0.0813 a better | -0.0013 tie | -0.0470 a better |
| adaptive-stale vs adaptive-wide | 200 | -0.0210 a better | -0.0011 tie | -0.0197 a better |
| adaptive-wide vs static-wide | 2000 | +0.0900 b better | -0.0616 a better | +0.0548 b better |
| adaptive-wide vs static-wide | 200 | -0.0177 a better | -0.0616 a better | +0.0236 b better |
| adaptive-stale vs oracle MWPM | 2000 | +0.0804 b better | +0.0099 b better | +0.0076 b better |
| adaptive-stale vs oracle MWPM | 200 | +0.0330 b better | +0.0100 b better | +0.0038 b better |
