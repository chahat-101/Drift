# Round 10 hypotheses (written before any run)
H1: On physics-based drift, the adaptive-stale ensemble (live) beats its frozen-fingerprint version on drift steps of phys_tls, phys_cosmic and phys_mixed (McNemar p < 0.001, and event-level sign test p < 0.01).
H2: The adaptive-stale ensemble beats calibrated static MWPM and the best WindowDEM on drift steps of phys_cosmic and phys_mixed (p < 0.001).
H3 (risk): On the physics base device with no drift, the neural models are worse than calibrated MWPM (distribution shift from training). If true, round 11 retrains on physics-informed noise.
No model is retrained in round 10.
