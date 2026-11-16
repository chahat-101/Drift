# External sources for numbers in the whitepaper that do not come from this project's results

## McEwen et al., Nature Physics 18, 107 (2022), arXiv:2104.05219

Verbatim quotes from the arXiv full text (v1), used in the Introduction and Section 3.1:

- Fig. 1 caption: "We use a 26 qubit subset (dark green) of a Google Sycamore processor."
- Fig. 1 caption: "The number of simultaneous qubit decay errors jumps from baseline ∼ 4 up to ∼ 24, effectively saturating the chip. The number of errors returns to baseline with an exponential time constant of ∼ 25 ms."
- Main text: "We find that the decay timescale is tightly grouped in the 25 to 30 ms range"
- Main text: "occurring on average every 10 s without significant bunching or anti-bunching."
- Abstract: "We track the events from their initial localised impact to high error rates across the chip."

## Klimov et al., Phys. Rev. Lett. 121, 090502 (2018), arXiv:1809.01043

Abstract (no numbers used): "energy-relaxation times, which can fluctuate unpredictably in frequency and time ... direct evidence that [two-level-system defects] are responsible for the largest fluctuations."

## Wall-clock length of one fingerprint window (Section 3.1) — rough estimate, not a hardware claim

Inputs supplied by the author as approximate values: about 1 µs per stabiliser round; 5 rounds per shot at d = 5; W = 2000 shots per window.
Arithmetic: 2000 shots × 5 rounds × 1 µs = 10 ms, plus reset and readout overhead between shots, so of order 10 ms.
