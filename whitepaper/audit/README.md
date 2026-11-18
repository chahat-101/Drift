# Audit scripts for `drift_whitepaper.tex`

These scripts recompute the paper's numbers from the saved per-shot error bits:
- every LER, Δ and exact McNemar *p* in Tables 1–3 and 6–8;
- the in-text numbers of §5–§6;
- the event-level sign tests;
- the NF localisation hit rates.

They never train or run a model. `claim_audit.md` (one folder up) lists every claim and the verdict it got from these
outputs. `check_numbers.py` cites `out/*.txt` as the source for numbers that were added after the audit.

## Data

- **In the repository:** the ensemble bits `results/exp6–9/d*/st*/ens_*_errors.npz`, the NF noise maps
  `results/exp9/noise/*.npz`, and the per-step CSVs whose `scale`/`n_hot` columns define the phases.
- **Not in the repository:** the MWPM baseline bits (static, oracle, WindowDEM) and the single-seed / h64/l6 bits.
  They are in `drift_data.tgz`. Extract only those into a folder outside the repository:

```sh
mkdir -p /path/to/bits
tar xzf drift_data.tgz -C /path/to/bits --wildcards 'results/*errors.npz' 'results/*baselines.npz'
export DRIFT_BITS=/path/to/bits        # default: <repo>/audit_bits
```

## Run

```sh
whitepaper/audit/run_all.sh            # uses ../../.venv/bin/python; override with PY=...
```

Each script's output goes to `out/<script>.txt`. The scripts need only NumPy. `t10.py` lists `drift_data.tgz`, which
takes a few minutes.

| script | what it recomputes |
|---|---|
| `lib.py` | Shared helpers: pooled bits, phase masks from the CSVs, log-space exact McNemar, best window per phase |
| `t1.py` | Table 1 (both distances): LERs, seed mean ± s.d., Δ and *p* against four references |
| `t2.py` | Table 2 (live − frozen) and every cell of the per-phase Tables 6–8 |
| `t3.py` | §5.1–5.3 text (onset/later vs oracle, after-step costs); §6.5 ensemble vs members |
| `t4.py` | §5.5 NF top-k hit rates |
| `t5.py` | Table 3 and §6.1–6.3: every variant minus adaptive-stale, all regimes × phases |
| `t6.py` | Test sets; static-wide vs static-nominal per seed (§5.2) |
| `t7.py` | §6.4 h64/l6 vs H_c=48 |
| `t8.py` | §5.2 frozen ensembles vs static-wide; nominal-step advantage over MWPM |
| `t9.py` | §4 event-level sign tests for every Table 1 / Table 2 comparison |
| `t10.py` | §3.3 nominal firing rate; Table 5 dataset sizes |

The *p*-values follow §4 of the paper: exact two-sided McNemar on shared shots, pooled over stream seeds 0–4.
