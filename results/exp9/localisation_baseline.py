"""Training-free localisation baseline for the report-9 noise-map evaluation (no model, no training).

    python results/exp9/localisation_baseline.py --streams <dir with regenerated streams> \
        --out results/exp9/localisation_baseline.json

Per step, from the live fingerprint (the same causal buffer as the models: before each 200-shot
sub-batch, the detection rate of every detector over the last 2000 shots, decoder.Fingerprint +
ensemble_eval.window_fps; the step's fingerprint is the mean over its 10 sub-batches, the same
per-shot averaging as noise_eval.step_predictions):
    excess_a = f_a - f_a(nominal)
f_a(nominal) is the frozen fingerprint, i.e. the nominal test set's 2000 fingerprint shots (as
ensemble_eval's frozen runs). Sensitivity check: f_a(nominal) = the exact nominal detection
probability from the nominal DEM.
score_q = sum of excess_a over the detectors adjacent to qubit q in the matching graph: the
endpoints of the matching-graph edges that a single-qubit Pauli fault on q produces (from the DEM of
a circuit whose only noise is DEPOLARIZE1 on q after every TICK, i.e. the hot-qubit noise model;
each of these edges is checked to be an edge of the nominal matching graph).
top-k: the same metric as noise_eval.metrics (k = number of hot qubits, stable argsort, steps with
hot qubits only), over the same 49 qubits (cfg aux_qubits) and phase masks.

Paired comparison with NF seed by seed, on identical steps, from results/exp9/noise/noise_auxfilm_s<k>.npz:
per-step diff = hit(baseline) - hit(NF); exact two-sided sign test over steps with diff != 0, and the
same test over burst events (a maximal run of consecutive steps with the same hot set; event diff =
mean step diff), since consecutive steps of one event are not independent.
"""
import argparse
import json
import math
import sys

import numpy as np
import stim
import torch

sys.path.insert(0, ".")
from circuits import make_circuit, matcher_from_circuit  # noqa: E402
from decoder import Fingerprint, load_checkpoint  # noqa: E402
from drift import DriftStream  # noqa: E402
from ensemble_eval import window_fps  # noqa: E402
from eval import phase_masks  # noqa: E402

REGIMES = ["bursts", "bursts_big", "hot_qubit"]
PHASES = ["drift", "onset", "later"]
VARIANTS = {"NF": "auxfilm", "N": "aux"}
SEEDS = [0, 1, 2]
STREAM_SEEDS = [0, 1, 2, 3, 4]


def adjacency(d, r, qubits, n_det, nominal_edges):
    """(n_qubits, n_det) 0/1: detectors on matching-graph edges caused by a single-qubit fault on each qubit."""
    A = np.zeros((len(qubits), n_det), dtype=np.float64)
    for i, q in enumerate(qubits):
        dem = make_circuit(d, r, 0.0, hot_qubits=(q,), hot_p=1e-3).detector_error_model(decompose_errors=True)
        for inst in dem.flattened():
            if inst.type != "error":
                continue
            comp = []
            for t in inst.targets_copy() + [stim.target_separator()]:
                if t.is_separator():
                    dets = tuple(sorted(comp))
                    if dets and dets not in nominal_edges:
                        raise SystemExit(f"qubit {q}: edge {dets} is not in the nominal matching graph")
                    A[i, list(dets)] = 1
                    comp = []
                elif t.is_relative_detector_id():
                    comp.append(t.val)
    return A


def nominal_edges(circuit):
    """Edges of the nominal matching graph as sorted detector tuples (boundary edges: 1-tuples)."""
    out = set()
    for u, v, _ in matcher_from_circuit(circuit).edges():
        out.add((u,) if v is None else tuple(sorted((u, v))))
    return out


def analytic_rates(circuit):
    """Exact per-detector firing probability under the circuit's noise: 0.5 (1 - prod(1 - 2 p_e))."""
    prod = np.ones(circuit.num_detectors)
    for inst in circuit.detector_error_model(decompose_errors=False).flattened():
        if inst.type == "error":
            p = inst.args_copy()[0]
            for t in inst.targets_copy():
                if t.is_relative_detector_id():
                    prod[t.val] *= 1 - 2 * p
    return 0.5 * (1 - prod)


def step_fingerprints(det, shots, window, fpr, every=200):
    """(steps, n_det) live fingerprint rates per step: mean over the step's sub-batch fingerprints, weighted by shots."""
    steps = len(det) // shots
    starts = [t * shots + i for t in range(steps) for i in range(0, shots, every)]
    f = window_fps(det, starts, window, fpr)[:, 0].double() / 10.0  # rates (the channel is rate x 10)
    sizes = torch.diff(torch.tensor(starts + [len(det)])).double()
    step_of = torch.tensor(starts) // shots
    out = torch.zeros(steps, f.shape[1], dtype=torch.float64).index_add_(0, step_of, f * sizes[:, None])
    return (out / shots).numpy()


def topk_hits(score, hot):
    """Per step with hot qubits: fraction of hot qubits among the k highest scores (as noise_eval.metrics)."""
    k = hot.sum(1)
    rows = np.flatnonzero(k > 0)
    return rows, np.array([len(set(np.argsort(-score[i], kind="stable")[: k[i]]) & set(np.flatnonzero(hot[i]))) / k[i]
                           for i in rows])


def false_picks(score, hot, A, is_data):
    """Over steps with hot qubits: the wrongly picked qubits among the top k, as counts
    (total, measure qubits, sharing >= 1 adjacent detector with a hot qubit)."""
    tot = meas = near = 0
    for i in np.flatnonzero(hot.any(1)):
        k = hot[i].sum()
        top = np.argsort(-score[i], kind="stable")[:k]
        hot_dets = A[hot[i]].any(0)
        for q in top[~hot[i][top]]:
            tot += 1
            meas += int(not is_data[q])
            near += int((A[q].astype(bool) & hot_dets).any())
    return np.array([tot, meas, near])


def sign_test(diff):
    """(n better, n worse, n tie, exact two-sided sign-test p) for diff = baseline - NF."""
    b, w = int((diff > 0).sum()), int((diff < 0).sum())
    n = b + w
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, w) + 1)) / 2**n)
    return b, w, int((diff == 0).sum()), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", required=True, help="folder with stream_<regime>_st<N>.npz and test_nominal_window.npz")
    ap.add_argument("--data", default="data/d5_wide")
    ap.add_argument("--out", default="results/exp9/localisation_baseline.json")
    args = ap.parse_args()
    man = json.load(open(f"{args.data}/manifest.json"))
    a = man["args"]
    stream = DriftStream(a["distance"], a["rounds"], a["p"])
    nominal = stream.nominal()
    n_det = nominal.num_detectors
    _, cfg = load_checkpoint(f"decoder_adaptive_d5_{VARIANTS['NF']}_long_s0.pt")
    for v in VARIANTS.values():
        for s in SEEDS:
            c = load_checkpoint(f"decoder_adaptive_d5_{v}_long_s{s}.pt")[1]
            assert c["window"] == cfg["window"] and c["aux_qubits"] == cfg["aux_qubits"] and c.get("fingerprint", "single") == "single"
    qubits, window = cfg["aux_qubits"], cfg["window"]
    fpr = Fingerprint(cfg, type("G", (), {"n": n_det})())

    A = adjacency(a["distance"], a["rounds"], qubits, n_det, nominal_edges(nominal))
    data_q = {t.value for t in [i for i in nominal.flattened() if i.name == "M"][-1].targets_copy()}  # final data readout
    is_data = np.array([q in data_q for q in qubits])
    wdet = np.load(f"{args.streams}/test_nominal_window.npz")["window_det"].astype(np.uint8)  # unpacked 0/1 (shots, n_det)
    assert wdet.shape == (a["test_window"], n_det)
    refs = {"test-set window": fpr.of_windows(torch.from_numpy(wdet))[0].double().numpy() / 10.0,
            "analytic DEM": analytic_rates(nominal)[None, :]}
    data = {"adjacency": {"detectors_per_qubit": A.sum(1).astype(int).tolist(), "qubits": qubits,
                          "is_data": is_data.tolist()},
            "reference": {k: {"mean_rate": float(v.mean())} for k, v in refs.items()}, "regimes": {}}

    nf = {v: {s: np.load(f"results/exp9/noise/noise_{tag}_s{s}.npz") for s in SEEDS} for v, tag in VARIANTS.items()}
    nfjson = {v: {s: json.load(open(f"results/exp9/noise/noise_{tag}_s{s}.json")) for s in SEEDS} for v, tag in VARIANTS.items()}
    for r in REGIMES:
        per = {ph: {"base": {k: [] for k in refs}, "events": [], **{f"{v}{s}": [] for v in VARIANTS for s in SEEDS}}
               for ph in PHASES}
        ev_off = 0
        fp_counts = {}  # drift steps: false picks (total, measure qubits, next to a hot qubit)
        for st in STREAM_SEEDS:
            z = np.load(f"{args.streams}/stream_{r}_st{st}.npz")
            steps = len(z["scale"])
            det = torch.from_numpy(np.unpackbits(z["det"], axis=-1)[..., :n_det].reshape(-1, n_det))
            f = step_fingerprints(det, a["shots_per_step"], window, fpr)
            hot = nf["NF"][0][f"{r}|st{st}|tmap"] > 0
            n_hot = (z["hot_qubits"] >= 0).sum(1)
            assert np.array_equal(hot.sum(1), n_hot)
            rows = [{"scale": float(s), "n_hot": int(h)} for s, h in zip(z["scale"], n_hot)]
            masks = phase_masks(rows)
            hq = [tuple(x[x >= 0]) for x in z["hot_qubits"]]
            event = np.zeros(steps, dtype=int)
            e = -1
            for t in range(steps):
                if n_hot[t] and (t == 0 or hq[t] != hq[t - 1]):
                    e += 1
                event[t] = e + ev_off if n_hot[t] else -1
            ev_off += e + 1
            base_rows = {}
            for k, ref in refs.items():
                score = (f - ref) @ A.T
                base_rows[k] = topk_hits(score, hot)
            model_rows = {f"{v}{s}": topk_hits(nf[v][s][f"{r}|st{st}|pred"], hot) for v in VARIANTS for s in SEEDS}
            dm = masks["drift"][:, None] & hot
            for key, sc in [("baseline", (f - refs["test-set window"]) @ A.T)] + \
                    [(f"NF s{s}", nf["NF"][s][f"{r}|st{st}|pred"]) for s in SEEDS]:
                fp_counts[key] = fp_counts.get(key, 0) + false_picks(sc, dm, A, is_data)
            rws = base_rows["test-set window"][0]
            for ph in PHASES:
                sel = masks[ph][rws]
                for k in refs:
                    per[ph]["base"][k].append(base_rows[k][1][sel])
                for key, (_, h) in model_rows.items():
                    per[ph][key].append(h[sel])
                per[ph]["events"].append(event[rws][sel])
        out = {}
        for ph in PHASES:
            P = {k: np.concatenate(v) for k, v in per[ph]["base"].items()}
            ev = np.concatenate(per[ph]["events"])
            cell = {"steps": int(len(ev)), "events": int(len(np.unique(ev))),
                    "baseline": {k: float(v.mean()) for k, v in P.items()}, "models": {}}
            for v in VARIANTS:
                for s in SEEDS:
                    h = np.concatenate(per[ph][f"{v}{s}"])
                    ref_val = nfjson[v][s]["metrics"][r][ph]["topk"]
                    assert abs(h.mean() - ref_val) < 1e-12, (r, ph, v, s, h.mean(), ref_val)  # reproduces report 9
                    m = {"topk": float(h.mean())}
                    if v == "NF":
                        for k, b in P.items():
                            d = b - h
                            de = np.array([d[ev == u].mean() for u in np.unique(ev)])
                            bs, ws, ts, p = sign_test(d)
                            be, we, te, pe = sign_test(de)
                            m[k] = {"mean_diff": float(d.mean()), "steps_better": bs, "steps_worse": ws, "steps_tie": ts,
                                    "p_steps": p, "events_better": be, "events_worse": we, "events_tie": te, "p_events": pe}
                    cell["models"][f"{v} s{s}"] = m
            out[ph] = cell
        out["false_picks_drift"] = {k: dict(zip(("total", "measure_qubits", "next_to_hot"), map(int, v)))
                                    for k, v in fp_counts.items()}
        data["regimes"][r] = out
        print(r, out["false_picks_drift"], {ph: (out[ph]["baseline"]["test-set window"], [out[ph]["models"][f"NF s{s}"]["topk"] for s in SEEDS])
                  for ph in PHASES}, flush=True)
    json.dump(data, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
