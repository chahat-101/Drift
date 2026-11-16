"""Figures for the Drift technical document, generated from the real code."""
import sys
sys.path.insert(0, "/home/claude/drift")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from circuits import make_circuit, nearest_qubits
from decoder import DetectorGrid
from drift import make_schedule, NoiseState, DriftStream

OUT = "/tmp/claude-0/-home-claude/b84ece17-bbe2-5335-ae76-16829ae92113/scratchpad/doc/fig/"
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
INK, MUTED, ACC, ACC2, GRID = "#1f2937", "#9ca3af", "#c2410c", "#1d4ed8", "#e5e7eb"

d = 5
stream = DriftStream(d, d, 0.003)
nom = stream.nominal()

# ---------- Fig 1: qubit layout ----------
data_q, meas_q = set(), set()
for inst in nom.flattened():
    if inst.name == "MR":
        meas_q |= {t.value for t in inst.targets_copy()}
    if inst.name == "M":
        data_q |= {t.value for t in inst.targets_copy()}
coords = {q: v[:2] for q, v in nom.get_final_qubit_coordinates().items()}
hot = nearest_qubits(nom, 2)
burst = nearest_qubits(nom, 5, center=coords[sorted(meas_q)[3]])
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
for ax, marked, title in [(axes[0], hot, "hot_qubit: 2 central qubits"),
                          (axes[1], burst, "bursts: a 5-qubit cluster")]:
    for q, (x, y) in coords.items():
        isd = q in data_q
        ax.scatter(x, y, s=70 if isd else 45, marker="o" if isd else "s",
                   facecolor=INK if isd else "white", edgecolor=INK, linewidth=0.9, zorder=2)
        if q in marked:
            ax.scatter(x, y, s=260, facecolor="none", edgecolor=ACC, linewidth=1.8, zorder=3)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(title, fontsize=9)
axes[0].scatter([], [], s=50, facecolor=INK, edgecolor=INK, label="data qubit")
axes[0].scatter([], [], s=35, marker="s", facecolor="white", edgecolor=INK, label="measure qubit")
axes[0].scatter([], [], s=120, facecolor="none", edgecolor=ACC, linewidth=1.8, label="extra noise")
fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(OUT + "layout.pdf"); plt.close(fig)

# ---------- Fig 2: one shot on the detector grid ----------
grid = DetectorGrid(nom)
T, X, Y = grid.shape
occ = np.zeros(grid.shape, bool)
occ.reshape(-1)[grid.flat] = True
st_hot = NoiseState(hot_qubits=hot, hot_p=0.03)
det_n, obs_n = stream.sample(NoiseState(), 4000, 7)
det_h, obs_h = stream.sample(st_hot, 4000, 7)
def pick(det):
    k = det.sum(1)
    return int(np.argmin(np.abs(k - np.median(k[k > 0]) - 1)))
cm = LinearSegmentedColormap.from_list("c", ["white", ACC])
fig, axes = plt.subplots(2, T, figsize=(7.0, 2.9))
for row, (det, lab) in enumerate([(det_n, "nominal"), (det_h, "2 hot qubits")]):
    i = pick(det)
    g = grid.to_grid(det[i:i + 1]).numpy()[0]
    for t in range(T):
        ax = axes[row, t]
        img = np.where(occ[t], 0.0, np.nan)
        ax.imshow(np.where(occ[t], 0.15, np.nan), cmap="Greys", vmin=0, vmax=1)
        yy, xx = np.nonzero(g[t] > 0)
        ax.scatter(xx, yy, s=55, color=ACC, marker="s", zorder=3)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        if row == 0: ax.set_title(f"t = {t}", fontsize=8)
    axes[row, 0].set_ylabel(f"{lab}\n{int(det[i].sum())} events", fontsize=8)
fig.tight_layout()
fig.savefig(OUT + "shot.pdf"); plt.close(fig)

# ---------- Fig 3: fingerprints ----------
states = [("nominal", NoiseState()), ("scale x2", NoiseState(scale=2.0)),
          ("2 hot qubits", st_hot), ("5-qubit burst", NoiseState(hot_qubits=burst, hot_p=0.05))]
maps = []
for _, st in states:
    w, _ = stream.sample(st, 2000, 11)
    f = w.mean(0)
    g = grid.to_grid(f[None, :]).numpy()[0]
    maps.append(np.where(occ, g, np.nan).mean(0) if False else np.nanmean(np.where(occ[1:-1], g[1:-1], np.nan), axis=0))
vmax = max(np.nanmax(m) for m in maps)
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2))
for ax, (lab, _), m in zip(axes, states, maps):
    im = ax.imshow(m, cmap=LinearSegmentedColormap.from_list("f", ["#fff7ed", ACC, "#7c2d12"]), vmin=0, vmax=vmax)
    ax.set_title(lab, fontsize=8.5); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cb.set_label("firing rate", fontsize=8)
fig.savefig(OUT + "fingerprint.pdf", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 4: drift regimes ----------
regs = ["static", "wander", "telegraph", "hot_qubit", "bursts"]
fig, axes = plt.subplots(len(regs), 1, figsize=(6.6, 4.6), sharex=True)
for ax, r in zip(axes, regs):
    sch = make_schedule(r, 150, np.random.default_rng(3), nom)
    s = [x.scale for x in sch]
    h = [len(x.hot_qubits) * x.hot_p for x in sch]
    ax.plot(s, color=ACC2, lw=1.4, label="global scale $s$")
    ax2 = ax.twinx()
    ax2.fill_between(range(150), h, step="mid", color=ACC, alpha=0.35, lw=0, label=r"hot load $|H|\,p_{\mathrm{hot}}$")
    ax.set_ylim(0.3, 2.7); ax2.set_ylim(0, 0.3)
    ax2.spines["top"].set_visible(False)
    ax.set_ylabel(r, fontsize=8, rotation=0, ha="right", va="center")
    ax.tick_params(labelsize=7); ax2.tick_params(labelsize=7, colors=ACC)
axes[-1].set_xlabel("step (2000 shots each)")
h1, l1 = axes[0].get_legend_handles_labels()
from matplotlib.patches import Patch
fig.legend(h1 + [Patch(color=ACC, alpha=0.35)], l1 + [r"hot load $|H|\,p_{\mathrm{hot}}$ (right axis)"],
           loc="upper center", ncol=2, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT + "regimes.pdf"); plt.close(fig)

# stats used in the text
print("grid", grid.shape, "occupied", occ.sum(), "of", occ.size)
for lab, st in states:
    w, o = stream.sample(st, 50000, 5)
    print(lab, "rate %.4f" % w.mean(), "events/shot %.2f" % w.sum(1).mean())
print("hot", hot, "burst", burst)
