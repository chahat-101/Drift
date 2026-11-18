"""Section 3.3 nominal firing rate (from results/exp9/localisation_baseline.json) and, if drift_data.tgz is present,
the on-disk size of each dataset in it (Table 5). The tar listing takes a few minutes."""
import json, os, subprocess
from lib import R
j = json.load(open(f"{R}/results/exp9/localisation_baseline.json"))["reference"]
a = 100 * j['analytic DEM']['mean_rate']
print(f"nominal mean firing rate d=5: analytic DEM {a:.3f} % (rounded {a:.2f} %), "
      f"nominal test-set window {100*j['test-set window']['mean_rate']:.3f} %")
tgz = os.path.join(R, "drift_data.tgz")
if os.path.exists(tgz):
    sizes = {}
    for line in subprocess.run(["tar", "tvzf", tgz], capture_output=True, text=True).stdout.splitlines():
        f = line.split()
        if len(f) >= 6 and f[5].startswith("data/") and f[5].count("/") >= 2:
            sizes[f[5].split("/")[1]] = sizes.get(f[5].split("/")[1], 0) + int(f[2])
    for k in sorted(sizes):
        m = sizes[k] / 2**20
        print(f"dataset {k}: {sizes[k]} bytes = {m:.1f} MiB (rounded {m:.0f} MiB) = {m/1024:.2f} GiB")
