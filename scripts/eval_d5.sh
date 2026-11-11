#!/bin/bash
# Evaluate the four d=5 models of one model seed (default 0) on 4 regimes x 5 stream seeds, with live and
# frozen fingerprints and the WindowDEM window sweep. JOBS runs at a time (default 4). A run is skipped if
# its JSON exists.
#
#   JOBS=4 scripts/eval_d5.sh 1
cd "$(dirname "$0")/.." && source .venv/bin/activate && export PYTHONUNBUFFERED=1
S=${1:-0}
run() {
  python eval.py --regime $1 --data data/d5_wide --stream-seed $2 --update-every 200 --frozen-fingerprint \
    --window-sizes 500,1000,2000,5000,10000 --device cuda --out results/exp3/d5 --tag s$S \
    --checkpoint decoder_static_d5_wide_s$S.pt --checkpoint decoder_static_d5_nominal_fixed_s$S.pt \
    --checkpoint decoder_adaptive_d5_wide_s$S.pt --checkpoint decoder_adaptive_d5_stale_s$S.pt \
    > results/exp3/d5/eval_$1_st$2_s$S.log 2>&1
}
for r in bursts bursts_big hot_qubit telegraph; do
  for st in 0 1 2 3 4; do
    [ -f results/exp3/d5/${r}_d5_r5_u200_st${st}_s${S}.json ] && continue
    while [ $(jobs -r | wc -l) -ge ${JOBS:-4} ]; do sleep 5; done
    run $r $st &
  done
done
wait
echo done $(date)
