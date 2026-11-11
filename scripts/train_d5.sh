#!/bin/bash
# Train the four d=5 models (static-wide, static-nominal, adaptive-wide, adaptive-stale) for each given
# model seed, JOBS at a time (default 4). fp32, batch 512, h48/l4, 2 epochs.
# A run is skipped if its checkpoint exists and its log says it was saved.
#
#   setsid nohup scripts/train_d5.sh 1 2 > results/exp3/train_d5_s12.out 2>&1 &
cd "$(dirname "$0")/.." && source .venv/bin/activate && export PYTHONUNBUFFERED=1
L=results/exp3
run() {  # name data extra-args...
  local name=$1 data=$2; shift 2
  python train.py --data $data --epochs 2 --val-every 5000 --seed $S --device cuda "$@" \
    --out decoder_${name}_s$S.pt > $L/train_${name}_s$S.log 2>&1
  echo "$name s$S exit $? $(date -u +%FT%TZ)"
}
for S in "$@"; do
  for spec in "static_d5_wide data/d5_wide" "static_d5_nominal_fixed data/d5_nominal" \
              "adaptive_d5_wide data/d5_wide --adaptive" "adaptive_d5_stale data/d5_wide --adaptive --stale-prob 0.3"; do
    read -r name rest <<< "$spec"
    if [ -f decoder_${name}_s$S.pt ] && grep -q "saved decoder_${name}_s$S.pt" $L/train_${name}_s$S.log 2>/dev/null; then
      echo "skip $name s$S"; continue
    fi
    while [ $(jobs -r | wc -l) -ge ${JOBS:-4} ]; do sleep 10; done
    run $spec &
    sleep 2
  done
done
wait
echo all done $(date -u +%FT%TZ)
