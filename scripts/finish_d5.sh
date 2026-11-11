#!/bin/bash
# Wait for the d=5 training and the seed-1 evals, then run the seed-2 evals, the latency
# measurements (on an otherwise idle machine) and the reports.
cd "$(dirname "$0")/.." && source .venv/bin/activate && export PYTHONUNBUFFERED=1
L=results/exp3
until grep -q "all done" $L/train_d5_s12.out 2>/dev/null && grep -q "^done" $L/eval_d5_s1.out 2>/dev/null; do sleep 30; done
JOBS=4 scripts/eval_d5.sh 2 > $L/eval_d5_s2.out 2>&1
for d in 3 5; do
  [ -f $L/latency_d$d.json ] && continue
  python latency.py --data data/d${d}_wide --checkpoint decoder_static_d${d}_wide_s0.pt \
    --checkpoint decoder_adaptive_d${d}_wide_s0.pt --windows 500,1000,2000,5000,10000 \
    --out $L/latency_d$d.json > $L/latency_d$d.log 2>&1
done
python report.py --exp2 --distance 5 --seeds 0,1,2 > $L/report3_d5.log 2>&1
python report.py --exp3 --seeds 0,1,2 > $L/report3_extra.log 2>&1
echo final done $(date -u +%FT%TZ)
