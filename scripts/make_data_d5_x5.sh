#!/bin/bash
# Make d5_wide_x5 and d5_nominal_x5: the d5_wide and d5_nominal datasets with 5x the train chunks.
# Seeds are unchanged, so val, test and streams are the same as in the originals.
cd "$(dirname "$0")/.." && source .venv/bin/activate && export PYTHONUNBUFFERED=1
R=regimes=hot_qubit,bursts,bursts_big,telegraph
[ -f data/d5_wide_x5/manifest.json ] || python make_dataset.py --distance 5 --rounds 5 --name d5_wide_x5 \
  --train-chunks 150000 --workers 6 > results/exp4/make_d5_wide_x5.log 2>&1
python make_dataset.py --streams-only --name d5_wide_x5 --stream-seeds 1,2,3,4 --$R >> results/exp4/make_d5_wide_x5.log 2>&1
[ -f data/d5_nominal_x5/manifest.json ] || python make_dataset.py --distance 5 --rounds 5 --nominal-only --window 0 \
  --name d5_nominal_x5 --train-chunks 150000 --workers 6 > results/exp4/make_d5_nominal_x5.log 2>&1
python make_dataset.py --streams-only --name d5_nominal_x5 --stream-seeds 1,2,3,4 --$R >> results/exp4/make_d5_nominal_x5.log 2>&1
echo done $(date -u +%FT%TZ)
