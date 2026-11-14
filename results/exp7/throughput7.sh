#!/bin/bash
# Training throughput of N = 1..6 concurrent d=3 mspair jobs (the heaviest variant): steps/s from the
# log times of step 500 -> 2500 (val every 500 included), peak GPU memory per process and in total.
# Writes results/exp7/concurrency.txt. Jobs write to a temp folder and are killed after step 2500.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python T=$(mktemp -d) OUT=results/exp7/concurrency.txt
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
echo "# d=3 mspair training (data/d3_wide, batch 512, h48/l4, stale 0.3, val every 500), $(date -u +%FT%TZ)" > $OUT
for N in 1 2 3 4 5 6; do
    pids=()
    for k in $(seq $N); do
        $PY -u train.py --data data/d3_wide --epochs 2 --adaptive --stale-prob 0.3 --fingerprint mspair \
            --seed $k --out $T/tp_$k.pt > $T/tp_${N}_$k.log 2>&1 &
        pids+=($!)
    done
    peak=0 peakp=0
    while true; do
        tot=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        [ $tot -gt $peak ] && peak=$tot
        p=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | sort -n | tail -1)
        [ -n "$p" ] && [ $p -gt $peakp ] && peakp=$p
        done_n=0
        for k in $(seq $N); do grep -q "^step   2500" $T/tp_${N}_$k.log && done_n=$((done_n + 1)); done
        [ $done_n = $N ] && break
        for pid in "${pids[@]}"; do kill -0 $pid 2>/dev/null || { echo "N=$N: a job died" >> $OUT; break 2; }; done
        sleep 2
    done
    kill "${pids[@]}" 2>/dev/null; wait 2>/dev/null
    rates=$(for k in $(seq $N); do
        awk '/^step +500 /{a=$NF} /^step +2500 /{b=$NF} END{sub("s","",a); sub("s","",b); printf "%.1f\n", 2000/(b-a)}' $T/tp_${N}_$k.log
    done | paste -sd' ')
    tot=$(echo $rates | tr ' ' '\n' | awk '{s+=$1} END{printf "%.1f", s}')
    echo "N=$N per-job steps/s: $rates  total $tot steps/s  peak GPU mem total ${peak} MiB, largest process ${peakp} MiB" | tee -a $OUT
done
rm -rf $T
