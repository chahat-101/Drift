#!/bin/bash
# Training throughput of N = 1..3 concurrent d=5 multiscale jobs (the round-8 command, minus --save-every/--resume):
# steps/s from the log times of step 1000 -> 5250 (the val pass at step 5000 included), peak GPU memory
# before the first val pass and overall (per process and in total). Writes results/exp8/concurrency.txt.
# Jobs write to a temp folder and are killed after step 5250 (or when one dies, e.g. out of memory).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python T=$(mktemp -d) OUT=results/exp8/concurrency.txt
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
echo "# d=5 multiscale training (data/d5_wide_x5, batch 512, h48/l4, stale 0.3, --compile, val every 5000), $(date -u +%FT%TZ)" > $OUT
for N in ${NS:-1 2 3}; do
    pids=()
    for k in $(seq $N); do
        $PY -u train.py --data data/d5_wide_x5 --adaptive --stale-prob 0.3 --epochs 2 --val-every 5000 \
            --device cuda --compile --fingerprint multiscale --seed $k --out $T/tp_$k.pt > $T/tp_${N}_$k.log 2>&1 &
        pids+=($!)
    done
    peak=0 peakp=0 pre=0 prep=0 died=
    while true; do
        tot=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        p=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | sort -n | tail -1)
        [ -z "$p" ] && p=0
        [ $tot -gt $peak ] && peak=$tot
        [ $p -gt $peakp ] && peakp=$p
        if ! grep -qsE "  val [0-9]" $T/tp_${N}_*.log; then [ $tot -gt $pre ] && pre=$tot; [ $p -gt $prep ] && prep=$p; fi
        done_n=0
        for k in $(seq $N); do grep -q "^step   5250" $T/tp_${N}_$k.log && done_n=$((done_n + 1)); done
        [ $done_n = $N ] && break
        for k in $(seq $N); do kill -0 ${pids[$((k - 1))]} 2>/dev/null || { died="job $k died: $(grep -m1 -i "error" $T/tp_${N}_$k.log)"; break 2; }; done
        sleep 1
    done
    kill "${pids[@]}" 2>/dev/null; wait 2>/dev/null
    rates=$(for k in $(seq $N); do
        awk '/^step +1000 /{a=$NF} /^step +5250 /{b=$NF} END{sub("s","",a); sub("s","",b); if (b == "") print "na"; else printf "%.1f\n", 4250/(b-a)}' $T/tp_${N}_$k.log
    done | paste -sd' ')
    tot=$(echo $rates | tr ' ' '\n' | awk '{s+=$1} END{printf "%.1f", s}')
    echo "N=$N per-job steps/s: $rates  total $tot steps/s  GPU mem before val: total ${pre} MiB, largest process ${prep} MiB;" \
         "peak incl. val: total ${peak} MiB, largest process ${peakp} MiB ${died}" | tee -a $OUT
    for k in $(seq $N); do grep -E "^data on|^step +(1000|5000|5250) " $T/tp_${N}_$k.log | sed "s/^/    N=$N job $k: /" >> $OUT; done
done
rm -rf $T
