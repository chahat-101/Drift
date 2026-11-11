#!/bin/bash
# Shut the VM down when no train.py/eval.py runs and the GPU is below 5% for IDLE_MIN minutes.
#
#   setsid nohup bash watchdog.sh > results/exp5/watchdog.out 2>&1 &
#
# Checks every 60 s. Independent of run5.sh. Env: DRY_RUN=1 echoes the shutdown;
# IDLE_MIN (default 20); OUT (default results/exp5) holds watchdog.log.
set -uo pipefail
cd "$(dirname "$0")"
OUT=${OUT:-results/exp5} IDLE_MIN=${IDLE_MIN:-20}
mkdir -p $OUT
note() { echo "$(date -u +%FT%TZ) $*" | tee -a $OUT/watchdog.log; }

note "watchdog start, pid $$, idle limit $IDLE_MIN min, DRY_RUN=${DRY_RUN:-0}"
idle_since=
while true; do
    procs=$(ps -eo comm=,args= | awk '$1 ~ /^python/ && /(train|eval)\.py/' | wc -l)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    [[ $util =~ ^[0-9]+$ ]] || util=0  # nvidia-smi failing counts as idle
    if [ $procs = 0 ] && [ $util -lt 5 ]; then
        [ -z "$idle_since" ] && idle_since=$(date +%s) && note "idle: no train/eval process, GPU $util%"
        if [ $(($(date +%s) - idle_since)) -ge $((IDLE_MIN * 60)) ]; then
            note "idle for $IDLE_MIN min (GPU $util%): sudo shutdown -h now"
            if [ "${DRY_RUN:-0}" = 1 ]; then note "DRY_RUN: sudo shutdown -h now"; else sudo shutdown -h now; fi
            exit 0
        fi
    elif [ -n "$idle_since" ]; then
        idle_since= && note "busy again ($procs train/eval processes, GPU $util%)"
    fi
    sleep 60
done
