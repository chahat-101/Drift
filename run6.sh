#!/bin/bash
# Round 6: seed ensembles (mean logit of model seeds 0-2 > 0) at d=5 and d=3, then report, then shut the VM down.
#
#   setsid nohup bash run6.sh > results/exp6/run6.out 2>&1 < /dev/null &
#
# A: ensemble_eval.py for each model type x stream seed, one at a time (1 process gave the highest
#    throughput, results/exp6/concurrency.txt), batch 16384; tests on stream seed 0
# B: python results/exp6/report6.py (files only)
# C: finish(): STATUS.txt, then sudo shutdown -h now
# A job is skipped if its output exists and is valid. A failed job is logged and the rest go on.
# Each job has EVAL_LIMIT seconds, cut so that nothing runs past DEADLINE_MIN (default 160 = 2 h 40 min)
# after the start. nvidia-smi is logged every 30 s to $OUT/gpu_util.csv.
# Env: DRY_RUN=1 echo the shutdown; TINY=1 rehearsal (1 regime, 1 stream seed, 2 ensembles) into
#      results/exp6_dryrun/; FAIL=1 add a job that fails (missing checkpoint); PRINT=1 print the jobs and exit.
set -uo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
if [ "${TINY:-0}" = 1 ]; then
    OUT=results/exp6_dryrun STREAM_SEEDS="0" REGIMES=bursts_big TESTS=
    TYPES="adaptive_wide_d5_long static_d3_wide"
else
    OUT=results/exp6 STREAM_SEEDS="0 1 2 3 4" REGIMES=bursts,bursts_big,hot_qubit,telegraph TESTS=--tests
    TYPES="adaptive_wide_d5_long adaptive_stale_d5_long static_wide_d5_long static_nominal_d5_long
           adaptive_d3_wide adaptive_d3_stale static_d3_wide static_d3_nominal"
fi
[ "${FAIL:-0}" = 1 ] && TYPES="$TYPES fail_test_d5_long"
EVAL_LIMIT=${EVAL_LIMIT:-900} REPORT_LIMIT=${REPORT_LIMIT:-900} BATCH=${BATCH:-16384}
END=$(($(date +%s) + ${DEADLINE_MIN:-160} * 60))
mkdir -p $OUT

NAMES=()
for t in $TYPES; do for st in $STREAM_SEEDS; do NAMES+=("ens_${t}_st$st"); done; done
NAMES+=(report6)

log() { echo "$(date -u +%T) $*"; }

valid() {  # valid JSON [tests]: ensemble summary is complete (every regime; test sets if asked) and its npz exists
    [ -s "$1" ] && [ -s "${1%.json}_errors.npz" ] || return 1
    $PY -c "import sys, json; j = json.load(open(sys.argv[1]))
assert set(sys.argv[2].split(',')) <= set(j['streams']) and (sys.argv[3] != '1' or j['tests'])" "$1" $REGIMES "$2" 2>/dev/null
}

job() {  # job NAME OUTPUT TESTS LIMIT CMD...: run CMD unless OUTPUT is valid; state goes to $OUT/jobs/NAME
    local name=$1 out=$2 tests=$3 limit=$4 t0=$SECONDS rc state; shift 4
    if [ "${PRINT:-0}" = 1 ]; then echo "$*"; return; fi
    if [ "$out" != - ] && valid "$out" $tests; then echo "skipped - 0" > $OUT/jobs/$name; log "skip $name"; return; fi
    local left=$((END - $(date +%s)))
    if [ $left -lt 60 ]; then echo "failed deadline 0" > $OUT/jobs/$name; log "deadline: not starting $name"; return; fi
    [ $left -lt $limit ] && limit=$left
    echo "running - 0" > $OUT/jobs/$name; log "start $name (limit ${limit}s)"
    echo "\$ $*" >> $OUT/$name.log
    timeout $limit "$@" >> $OUT/$name.log 2>&1 &  # in the background + wait, so a signal runs finish() at once
    wait $!; rc=$?
    if [ $rc = 0 ] && { [ "$out" = - ] || valid "$out" $tests; }; then state=done; else state=failed; fi
    echo "$state $rc $((SECONDS - t0))" > $OUT/jobs/$name; log "$state $name rc=$rc $((SECONDS - t0))s"
}

ensemble() {  # ensemble TYPE STREAM_SEED: members = seeds 0, 1, 2 of TYPE
    local t=$1 st=$2 d=5 pat data=data/d5_wide tests=0 tflag=
    case $t in
    *_d5_long) pat=decoder_${t%_d5_long}_d5_long_s ;;
    *) d=3 pat=decoder_${t}_s data=data/d3_r3_p0.003_s0 ;;
    esac
    [ $st = 0 ] && [ -n "$TESTS" ] && tests=1 tflag=--tests
    job ens_${t}_st$st $OUT/d$d/st$st/ens_$t.json $tests $EVAL_LIMIT $PY -u ensemble_eval.py \
        --checkpoints ${pat}0.pt,${pat}1.pt,${pat}2.pt --data $data --streams $REGIMES --stream-seed $st $tflag \
        --batch $BATCH --out $OUT/d$d/st$st --name ens_$t
}

if [ "${PRINT:-0}" = 1 ]; then for t in $TYPES; do for st in $STREAM_SEEDS; do ensemble $t $st; done; done; exit 0; fi

kill_tree() {  # kill a process, then its descendants
    local kids=$(pgrep -P $1); kill $1 2>/dev/null
    for c in $kids; do kill_tree $c; done
}

finish() {  # stop leftover jobs and the GPU logger, write STATUS.txt, shut down
    local rc=$? name s r t
    [ $BASHPID = $$ ] || return
    trap '' INT TERM HUP
    for c in $(pgrep -P $$); do kill_tree $c; done
    wait
    {
        echo "run6.sh exit $rc at $(date -u +%FT%TZ) after $((SECONDS / 60)) min"
        printf "%-40s %-12s %-4s %s\n" job state rc seconds
        for name in "${NAMES[@]}"; do
            s=not_started r=- t=-
            [ -f $OUT/jobs/$name ] && read -r s r t < $OUT/jobs/$name
            [ "$s" = running ] && s=interrupted
            printf "%-40s %-12s %-4s %s\n" $name $s $r $t
        done
        for name in "${NAMES[@]}"; do
            if grep -qs "^failed" $OUT/jobs/$name; then
                echo; echo "--- $name: last 20 lines of $OUT/$name.log ---"; tail -20 $OUT/$name.log
            fi
        done
    } > $OUT/STATUS.txt
    cat $OUT/STATUS.txt
    if [ "${DRY_RUN:-0}" = 1 ]; then log "DRY_RUN: sudo shutdown -h now"; else log "sudo shutdown -h now"; sudo shutdown -h now; fi
}

exec 9> $OUT/run6.lock
flock -n 9 || { echo "run6.sh is already running"; exit 1; }
rm -rf $OUT/jobs && mkdir -p $OUT/jobs
echo $$ > $OUT/run6.pid
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

[ -f $OUT/gpu_util.csv ] || echo "timestamp, utilization.gpu [%], memory.used [MiB], phase" > $OUT/gpu_util.csv
while true; do
    echo "$(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader), run6" >> $OUT/gpu_util.csv
    sleep 30
done &

log "start: ${#NAMES[@]} jobs -> $OUT (DRY_RUN=${DRY_RUN:-0} TINY=${TINY:-0} FAIL=${FAIL:-0}), deadline $(date -u -d @$END +%T)"
log "step A: ensemble evals"
for t in $TYPES; do for st in $STREAM_SEEDS; do ensemble $t $st; done; done
log "step B: report"
job report6 - 0 $REPORT_LIMIT $PY -u results/exp6/report6.py --exp $OUT --stream-seeds ${STREAM_SEEDS// /,} --regimes $REGIMES
log "all steps finished"
