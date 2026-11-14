#!/bin/bash
# Round 8: the multiscale fingerprint (round 7's variant B) at d=5, then report, then shut the VM down.
#
#   setsid nohup bash run8.sh > results/exp8/run8.out 2>&1 < /dev/null &
#
# A: train decoder_adaptive_d5_ms_long_s{0,1,2}.pt one at a time (1 job gave the highest total steps/s,
#    results/exp8/concurrency.txt). Command = the seed-0 adaptive-stale d=5 long command
#    (decoder_adaptive_stale_d5_long_s0.pt) plus --fingerprint multiscale; only --seed and --out differ.
# B: ensemble_eval.py per stream seed (members + mean-logit ensemble, live and frozen), batch 16384;
#    test sets on stream seed 0. Baseline, static-wide and classical results are reused (exp6, exp3, exp4).
# C: python results/exp8/report8.py (files only)
# D: finish(): STATUS.txt, then sudo shutdown -h now
# A job is skipped if its output exists and is valid. A failed job is logged and the rest go on.
# Each job has a time limit, cut so that nothing runs past DEADLINE_MIN (default 390 = 6 h 30 min)
# after the start. nvidia-smi is logged every 30 s to $OUT/gpu_util.csv.
# Env: DRY_RUN=1 echo the shutdown; TINY=1 rehearsal (600 training steps, checkpoints in the dryrun folder,
#      1 regime, 1 stream seed) into results/exp8_dryrun/; FAIL=1 add a job that fails (missing checkpoint);
#      PRINT=1 print the jobs and exit.
set -uo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
SEEDS="0 1 2"
if [ "${TINY:-0}" = 1 ]; then
    OUT=results/exp8_dryrun CK=results/exp8_dryrun STREAM_SEEDS="0" REGIMES=bursts_big
    LEN="--epochs 2 --steps 600 --val-every 300 --save-every 300" STEPS=600
else
    OUT=results/exp8 CK=. STREAM_SEEDS="0 1 2 3 4" REGIMES=bursts,bursts_big,hot_qubit,telegraph
    LEN="--epochs 2 --val-every 5000 --save-every 30000" STEPS=300000
fi
TRAIN_LIMIT=${TRAIN_LIMIT:-6000} EVAL_LIMIT=${EVAL_LIMIT:-900} REPORT_LIMIT=${REPORT_LIMIT:-900} BATCH=${BATCH:-16384}
END=$(($(date +%s) + ${DEADLINE_MIN:-390} * 60))
mkdir -p $OUT

NAMES=()
for s in $SEEDS; do NAMES+=("train_ms_s$s"); done
EVARIANTS="ms"
[ "${FAIL:-0}" = 1 ] && EVARIANTS="$EVARIANTS failtest"
for v in $EVARIANTS; do for st in $STREAM_SEEDS; do NAMES+=("ens_${v}_st$st"); done; done
NAMES+=(report8)

log() { echo "$(date -u +%T) $*"; }

valid() {  # valid OUTPUT KIND: a finished checkpoint (KIND = fingerprint) or ensemble JSON (KIND = 1 if tests needed)
    case $1 in
    *.pt)
        [ -s "$1" ] && $PY -c "import sys, torch; c = torch.load(sys.argv[1], weights_only=False)['cfg']
assert c['fingerprint'] == sys.argv[2] and c['steps'] == int(sys.argv[3])" "$1" "$2" $STEPS 2>/dev/null ;;
    *.json)
        [ -s "$1" ] && [ -s "${1%.json}_errors.npz" ] || return 1
        $PY -c "import sys, json; j = json.load(open(sys.argv[1]))
assert set(sys.argv[2].split(',')) <= set(j['streams']) and (sys.argv[3] != '1' or j['tests'])" "$1" $REGIMES "$2" 2>/dev/null ;;
    *) return 1 ;;
    esac
}

job() {  # job NAME OUTPUT KIND LIMIT CMD...: run CMD unless OUTPUT is valid; state goes to $OUT/jobs/NAME
    local name=$1 out=$2 kind=$3 limit=$4 t0=$SECONDS rc state; shift 4
    if [ "${PRINT:-0}" = 1 ]; then echo "$*"; return; fi
    if [ "$out" != - ] && valid "$out" $kind; then echo "skipped - 0" > $OUT/jobs/$name; log "skip $name"; return; fi
    local left=$((END - $(date +%s)))
    if [ $left -lt 60 ]; then echo "failed deadline 0" > $OUT/jobs/$name; log "deadline: not starting $name"; return; fi
    [ $left -lt $limit ] && limit=$left
    echo "running - 0" > $OUT/jobs/$name; log "start $name (limit ${limit}s)"
    echo "\$ $*" >> $OUT/$name.log
    timeout $limit "$@" >> $OUT/$name.log 2>&1 &  # in the background + wait, so a signal runs finish() at once
    wait $!; rc=$?
    if [ $rc = 0 ] && { [ "$out" = - ] || valid "$out" $kind; }; then state=done; else state=failed; fi
    echo "$state $rc $((SECONDS - t0))" > $OUT/jobs/$name; log "$state $name rc=$rc $((SECONDS - t0))s"
}

train() {  # train SEED: the decoder_adaptive_stale_d5_long_s0 command with --fingerprint multiscale
    local s=$1 ck=$CK/decoder_adaptive_d5_ms_long_s$1.pt
    job train_ms_s$s $ck multiscale $TRAIN_LIMIT $PY -u train.py --data data/d5_wide_x5 --adaptive --stale-prob 0.3 \
        $LEN --device cuda --compile --resume --fingerprint multiscale --seed $s --out $ck
}

ensemble() {  # ensemble VARIANT STREAM_SEED: members = seeds 0, 1, 2
    local v=$1 st=$2 tests=0 tflag= p=$CK/decoder_adaptive_d5_${1}_long_s
    [ $st = 0 ] && tests=1 tflag=--tests
    job ens_${v}_st$st $OUT/d5/st$st/ens_adaptive_d5_${v}_long.json $tests $EVAL_LIMIT $PY -u ensemble_eval.py \
        --checkpoints ${p}0.pt,${p}1.pt,${p}2.pt --data data/d5_wide --streams $REGIMES --stream-seed $st \
        $tflag --batch $BATCH --out $OUT/d5/st$st --name ens_adaptive_d5_${v}_long
}

if [ "${PRINT:-0}" = 1 ]; then
    for s in $SEEDS; do train $s; done
    for v in $EVARIANTS; do for st in $STREAM_SEEDS; do ensemble $v $st; done; done
    exit 0
fi

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
        echo "run8.sh exit $rc at $(date -u +%FT%TZ) after $((SECONDS / 60)) min"
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

exec 9> $OUT/run8.lock
flock -n 9 || { echo "run8.sh is already running"; exit 1; }
rm -rf $OUT/jobs && mkdir -p $OUT/jobs
echo $$ > $OUT/run8.pid
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

[ -f $OUT/gpu_util.csv ] || echo "timestamp, utilization.gpu [%], memory.used [MiB], phase" > $OUT/gpu_util.csv
while true; do
    echo "$(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader), run8" >> $OUT/gpu_util.csv
    sleep 30
done &

log "start: ${#NAMES[@]} jobs -> $OUT (DRY_RUN=${DRY_RUN:-0} TINY=${TINY:-0} FAIL=${FAIL:-0}), deadline $(date -u -d @$END +%T)"
log "step A: training"
for s in $SEEDS; do train $s; done
log "step B: ensemble evals"
for v in $EVARIANTS; do for st in $STREAM_SEEDS; do ensemble $v $st; done; done
log "step C: report"
job report8 - - $REPORT_LIMIT $PY -u results/exp8/report8.py --exp $OUT --stream-seeds ${STREAM_SEEDS// /,} --regimes $REGIMES
log "all steps finished"
