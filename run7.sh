#!/bin/bash
# Round 7: richer fingerprints for the d=3 adaptive-stale decoder (multiscale, multiscale+pairs), then report,
# then shut the VM down.
#
#   setsid nohup bash run7.sh > results/exp7/run7.out 2>&1 < /dev/null &
#
# A: train decoder_adaptive_d3_{ms,mspair}_s{0,1,2}.pt one at a time (1 job gave the highest total
#    steps/s, results/exp7/concurrency.txt), seeds interleaved: ms s0, mspair s0, ms s1, ...
#    Same command as decoder_adaptive_d3_stale_s*.pt plus --fingerprint (and --save-every/--resume).
# B: ensemble_eval.py per variant x stream seed (members + mean-logit ensemble, live and frozen),
#    batch 16384; test sets on stream seed 0. The baseline and classical decoders are reused (exp6, exp2).
# C: python results/exp7/report7.py (files only)
# D: finish(): STATUS.txt, then sudo shutdown -h now
# A job is skipped if its output exists and is valid. A failed job is logged and the rest go on.
# Each job has a time limit, cut so that nothing runs past DEADLINE_MIN (default 220 = 3 h 40 min)
# after the start. nvidia-smi is logged every 30 s to $OUT/gpu_util.csv.
# Env: DRY_RUN=1 echo the shutdown; TINY=1 rehearsal (600 training steps, checkpoints in the dryrun folder,
#      1 regime, 1 stream seed) into results/exp7_dryrun/; FAIL=1 add a job that fails (missing checkpoint);
#      PRINT=1 print the jobs and exit.
set -uo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
VARIANTS="ms mspair" SEEDS="0 1 2"
if [ "${TINY:-0}" = 1 ]; then
    OUT=results/exp7_dryrun CK=results/exp7_dryrun STREAM_SEEDS="0" REGIMES=bursts_big
    TRAIN_EXTRA="--steps 600 --val-every 300 --save-every 300" STEPS=600
else
    OUT=results/exp7 CK=. STREAM_SEEDS="0 1 2 3 4" REGIMES=bursts,bursts_big,hot_qubit,telegraph
    TRAIN_EXTRA="--save-every 20000" STEPS=60000
fi
TRAIN_LIMIT=${TRAIN_LIMIT:-1800} EVAL_LIMIT=${EVAL_LIMIT:-600} REPORT_LIMIT=${REPORT_LIMIT:-900} BATCH=${BATCH:-16384}
END=$(($(date +%s) + ${DEADLINE_MIN:-220} * 60))
declare -A FPNAME=([ms]=multiscale [mspair]=mspair)
mkdir -p $OUT

NAMES=()
for s in $SEEDS; do for v in $VARIANTS; do NAMES+=("train_${v}_s$s"); done; done
EVARIANTS="$VARIANTS"
[ "${FAIL:-0}" = 1 ] && EVARIANTS="$EVARIANTS failtest"
for v in $EVARIANTS; do for st in $STREAM_SEEDS; do NAMES+=("ens_${v}_st$st"); done; done
NAMES+=(report7)

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

train() {  # train VARIANT SEED: the decoder_adaptive_d3_stale_s* command with --fingerprint
    local v=$1 s=$2 ck=$CK/decoder_adaptive_d3_${1}_s$2.pt
    job train_${v}_s$s $ck ${FPNAME[$v]} $TRAIN_LIMIT $PY -u train.py --data data/d3_wide --epochs 2 --adaptive \
        --stale-prob 0.3 --fingerprint ${FPNAME[$v]} --seed $s --resume $TRAIN_EXTRA --out $ck
}

ensemble() {  # ensemble VARIANT STREAM_SEED: members = seeds 0, 1, 2
    local v=$1 st=$2 tests=0 tflag= p=$CK/decoder_adaptive_d3_${1}_s
    [ $st = 0 ] && tests=1 tflag=--tests
    job ens_${v}_st$st $OUT/d3/st$st/ens_adaptive_d3_$v.json $tests $EVAL_LIMIT $PY -u ensemble_eval.py \
        --checkpoints ${p}0.pt,${p}1.pt,${p}2.pt --data data/d3_r3_p0.003_s0 --streams $REGIMES --stream-seed $st \
        $tflag --batch $BATCH --out $OUT/d3/st$st --name ens_adaptive_d3_$v
}

if [ "${PRINT:-0}" = 1 ]; then
    for s in $SEEDS; do for v in $VARIANTS; do train $v $s; done; done
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
        echo "run7.sh exit $rc at $(date -u +%FT%TZ) after $((SECONDS / 60)) min"
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

exec 9> $OUT/run7.lock
flock -n 9 || { echo "run7.sh is already running"; exit 1; }
rm -rf $OUT/jobs && mkdir -p $OUT/jobs
echo $$ > $OUT/run7.pid
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

[ -f $OUT/gpu_util.csv ] || echo "timestamp, utilization.gpu [%], memory.used [MiB], phase" > $OUT/gpu_util.csv
while true; do
    echo "$(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader), run7" >> $OUT/gpu_util.csv
    sleep 30
done &

log "start: ${#NAMES[@]} jobs -> $OUT (DRY_RUN=${DRY_RUN:-0} TINY=${TINY:-0} FAIL=${FAIL:-0}), deadline $(date -u -d @$END +%T)"
log "step A: training"
for s in $SEEDS; do for v in $VARIANTS; do train $v $s; done; done
log "step B: ensemble evals"
for v in $EVARIANTS; do for st in $STREAM_SEEDS; do ensemble $v $st; done; done
log "step C: report"
job report7 - - $REPORT_LIMIT $PY -u results/exp7/report7.py --exp $OUT --stream-seeds ${STREAM_SEEDS// /,} --regimes $REGIMES
log "all steps finished"
