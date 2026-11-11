#!/bin/bash
# Round 5: finish 3 model seeds of the d=5 round-4 setup, then shut the VM down.
#
#   setsid nohup bash run5.sh > results/exp5/run5.out 2>&1 &
#
# A: full evals of adaptive-wide / static-nominal seeds 1, 2 (2 at a time)
# B: train static-wide / adaptive-stale seeds 1, 2 (2 at a time, one stale + one static, --resume;
#    4 at once ran out of GPU memory at the val step in the rehearsal)
# C: full evals of the 4 new models (2 at a time)
# A job is skipped if its output exists and is valid. A failed job is logged and
# the rest go on. On any exit finish() writes STATUS.txt and shuts the VM down.
# Job timeouts: TRAIN_LIMIT / EVAL_LIMIT seconds, cut so that no job runs past DEADLINE_MIN
# minutes after the start (default 540 = 9 h, before the VM's 10 h hard timer).
# Env: DRY_RUN=1 echo the shutdown; TINY=1 rehearsal size, into results/exp5_dryrun/;
#      PRINT=1 print the job commands and exit; SRC=folder of the existing s1/s2 checkpoints.
set -uo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
SRC=${SRC:-}
if [ "${TINY:-0}" = 1 ]; then
    OUT=results/exp5_dryrun CK=results/exp5_dryrun/ EVAL_OUT=results/exp5_dryrun/full
    STREAM_SEEDS="0" REGIMES=bursts_big STEPS=600
    LEN="--epochs 2 --steps 600 --val-every 5000 --save-every 300"
    mkdir -p $EVAL_OUT/st0 && cp -n results/exp4/full/st0/baselines.npz $EVAL_OUT/st0/  # MWPM test cache
else
    OUT=results/exp5 CK= EVAL_OUT=results/exp4/full
    STREAM_SEEDS="0 1 2 3 4" REGIMES=bursts,bursts_big,hot_qubit,telegraph STEPS=300000
    LEN="--epochs 2 --val-every 5000 --save-every 30000"
fi
TRAIN_LIMIT=${TRAIN_LIMIT:-14400} EVAL_LIMIT=${EVAL_LIMIT:-1200}
END=$(($(date +%s) + ${DEADLINE_MIN:-540} * 60))
declare -A FLAGS=([static_wide]="" [adaptive_stale]="--adaptive --stale-prob 0.3")
mkdir -p $OUT

# the plan: A and C are "checkpoint stream_seed", B is "model seed"
A=() B=() C=() NAMES=()
for m in adaptive_wide static_nominal; do for s in 1 2; do for st in $STREAM_SEEDS; do
    A+=("${SRC}decoder_${m}_d5_long_s$s.pt $st") NAMES+=("eval_decoder_${m}_d5_long_s${s}_st$st"); done; done; done
for s in 1 2; do for m in static_wide adaptive_stale; do
    B+=("$m $s") NAMES+=("train_${m}_d5_long_s$s"); done; done
for m in static_wide adaptive_stale; do for s in 1 2; do for st in $STREAM_SEEDS; do
    C+=("${CK}decoder_${m}_d5_long_s$s.pt $st") NAMES+=("eval_decoder_${m}_d5_long_s${s}_st$st"); done; done; done

log() { echo "$(date -u +%T) $*"; }

valid() {  # valid FILE: final checkpoint (.pt) or eval summary (.json) exists and is complete
    [ -s "$1" ] || return 1
    case $1 in
    *.pt) $PY -c "import sys, torch; c = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
assert 'state_dict' in c and c['cfg']['steps'] == int(sys.argv[2])" "$1" $STEPS 2>/dev/null ;;
    *.json) [ -s "${1%.json}_errors.npz" ] && $PY -c "import sys, json; j = json.load(open(sys.argv[1]))
assert set(sys.argv[2].split(',')) <= set(j['streams'])" "$1" $REGIMES 2>/dev/null ;;
    esac
}

job() {  # job NAME OUTPUT LIMIT CMD...: run CMD unless OUTPUT is valid; state goes to $OUT/jobs/NAME
    local name=$1 out=$2 limit=$3 t0=$SECONDS rc state; shift 3
    if [ "${PRINT:-0}" = 1 ]; then echo "$*"; return; fi
    if valid "$out"; then echo "skipped - 0" > $OUT/jobs/$name; log "skip $name"; return; fi
    local left=$((END - $(date +%s)))
    if [ $left -lt 60 ]; then echo "failed deadline 0" > $OUT/jobs/$name; log "deadline: not starting $name"; return; fi
    [ $left -lt $limit ] && limit=$left
    echo "running - 0" > $OUT/jobs/$name; log "start $name (limit ${limit}s)"
    echo "\$ $*" >> $OUT/$name.log
    timeout $limit "$@" >> $OUT/$name.log 2>&1; rc=$?
    if [ $rc = 0 ] && valid "$out"; then state=done; else state=failed; fi
    echo "$state $rc $((SECONDS - t0))" > $OUT/jobs/$name; log "$state $name rc=$rc $((SECONDS - t0))s"
}

train() {  # train MODEL SEED: same command as seed 0 in round 4, only --seed and --out differ
    local out=${CK}decoder_$1_d5_long_s$2.pt
    job train_$1_d5_long_s$2 $out $TRAIN_LIMIT $PY -u train.py --data data/d5_wide_x5 ${FLAGS[$1]} $LEN \
        --device cuda --compile --resume --seed $2 --out $out
}

evaluate() {  # evaluate CHECKPOINT STREAM_SEED: 4 regimes, update every 200, live (+ frozen if adaptive)
    local base=$(basename $1 .pt)
    job eval_${base}_st$2 $EVAL_OUT/st$2/$base.json $EVAL_LIMIT $PY -u curve_eval.py --checkpoint $1 \
        --streams $REGIMES --stream-seed $2 --baseline-tag s${base##*_s} --device cuda --out $EVAL_OUT/st$2
}

pool() {  # pool N FUNC ITEM...: run FUNC on each ITEM, N at a time
    local n=$1 f=$2 it; shift 2
    for it in "$@"; do
        while [ $(jobs -rp | wc -l) -ge $n ]; do wait -n; done
        $f $it &
    done
    wait
}

if [ "${PRINT:-0}" = 1 ]; then pool 1 evaluate "${A[@]}"; pool 1 train "${B[@]}"; pool 1 evaluate "${C[@]}"; exit 0; fi

kill_tree() {  # kill a process, then its descendants
    local kids=$(pgrep -P $1); kill $1 2>/dev/null
    for c in $kids; do kill_tree $c; done
}

finish() {  # stop leftover jobs, write STATUS.txt, shut down
    local rc=$? name s r t
    [ $BASHPID = $$ ] || return
    trap '' INT TERM HUP
    for c in $(pgrep -P $$); do kill_tree $c; done
    wait
    {
        echo "run5.sh exit $rc at $(date -u +%FT%TZ) after $((SECONDS / 60)) min"
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

exec 9> $OUT/run5.lock
flock -n 9 || { echo "run5.sh is already running"; exit 1; }
rm -rf $OUT/jobs && mkdir -p $OUT/jobs
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

log "start: ${#A[@]} evals, ${#B[@]} trainings, ${#C[@]} evals -> $OUT (DRY_RUN=${DRY_RUN:-0} TINY=${TINY:-0})"
log "step A"; pool 2 evaluate "${A[@]}"
log "step B"; pool 2 train "${B[@]}"
log "step C"; pool 2 evaluate "${C[@]}"
log "all steps finished"
