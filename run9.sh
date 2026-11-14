#!/bin/bash
# Round 9: noise-map head (N, --aux-noise) and noise-map head + spatial FiLM (NF, --aux-noise --film) at d=5,
# then evals, noise-map diagnostic and report, then shut the VM down.
#
#   setsid nohup bash run9.sh > results/exp9/run9.out 2>&1 < /dev/null &
#
# A: train decoder_adaptive_d5_{aux,auxfilm}_long_s{0,1,2}.pt one at a time, alternating variants by seed
#    (N s0, NF s0, N s1, ...). Command = the seed-0 adaptive-stale d=5 long command
#    (decoder_adaptive_stale_d5_long_s0.pt) plus --aux-noise [--film]; only --seed and --out differ.
#    A training job starts only if TRAIN_NEED seconds plus EVAL_RESERVE are left before the deadline.
# B: ensemble_eval.py per variant x stream seed (members + mean-logit ensemble, live and frozen), batch 16384;
#    test sets on stream seed 0. Members = the seeds whose checkpoints exist for BOTH variants (all 3 unless
#    the deadline cut training). Baseline, static-wide and classical results are reused (exp6, exp3, exp4).
# C: noise_eval.py per model (noise-map quality, live fingerprint, 5 stream seeds).
# D: python results/exp9/report9.py (files only)
# E: finish(): STATUS.txt, then sudo shutdown -h now
# A job is skipped if its output exists and is valid. A failed job is logged and the rest go on.
# Each job has a time limit, cut so that nothing runs past the deadline: DEADLINE_MIN (default 510 = 8 h 30 min)
# after the start, or HARD_END (epoch seconds, default unset) if earlier. nvidia-smi is logged every 30 s.
# Launched with HARD_END = 2026-11-14 03:40 UTC (user VM timer ~03:56 UTC).
# Env: DRY_RUN=1 echo the shutdown; TINY=1 rehearsal (600 training steps, checkpoints in the dryrun folder,
#      1 regime, 1 stream seed, model seeds 0 and 1) into results/exp9_dryrun/; FAIL=1 add a job that fails (missing checkpoint);
#      PRINT=1 print the jobs and exit.
set -uo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
SEEDS="0 1 2" VARIANTS="aux auxfilm"
if [ "${TINY:-0}" = 1 ]; then
    OUT=results/exp9_dryrun CK=results/exp9_dryrun STREAM_SEEDS="0" REGIMES=bursts_big SEEDS="0 1"
    LEN="--epochs 2 --steps 600 --val-every 300 --save-every 300" STEPS=600
else
    OUT=results/exp9 CK=. STREAM_SEEDS="0 1 2 3 4" REGIMES=bursts,bursts_big,hot_qubit,telegraph
    LEN="--epochs 2 --val-every 5000 --save-every 30000" STEPS=300000
fi
TRAIN_LIMIT=${TRAIN_LIMIT:-6000} TRAIN_NEED=${TRAIN_NEED:-4800} EVAL_RESERVE=${EVAL_RESERVE:-2100}
EVAL_LIMIT=${EVAL_LIMIT:-900} REPORT_LIMIT=${REPORT_LIMIT:-900} BATCH=${BATCH:-16384}
END=$(($(date +%s) + ${DEADLINE_MIN:-510} * 60))
[ -n "${HARD_END:-}" ] && [ $HARD_END -lt $END ] && END=$HARD_END
mkdir -p $OUT

NAMES=()
for s in $SEEDS; do for v in $VARIANTS; do NAMES+=("train_${v}_s$s"); done; done
EVARIANTS=$VARIANTS
[ "${FAIL:-0}" = 1 ] && EVARIANTS="$EVARIANTS failtest"
for v in $EVARIANTS; do for st in $STREAM_SEEDS; do NAMES+=("ens_${v}_st$st"); done; done
for v in $VARIANTS; do for s in $SEEDS; do NAMES+=("noise_${v}_s$s"); done; done
NAMES+=(report9)

log() { echo "$(date -u +%T) $*"; }

valid() {  # valid OUTPUT KIND: a finished checkpoint (KIND = aux|auxfilm), ensemble JSON (KIND = TESTS:MEMBERS,
           # TESTS = 1 if tests needed, MEMBERS = the comma-separated checkpoints) or noise-map JSON (KIND = noise)
    case $1 in
    *.pt)
        [ -s "$1" ] && $PY -c "import sys, torch; c = torch.load(sys.argv[1], weights_only=False)['cfg']
assert c['aux_noise'] and c['film'] == (sys.argv[2] == 'auxfilm') and c['steps'] == int(sys.argv[3])" "$1" "$2" $STEPS 2>/dev/null ;;
    *.json)
        [ -s "$1" ] || return 1
        if [ "$2" = noise ]; then
            [ -s "${1%.json}.npz" ] && $PY -c "import sys, json; j = json.load(open(sys.argv[1]))
assert set(sys.argv[2].split(',')) <= set(j['metrics'])" "$1" $REGIMES 2>/dev/null
        else
            [ -s "${1%.json}_errors.npz" ] && $PY -c "import sys, json; j = json.load(open(sys.argv[1]))
t, m = sys.argv[3].split(':')
assert set(sys.argv[2].split(',')) <= set(j['streams']) and (t != '1' or j['tests']) and j['checkpoints'] == m.split(',')" \
                "$1" $REGIMES "$2" 2>/dev/null
        fi ;;
    *) return 1 ;;
    esac
}

job() {  # job NAME OUTPUT KIND LIMIT NEED CMD...: run CMD unless OUTPUT is valid; start only if NEED s are left
         # (beyond the reserve for later steps); state goes to $OUT/jobs/NAME
    local name=$1 out=$2 kind=$3 limit=$4 need=$5 t0=$SECONDS rc state reserve=0; shift 5
    if [ "${PRINT:-0}" = 1 ]; then echo "$*"; return; fi
    if [ "$out" != - ] && valid "$out" $kind; then echo "skipped - 0" > $OUT/jobs/$name; log "skip $name"; return; fi
    [[ $name == train_* ]] && reserve=$EVAL_RESERVE
    local left=$((END - $(date +%s) - reserve))
    if [ $left -lt $need ]; then echo "failed deadline 0" > $OUT/jobs/$name; log "deadline: not starting $name (${left}s left)"; return; fi
    [ $left -lt $limit ] && limit=$left
    echo "running - 0" > $OUT/jobs/$name; log "start $name (limit ${limit}s)"
    echo "\$ $*" >> $OUT/$name.log
    timeout $limit "$@" >> $OUT/$name.log 2>&1 &  # in the background + wait, so a signal runs finish() at once
    wait $!; rc=$?
    if [ $rc = 0 ] && { [ "$out" = - ] || valid "$out" $kind; }; then state=done; else state=failed; fi
    echo "$state $rc $((SECONDS - t0))" > $OUT/jobs/$name; log "$state $name rc=$rc $((SECONDS - t0))s"
}

ckpt() { echo $CK/decoder_adaptive_d5_${1}_long_s$2.pt; }

train() {  # train VARIANT SEED: the decoder_adaptive_stale_d5_long_s0 command plus --aux-noise [--film]
    local v=$1 s=$2 ck=$(ckpt $1 $2) extra=--aux-noise
    [ $v = auxfilm ] && extra="--aux-noise --film"
    job train_${v}_s$s $ck $v $TRAIN_LIMIT $TRAIN_NEED $PY -u train.py --data data/d5_wide_x5 --adaptive --stale-prob 0.3 \
        $LEN --device cuda --compile --resume $extra --seed $s --out $ck
}

members() {  # seeds whose checkpoints are valid for every variant
    local s v ok out=
    for s in $SEEDS; do
        ok=1
        for v in $VARIANTS; do valid $(ckpt $v $s) $v || ok=0; done
        [ $ok = 1 ] && out="$out $s"
    done
    echo $out
}

ensemble() {  # ensemble VARIANT STREAM_SEED: members = $MEMBERS
    local v=$1 st=$2 tests=0 tflag= list=
    [ $st = 0 ] && tests=1 tflag=--tests
    for s in $MEMBERS; do list="$list,$(ckpt $v $s)"; done
    [ $v = failtest ] && list=",$(ckpt failtest 0)"
    job ens_${v}_st$st $OUT/d5/st$st/ens_adaptive_d5_${v}_long.json $tests:${list#,} $EVAL_LIMIT 60 $PY -u ensemble_eval.py \
        --checkpoints ${list#,} --data data/d5_wide --streams $REGIMES --stream-seed $st \
        $tflag --batch $BATCH --out $OUT/d5/st$st --name ens_adaptive_d5_${v}_long --force
}

noise() {  # noise VARIANT SEED
    job noise_${1}_s$2 $OUT/noise/noise_${1}_s$2.json noise $EVAL_LIMIT 60 $PY -u noise_eval.py --checkpoint $(ckpt $1 $2) \
        --data data/d5_wide --streams $REGIMES --stream-seeds ${STREAM_SEEDS// /,} --batch $BATCH \
        --out $OUT/noise --name noise_${1}_s$2
}

if [ "${PRINT:-0}" = 1 ]; then
    MEMBERS=$SEEDS
    for s in $SEEDS; do for v in $VARIANTS; do train $v $s; done; done
    for v in $EVARIANTS; do for st in $STREAM_SEEDS; do ensemble $v $st; done; done
    for v in $VARIANTS; do for s in $SEEDS; do noise $v $s; done; done
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
        echo "run9.sh exit $rc at $(date -u +%FT%TZ) after $((SECONDS / 60)) min"
        echo "ensemble members (seeds): ${MEMBERS:-not computed}"
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

exec 9> $OUT/run9.lock
flock -n 9 || { echo "run9.sh is already running"; exit 1; }
rm -rf $OUT/jobs && mkdir -p $OUT/jobs
echo $$ > $OUT/run9.pid
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

[ -f $OUT/gpu_util.csv ] || echo "timestamp, utilization.gpu [%], memory.used [MiB], phase" > $OUT/gpu_util.csv
while true; do
    echo "$(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader), run9" >> $OUT/gpu_util.csv
    sleep 30
done &

log "start: ${#NAMES[@]} jobs -> $OUT (DRY_RUN=${DRY_RUN:-0} TINY=${TINY:-0} FAIL=${FAIL:-0}), deadline $(date -u -d @$END +%T)"
log "step A: training"
for s in $SEEDS; do for v in $VARIANTS; do train $v $s; done; done
MEMBERS=$(members)
log "step B: ensemble evals, members: seeds ${MEMBERS:-none}"
for v in $EVARIANTS; do for st in $STREAM_SEEDS; do ensemble $v $st; done; done
log "step C: noise-map evals"
for v in $VARIANTS; do for s in $SEEDS; do noise $v $s; done; done
log "step D: report"
job report9 - - $REPORT_LIMIT 60 $PY -u results/exp9/report9.py --exp $OUT --stream-seeds ${STREAM_SEEDS// /,} \
    --regimes $REGIMES --seeds ${SEEDS// /,}
log "all steps finished"
