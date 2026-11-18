#!/bin/sh
# Rerun every audit script and save its output under out/ (see README.md). Needs DRIFT_BITS.
set -e
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
mkdir -p out
for t in t1 t2 t3 t4 t5 t6 t7 t8 t9 t10; do
    echo "== $t"
    $PY $t.py > out/$t.txt
done
