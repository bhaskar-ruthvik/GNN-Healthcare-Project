#!/bin/zsh
# Final evaluation (2026-09-25): test split for the final models + quality re-ranking + slide-3 before/after.
set -u
PY=env/bin/python; F='FutureWarning|warnings.warn|RuntimeWarning|Mean of empty|pct_of|Pandas4Warning|keyword-only|eval (validation|test):'
U=$(ls -d runs/2*_v4_unmasked_s42); M=$(ls -d runs/2*_v4_masked_s42); B=$(ls -d runs/2*_tn_B_s42)
echo "=== 1 test ranking, three pools  $(date '+%H:%M:%S')"; $PY scripts/compare_pilots.py --split test $U $M 2>&1 | grep -v -E "$F" | tail -4
echo "=== 2 test MIPS allocation  $(date '+%H:%M:%S')"; $PY scripts/mips_analysis.py --split test --runs $U $M --heuristics 2>&1 | grep -v -E "$F" | sed -n '/allocation/,/by severity/p'
echo "=== 3 quality re-ranking, validation  $(date '+%H:%M:%S')"; $PY scripts/mips_rerank.py $U $M --split validation 2>&1 | grep -v -E "$F"
echo "=== 4 quality re-ranking, test  $(date '+%H:%M:%S')"; $PY scripts/mips_rerank.py $U $M --split test 2>&1 | grep -v -E "$F"
echo "=== 5 slide-3 familiarity, before/after (validation)  $(date '+%H:%M:%S')"; for r in $B $M; do $PY scripts/familiarity_check.py $r 2>&1 | grep -v -E "$F"; done
echo "=== FINAL EVAL DONE $(date '+%H:%M:%S')"
