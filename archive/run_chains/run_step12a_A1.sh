#!/bin/zsh
# STEP12a Amendment A1 rev 1: waits for the extra-G-seed chain, applies the strict volume mode (referral_gnn/ edits are deferred
# until no training process imports them), checks it, then trains LightGBM strict (reference config, seeds 42/1, CPU) alongside
# the neural strict cell (6 seeds, MPS). From Implementation/. Usage: zsh scripts/run_step12a_A1.sh <PID of the G-seed chain>
set -e
WAIT=${1:?pid}
while kill -0 $WAIT 2>/dev/null; do sleep 30; done
echo "=== G-seed chain exited; applying strict mode $(date '+%H:%M:%S')"
env/bin/python scripts/_step12a_apply_strict.py
env/bin/python - <<'PY' 2>&1 | grep -v -i warn
import sys; sys.path.insert(0, "."); import numpy as np
from referral_gnn.three_node_train import load_three_node_inputs
from referral_gnn import volume as V
mi = load_three_node_inputs("prepared_data/three_node_v5/model_inputs_PRIVATE.pkl"); st = V.stats(mi); X = V._last_snapshot(mi)["specialist_num"]
Xs = V.transform_np(X, st, "strict"); vary = [j for j in range(X.shape[1]) if np.ptp(Xs[:, j]) > 0]
names = [f"{n}[{k}]" for n in mi.spec_encoder.numeric_cols for k in ("v", "m", "a")] + ["cat_amb0", "cat_amb1", "mips_x0", "mips_x1"]
print("strict check — varying specialist_num columns:", [names[j] for j in vary])
PY
echo "=== LightGBM strict (CPU) + neural strict (MPS) $(date '+%H:%M:%S')"
env/bin/python -W ignore scripts/step12_lgbm.py --objective lambdarank --volume strict --mode fixed --config 127:50:1000 --seeds 42 1 --threads 6 2>&1 | grep -v -i warn | grep -E "wrote|done|Error|Trace" &
zsh scripts/run_runlist.sh grace/runlists/step12a_A1_strict.tsv
wait
echo "=== A1 TRAINING DONE $(date '+%H:%M:%S')"
