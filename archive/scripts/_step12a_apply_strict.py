# applied after the A1 G-seed chain exits (referral_gnn/ is imported by training processes)
import shutil, os
B="runs/_tests/builder_backup_20261013c/referral_gnn"; os.makedirs(B, exist_ok=True)
for f in ("volume.py","three_node_model.py","tabular.py"): shutil.copy(f"referral_gnn/{f}", f"{B}/{f}")
def rep(p,a,b):
    s=open(p).read(); assert s.count(a)==1,(p,a[:60]); open(p,"w").write(s.replace(a,b))
rep("referral_gnn/volume.py",'''PLUS = ["rating_count", "practice_size"]''','''PLUS = ["rating_count", "practice_size"]
# STEP12a Amendment A1 (H3′): "strict" also removes every specialist column that carries within-specialty volume information
# (greedy CV-R² elimination, 2026-10-13): within-specialty R² of log beneficiaries from what remains ≈ −0.05.
STRICT = VOL + ["average_panel_age", "average_panel_risk", "practice_size", "rating_count", "mips_score"]''')
rep("referral_gnn/volume.py",'''    assert mode in ("log1p", "shares")''','''    assert mode in ("log1p", "shares", "strict")''')
rep("referral_gnn/volume.py",'''    if mode == "log1p":''','''    if mode == "strict":                                   # constant 0 (value and flags) for STRICT columns and the MIPS extras (v5 layout)
        assert X.shape[1] == st["extras_start"] + 2, "strict mode is defined for the v5 specialist snapshot (num_dim + 2 MIPS extras)"
        for n in STRICT:
            for j in st["cols_all"][n]: X[:, j] = 0.0
        X[:, st["extras_start"]:] = 0.0
        return X
    if mode == "log1p":''')
rep("referral_gnn/volume.py",'''    out = {"cols": {n: list(col(mi, n)) for n in VOL + PLUS}, "scaler": {n: list(_scaler(mi, n)) for n in VOL + PLUS},''','''    out = {"cols": {n: list(col(mi, n)) for n in VOL + PLUS}, "scaler": {n: list(_scaler(mi, n)) for n in VOL + PLUS},
           "cols_all": {n: list(col(mi, n)) for n in STRICT}, "extras_start": int(mi.spec_encoder.num_dim),''')
rep("referral_gnn/three_node_model.py",'''        assert cfg.volume_mode in ("keep", "log1p", "shares")''','''        assert cfg.volume_mode in ("keep", "log1p", "shares", "strict")''')
rep("referral_gnn/tabular.py",'''volume in ("keep", "shares")''','''volume in ("keep", "shares", "strict")''')
print("strict applied")
shutil.copy("scripts/three_node_train.py", "runs/_tests/builder_backup_20261013c/three_node_train_SCRIPT.py")
rep("scripts/three_node_train.py",'''choices=["keep", "log1p", "shares"]''','''choices=["keep", "log1p", "shares", "strict"]''')
print("cli updated")

rep("referral_gnn/volume.py",'''                   "shares" (B2): beneficiary/service counts removed (constant 0, flags 0), panel counts as standardised shares.''','''                   "shares" (B2): beneficiary/service counts removed (constant 0, flags 0), panel counts as standardised shares;
                   "strict" (Amendment A1, H3′): every specialist column carrying within-specialty volume information set to
                   constant 0 (counts, panel block, average panel age/risk, practice size, rating count, MIPS and its extras);
                   specialty, credentials, years in Medicare and rating kept.''')
print("docstring updated")
