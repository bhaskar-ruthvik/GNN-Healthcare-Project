"""three-node-v4 inputs (STEP9_PROTOCOL §2): v3 plus
  - encounter severity (deterministic per diagnosis code) as a categorical + severity_conflict flag,
  - patient race (computed/imputed source field) as an encounter categorical,
  - MIPS missing-as-low: specialist mips_score missing (not ambiguous) -> scaled value of raw 0.
Masking of demographics is a MODEL flag (ThreeNodeConfig.mask_demographics); the artifact is shared.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Vocab
from .three_node_v3_data import ThreeNodeInputsV3

SEVERITY_ORDER = {"Simple": 1, "Moderate": 2, "Complex": 3}


class ThreeNodeInputsV4(ThreeNodeInputsV3):
    schema_version = "three-node-v4"

    def __init__(self, v3: ThreeNodeInputsV3, row_severity: pd.DataFrame, row_event_map: pd.DataFrame, patient_race: pd.DataFrame):
        self.__dict__.update(v3.__dict__)
        self.encounter_blocks = dict(v3.encounter_blocks)
        self.pair_feature_blocks = dict(v3.pair_feature_blocks)
        self._enc_static_cache = None
        train = self.enc_split == "train"
        # ---- severity per encounter: most severe of its rows + conflict flag --------------------------
        m = row_event_map.merge(row_severity, on="_source_row", how="left")
        m["enc"] = m["encounter_id"].astype(str).map(self.enc_index)
        m = m.dropna(subset=["enc"]); m["enc"] = m["enc"].astype(int)
        m["rank"] = m["severity"].map(SEVERITY_ORDER).fillna(0).astype(int)
        g = m.groupby("enc")["rank"]
        max_rank = g.max().reindex(np.arange(self.n_encounter)).fillna(0).astype(int).to_numpy()
        n_distinct = m[m["rank"] > 0].groupby("enc")["rank"].nunique().reindex(np.arange(self.n_encounter)).fillna(0).astype(int).to_numpy()   # blanks are not a severity
        self.enc_severity_conflict = (n_distinct > 1)
        inv = {v: k for k, v in SEVERITY_ORDER.items()}
        sev_labels = np.array([inv.get(r) for r in max_rank], dtype=object)
        self.severity_vocab = Vocab.fit(sev_labels[train])
        self.enc_severity = self.severity_vocab.transform(sev_labels)
        self.enc_severity_label = sev_labels
        # ---- patient race per encounter ------------------------------------------------------------------
        pr = patient_race.set_index("mso_membership_id")["race"].reindex(self.patient_ids)
        race_by_patient = pr.astype(object).to_numpy()
        self.race_vocab = Vocab.fit(race_by_patient[np.unique(self.enc_patient[train])])
        self.patient_race_idx = self.race_vocab.transform(race_by_patient)
        self.enc_race = self.patient_race_idx[self.enc_patient]
        # ---- encounter categoricals: [gender, insurance, aco, severity, race] ----------------------------------
        self.enc_patient_cat = np.concatenate([v3.enc_patient_cat, self.enc_severity[:, None], self.enc_race[:, None]], axis=1)
        self.encounter_cat_sizes = [self.patient_vocabs[c].size for c in ("gender", "insurance_company", "aco_status")] + [self.severity_vocab.size, self.race_vocab.size]
        # demographic columns for masking: categorical indices and ambiguity-flag columns in the numeric block
        self.demographic_cat_cols = [0, 4]            # gender, race
        self.demographic_flag_cols = [3]              # gender_ambiguous inside patient_context block (age, age_miss, age_amb, gender_amb, ...)
        # ---- numeric block: insert severity_conflict flag after patient_context; shift later blocks ----------
        p = self.encounter_blocks["patient_context"][1]
        shift = 1
        new_blocks = {"patient_context": (0, p), "severity_flags": (p, p + shift)}
        for k, (a, b) in v3.encounter_blocks.items():
            if k == "patient_context": continue
            new_blocks[k] = (a + shift, b + shift)
        self.encounter_blocks = new_blocks
        # ---- MIPS missing-as-low (missing & not ambiguous -> scaled raw 0), in as-of rows; snapshot path below ----
        j = 3 * self.spec_encoder.numeric_cols.index("mips_score")
        self.mips_cols = (j, j + 1, j + 2)
        self.mips_scaled_zero = float(self.spec_encoder.scalers["mips_score"].transform([0.0])[0][0])
        v, miss, amb = self.mips_cols
        rows = (self.spec_hist_num[:, miss] == 1) & (self.spec_hist_num[:, amb] == 0)
        self.spec_hist_num = self.spec_hist_num.copy(); self.spec_hist_num[rows, v] = self.mips_scaled_zero
        self.n_mips_missing_rows_reencoded = int(rows.sum())

    def _encounter_static(self) -> np.ndarray:
        if getattr(self, "_enc_static_cache", None) is None:
            base = super()._encounter_static()
            p = self.encounter_blocks["patient_context"][1]
            self._enc_static_cache = np.concatenate([base[:, :p], self.enc_severity_conflict[:, None].astype(np.float32), base[:, p:]], axis=1)
        return self._enc_static_cache

    def provider_snapshot(self, day: int):
        snap = super().provider_snapshot(day)
        v, miss, amb = self.mips_cols
        x = snap["specialist_num"]
        rows = (x[:, miss] == 1) & (x[:, amb] == 0)               # includes specialists with no earlier row
        x[rows, v] = self.mips_scaled_zero
        return snap

    def dims(self):
        d = super().dims()
        d["encounter_cat_sizes"] = list(self.encounter_cat_sizes)
        d["encounter_num"] = self.encounter_blocks["predecessor_gap"][1]
        return d
