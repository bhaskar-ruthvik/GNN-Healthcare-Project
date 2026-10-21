"""three-node-v2 inputs: v1 plus patient-origin distance to each candidate specialist.

Patient ZIP (new source column `zipDFW_patient`, one ZIP per patient) -> 2020 ZCTA
representative point (same public reference as provider locations) -> great-circle
miles to each specialist's office point, computed on the fly per pair. Scaler fit on
training positive pairs only. Missing (unmapped ZIP or provider) stays missing with a
flag; never zero-filled. Everything else is inherited unchanged from v1.
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from .data import NumericScaler, PAIR_FEATURE_NAMES
from .three_node_data import ThreeNodeInputs

V2_PAIR_NAMES = PAIR_FEATURE_NAMES + ["patient_distance_scaled", "patient_distance_missing"]


def load_zcta_points(path: str) -> pd.DataFrame:
    zf = zipfile.ZipFile(path)
    gaz = pd.read_csv(io.BytesIO(zf.read(zf.namelist()[0])), sep="\t", dtype=str)
    gaz.columns = [c.strip() for c in gaz.columns]
    return pd.DataFrame({"lat": gaz["INTPTLAT"].astype(float).to_numpy(), "lon": gaz["INTPTLONG"].astype(float).to_numpy()}, index=gaz["GEOID"].str.zfill(5))


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 3958.7613 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class ThreeNodeInputsV2(ThreeNodeInputs):
    schema_version = "three-node-v2"

    def __init__(self, v1: ThreeNodeInputs, patient_zip: pd.DataFrame, zcta_points: pd.DataFrame, specialist_locations: pd.DataFrame):
        self.__dict__.update(v1.__dict__)
        self.encounter_blocks = dict(v1.encounter_blocks)
        self.pair_feature_blocks = dict(v1.pair_feature_blocks)
        # patient coordinates aligned to patient index
        pz = patient_zip.set_index("mso_membership_id")["zip5"].reindex(self.patient_ids)
        pts = zcta_points.reindex(pz.to_numpy())
        self.patient_lat = pts["lat"].to_numpy(dtype=float); self.patient_lon = pts["lon"].to_numpy(dtype=float)
        self.patient_zip_missing = pz.isna().to_numpy(); self.patient_zip_unmapped = (~pz.isna().to_numpy()) & np.isnan(self.patient_lat)
        # specialist coordinates aligned to specialist index (same table used for PCP-specialist distance)
        sl = specialist_locations.copy(); sl.index = sl.index.astype(str); sl = sl.reindex(self.specialist_ids)
        self.spec_lat = sl["lat"].astype(float).to_numpy(); self.spec_lon = sl["lon"].astype(float).to_numpy()
        # scaler on training positive pairs
        tr = self.query_enc["train"]
        d_train = self.patient_specialist_distance(self.enc_patient[tr], self.enc_target[tr])
        self.patient_distance_scaler = NumericScaler.fit(d_train)
        self.pair_feature_blocks["patient_distance"] = [8, 9]
        self.pair_feature_dim = len(V2_PAIR_NAMES)
        self.pair_feature_names = V2_PAIR_NAMES

    def patient_specialist_distance(self, patient_idx, spec_idx) -> np.ndarray:
        p = np.asarray(patient_idx, np.int64); s = np.asarray(spec_idx, np.int64)
        return haversine_miles(self.patient_lat[p], self.patient_lon[p], self.spec_lat[s], self.spec_lon[s])   # NaN where either is missing

    def pair_features(self, enc_idx, spec_idx, day=None) -> np.ndarray:
        base = super().pair_features(enc_idx, spec_idx, day)
        enc_idx = np.asarray(enc_idx, np.int64); spec_idx = np.asarray(spec_idx, np.int64)
        d = self.patient_specialist_distance(self.enc_patient[enc_idx], spec_idx)
        ds, miss = self.patient_distance_scaler.transform(d)
        return np.concatenate([base, ds[:, None], miss[:, None].astype(np.float32)], axis=1)

    def dims(self):
        d = super().dims(); d["pair"] = self.pair_feature_dim; return d
