"""three-node-v3 inputs: v2 plus two history/affiliation pair features.

  pair[10] insurer_history   1 if the candidate specialist saw at least one patient with this
                             encounter's insurer strictly before the query day (history only)
  pair[11] insurer_unknown   1 if the encounter's insurer is missing (then insurer_history = 0)
  pair[12] shared_affiliation 1 if the PCP and the candidate list a common hospital affiliation
  pair[13] affiliation_unknown 1 if the candidate lists no affiliation (then shared = 0); PCP with
                             none listed -> shared = 0 for all candidates and this flag = 1

Affiliations are provider-level strings from the source (static; no dates), so they are a
snapshot like other provider attributes. Insurer history is cutoff-safe (strictly earlier).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import AsOfIndex
from .three_node_v2_data import ThreeNodeInputsV2, V2_PAIR_NAMES

V3_PAIR_NAMES = V2_PAIR_NAMES + ["insurer_history", "insurer_unknown", "shared_affiliation", "affiliation_unknown"]


class ThreeNodeInputsV3(ThreeNodeInputsV2):
    schema_version = "three-node-v3"

    def __init__(self, v2: ThreeNodeInputsV2, encounter_insurer: np.ndarray, specialist_insurer_days: pd.DataFrame, affiliations: dict):
        self.__dict__.update(v2.__dict__)
        self.encounter_blocks = dict(v2.encounter_blocks)
        self.pair_feature_blocks = dict(v2.pair_feature_blocks)
        # insurer vocabulary over both sources
        ins = sorted(set(x for x in encounter_insurer if isinstance(x, str)) | set(specialist_insurer_days["insurer"].dropna()))
        self.insurer_index = {v: i for i, v in enumerate(ins)}
        self.enc_insurer = np.array([self.insurer_index.get(x, -1) if isinstance(x, str) else -1 for x in encounter_insurer], np.int64)
        sid = specialist_insurer_days.copy()
        sid["s"] = sid["rendering_provider_npi"].astype(str).map(self.specialist_index); sid["i"] = sid["insurer"].map(self.insurer_index)
        sid = sid.dropna(subset=["s", "i"])
        self.n_insurer = len(ins)
        self._ins_idx = AsOfIndex((sid["s"].astype(np.int64) * self.n_insurer + sid["i"].astype(np.int64)).to_numpy(), sid["day"].to_numpy(np.int64))
        # affiliations -> sets of small ints per provider
        names = sorted(set().union(*affiliations["pcp"].values(), *affiliations["specialist"].values()))
        aid = {n: i for i, n in enumerate(names)}
        self.pcp_affil = [frozenset(aid[a] for a in affiliations["pcp"].get(p, ())) for p in self.pcp_ids]
        self.spec_affil = [frozenset(aid[a] for a in affiliations["specialist"].get(s, ())) for s in self.specialist_ids]
        self.spec_has_affil = np.array([len(a) > 0 for a in self.spec_affil])
        self.pcp_has_affil = np.array([len(a) > 0 for a in self.pcp_affil])
        self.pair_feature_blocks["insurer"] = [10, 11]; self.pair_feature_blocks["affiliation"] = [12, 13]
        self.pair_feature_dim = len(V3_PAIR_NAMES); self.pair_feature_names = V3_PAIR_NAMES

    def insurer_history(self, enc_idx, spec_idx) -> np.ndarray:
        enc_idx = np.asarray(enc_idx, np.int64); spec_idx = np.asarray(spec_idx, np.int64)
        ins = self.enc_insurer[enc_idx]; known = ins >= 0
        rows = self._ins_idx.lookup(spec_idx * self.n_insurer + np.clip(ins, 0, None), self.enc_day[enc_idx])   # strictly before query day
        return (known & (rows >= 0)).astype(np.float32), (~known).astype(np.float32)

    def shared_affiliation(self, enc_idx, spec_idx) -> np.ndarray:
        enc_idx = np.asarray(enc_idx, np.int64); spec_idx = np.asarray(spec_idx, np.int64)
        pcps = self.enc_pcp[enc_idx]
        shared = np.fromiter((bool(self.pcp_affil[p] & self.spec_affil[s]) for p, s in zip(pcps, spec_idx)), dtype=bool, count=len(pcps))
        unknown = (~self.spec_has_affil[spec_idx]) | (~self.pcp_has_affil[pcps])
        return shared.astype(np.float32), unknown.astype(np.float32)

    def pair_features(self, enc_idx, spec_idx, day=None) -> np.ndarray:
        base = super().pair_features(enc_idx, spec_idx, day)
        ih, iu = self.insurer_history(enc_idx, spec_idx); sa, au = self.shared_affiliation(enc_idx, spec_idx)
        return np.concatenate([base, ih[:, None], iu[:, None], sa[:, None], au[:, None]], axis=1)

    def dims(self):
        d = super().dims(); d["pair"] = self.pair_feature_dim; return d
