"""Versioned three-node model inputs; leaves the four-node pipeline untouched.

Clinical assignments come from the existing source-derived category column.
Only the vocabulary is learned, using training encounters. Patient identities
remain private linkage metadata, never graph nodes or numerical features.
"""
from bisect import bisect_left
import numpy as np
import torch
from .data import ModelInputs, DxVocab

TEMPORAL_EDGE = ('encounter', 'precedes', 'encounter')
NODE_TYPES = ('pcp', 'encounter', 'specialist')


class ThreeNodeInputs(ModelInputs):
    schema_version = 'three-node-v1'

    def __init__(self, baseline, encounter_codes, category_mapping):
        # Reuse immutable linkage, histories and fitted nonclinical preprocessing.
        self.__dict__.update(baseline.__dict__)
        self.encounter_blocks = dict(baseline.encounter_blocks)
        self._enc_static_cache = None
        categories = []
        unmapped_codes = 0
        for codes in encounter_codes:
            names = set()
            for code in codes:
                if code not in category_mapping:
                    unmapped_codes += 1
                name = category_mapping.get(code, '')
                # None denotes a present code with no source category, not padding.
                names.add(name if name else None)
            categories.append(sorted((s for s in names if s is not None)) + ([None] if None in names else []))
        self.dx_vocab = DxVocab.fit([c for i, cs in enumerate(categories)
                                    if self.enc_split[i] == 'train' for c in cs if c is not None])
        width = max(1, max(map(len, categories), default=0))
        ids = np.zeros((self.n_encounter, width), np.int64)
        mask = np.zeros_like(ids, dtype=bool)
        for i, cs in enumerate(categories):
            ids[i, :len(cs)] = self.dx_vocab.transform(cs)
            mask[i, :len(cs)] = True
        self.enc_dx, self.enc_dx_mask = ids, mask
        self.enc_dx_empty = ~mask.any(axis=1)
        self.enc_dx_unk_share = ((ids == 1).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)).astype(np.float32)
        self.clinical_representation = 'source disease categories; distinct multi-category bag'
        self.category_audit = {
            'source_labels': len({c for cs in categories for c in cs if c is not None}),
            'train_labels': len(self.dx_vocab.index),
            'unmapped_code_occurrences': unmapped_codes,
            'max_categories_per_encounter': width,
            'empty_encounters': int(self.enc_dx_empty.sum()),
            'unknown_category_encounters_by_split': {
                s:int(((self.enc_dx_unk_share > 0) & (self.enc_split == s)).sum())
                for s in ('train','validation','test')},
        }
        end = self.encounter_blocks['time'][1]
        self.encounter_blocks['predecessor_gap'] = (end, end+2)
        self._build_temporal_index()
        # No standalone patient feature or Patient–PCP edge arrays retained.
        for name in ('patient_num','patient_cat','seen_by_src','seen_by_dst','seen_by_first_day'):
            self.__dict__.pop(name, None)

    def _build_temporal_index(self):
        groups = {}
        for i, (p,d) in enumerate(zip(self.enc_patient, self.enc_day)):
            groups.setdefault(int(p), {}).setdefault(int(d), []).append(i)
        self.patient_date_groups = {
            p:(sorted(gs), {d:np.asarray(es, np.int64) for d,es in gs.items()})
            for p,gs in groups.items()}
        src, dst = [], []
        self.historical_gap_days = np.full(self.n_encounter, -1, np.int64)
        for days, gs in self.patient_date_groups.values():
            for prev, day in zip(days, days[1:]):
                earlier, later = gs[prev], gs[day]
                src.extend(np.repeat(earlier, len(later)))
                dst.extend(np.tile(later, len(earlier)))
                self.historical_gap_days[later] = day-prev
        self.historical_precedes = np.asarray([src, dst], np.int64).reshape(2,-1)

    def temporal_context(self, cutoff, query_enc):
        """Historical temporal edges plus isolated queries' latest pre-cutoff date."""
        query_enc = np.asarray(query_enc, np.int64)
        if len(np.unique(query_enc)) != len(query_enc):
            raise ValueError('Duplicate query indices')
        if np.any(self.enc_day[query_enc] < cutoff):
            raise ValueError('Query before cutoff')
        h = self.historical_precedes
        edges = [h[:, self.enc_day[h[1]] < cutoff]]
        gaps = np.full(self.n_encounter, -1, np.int64)
        hist = self.enc_day < cutoff
        gaps[hist] = self.historical_gap_days[hist]
        for q in query_enc:
            days, groups = self.patient_date_groups[int(self.enc_patient[q])]
            pos = bisect_left(days, int(cutoff))-1
            if pos < 0:
                continue
            pd = days[pos]
            sources = groups[pd]
            edges.append(np.stack([sources, np.full(len(sources), q, np.int64)]))
            gaps[q] = self.enc_day[q]-pd
        return np.concatenate(edges, axis=1), gaps

    def graph_batch(self, cutoff, batch_enc, device='cpu', temporal_edges=True):
        """Build exactly three node types; both arms share features including gaps."""
        cutoff = int(cutoff)
        queries = np.asarray(batch_enc, np.int64)
        temporal, gaps = self.temporal_context(cutoff, queries)
        hist = self.enc_day < cutoff
        visible = hist.copy(); visible[queries] = True
        hi = np.flatnonzero(hist)
        vi = np.flatnonzero(visible)
        snap = self.provider_snapshot(cutoff)
        from .data import time_encoding
        gap_num = np.stack([np.log1p(np.maximum(gaps,0)), gaps < 0], axis=1).astype(np.float32)
        enc_num = np.concatenate([self._encounter_static(),
            time_encoding(np.where(hist, cutoff-self.enc_day, 0)), gap_num], axis=1)
        t = lambda a: torch.as_tensor(a, device=device)
        edge_index = {
            ('pcp','has_encounter','encounter'): t(np.stack([self.enc_pcp[vi],vi])),
            ('encounter','rev_has_encounter','pcp'): t(np.stack([hi,self.enc_pcp[hi]])),
            ('encounter','referred_to','specialist'): t(np.stack([hi,self.enc_target[hi]])),
            ('specialist','rev_referred_to','encounter'): t(np.stack([self.enc_target[hi],hi])),
            TEMPORAL_EDGE: t(temporal if temporal_edges else np.empty((2,0),np.int64)),
        }
        x = {f'{nt}_{kind}': t(snap[f'{nt}_{kind}']) for nt in ('pcp','specialist') for kind in ('num','cat')}
        x.update(encounter_num=t(enc_num),encounter_cat=t(self.enc_patient_cat),
                 encounter_dx=t(self.enc_dx),encounter_dx_mask=t(self.enc_dx_mask))
        return {'day':cutoff, 'cutoff':cutoff, 'query_enc':t(queries), 'x':x,
                'edge_index':edge_index, 'node_types':NODE_TYPES,
                'temporal_gap_days':t(self.enc_day[temporal[1]]-self.enc_day[temporal[0]]) if temporal_edges else t(np.empty(0,np.int64)),
                'available':t(self.available_specialists(cutoff)),
                'n_visible_encounters':int(visible.sum())}

    def dims(self):
        d = super().dims()
        for k in ('patient_num','patient_cat_sizes','n_patient'):
            d.pop(k, None)
        d['encounter_num'] = self.encounter_blocks['predecessor_gap'][1]
        return d
