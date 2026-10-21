"""Build new private inputs and verify temporal construction without training."""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT = workspace()
import os
CODE_ROOT = code_file('.')
sys.path.insert(0, str(CODE_ROOT))
from referral_gnn.data import ModelInputs, default_bundle_path
from referral_gnn.three_node_data import ThreeNodeInputs, TEMPORAL_EDGE

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()

def main():
    dest=ROOT/'prepared_data/three_node_v1'
    output=dest/'model_inputs_PRIVATE.pkl'
    if output.exists():raise FileExistsError(output)
    source=Path(default_bundle_path(str(ROOT)))
    baseline_path=source/'model_inputs_PRIVATE.pkl'
    before=digest(baseline_path)
    source_manifest=json.loads((source/'manifest_PRIVATE.json').read_text())
    source_path=ROOT/'data'/source_manifest['source_filename']
    if digest(source_path) != source_manifest['source_sha256']:
        raise ValueError('Source workbook differs from prepared bundle')
    b=ModelInputs.load(str(baseline_path))
    enc=pd.read_pickle(source/'encounter_features_PRIVATE.pkl').set_index('encounter_id').loc[b.encounter_ids]
    mapping_path=dest/'disease_mapping_PRIVATE.json'
    mapping=json.loads(mapping_path.read_text())
    mi=ThreeNodeInputs(b,enc.diagnosis_codes.tolist(),mapping)
    checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name); print('PASS',name,flush=True)
    check('all encounter diagnosis codes found in source mapping',mi.category_audit['unmapped_code_occurrences']==0)
    train_labels={mapping[c] for codes in enc.loc[mi.enc_split=='train','diagnosis_codes'] for c in codes if mapping.get(c)}
    check('category vocabulary exactly training labels',set(mi.dx_vocab.index)==train_labels)
    check('PAD and UNK separate; mask consistent',np.array_equal(mi.enc_dx==0,~mi.enc_dx_mask))
    check('patient feature arrays removed',not hasattr(mi,'patient_num') and 'n_patient' not in mi.dims())
    check('original candidates, split and validation sample preserved',all(np.array_equal(getattr(mi,k),getattr(b,k)) for k in ('tc_enc','tc_spec','tc_label','enc_split','validation_sample')))
    # Adversarial small example: previous-date ties, same-day ties, no past history,
    # another patient's encounters and a query after an in-week encounter.
    s=object.__new__(ThreeNodeInputs)
    s.enc_patient=np.array([0,0,0,0,0,1,1]); s.enc_day=np.array([1,1,3,3,5,2,6]);s.n_encounter=7
    s._build_temporal_index()
    edges,gaps=s.temporal_context(3,np.array([2,3,4,6]))
    expected={(0,2),(1,2),(0,3),(1,3),(0,4),(1,4),(5,6)}
    check('synthetic weekly tie/cutoff/multi-patient edges exact',set(map(tuple,edges.T))==expected and gaps[4]==4)
    edges,gaps=s.temporal_context(5,np.array([4]))
    check('synthetic daily predecessor includes every latest-date tie',set(map(tuple,edges[:,edges[1]==4].T))=={(2,4),(3,4)})
    edges,gaps=s.temporal_context(1,np.array([0,1]))
    check('synthetic no history has empty edges and missing gap',edges.shape==(2,0) and (gaps[:2]==-1).all())
    h=mi.historical_precedes
    check('all historical temporal edges same patient strictly earlier',np.all(mi.enc_patient[h[0]]==mi.enc_patient[h[1]]) and np.all(mi.enc_day[h[0]]<mi.enc_day[h[1]]))
    checked=0
    for split in ('train','validation','test'):
        for policy in ('day','week'):
            buckets=mi.buckets(split,policy)
            for idx in (0,len(buckets)//2,len(buckets)-1):
                cutoff,batch,queries=buckets[idx]
                bt=mi.graph_batch(cutoff,batch)
                ei=bt['edge_index']; te=ei[TEMPORAL_EDGE].numpy()
                assert set(bt['node_types'])=={'pcp','encounter','specialist'}
                assert all(k[0]!='patient' and k[2]!='patient' for k in ei)
                assert not any(k.startswith('patient_') for k in bt['x'])
                for (src,_,dst),edges in ei.items():
                    if src=='encounter':assert (mi.enc_day[edges[0].numpy()]<cutoff).all()
                assert (mi.enc_patient[te[0]]==mi.enc_patient[te[1]]).all()
                assert (mi.enc_day[te[0]]<mi.enc_day[te[1]]).all()
                assert np.isfinite(bt['x']['encounter_num'].numpy()).all()
                control=mi.graph_batch(cutoff,batch,temporal_edges=False)
                assert control['edge_index'][TEMPORAL_EDGE].numel()==0
                assert all(torch.equal(v,control['x'][k]) for k,v in bt['x'].items())
                q=int(queries[0]); solo=mi.graph_batch(cutoff,np.array([q]))
                assert torch.equal(bt['x']['encounter_num'][q],solo['x']['encounter_num'][q])
                for key,edges in ei.items():
                    if key[2]=='encounter':
                        assert torch.equal(edges[:,edges[1]==q],solo['edge_index'][key][:,solo['edge_index'][key][1]==q])
                    else:assert torch.equal(edges,solo['edge_index'][key])
                # Verify actual chosen date is latest permitted date for sampled queries.
                for q in queries[:5]:
                    allowed=np.flatnonzero((mi.enc_patient==mi.enc_patient[q]) & (mi.enc_day<cutoff))
                    actual=te[0,te[1]==q]
                    expected=allowed[mi.enc_day[allowed]==mi.enc_day[allowed].max()] if len(allowed) else allowed
                    assert set(actual)==set(expected)
                checked+=1
    check('18 real day/week graphs: cutoffs, ties, isolation, matched control features',checked==18)
    # Perturb all current/future referral labels; edge inputs must be identical.
    cutoff,batch,_=mi.buckets('validation','week')[2]
    original=mi.graph_batch(cutoff,batch)
    mutated=copy.copy(mi);mutated.enc_target=mi.enc_target.copy()
    mask=mi.enc_day>=cutoff;mutated.enc_target[mask]=(mi.enc_target[mask]+1)%mi.n_specialist
    changed=mutated.graph_batch(cutoff,batch)
    check('withheld target labels do not enter graph',all(torch.equal(v,changed['edge_index'][k]) for k,v in original['edge_index'].items()))
    check('four-node input hash unchanged',digest(baseline_path)==before)
    mi.save(str(output))
    reread=ModelInputs.load(str(output))
    check('saved subclass roundtrip',isinstance(reread,ThreeNodeInputs) and np.array_equal(reread.historical_precedes,mi.historical_precedes))
    report={'schema_version':mi.schema_version,'category_audit':mi.category_audit,'dims':mi.dims(),
        'checks_passed':checks,'graph_cases':checked,'historical_temporal_edges':h.shape[1],
        'scope':'input-construction verification only; new model forward/backward and score invariance not run',
        'source_workbook_sha256':source_manifest['source_sha256'],'baseline_sha256':before,'mapping_sha256':digest(mapping_path),'inputs_sha256':digest(output),
        'source_manifest_sha256':digest(source/'manifest_PRIVATE.json'),
        'code_sha256':{p.name:digest(p) for p in [Path(__file__),code_file('referral_gnn/three_node_data.py'),script_path('extract_disease_mapping.py')]}}
    (dest/'manifest.json').write_text(json.dumps(report,indent=2))
    (dest/'COMPLETE.json').write_text(json.dumps({'manifest_sha256':digest(dest/'manifest.json')}))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
