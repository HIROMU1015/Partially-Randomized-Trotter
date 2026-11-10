"""Author bookkeeping/source checks only; never fit, predict, sample, compile or sum G."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import r_price_phase1 as m
from trottertracks.r_price_phase1_runner import read, check_source
OUT=Path(__file__).resolve().parent
RUN=OUT/'run01'
FROZEN='7eeabf0ae482ee610d667e28470692d209f2bb21'
BASE='c596e88d2a88fb219c2e373fb0d72c6e39cb723a'


def main():
    plan=read(RUN/'PLAN.json');m.validate_plan(plan);check_source(ROOT,plan)
    assert plan==read(ROOT/'docs/research/r_price_phase1_preparation_20261111/PLAN.json')
    stop=read(RUN/'STOP.json');assert stop['status']=='PHASE1_COMPLETE_MANDATORY_STOP'
    for name in ('FROZEN_MODELS.json','CALIBRATION.json','CALIBRATION_EVALUATION.json','PLAN.json'):
        assert (RUN/name).read_bytes()==m.blob(ROOT,FROZEN,str((RUN/name).relative_to(ROOT)))
    frozen=read(RUN/'FROZEN_MODELS.json');assert not frozen['validation_prices_observed'] and not frozen['validation_refit']
    cal=read(RUN/'CALIBRATION.json');val=read(RUN/'VALIDATION.json');rare=read(RUN/'RARE_DIAGNOSTICS.json')
    assert set(cal)=={c['id'] for c in plan['cells'] if c['role']=='CALIBRATION'}
    assert set(val['data'])=={c['id'] for c in plan['cells'] if c['role']=='VALIDATION'}
    assert val['frozen_model_commit']==FROZEN and val['validation_refit'] is False and val['full_rpe_resources'] is None
    normal=[];saved=0;counts=Counter();private_bindings=[]
    def raw_check(r,conditional):
        assert r['conditional'] is conditional and r['compiler']==m.p.PROFILE['compiler']
        file=Path(r['source']['private_result']);assert read(file)==r
        evidence=read(file.with_name('events_PRIVATE.json'));assert evidence['conditional'] is conditional
        events=evidence['events'];assert m.digest(events)==r['event_digest']
        assert read(file.with_name('STRUCTURAL.json'))['S']==r['S']
        orders=Counter(e['taylor_order'] for batch in events for e in batch)
        for j,axis in enumerate(m.AXES):
            price=read(file.with_name(axis+'.json'))
            assert price['metrics']['rz_count']==r['C'][j] and price['fingerprint']==r['fingerprints'][j]
            assert price['compiler']==m.p.PROFILE['compiler']
        assert r['cache_hits']+r['new_compiles']==2
        private_bindings.append(dict(candidate_id=r['candidate_id'],index=r['index'],conditional=conditional,
            event_digest=r['event_digest'],result_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            fingerprints=r['fingerprints'],compiler=r['compiler']))
        return orders
    for cell in plan['cells']:
        rows=cal[cell['id']] if cell['role']=='CALIBRATION' else val['data'][cell['id']]
        m.validate_records(plan,cell,rows)
        if cell['saved']:saved+=len(rows);continue
        for r in rows:counts.update(raw_check(r,False));normal.append(r)
    assert len(normal)==32
    pairs=rare['pairs'];assert len(pairs)==4 and {(p['system'],p['pair']) for p in pairs}=={(s,i) for s in ('H4','H6') for i in (0,1)}
    forced=[]
    for pair in pairs:
        assert pair['unit']=='FORCED_LEGAL_PAIR_NOT_UNCONDITIONAL_COHORT' and pair['all_law_weighted_contribution'] is None
        a,b=pair['paths'];zero=raw_check(a,True);two=raw_check(b,True)
        assert zero.get(2,0)==0 and two[2]==1 and sum(zero.values())==sum(two.values())
        assert pair['compiled_difference']==[y-x for x,y in zip(a['C'],b['C'])]
        forced.extend([a,b])
    assert len(forced)==8 and sum(r['new_compiles'] for r in normal)<=64 and sum(r['new_compiles'] for r in forced)<=16
    for model in frozen['models'].values():assert model['rank']==3 and model['cell_level_lack_of_fit_DOF']==0 and model['qualified_for_candidate'] is False
    prices=read(RUN/'CALIBRATION_EVALUATION.json')['prices']+val['prices']
    assert len(prices)==32 and all(p['expectation']=='FINITE_COHORT_ONLY' and p['normalization_is_price_multiplier'] is False for p in prices)
    assert all(p['model'] is None or p['model']['qualified_for_candidate'] is False for p in prices)
    used=['src/trottertracks/pf_pr_rpe_pipeline.py','src/trottertracks/h4_df10_price_provider.py','src/trottertracks/h6_df19_price_provider.py','src/trottertracks/track_d_native_resources.py','scripts/h46_long_time_native.py']
    for name in used:assert (ROOT/name).read_bytes()==m.blob(ROOT,BASE,name)
    inputs={};cfg=read(plan['runtime']['input_config'])
    for system in ('H4','H6'):
        prior=m.p.SavedArtifacts(ROOT).read(m.ref(m.H4V1 if system=='H4' else m.H6V1,
            f'docs/research/{system.lower()}_df'+('10' if system=='H4' else '19')+'_price_v1_20261110/DEVELOPMENT.json'))
        for name,expected in prior['input_sha256'].items():
            actual=hashlib.sha256(Path(cfg['systems'][system]['inputs'][name]['path']).read_bytes()).hexdigest()
            assert actual==expected;inputs[system+':'+name]=actual
    out=dict(status='AUTHOR_SOURCE_AND_SCOPE_CHECKS_PASS',execution_source=plan['source'],frozen_model_commit=FROZEN,
        normal_new_paths=len(normal),forced_diagnostic_paths=len(forced),saved_C_S_paths_reused=saved,
        normal_new_compiles=sum(r['new_compiles'] for r in normal),forced_new_compiles=sum(r['new_compiles'] for r in forced),
        cache_hits=sum(r['cache_hits'] for r in normal+forced),normal_observed_event_orders=dict(counts),
        frozen_calibration_bytes_preserved=True,validation_refit=False,conditional_cohort_pooling=False,
        inputs_preserved=inputs,old_code_preserved=used,private_path_bindings=private_bindings,
        new_signal_N_G=0,SCF_DF_state_regeneration=0,full_wrapper_new_statevector_validation=False,
        independent_scientific_review=False,all_law_expected_price=None,joint_covariance=None,total_CI=None,full_rpe_resources=None,mandatory_stop=True)
    with (OUT/'CHECKS.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('private_path_bindings','inputs_preserved','old_code_preserved')}))


if __name__=='__main__':main()
