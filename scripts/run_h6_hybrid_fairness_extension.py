#!/usr/bin/env python3
"""Fresh H6 fixed-four workers. Reuse old kernels, dispatch and price_job."""
import importlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.phase-a-numeric'))
OUT=ROOT/'artifacts/h6_hybrid_fairness_extension/2026-10-27'
PREP=ROOT/'analysis/h6_hybrid_fairness_readiness_20261027'
def module(name,path):
    s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
R=module('hybrid_original_price','scripts/pauli_h2_h6_study.py');R.OUT=OUT
D=module('hybrid_original_dispatch','scripts/run_pauli_rte_event_extension.py');D.OUT=OUT
E=importlib.import_module('hx_study.h6_hybrid_fairness_extension')
F=importlib.import_module('hx_study.h6_rte_fairness')
read,save,sha=R.read,R.save,R.sha
BATCH=OUT/'H6/batch01'
def gitbytes(ref,path):return subprocess.check_output(['git','show',ref+':'+path],cwd=ROOT)
def check():
    if (OUT/'STUDY_STOP.json').exists():raise ValueError('COMPLETED_STUDY_NO_RESUME')
    s=read(OUT/'SOURCE.json')
    for p,h in {**s['code_sha256'],**s['input_sha256']}.items():
        if sha(ROOT/p)!=h:raise ValueError('SCIENTIFIC_SOURCE_OR_INPUT_CHANGED '+p)
    return s
def model():return read(OUT/'H6/inputs/model.json')
def ham(m):
    from hx_study.pauli_signal_adapter import PauliHamiltonian
    h=m['hamiltonian'];return PauliHamiltonian(m['n_qubits'],h['constant'],tuple(map(tuple,h['terms'])),json.dumps(h.get('metadata',{})))
def case(cid):E.signature(cid);return BATCH/'cases'/cid
def initialize():
    import importlib.metadata as md
    if OUT.exists():raise ValueError('FRESH_OUTPUT_REQUIRED')
    OUT.mkdir(parents=True);(OUT/'H6/inputs').mkdir(parents=True)
    p=read(PREP/'SOURCE_INPUTS.json');inputs={}
    for item in p['inputs']:
        path=OUT/'H6/inputs'/Path(item['path']).name;path.write_bytes(gitbytes(item['ref'],item['path']))
        assert sha(path)==item['sha256'];inputs[str(path.relative_to(ROOT))]=sha(path)
    E.bind_saved_inputs((OUT/'H6/inputs/model.json').read_bytes(),(OUT/'H6/inputs/reference.json').read_bytes())
    contract=dict(candidates=list(E.CANDIDATES),target_max=4,law_max=4,scalar_bits=128,vector_bits=53,
                  target_per_candidate=1,new_Hamiltonian_state_reference=0,automatic_retry=False,
                  price_before_design=False,confirmation_top_up=False,mandatory_STOP=True,
                  preparation_commit='bcc2eb6b201e1434bf47560775f5afb3e14d3ca8')
    save(OUT/'target_contract.json',contract);inputs[str((OUT/'target_contract.json').relative_to(ROOT))]=sha(OUT/'target_contract.json')
    codes=list(dict.fromkeys(R.CODE+['src/trotterlib/h6_hybrid_fairness_extension.py','src/trotterlib/pauli_vector_tight_enclosure.py','src/trotterlib/h6_rte_fairness.py','scripts/run_pauli_rte_event_extension.py','scripts/run_h6_hybrid_fairness_extension.py']))
    source=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),code_sha256={p:sha(ROOT/p) for p in codes},input_sha256=inputs,
                model_sha256=sha(OUT/'H6/inputs/model.json'),reference_sha256=sha(OUT/'H6/inputs/reference.json'),study_plan_sha256=sha(OUT/'target_contract.json'))
    save(OUT/'SOURCE.json',source)
    versions={p:md.version(p) for p in ['numpy','scipy','qiskit','python-flint']}
    assert versions==dict(numpy='1.26.4',scipy='1.14.1',qiskit='1.3.0',**{'python-flint':'0.9.0'})
    save(OUT/'runtime.json',versions)
    (OUT/'.gitignore').write_text('/H6/\n/stages/\n/cache/\n/*.tar.xz\n')
    BATCH.mkdir(parents=True);candidates=[];h=ham(model())
    for cid,(prefix,q,r,K) in E.CANDIDATES.items():
        cfg=dict(method='PR',L_D=prefix,q=q,r=r,K=K,T=.8)
        d=case(cid);d.mkdir(parents=True);law=E.law_descriptor(h,cid)
        c=dict(candidate_id=cid,molecule=6,config=cfg,law=law)
        save(d/'spec.json',c);candidates.append(c)
    save(BATCH/'plan.json',dict(molecule=6,candidates=candidates,SOURCE=source))
    print('FOUR_LAWS_SAVED_FRESH_SOURCE',flush=True)
def target(cid):
    check();d=case(cid)
    with (d/'target_call_started.json').open('x') as f:json.dump(dict(max_state_actions=1),f)
    m=model();ref=read(OUT/'H6/inputs/reference.json');law=read(d/'spec.json')['law']
    result=E.mean_target(m,ref,law,cid)
    assert result['raw_norm_exact']==ref['raw_norm_exact'] and result['B_applied_count']==1 and not result['state_renormalized']
    save(d/'target.json',result)
    print(cid,'TARGET_SAVED',flush=True)
def targets():
    check();jobs=[([R.COMP,str(Path(__file__).resolve()),'target',cid],cid) for cid in E.CANDIDATES]
    D.dispatch(jobs,OUT/'stages/targets');print('FOUR_TARGETS_NATIVE_EXIT0_REAPED',flush=True)
def price(cid,cohort,index):
    check();case(cid);assert cohort in ['exploration','confirmation']
    plan=read(OUT/('sampling_design.json' if cohort=='exploration' else 'confirmation_plan.json'))
    n=plan['n_E'] if cohort=='exploration' else plan['candidates'][cid]['n_C']
    assert cid in plan['candidates'] and 0<=index<n
    if not any(R.A.joint_status(read(case(cid)/'target.json'),e)=='CONDITIONAL_GUARD_ELIGIBLE' for e in R.EPS):raise ValueError('NO_ELIGIBLE_SIGNAL')
    R.price_job(BATCH,cid,cohort,index)
def prices(cohort):
    check();p=read(OUT/('sampling_design.json' if cohort=='exploration' else 'confirmation_plan.json'));jobs=[]
    for cid in p['candidates']:
        n=p['n_E'] if cohort=='exploration' else p['candidates'][cid]['n_C']
        for index in range(n):jobs.append(([R.COMP,str(Path(__file__).resolve()),'price',cid,cohort,str(index)],cid+f'_{index:05d}'))
    D.dispatch(jobs,OUT/'stages'/cohort);print(cohort,len(jobs),'TRAJECTORIES_EXIT0_REAPED',flush=True)
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='init':initialize()
    elif mode=='targets':targets()
    elif mode=='target':target(sys.argv[2])
    elif mode=='price':price(sys.argv[2],sys.argv[3],int(sys.argv[4]))
    elif mode in ['exploration','confirmation']:prices(mode)
    else:raise ValueError('FIXED_WORKER_MODE')
