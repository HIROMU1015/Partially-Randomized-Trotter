"""Admit historical p7 receipts and project existing PR-DF evidence metadata.

No molecular input, circuit generator, sampler, compiler, fit or G entry.
Saved QPY objects are read only to check their measurement/scalar metadata.
Previously checked canonical hashes are reused only for unchanged QPY bytes.
"""
import argparse
import ast
from copy import deepcopy
import csv
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path

from trottertracks.r_price_phase1_science import functions

ROOT = Path(__file__).resolve().parents[1]
_spec=importlib.util.spec_from_file_location('m5_saved_receipt_helper',ROOT/'scripts/check_m5_m4_receipt_admission.py')
M=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(M)
PID = '13d1cd4869951cdf1e2c'
STAGED_D = 'c848bbefa30c23a9ee69170482fb02accae31601'
OLD_ADMISSION = '2fa88065318f0f2ab13937ef610828200b35874c'
OLD_DIR = M.OUTPUT
OUTPUT = 'docs/research/truncated_p7_admission_pr_df_evidence_20261114'
CACHE = Path('/home/AbeHiromu/projects/h46-long-time-pf-pr/20261025/compiled_cache')
FINGERPRINTS = dict(X='b0026b0eb54be07801a02a628e27f19302a93ba4c1cc9ddf8dbf0c91de5b11b6',
                    Y='8b523455c6b57793b1edf4cabf57f84e24d5777dca11ad606bced1edfc3389e5')
PR_EVIDENCE = dict(commit='3ae2e605dd6efcc63ed3742fe3076a8be3e6afae',
    path='docs/research/rpe_price_evidence_validation_20261113/COVERAGE.json')
INDEX = dict(commit='7a8bc7f132e166390c943e6b32c92cc6abe1baf4',
    path='artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json')


def revise_plan(old):
    """Exactly two action changes; old plan and every physical field preserved."""
    new = deepcopy(old); changed = []
    for i, row in enumerate(new):
        if row['route']=='TRUNCATED_PF' and row['physical_candidate']['id']==PID:
            if (row['m'],row['formula'],row['width_over_h'],row['features']['q'],row['role'],row['action']) != (
                    4,'4th(m5_best)','16',1,'CALIBRATION','NEW_COMPILE'):
                raise ValueError('Unexpected old saved p7 cell')
            row['action']='SAVED_RECEIPT_FOR_C_ACCEPTANCE'; changed.append(i)
    if len(changed)!=2 or {new[i]['axis'] for i in changed}!={'X','Y'}:
        raise ValueError('Exactly the saved p7 X/Y pair required')
    return new,changed


def wrapper_axis(qc, axis, scalar, T):
    """Exact source-defined suffix and scalar checks, without constructing gates."""
    control=8; cut=2 if axis=='X' else 3
    def word(item):
        return (item.operation.name,tuple(qc.find_bit(q).index for q in item.qubits),
                tuple(qc.find_bit(c).index for c in item.clbits))
    expected=[('h',(control,),()),('measure',(control,),(0,))]
    if axis=='Y': expected.insert(0,('sdg',(control,),()))
    if (qc.num_qubits,qc.num_clbits)!=(9,1) or word(qc.data[0])!=('h',(control,),()) or [word(x) for x in qc.data[-cut:]]!=expected:
        raise ValueError('Saved measured X/Y wrapper differs')
    phase=qc.data[-cut-1]
    if word(phase)!=('p',(control,),()) or float(phase.operation.params[0]).hex()!=float(-scalar*T).hex():
        raise ValueError('Saved B0I total scalar phase differs')
    # Source compose keeps every piece scalar=0 and applies p(-scalar*T) once.
    if sum(word(x)==('p',(control,),()) for x in qc.data)!=1:
        raise ValueError('Scalar not applied once at total T')
    return dict(axis=axis,control=control,measured_classical_bit=0,
                suffix=[x[0] for x in expected],scalar_angle_hex=float(phase.operation.params[0]).hex())


def inspect_wrappers(trial, checked):
    """Reuse canonical inspections only for the identical saved artifact bytes."""
    from qiskit import qpy
    out={}
    for axis,fp in FINGERPRINTS.items():
        ref=CACHE/(fp+'.json'); raw=ref.read_bytes(); receipt=json.loads(raw)
        wrapper=CACHE/(fp+'_wrapper.qpy.gz'); data=wrapper.read_bytes(); known=checked[axis]
        if (receipt['fingerprint']!=fp or receipt['compiler']!=M.document(ROOT,dict(commit=M.CARRIER,path=M.BASE+'scientific_provenance.json'))['compiler']['hash']
                or receipt['metrics']!=trial['prices']['cosine' if axis=='X' else 'sine']
                or known['fingerprint_saved']!=fp or known['fingerprint_loaded']!=fp or known['match'] is not True
                or hashlib.sha256(data).hexdigest()!=known['wrapper_sha256']):
            raise ValueError('Saved wrapper fingerprint/receipt mismatch: '+axis)
        qc=qpy.load(io.BytesIO(gzip.decompress(data)))[0]  # deserialize, never build a new physical circuit
        details=wrapper_axis(qc,axis,trial['scalar'],trial['candidate']['T'])
        compiled=CACHE/(fp+'_compiled.qpy.gz')
        out[axis]=dict(details,**known,receipt=dict(path=str(ref),sha256=hashlib.sha256(raw).hexdigest()),
            compiled_artifact=dict(path=str(compiled),sha256=hashlib.sha256(compiled.read_bytes()).hexdigest()),
            wrapper_path=str(wrapper),canonical_source=dict(commit=M.C_SOURCE,
                path='src/trottertracks/resource_applicability/ax2b_stream_fingerprint_v5.py'))
    if out['X']['common_body_digest']!=out['Y']['common_body_digest'] or FINGERPRINTS['X']==FINGERPRINTS['Y']:
        raise ValueError('X/Y must share the exact evolution word but distinct measured wrappers')
    return out


def canonical_wrapper_checks(cache=CACHE):
    """Reproduce the technical saved-QPY hash inspection, no gate construction.

    Not called by prepare: reuse the recorded check when bytes are unchanged.
    The body view is an iterator/metadata view, not a new QuantumCircuit.
    """
    from types import SimpleNamespace
    from qiskit import qpy
    from trottertracks.resource_applicability.ax2b_stream_fingerprint_v5 import _Writer,_HashSink,canonical_qiskit_circuit_fingerprint
    out={}
    for axis,fp in FINGERPRINTS.items():
        raw=(cache/(fp+'_wrapper.qpy.gz')).read_bytes();qc=qpy.load(io.BytesIO(gzip.decompress(raw)))[0]
        actual=canonical_qiskit_circuit_fingerprint(qc);cut=2 if axis=='X' else 3
        view=SimpleNamespace(parameters=qc.parameters,num_qubits=qc.num_qubits,num_clbits=qc.num_clbits,
            global_phase=qc.global_phase,data=qc.data[:-cut],find_bit=qc.find_bit)
        sink=_HashSink();_Writer(sink).circuit(view);sink.flush()
        out[axis]=dict(fingerprint_saved=fp,fingerprint_loaded=actual,match=fp==actual,
            wrapper_sha256=hashlib.sha256(raw).hexdigest(),common_body_digest=sink.hasher.hexdigest())
    return out


def prepare(root, wrapper_checks):
    t,f=M.d_callers(root); template=next(p for p in t.plans(root) if p['formula']=='4th(m5_best)')
    cell=next(c for c in template['cells'] if c['m']==4)
    carrier=json.loads(gzip.decompress(M.C.blob(root,M.CARRIER,M.BASE+'exploration_costs.json.gz')))[PID]
    raw=Path(carrier['trial_files'][0]).read_bytes();trial=json.loads(raw)
    table_raw=M.C.blob(root,M.CARRIER,M.BASE+'per_circuit_metrics.csv')
    table=[r for r in csv.DictReader(io.StringIO(table_raw.decode())) if r['id']==PID]
    if len(table)!=2 or {r['axis'] for r in table}!={'cosine','sine'}:raise ValueError('Original axis rows missing')
    for row in table:
        if any(Fraction(row[k])!=v for k,v in trial['prices'][row['axis']].items()):
            raise ValueError('Original CSV and trial metrics differ')
    # A local table-match record, NOT an assertion by the Full-PF-only D catalogue.
    observed=dict(physical_candidate=carrier['candidate'],prices=[dict(axes={
        axis:dict(status='EXACT_METADATA_MATCH',rz_count=Fraction(next(r for r in table if r['axis']==unit)['rz_count']))
        for axis,unit in (('X','cosine'),('Y','sine'))})])
    prov=json.loads(M.C.blob(root,M.CARRIER,M.BASE+'scientific_provenance.json'))
    context_ref=M.document(root,INDEX)['T20_reference_files']['H4']; context_raw=Path(context_ref['path']).read_bytes()
    if hashlib.sha256(context_raw).hexdigest()!=context_ref['sha256']:raise ValueError('Saved context differs')
    context=json.loads(context_raw); shared=M.document(root,M.MANIFEST)['shared']['full_h_preparation']
    total=shared['scalar']['constant']+context['prefixes']['7']['scalar']  # original binary64 metadata addition only
    if total.hex()!=float(2.293146549384713).hex():raise ValueError('Requested B0I scalar differs')
    config=template['plan']['configuration']; profile=config['spec']['profile']
    admission=M.inspect(trial,carrier,observed,cell['row'],prov,profile,total,method='B0I',prefix=7,pid=PID)
    for key,value in prov['inputs']['H4']['binding'].items():
        if config['spec']['binding'][key]!=value or context['binding'][key]!=value:raise ValueError('H/state binding differs: '+key)
    previous=M.document(root,dict(commit=OLD_ADMISSION,path=OLD_DIR+'/ADMISSION.json'))
    # Reuse the prior source-route proof; check only the changed D stage's callers.
    changes={}
    for name in ('plans','price_bindings','checked_trial'):
        path='scripts/run_truncated_p7_prices.py'
        def node(commit):return ast.dump(next(n for n in ast.parse(M.C.blob(root,commit,path)).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
        changes[name]=node(M.D_SOURCE)==node(STAGED_D)
    if not all(changes.values()):raise ValueError('D stage changed physical/record route')
    wrappers=inspect_wrappers(trial,wrapper_checks)
    source=dict(commit=M.CARRIER,path=M.BASE+'exploration_costs.json.gz',key=PID,
        original_trial=dict(path=carrier['trial_files'][0],sha256=hashlib.sha256(raw).hexdigest()),
        original_table=dict(commit=M.CARRIER,path=M.BASE+'per_circuit_metrics.csv',id=PID),
        original_axis_receipts={a:w['receipt'] for a,w in wrappers.items()})
    t.price_bindings.__globals__['COHORT']=trial['cohort']
    prices,sides,proofs=t.price_bindings(root,template['plan'],cell,trial,price_source=source,
        evidence_commit='UNPUBLISHED_RECEIPT_BUNDLE',evidence_path=OUTPUT+'/IMPLEMENTATION_EVIDENCE.json')
    for price,side in zip(prices,sides):
        axis=price['axis'];price['route']='TRUNCATED_PF'
        price['diagnostic']['compiled_wrapper']=deepcopy(wrappers[axis])
        side['implementation'].update(route='TRUNCATED_PF',original_wrapper_fingerprint=FINGERPRINTS[axis])
        key=side['implementation_evidence']['selector'][0]
        proofs[key]['implementation']=deepcopy(side['implementation'])
    records=f.fit_inputs(prices,route='TRUNCATED_PF',formula=template['formula'],width_over_h=template['width_over_h'])
    old=M.document(root,dict(commit=OLD_ADMISSION,path=OLD_DIR+'/CELL_PLAN.json'));new,changed=revise_plan(old)
    admission.update(source=source,configuration_id=config['id'],physical_candidate=cell['row'],profile=profile,
        original_checks=trial['checks'],wrapper_checks=wrappers,saved_tail_scalar=context['prefixes']['7']['scalar'],
        scalar_context=dict(index=INDEX,context=context_ref),retained_blocks=8,fragment_order=list(range(7)),
        source_proof_reused=dict(commit=OLD_ADMISSION,path=OLD_DIR+'/ADMISSION.json',selector=['native_source_checks']),
        staged_D_same_functions=changes,comparison_policy_adopted=False,model_fit=None,G=None)
    return dict(admission=admission,prices=prices,sides=sides,proofs=proofs,fit_records=records,
        cells=new,delta=dict(old_source=dict(commit=OLD_ADMISSION,path=OLD_DIR+'/CELL_PLAN.json'),
            changed_indices=changed,changed_field='action',new_compile=32,new_calibration=20,new_holdout=12,saved=4),
        raw=raw,original_table=table,plan=template['plan'])


def pr_coverage(root):
    """Reuse existing metadata coverage; do not touch S/C values or predictors."""
    old=M.document(root,PR_EVIDENCE);out=[]
    p1_ref=dict(commit='39ddd59edf54a3a901e5302207b2765ddbd03fbb',
        path='docs/research/d_p1_efficient_czz_20261113/run02/RESULT.json')
    p1=M.document(root,p1_ref)
    for i,row in enumerate(old['coverage']):
        if row['label'] not in ('PR-LD1','PR-LD2'):continue
        r=deepcopy(row)
        if r['label']=='PR-LD1':
            for j,s in enumerate(p1['summary']):
                if s['m']==r['m']:
                    r['finite_cohort'].append(dict(source='P1',selector=['summary',j],axis=r['axis'],
                        cohort=p1['cohort'],n_paths=s['n_paths'],expectation=s['expectation'],
                        consumer_role='REFERENCE_DEVELOPMENT_NOT_NEW_HOLDOUT',
                        lowering_id='CONTROLLED_ZZ_CX_CRZ_CX_V1',consumer_bridge='PENDING_NO_ADMISSION',
                        status='SAVED_PAIRED_FINITE_COHORT_ONLY'))
        r.update(reused_source=dict(PR_EVIDENCE,selector=['coverage',i]),
            saved_direct_path_price=bool(r['finite_cohort']),finite_cohort_available=bool(r['finite_cohort']),
            structural_expectation_available=bool(r['structural_mean']),residual_model_in_current_domain=False,
            qualified_compiled_expectation_available=False,unqualified_model_implementation=True,
            model_prediction=None,G_paper_tier=('FINITE_COHORT_SCENARIO_PENDING_BINDING' if r['finite_cohort'] else 'MISSING_COMPILED_PRICE'),
            level2_qualified=False,unknown=['LAW_DISTANCE','RARE_CONTRIBUTION','TRANSFER_ERROR','TOTAL_CI'])
        out.append(r)
    if len(out)!=52 or {(r['label'],r['m'],r['axis']) for r in out}!={(l,m,a) for l in ('PR-LD1','PR-LD2') for m in range(13) for a in ('X','Y')}:
        raise ValueError('Incomplete fixed PR coverage')
    return dict(status='REUSED_STATIC_EVIDENCE_NO_NEW_S_C_ANALYSIS',source=PR_EVIDENCE,
        sources=dict(old['sources'],P1=p1_ref),coverage=out,price_means_recomputed=False,fit_or_prediction=False,G=None)


def c1_catalog(root):
    ref=dict(commit='2e6a0cdf5a647058cc7afeeb5741bfafa24c974b',
             path='docs/research/r_price_phase1_results_20261111/run01/PLAN.json')
    plan=M.document(root,ref);out=[]
    for i,cell in enumerate(plan['cells']):
        if cell['system']!='H4':continue
        identity=cell['request']['identity'];c=identity['candidate']
        for axis in ('X','Y'):
            data='CALIBRATION.json' if cell['role']=='CALIBRATION' else 'VALIDATION.json'
            selector=[cell['id']] if data=='CALIBRATION.json' else ['data',cell['id']]
            out.append(dict(cell_id=cell['id'],physical_candidate=deepcopy(c),m=cell['m'],axis=axis,
                role=cell['role'],cohort=cell['cohort'],n_paths=cell['n'],
                plan_source=dict(ref,selector=['cells',i]),
                path_prices_source=dict(commit=ref['commit'],path=ref['path'].replace('PLAN.json',data),selector=selector),
                original_lowering='CONTROLLED_ZZ_GENERIC_RZZ_CONTROL1_PROVIDER_C31EF424',
                full_measured_wrapper_profile=deepcopy(identity['profile']),
                expectation='FINITE_COHORT_ONLY',qualified_for_candidate=False,
                active_LD1_LD2_schedule_match=False,price_values_not_projected=True))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wrapper-checks',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();p=prepare(ROOT,json.loads(args.wrapper_checks.read_text()))
    args.output.mkdir(parents=True,exist_ok=False)
    for file,key in (('ADMISSION.json','admission'),('PRICES.json','prices'),('SIDECARS_UNBOUND.json','sides'),
        ('IMPLEMENTATION_EVIDENCE.json','proofs'),('SAVED_CALIBRATION_INPUTS.json','fit_records'),
        ('CELL_PLAN_V2.json','cells'),('CELL_PLAN_DELTA.json','delta'),('ORIGINAL_TABLE_ROWS.json','original_table')):
        (args.output/file).write_text(json.dumps(p[key],indent=2)+'\n')
    (args.output/'ORIGINAL_TRIAL.json').write_bytes(p['raw'])
    (args.output/'SAVED_WRAPPER_CHECKS.json').write_text(json.dumps(p['admission']['wrapper_checks'],indent=2)+'\n')
    (args.output/'PR_DF_COVERAGE.json').write_text(json.dumps(pr_coverage(ROOT),indent=2)+'\n')
    (args.output/'PR_DF_C1_CATALOG.json').write_text(json.dumps(c1_catalog(ROOT),indent=2)+'\n')
    print('p7 saved receipts admitted; 4 saved + 20 new calibration + 12 holdout. No science calls.')


if __name__=='__main__':main()
