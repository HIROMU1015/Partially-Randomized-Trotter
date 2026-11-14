"""Metadata-only admission of the fixed 12 m3 saved deterministic PF prices.

Reuse D's physical-equivalence evidence and C's planner/implementation guard.
No circuit, array, prediction, fit, price averaging or cost evaluation entry.
"""
import argparse
from copy import deepcopy
import csv
from functools import lru_cache
import gzip
import importlib.util
import io
import json
from pathlib import Path
import subprocess

from trottertracks import baseline_signal_pipeline as C
from trottertracks import price_implementation_guard as G

ROOT=Path(__file__).resolve().parents[1]
_spec=importlib.util.spec_from_file_location('_m3_existing_receipt_reader',ROOT/'scripts/check_m5_m4_receipt_admission.py')
M=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(M)
BASE='docs/research/pf_m3_saved_price_admission_20261115'
D=dict(commit='bcb67b2b2bb78776db1dd8d0625a3facb37cd298',path='docs/research/d_pf_allround_price_coverage_plan_20261115/COVERAGE_DELTA.json')
ALIGN=dict(commit='0b4b2d941cd669aaa45231e1236bc2d3158a9d12',path='docs/research/h4_five_method_candidate_alignment_20261115/ALIGNMENT.json')
CARRIER=dict(commit='155529e20a8fac39e00786c40b57830b71203c4f',path=M.BASE+'reused_compiled_prices.json.gz')
PFC_BINDING=dict(commit='0b1886f53f7c3a5e99e0c13cb5f7090581c5e38c',path='artifacts/resource_applicability/track_a_pf_common_H4_results/2026-10-22/source_input_identity.json')
PLACEHOLDER='UNPUBLISHED_M3_METADATA_BUNDLE'
ROLE='SAVED_M3_DIRECT_OBSERVATION_NOT_CALIBRATION_OR_HOLDOUT'


@lru_cache(maxsize=32)
def document(root,commit,path):
    raw=C.blob(root,commit,path)
    return json.loads(gzip.decompress(raw) if path.endswith('.gz') else raw)


def read(root,ref):
    out=document(str(root),ref['commit'],ref['path'])
    for k in ref.get('selector',[]):out=out[k]
    return deepcopy(out)


def at(ref,*keys):return dict(ref,selector=list(ref.get('selector',[]))+list(keys))


def primary_axis(root,saved,axis):
    """Original per-axis metadata; do not reopen the M5 private trial/QPY."""
    o=saved['origin'];name={'X':'cosine','Y':'sine'}[axis]
    if 'public_path' in o:
        rows=list(csv.DictReader(io.StringIO(C.blob(root,o['public_commit'],o['public_path']).decode())))
        index=next(i for i,r in enumerate(rows) if r['candidate_id']==o['legacy_id'] and r['axis']==name)
        receipt=rows[index]
        if receipt['cost_type']!='DIRECT_MEASURED_NEW_RUN01' or receipt['SOURCE']!=saved['generation_SOURCE']:
            raise ValueError('M5 receipt is not the original direct measured price')
        return dict(metrics={k:int(receipt[k]) for k in ('rz_count','cx_count','total_depth','rz_depth','cx_depth','circuit_size')},
            fingerprint=receipt['fingerprint'],compiler=receipt['compiler_settings_hash'],
            preparation_hash=receipt['preparation_hash'],original_cohort=None,
            original_candidate=read(root,at(CARRIER,saved['physical_id']))['candidate'],
            legacy_canonical_id=None,legacy_trial_id=o['legacy_id'],generation_SOURCE=receipt['SOURCE'],
            original_native_policy=receipt['native_policy'],
            result_sha256=receipt['result_sha256'],
            receipt=dict(commit=o['public_commit'],path=o['public_path'],
                         csv_selector=dict(candidate_id=o['legacy_id'],axis=name),csv_data_row=index),
            original_trial=o,original_receipt_sha256=receipt['cost_receipt_sha256'],
            scalar=saved['saved_total_scalar'],scalar_basis='D_FIXED_PRIMARY_TRIAL_HASH_AND_SCALAR_CHECK',
            scientific_binding_basis=at(D,'saved_m3',4))
    trial=read(root,o)
    if trial['SOURCE']!=saved['generation_SOURCE'] or trial['status']!='PASS':
        raise ValueError('Original generation SOURCE/status differs')
    if 'prices' in trial:
        r=trial['prices'][name]
        if not all(x['PASS'] for x in trial['checks']):raise ValueError('Original saved action gate failed')
        receipt=at(o,'prices',name)
        binding=read(root,PFC_BINDING)['binding']
        legacy=trial.get('identity',{}).get('candidate',{}).get('id')
        original=trial['candidate']
    else:
        ri=next(i for i,x in enumerate(trial['records']) if x['axis']==name);r=trial['records'][ri]
        receipt=at(o,'records',ri);binding=trial['binding'];legacy=None;original=None
    return dict(metrics=r['metrics'],fingerprint=r['fingerprint'],
        compiler=r.get('compiler',r.get('compiler_settings_hash')),preparation_hash=trial['preparation_hash'],
        original_cohort=trial.get('cohort'),original_candidate=original,
        legacy_canonical_id=legacy,legacy_trial_id=trial.get('task'),generation_SOURCE=trial['SOURCE'],
        original_native_policy=trial['native_policy'],
        result_sha256=o.get('result_sha256',o.get('sha256')),receipt=receipt,original_trial=o,
        scalar=trial['scalar'],binding=binding,
        scalar_basis='ORIGINAL_TRIAL_SCALAR',scientific_binding_basis=PFC_BINDING if 'prices' in trial else o)


def check_axis(row,saved,origin,context,scalar):
    """Only identity/metadata equality and existing PASS evidence, no reanalysis."""
    physical=context['physical_candidate'];axis=row['axis'];receipt=saved['axes'][axis]
    if row['m']!=3 or row['physical_id']!=saved['physical_id'] or row['physical_candidate']!=physical:
        raise ValueError('ALIGNMENT/current physical candidate or round differs')
    G._same((physical['T'],physical['q'],physical['delta'],physical['formula'],physical['prefix']),
            (row['physical_candidate']['T'],row['physical_candidate']['q'],row['physical_candidate']['delta'],
             row['physical_candidate']['formula'],row['physical_candidate']['prefix']),'Actual schedule differs')
    if origin['metrics']['rz_count']!=receipt['rz_count'] or origin['fingerprint']!=receipt['fingerprint']:
        raise ValueError('Original axis price/fingerprint differs from fixed D evidence')
    if origin['preparation_hash']!=saved['preparation_hash'] or origin['scalar']!=saved['saved_total_scalar']:
        raise ValueError('Original preparation/total scalar differs')
    if origin['scalar']!=scalar or origin['compiler']!=context['profile']['compiler']:
        raise ValueError('C expected scalar/compiler differs')
    if origin['generation_SOURCE']!=saved['generation_SOURCE']:
        raise ValueError('Original generation SOURCE differs')
    if origin.get('binding'):
        for k,v in origin['binding'].items():G._same(v,context['binding'][k],'H/state/snapshot binding differs: '+k)
    if origin.get('original_candidate'):
        for k,v in origin['original_candidate'].items():G._same(v,physical[k],'Original candidate differs: '+k)
    return 'SOURCE_AND_PHYSICAL_AXIS_METADATA_MATCH'


def axis_binding(root,plan,row,origin,route):
    """Thin binding in the existing D/C price/sidecar schema, per axis."""
    m=3;axis=row['axis'];req=next(r for r in M.PN.I.price_requests(root,plan) if r['m']==m and r['axis']==axis)
    profile=plan['configuration']['spec']['profile'];gen=origin['generation_SOURCE']
    native=dict(commit=gen,path='scripts/h46_m5_native.py')
    # Legacy caller commits use a separately installed provider tree. D already
    # checked the measured wrapper's equivalence; keep that reference distinct
    # from the original caller SOURCE rather than inventing a file there.
    wrapper=dict(commit='af284d853890155bacd19e61850c196bd1d5c660',
                 path='src/trottertracks/resource_applicability/ax2b_native_df_v5.py')
    source=dict(**origin['receipt'],generation_SOURCE=gen,original_trial=origin['original_trial'])
    price=deepcopy(req)
    price.update(source=source,cohort=origin['original_cohort'],n_paths=1,
        mean=dict(rz_count=origin['metrics']['rz_count']),metrics=deepcopy(origin['metrics']),
        classification='DIRECT_COMPILED_DETERMINISTIC',expectation='DETERMINISTIC',
        price_status='DIRECT_DETERMINISTIC_AVAILABLE',record_schema='D_PRICE_RECORD_B_KINDS_V2',
        diagnostic=dict(record_role=ROLE,model_group=None,model_qualified=False,new_compile=False),
        price_profile=dict(profile,lowering_id='H46_LONG_TIME_NATIVE_COMPOSE'),
        profile_extension=dict(lowering_id='H46_LONG_TIME_NATIVE_COMPOSE',source=native),
        normalization_is_price_multiplier=False)
    facts=dict(method=route,profile=deepcopy(profile),native_policy='H46_LONG_TIME_NATIVE_COMPOSE',
        rte_rotation_lowering=G.NO_RTE,rte_lowering_source=None,native_source=native,wrapper_source=wrapper,
        wrapper_source_role='D_CHECKED_SOURCE_EQUIVALENT_REFERENCE_NOT_ORIGINAL_CALLER_COMMIT',
        wrapper_equivalence_source=D,original_generation_SOURCE=gen,
        original_native_policy=origin['original_native_policy'],
        scalar_source=dict(native,selector='saved original total scalar; applied once after signed native compose'),
        scalar_quantity='HAMILTONIAN_CONSTANT' if route=='FULL_PF' else 'HAMILTONIAN_CONSTANT_PLUS_B0I_TAIL_IDENTITY',
        scalar_application='ONCE_AT_TOTAL_T',measurement=M.PN.I.providers(str(root)).B.p.measurement(plan['configuration']['rounds'][m],axis),
        resolved_scalar=origin['scalar'],expected_scalar=origin['scalar'])
    key=row['configuration_id']+':m3:'+axis
    side=dict(identity=deepcopy(req['identity']),law_binding=deepcopy(req['law_binding']),axis=axis,
        cohort=origin['original_cohort'],price_source=source,physical_context=deepcopy(req['physical_context']),
        profile_extension=deepcopy(price['profile_extension']),implementation=facts,
        implementation_evidence=dict(commit=PLACEHOLDER,path=BASE+'/IMPLEMENTATION_EVIDENCE.json',selector=[key]))
    proof={k:deepcopy(side[k]) for k in ('price_source','identity','law_binding','axis','cohort','physical_context','profile_extension','implementation')}
    return price,side,key,proof


def existing36(root):
    """Index already accepted Stage1/2 receipts, without numerical rejudgement."""
    index_ref=dict(commit='b190bd815c886446291fac8b281c3c7df7343372',path='docs/research/d_pf_stage1_calibration_20261115/C_HANDOFF_INDEX.json')
    hold_ref=dict(commit='a55813373b84b9128cbd10c6812c0d26143a4280',path='docs/research/d_pf_stage2_holdout_20261115/run01/RESULT.json')
    cells=M.document(root,dict(commit='43c15bfa34a22f313e55b824223b200332b7f23f',path='docs/research/truncated_p7_admission_pr_df_evidence_20261114/CELL_PLAN_V2.json'))
    rows=[]
    for group,g in read(root,index_ref)['groups'].items():
        for i,e in enumerate(g['entries']):
            cell=next(c for c in cells if c['model_group']==g['model_group'] and c['m']==e['m'])
            receipt=e.get('price_reference') or e['references']['PRICES.json']
            rows.append(dict(configuration_id=cell['configuration_id'],physical_id=cell['physical_candidate']['id'],
                m=e['m'],axis=cell['axis'],role='CALIBRATION',receipt=receipt,
                existing_acceptance_source=at(index_ref,'groups',group,'entries',i)))
    for i,e in enumerate(read(root,hold_ref)['rows']):
        cell=next(c for c in cells if '|'.join(c['model_group'])==e['model_group'] and c['m']==6)
        G._same(cell['physical_candidate'],e['physical_candidate'],'Existing m6 physical receipt differs from CELL_PLAN')
        rows.append(dict(configuration_id=cell['configuration_id'],physical_id=cell['physical_candidate']['id'],m=6,
            axis=e['axis'],role='HOLDOUT',receipt=e['receipt'],existing_acceptance_source=at(hold_ref,'rows',i),
            existing_reported_decision=e['decision'],new_judgement=False))
    if len(rows)!=36 or len({(r['configuration_id'],r['m'],r['axis']) for r in rows})!=36:
        raise ValueError('Existing 36-point reference set incomplete or duplicated')
    return rows


def prepare(root):
    delta=read(root,D);alignment=read(root,ALIGN)
    if delta['status']!='METADATA_ONLY_PASS' or delta['passed_checks']!=len(delta['checks']):
        raise ValueError('D physical-equivalence checks incomplete')
    t,f=M.d_callers(root);templates={p['configuration_id']:p for p in f.plans(root)+t.plans(root)}
    frozen=read(root,dict(commit='12372fa6b53c878dc5dd124e036572ba0cd82061',path='docs/research/pf_stage1_model_freeze_20261115/run01/FROZEN_MODELS.json'))
    rows=[];prices=[];sides=[];proofs={}
    for si,saved in enumerate(delta['saved_m3']):
        for ai in saved['alignment_rows']:
            row=alignment['rows'][ai];axis=row['axis'];plan=templates[row['configuration_id']]['plan']
            context=M.PN.I.price_context(plan,3)
            route='FULL_PF' if context['physical_candidate']['method']=='FULL' else 'TRUNCATED_PF'
            entry=dict(alignment_source=at(ALIGN,'rows',ai),configuration_id=row['configuration_id'],
                physical_id=row['physical_id'],physical_candidate=row['physical_candidate'],axis=axis,m=3,route=route,
                canonical_id=plan['configuration']['rounds'][3]['candidate']['id'],model_group=None,role=ROLE,
                D_equivalence_source=at(D,'saved_m3',si),status='PENDING_C_GUARD',missing=[])
            try:
                G._same(row['original_canonical_id'],entry['canonical_id'],'ALIGNMENT/current canonical identity differs')
                origin=primary_axis(root,saved,axis)
                scalar=next(m for m in frozen['models'].values() if m['input_locators'][0]['configuration_id']==row['configuration_id'])['input_locators'][0]['scalar']['value']
                check_axis(row,saved,origin,context,scalar)
                price,side,key,proof=axis_binding(root,plan,row,origin,route)
                prices.append(price);sides.append(side);proofs[key]=proof
                entry.update(origin=origin,price_source=price['source'],original_cohort=origin['original_cohort'],
                    legacy_canonical_id=origin['legacy_canonical_id'],legacy_trial_id=origin['legacy_trial_id'],
                    origin_cohort_note='PRESERVED_NULL_WHEN_NOT_RECORDED_IN_ORIGINAL_DETERMINISTIC_TRIAL',
                    saved_RZ=origin['metrics']['rz_count'],guard_key=key)
            except (ValueError,KeyError,StopIteration) as exc:
                entry.update(status='HELD_MISSING_OR_CONTRADICTORY_EVIDENCE',missing=[type(exc).__name__+': '+str(exc)])
            rows.append(entry)
    if len(rows)!=12 or len({(x['configuration_id'],x['axis']) for x in rows})!=12:
        raise ValueError('Expected the fixed six m3 candidates x two axes')
    return dict(rows=rows,prices=prices,sides=sides,proofs=proofs,templates=templates,prior36=existing36(root))


def bind(root,packet,bundle_commit):
    """Existing guard per axis. Failure holds that axis, without substitution."""
    rows=deepcopy(packet['rows']);prices=[];sides=[];checks=[]
    for row in rows:
        if row['status']!='PENDING_C_GUARD':continue
        key=row['guard_key'];side=next(deepcopy(s) for s in packet['sides'] if s['implementation_evidence']['selector']==[key])
        price=next(p for p in packet['prices'] if p['identity']==side['identity'] and p['axis']==side['axis'])
        side['implementation_evidence']['commit']=bundle_commit
        config=packet['templates'][row['configuration_id']]['plan']['configuration'];route=row['route']
        policy=dict(id='C_SAVED_M3_METADATA_COMPATIBILITY_ONLY',source=D,profile=config['spec']['profile'],
            by_method={route:dict(native_policy='H46_LONG_TIME_NATIVE_COMPOSE',rte_rotation_lowering=G.NO_RTE,
                scalar_quantity=side['implementation']['scalar_quantity'])})
        try:
            check=G.check(root,config,[price],cohorts={(row['canonical_id'],row['axis']):row['original_cohort']},bindings=[side],policy=policy)
            if len(check['checked'])!=1:raise ValueError('Guard did not check this axis')
            tier=M.PN._tier(C.published_api(str(root)).B,price,route)
            if tier!='DETERMINISTIC_DIRECT':raise ValueError('Existing price consumer tier is not deterministic direct')
            row['status']='ACCEPTED_SAVED_DIRECT_POINT';row['price_tier']=tier
            row['bound_evidence']=side['implementation_evidence'];prices.append(price);sides.append(side);checks.append(check)
        except (ValueError,KeyError,subprocess.CalledProcessError) as exc:
            row.update(status='HELD_C_GUARD',missing=[type(exc).__name__+': '+str(exc)])
    accepted=sum(r['status']=='ACCEPTED_SAVED_DIRECT_POINT' for r in rows)
    calibration=sum(x['role']=='CALIBRATION' for x in packet['prior36']);holdouts=sum(x['role']=='HOLDOUT' for x in packet['prior36'])
    return dict(rows=rows,prices=prices,sides=sides,guard=checks,counts=dict(m3_requested=12,m3_accepted=accepted,
        m3_held=12-accepted,unchanged_m4_m5_calibration=calibration,unchanged_m6_holdout=holdouts,total_accepted_direct_axes=len(packet['prior36'])+accepted),
        previous36_sources=dict(stage1=read(root,D)['stage1_source'],stage2=read(root,D)['stage2_source']),
        previous36_references=BASE+'/PREVIOUS36_REFERENCES.json',
        role=ROLE,model_refit=False,predictions=0,compile_calls=0,N=None,G=None,winner=None,mandatory_stop=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--bind',metavar='BUNDLE_COMMIT');args=ap.parse_args()
    out=ROOT/BASE;packet=prepare(ROOT);out.mkdir(parents=True,exist_ok=True)
    if args.bind:
        if (out/'ADMISSION.json').exists():raise ValueError('Preserve existing admission result')
        result=bind(ROOT,packet,args.bind)
        for filename,key in (('ADMISSION.json',None),('PRICES.json','prices'),('SIDECARS.json','sides'),('GUARD_CHECK.json','guard')):
            (out/filename).write_text(json.dumps(result if key is None else result[key],indent=2)+'\n')
        with (out/'ADMISSION.csv').open('w',newline='') as stream:
            w=csv.writer(stream,lineterminator='\n');w.writerow(['configuration_id','physical_id','axis','canonical_id','origin_locator','status','missing'])
            for r in result['rows']:w.writerow([r['configuration_id'],r['physical_id'],r['axis'],r['canonical_id'],json.dumps(r.get('price_source'),sort_keys=True),r['status'],';'.join(r['missing'])])
        print(json.dumps(result['counts']))
    else:
        if (out/'IMPLEMENTATION_EVIDENCE.json').exists():raise ValueError('Preserve existing metadata bundle')
        for filename,value in (('PREPARED_ROWS.json',packet['rows']),('PRICES_UNBOUND.json',packet['prices']),
                ('SIDECARS_UNBOUND.json',packet['sides']),('IMPLEMENTATION_EVIDENCE.json',packet['proofs']),
                ('PREVIOUS36_REFERENCES.json',packet['prior36'])):
            (out/filename).write_text(json.dumps(value,indent=2)+'\n')
        print(json.dumps(dict(axes=len(packet['rows']),pending=sum(r['status']=='PENDING_C_GUARD' for r in packet['rows']),scientific_calls=0)))


if __name__=='__main__':main()
