"""One fixed normal m0 path, two measured wrappers, empirical per-shot references."""
from pathlib import Path
from datetime import datetime,timezone
from unittest.mock import patch
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_full_h_kappa2_p2 as P
import full_h_p2_acceleration as A
from complete_full_h_p2_checkpoint import observe_child

CALLER='scripts/run_full_h_m0_reference.py'
STUDY='docs/research/d_full_h_m0_reference_20261115'
FROZEN=STUDY+'/FROZEN_JOB.json'
RULE=dict(commit='4a1cb0d7364fece475437749129fdce740fd8fc4',path='docs/research/full_h_m0_price_rule_20261115/PROPOSAL.json')
ADMISSION=dict(commit='61f249e6e18d5afab9ba8797bef5e8f9f14e29de',path='docs/research/full_h_d7_holdout_admission_plan_20261115/ADMISSION.json')
GATES={'low_control','high_control_leakage','independent_occupation_vector','Hadamard_cosine',
    'Hadamard_sine','compiled_vector_cosine','compiled_vector_sine'}


def contract(job):
    proposal=json.loads(P.P1.blob(RULE['commit'],RULE['path']))
    admission=json.loads(P.P1.blob(ADMISSION['commit'],ADMISSION['path']))
    acquisition=proposal['proposed_acquisition'];row=proposal['physical_candidate']
    if (job['rule_source']!=RULE or job['admission_source']!=ADMISSION
            or job['rule_id']!=proposal['rule_id'] or job['row']!=row
            or job['cohort']!=acquisition['cohort'] or job['index']!=0
            or job['seeds']!=[15612141344395026302] or job['seeds']!=[acquisition['seed_u64']]
            or job['maximum_compiles']!=2 or job['independent_paths']!=1
            or job['profile']!=proposal['profile'] or job['implementation']!=proposal['implementation']
            or row['id']!='FULL_H_2a9a68a11d08af234a8c'
            or (row['T'],row['q'],row['R_outer'],row['K'],row['prefix'])!=(.1,1,9,2,0)
            or P.P1.seed_function()(row,job['cohort'],0,0,0)!=job['seeds'][0]):
        raise ValueError('Fixed m0 scope/physical/profile/cohort/seed differs')
    for i,axis in enumerate(('X','Y')):
        unit=proposal['units'][i];fixed=admission['rows'][i]
        if (unit['axis']!=axis or fixed['axis']!=axis or fixed['physical_candidate']!=row
                or unit['admission']!=dict(ADMISSION,selector=['rows',i])
                or any(unit[k]!=fixed[k] for k in ('physical_id','m','T_exact','original_configuration_id',
                    'total_configuration_id','original_candidate_id','total_candidate_id'))):
            raise ValueError('C admission axis/canonical binding differs')
    return proposal


def context_check(proposal,full,ctx,private):
    prep=ctx.preparations[0];saved=json.loads((Path(private)/'H4_context.json').read_text())
    if (prep.preparation_hash!=proposal['target_preparation_hash']
            or saved['preparation_hash']!=prep.preparation_hash
            or any(saved['binding'][k]!=v for k,v in proposal['H_state_snapshot'].items())
            or saved['binding']['geometry']!='linear_1Angstrom' or saved['binding']['basis']!='STO-3G'
            or saved['binding']['rank']!=10 or saved['binding']['rank_policy']!='tol_only_1e-8'
            or saved['constant']!=proposal['scalar']['constant'] or saved['extracted_scalar']!=proposal['scalar']['extracted_identity']
            or ctx.ham.constant+prep.extracted_identity_coefficient!=proposal['scalar']['total']
            or list(prep.randomized_block_indices)!=list(range(10)) or saved['deterministic_blocks']!=0):
        raise ValueError('D7 preparation/H/state/scalar/component order differs')
    return dict(status='PASS',saved_context=saved,preparation_source=P.P1.FULL_H,
        component_order=[c.component_id for c in prep.rte_preparation.symbolic_tail.components],
        component_order_binding='D7 preparation hash and unchanged prepare_full_h SOURCE',
        compiler_profile=proposal['profile'],implementation=proposal['implementation'])


def prices(job,path,source):
    trial=path['trial'];checks=path['checks'];proposal=contract(job)
    if (path['row']!=job['row'] or path['cohort']!=job['cohort'] or path['index']!=0 or path['seeds']!=job['seeds']
            or trial['candidate']!=job['row'] or trial['cohort']!=job['cohort'] or trial['index']!=0 or trial['status']!='PASS'
            or trial['preparation_hash']!=proposal['target_preparation_hash'] or trial['scalar']!=proposal['scalar']['total']
            or {g['check'] for g in trial['checks']}!=GATES
            or any(not g['error']<g['tolerance'] or g['phase_alignment'] for g in trial['checks'])
            or checks['parity']['status']!='PASS' or checks['scalar_event_phases']['status']!='PASS'
            or checks['builder_lowering_id']!=P.P1.LOWERING_ID or path['xy_phase_gate']['status']!='PASS'
            or path['sampling']!='ORIGINAL_NORMAL_SAMPLER' or path.get('parents')
            or len(path['events'])!=1 or len(path['events'][0])!=9
            or len(set(path['fingerprints'].values()))!=2):raise ValueError('Required m0 path/gates/fingerprints differ')
    rows=[]
    for i,(axis,a) in enumerate((('X','cosine'),('Y','sine'))):
        receipt=path['compile_receipts'][a];record=receipt['record'];value=record['metrics']['rz_count']
        if (record['compiler']!=job['profile']['compiler'] or record['fingerprint']!=path['fingerprints'][a]
                or receipt['fingerprint']!=record['fingerprint'] or record['metrics']!=trial['prices'][a]
                or type(value) is not int or value<0 or receipt['reused'] or not receipt['locator']['sha256']):
            raise ValueError('Original axis compiler record/price differs')
        rows.append(dict(rule_id=job['rule_id'],axis=axis,m=0,physical_id=job['row']['id'],physical=job['row'],
            per_shot_RZ=value,symbol='C_0'+axis,path_index=0,cohort=job['cohort'],seeds=job['seeds'],
            independent_paths=1,shared_path_for_X_Y=True,kind='DIRECT_COMPILED_FINITE_SAMPLE',
            evidence_kind='SINGLE_PATH_EMPIRICAL_REFERENCE',expectation='FINITE_COHORT_ONLY',
            shot_multiplier_applied=False,RTE_normalizer_price_multiplier=False,
            measurement=dict(axis=axis,native_axis=a,include_measurement=True,state_preparation_cost_included=False),
            fingerprint=record['fingerprint'],compile_receipt=receipt,source=source,
            admission=proposal['units'][i]['admission'],canonical_binding={k:proposal['units'][i][k] for k in
                ('original_configuration_id','total_configuration_id','original_candidate_id','total_candidate_id','B_join')},
            acceptance='D1_REQUIRED_GATES_AND_BINDING_PASS',C_scientific_admission='NOT_PERFORMED',
            law_expectation_qualified=False,E_C=None,CI=None))
    return rows


def child(filename):
    job=json.loads(Path(filename).read_text());private=Path(job['private']);started=time.monotonic();draws=0;result={}
    with (private/'PROGRESS.jsonl').open('x') as log:
        def progress(stage):
            log.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),stage=stage,
                elapsed_seconds=time.monotonic()-started))+'\n');log.flush()
        try:
            proposal=contract(job)
            import h46_long_time_signal as S
            S.install();P.P1.provider_checks();full=P.P1.full_h_module(job['output'],job['private'])
            progress('restore_fixed_H4_context');ctx=full.context('H4')
            result['context_check']=context_check(proposal,full,ctx,private)
            native=full.setup();original=native.sample_requests
            def once(*args,**kwargs):
                nonlocal draws
                if draws:raise ValueError('No second normal path draw permitted')
                draws+=1;return original(*args,**kwargs)
            with patch.object(native,'sample_requests',once):
                result['path']=A.accelerated_path(P,full,job['row'],job['cohort'],0,job['seeds'],progress=progress)
            if draws!=1:raise ValueError('Exactly one normal path required')
            result.update(status='PASS',normal_sampler_calls=draws)
            P.P1.write(private/'child.json',result);progress('COMPLETE');return 0
        except BaseException:
            result.update(status='STOP',normal_sampler_calls=draws,error=traceback.format_exc())
            P.P1.write(private/'child.json',result);progress('STOP');return 2


def execute(job,source):
    contract(job);out=(ROOT/job['output']).resolve();private=Path(job['private'])
    if not out.is_relative_to(ROOT) or out.exists() or private.exists():raise ValueError('Fresh isolated m0 output required')
    P.P1.source_checks();P.P1.child_import_preflight(ROOT)
    out.mkdir(parents=True);private.mkdir(parents=True)
    runtime=dict(job,output=str(out),private=str(private));P.P1.write(private/'job.json',runtime)
    code=observe_child([sys.executable,'-B',str(ROOT/CALLER),'--child',str(private/'job.json')],cwd=ROOT,private=private)
    result=json.loads((private/'child.json').read_text()) if (private/'child.json').exists() else None
    provenance=dict(commit=source,caller=CALLER,result_path=job['output']+'/RESULT.json',rule_id=job['rule_id'])
    error=None;rows=[]
    try:
        if code or result is None or result['status']!='PASS':raise ValueError('Child did not pass required path checks')
        rows=prices(job,result['path'],provenance)
        for row in rows:
            receipt=row['compile_receipt'];raw=Path(receipt['locator']['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=receipt['locator']['sha256'] or json.loads(raw)!=receipt['record']:
                raise ValueError('Saved original compiler bytes differ')
            p=out/'COMPILE_RECORDS'/('p0_'+row['axis']+'.json');p.parent.mkdir(exist_ok=True);p.write_bytes(raw)
    except Exception:error=traceback.format_exc();rows=[]
    ok=bool(rows) and error is None
    if not ok:rows=[dict(axis=axis,m=0,physical_id=job['row']['id'],per_shot_RZ=None,
        acceptance='HELD_NULL_REQUIRED_EVIDENCE_INCOMPLETE',reason=error,source=provenance) for axis in ('X','Y')]
    P.P1.write(out/'PRICES.json',rows)
    saved_records=sorted((private/'compiled_cache').glob('*.json'))
    if not ok:
        for record in saved_records:
            destination=out/'UNQUALIFIED_COMPILE_RECORDS'/record.name;destination.parent.mkdir(exist_ok=True)
            shutil.copy2(record,destination)
    P.P1.write(out/'RESULT.json',dict(status='M0_REFERENCE_COMPLETE' if ok else 'STOP',source=provenance,
        child=result,error=error,child_exit_code=code,child_exit_signal=-code if code<0 else None,
        independent_paths=1,maximum_compiles=2,saved_compile_records=len(saved_records),
        per_shot_only=True,E_C=None,CI=None,mandatory_stop=True))
    native=private/'cost'/job['cohort']/job['row']['id']/'0'
    if native.exists():shutil.copytree(native,out/'NATIVE_TRIAL')
    if (private/'H4_context.json').exists():shutil.copy2(private/'H4_context.json',out/'H4_context.json')
    return 0 if ok else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute-fixed-job');parser.add_argument('--child');args=parser.parse_args()
    if args.child:return child(args.child)
    if not args.execute_fixed_job:parser.error('Fixed m0 job required')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path in (CALLER,'scripts/run_full_h_kappa2_p2.py','scripts/full_h_p2_acceleration.py','scripts/complete_full_h_p2_checkpoint.py'):
        if (ROOT/path).read_bytes()!=P.P1.blob(source,path):raise ValueError('m0 execution SOURCE differs from committed files')
    job=json.loads(Path(args.execute_fixed_job).read_text())
    if job!=json.loads(P.P1.blob(source,FROZEN)):raise ValueError('m0 input differs from fixed SOURCE')
    return execute(job,source)


if __name__=='__main__':sys.exit(main())
