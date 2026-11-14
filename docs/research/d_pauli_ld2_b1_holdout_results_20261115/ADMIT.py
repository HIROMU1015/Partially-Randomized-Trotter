"""Reuse fixed D2 receipt export and canonical implementation guard; no simulation/fit."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
import run_pauli_b1_prices as P
BASE=Path(__file__).resolve().parent
CIDS=('M5_ROUND','M6_ROUND')
AXES={'X':'cosine','Y':'sine'}

def read(path):return json.loads(path.read_text())
def write(path,data):P.write(path,data)
def raw(commit,path):return json.loads(P.B0.blob(commit,str(path.relative_to(ROOT))))
def export_receipts():
    outcomes=[]
    for cid in CIDS:
        out=BASE/'run01'/cid
        job=read(BASE/'JOBS'/f'{cid}.json')
        try:
            result=read(out/'RESULT.json')
            paths=(result.get('child') or {}).get('paths',[])
            packet=P.canonical(2)[1]
            receipts=P.receipts().compile_receipts(paths,result['source'],packet['dictionary_identity'],job['private'])
            write(out/'COMPILE_RECEIPTS.json',receipts)
            outcomes.append(dict(cell_id=cid,status='EXPORTED',receipts=len(receipts)))
        except Exception as exc:
            outcomes.append(dict(cell_id=cid,status='INCOMPLETE_OR_INVALID',reason=type(exc).__name__+': '+str(exc)))
    write(BASE/'RECEIPT_EXPORT_STATUS.json',outcomes)
    print(json.dumps(outcomes))

def admit(commit):
    pre=read(BASE/'PREOPEN.json');units=[];round_checks=[]
    expected_dict=P.canonical(2)[1]['dictionary_identity']
    for cid in CIDS:
        out=BASE/'run01'/cid; job=read(BASE/'JOBS'/f'{cid}.json');plan=job['plan']
        cell=next(c for c in plan['cells'] if c['cell_id']==cid)
        request=P.request_for(plan,cell)[0];seeds=job['step_seeds'];m=request['m']
        def loc(name,selector):return dict(commit=commit,path=str((out/name).relative_to(ROOT)),selector=selector)
        global_reasons=[]; data={}; receipts={}; gates=None
        try:
            data=raw(commit,out/'RESULT.json');receipts=raw(commit,out/'COMPILE_RECEIPTS.json')
            source=data['source']
            if (source['commit']!=pre['execution_source'] or source['cell']!=cid or source['cohort']!=cell['cohort'] or
                source['step_seeds']!=seeds or source['prefix']!=2 or source['kind']!='B1-CAL' or
                source['caller']!=P.CALLER or source['result_path']!=str((out/'RESULT.json').relative_to(ROOT))):
                raise ValueError('Execution SOURCE/cell/cohort/seed/result binding differs')
            if (data['status']!='FINITE_CELL_COMPLETE' or data['child']['status']!='PASS' or
                len(data['child']['paths'])!=cell['n_paths']):
                global_reasons.append('INCOMPLETE_CELL')
            for name in P.REPLAY_SOURCES:
                if P.B0.blob(source['commit'],name)!=P.B0.blob(pre['implementation_source'],name):
                    raise ValueError('Fixed implementation SOURCE differs: '+name)
            gates=P.check_round_records(plan,cell,raw(commit,out/'PRICES.json'),raw(commit,out/'SIDECARS_UNBOUND.json'),evidence_commit=commit)
            write(out/'C_GUARD_CHECK.json',gates)
            write(out/'SIDECARS_PUBLISHED.json',gates['bound_sidecars'])
            round_checks.append(dict(cell_id=cid,check=str((out/'C_GUARD_CHECK.json').relative_to(ROOT)),status=gates['status']))
        except Exception as exc:
            global_reasons.append(type(exc).__name__+': '+str(exc))
            round_checks.append(dict(cell_id=cid,status='INCOMPLETE_OR_INVALID',reasons=global_reasons.copy()))
        saved=(data.get('child') or {}).get('paths',[])
        for axis,key in AXES.items():
            unit=dict(m=m,axis=axis,cell_id=cid,canonical_physical_id=request['candidate_id'],paths=[],reasons=global_reasons.copy())
            for i in range(cell['n_paths']):
                row=dict(index=i,axis=axis,valid=False,compiled_RZ=None,reason=None,
                    path=loc('RESULT.json',['child','paths',i]),receipt=loc('COMPILE_RECEIPTS.json',[f'p{i}{axis}']),
                    implementation_evidence=loc('IMPLEMENTATION_EVIDENCE.json',[f'm{m}{axis}']),
                    execution_commit=pre['execution_source'],canonical_physical_id=request['candidate_id'],cohort=cell['cohort'],seeds=seeds[i])
                try:
                    if global_reasons:raise ValueError('CELL_GATE_OR_SOURCE_NOT_ADMITTED')
                    matches=[p for p in saved if p['index']==i]
                    if len(matches)!=1:raise ValueError('MISSING_OR_DUPLICATE_PATH')
                    path=matches[0]
                    P.receipts().check_path(path,request=request,dictionary=expected_dict,cohort=cell['cohort'],index=i,seeds=seeds[i])
                    value=path['result'];receipt=receipts[f'p{i}{axis}'];rec=receipt['record'];fp=value['fingerprints'][key]['fingerprint']
                    expected={'cell_id':cid,'axis':axis,'index':i,'cohort':cell['cohort'],'seeds':seeds[i],
                              'request':request,'dictionary_identity':expected_dict,'execution_source':data['source']}
                    if any(receipt.get(k)!=v for k,v in expected.items()):raise ValueError('Individual receipt identity differs')
                    if (rec!=value['compile_receipts'][key] or rec['metrics']!=value['prices'][key] or
                        rec['fingerprint']!=fp or rec['compiler']!=cell['profile']['compiler']):
                        raise ValueError('Individual metrics/fingerprint/compiler receipt differs')
                    if value['sampling']!='ORIGINAL_NORMAL_SAMPLER':raise ValueError('Independent fresh normal sampler required')
                    if value['scalar']!=P.B0.connection().total_scalar(P.canonical(2)[2],P.B0.HAMILTONIAN_CONSTANT):
                        raise ValueError('Once-at-total-T scalar quantity differs')
                    price=rec['metrics']['rz_count']
                    if type(price) is not int or price<0:raise ValueError('Nonnegative integer RZ price required')
                    row.update(valid=True,compiled_RZ=price,fingerprint=fp,cache_reused=value['fingerprints'][key]['cache_reused'],
                               action_gate='PASS',implementation_gate='CHECKED_NOT_MODEL_QUALIFICATION',checks=value['checks'])
                except Exception as exc:row['reason']=type(exc).__name__+': '+str(exc)
                unit['paths'].append(row)
            units.append(unit)
    result=dict(status='ALL_SIX_RECEIPTS_ADMITTED' if all(p['valid'] for u in units for p in u['paths']) else 'INCOMPLETE_OR_INVALID',
        raw_publication_commit=commit,implementation_source=pre['implementation_source'],execution_source=pre['execution_source'],
        units=units,canonical_round_checks=round_checks,model_verdict='NOT_ASSESSED_NO_ADOPTED_TOLERANCE',fit_run=False)
    write(BASE/'RECEIPT_ADMISSION.json',result)
    print(json.dumps({'status':result['status'],'valid_receipts':sum(p['valid'] for u in units for p in u['paths']),'round_checks':round_checks}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-only',action='store_true');parser.add_argument('--publication')
    args=parser.parse_args()
    if args.export_only:export_receipts()
    else:admit(args.publication)
