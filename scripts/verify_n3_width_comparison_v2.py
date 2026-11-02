#!/usr/bin/env python3
"""Independent saved proof verification. Does not import science producer or solvers."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.representation_exploration.n3_width_contract_v2 import active_data,digest,rat,CAPS,DELTA_TIERS
from trottertracks.representation_exploration.n3_width_check_v2 import verify_bounds,Checks,verify_full_lower,verify_witness,verify_sdp_refutation,verify_endpoints


def verify_record(case,record,source_fingerprint=None):
    if record['case_sha256']!=digest(case) or record['input_id']!=case['id'] or record['mandatory_stop'] is not True or record['next_stage_authorized'] is not False:
        raise ValueError('SAVED_CASE_IDENTITY_OR_STOP')
    if source_fingerprint is not None and record.get('source_fingerprint')!=source_fingerprint:raise ValueError('SAVED_SOURCE_BINDING')
    counts={'certificates':0,'refutations':0,'PSD_checks':0,'operations':0}
    blocks=record.get('blocks',record.get('completed_blocks',[]))
    if 'partial_block' in record:blocks=blocks+[record['partial_block']]
    for block in blocks:
        for method,path in block['methods'].items():
            if path['method']!=method:raise ValueError('METHOD_NAME_BINDING')
            for row in path['mode_records']:
                d=row['external']
                if row['method']!=method:raise ValueError('MODE_METHOD_BINDING')
                maximum={'NORM':1,'FIVE':5,'OLD':4,'OCC':4,'SDP_EQUIVALENT':3,'DENSITY_SOS':3,'FULL_H_LOWER':1}[method]
                if len(row['candidates'])>maximum or len(row['refutations'])>3:raise ValueError('PROPOSAL_COUNT_CONTRACT')
                stats=row.get('resources',path.get('resources'))
                if stats is not None:
                    for key,cap in [('eigh',CAPS['eigh_per_mode']),('solver_calls',CAPS['solver_calls_per_mode']),('PSD_probes',CAPS['PSD_probes_per_mode'])]:
                        value=stats['calls'][key]
                        if type(value) is not int or not 0<=value<=cap:raise ValueError('PRIMITIVE_CAP')
                if row['bounds'] is not None:
                    U,c=verify_bounds(case,d,row['bounds'],Checks())
                    if row['status']=='BOUND_PRECONDITION_FAILED' and c>U:raise ValueError('FALSE_BOUND_PRECONDITION_FAILURE')
                for w in row['candidates']:
                    expected_kind={'NORM':'NORM','FIVE':'OLD','OLD':'OLD','OCC':'OCC','SDP_EQUIVALENT':'OCC','DENSITY_SOS':'SOS','FULL_H_LOWER':'FULL_H_LOWER'}[method]
                    if w['kind']!=expected_kind:raise ValueError('METHOD_CERTIFICATE_ATTRIBUTION')
                    if w['kind']=='FULL_H_LOWER':
                        if w['case_sha256']!=digest(case) or w['external']!=d:raise ValueError('FULL_LOWER_BINDING')
                        audit=verify_full_lower(case,d,w['upper'],w['full_lower'])
                        width=rat(w['full_lower']['width'])
                        if rat(w['width']['high'])!=width or rat(w['width']['low'])!=width or rat(w['width']['positive_proof_delta'])<=0 or rat(w['width']['positive_proof_delta'])<width:raise ValueError('FULL_WIDTH')
                    else:audit=verify_witness(case,d,row['bounds'],w)
                    counts['certificates']+=1;counts['PSD_checks']+=audit['PSD_checks'];counts['operations']+=audit['operations']
                for ref in row['refutations']:
                    _,A,v,J,E,hdd,nu,eps=active_data(case,d);U,c=verify_bounds(case,d,row['bounds'],Checks())
                    if method!='SDP_EQUIVALENT' or rat(ref['delta']) not in [eps*tier for tier in DELTA_TIERS]:raise ValueError('REFUTATION_SCOPE')
                    verify_sdp_refutation(A,v,nu,U,c,rat(ref['delta']),ref['certificate']);counts['refutations']+=1
        if 'endpoints' in block:verify_endpoints(case,block['methods'],block['endpoints'])
    return {'status':'SAVED_EXACT_PROOFS_PASS','input_id':case['id'],**counts,'new_solver_or_producer_calls':0}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--pack',type=Path,required=True);a=p.parse_args()
    from trottertracks.representation_exploration.n3_width_execution_v2 import preflight,file_sha
    plan_path=ROOT/'docs/research/n3_width_comparison_v2_execution_plan_20261012.json';plan=json.loads(plan_path.read_bytes());preflight(ROOT,plan)
    if file_sha(a.pack)!=plan['input_sha256']:raise ValueError('SAVED_PACK_IDENTITY')
    launch=json.loads((a.directory/'launch_receipt.json').read_bytes())
    if launch['plan_sha256']!=file_sha(plan_path) or launch['source_fingerprint']!=plan['source_fingerprint'] or launch['input_sha256']!=plan['input_sha256']:raise ValueError('LAUNCH_LINEAGE')
    cases=json.loads(a.pack.read_bytes())['cases'];rows=[]
    for case in cases:
        record=json.loads((a.directory/(case['id']+'.json')).read_bytes());rows.append(verify_record(case,record,plan['source_fingerprint']))
    print(json.dumps({'status':'INDEPENDENT_SAVED_COMPARISON_PROOFS_PASS','cases':rows,'mandatory_stop':True},indent=2))


if __name__=='__main__':main()
