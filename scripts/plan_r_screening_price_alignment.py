"""Align fixed A/C experiment metadata. No normalization, N/C model or G calls."""
import argparse
import ast
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import pf_pr_rpe_pipeline as p
from trottertracks import rpe_r_selection as r
from trottertracks import rpe_provider_integration as b

A='f98982118324bcc8402ed513b5c720b37bb03e87'
C='8457c443554af5fadaa3ba36f97f6316ca3c5050'
B='ac3e1ce87b5ed7d9114abc316545366dd097209e'
AP='docs/research/pr_r_screening_20261111/'
CP='docs/research/r_variable_compiled_price_design_20261111/'


def blob(root,commit,path):return subprocess.check_output(['git','show',commit+':'+path],cwd=root)


def phase1_specs(text):
    rows=re.findall(r'^\| (H[46]) DF\d+ \| (\d+) / (\d+) \| (\d+) / (\d+) \|',text,re.M)
    specs=[(s,*map(int,values)) for s,*values in rows]
    if specs!=[('H4',8,16,4,8),('H6',1,2,16,32)]:raise ValueError('C fixed Phase1 rectangle differs; do not substitute candidates')
    return specs


def fixed_C_describer(root):
    path='src/trottertracks/r_variable_compiled_price.py';raw=blob(root,C,path)
    node=next(v for v in ast.parse(raw).body if isinstance(v,ast.FunctionDef) and v.name=='describe_request')
    scope=dict(deepcopy=deepcopy,hashlib=hashlib,json=json)
    exec(compile(ast.Module(body=[node],type_ignores=[]),C+':'+path,'exec'),scope)
    return scope['describe_request']


def validate_A(packet):
    menu=packet['menu'];p.validate_task(menu)
    if packet['R_upper_bound'] is not None or packet['range_complete'] or packet['scientific_R_adopted']:
        raise ValueError('A finite open-range proposal required')
    for row in packet['rows']:
        target=next(v for v in menu['candidates'] if v['candidate']['id']==row['candidate_id'])
        expected=r.request_context(menu,target)
        if not p.same(row['request'],expected) or not p.same(row['structure'],target['structure']):
            raise ValueError('A full recipe/identity/law differs')
        if any(row[k]!=target['candidate'][ck] for k,ck in [('T','T'),('q_PF','q'),('R_outer','R_outer'),('delta','delta'),('prefix','prefix'),('K','K'),('formula','formula')]):
            raise ValueError('A descriptor/candidate differs')
        if row['q_RPE']!=target['q_RPE'] or row['m']!=target['m'] or row['system']!=target['candidate']['system'] or not row['detailed_N_required']:
            raise ValueError('Retain all A candidates as detailed-N-needed')


def align(packets, aplan, ccoverage, fixed_coverage, specs, describe):
    """Pure recipe/record metadata join; never inspect stored price/N values."""
    for packet in packets.values():validate_A(packet)
    all_A={row['candidate_id']:row for packet in packets.values() for row in packet['rows']}
    norm=aplan['phase0']['candidates'];norm_ids={v['candidate_id'] for v in norm}
    if len(norm)!=24 or len(norm_ids)!=24 or not norm_ids<=all_A.keys():raise ValueError('A fixed normalization24 scope differs')
    for row in norm:
        source=all_A[row['candidate_id']]
        if any(row[k]!=source[k] for k in row):raise ValueError('Normalization descriptor differs from A menu')
    n_pilot=aplan['phase1_candidate_example']['candidates']
    if not {v['candidate_id'] for v in n_pilot}<=norm_ids:raise ValueError('A minimum N pilot must retain its normalization provenance')
    for row in n_pilot:
        if any(row[k]!=all_A[row['candidate_id']][k] for k in row):raise ValueError('A minimum N pilot descriptor differs')
    saved_N={v['candidate']['id']:v for system in fixed_coverage['systems'].values() for v in system['rows']}
    cells=[]
    for system,qL,qH,R0,R1 in specs:
        packet=packets[system];base=packet['base']
        original={row['candidate_id']:row for row in packet['rows']}
        by_q={row['q_PF']:row for row in base['candidates'] if p.arm(row['candidate'])=='PR' and row['q_PF'] in (qL,qH)
              and row['candidate']['delta']==(.1 if system=='H4' else 1.6)}
        if set(by_q)!={qL,qH}:raise ValueError('C q rectangle absent from fixed N recipe')
        menu=r.build_menu(base,{row['m']:[R0,R1] for row in by_q.values()},sampling_law=packet['menu']['sampling_law'])
        bmenu=r.build_menu(base,{row['m']:[R0,R1] for row in by_q.values()},sampling_law=b.LAW)
        for q,R,role,npaths in [(qL,R0,'SAVED_CALIBRATION',9),(qH,R0,'SAVED_CALIBRATION',33 if system=='H4' else 8),
                                (qL,R1,'NEW_CALIBRATION',8),(qH,R1,'INDEPENDENT_VALIDATION',8)]:
            row=next(v for v in menu['candidates'] if v['m']==by_q[q]['m'] and v['q_PF']==q and v['candidate']['R_outer']==R)
            cid=row['candidate']['id'];request=r.request_context(menu,row)
            descriptions=[describe(menu,row,axis,pipeline=p) for axis in p.AXES]
            if any(not p.same(v['identity'],request['identity']) or v['allocation']!=row['structure']['allocation'] for v in descriptions):
                raise ValueError('C metadata recipe differs from A/B')
            if cid in original and not p.same(original[cid]['request'],request):raise ValueError('A/C matching ID has different conditions')
            brow=next(v for v in bmenu['candidates'] if v['candidate']['id']==cid)
            breq=r.request_context(bmenu,brow)
            if not p.same(breq['identity'],request['identity']):raise ValueError('B/A physical identity differs')
            evidence=[deepcopy(v) for v in ccoverage['fixed_provider_evidence'] if v['candidate']['id']==cid]
            if any(not p.same(v['candidate'],row['candidate']) for v in evidence):raise ValueError('Saved C candidate differs')
            old_N=saved_N.get(cid)
            if old_N and not p.same(old_N['candidate'],row['candidate']):raise ValueError('Saved N candidate differs')
            if (role=='SAVED_CALIBRATION')!=bool(evidence):raise ValueError('Saved/new C role conflicts with fixed metadata corpus')
            calibration=[v for v in evidence if v['classification']=='DIRECT_COMPILED_FINITE_SAMPLE' and v['stored_n_paths']==npaths]
            if role=='SAVED_CALIBRATION' and len(calibration)!=1:raise ValueError('Explicit C calibration cohort is ambiguous')
            cells.append(dict(system=system,m=row['m'],candidate=deepcopy(row['candidate']),candidate_id=cid,
                q_RPE=row['q_RPE'],q_PF=q,structure=deepcopy(row['structure']),profile=deepcopy(menu['profile']),
                binding=deepcopy(menu['binding']),role=role,planned_price_paths=npaths,
                in_A_auto_menu=cid in original,normalization_in_A24=cid in norm_ids,
                intake_lane='A_AUTO_MENU' if cid in original else 'EXPLICIT_C_CALIBRATION_VALIDATION_ONLY',
                A_request_law_binding=request['law_binding'],B_request_law_binding=breq['law_binding'],
                law_binding_raw_equal=p.same(request['law_binding'],breq['law_binding']),
                actual_sampler_transfer_qualified=False,
                N_saved=old_N is not None,N_status=None if old_N is None else old_N['N_status'],
                N_source=None if old_N is None else old_N['N_source'],
                same_N_allowance_policy_for_R_comparison='NOT_ESTABLISHED_REQUIRES_SEPARATELY_PERMITTED_SCALAR_REANALYSIS',
                C_saved_metadata=evidence,C_selected_calibration_metadata=calibration,price_values_exported=False,
                paired_C_S_linkage='VERIFY_EXISTING_PATH_BINDING_BEFORE_FIT_NO_RECOUNT_PERFORMED',
                old_R0_validation_not_a_new_holdout=role=='SAVED_CALIBRATION',
                new_N_required_for_C_price_pilot=False,new_N_required_for_same_point_G=old_N is None,
                C_join_keys={v['axis']:v['join_key'] for v in descriptions},
                long_q_transfer_qualified=False,full_law_expected_price_qualified=False))
    cids={v['candidate_id'] for v in cells};nids={v['candidate_id'] for v in n_pilot}
    return dict(C_phase1_cells=cells,A_normalization24=deepcopy(norm),A_minimum_N_pilot=deepcopy(aplan['phase1_candidate_example']),
        sets=dict(A_auto_PR=sum(len(v['rows']) for v in packets.values()),C_cells=len(cells),
            C_in_A_auto=sum(v['in_A_auto_menu'] for v in cells),C_explicit_outside_A=sum(not v['in_A_auto_menu'] for v in cells),
            A24_C_overlap=sorted(norm_ids&cids),A_N_pilot_C_overlap=sorted(nids&cids),
            A_N_pilot_is_subset_of_A24=nids<=norm_ids),
        C_execution_units=dict(new_whole_paths=32,axes_per_path=2,ordinary_wrappers_max=64,
            rare_diagnostic_pairs_per_system=2,rare_diagnostic_wrappers_max=16,all_wrappers_max=80,
            rare_paths_pooled_into_unconditional_cohort=False),
        A_screen_scope_automatically_extended=False,A_comparison_menu_automatically_reduced=False,
        R_upper_bound=None,range_complete=False,scientific_R_adopted=False)


def build(root=ROOT):
    store=p.SavedArtifacts(root)
    packets={s:store.read(dict(commit=A,path=AP+s+'_METADATA_PROPOSAL.json')) for s in ('H4','H6')}
    aplan=store.read(dict(commit=A,path=AP+'EXECUTION_PLAN.json'))
    cov=store.read(dict(commit=C,path=CP+'SAVED_PRICE_COVERAGE.json'))
    csource=store.read(dict(commit=C,path=CP+'SOURCE_INDEX.json'))
    fixed=store.read(dict(commit=B,path='docs/research/rpe_provider_integration_20261111/fixed_coverage/COVERAGE.json'))
    text=blob(root,C,CP+'MINIMAL_SCIENTIFIC_PLAN.md').decode()
    design=blob(root,C,CP+'MODEL_DESIGN.md').decode()
    if p.PROFILE['compiler'] not in design:raise ValueError('C compiler profile not established')
    for packet in packets.values():
        if not p.same(packet['menu']['profile'],p.PROFILE):raise ValueError('A/C compiler profile differs')
    result=align(packets,aplan,cov,fixed,phase1_specs(text),fixed_C_describer(root))
    law=csource['finite_law'];A_law=packets['H4']['menu']['sampling_law']['source']
    same=blob(root,law['commit'],law['path'])==blob(root,A_law['commit'],A_law['path'])
    result.update(status='METADATA_ALIGNMENT_AND_UNEXECUTED_PLAN',sources=dict(A=A,B=B,C=C),
        law_definition=dict(A_source=A_law,C_source=law,implementation_bytes_equal=same,
            raw_binding_alias_automatically_applied=False,actual_finite_sampler_distance_qualified=False,
            intake='Retain A declaration; match to B request_context after sourced-law evidence. Do not relabel actual sampling law.'),
        new_normalization_N_C_reanalysis_fit_sampling_compile_G=0,total_resources=None,ranking=None,mandatory_stop=True)
    if not same:raise ValueError('A/C finite-law definition differs; explicit investigation required')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True)
    out=Path(parser.parse_args().output_dir)
    if out.exists():raise FileExistsError('Use a fresh technical metadata output directory')
    result=build();out.mkdir(parents=True)
    (out/'ALIGNMENT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    fields=['system','m','T','q_PF','delta','R_outer','candidate_id','role','planned_price_paths','in_A_auto_menu','normalization_in_A24','N_saved','intake_lane']
    with (out/'C_PHASE1_CANDIDATES.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fields,lineterminator='\n');writer.writeheader()
        for row in result['C_phase1_cells']:writer.writerow({key:row['candidate'][key] if key in ('T','delta','R_outer') else row[key] for key in fields})
    print(json.dumps(dict(status=result['status'],C_cells=8,A_normalization_candidates=24,max_C_wrappers=80,new_scientific_calls=0)))


if __name__=='__main__':main()
