#!/usr/bin/env python3
"""Fixed Git blob/identity/literal audit, NOT probability or statistics evaluation.

Does not import molecular modules or evaluate saved cost, probability, SE, ratio,
shot or sensitivity formulas. Floating JSON numbers are transcribed as strings.
Only metadata identities/counts/seed derivation and literal equality are tested.
"""
import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
COMPILER='1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
PROFILE='622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12'


def require(ok,reason):
    if not ok:raise ValueError(reason)


def git(*args):
    return subprocess.check_output(['git','--no-optional-locks','-C',str(ROOT),*args],stderr=subprocess.PIPE)


def literal_constants(source):
    out={}
    for node in ast.parse(source).body:
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant):
            for name in node.targets:
                if isinstance(name,ast.Name):out[name.id]=node.value.value
    return out


def audit():
    manifest=json.loads((ROOT/'references/h8_rare_tail_selection_inputs_v1.json').read_text())
    raw={};catalog=[]
    for r in manifest['inputs']:
        require(len(r['commit'])==40,'EXACT_COMMIT')
        path=Path(r['path']);require(not path.is_absolute() and '..' not in path.parts,'REPO_PATH')
        b=git('show',r['commit']+':'+r['path'])
        blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        require(len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
                and blob==r['git_blob'],'FIXED_EVIDENCE_BYTES')
        raw[r['role']]=b;catalog.append(r)
    J=lambda role:json.loads(raw[role],parse_float=str)
    C=lambda role:list(csv.DictReader(io.StringIO(raw[role].decode())))
    ma,ps,ref=J('map_audit'),J('pilot_summary'),J('pilot_evidence_reference')
    require(ma['compiler_hash']==ps['compiler']==ref['compiler']==COMPILER,'COMPILER_JOIN')
    require(ps['profile']==ref['profile']==PROFILE,'PROFILE_JOIN')
    require(ps['provider']==ma['SOURCE']==ref['original_science_source'],'PROVIDER_JOIN')
    require(ps['status']=='PASS_MANDATORY_STOP' and not ps['next_stage_authorized'],'PILOT_STOP')
    require(J('failed_terminal')['status']=='STOP','OLD_STOP_PRESERVED')
    require(J('current_final_gate')['status']=='PASS_MANDATORY_STOP','CURRENT_RESULT_GATE')
    samples=J('pilot_sampling')['receipts'];laws={}
    binding=J('pilot_binding');closure=binding['source_closure'];verified_closure=[]
    require(binding['SOURCE']==closure['SOURCE']==ps['SOURCE'],'PILOT_SOURCE_BINDING')
    for field in ('compiler','profile','plan_sha256','grant_sha256','config_sha256'):
        require(binding[field]==ps[field],'PILOT_RECEIPT_BINDING_'+field)
    for path,expected in {**closure['files'],**closure['auxiliary_existing_test']}.items():
        b=git('show',ps['SOURCE']+':'+path)
        require(hashlib.sha256(b).hexdigest()==expected,'PILOT_SOURCE_CLOSURE')
        verified_closure.append(dict(commit=ps['SOURCE'],path=path,bytes=len(b),sha256=expected,
            git_blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()))
    compiled=J('pilot_costs')['trajectories'];require(len(compiled)==16,'PILOT_16_PAIRS')
    sample_by_id={x['task']['id']:x for x in samples}
    require(len(sample_by_id)==16,'PILOT_SAMPLE_IDENTITIES')
    for t in compiled:
        a=sample_by_id[t['task']['id']]
        require(t['task']==a['task'] and t['event_digest']==a['event_digest']
                and t['native_exit']==0 and t['reaped'] is True,'PILOT_NATIVE_SAMPLE_JOIN')
        require(len(t['rows'])==2 and {x['axis'] for x in t['rows']}=={'cosine','sine'},'PILOT_PAIR')
        require(all(x['seed']==t['task']['seed'] and x['event_digest']==a['event_digest']
            and x['compiler_hash']==COMPILER for x in t['rows']),'PILOT_AXIS_BINDING')
    for q in (4,8):
        rows=[x['receipt'] for x in samples if x['receipt']['q']==q]
        require(len(rows)==8 and all(x['order2_count']==1 and x['orders'].count(4)==0
                and len(x['orders'])==q*4 for x in rows),'PILOT_ORDER_COVERAGE')
        first=rows[0];candidate=f'B2_p7_q{q}_2nd_r4_K4'
        originals=[x for x in J('map_rare')['cells'] if x['cell_id']==candidate]
        require(all(x['distribution']==first['original_finite_law']==ref['laws'][candidate]
                for x in originals),'FINITE_LAW_LITERAL_JOIN')
        require(all(x['law']==first['law'] and x['original_finite_law']==first['original_finite_law']
                for x in rows),'SAVED_STRATUM_LAW_LITERAL_JOIN')
        law=first['law']
        laws[candidate]=dict(q=q,R=first['R'],p0_p2_p4=law['original_probabilities'],
            saved_pi0=law['pi0'],saved_pi2=law['pi2'],saved_pi4=law['pi4'],
            conditional_N2_counts=law['counts'],conditional_N2_probabilities=law['count_probabilities'],
            saved_P_N2_eq_1_given_s2=law['count_probabilities'][0],
            unconditional_single_mass=None,unconditional_multi_mass=None,
            unconditional_mass_status='NOT_SAVED_AS_SEPARATE_SCALAR_NOT_RECOMPUTED',
            unconditional_single_formula='R*p2*p0**(R-1)',
            unconditional_multi_formula='(p0+p2)**R-p0**R-R*p2*p0**(R-1)',
            conditional_multi_formula='1-P(N2=1|s2)',
            probability_evaluation_executed=False,formula_implementation='pinned pilot sampling SOURCE; no formula reevaluation',
            original_observation=originals,pilot_N2_counts=[x['order2_count'] for x in rows],
            s4_compiled_cost=None,s2_multi_compiled_cost=None)
    groups=C('map_groups');wrappers=C('map_wrappers')
    core=literal_constants(raw['source_contract']);pre=literal_constants(raw['source_preflight'])
    group_keys=set();global_seeds=set();by_phase={};confirmation={}
    for r in groups:
        key=(r['phase'],r['cell_id'],r['replica'])
        require(key not in group_keys,'DUPLICATE_GROUP');group_keys.add(key)
        seed=int.from_bytes(hashlib.sha256(
            f"{core['NAMESPACE']}|{ma['seed_source']}|{pre['INPUT_SHA']}|{r['cell_id']}|{r['phase']}|{r['replica']}".encode()).digest()[:8],'big')
        require(str(seed)==r['seed'],'DATASET_SEED_SOURCE')
        require(r['seed'] not in global_seeds,'SEED_COLLISION');global_seeds.add(r['seed'])
        by_phase.setdefault(r['phase'],set()).add(r['seed'])
        if r['phase']=='confirmation':confirmation.setdefault(r['cell_id'],[]).append(r)
        pair=[w for w in wrappers if (w['phase'],w['cell_id'],w['replica'])==key
              and w['control']=='symmetric_directional']
        require(len(pair)==2 and {x['axis'] for x in pair}=={'cosine','sine'},'PAIRED_WRAPPERS')
        require(all(x['seed']==r['seed'] and x['event_digest']==r['event_digest']
                and x['compiler_hash']==COMPILER and x['source']==r['source'] for x in pair),'PAIR_IDENTITY_JOIN')
    require(set(confirmation)==set(ma['selected_ids']) and len(confirmation)==6
            and all(len(x)==32 for x in confirmation.values()),'UNION6_N32')
    require(not by_phase['exploration']&by_phase['confirmation'],'EXPLORATION_CONFIRMATION_SEED_OVERLAP')
    require('selection_before_confirmation.json' in ma['result_file_SHA'],'SELECTION_RECEIPT_HASH')
    execution=raw['source_execution'].decode()
    require(execution.index("w.write('selection_before_confirmation.json'")<
            execution.index("supervisor(root,run,'confirmation'"),'SELECTION_SOURCE_ORDER')
    original=[]
    confirmation_literals=C('map_confirmation_rows')
    for r in J('map_precision')['rows']:
        fixed=r['exploration']['B2']['cell_id']
        fixed_rows=[v for v in confirmation_literals if v['cell_id']==fixed and
                    v['epsilon']==r['epsilon'] and v['metric']=='RZ' and v['allowance_factor']=='1.0']
        require(len(fixed_rows)==1,'FIXED_CANDIDATE_CONFIRMATION_LITERAL_ROW')
        original.append(dict(epsilon=r['epsilon'],exploration_fixed_candidate=fixed,
            fixed_candidate_confirmation=fixed_rows[0],
            post_confirmation_point_candidate=r['confirmation']['B2']['cell_id'],
            post_confirmation_candidate_evaluation=r['confirmation']['B2'],
            post_confirmation_selection_flag=r['post_confirmation_point_selection']))
    current=C('current_comparisons')
    require(len(current)==40,'FORTY_SAVED_COMPARISONS')
    saved_confirmation=[r for r in C('map_confirmation_rows') if r['method']=='B2' and r['metric']=='RZ']
    current_pr=[r for r in C('current_accounting') if r['cell_id'].startswith('B2_')]
    seed_audit=dict(status='DISTINCT_SEED_STREAMS_SOURCE_SUPPORTED',
        source_seed_identity=ma['seed_source'],execution_provider=ma['SOURCE'],
        matched_group_registrations=len(groups),unique_master_seeds=len(global_seeds),
        exploration_confirmation_seed_overlap=0,confirmation_candidates=sorted(confirmation),
        each_confirmation_n=32,within_trajectory_axes_paired=True,
        cross_candidate_paired_design=False,cross_candidate_covariance=None,
        mathematical_PRNG_independence_proved=False,
        independence_assumption='distinct candidate/phase-keyed PCG64 streams, not a proof from seed uniqueness',
        common_replica_index_implies_pairing=False,
        selection_before_confirmation_hash=ma['result_file_SHA']['selection_before_confirmation.json'],
        selection_receipt_original_bytes_verified=False,
        selection_receipt_evidence='published hash + fixed SOURCE write-before-dispatch + final coverage; private receipt not opened',
        epsilon_0005_exploration_selection='NOT_IN_ORIGINAL_FOUR_EPSILON_EXPLORATION',
        endpoint_reuse='same candidate n32 cost samples across epsilon, allowance1/10 and bias modes; saved shot weights differ')
    pilot=dict(SOURCE=ps['SOURCE'],result='b21e90b8368645a5066c06a2a0a2ceb2850d162e',
        status=ps['status'],compiler=ps['compiler'],profile=ps['profile'],
        descriptive=ps['descriptive'],new_actual_calls=ps['new_actual_calls'],
        budget=ps['budget'],old_run01=J('failed_terminal'),
        plan_sha256=ps['plan_sha256'],grant_sha256=ps['grant_sha256'],
        config_sha256=ps['config_sha256'],raw_private_inputs_opened=False)
    # No means/SE/probabilities/rankings computed: only stored fields and identities.
    return dict(schema='H8_RARE_TAIL_SELECTION_METADATA_AUDIT_V1',status='PASS_METADATA_ONLY',
        fixed_evidence_files=len(catalog),catalog=catalog,verified_pilot_SOURCE_closure=verified_closure,laws=laws,seed_and_selection=seed_audit,
        original_fixed_candidate_evaluations=original,
        original_confirmation_RZ_literal_rows=saved_confirmation,
        current_PR_saved_literal_rows=current_pr,current_comparison_literal_rows=current,
        pilot=pilot,scientific_calls=0,numerical_statistics_executed=False,
        probabilities_recomputed=False,costs_reaccounted=False,rankings_recomputed=False,
        source_code_executed_from_original_git_blobs=False,independent_review=False,
        next_stage_authorized=False,mandatory_STOP=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');args=p.parse_args()
    result=audit()
    if args.output:
        out=Path(args.output).resolve()
        expected=ROOT/'artifacts/resource_applicability/track_a_h8_rare_tail_selection_readiness/2026-10-18'
        require(out==expected/'evidence_audit_v1.json','OWN_NEW_AUDIT_OUTPUT_ONLY')
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in
        ('catalog','verified_pilot_SOURCE_closure','laws','seed_and_selection','original_fixed_candidate_evaluations',
         'original_confirmation_RZ_literal_rows','current_PR_saved_literal_rows',
         'current_comparison_literal_rows','pilot')},sort_keys=True))


if __name__=='__main__':main()
