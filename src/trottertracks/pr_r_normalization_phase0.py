"""Fixed24 normalization-only preparation and future execution. No N/C/G.

Preflight copies saved tail values and hashes inputs, without normalization
arithmetic, Hamiltonian/state-array loading, or approval-proof machinery.
"""
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from . import pr_r_screening as screen

UPSTREAM='f98982118324bcc8402ed513b5c720b37bb03e87'
PLAN_PATH='docs/research/pr_r_normalization_phase0_preparation_20261111/PLAN.json'
UP_DIR='docs/research/pr_r_screening_20261111/'
INDEX=dict(commit='7a8bc7f132e166390c943e6b32c92cc6abe1baf4',path='artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json')
MENU={'H4':{0:[2,4,8],6:[4,128,256,512],12:[4,8,8192,16384,32768]},
      'H6':{4:[11,16,32],8:[16,128,256,512],12:[16,32,2048,4096,8192]}}
REUSED_FILES=('src/trottertracks/pr_r_screening.py','src/trottertracks/pr_r_dependent_n.py',
 'src/trottertracks/pf_pr_rpe_pipeline.py','src/trotterlib/h46_m5_plan.py','scripts/h46_new2_core.py',
 'scripts/h46_long_time_signal.py','src/trotterlib/rte.py')
NEW_FILES=('src/trottertracks/pr_r_normalization_phase0.py','scripts/run_pr_r_normalization_phase0.py')


def blob(root,commit,path):return subprocess.check_output(['git','show',commit+':'+path],cwd=root)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(root):
    """Static JSON/receipt/hash checks only. Paths remain internal, not public."""
    root=Path(root);store=screen.p.SavedArtifacts(root)
    original=store.read(dict(commit=UPSTREAM,path=UP_DIR+'EXECUTION_PLAN.json'))
    declared=original['phase0']['candidates']
    if len(declared)!=24:raise ValueError('Frozen Phase0 must have24 candidates')
    B=screen.b_metadata(root);proposals={}
    for system,Rmap in MENU.items():
        saved=store.read(dict(commit=UPSTREAM,path=UP_DIR+system+'_METADATA_PROPOSAL.json'))
        selected=[r for r in saved['rows'] if r['m'] in Rmap]
        expected=[(m,R) for m,values in Rmap.items() for R in values]
        if [(r['m'],r['R_outer']) for r in selected]!=expected:raise ValueError('Fixed24 R menu differs')
        menu=B.build_menu(saved['base'],Rmap,sampling_law=saved['menu']['sampling_law'])
        for row in selected:
            matches=[d for d in declared if d['system']==system and d['candidate_id']==row['candidate_id']]
            expected_row=dict(system=system,m=row['m'],candidate_id=row['candidate_id'],R_outer=row['R_outer'],q_PF=row['q_PF'],T=row['T'])
            if matches!=[expected_row]:raise ValueError('Phase0 candidate identity differs')
            c=next(r for r in menu['candidates'] if r['candidate']['id']==row['candidate_id'])
            if not screen.p.same(row['request'],B.request_context(menu,c)):raise ValueError('Law/request differs')
        proposals[system]=dict(saved,rows=deepcopy(selected),menu=menu,scope='FROZEN_PHASE0_ONLY')
    index=store.read(INDEX);tails={};input_files={};input_hashes={};historical={}
    for system in MENU:
        specs=index['saved_inputs'][system]['inputs'];context_spec=index['T20_reference_files'][system]
        for name,spec in list(specs.items())+[(system+'_context.json',context_spec)]:
            if sha(spec['path'])!=spec['sha256']:raise ValueError('Saved input/context hash differs')
            input_files[system+':'+name]=spec;input_hashes[system+':'+name]=spec['sha256']
        context=json.loads(Path(context_spec['path']).read_bytes());binding=proposals[system]['base']['binding']
        prefix=1 if system=='H4' else 7;tail=context['prefixes'][str(prefix)]
        if (not screen.p.same(context['binding'],binding) or context['sector_dimension']!=(36 if system=='H4' else 400)
                or type(tail['lambda_R']) not in (int,float) or not math.isfinite(tail['lambda_R']) or tail['lambda_R']<=0):raise ValueError('Saved tail/H/state/sector binding differs')
        if system=='H4':
            receipt=json.loads(Path(specs['H4_tol_receipt.json']['path']).read_bytes())
            if (receipt['hamiltonian_hash']!=binding['hamiltonian'] or receipt['state_hash']!=binding['state'] or receipt['rank']!=10
                    or receipt['sha256']!=specs['H4_tol_snapshot.npz']['sha256']):raise ValueError('H4 receipt differs')
            historical[system]=dict(status=receipt['status'],ground_state_certified=receipt['ground_state_certified'])
        else:
            df=json.loads(Path(specs['H6_df_receipt.json']['path']).read_bytes());snap=json.loads(Path(specs['H6_snapshot_receipt.json']['path']).read_bytes())
            if (df['hamiltonian_hash']!=binding['hamiltonian'] or df['actual_rank']!=19 or snap['sha256']!=specs['H6_snapshot.npz']['sha256']
                    or snap['metadata']['hamiltonian_hash']!=binding['hamiltonian'] or snap['metadata']['state_hash']!=binding['state']):raise ValueError('H6 receipt differs')
            historical[system]=dict(status=snap['H6_status'],ground_state_certified=snap['metadata']['ground_state_certified'])
        tails[system]=dict(binding=deepcopy(binding),prefix=prefix,lambda_R=tail['lambda_R'],
            source=dict(index=INDEX,context_field='T20_reference_files.'+system,context_sha256=context_spec['sha256'],value_field='prefixes.'+str(prefix)+'.lambda_R',qualification='COPIED_SAVED_VALUE_NOT_RECOMPUTED'))
    sources={p:sha(root/p) for p in REUSED_FILES+NEW_FILES}
    for p in REUSED_FILES:
        if (root/p).read_bytes()!=blob(root,UPSTREAM,p):raise ValueError('Reused scientific SOURCE changed')
    return dict(proposals=proposals,tails=tails,input_files=input_files,input_hashes=input_hashes,historical_status=historical,
        sources=sources,saved_sources=store.sources,B_metadata_source=B.source,upstream_candidates=declared)


def public_plan(packet):
    return dict(status='FIXED24_PREPARATION_ONLY_NO_REAL_NORMALIZATION',upstream_commit=UPSTREAM,
        upstream_candidates=packet['upstream_candidates'],fixed_R_menu={s:{str(m):R for m,R in v.items()} for s,v in MENU.items()},
        requests=[deepcopy(r) for s in MENU for r in packet['proposals'][s]['rows']],
        saved_tails=deepcopy(packet['tails']),input_hashes=deepcopy(packet['input_hashes']),historical_status=deepcopy(packet['historical_status']),
        sources=deepcopy(packet['sources']),saved_sources=deepcopy(packet['saved_sources']),B_metadata_source=deepcopy(packet['B_metadata_source']),
        scope=dict(normalization_candidates=24,piece_normalization_occurrences=144,signal_phase_radius_N_C_G=0,H_state_array_loads=0),
        shot_inflation_budget=None,scientific_R_adopted=False,full_RPE_qualified=False,mandatory_stop=True)


def exp_display(exponent):
    if exponent is None:return dict(value=None,status='UNAVAILABLE_NORMALIZATION')
    if not math.isfinite(exponent):return dict(value=None,status='NONFINITE_EXPONENT')
    try:value=math.exp(exponent)
    except OverflowError:return dict(value=None,status='OVERFLOW_BINARY64_KEEP_LOG')
    if value==0 and exponent<0:return dict(value=0.,status='UNDERFLOW_TO_ZERO_KEEP_LOG')
    return dict(value=value,status='FINITE_BINARY64')


def row_output(row):
    n=deepcopy(row['normalization']);logB=n.get('log_B')
    return dict(system=row['system'],m=row['m'],T=row['T'],q_RPE=row['q_RPE'],q_PF=row['q_PF'],delta=row['delta'],
        candidate_id=row['candidate_id'],R_outer=row['R_outer'],formula=row['formula'],prefix=row['prefix'],K=row['K'],
        structure=deepcopy(row['structure']),request=deepcopy(row['request']),normalization=n,
        attenuation=exp_display(-logB if logB is not None else None),
        shot_inflation=exp_display(n.get('optimistic_shot_inflation_log')),
        shot_inflation_interpretation='exp(2 log B), zero-error conditional geometry coefficient; not actual N or a shot lower bound',
        N=None,C=None,G=None,phase_radius_N_status='NOT_EVALUATED',scientific_R_adopted=False,
        recommended_R=None,scientific_candidate_PASS_FAIL=None)


def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def execute(packet,root,source,out,*,score_provider=None):
    """Future science only. Artificial tests substitute tails/provider explicitly."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    result=dict(status='PHASE0_RUNNING',scientific_source=source,records=[],errors=[],real_N_C_G=0,
        signal_phase_radius_actions=0,H_state_array_loads=0,automatic_retry=False,shot_inflation_budget=None,mandatory_stop=True)
    save(out/'progress.json',result)
    scorer=score_provider or screen.screen_normalization
    try:
        for system in MENU:
            scored=scorer(root,packet['proposals'][system],packet['tails'][system],shot_inflation_budget=None,budget_source=None)
            expected=[r['candidate_id'] for r in packet['proposals'][system]['rows']]
            if [r['candidate_id'] for r in scored['rows']]!=expected:raise ValueError('No candidate expansion/drop permitted')
            for original,row in zip(packet['proposals'][system]['rows'],scored['rows']):
                if any(not screen.p.same(original[k],row[k]) for k in ('system','m','T','q_RPE','q_PF','delta','formula','prefix','K','R_outer','structure','request')):raise ValueError('Scored candidate/law binding changed')
            result['records'].extend(row_output(row) for row in scored['rows'])
        result['status']='PHASE0_NORMALIZATION_RECORDED' if all(r['normalization']['status']=='NORMALIZATION_SCORED_NOT_N_QUALIFIED' for r in result['records']) else 'PHASE0_NORMALIZATION_NUMERICAL_UNRESOLVED'
    except Exception as exc:result.update(status='PHASE0_STOP_TECHNICAL_OR_INPUT',errors=[str(exc)])
    result['wall_seconds']=time.monotonic()-started
    try:result['source_preserved']=all(sha(Path(root)/p)==h for p,h in packet['sources'].items())
    except OSError:result['source_preserved']=False
    try:result['input_preserved']=all(sha(v['path'])==v['sha256'] for v in packet['input_files'].values())
    except OSError:result['input_preserved']=False
    if not result['source_preserved'] or not result['input_preserved']:result['status']='PHASE0_STOP_SOURCE_OR_INPUT_CHANGED'
    result['scope_observed']=dict(normalization_rows=len(result['records']),piece_slots_represented=sum(len(r['structure']['allocation']) for r in result['records']),signal_phase_radius_N_C_G=0)
    save(out/'result.json',result)
    save(out/'HANDOFF.json',dict(status=result['status'],scientific_source=source,records=result['records'],
        no_threshold_no_candidate_selection=True,normalization_evidence_valid=result['source_preserved'] and result['input_preserved'] and result['status']=='PHASE0_NORMALIZATION_RECORDED',N_C_G=None,scientific_R_adopted=False,full_RPE_qualified=False,mandatory_stop=True))
    fields=['system','m','T','q_RPE','q_PF','R_outer','candidate_id','event_count','allocation','log_B','attenuation','attenuation_status','log_shot_inflation','shot_inflation','shot_inflation_status','one_ulp_log_sensitivity','normalization_status']
    with (out/'NORMALIZATION.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for r in result['records']:
            n=r['normalization'];writer.writerow(dict(**{k:r[k] for k in ('system','m','T','q_RPE','q_PF','R_outer','candidate_id')},event_count=r['structure']['event_count'],allocation=json.dumps(r['structure']['allocation']),
                log_B=n.get('log_B'),attenuation=r['attenuation']['value'],attenuation_status=r['attenuation']['status'],log_shot_inflation=n.get('optimistic_shot_inflation_log'),
                shot_inflation=r['shot_inflation']['value'],shot_inflation_status=r['shot_inflation']['status'],one_ulp_log_sensitivity=n.get('one_ulp_accumulation_log_sensitivity'),normalization_status=n['status']))
    return result
