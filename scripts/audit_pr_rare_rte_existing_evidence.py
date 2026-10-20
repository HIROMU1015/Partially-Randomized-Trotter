"""Documentary extraction and byte/structure audit only; no scientific imports.

Never evaluates probabilities, fractions, costs, variance, bounds, or rankings.
Reads fixed Git blobs. Copies stored JSON fields and CSV strings literally.
"""
import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/resource_applicability/pr_rare_rte_existing_evidence/2026-10-21'
BASE = '80b7a35c4dbf683f952293eeb55944bc10b9efc8'
H2 = 'f2d1a86c431f835db700f348ce618d8604e7ee23'
H4 = '821cc0f6ced57168293ea166644024867b064ffb'
H6 = '3cccfcb2051dacf29e91473e2e1152acef1574bd'
H8 = 'e6eea5387a6e84d1e39a9adb3b61975acc74e200'
PACKET = 'a6246f03c6297438ab624fc62e5061a58e53145f'
PROFILE = '5f267e96ea313f373eb78a75040a43fac3a2ee1b'
A2 = 'artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/P_completed_run01/'
A4 = 'artifacts/resource_applicability/track_a_h4_tol_p_completed/2026-10-19/'
A6 = 'artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/'
A8 = 'artifacts/resource_applicability/track_a_h8_new_pr_confirmation_results/2026-10-19/'
PK = 'artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/candidate/'
PS = 'artifacts/resource_applicability/track_a_h8_original_artifacts_final_search/2026-10-20/share_candidate/'
SPECS = {}

def add(eid, commit, path, role):
    SPECS[eid] = dict(commit=commit, path=path, role=role)

add('PROTOCOL', BASE, 'docs/research/track_a_pr_resource_evaluation_protocol_v2.md', 'normative')
add('CONVENTIONS', BASE, 'docs/rte_conventions.md', 'normative_probability')
for name, ref, prefix, report in [
    ('H2',H2,A2,'track_a_h2_p_completed_20261019.md'),
    ('H4',H4,A4,'track_a_h4_tol_p_completed_20261019.md'),
    ('H6',H6,A6,'track_a_h6_pr_confirmation_science_results_20261019.md'),
    ('H8',H8,A8,'track_a_h8_new_pr_confirmation_results_20261019.md')]:
    add(name+'_REPORT',ref,'docs/research/'+report,'P_report')
for eid,ref,prefix,files in [
    ('H2',H2,A2,['confirmation_result.json','trajectory_receipts_projection.json','paired_costs_projection.json','execution_binding.json','native_completion_audit.json','native_exit.json','terminal.json']),
    ('H4',H4,A4,['result.json','event_label_audit.json','paired_costs_projection.json.gz','execution_binding.json','completion_audit.json','native_exit.json','terminal.json']),
    ('H6',H6,A6,['result.json','axis_costs.csv','trajectory_receipts.csv','source_and_preservation.json','native_completion.json','ledger_audit.json','scope_boundary.json','csv_serialization_audit.json']),
    ('H8',H8,A8,['comparisons.csv','paired_metrics.csv'])]:
    for p in files:add(eid+'_'+p.split('.')[0],ref,prefix+p,'P_result_or_audit')
for p in ['01_results_and_source.json','02_execution_receipts.json','03_worker_receipts.json','04_grant_ledger_marker_stop.json','05_seed_cohort_audit.json','06_task_axis_links.json','07_input_runtime_bindings.json','08_provenance_and_redactions.json','09_missing_evidence.csv','10_source_conditions.md','packet_manifest.json']:
    add('PACKET_'+p.split('.')[0],PACKET,PK+p,'redacted_original_receipt_projection')
for p in ['01_original_profile_receipt.json','02_saved_PR_circuit_search.json','03_axis_coverage.csv','04_source_and_provenance.json','05_missing_status_changes.json','06_preservation_and_scope_audit.json','supplement_manifest.json']:
    add('PROFILE_'+p.split('.')[0],PROFILE,PS+p,'profile_recovery_supplement')
for eid,ref,path in [
 ('H2_SOURCE','ce1f511c4f7429190adcee58cde86be8a87a2fb1','scripts/h2_p_science.py'),
 ('H4_SOURCE','3c664ca4a9f003880b9f6d56f2a0e55ed5b56603','scripts/h4_tol_p_science.py'),
 ('H4_NATIVE','3c664ca4a9f003880b9f6d56f2a0e55ed5b56603','scripts/h4_native_science.py'),
 ('H6_SOURCE','bf0c26dae4477de5fe7264a7c91e077661ef9b38','scripts/h6_pr_confirmation.py'),
 ('H8_SOURCE','745a674546f2cb6258526fbfde1d9a00601689f8','scripts/h8_pr_confirmation.py'),
 ('NATIVE_COST','745a674546f2cb6258526fbfde1d9a00601689f8','scripts/pr_cost_native.py'),
 ('RTE_SOURCE','87b6b9f46b7e8ba9abe132ee21478c114b6f46f7','src/trotterlib/rte.py'),
 ('NATIVE_PROVIDER','c31ef424175c02379eafebe8fc6c7548c1d9aabe','src/trottertracks/resource_applicability/ax2b_native_df_v5.py'),
 ('EXACT_MC_API',BASE,'src/trotterlib/rte_compiled_cost.py'),
 ('COMPILED_METRICS',BASE,'src/trotterlib/rte_compiled_cost.py'),
 ('NATIVE_MODEL',BASE,'src/trotterlib/native_df_gate_model.py'),
 ('RARE_SOURCE','3c0c9546701fd953de3b299be561344375c1a78b','src/trotterlib/h8_rare_tail_selection.py')]:
    add(eid,ref,path,'static_SOURCE')
for eid,ref,path in [
 ('H6_RARE','eee1343f6e99aa7661892ddfbe1067e974e2b74a','docs/research/track_a_h6_rare_order_cost_reliability_v1.md'),
 ('H6_RARE_RESULT','eee1343f6e99aa7661892ddfbe1067e974e2b74a','artifacts/resource_applicability/track_a_h6_rare_order_cost_analysis_v1/2026-10-13/cost_reliability_analysis_v1.json'),
 ('H6_DBA','648184eb','docs/research/track_a_h6_dba_20261018.md'),
 ('H6_S2',BASE,'artifacts/resource_applicability/track_a_h4_b0i_manuscript_integration/2026-10-14/references/track_a_h6_conditional_s2_pilot_result_v1.md'),
 ('H8_S2','b21e90b8368645a5066c06a2a0a2ceb2850d162e','docs/research/track_a_h8_conditional_s2_run02_results_20261018.md'),
 ('H8_S2_RESULT','b21e90b8368645a5066c06a2a0a2ceb2850d162e','artifacts/resource_applicability/track_a_h8_conditional_s2_run02/2026-10-18/result_summary.json'),
 ('H8_S2_LAW','b21e90b8368645a5066c06a2a0a2ceb2850d162e','artifacts/resource_applicability/track_a_h8_conditional_s2_run02/2026-10-18/sampling_receipts.json'),
 ('H8_RARE',BASE,'docs/research/track_a_h8_rare_sensitivity_selection_run01_results_20261018.md'),
 ('H8_RARE_RESULT',BASE,'artifacts/resource_applicability/track_a_h8_saved_analysis_run01/2026-10-18/rare_sensitivity/result.json'),
 ('H8_RARE_DESIGN',BASE,'docs/research/track_a_h8_rare_tail_sampling_design_20261018.md'),
 ('H4_MC','295b93d50662a0f8af3ec469df43099a3f04af60','docs/research/track_a_h4_pr_cost_variability_20261019.md'),
 ('B_DIAGNOSTIC',BASE,'artifacts/resource_applicability/track_a_pr_cost_batch_results/2026-10-18/b_diagnostic_scope_summary.json'),
 ('B_COMPLETION',BASE,'docs/research/track_a_pr_cost_batch_completed_results_20261018.md'),
 ('TIER_R','bf8c19057fd82c900c6234152e152e79cb52e707','artifacts/resource_applicability/track_a_final_technical_audit/2026-10-19/A_rare_event/report.md'),
 ('TIER_R_GATES',BASE,'artifacts/resource_applicability/track_a_final_technical_audit/2026-10-19/A_rare_event/native_gate_static_table.csv'),
 ('TIER_R_SOURCES',BASE,'artifacts/resource_applicability/track_a_final_technical_audit/2026-10-19/A_rare_event/source_manifest.json'),
 ('THEORY_COMPARE','7e454b91185ed7b0f92a3790f800ba5d65d5f156','docs/research/track_a_pr_theory_vs_compiled_completed_20261018.md'),
 ('THEORY_REUSE',BASE,'docs/research/track_a_v2_existing_theory_reconciliation_20261016.md'),
 ('COST_HISTORY',BASE,'docs/rte_compiled_cost_validation_summary.md'),
 ('EVENT_API',BASE,'docs/rte_compiled_event_cost.md'),
 ('Q100',BASE,'docs/research/track_a_pf_q_model_evidence_integration_20261020.md'),
 ('Q100_METRICS',BASE,'artifacts/resource_applicability/track_a_pf_q_model_evidence_integration/2026-10-20/adopted_metrics.json')]:
    add(eid,ref,path,'existing_study')

add('EVENT_EXACT_SAVED',BASE,'artifacts/rte_boundary_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k0_v1.json','historical_exact_single_event_result')

add('H4_RARE_COMPLETED','97d25cc431516d720ce91c7cb4f148e061d8ea34','docs/research/track_a_h4_followup_20261018.md','completed_legacy_conditional_pilot')
add('H8_RARE_COMPLETED','839c403004cf9343730c43adb346024aea41999b','docs/research/track_a_h8_ab_20261018.md','completed_conditional_pilot')

def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)

def blob(eid):
    row=SPECS[eid]
    return git('show',row['commit']+':'+row['path'])

def read(eid):
    b=blob(eid)
    if SPECS[eid]['path'].endswith('.gz'):b=gzip.decompress(b)
    return json.loads(b)

def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def csvsave(name,rows):
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)

LITERALS=[]
def literal(eid,pointer):
    value=read(eid)
    for key in pointer.strip('/').split('/'):
        value=value[int(key)] if isinstance(value,list) else value[key.replace('~1','/').replace('~0','~')]
    LITERALS.append(dict(evidence_id=eid,json_pointer=pointer,value=value))
    return value

def link(eid):
    r=SPECS[eid]
    commit=git('rev-parse',r['commit']+'^{commit}').decode().strip()
    return 'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/'+commit+'/'+r['path']

def conditions():
    rows=[]
    for i,(q,delta,strata) in enumerate([(1,'.8',{'S0_ONLY':79,'S2_SINGLE':1}),(4,'.2',{'S0_ONLY':12})]):
        eid='H2_confirmation_result';p='/accounting/comparisons/'+str(i);x=literal(eid,p)
        saved=literal('H2_trajectory_receipts_projection','/stratum_counts/'+x['cell_id'])
        assert saved==strata
        rows.append(dict(condition='H2_'+str(x['epsilon']),system='H2',DF_policy='tol_only_1e-8',rank=3,
            candidate=x['cell_id'],epsilon=x['epsilon'],q=q,r=2,K=4,delta=delta,n=x['n'],
            SOURCE=read(eid)['SOURCE'],result_commit=H2,events_saved='256 combined; per-condition aggregate not quoted',
            event_orders_saved=None,strata_saved=saved,mean_exact=x['G_exact'],paired_SE=x['paired_SE'],
            fixed_PF=x['PF']['cell_id'],PF_G_exact=x['PF']['G_RZ_exact'],shots=x['shots'],
            metrics='RZ,CX,RZ_depth,CX_depth,depth,size; paired_costs_projection',full_wrapper=True,
            missing='q1:S2_MULTI/S4; q4:S2_SINGLE/S2_MULTI/S4; component-support costs in all strata',
            route='A:saved sample; B:law/stratum analysis; C:matched rare costs if needed; D:all-support compiler proof',
            result_evidence=eid,receipt_evidence='H2_trajectory_receipts_projection'))
    for i in range(2):
        eid='H4_result';x=literal(eid,'/comparisons/'+str(i));cell=x['cell'];orders=literal('H4_event_label_audit','/rows/'+cell['id'])
        rows.append(dict(condition='H4_'+str(x['epsilon']),system='H4',DF_policy='tol_only_1e-8',rank=10,
            candidate=cell['id'],epsilon=x['epsilon'],q=cell['q'],r=cell['r'],K=cell['K'],delta=['.8','.1'][i],n=x['n'],
            SOURCE=read(eid)['SOURCE'],result_commit=H4,events_saved='5680 combined; per-condition saved order counts below',
            event_orders_saved=orders,strata_saved=x['strata'],mean_exact=x['result']['G_exact'],paired_SE=x['result']['paired_SE'],
            fixed_PF=x['PF_id'],PF_G_exact=x['PF_G_exact'],shots=x['saved_shots'],
            metrics='RZ,CX,RZ_depth,CX_depth,depth,size; paired_costs_projection.json.gz',full_wrapper=True,
            missing=['S2_MULTI; S4 outside K2 support; unobserved component sequences','S2_MULTI/S4; unobserved component sequences'][i],
            route='A:saved sample and existing MC diagnostics; B:law/sensitivity; C:matched rare costs if needed; D:compiler closure',
            result_evidence=eid,receipt_evidence='H4_event_label_audit'))
    for i in range(2):
        eid='H6_result';x=literal(eid,'/scopes/'+str(i));cell=x['cell'];z=x['primary']
        rows.append(dict(condition='H6_'+str(x['epsilon']),system='H6',DF_policy='tol_only_1e-8',rank=19,
            candidate=cell['id'],epsilon=x['epsilon'],q=cell['q'],r=cell['r'],K=cell['K'],delta=['.4','.1'][i],n=z['n'],
            SOURCE=read(eid)['SOURCE'],result_commit=H6,events_saved=x['calls']['RTE_event'],
            event_orders_saved=x['event_order_counts'],strata_saved=None,mean_exact=z['mean_exact'],paired_SE=z['paired_SE'],
            fixed_PF=x['fixed_PF_id'],PF_G_exact=z['fixed_PF_G_exact'],shots=x['shots'],
            metrics='RZ,CX,RZ_depth,CX_depth,depth,size; axis_costs.csv',full_wrapper=True,
            missing=['S2_SINGLE/MULTI counts not saved in public aggregate; full component support','one order2 event only; multi not covered; component support'][i],
            route='A:saved sample; B:P-specific law and stratum linkage; C:p4 rare costs if needed; D:compiler closure',
            result_evidence=eid,receipt_evidence='H6_axis_costs'))
    x=read('PACKET_01_results_and_source')
    csvrows=list(csv.DictReader(io.StringIO(blob('H8_comparisons').decode())))
    for i,c in enumerate(x['conditions']):
        z=c['saved_comparison_rows'][0]
        assert z in csvrows
        LITERALS.append(dict(evidence_id='PACKET_01_results_and_source',json_pointer='/conditions/'+str(i),value=c))
        rows.append(dict(condition='H8_'+z['epsilon'],system='H8',DF_policy='tol_only_1e-8',rank=28,
            candidate=c['candidate']['id'],epsilon=z['epsilon'],q=c['candidate']['q'],r=4,K=4,delta=c['delta_literal'],n=c['n'],
            SOURCE=c['science_SOURCE'],result_commit=H8,events_saved=[384,3072][i],
            event_orders_saved={'0':[384,3072][i],'2':0,'4':0},strata_saved='report: all observed events order0; no new histogram computed',
            mean_exact=z['mean_exact'],paired_SE=z['paired_SE'],fixed_PF=z['fixed_PF_id'],PF_G_exact=z['fixed_PF_G_exact'],
            shots={'cosine':z['shots_cosine'],'sine':z['shots_sine']},metrics='RZ,CX,RZ_depth,CX_depth,depth,size; paired_metrics.csv',full_wrapper=True,
            missing='S2_SINGLE/S2_MULTI/S4 at this candidate; saved PR circuit bodies not acquired',
            route='A:saved sample plus packet/profile update; B:matched law/sensitivity; C:p8q2 or p7q16 rare costs if needed; D:closure/inference',
            result_evidence='H8_comparisons',receipt_evidence='PACKET_06_task_axis_links'))
    return rows

def inventory():
    rows=[]
    for eid,r in SPECS.items():
        commit=git('rev-parse',r['commit']+'^{commit}').decode().strip();b=blob(eid)
        rows.append(dict(evidence_id=eid,commit=commit,path=r['path'],role=r['role'],
            git_blob=git('rev-parse',commit+':'+r['path']).decode().strip(),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),url=link(eid),
            acquisition='fixed_git_blob_read',scientific_execution=False))
    return rows

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=inventory();save('provenance.json',rows);csvsave('provenance.csv',rows)
    cs=conditions();save('conditions.json',cs)
    csvsave('conditions.csv',[{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in cs])
    save('literal_extracts.json',LITERALS)
    sources=[]
    for eid in ['H2_execution_binding','H4_execution_binding','H6_source_and_preservation']:
        d=read(eid)
        for path,expected in d['source_hashes'].items():
            b=git('show',d['SOURCE']+':'+path)
            assert hashlib.sha256(b).hexdigest()==expected,(eid,path)
            sources.append(dict(receipt=eid,SOURCE=d['SOURCE'],path=path,sha256=expected,match=True))
    save('source_receipt_matches.json',sources)
    remote=json.loads((OUT/'remote_acquisition.json').read_text())
    for r in remote['files']:
        assert r['success'] and r['content_retrieved']
        assert git('rev-parse',r['commit']+':'+r['path']).decode().strip()==r['remote_declared_git_blob']
    save('remote_blob_validation.json',dict(status='PASS_REMOTE_DECLARED_GIT_BLOB_IDS_MATCH_LOCAL',files=len(remote['files']),
        remote_content_returned=True,remote_content_independently_rehashed=False,remote_commit_count=len(remote['commits']),
        limitation='Connector returned full content and Git blob IDs. IDs compared to local Git; no claim of independent remote-byte digest recomputation.'))
    # Hash metadata of this worktree's original tracked files; sparse files only.
    before={}
    for p in git('ls-files','-z').decode().split('\0'):
        f=ROOT/p
        if p and f.is_file():before[p]=hashlib.sha256(f.read_bytes()).hexdigest()
    save('preservation_start.json',dict(base_commit=BASE,tracked_materialized_sha256=before,
        scope='This task worktree only. Other worktree HEAD/status are not invariants. Fixed Git evidence blobs separately inventoried.',
        previous_scientific_sources_results_grants_ledgers_markers_STOP_write_authorized=False))
    print('BUILT_LITERAL_AND_PROVENANCE_METADATA')

def check():
    old=json.loads((OUT/'provenance.json').read_text());assert old==inventory()
    expected=json.loads((OUT/'conditions.json').read_text());assert expected==conditions()
    # Validate the four single saved metric examples without pooling or arithmetic.
    assert read('H2_paired_costs_projection')['rows'][0]['axes']['cosine']['RZ']==301
    assert read('H2_paired_costs_projection')['rows'][0]['axes']['cosine']['CX']==148
    assert read('H4_paired_costs_projection')['rows'][0]['axes']['cosine']['RZ']==3433
    assert read('H4_paired_costs_projection')['rows'][0]['axes']['cosine']['CX']==1500
    for eid,rz,cx in [('H6_axis_costs','21787','8992'),('H8_paired_metrics','75764','30936')]:
        row=next(csv.DictReader(io.StringIO(blob(eid).decode())))
        assert row['RZ']==rz and row['CX']==cx
    exact=read('EVENT_EXACT_SAVED')['single_event_exact']
    assert exact['estimate_kind']=='exact_compiled_expectation'
    assert str(exact['expected_cost']['rz_count'])=='289.24012667527643'
    assert str(exact['expected_cost']['cx_count'])=='84.98664101463322'
    assert exact['enumerated_event_count']==218 and exact['standard_error'] is None
    for r in json.loads((OUT/'historical_pilot_tokens.json').read_text()):
        assert r['source_literal'] in blob(r['evidence_id']).decode(),r
    tree=ast.parse(Path(__file__).read_text())
    imported={n.names[0].name for n in ast.walk(tree) if isinstance(n,ast.Import)}
    imported|={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
    assert imported <= {'argparse','ast','csv','gzip','hashlib','io','json','pathlib','re','subprocess'}
    forbidden={'eval','exec','compile','mean','variance','std','bootstrap','sample','fit','transpile'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ''
            assert name not in forbidden,name

    assert json.loads((OUT/'literal_extracts.json').read_text())==LITERALS
    remote=json.loads((OUT/'remote_acquisition.json').read_text())
    for r in remote['files']:assert git('rev-parse',r['commit']+':'+r['path']).decode().strip()==r['remote_declared_git_blob']
    allowed={'PROJECT_MAP.md','VALIDATION_STATUS.md','docs/README.md','docs/research/研究概要・現状.md'}
    before=json.loads((OUT/'preservation_start.json').read_text())
    changed=[]
    for p,digest in before['tracked_materialized_sha256'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=digest:
            assert p in allowed,p
            # Existing index history must remain an unchanged suffix.
            assert (ROOT/p).read_bytes().endswith(git('show',BASE+':'+p)),p
            changed.append(p)
    checked_links=[]
    docs=[ROOT/'docs/research/pr_rare_rte_existing_evidence_20261021.md',*OUT.glob('*.md')]
    for path in docs:
        for dest in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if dest.startswith(('http:','https:','#')):continue
            target=(path.parent/dest.split('#')[0])
            if not target.exists():
                assert target.parent==OUT and target.name in {'structure_validation.json','preservation_audit.json'},(path.name,dest)
            checked_links.append(dict(document=str(path.relative_to(ROOT)),target=dest))
    new_texts=[p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.md','.json','.csv')]
    for p in new_texts:
        t=p.read_text()
        assert not re.search(r'/(?:home|Users|tmp)/',t),p.name
        assert not re.search(r'\[\[[A-Z0-9_]+\]\]',t),p.name
        assert not re.search(r'gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----',t),p.name
    receipt=dict(status='PASS_DOCUMENTARY_STRUCTURE_LITERAL_PROVENANCE_PRESERVATION',
        date_JST='2026-10-21',base_commit=BASE,conditions=8,fixed_blob_entries=len(old),
        remote_files=len(remote['files']),source_receipt_matches=len(json.loads((OUT/'source_receipt_matches.json').read_text())),
        modified_existing_files=changed,old_index_history_preserved_as_suffix=True,
        original_scientific_SOURCE_results_raw_grants_ledgers_markers_STOP_modified=False,
        read_only_gzip_decoding=True,scientific_array_decode=False,scientific_imports=False,
        sampling=False,simulation=False,signal=False,circuit_generation=False,compile_transpile=False,
        new_probability_cost_variance_bound_numbers=False,statistical_or_numerical_reanalysis=False,reranking=False,
        other_job_intervention=False,AI_subagents=False,independent_scientific_review=False,final_evidence_freeze=False,
        publication_push=False,private_raw_or_seed_values_copied=False,link_checks=checked_links,
        limitations='Primary-agent metadata checks. Historical runtime/private originals not re-executed. Pattern scan is limited, not a credential-security guarantee.',mandatory_STOP=True)
    save('structure_validation.json',receipt)
    save('preservation_audit.json',dict(status='PASS_SCOPED_PRESERVATION',base_commit=BASE,
        fixed_evidence_blob_sha256_rechecked=True,materialized_original_files_rechecked=True,
        allowed_index_prefixes_only=changed,unrelated_worktrees_frozen=False,existing_scientific_artifacts_unchanged=True))
    for row in checked_links:
        assert (ROOT/row['document']).parent.joinpath(row['target']).exists(),row
    print(json.dumps({k:v for k,v in receipt.items() if k!='link_checks'},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','check']);args=p.parse_args()
    if args.action=='build':build()
    else:check()
