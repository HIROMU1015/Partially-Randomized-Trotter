"""Metadata-only PF compile inventory. Reads fixed Git objects; never imports science.

Only literal transcription, identifier joins, hashes, and inventory counts. No
model evaluation, statistics, quantum circuit construction or execution.
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
BASE = '3826f566b04ad9c2ead9abf1b80c864f7b34b785'
B = 'artifacts/resource_applicability/'
OUT = ROOT / (B+'track_a_pf_direct_compile_plan/2026-10-19')
SOURCES = {}
COMPILER = '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def source(label, path, commit=BASE, fmt='json'):
    raw = git('show', commit+':'+path)
    commit = git('rev-parse', commit).decode().strip()
    SOURCES[label] = dict(id=label, commit=commit, path=path,
        sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
        git_blob_sha=git('rev-parse', commit+':'+path).decode().strip(),
        url=f'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/{commit}/{path}')
    if fmt=='bytes':return raw
    if path.endswith('.gz'):raw=gzip.decompress(raw)
    if fmt=='csv':return list(csv.DictReader(io.StringIO(raw.decode())))
    if fmt=='jsonl':return [json.loads(x, parse_float=str) for x in raw.splitlines()]
    return json.loads(raw, parse_float=str)


def write_json(name, data):
    (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def write_csv(name, rows, fields=None):
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)


def collect():
    SOURCES.clear()
    registrations=source('REGISTRATIONS', B+'track_a_protocol_v2_draft/2026-10-19/candidate_registrations.json')['registries']
    domains=source('DOMAINS',B+'track_a_final_technical_audit/2026-10-19/B_domain/registered_candidate_domains.csv',fmt='csv')
    systems={
        'H2_DF3': registrations['H2_D70']['cells'],
        'H4_DF10': registrations['H4_TOL_D']['cells'],
        'H6_DF19': registrations['H6_strengthened_S2_S4']['rows'],
        'H8_DF28': registrations['H8_strengthened184']['rows'],
        'H4_LEGACY_DF12': [r for r in domains if r['domain']=='H4_registered_D96_plus_old8'],
    }
    cells={(s,r['id']):r for s,rs in systems.items() for r in rs}
    evidence=[]
    def add(system,cid,axis,metrics,label,locator,kind='DIRECT',origin='',reuse='SAVED_BINDING_RECHECK_REQUIRED',model=''):
        if (system,cid) not in cells:return
        assert axis in ('cosine','sine')
        evidence.append(dict(evidence_id='PF_E'+str(len(evidence)+1),system=system,candidate_id=cid,axis=axis,
            kind=kind,RZ_literal=str(metrics.get('measured_RZ', metrics.get('RZ', metrics.get('measured','') if not isinstance(metrics.get('measured'),bool) else ''))),
            CX_literal=str(metrics.get('CX','')),depth_literal=str(metrics.get('depth','')),
            size_literal=str(metrics.get('size','')),
            compile_seconds_literal=str(metrics.get('compile_wall_seconds',metrics.get('compile_seconds',metrics.get('seconds',metrics.get('recorded_compile_wall_seconds',''))))),
            peak_RSS_bytes_literal=str(metrics.get('peak_RSS_bytes',metrics.get('worker_peak_rss_bytes',''))),
            fingerprint=metrics.get('fingerprint',metrics.get('numeric_fingerprint',metrics.get('fingerprint_before',''))),
            compiler_hash=metrics.get('compiler_hash',''),
            profile_hash=metrics.get('profile_sha256',metrics.get('profile_hash','')) or '',
            origin_SOURCE=metrics.get('SOURCE',origin) or '',model_sha256=metrics.get('model_sha256',model),
            reuse_assessment=reuse,source_id=label,source_locator=locator))
        if kind=='DIRECT':
            prediction=metrics.get('exact_prediction',metrics.get('prediction'))
            if prediction is not None:
                add(system,cid,axis,dict(RZ=prediction,model_sha256=metrics.get('model_sha256',model)),label,
                    locator+('/exact_prediction' if 'exact_prediction' in metrics else '/prediction'),kind='MODEL',reuse='SAVED_PREDICTION_NO_NEW_EVALUATION')
    # H2: both newly acquired and explicitly accepted FP pairs, deterministic only.
    h2=source('H2_D',B+'track_a_h2_pr_resource/2026-10-19/D_completed_run02/result.json','7b202a1a569c6839c4c7a5dfb18b6ff51ec3f3eb')
    for key in ('measurements','reused_FP_measurements'):
        for i,row in enumerate(h2[key]):
            for a,m in row['axes'].items():add('H2_DF3',row['cell_id'],a,m,'H2_D',f'/{key}/{i}/axes/{a}',origin=h2['SOURCE'] if key=='measurements' else h2['original_SOURCE'])
    h4=source('H4_D',B+'track_a_h4_tol_d_completed/2026-10-19/result.json','406f70570323366da185355550a3900388717fbb')
    h4pairs=source('H4_D_PAIRS',B+'track_a_h4_tol_d_completed/2026-10-19/paired_measurements.json.gz','406f70570323366da185355550a3900388717fbb')
    for i,row in enumerate(h4pairs['rows']):
        for j,a in enumerate(row['rows']):add('H4_DF10',row['cell_id'],a['axis'],dict(a['metrics'],**{k:v for k,v in a.items() if k!='metrics'}),'H4_D_PAIRS',f'/rows/{i}/rows/{j}',origin=h4['SOURCE'])
    fp=source('H4_FP_REUSE',B+'track_a_h4_tol_main_phase0/2026-10-19/FP_reuse_evidence.json','406f70570323366da185355550a3900388717fbb')
    assert fp['binding']==h4['binding'] and fp['same_input_state_compiler_and_fixed_native_SOURCE'] is True
    for i,row in enumerate(fp['measured_axes']):
        if row['cell_id'].startswith('B2'):continue
        for j,a in enumerate(row['axes']):add('H4_DF10',row['cell_id'],a['axis'],dict(a['metrics'],**{k:v for k,v in a.items() if k!='metrics'}),'H4_FP_REUSE',f'/measured_axes/{i}/axes/{j}',origin=fp['SOURCE'])
    h6axis=source('H6_SAVED_AXIS_COSTS',B+'track_a_v1_reaccount_result/2026-10-16/axis_costs.json')
    for i,row in enumerate(h6axis['rows']):
        add('H6_DF19',row['cell_id'],row['axis'],row,'H6_SAVED_AXIS_COSTS',f'/rows/{i}',kind='DIRECT' if row.get('measured') else 'MODEL',
            reuse='PUBLIC_PRICE_ONLY_USE_MATCHED_METRICS_RECEIPT' if row.get('measured') else 'NOT_A_MEASUREMENT')
    h6anchors=source('H6_ANCHORS',B+'track_a_v1_phase_b_readiness/2026-10-16/saved_cost_coverage.json')
    for i,row in enumerate(h6anchors['anchors']):
        add('H6_DF19',row['cell_id'],row['stratum']['axis'],dict(row,**row['stratum']),'H6_ANCHORS',f'/anchors/{i}',origin=row.get('origin_SOURCE',''),reuse='DECLARED_PROFILE_LIMITS_NOT_BINARY_EQUIVALENCE')
    h6raw=source('H6_OLD_METRICS',B+'track_a_h6_exploration_summary_v1/2026-10-12/wrapper_metrics_v1.csv','df99d6ee28942df8d12aac3c4db7cd202df55d77',fmt='csv')
    for i,row in enumerate(h6raw):
        if row['control']=='symmetric_directional':add('H6_DF19',row['cell_id'],row['axis'],row,'H6_OLD_METRICS',f'CSV_RECORD:{i+1}',origin=row['source_commit'],reuse='OLD_PROFILE_BYTES_UNRESOLVED_REACQUIRE_IF_NOT_RECOVERED')
    b0i=source('H6_B0I_METRICS',B+'track_a_h6_native_b0i_full_family_result_v1/2026-10-15/axis_compiled_metrics_v1.csv',fmt='csv')
    for i,row in enumerate(b0i):add('H6_DF19',f"B0I_p{row['prefix']}_q{row['q']}_2nd",row['axis'],row,'H6_B0I_METRICS',f'CSV_RECORD:{i+1}',origin=row['origin_SOURCE'],reuse='DECLARED_PROFILE_LIMITS_NOT_BINARY_EQUIVALENCE')
    v1=source('H6_S4_VALIDATION',B+'track_a_v1_phase_b_run01/2026-10-16/axis_validation.csv',fmt='csv')
    for i,row in enumerate(v1):add('H6_DF19',row['cell_id'],row['axis'],row,'H6_S4_VALIDATION',f'CSV_RECORD:{i+1}',origin='fcf37e6794e2e51fc95e958a75fc8f75fb049423')
    for axis in ('cosine','sine'):
        label='H6_P7_'+axis.upper();row=source(label,B+f'track_a_v1_p7_compile_run01/2026-10-16/{axis}_measurement.json')
        add('H6_DF19',row['cell_id'],axis,row,label,'/',origin='84b9ea22bf78c7193f2137ce42a04400292b8e68')
    h6d=source('H6_D_LATEST',B+'track_a_h6_dba/2026-10-18/completion/D_terminal_result.json','648184eb')
    assert h6d['native_exit']['exit_code']==0
    for i,row in enumerate(h6d['result']['rows']):add('H6_DF19',h6d['result']['cell_id'],row['axis'],row,'H6_D_LATEST',f'/result/rows/{i}',origin=h6d['SOURCE'])
    h8account=source('H8_SAVED_ACCOUNTING',B+'track_a_h8_six_axis_reaccount_run01/2026-10-17/accounting_rows.jsonl.gz',fmt='jsonl')
    # Axis prices are copied once per distinct saved price/status/provenance, never averaged.
    seen=set()
    for i,row in enumerate(h8account):
        if row['bias_kind']!='USUAL_SIGNED_SAVED':continue
        for a,m in row['updated'].get('axis_costs',{}).items():
            cid=row['updated']['cell_id'];key=(cid,a,json.dumps(m,sort_keys=True))
            if key in seen:continue
            seen.add(key);prov=m.get('provenance',{})
            add('H8_DF28',cid,a,dict(prov,**m),'H8_SAVED_ACCOUNTING',f'JSONL_RECORD:{i+1}/updated/axis_costs/{a}',kind='DIRECT' if m.get('measured') else 'MODEL',origin=prov.get('origin_SOURCE',''),reuse='LEGACY_PROFILE_UNKNOWN_REACQUIRE_IF_NOT_RECOVERED' if m.get('origin')=='LEGACY_FULL_PF_DECLARED_LIMITS' else 'DECLARED_SAVED_PROVENANCE_RECHECK_REQUIRED' if m.get('measured') else 'NOT_A_MEASUREMENT')
            if 'previous_unit_evidence' in m:
                old=m['previous_unit_evidence']
                assert old.get('measured') is False
                add('H8_DF28',cid,a,old,'H8_SAVED_ACCOUNTING',f'JSONL_RECORD:{i+1}/updated/axis_costs/{a}/previous_unit_evidence',kind='MODEL',reuse='RETAINED_PRE_DIRECT_MODEL_VALUE')
    h8model=source('H8_MODEL',B+'track_a_h8_phase_b1_run01/2026-10-17/frozen_models_v1.json')
    # Fit anchors remain direct observations even when absent from eligible accounting rows.
    for i,m in enumerate(h8model['models']):
        for j,row in enumerate(m['checks']):
            cid,axis=row['id'].rsplit('_',1)
            add('H8_DF28',cid,axis,dict(row,compiler_hash=h8model['compiler_hash'],profile_sha256=h8model['profile_sha256']),'H8_MODEL',f'/models/{i}/checks/{j}',reuse='ANCHOR_OBSERVATION_PROVENANCE_RECHECK_REQUIRED')
    h8six=source('H8_SIX_DIRECT',B+'track_a_h8_six_axis_reaccount_run01/2026-10-17/six_axis_provenance.json')
    for i,row in enumerate(h8six['selected_axes']):add('H8_DF28',row['case_id'],row['axis'],row,'H8_SIX_DIRECT',f'/selected_axes/{i}')
    for cid in ('B0_p15_q8_2nd','B0I_p16_q4_4th'):
        label='H8_D_'+cid;d=source(label,B+f'track_a_h8_dba/2026-10-18/completion/{cid}_result.json')
        for i,row in enumerate(d['rows']):add('H8_DF28',cid,row['axis'],row,label,f'/rows/{i}',origin=d['SOURCE'])
    legacy=source('H4_LEGACY_SAVED',B+'track_a_h4_dp_phase0/2026-10-19/saved_measurements.json','504f3b65023e9f3f3497b3881878b90be154d417')
    lc=source('H4_LEGACY_COVERAGE',B+'track_a_h4_dp_phase0/2026-10-19/deterministic_coverage.csv','504f3b65023e9f3f3497b3881878b90be154d417',fmt='csv')
    cov={r['candidate_id']:r for r in lc}
    for cid,axes in legacy.items():
        for a,m in axes.items():add('H4_LEGACY_DF12',cid,a,m,'H4_LEGACY_SAVED',f'/{cid}/{a}',origin=cov[cid]['measurement_SOURCE'])
    ln=source('H4_LEGACY_NEW',B+'track_a_h4_d_completed/2026-10-19/new_native_measurements.json','4e8ad1936363266d1b55bb689c36846e6c3dfcdb')
    for i,row in enumerate(ln):
        for a,m in row['axes'].items():add('H4_LEGACY_DF12',row['cell']['id'],a,m,'H4_LEGACY_NEW',f'/{i}/axes/{a}',origin=row['SOURCE'])
    source('H4_LEGACY_COMPLETION_REPORT','docs/research/track_a_h4_d_completed_20261019.md','4e8ad1936363266d1b55bb689c36846e6c3dfcdb',fmt='bytes')
    # Reusable implementation identities only: reading code bytes does not import it.
    for label,commit,path in [
        ('H2_COMPILE_CODE','cf99e4452df96291d7e9d37d76d2e91be5cbc413','scripts/h2_d_recovery_science.py'),
        ('H4_COMPILE_CODE','84db3909639b34ec521004c3d8f7ef9b5e6ad851','scripts/h4_tol_main_science.py'),
        ('H4_LEGACY_COMPILE_CODE','8a82b9272ee24896e7d7cbb7400d9b8c12d8f804','scripts/h4_dp_science.py'),
        ('H6_COMPILE_CODE','579d03c9c8a2b1ce7c22eb751a08642e2d813c3c','scripts/h6_dba_native.py'),
        ('H8_COMPILE_CODE','49879a9c73477c49081a3059cec70e4f4c6a5924','scripts/h8_dba_direct.py')]:
        source(label,path,commit,fmt='bytes')
    resource_receipts={
        'H4_D_STAGE':source('H4_D_COMPLETION',B+'track_a_h4_tol_d_completed/2026-10-19/completion_audit.json'),
        'H6_S4_STAGE':source('H6_S4_COMPLETION',B+'track_a_v1_phase_b_run01/2026-10-16/completion_audit.json'),
        'H6_P11_S4_STAGE':h6d['native_exit'],
        'H8_B0_P15_Q8_STAGE':source('H8_B0_D_TERMINAL',B+'track_a_h8_dba/2026-10-18/completion/B0_p15_q8_2nd_terminal.json'),
        'H8_B0I_P16_Q4_STAGE':source('H8_B0I_D_TERMINAL',B+'track_a_h8_dba/2026-10-18/completion/B0I_p16_q4_4th_terminal.json')}
    # Fixed input/profile and model identity records, copied verbatim into separate metadata.
    bindings={
        'H2_DF3':h2['binding'],'H4_DF10':h4['binding'],
        'H6_DF19':source('H6_BINDING',B+'track_a_h6_dba/2026-10-18/input_binding_v1.json'),
        'H8_DF28':source('H8_BINDING',B+'track_a_h8_dba/2026-10-18/saved_binding_v1.json'),
        'H4_LEGACY_DF12':source('H4_LEGACY_BINDING',B+'track_a_h4_native_validation/2026-10-18/input_binding.json','4e8ad1936363266d1b55bb689c36846e6c3dfcdb')}
    h6model=source('H6_MODEL',B+'track_a_v1_phase_b_readiness/2026-10-16/frozen_model.json')
    # Do not run old fit or prediction functions. Formula strings/coefficients only.
    models=[]
    for system,d,label in [('H6_DF19',h6model,'H6_MODEL'),('H8_DF28',h8model,'H8_MODEL')]:
        for i,m in enumerate(d['models']):
            stratum=m.get('stratum',m)
            models.append(dict(model_id=f'{system}_{i+1}',system=system,method=stratum['method'],axis=stratum['axis'],
                S2_formula='a+b*q+d*prefix*q',S4_formula='a+3*b*q+3*d*prefix*q',
                coefficients_literal=json.dumps(m['coefficients']),source_id=label,source_locator=f'/models/{i}',
                evaluated_this_task=False,fit_this_task=False))
    grouped={}
    for row in evidence:grouped.setdefault((row['system'],row['candidate_id'],row['axis']),[]).append(row)
    inventory=[];axis_inventory=[]
    for (s,cid),cell in cells.items():
        axis_status=[]
        for axis in ('cosine','sine'):
            rs=grouped.get((s,cid,axis),[]);direct=[r for r in rs if r['kind']=='DIRECT'];est=[r for r in rs if r['kind']=='MODEL']
            # Several receipts may describe the same measurement. They must never count as extra axes.
            prices={r['RZ_literal'] for r in direct if r['RZ_literal']}
            assert len(prices)<=1,('CONFLICTING_DIRECT_PRICES',s,cid,axis,prices)
            status='DIRECT_OBSERVED' if direct else 'MODEL_ONLY' if est else 'NO_ACCEPTED_AXIS_EVIDENCE_IN_FIXED_SOURCES'
            axis_status.append(status)
            axis_inventory.append(dict(system=s,candidate_id=cid,axis=axis,PF=cell['formula'],prefix=cell['prefix'],q=cell['q'],
                status=status,direct_RZ_literal=next(iter(prices),'') if prices else '',
                direct_evidence_ids=';'.join(r['evidence_id'] for r in direct),model_evidence_ids=';'.join(r['evidence_id'] for r in est),
                model_formula='a+b*q+d*prefix*q' if cell['formula']=='2nd' and cell['method'] in ('B0','B0I') and s in ('H6_DF19','H8_DF28') else 'a+3*b*q+3*d*prefix*q' if cell['method'] in ('B0','B0I') and s in ('H6_DF19','H8_DF28') else 'NONE_REGISTERED_IN_THIS_INPUT_SERIES',
                compiler_id='QISKIT130_RZSXXCX_OPT1_SEED17',control='symmetric_directional',
                input_binding_ref='system_bindings.json#'+s,
                reuse_status=';'.join(sorted({r['reuse_assessment'] for r in direct})) if direct else 'NEW_COMPILE_REQUIRED_AFTER_APPROVAL',
                future_circuit_identity_checked=False,execution_authorized=False))
        n=axis_status.count('DIRECT_OBSERVED')
        inventory.append(dict(system=s,candidate_id=cid,method=cell['method'],prefix=cell['prefix'],q=cell['q'],PF=cell['formula'],
            direct_axes=n,status='BOTH_AXES_DIRECT' if n==2 else 'PARTIAL_DIRECT' if n else 'NO_DIRECT_AXIS_FOUND',
            model_axes=axis_status.count('MODEL_ONLY'),missing_axes=2-n,
            recommended_action='REUSE_ONLY_IF_ALL_IDENTITY_GATES_MATCH' if n==2 else 'COMPILE_MISSING_AXES_REGARDLESS_OF_PRECISION_ELIGIBILITY',
            priority='FIRST_H6_EPS005' if s=='H6_DF19' and cid=='B0_p11_q4_2nd' else 'ALL_REGISTERED_Q_INCLUDED',
            future_fingerprint_checked=False))
    comparisons=source('SAVED_D_COMPARISONS',B+'track_a_pr_cost_batch_results/2026-10-18/d_saved_comparison_summary.json')
    mixed=[]
    for system,d in comparisons.items():
        for i,r in enumerate(d['comparisons']):
            if r['factor']!='1.0' and r['factor']!=1:continue
            mixed.append(dict(system=system,series='H4_LEGACY_DF12' if system=='H4' else system+'_DF'+('19' if system=='H6' else '28'),
                epsilon=r['epsilon'],bias_mode=r['bias_mode'],comparison_scope=r['scope'],
                saved_selected_PF=r['deterministic']['cell_id'],saved_cost_status=r['deterministic']['cost_status'],
                whole_domain_direct=False,selection_recomputed=False,
                source_id='SAVED_D_COMPARISONS',source_locator=f'/{system}/comparisons/{i}'))
    counts=[]
    for s,rs in systems.items():
        inv=[r for r in inventory if r['system']==s];ax=[r for r in axis_inventory if r['system']==s]
        counts.append(dict(system=s,registered_candidates=len(rs),direct_both=sum(r['direct_axes']==2 for r in inv),
            partial_direct=sum(r['direct_axes']==1 for r in inv),no_direct_candidate=sum(r['direct_axes']==0 for r in inv),
            direct_axes=sum(r['status']=='DIRECT_OBSERVED' for r in ax),model_only_axes=sum(r['status']=='MODEL_ONLY' for r in ax),
            no_axis_evidence=sum(r['status']=='NO_ACCEPTED_AXIS_EVIDENCE_IN_FIXED_SOURCES' for r in ax),
            mandatory_new_compile_axes=sum(r['missing_axes'] for r in inv),
            all_q_retained=True,counts_are_evidence_inventory_not_new_science=True))
    return dict(inventory=inventory,axis_inventory=axis_inventory,evidence=evidence,counts=counts,
        bindings=bindings,models=models,mixed=mixed,resource_receipts=resource_receipts)


def build():
    data=collect();OUT.mkdir(parents=True,exist_ok=True)
    for key,file in [('inventory','candidate_inventory.csv'),('axis_inventory','axis_inventory.csv'),('evidence','axis_evidence.csv'),('counts','inventory_counts.csv'),('models','model_registry.csv'),('mixed','saved_mixed_comparisons.csv')]:write_csv(file,data[key])
    write_json('system_bindings.json',data['bindings']);write_json('saved_resource_receipts.json',data['resource_receipts']);write_json('source_inventory.json',list(SOURCES.values()))
    write_json('MANDATORY_STOP.json',dict(status='MANDATORY_STOP_PLAN_ONLY',scientific_calls=0,compile=0,sampling=0,fit=0,cost_reaccounting=0,detailed_scientific_review=False,final_evidence_freeze=False,execution_authorized=False,grant_issued=False))
    print(json.dumps(data['counts'],indent=2))


def check():
    data=collect()
    for key,file in [('inventory','candidate_inventory.csv'),('axis_inventory','axis_inventory.csv'),('evidence','axis_evidence.csv'),('counts','inventory_counts.csv'),('models','model_registry.csv'),('mixed','saved_mixed_comparisons.csv')]:
        actual=list(csv.DictReader((OUT/file).open()));expected=[{k:str(v) for k,v in row.items()} for row in data[key]]
        assert actual==expected,file
    assert json.loads((OUT/'system_bindings.json').read_text())==data['bindings']
    assert json.loads((OUT/'saved_resource_receipts.json').read_text())==data['resource_receipts']
    assert json.loads((OUT/'source_inventory.json').read_text())==list(SOURCES.values())
    assert len(data['inventory'])==684 and len(data['axis_inventory'])==1368
    special=[r for r in data['axis_inventory'] if r['system']=='H6_DF19' and r['candidate_id']=='B0_p11_q4_2nd']
    assert len(special)==2 and all(r['status']=='MODEL_ONLY' for r in special)
    for r in special:
        assert any(e['RZ_literal']=='433317/5' and e['kind']=='MODEL' for e in data['evidence'] if e['evidence_id'] in r['model_evidence_ids'].split(';'))
    for c in data['counts']:
        assert c['direct_both']+c['partial_direct']+c['no_direct_candidate']==c['registered_candidates']
        assert c['direct_axes']+c['model_only_axes']+c['no_axis_evidence']==2*c['registered_candidates']
    # No tracked previous evidence or scientific SOURCE may be changed.
    allowed={'PROJECT_MAP.md','VALIDATION_STATUS.md','docs/README.md','docs/research/README.md','docs/research/研究概要・現状.md','scripts/README.md','src/trotterlib/README.md','artifacts/validation_manifest.json'}
    modified={p for p in git('diff','--name-only','-z',BASE).decode().split('\0') if p}
    assert all(p in allowed or p.startswith(B+'track_a_pf_direct_compile_plan/') or p.startswith('docs/research/track_a_pf_direct_compile') or p=='scripts/prepare_pf_direct_compile_inventory.py' or p=='docs/research/研究ノート/2026-10-19_pf_direct_compile_plan.md' for p in modified),modified
    for path in allowed-{'artifacts/validation_manifest.json'}:
        assert (ROOT/path).read_bytes().endswith(git('show',BASE+':'+path)),('OLD_INDEX_TEXT_CHANGED',path)
    old=json.loads(git('show',BASE+':artifacts/validation_manifest.json'))
    new=json.loads((ROOT/'artifacts/validation_manifest.json').read_text())
    assert set(new)-set(old)=={'track_a_pf_direct_compile_plan_20261019'}
    assert all(new[k]==v for k,v in old.items()),'OLD_VALIDATION_ENTRY_CHANGED'
    remote=json.loads((OUT/'github_read_receipt.json').read_text())
    assert len(remote['files'])==12
    byid={r['id']:r for r in SOURCES.values()}
    for r in remote['files']:
        assert r['exact_fixed_Git_object_match'] is True and r['sha256']==byid[r['source_id']]['sha256']
    for name in ('execution_contract_DRAFT.json','future_data_schema.json','publication_scope.json'):
        assert (OUT/name).is_file(),name
    contract=json.loads((OUT/'execution_contract_DRAFT.json').read_text());assert not contract['execution_authorized'] and not contract['grant_issued']
    tasks=list(csv.DictReader((OUT/'proposed_compile_axes.csv').open()))
    required={(r['system'],r['candidate_id'],r['axis']) for r in data['axis_inventory'] if r['status']!='DIRECT_OBSERVED'}
    assert {(r['system'],r['candidate_id'],r['axis']) for r in tasks}==required
    assert all(r['execution_authorized']=='False' for r in tasks)
    axis_lookup={(r['system'],r['candidate_id'],r['axis']):r for r in data['axis_inventory']}
    for task in tasks:
        original=axis_lookup[(task['system'],task['candidate_id'],task['axis'])]
        assert all(task[k]==str(original[k]) for k in ('prefix','q','PF'))
        assert task['precision_ineligibility_does_not_exclude']=='True'
    conditional=list(csv.DictReader((OUT/'conditional_reacquisition_axes.csv').open()))
    expected_conditional={(r['system'],r['candidate_id'],r['axis']) for r in data['axis_inventory'] if r['status']=='DIRECT_OBSERVED' and any(s in r['reuse_status'] for s in ('OLD_PROFILE','LEGACY_PROFILE'))}
    assert {(r['system'],r['candidate_id'],r['axis']) for r in conditional}==expected_conditional
    assert len(conditional)==48 and len(tasks)==788
    assert contract['proposed_axes']==len(tasks) and contract['optional_reacquisition_axes']==len(conditional)
    assert contract['proposed_missing_only_call_totals']['wrapper_compile']==len(tasks)
    assert contract['parallelism']['worker_limit'] is None and contract['parallelism']['RAM_limit'] is None
    tree=ast.parse(Path(__file__).read_text())
    imports={n.names[0].name.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.Import)}
    assert imports<={'argparse','ast','csv','gzip','hashlib','io','json','re','subprocess'}
    assert all(n.module=='pathlib' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('eval','exec','compile','__import__') for n in ast.walk(tree))
    if (OUT/'artifact_manifest.json').is_file():
        m=json.loads((OUT/'artifact_manifest.json').read_text())
        for path,digest in m['files'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
        assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==m['builder_sha256']
    report=dict(status='PASS_METADATA_STRUCTURE_ONLY',candidate_count=684,axis_count=1368,source_files=len(SOURCES),
        literal_transcription_and_source_joins=True,inventory_counts_only=True,new_scientific_calls=0,new_compile=0,
        new_model_predictions=0,new_fit=0,new_statistics=0,new_cost_reaccounting=0,
        future_circuit_fingerprint_check_performed=False,independent_scientific_review=False,
        final_evidence_freeze=False,primary_agent_self_check=True)
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',action='store_true');a=p.parse_args()
    if a.build:build()
    else:check()
