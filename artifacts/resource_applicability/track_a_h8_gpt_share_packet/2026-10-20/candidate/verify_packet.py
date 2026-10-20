"""Offline structural checks of redacted H8 evidence; no scientific imports or calculations."""
import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import re

SCOPES=['H8_CONFIRM_P8_Q2','H8_CONFIRM_P7_Q16']
SOURCE='745a674546f2cb6258526fbfde1d9a00601689f8'
ORIGIN='1549b6f891a15ff48379996c34b344118ba2606c'
HEX=re.compile(r'^[0-9a-f]{64}$')
LEDGER_KEYS={'kind','count','total','nonrefundable'}

def require(ok, message):
    if not ok:raise ValueError(message)

def digest(raw):return hashlib.sha256(raw).hexdigest()

def scan_text(text):
    patterns=[r'/(?:home|tmp|root|Users|mnt|srv|scratch|data)/',r'(?<![A-Za-z])[A-Za-z]:[\\/]',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',r'\bgh[pousr]_[A-Za-z0-9]{20,}',r'github_pat_[A-Za-z0-9_]{20,}',r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',r'\bBearer\s+[A-Za-z0-9._~+/-]{16,}',r'\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+',r'(?i)(?:password|api[_-]?key|access[_-]?token|secret[_-]?key)\s*[:=]\s*[\"\x27][A-Za-z0-9_+/=-]{12,}[\"\x27]']
    require(not any(re.search(p,text) for p in patterns),'PRIVATE_PATH_OR_CREDENTIAL_PATTERN')

def scan_json(obj):
    forbidden={'seed','seeds','seed_plan_digest','events','old_seed_values','excluded','password','credentials','access_token','api_key','pid','uid','identity','allowed_cpus','driver_cpu','directory','user_instruction','amplitudes','state_vector','matrix','raw_shots'}
    if isinstance(obj,dict):
        require(not (set(obj)&forbidden),'FORBIDDEN_PRIVATE_FIELD')
        for value in obj.values():scan_json(value)
    elif isinstance(obj,list):
        for value in obj:scan_json(value)
    elif isinstance(obj,str):
        scan_text(obj)
        if obj.lstrip().startswith('{'):
            try:embedded=json.loads(obj)
            except json.JSONDecodeError:return
            scan_json(embedded)

def semantic(data):
    results=data['01_results_and_source.json'];execution=data['02_execution_receipts.json'];native=data['03_worker_receipts.json'];contracts=data['04_grant_ledger_marker_stop.json'];seeds=data['05_seed_cohort_audit.json'];tasks=data['06_task_axis_links.json']['tasks'];bindings=data['07_input_runtime_bindings.json'];provenance=data['08_provenance_and_redactions.json'];sources=data['11_public_sources.json']
    artifacts={a['evidence_id']:a for a in provenance['artifacts']}
    require(len(artifacts)==len(provenance['artifacts']),'DUPLICATE_ARTIFACT')
    require(all(HEX.fullmatch(a['original_sha256']) and a['original_bytes']>0 for a in artifacts.values()),'ORIGINAL_ARTIFACT_HASH_FORMAT')
    def ref(r):
        require(r['evidence_id'] in artifacts,'UNKNOWN_ORIGINAL_REFERENCE')
        return artifacts[r['evidence_id']]
    full_receipts=set()
    def full(r):
        a=artifacts[r['evidence_id']];raw=r['original_json_utf8'].encode()
        require(a['original_bytes_in_packet'] is True and digest(raw)==a['original_sha256'] and len(raw)==a['original_bytes'],'EXACT_AUDIT_RECEIPT_SHA')
        require(json.loads(raw)==r['value'],'FULL_AUDIT_PARSED_FIELDS')
        scan_json(r['value']);full_receipts.add(r['evidence_id'])
        return r['value']
    def ledger(rows,expected):
        values=[full(r) for r in rows]
        require(all(set(v)==LEDGER_KEYS and v['nonrefundable'] is True and type(v['count']) is int and v['count']>0 for v in values),'LEDGER_ALLOWED_FIELDS')
        require({v['kind'] for v in values}==set(expected),'LEDGER_KIND_COVERAGE')
        for kind,total in expected.items():
            group=[v for v in values if v['kind']==kind]
            require(sum(v['count'] for v in group)==total and max(v['total'] for v in group)==total,'NONREFUNDABLE_RESERVATION_TOTAL')
    nodes={n['node_id']:n for n in native['nodes']}
    require(len(nodes)==len(native['nodes']),'DUPLICATE_NATIVE_NODE')
    for n in nodes.values():
        ref(n['origin'])
        if n['parent_id'] is not None:
            require(n['parent_id'] in nodes,'NATIVE_PARENT_MISSING')
            require(n['started_identity_alias']==n['node_id'] and n['locally_checked_identity_equal'] is True,'REDACTED_IDENTITY_LINK')
            ref(n['started_origin'])
        if 'result_evidence_id' in n:
            require(artifacts[n['result_evidence_id']]['original_sha256']==n['result_sha256'],'NATIVE_RESULT_SHA_LINK')
    for group in native['maps']:
        ref(group['origin']);require(group['parent_id'] in nodes,'MAP_PARENT')
        require(all(n in nodes and nodes[n]['parent_id']==group['parent_id'] for n in group['workers']),'MAP_WORKER_RELATION')
        require(group['fields']['all_reaped'] is True and not group['fields']['cleanup_errors'],'MAP_REAP_CLEANUP')
        if group['attempt']=='continuation':
            require(group['fields']['status']=='PASS' and group['fields']['first_error'] is None and group['fields']['automatic_retry'] is False,'CONTINUATION_NATIVE_PASS')
            require(all(nodes[n]['exit_code']==0 and nodes[n]['reaped'] is True for n in group['workers']),'CONTINUATION_NATIVE_EXIT')
    require({r['scope'] for r in results['conditions']}==set(SCOPES),'TWO_RESULT_CONDITIONS')
    batch=execution['batch']
    require(batch['SOURCE']==SOURCE and batch['mandatory_STOP'] is True and batch['automatic_retry'] is False and batch['next_stage_authorized'] is False,'BATCH_STOP')
    final_ids=[]
    for scope in SCOPES:
        r=next(x for x in results['conditions'] if x['scope']==scope);e=next(x for x in execution['scopes'] if x['scope']==scope);c=next(x for x in contracts['scopes'] if x['scope']==scope);b=next(x for x in bindings['scopes'] if x['scope']==scope)
        localtasks=[t for t in tasks if t['scope']==scope]
        require(r['n']==48 and r['axes']==96 and len(localtasks)==48 and {t['replica'] for t in localtasks}==set(range(48)),'N48_PAIR_COVERAGE')
        require(r['science_SOURCE']==e['SOURCE']==SOURCE and r['seed_derivation_SOURCE']==ORIGIN,'SCIENCE_SOURCE')
        require(e['startup']['status']=='PASS' and e['startup']['SOURCE']==SOURCE and e['startup']['technical_digest']==e['seal']['technical_digest'],'STARTUP_SEAL_BINDING')
        require(e['seal']['approved'] is False and e['seal']['runtime_authorization'] is False and e['seal']['technical_sealed'] is True,'TECHNICAL_SEAL_NOT_GRANT')
        d=nodes[e['driver_node']]
        require(d['exit_code']==0 and d['reaped'] is True and d['kind']=='driver','DRIVER_EXIT_REAP')
        terminal=full(e['terminal'])
        require(terminal['status']=='COMPLETE_MANDATORY_STOP' and terminal['SOURCE']==SOURCE and terminal['result_sha256']==artifacts[e['raw_result_evidence_id']]['original_sha256'] and terminal['retry'] is False and terminal['next_stage_authorized'] is False,'TERMINAL_RAW_RESULT')
        require(e['all_reaped'] is True and e['reused_task_count']==r['reused_complete_trajectories'],'NATIVE_REUSE')
        require(c['resume']['original_STOP_preserved'] is True and c['resume']['failed_or_unfinished_charge_refunded'] is False,'ORIGIN_CARRY_NOT_REFUNDED')
        require(c['result_carry_evidence_ids']==[x['evidence_id'] for x in c['carried_reservations']],'CARRY_RESULT_MATCH')
        for row in c['carried_reservations']:require(set(full(row))==LEDGER_KEYS,'CARRY_RECEIPT_SCHEMA')
        stop=full(c['origin_stop'])
        require(stop['reason'] in ('ValueError:NATIVE_START_IDENTITY','KeyboardInterrupt:OWN_RUN_SIGNAL_15') and stop['retry'] is False,'HISTORICAL_STOP_RETAINED')
        for item in c['grants']:
            g=item['grant'];a=item['approval'];s=item['seal'];marker=full(item['marker'])
            require(all(g[k] is True for k in ['approved','approved_by_user','runtime_authorization','one_shot']),'EXPLICIT_HISTORICAL_GRANT')
            require(g['scope']==s['scope']==scope and g['SOURCE']==s['SOURCE'] and g['technical_digest']==s['technical_digest'],'GRANT_SEAL_BINDING')
            require(g['retry'] is False and g['next_stage_authorized'] is False and a['approved_by_user'] is True,'GRANT_SCOPE_BOUNDARY')
            require(g['approval_receipt_sha256']==ref(item['approval_origin'])['original_sha256'],'APPROVAL_RECEIPT_HASH_LINK')
            require(a['n_per_candidate']==48 and a['wrapper_compiles']==192 and set(a['scopes'])==set(SCOPES) and a['design_digest']==s['design_digest'],'APPROVAL_SCOPE_SELECTION')
            require(marker['SOURCE']==g['SOURCE'] and marker['scope']==scope and marker['nonrefundable_attempt'] is True,'ONE_SHOT_MARKER')
            if item['attempt']=='origin':
                require(s['source_hashes']['scripts/h8_pr_confirmation.py']==sources['origin_source_pin']['sha256'],'ORIGIN_SOURCE_CLOSURE')
        ledger(c['parent_ledger'],c['parent_calls'])
        require(len(b['input_rows'])==(65 if scope==SCOPES[0] else 58),'INPUT_COVERAGE')
        for row in b['input_rows']:
            a=artifacts[row['origin_evidence_id']]
            require(row['sha256']==a['original_sha256'] and row['bytes']==a['original_bytes'],'INPUT_ORIGINAL_HASH_LINK')
        require(b['provider_closure']['expected_files']==931 and b['runtime_closure']['expected_files']==34273,'PROVIDER_RUNTIME_COVERAGE')
        require(b['runtime_closure']['runtime_receipt_sha256']==ref(b['runtime_closure']['runtime_receipt_origin'])['original_sha256'],'RUNTIME_RECEIPT_LINK')
        require(b['compiler_options']==dict(basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17),'RECORDED_COMPILER_OPTIONS')
        ref(b['compiler_config_origin'])
        public_source_hashes={x['path']:x['sha256'] for x in sources['previous_public_read_receipt']['files'] if x['commit']==SOURCE}
        require(all(public_source_hashes.get(path)==value for path,value in b['source_hashes'].items()),'CURRENT_SOURCE_CLOSURE')
        require(b['runtime_closure']['previous_check']['status']=='PASS' and b['provider_closure']['previous_check']['status']=='PASS','PRIOR_BYTE_CHECK')
        require(b['profile_reference']['original_profile_file_bytes_verified'] is False and b['profile_reference']['old_compiler_output_bit_identity_verified'] is False,'MISSING_PROFILE_NOT_UPGRADED')
        prefix=str(r['candidate']['prefix']);prep=b['preparation_binding']['prefixes'][prefix]
        require(prep['preparation_hash']==b['target_preparation_sha256']==b['preparation_receipt']['prefixes'][prefix]['preparation_hash'],'PREPARATION_BINDING')
        require(prep['saved_exact_export_match'] is (scope==SCOPES[1]),'EXPORT_MATCH_CAVEAT')
        reused=[]
        for t in localtasks:
            require(t['task_id']==t['candidate']+'_'+str(t['replica']) and t['candidate']==r['candidate']['id'],'TASK_CANDIDATE')
            n=nodes[t['node_id']];final_ids.append(t['task_id'])
            require(n['exit_code']==0 and n['reaped'] is True and n['task_id']==t['task_id'],'ACCEPTED_TASK_NATIVE')
            require(t['measurement_SOURCE']==(ORIGIN if t['reused_complete_trajectory'] else SOURCE),'TASK_MEASUREMENT_SOURCE')
            if t['reused_complete_trajectory']:reused.append(t['task_id'])
            require(t['synthetic'] is False and t['normalization_unchanged'] is True and t['trace_flags']['conditional'] is False and t['trace_flags']['quantum_weights_changed'] is False,'UNCONDITIONAL_COHORT_FLAGS')
            require(t['cohort']=='INDEPENDENT_UNCONDITIONAL_CONFIRMATION_N48' and t['law_digest']==t['expected_law_digest'],'COHORT_LAW_LINK')
            require(ref(t['task_result_origin'])['original_sha256']==n['result_sha256'],'TASK_RESULT_HASH')
            require(set(t['axes'])=={'cosine','sine'},'EXACT_PAIR')
            ledger(t['ledger'],t['calls'])
            for axis,row in t['axes'].items():
                m=row['metrics'];require(m['event_digest']==t['event_digest'] and m['compiler_hash']==b['binding']['compiler'] and HEX.fullmatch(m['fingerprint']),'AXIS_INPUT_COMPILER_DIGEST')
                children=[x for x in nodes.values() if x['parent_id']==t['node_id'] and x['kind']=='axis' and x['task_id']==axis]
                if not t['reused_complete_trajectory']:
                    require(len(children)==1 and children[0]['axis_metrics']==m,'AXIS_NATIVE_METRIC_EQUALITY')
            if t['replica']==0:
                control=t['control']['fields']
                require(control['status']=='PASS' and control['tolerance']==1e-8 and 0<=control['max_error']<1e-8 and control['control_probes']==4 and control['independent_event_oracles']==1,'SAVED_CONTROL_GATE')
                require(control['full_Fock_scalar_once'] is True and control['reference_rotated'] is False and control['ordinary_ancilla_branch'] is True,'CONTROL_CONVENTIONS')
            if t['task_id'] in c['previous_event_digests']:
                require(t['event_digest']==c['previous_event_digests'][t['task_id']],'PREVIOUS_EVENT_STREAM')
        require(set(reused)==set(c['accepted_task_ids']) and len(reused)==r['reused_complete_trajectories'],'ACCEPTED_REUSE_POLICY')
        for mode,claim in r['saved_result_claims'].items():
            require(claim['old_samples_pooled'] is False and claim['confidence_interval'] is None and claim['p_value'] is None,'SAVED_LIMITS')
    require(len(tasks)==96 and len(set(final_ids))==96,'ALL96_FINAL_TASKS')
    require(seeds['mathematical_independence_proven'] is False and seeds['declaration']['old_cohorts_not_reused'] is True and seeds['declaration']['redraw_on_collision'] is False,'SEED_SCOPE_LIMIT')
    require(seeds['seed_plan_derivation_key_withheld'] is True and 'seed_plan_digest' not in seeds['declaration'],'SEED_DERIVATION_INPUT_WITHHELD')
    require({t['id'] for t in seeds['registered_tasks']}==set(final_ids),'REGISTERED_TASK_COVERAGE')
    a=seeds['prior_local_audit']
    require(a['old_excluded_values']==4136 and a['A_registered_masters']==128 and a['merged_excluded_values']==5000 and a['registered_confirmation_tasks']==96,'EXCLUSION_AUDIT_SCOPE')
    require(all(r['masters_missing']==0 and r['derived_values_missing']==0 for r in a['old_registration_checks']),'OLD_REGISTRATION_AUDIT')
    require(provenance['new_scientific_calls']==provenance['new_sampling']==provenance['new_compile']==provenance['new_statistics']==0 and provenance['result_recalculation'] is False and provenance['scientific_validity_review_started'] is False and provenance['final_evidence_freeze'] is False,'NO_SCIENCE_OR_FREEZE')
    require(provenance['actual_external_sharing_authorized'] is False and provenance['actual_external_sharing_performed'] is False,'PREPARATION_ONLY')
    require({r['evidence_id'] for r in artifacts.values() if r['original_bytes_in_packet']}==full_receipts,'EXACT_AUDIT_BYTE_COVERAGE')
    return dict(status='PASS_PACKET_STRUCTURE_ONLY',conditions=2,paired_trajectories=96,axes=192,native_nodes=len(nodes),exact_audit_receipts_sha_verified=len(full_receipts),original_artifact_links=len(artifacts),new_scientific_calls=0,original_authenticity_proven=False,mathematical_independence_proven=False,final_evidence_freeze=False)

def validate(root):
    root=Path(root);manifest=json.loads((root/'packet_manifest.json').read_text());files=manifest['files']
    require(manifest['publication_status']=='LOCAL_CANDIDATE_NOT_SENT' and manifest['self_excluded'] is True,'MANIFEST_SHARING_BOUNDARY')
    expected={r['filename'] for r in files}|{'packet_manifest.json'}
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    require(expected==actual,'PACKET_FILE_COVERAGE')
    data={}
    for row in files:
        p=Path(row['filename']);require(not p.is_absolute() and '..' not in p.parts,'SAFE_RELATIVE_PATH')
        raw=(root/p).read_bytes();require(digest(raw)==row['sha256'] and len(raw)==row['bytes'],'PACKET_FILE_SHA')
        text=raw.decode();scan_text(text)
        if p.suffix=='.json':
            obj=json.loads(text);scan_json(obj);data[str(p)]=obj
    for f in data['11_public_sources.json']['public_files_in_packet']:
        raw=(root/f['packet_file']).read_bytes()
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        require(digest(raw)==f['sha256'] and blob==f['git_blob_sha'],'FIXED_PUBLIC_FILE_IDENTITY')
    public_rows=list(csv.DictReader(io.StringIO((root/'public/paired_metrics.csv').read_text())))
    comparisons=list(csv.DictReader(io.StringIO((root/'public/comparisons.csv').read_text())))
    task_rows=[]
    for t in data['06_task_axis_links.json']['tasks']:
        for axis,item in t['axes'].items():
            m=item['metrics'];row=dict(scope=t['scope'],candidate=t['candidate'],replica=str(t['replica']),axis=axis,cohort=t['cohort'],reused_complete_trajectory=str(t['reused_complete_trajectory']))
            row.update({k:str(m[k]) for k in ['RZ','CX','depth','RZ_depth','CX_depth','size','wall_seconds']});task_rows.append(row)
    order=lambda r:(r['scope'],int(r['replica']),r['axis'])
    require(sorted(task_rows,key=order)==sorted(public_rows,key=order),'PUBLIC192_PROJECTION_MATCH')
    saved=[row for r in data['01_results_and_source.json']['conditions'] for row in r['saved_comparison_rows']]
    require(saved==comparisons,'SAVED_COMPARISON_TOKENS')
    return semantic(data),data

def self_test(data):
    tests=[]
    def reject(name,edit):
        mutated=copy.deepcopy(data);edit(mutated)
        try:semantic(mutated)
        except (ValueError,KeyError):tests.append(dict(name=name,status='PASS_REJECTED'))
        else:raise ValueError('MUTATION_NOT_REJECTED:'+name)
    reject('driver nonzero exit',lambda d:d['03_worker_receipts.json']['nodes'][0].update(exit_code=7))
    reject('worker not reaped',lambda d:next(n for n in d['03_worker_receipts.json']['nodes'] if n['kind']=='axis').update(reaped=False))
    reject('grant digest mismatch',lambda d:d['04_grant_ledger_marker_stop.json']['scopes'][0]['grants'][0]['grant'].update(technical_digest='0'*64))
    reject('paired axis removed',lambda d:d['06_task_axis_links.json']['tasks'][0]['axes'].pop('sine'))
    reject('refunded historical carry',lambda d:d['04_grant_ledger_marker_stop.json']['scopes'][0]['resume'].update(failed_or_unfinished_charge_refunded=True))
    reject('profile gap erased',lambda d:d['07_input_runtime_bindings.json']['scopes'][0]['profile_reference'].update(original_profile_file_bytes_verified=True))
    for name,obj in [('seed value',{'seed':123456789}),('seed derivation input',{'seed_plan_digest':'0'*64}),('private path',{'x':'/'+'home'+'/example/private'}),('credential',{'x':'ghp_'+'x'*30})]:
        try:scan_json(obj)
        except ValueError:tests.append(dict(name=name,status='PASS_REJECTED'))
        else:raise ValueError('PRIVACY_MUTATION_NOT_REJECTED:'+name)
    return tests

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,default=Path(__file__).parent);parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    result,data=validate(args.directory)
    if args.self_test:result['mutation_checks']=self_test(data)
    print(json.dumps(result,ensure_ascii=False,indent=2))
