"""Stdlib saved-record audit only. Never import any scientific backend."""
import builtins
import hashlib
import json
from pathlib import Path
import stat
import subprocess

ROOT = Path(__file__).resolve().parents[4]
SOURCE = 'b228f2307f5fea77f066ee13e11ba6d2d8b9bea7'
AUTH = 'a699a74dbf7a10a4e43c32ed6b6d3cb9d402a708'
SEAL = 'd2b1511cd87ee51e2a08d0f7c5025e55e920745d'
META = 'artifacts/resource_applicability/track_a_ax2b_h4_limited_execution_v2/2026-10-10'
OUTPUT = 'artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2'
PREPARATION = 'artifacts/resource_applicability/track_a_ax2b_h4_limited_seal_v2/2026-10-10'
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy','scipy','mpmath','qiskit','openfermion','pyscf'}:
        raise RuntimeError('NUMERICAL_IMPORT_FORBIDDEN_IN_SAVED_AUDIT:' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded

def sha(data):return hashlib.sha256(data).hexdigest()
def git_blob(path, commit):return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
def strict_json(data):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('DUPLICATE_JSON_KEY')
            result[key]=value
        return result
    def invalid(value):raise ValueError('NONFINITE_JSON:' + value)
    return json.loads(data,object_pairs_hook=pairs,parse_constant=invalid)

def main():
    directory=ROOT/OUTPUT
    assert not directory.is_symlink()
    manifest_bytes=(ROOT/PREPARATION/'sealed_preparation_manifest_v2.json').read_bytes()
    manifest=strict_json(manifest_bytes)
    assert manifest_bytes==git_blob(PREPARATION+'/sealed_preparation_manifest_v2.json',SEAL)
    caps=manifest['plan']['caps_proposed']
    paths=list(directory.iterdir())
    assert all(stat.S_ISREG(path.lstat().st_mode) for path in paths)
    assert sum(path.stat().st_size for path in paths)<=caps['output_bytes']
    blobs={path.name:path.read_bytes() for path in paths}
    rows, parse_errors = {}, {}
    for name,data in blobs.items():
        if name.endswith('.json'):
            try:rows[name]=strict_json(data)
            except (ValueError,UnicodeError) as error:
                parse_errors[name]=type(error).__name__+':'+str(error)[:200]
    assert blobs['frozen_preparation.json']==manifest_bytes
    auth_bytes=(ROOT/META/'authorization_v2.json').read_bytes()
    assert blobs['authorization.json']==auth_bytes==git_blob(META+'/authorization_v2.json',AUTH)
    auth=strict_json(auth_bytes)
    assert sha(auth_bytes)=='5732dba41c30783f9e57c2b8d91af84ea2531796e15469b64ede9a07304b43d3'
    assert auth['source_commit']==SOURCE and auth['caps']==caps and auth['approved_by_user'] is True
    assert auth['retry'] is auth['resume'] is False and auth['exclusive_output']==str(directory)
    assert rows['worker_claim.json']['assigned_cpu']==3 and rows['worker_claim.json']['resume'] is False
    assert rows['launch_binding.json']['manifest_digest']==rows['worker_claim.json']['manifest_digest']==auth['manifest_digest']
    parent=rows['terminal_status.json'];worker=rows.get('worker_terminal.json')
    assert parent['mandatory_stop'] is True and parent['next_stage_authorized'] is False and parent['N'] is parent['G'] is None
    assert parent['status'] in ('H4_LIMITED_COMPLETE','H4_LIMITED_STOP') and parent['retry'] is False
    assert parent['worker_log_bytes']==len(blobs['worker.log'])<=caps['log_bytes']
    if worker is not None:
        assert parent['worker_terminal']==worker and worker['mandatory_stop'] is True
        assert worker['N'] is worker['G'] is None and worker['numerical_allowance_certified'] is False
        assert worker['accuracy_eligibility']=='UNDETERMINED' and worker['next_stage_authorized'] is False
        for name,value in worker['calls'].items():assert type(value) is int and 0<=value<=caps[name]
        assert all(worker['calls'][name]==0 for name in ('compile','trajectory','occurrence'))
    comparison=rows['coverage_comparison_v3.json']
    assert comparison['equal'] is True and comparison['expected_sha256']==comparison['actual_sha256']
    assert comparison['differences']==[]
    assert rows['actual_coverage.json']==manifest['coverage_binding']['expected_bounds']
    reference=rows['input_reference.json'];primitives=rows['primitive_validation.json']
    assert reference['reference_matvecs']==36 and reference['ground_state_certified'] is False
    assert primitives['actual_time_count']==179 and primitives['probe_count']==3 and primitives['certified'] is False
    cells=manifest['plan']['cells'];correctness={}
    mp=[]
    for cell in cells:
        name=cell['id']+'_correctness.json'
        if name in rows:
            record=rows[name];assert record['cell']==cell
            assert record['N'] is record['G'] is None and record['numerical_allowance_certified'] is False
            assert record['accuracy_eligibility']=='UNDETERMINED'
            assert all(cell['id']+'_mp'+str(dps)+'.json' in rows for dps in manifest['plan']['dps'])
            correctness[cell['id']]={'classical_cell_wall_seconds':record['classical_cell_wall_seconds'],
                'worker_peak_rss_bytes_at_cell_end':record['worker_peak_rss_bytes'],
                'reference_difference_mp80':record['reference_difference_mp80'],
                'reference_difference_mp120':record['reference_difference_mp120'],
                'signals':record['signals'],'action_counts':record['action_counts'],'log_B':record['log_B'],
                'precision_comparison':record['precision_comparison']}
        for dps in manifest['plan']['dps']:
            mname=cell['id']+'_mp'+str(dps)+'.json'
            if mname in rows:
                assert rows[mname]['dps']==dps and rows[mname]['certified'] is False
                mp.append(mname)
    events=[id_+'_explicit_order'+str(order)+'.json' for id_ in ('H4_B2_K2','H4_B3_K6') for order in (0,2)]
    for name in events:
        if name in rows:assert rows[name]['sampling_performed'] is False and rows[name]['full_molecular_event_mean_enumerated'] is False
    expected={'authorization.json','frozen_preparation.json','launch_binding.json','worker_claim.json','worker.log',
              'actual_coverage.json','coverage_comparison_v3.json','input_reference.json','primitive_validation.json',
              'phase_input_reference.json','phase_correctness.json','phase_wrapper_cost.json','terminal_status.json','worker_terminal.json'}
    expected.update(cell['id']+'_correctness.json' for cell in cells)
    expected.update(cell['id']+'_mp'+str(dps)+'.json' for cell in cells for dps in manifest['plan']['dps'])
    expected.update(events)
    assert set(blobs).issubset(expected)
    missing=sorted(expected-set(blobs))
    if worker is not None:assert worker['completed_correctness_cells']==len(correctness) and worker['compiled_wrappers']==0
    if parent['status']=='H4_LIMITED_COMPLETE':
        assert worker is not None and len(correctness)==8 and len(mp)==16 and not missing and not parse_errors
        assert worker['calls']['primitive']==537 and worker['calls']['reference_matvec']==36 and worker['calls']['control_probe']==100
    freeze=strict_json((ROOT/PREPARATION/'preparation_source_freeze_v2.json').read_bytes())
    for hashes in (manifest['source_hashes'],freeze['source_hashes']):
        for path,expected_hash in hashes.items():assert sha((ROOT/path).read_bytes())==sha(git_blob(path,SOURCE))==expected_hash,path
    assert {path.name:sha(path.read_bytes()) for path in directory.iterdir()}=={name:sha(data) for name,data in blobs.items()}
    report={'schema':'track_a_ax2b_h4_limited_saved_execution_audit_v2','status':'SAVED_EXECUTION_CONSISTENCY_PASS',
        'science_run_status':parent['status'],'parent_reason':parent.get('reason'),'worker_reason':worker.get('reason') if worker else None,
        'worker_terminal_saved':worker is not None,'science_source_commit':SOURCE,'authorization_commit':AUTH,'manifest_commit':SEAL,
        'file_hashes':{OUTPUT+'/'+name:sha(data) for name,data in blobs.items()},'raw_files':len(blobs),'raw_bytes':sum(map(len,blobs.values())),
        'parent_wall_seconds':parent['wall_seconds'],'correctness_cells_completed':len(correctness),'MP_records_saved':len(mp),
        'explicit_event_records_saved':sum(name in rows for name in events),'correctness_summary':correctness,
        'observed_calls':worker['calls'] if worker else None,'call_counts_without_terminal':'raw completed stage records only; no guessed unfinished-cell counts',
        'coverage_comparison':comparison,'input_reference_summary':{key:reference[key] for key in ('reference_matvecs','reference','reference_kind','saved_state_norm_before','independent_occupation_all_columns_error')},
        'primitive_summary':primitives,'missing_records':missing,'partial_or_invalid_json_records':parse_errors,
        'source_hashes_checked':173,'preparation_freeze_checked':195,
        'worker_peak_RSS_lower_bound_from_completed_cells':max((record['worker_peak_rss_bytes_at_cell_end'] for record in correctness.values()),default=None),
        'scientific_acceptance_or_u_certification':False,'launch_count':1,'new_grant_consumed':True,'retry':False,'resume':False,
        'N':None,'G':None,'accuracy_eligibility':'UNDETERMINED','numerical_allowance_certified':False,
        'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION','mandatory_stop':True,'next_stage_authorized':False}
    print(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False))

if __name__=='__main__':main()
