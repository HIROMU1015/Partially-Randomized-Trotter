"""Verify saved metadata/bytes only; never run any scientific stage."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
read=lambda p:json.loads((HERE/p).read_text())
sha=lambda data:hashlib.sha256(data).hexdigest()


def main():
    for name,receipt in read('manifest.json')['files'].items():
        raw=(HERE/name).read_bytes()
        assert len(raw)==receipt['bytes'] and sha(raw)==receipt['sha256'],name
    index=read('stage_commit_index.json')['stages']
    counter_hashes=[]
    for stage in (1,2,3):
        terminal=read(f'stage{stage}/terminal.json');native=read(f'stage{stage}/native_exit.json')
        projection=read(f'stage{stage}/audit_projection.json')
        assert terminal['status']=='COMPLETE_MANDATORY_STOP' and terminal['first_stop'] is None
        assert native['native_exit_code']==0 and native['wait_reap_completed']
        assert terminal['SOURCE']==native['SOURCE']==projection['SOURCE']==index[stage-1]['SOURCE']
        assert terminal['final_source_input_environment_gate'] and not projection['automatic_retry']
        assert projection['grant_status']=='CONSUMED_COMPLETE' and projection['this_projection_is_not_a_grant']
        source=terminal['SOURCE']
        for path,expected in projection['source_closure'].items():
            assert sha(subprocess.check_output(['git','-C',str(ROOT),'show',f'{source}:{path}']))==expected,path
        counter_hashes.append(projection['source_closure']['src/trotterlib/native_df_gate_model.py'])
    assert len(set(counter_hashes))==1
    tests=read('stage1/tests_summary.json');assert tests==dict(tests=68,failures=0,errors=0,skipped=0)
    for record in read('stage1/stage_time_hex.json'):
        assert all(s['model_time']==s['native_time'] for s in record['stages'])
    acceptance=read('stage2/acceptance.json');assert len(acceptance)==8 and all(r['actual']==r['expected'] and r['exact_integer_match'] for r in acceptance)
    primitive=read('stage2/primitive_reuse_checks.json');assert sum(map(len,primitive.values()))==33
    assert all(r['full_primitive_receipt_match'] for rows in primitive.values() for r in rows)
    workers=read('stage2/worker_native_exits.json');assert len(workers)==2 and all(r['native_exit_code']==0 and r['wait_reap_completed'] for r in workers)
    summary=read('stage3/result_summary.json')
    assert (summary['expectation_comparison_rows'],summary['same_event_comparison_rows'],summary['pf_pr_comparison_rows'])==(84,84,360)
    assert summary['changed_rows']==dict(expectation_vs_saved=24,same_event_comparisons=24,pf_pr_comparisons=168)
    assert summary['direction_changes']==0 and summary['theory_compiled_direction_disagreements']==6
    assert len(read('stage3/old_new_differences.json'))==528
    same=read('stage3/same_event_comparisons.json')
    mixed=[r for r in same if r['model']=='NATIVE_MIXED']
    assert len(mixed)==28 and all(r['counter_native_count_match'] for r in mixed)
    assert len(read('stage3/preserved_shots.json'))==70
    for name,n in (('expectation_vs_saved',84),('same_event_comparisons',84),('pf_pr_comparisons',360)):
        rows=read('stage3/'+name+'.json');assert len(rows)==n
        for row in rows:
            assert row['SOURCE_accounting']==index[2]['SOURCE'] and row['compiled_values_reused']
    protected=read('preservation_audit.json')['protected_paths']
    assert not subprocess.check_output(['git','-C',str(ROOT),'diff','cb501906269ef31fe8733a3391db946b34467163','HEAD','--',*protected]).strip()
    before=subprocess.check_output(['git','-C',str(ROOT),'show','536a3671c0ec2939ff3886e40467db26a27ec65a:src/trotterlib/native_df_gate_model.py'])
    after=subprocess.check_output(['git','-C',str(ROOT),'show',index[0]['SOURCE']+':src/trotterlib/native_df_gate_model.py'])
    def rest(raw):
        tree=ast.parse(raw);tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name!='pf_schedule'];return ast.dump(tree,include_attributes=False)
    assert rest(before)==rest(after)
    print('PASS saved evidence: 3 SOURCE/STOP/native exits, 68 tests, 8 exact matches, 33 primitive checks, 84/84/360 rows, old evidence preserved; science calls=0')


if __name__=='__main__':main()
