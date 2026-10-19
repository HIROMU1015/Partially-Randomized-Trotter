"""Independently reread fresh faults and fixed pre-repair evidence."""
from pathlib import Path
from hashlib import sha256
import importlib.util,json,subprocess,tempfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('run',HERE/'run_acceptance.py');R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
A,I=R.A,R.I

def main():
    public=R.reference();fresh=[]
    for mode in ('S','P1'):
        case=HERE/'independent_writer_faults_final'/mode
        c=json.loads((case/'pilot_contract.json').read_text())
        expected=dict(public.template,runtime_root=c['runtime_root'],python=c['python'])
        assert A.I.strict_equal(c,expected)
        ref=I.Reference(HERE/'public_bytes',public.old.tree,ROOT/'docs/research/track_b_g6_w2c_b3_v3_offline_acceptance_20261019/fixed_reference',c)
        result=A.verify(case,ref);assert result['status']=='INCOMPLETE_EVIDENCE' and not result['issues'],result
        assert result['calls']['reserved']==1 and result['calls']['started_observed']==0 and result['calls']['cancelled_before_start']==1
        (case/'independent_verification.json').write_text(json.dumps(result,indent=2,default=lambda x:{'exact':A.V.canonical(x)})+'\n')
        observation=json.loads((case/'fault_result.json').read_text())
        fresh.append(dict(mode=mode,status=result['status'],calls=result['calls'],at_actual_writer_fault=observation['at_fault'],raw_sha256=result['raw_archive_sha256'],environment_delta=dict(runtime_root=c['runtime_root'],all_non_environment_candidate_fields_equal=True),scientific_calls=0))
    old=HERE/'public_bytes'/I.PREFIX/'before_repair/fresh_trace_reserve_write_failure'
    report=A.A.safe_values(A.V.plain((old/'observations.json').read_bytes()))['report']
    with tempfile.TemporaryDirectory(prefix='b3-pre-repair-independent-') as tmp:
        out=A.B.extract_archive(old/'artificial_raw.tar.gz',Path(tmp)/'raw');tasks,planned=A.G.plan(report['settings']['profile_n']);state=A.G.state_counts(out/'trial_states.bin',tasks)
        try:A.B.accounting_v3(state,report,planned)
        except A.V.Invalid as exc:issue=str(exc)
        else:raise AssertionError('PRE_REPAIR_DEFECT_NOT_DETECTED')
    assert issue=='GLOBAL_STATE_COUNTER_BINDING' and report['reserved']==0 and state['counts']['START_RESERVED']==1
    source='src/trottertracks/algorithm_codesign/g6_w2c_b3_pilot_candidate.py'
    before=subprocess.check_output(['git','show','76ca3c4089b201d55da0fede08e1f77733ac38a5:'+source],cwd=ROOT)
    assert before==(HERE/'public_bytes'/source).read_bytes()
    old_result=dict(public_SOURCE='76ca3c4089b201d55da0fede08e1f77733ac38a5',source_path=source,source_sha256=sha256(before).hexdigest(),case='fixed_before_repair_S',report_reserved=report['reserved'],state_counts=state['counts'],independent_rejection=issue,raw_sha256=A.V.file_digest(old/'artificial_raw.tar.gz'))
    runtime=[];py=public.template['python']
    for name,meta in [('executable',dict(path=py['executable'],sha256=py['executable_sha256'])),*py['libraries'].items()]:
        path=Path(meta['path']);actual=A.V.file_digest(path) if path.is_file() else None
        runtime.append(dict(name=name,path=str(path),recorded_sha256=meta['sha256'],current_server_sha256=actual,current_bytes_match=actual==meta['sha256']))
    packet=dict(schema='b3_v2_independent_reservation_comparison_v1',before=old_result,new_external_writer_faults=fresh,fresh_fault_count=2,producer_self_checker_called=False,current_runtime_bytes=runtime,historical_loaded_runtime_independently_proven=False,real_scientific_calls=0,mandatory_STOP=True)
    (HERE/'reservation_before_after_independent.json').write_text(json.dumps(packet,indent=2)+'\n');print(json.dumps(dict(before=issue,fresh_faults=[(r['mode'],r['status']) for r in fresh],current_runtime_all_match=all(r['current_bytes_match'] for r in runtime))))
if __name__=='__main__':main()
