"""Read saved raw in a producer-free process; fixed checker remains unchanged."""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json,sys,tempfile
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign import g6_w2c_b3_storage_verify_v2 as N
from trottertracks.algorithm_codesign import g6_w2c_b3_offline_result as V
from trottertracks.algorithm_codesign import g6_w2c_b3_parallel_verifier_core as G
from trottertracks.algorithm_codesign import g6_w2c_b3_v3_offline_adapter as B
from trottertracks.algorithm_codesign import g6_w2c_b3_parallel_offline_adapter as W
from trottertracks.algorithm_codesign import g6_w2c_b3_pilot_independent_adapter as E
from trottertracks.algorithm_codesign import g6_w2c_b3_pilot_independent_identity as I


def main():
    packet=ROOT/'docs/research/track_b_g6_w2c_b3_limited_pilot_independent_acceptance_20261019'
    ref=I.Reference(packet/'fixed_reference/tree',ROOT/'docs/research/track_b_g6_w2c_b3_v3_offline_acceptance_20261019/fixed_reference')
    before=HERE/'before_repair/fresh_trace_reserve_write_failure'
    c=V.plain((before/'pilot_contract.json').read_text());normalized=deepcopy(c)
    for k in('runtime_root','python'):normalized[k]=ref.template[k]
    assert I.I.strict_equal(normalized,ref.template)
    # Same explicit runtime-only rebind used in the fixed external replay tests.
    old_template=ref.template;ref.template=c;old_result=E.verify(before,ref);ref.template=old_template
    assert old_result['status']=='INTEGRITY_MISMATCH'and 'GLOBAL_STATE_COUNTER_BINDING'in old_result['issues']
    rows=[]
    for case in sorted((HERE/'fault_cases_final').iterdir()):
        if not case.is_dir():continue
        c=V.plain((case/'pilot_contract.json').read_text());inner=V.plain((case/'inner_contract.json').read_text())
        obs=W.safe_values(V.plain((case/'observations.json').read_text()));r=obs['report']
        store=I.admission_snapshot(case/'FAKE_ONLY_claims.sqlite3',inner)
        common=None;verified=obs['verification']
        if r is not None:
            with tempfile.TemporaryDirectory(prefix='b3-storage-saved-read-')as d:
                out=B.extract_archive(case/'artificial_raw.tar.gz',Path(d)/'raw')
                verified=N.verify(out,c,inner,r);assert verified['status']==obs['verification']['status']
                tasks,n=G.plan(16);snapshot=G.state_counts(out/'trial_states.bin',tasks)
                if r['evidence_v2']['statefile_complete']:
                    common=B.accounting_v3(snapshot,r,n)
                    assert common['reserved']==r['reserved']
        unsupported=E.verify(case,ref)
        assert unsupported['status']!='VERIFIED_COMPLETE'
        rows.append(dict(case=case.name,status=verified['status'],fake_store=store,
            common_fixed_accounting=common,statefile_available_for_exact_comparison=common is not None,
            reserved=r['reserved']if r else None,started=r['started_observed']if r else None,
            complete=r['complete']if r else False,fixed_full_adapter_status=unsupported['status'],
            fixed_full_adapter_issues=unsupported['issues'],
            full_adapter_applicability='UNSUPPORTED_NEW_V2_IDENTITY_NOT_AN_ACCOUNTING_PASS',
            local_Codex_final_acceptance_pending=True))
    assert len(rows)==108
    forbidden=[n for n in sys.modules if n.rsplit('.',1)[-1]in(
        'g6_w2c_b3_pilot_candidate','g6_w2c_b3_pilot_storage_v2','g6_w2c_b3_parallel_scheduler',
        'g6_w2c_b3_parallel_admission','g7_generator','g7_provider','return_aggregation',
        'g6_f3_contract','g6_f3_precision','g6_f3_generator','g9_native')]
    assert not forbidden
    value=dict(before_fixed_SOURCE=dict(status=old_result['status'],issues=old_result['issues']),
        after_cases=rows,classifications=dict(Counter(r['status']for r in rows)),
        fixed_checker_commit='49a0dcc9c8ec6dfc96eba5bb862cfdcb3d0fc866',
        external_checker_SOURCE_unchanged=True,forbidden_producer_science_imports=forbidden,
        author_run_supplemental_check=True,local_Codex_final_independent_acceptance='PENDING',successful=True)
    (HERE/'saved_matrix_verification.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(cases=len(rows),classifications=value['classifications'],before=value['before_fixed_SOURCE'],successful=True)))


if __name__=='__main__':main()
