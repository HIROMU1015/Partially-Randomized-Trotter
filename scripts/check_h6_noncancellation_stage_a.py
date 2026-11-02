"""Post-run structural and saved-check audit; no second numeric analysis pass."""
from collections import Counter
import csv
import json
import sys

from run_h6_noncancellation_stage_a import OUT,ROOT,core,read,sha,verify,write


def main():
    c=read(OUT/'sealed_contract.json')
    verify(c)
    rows=read(OUT/'endpoints.json')['rows']
    parts=read(OUT/'components.json')['rows']
    checks=read(OUT/'numeric_validation.json')
    summary=read(OUT/'summary.json')
    ledger=read(OUT/'ledger.json')
    native=read(OUT/'native_exit_evidence.json')
    completion=read(OUT/'numeric_completion.json')
    assert len(rows)==120 and len(parts)==12
    assert {p['candidate_id'] for p in parts}==set(c['candidate_ids'])
    assert len({(r['candidate_id'],r['epsilon'],r['allowance_factor']) for r in rows})==120
    assert all(r['eligible'] is not None and r['previous_eligibility'] is None for r in rows)
    assert Counter(r['status'] for r in rows)==Counter(summary['eligibility_counts'])
    assert sum(p['component_origin']=='REUSED_PUBLIC_LITERAL_COMPONENTS' for p in parts)==1
    assert next(p['candidate_id'] for p in parts if p['component_origin']=='REUSED_PUBLIC_LITERAL_COMPONENTS')==core.REUSED
    assert ledger['reserved']==ledger['completed']==core.CAPS
    assert all(v==0 for v in ledger['prohibited_calls'].values())
    assert ledger['final_input_SOURCE_runtime_protected_evidence_gate']=='PASS'
    assert checks['status']=='PASS_AUTHOR_NUMERIC_CHECKS' and len(checks['checks'])==132
    assert all((x.get('candidate_check') or x['endpoint_check'])['status']=='PASS' for x in checks['checks'])
    assert native['native_exit_code']==0 and native['same_parent_wait4'] and native['reaped']
    assert native['live_owned_workers']==0
    assert sha(OUT/'native_stdout.txt')==native['stdout_sha256']
    assert sha(OUT/'native_stderr.txt')==native['stderr_sha256']
    for name,value in completion['result_sha256'].items():assert sha(OUT/name)==value,name
    csv_rows=list(csv.DictReader((OUT/'endpoints.csv').open()))
    assert len(csv_rows)==120
    for a,b in zip(rows,csv_rows):
        for key in ('candidate_id','prefix','q','epsilon','allowance_factor','status','raw_input_sha256'):
            assert b[key]==str(a[key]),key
        assert json.loads(b['eligible'])==a['eligible']
        for key,value in [('N_cosine',None if a['shots'] is None else a['shots']['cosine']),
                          ('N_sine',None if a['shots'] is None else a['shots']['sine']),('N_total',a['N_total'])]:
            assert b[key]==('null' if value is None else str(value)),key
    write('completion_audit.json',dict(status='PASS_STAGE_A_COMPLETE_NATIVE_REAP_PROTECTED_EVIDENCE',
        SOURCE=c['SOURCE'],candidate_count=12,endpoint_count=120,eligibility_counts=summary['eligibility_counts'],
        unresolved_endpoints=summary['eligibility_counts']['UNRESOLVED'],
        closure_candidates=12,primary_and_validation_budget=ledger['completed'],
        numeric_check_records=132,numeric_checks_were_inside_one_shot_child=True,
        second_primary_numeric_pass=False,CSV_JSON='PASS',result_SHA='PASS',
        SOURCE_input_runtime_protected_old_evidence='PASS',native_exit0_same_parent_reap=True,
        live_owned_workers=0,stage_B_executed=False,new_compile_signal_sampling_fit=0,
        checks_by='Primary author agent; not a separate independent reviewer',
        new_publication_authorized=False,detailed_scientific_review=False,final_evidence_freeze=False,
        mandatory_STOP=True))
    write('STOP.json',dict(status='STAGE_A_COMPLETE_MANDATORY_STOP',SOURCE=c['SOURCE'],
        eligibility_counts=summary['eligibility_counts'],unresolved_endpoints=summary['eligibility_counts']['UNRESOLVED'],
        native_exit_code=0,reaped=True,live_scientific_workers=0,grant_consumed=True,
        further_stage_A_execution_authorized=False,stage_B_authorized=False,automatic_retry=False,
        old_evidence_preserved=True,new_publication_authorized=False,detailed_scientific_review=False,
        final_evidence_freeze=False,mandatory_STOP=True))
    print(json.dumps(dict(status='PASS',endpoints=120,**summary['eligibility_counts'])))


if __name__=='__main__':main()
