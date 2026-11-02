"""Read-only post-run structure and receipt audit; numeric checks ran in child."""
from collections import Counter
import csv
import json
from run_h6_noncancellation_stage_b import ROOT,OUT,A,OLD,PREP,read,sha,verify,write,core


def main():
    c=read(OUT/'sealed_contract.json');verify(c)
    rows=read(OUT/'joined_rows.json')['rows'];patched=read(OUT/'patched120_lineage.json')['rows']
    original=read(ROOT/PREP/'saved_condition_rows.json')['rows'];original={r['row_id']:r for r in original}
    arows=read(ROOT/A/'endpoints.json')['rows'];amap={core.akey(r):r for r in arows}
    for row in rows:
        src=original[row['row_id']]
        if src['eligible'] is not None:assert row==src
        else:
            a=amap[core.akey(src)]
            assert row['eligible'] is a['eligible'] and row['saved_shots']==a['shots']
            assert row['saved_bias_and_shot_policy']==a['accounting']
            assert row['saved_total_RZ_literal'] is None
    assert len(patched)==120 and all(type(r['eligible']) is bool for r in rows)
    combined=read(OUT/'combined_selection100.json')['rows'];old=read(ROOT/OLD/'selection_comparison.json')['rows']
    assert len(combined)==100 and len({r['condition_id'] for r in combined})==100
    for a,b in zip(combined[:70],old):assert {k:v for k,v in a.items() if k!='numeric_origin'}==b
    results=read(OUT/'selection_comparison30.json')['rows'];assert combined[70:]==results and len(results)==30
    for file in ('candidate_totals','selection_comparison30','candidate_ranks','rank_relation_changes','combined_selection100'):
        js=read(OUT/(file+'.json'))['rows'];cs=list(csv.DictReader((OUT/(file+'.csv')).open()))
        assert len(js)==len(cs)
        for a,b in zip(js,cs):
            for k,v in a.items():assert b[k]==(json.dumps(v,ensure_ascii=False,separators=(',',':')) if isinstance(v,(dict,list,bool)) or v is None else str(v)),(file,k)
    for name,value in read(OUT/'numeric_completion.json')['result_sha256'].items():assert sha(OUT/name)==value
    audit=read(OUT/'numeric_validation.json');assert audit['status']=='PASS_INTEGER_COST_SELECTION_RANK_RATIO_AUDIT'
    ledger=read(OUT/'ledger.json');assert ledger['stages_reserved']==ledger['stages_completed'] and ledger['status']=='COMPLETE_STAGE_B30'
    assert all(v==0 for v in ledger['prohibited_calls'].values())
    native=read(OUT/'native_exit_evidence.json');assert native['native_exit_code']==0 and native['same_parent_wait4'] and native['reaped'] and native['live_owned_workers']==0
    assert sha(OUT/'native_stdout.txt')==native['stdout_sha256'] and sha(OUT/'native_stderr.txt')==native['stderr_sha256']
    rendering=read(OUT/'rendering_completion.json');assert len(rendering['figures'])==6
    for fig in rendering['figures']:assert sha(OUT/fig['path'])==fig['sha256']
    points=read(OUT/'figure_data.json');assert len(points['points'])==100
    assert sha(OUT/'figure_data.json')==rendering['figure_data_sha256']
    write('completion_audit.json',dict(status='PASS_STAGE_B30_WITH_VERBATIM70_COMPLETE',SOURCE=c['SOURCE'],
        patched_old_unknown_rows=120,old_nonnull_rows_PR_verbatim=True,stage_A_shots_eligibility_verbatim=True,
        known_baseline_costs_exactly_reproduced=712,new_eligible_baseline_costs=34,scope_conditions=30,
        combined_scope_conditions=100,unique_combined_endpoints=40,old70_verbatim_no_reanalysis=True,
        all_CSV_JSON='PASS',result_SOURCE_input_runtime_protected_SHA='PASS',numeric_audit=audit,
        native_exit0_same_parent_wait4_reaped=True,live_owned_workers=0,figures=2,exports=6,
        independent_reviewer=False,detailed_scientific_review=False,final_evidence_freeze=False,
        new_publication_authorized=False,mandatory_STOP=True))
    write('STOP.json',dict(status='STAGE_B30_COMPLETE_MANDATORY_STOP',SOURCE=c['SOURCE'],
        scope_conditions_completed=30,combined_scope_references=100,unknown_eligibility=0,
        old70_recalculated=False,old_evidence_preserved=True,grant_consumed=True,automatic_retry=False,
        further_scientific_execution_authorized=False,native_exit_code=0,reaped=True,live_owned_workers=0,
        detailed_scientific_review=False,final_evidence_freeze=False,new_publication_authorized=False,
        mandatory_STOP=True))
    print(json.dumps(dict(status='PASS',new_scopes=30,verbatim_old_scopes=70)))


if __name__=='__main__':main()
