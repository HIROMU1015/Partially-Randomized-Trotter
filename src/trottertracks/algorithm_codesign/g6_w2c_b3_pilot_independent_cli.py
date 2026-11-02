"""Fixed artificial pilot case -> fresh independent report directory."""
import argparse,csv,json
from pathlib import Path
from . import g6_w2c_b3_pilot_independent_adapter as A
from . import g6_w2c_b3_pilot_independent_identity as I
from . import g6_w2c_b3_offline_result as V
from .g6_w2c_b3_offline_cli import write_json


def acceptance(v):
    checks=[];raw='history' in v;identity=v.get('identity',{});adm=v.get('admission',{})
    def add(cid,expected,observed,passed=False,reason=None,evidence='RAW_RECOMPUTATION'):
        checks.append(dict(check_id=cid,target_SOURCE='fixed pilot preparation / independent consumer',
            public_commit=I.PUBLIC,content_commit=I.CONTENT,measured_SOURCE_commit=I.MEASURED,
            input_identity=dict(case=v.get('case'),pilot_digest=identity.get('pilot_digest'),inner_digest=identity.get('inner_digest')),
            expected=expected,observed=observed,judgment='PASS' if passed else 'PENDING_EVIDENCE',
            evidence=evidence,evidence_files=['pilot_contract.json','inner_contract.json','observations.json','FAKE_ONLY_claims.sqlite3']+(['artificial_raw.tar.gz'] if raw else []),
            unresolved=reason,formal_adoption_required=False))
    add('INPUT_INTEGRITY','No detected input inconsistency',v['status'],v['status'] in ('VERIFIED_COMPLETE','TECHNICAL_FAILURE') or bool(identity))
    if v['status'] in ('INTEGRITY_MISMATCH','UNSUPPORTED_SCHEMA'):checks[-1]['judgment']='FAIL'
    add('PILOT_IDENTITY','Published closed candidate / fixed SOURCE/input/schema pins',identity,bool(identity))
    add('FAKE_SNAPSHOT','Read-only single store/scope/claim/digest/spent binding',adm,bool(adm))
    add('ATOMIC_HISTORY','Atomic admission/race/transition history','Snapshot and saved observations',False,
        'Snapshot cannot prove atomic transition order; independent fresh artificial replay is separate evidence.',evidence='SAVED_SNAPSHOT_AND_CLAIM')
    add('ORDERED_TRACE','Canonical chain, seq/run/trial, phase/STOP/drain, state witnesses',v.get('trace'),raw,
        None if raw else 'No complete operation trace retained.')
    add('CALL_BUDGET','Independent counts and global cap',v.get('calls'),raw)
    add('BYTE_BUDGET','Trace self bytes + nontrace credits; ACK <= physical <= reserved',v.get('output_budget'),raw)
    add('F01_PREFIX','Engine commit is distinct from physical row/prefix',len(v.get('pairs',[])),raw)
    add('SUCCESS_RECEIPT','Summary/receipt/trace-prefix joined',v.get('control'),v['status']=='VERIFIED_COMPLETE',
        None if v['status']=='VERIFIED_COMPLETE' else 'Failed or admission-only case has no successful completion proof.')
    cost=v.get('cost',{})
    add('COST_BOUNDARIES','Host parent+all workers, outer wall, alternative coarse/fine, null unknown',cost.get('status'),
        cost.get('status')=='VERIFIED_PARALLEL_COST_BOUNDARIES',None if raw else 'No process totals retained.')
    if cost.get('status')=='COST_INTEGRITY_MISMATCH':checks[-1]['judgment']='FAIL'
    add('HISTORICAL_PENDING','Old 47 cases unchanged',v.get('historical_pending_reclassified') is False,
        v.get('historical_pending_reclassified') is False)
    add('TWO_CONDITION_BRIDGE','S/P1 global scientific bundle implementation','NOT_IMPLEMENTED',False,
        'Scientific bridge and adoption are separate future work.')
    checks[-1]['formal_adoption_required']=True
    add('PILOT_SCOPE_ADOPTION','User adoption of N/modes/seed/measurement contract','UNADOPTED_EXECUTION_CLOSED',False,'Static arithmetic is checked; scientific scope remains a user/GPT decision.')
    checks[-1]['formal_adoption_required']=True
    return dict(schema='b3_limited_pilot_independent_acceptance_v1',checks=checks,scientific_GO=False,
                historical_pending_reclassified=False,mandatory_STOP=True)


def run(case,tree,legacy,output):
    case=Path(case).resolve();tree=Path(tree).resolve();legacy=Path(legacy).resolve();output=Path(output).resolve()
    for root in (case,tree,legacy):V.require(not output.is_relative_to(root) and not root.is_relative_to(output),'OUTPUT_OUTSIDE_INPUT')
    V.require(not output.exists(),'NO_OVERWRITE')
    v=A.verify(case,I.Reference(tree,legacy));h=acceptance(v)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'verification_report.json',v);write_json(output/'acceptance_report.json',h)
    (output/'verification_report.md').write_text('Status: **'+v['status']+'**.\n\nArtificial preparation only. Snapshot is not atomic history; old pending is unchanged.\n')
    fields=['task_key','phase','pass_name','arm','layer','outer_cpu_wall_ns','setup_cpu_wall_ns','online_cpu_wall_ns','failed_prefix']
    with (output/'comparison_table.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore',lineterminator='\n');w.writeheader()
        c=v.get('cost',{})
        if v['status']=='VERIFIED_COMPLETE' and c.get('comparison_eligible') and c.get('status')=='VERIFIED_PARALLEL_COST_BOUNDARIES':w.writerows(p for p in c.get('pairs',[]) if not p['failed_prefix'])
    return v,h


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('case','fixed-tree','legacy-reference','new-output'):p.add_argument('--'+n,required=True,type=Path)
    a=p.parse_args(argv);v,_=run(a.case,a.fixed_tree,a.legacy_reference,a.new_output)
    print(json.dumps(dict(status=v['status'],mandatory_STOP=True)))
    return 0 if v['status'] in ('VERIFIED_COMPLETE','TECHNICAL_FAILURE') else 2

if __name__=='__main__':raise SystemExit(main())
