"""Machine-readable v3 P01--P11; saved records do not prove missing history."""
from . import g6_w2c_b3_offline_result as V
from . import g6_w2c_b3_v3_offline_identity as I


def matrix(v,cost):
    rows=[];usable=v['status'] in ('VERIFIED_COMPLETE','TECHNICAL_FAILURE') and 'calls' in v
    def add(cid,expected,observed,proven=True,evidence=V.RAW,reason=None):
        rows.append(dict(check_id=cid,target_SOURCE='fixed integrated v3 / independent consumer',
                    fixed_commit=I.PUBLIC,measured_source_commit=I.MEASURED,
                    schema_version='b3_parallel_wire_and_storage_v3',expected=expected,observed=observed,
                    judgment='PASS' if usable and proven else 'PENDING_EVIDENCE',evidence=evidence,unresolved=reason))
    control=v.get('control',{});calls=v.get('calls');binding=v.get('binding_evidence',{})
    add('P01','Fixed task identity/order/offset; exact pair chain and file bytes',v['status'])
    add('P02','Logical committed state vs physical F01 prefix/in-flight',calls)
    add('P03','Global planned/reserved/started/completed/failed/cancelled/unknown',calls)
    add('P04','Reserved/ACK/physical/unknown credit; no credit refund',v.get('output_budget'),v.get('output_budget') is not None)
    add('P05','Conservative worker/parent loss and unknown accounting',calls,
        reason='Saved worker exit/join declarations do not independently prove historical OS process state.')
    add('P06','Phase barrier and persisted worker/failure order',control,not control.get('evidence_gaps') and control.get('started_workers')==control.get('ended_workers'),
        reason='Log failure may leave only an ordered subsequence; absent notices are not reconstructed.')
    add('P07','Atomic fresh scope reservation, approval, spent state, SOURCE/runtime binding',binding,False,V.CLAIM,
        'Contract digests can be recomputed when published; private approval/SQLite bytes and historical Python bytes are unavailable. No real atomic claim proof.')
    add('P08','No new ENTER permission after parent STOP','No per-trial operation sequence in published control',False,V.UNKNOWN,
        'Latest statefile is not transition history. Separate normalized artificial operation tests detect violations only in that supplied history.')
    add('P09','Host parent+worker CPU; exclusive coarse/fine separately; unknown nonzero semantics',cost['status'],
        cost['status']=='VERIFIED_PARALLEL_COST_BOUNDARIES',reason='Parent crash lacks outer CPU; isolated observer overhead is unknown.')
    add('P10','Summary/receipt digest and all-worker join declaration',control.get('receipt_present'),
        control.get('receipt_present') is True,V.RAW,'Absent success receipt is expected on failure; physical joins remain saved declarations.')
    add('P11','Artificial retry0/resume=false flags; no reuse authority',v.get('fake_claim_status'),False,V.CLAIM,
        'Flags and STOP claim checked; private v3 store nonreuse/atomic history not independently replayed. Existing artificial ledger regressions are separate evidence.')
    return dict(schema='b3_v3_offline_acceptance_v1',run_id=v.get('run_id'),rows=rows,
                pending=[r['check_id'] for r in rows if r['judgment']=='PENDING_EVIDENCE'],
                scientific_GO=False,real_authorization_verified=False,mandatory_STOP=True)
