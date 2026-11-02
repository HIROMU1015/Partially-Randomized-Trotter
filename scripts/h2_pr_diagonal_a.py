"""Independent diagonal receipt protocol and mock-only call orchestration.

No formula or primitive algorithm is implemented here. Production evaluation
has no authorized entry. Existing scientific functions are reused only by the
separate dormant adapter. QPY is read only after the independent stage finishes.
"""
from copy import deepcopy
import math
from functools import lru_cache
import h2_pr_identity_recovery as r
import h2_pr_stream_conformance as stream
import h2_pr_equivalence_contract as alternative
import audit_h2_pr_original_identity as original

BASE='65085ff7f818fd92af3424b1980f6bd9dbf1c742'
PRIOR='artifacts/resource_applicability/track_a_h2_pr_equivalence_contract/2026-10-23/'
TIMES=(('-0x1.5ca78b1289ec1p-1','-0x1.5ca78b1289ec1p-2'),
       ('0x1.14ba2befab5c7p-1','0x1.14ba2befab5c7p-2'),
       ('0x1.999999999999ap-3','0x1.999999999999ap-4'))
BLOCKS=('one_body','df_fragment')
BUDGET=dict(bootstrap=1,setup=1,prepare=1,diagonal_primitive_evaluation=6,archived_axis_parameter_compare=134,
 sampling=0,request_factory=0,rte_event=0,circuit_build=0,lowering=0,compile=0,transpile=0,
 simulation=0,u_N_G=0,allowance=0,ranking=0)
ALLOWED_INDEPENDENT_ROLES=('snapshot','snapshot_receipt','scientific_SOURCE','runtime','common_preparation_receipt')


def require(ok,message):r.require(ok,message)


def canonical_hex(value):
    require(type(value) is str,'BINARY64_HEX_STRING')
    try:x=float.fromhex(value)
    except ValueError:raise r.EvidenceError('INVALID_BINARY64_HEX')
    require(math.isfinite(x) and x.hex()==value,'CANONICAL_FINITE_BINARY64_HEX')
    return x


def time_pair(piece,argument):
    require((piece,argument) in TIMES,'EXACT_PIECE_AND_PRIMITIVE_TIME_MAPPING')
    require((canonical_hex(piece)/2.0).hex()==argument,'SINGLE_HALF_TIME_APPLICATION')


def draft(binding,producer_sources):
    require(set(binding)=={'snapshot','snapshot_receipt','hamiltonian','state','preparation','common_operations'},'BINDING_KEYS')
    for key,value in binding.items():alternative.strict.policy.sha(value,key)
    require(set(producer_sources)=={'prepare','one_body','df_fragment'},'EXISTING_PRODUCER_FUNCTIONS')
    for key,value in producer_sources.items():alternative.strict.policy.sha(value,key)
    return dict(schema='H2_DIAGONAL_A_REUSE_DRAFT_V1',status='UNAPPROVED_SCIENCE_EXECUTION_DISABLED',
      base_commit=BASE,condition='absolute_deterministic_diagonal_vs_common_preparation',
      binding=deepcopy(binding),producer_sources=deepcopy(producer_sources),
      times=[dict(piece_time_hex=p,primitive_argument_hex=t,division_count=1) for p,t in TIMES],
      deterministic_blocks=list(BLOCKS),budget=deepcopy(BUDGET),prefix=1,
      original_piece_identifiers_preserved=True,primitive_formula_reimplemented=False,
      scientific_execution_authorized=False,alternative_adopted=False,COMPARE_ready=False,
      new_grant=None,new_ledger=None,new_marker=None,new_run_root=None,retry=False,mandatory_STOP=True)


def validate_draft(value):
    require(type(value) is dict and 'binding' in value and 'producer_sources' in value,'DRAFT_REQUIRED')
    require(alternative.strict.policy.strict_equal(value,draft(value['binding'],value['producer_sources'])),
            'FIXED_DRAFT_SCOPE_OR_BUDGET_CHANGED')


def validate_independent_receipt(packet,plan):
    validate_draft(plan)
    require(set(packet)=={'schema','binding','producer_sources','input_dependencies','origin','rows','input_stage_closed'},'INDEPENDENT_RECEIPT_SCHEMA')
    require(packet['schema']=='H2_INDEPENDENT_DIAGONAL_PRIMITIVES_V1' and
      packet['binding']==plan['binding'] and packet['producer_sources']==plan['producer_sources'],'SOURCE_INPUT_PREPARATION_BINDING')
    require(packet['origin']=='COMMON_HAMILTONIAN_PREPARATION_EXISTING_PRIMITIVES' and
            packet['input_stage_closed'] is True,'INDEPENDENCE_NOT_JUST_DUPLICATE_VALUES')
    require(type(packet['input_dependencies']) is list and bool(packet['input_dependencies']),'INDEPENDENT_INPUT_MANIFEST')
    for dep in packet['input_dependencies']:
        require(set(dep)=={'role','sha256'} and dep['role'] in ALLOWED_INDEPENDENT_ROLES,'ARCHIVED_QPY_EVENT_OR_COST_MUST_NOT_FEED_PRODUCER')
        alternative.strict.policy.sha(dep['sha256'],'INPUT_DEPENDENCY_SHA')
    for key in ('snapshot','snapshot_receipt'):
        require(any(x['role']==key and x['sha256']==plan['binding'][key] for x in packet['input_dependencies']), 'BOUND_SNAPSHOT_DEPENDENCY')
    require(type(packet['rows']) is list and len(packet['rows'])==6,'SIX_INDEPENDENT_PRIMITIVES')
    require([(x['piece_time_hex'],x['block']) for x in packet['rows']]==[(p,b) for p,t in TIMES for b in BLOCKS],
            'EXACT_TIME_BLOCK_ORDER')
    for item in packet['rows']:
        require(set(item)=={'piece_time_hex','primitive_argument_hex','block','global_phase_hex','rz','rzz'},'PRIMITIVE_ROW_KEYS')
        time_pair(item['piece_time_hex'],item['primitive_argument_hex']);canonical_hex(item['global_phase_hex'])
        expected_rz=[[k] for k in range(4)];expected_rzz=[] if item['block']=='one_body' else [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]
        for key,expected in (('rz',expected_rz),('rzz',expected_rzz)):
            require(type(item[key]) is list and [x['wires'] for x in item[key]]==expected,'ORDERED_PRIMITIVE_SUPPORT')
            for op in item[key]:
                require(set(op)=={'wires','angle_hex'},'PRIMITIVE_OPERATION_KEYS')
                require(canonical_hex(op['angle_hex'])!=0.0,'FIXED_NONZERO_H2_PRIMITIVE_PATTERN')
    return {(x['piece_time_hex'],x['block']):x for x in packet['rows']}


def serialize_primitive(value,piece,argument,block):
    """Record outputs of original functions, never derive the primitive formula."""
    time_pair(piece,argument)
    def hx(x):
        result=float(x).hex();canonical_hex(result);return result
    return dict(piece_time_hex=piece,primitive_argument_hex=argument,block=block,
      global_phase_hex=hx(value.global_phase),
      rz=[dict(wires=[int(k)],angle_hex=hx(a)) for k,a in value.rz],
      rzz=[dict(wires=[int(i),int(j)],angle_hex=hx(a)) for i,j,a in value.rzz])


class Ledger:
    def __init__(self):self.used=dict.fromkeys(BUDGET,0)
    def call(self,name,fn,*args):
        require(name in BUDGET and self.used[name]<BUDGET[name],'EXACT_CALL_BUDGET:'+name)
        self.used[name]+=1
        return fn(*args)


def synthetic_workflow(plan,backend,axis_jobs):
    """Run only injected artificial callbacks; actual H2 backend always denied."""
    validate_draft(plan)
    require(getattr(backend,'domain',None)=='ARTIFICIAL','REAL_H2_PRIMITIVE_EXECUTION_UNAUTHORIZED')
    require(type(axis_jobs) is list and len(axis_jobs)==134,'EXACT_AXIS_JOB_COUNT')
    ledger=Ledger();ledger.call('bootstrap',backend.bootstrap)
    model=ledger.call('setup',backend.setup)
    prepared=ledger.call('prepare',backend.prepare,model)
    rows=[]
    for piece,argument in TIMES:
        for block in BLOCKS:
            value=ledger.call('diagonal_primitive_evaluation',backend.primitive,prepared,block,argument)
            rows.append(serialize_primitive(value,piece,argument,block))
    packet=dict(schema='H2_INDEPENDENT_DIAGONAL_PRIMITIVES_V1',binding=deepcopy(plan['binding']),
      producer_sources=deepcopy(plan['producer_sources']),input_dependencies=backend.input_dependencies(),
      origin='COMMON_HAMILTONIAN_PREPARATION_EXISTING_PRIMITIVES',rows=rows,input_stage_closed=True)
    validate_independent_receipt(packet,plan)
    # Seal immutable serialized output before comparator can open a QPY.
    raw=__import__('json').dumps(packet,sort_keys=True,separators=(',',':')).encode()
    receipt_sha=__import__('hashlib').sha256(raw).hexdigest()
    for job in axis_jobs:
        ledger.call('archived_axis_parameter_compare',backend.compare,job,raw,receipt_sha)
    require(ledger.used==BUDGET,'EXACT_WORKFLOW_COMPLETE_BUDGET')
    return dict(domain='ARTIFICIAL',packet_sha256=receipt_sha,used=ledger.used,scientific_GO=False)


class IndependentCursor(stream.Cursor):
    def __init__(self,circuit,lookup,piece,groups):
        super().__init__(circuit);self.lookup=lookup;self.piece=piece;self.groups=groups
    def row(self,operations):
        require(operations in self.groups[:2],'EXACT_DETERMINISTIC_BLOCK')
        return self.lookup[(self.piece,BLOCKS[self.groups.index(operations)])]
    def diagonal(self,operations,fragment):
        saved=super().diagonal(operations,fragment);row=self.row(operations)
        for entries,key in zip(saved,('rzz','rz')):
            actual=[dict(wires=w,angle_hex=a.hex()) for w,a in entries]
            require(actual==row[key],'ABSOLUTE_DIAGONAL_ANGLE_SUPPORT_SIGN_ORDER_MISMATCH')
        return saved
    def directional(self,operations,saved):
        phase=super().directional(operations,saved)
        require(phase.hex()==self.row(operations)['global_phase_hex'],'INDEPENDENT_BLOCK_PHASE_MISMATCH')
        return phase


@lru_cache(maxsize=67)
def expected_piece_times(task):
    saved=r.loads(r.git_bytes(r.BASE,r.RECEIVED+'rte_schedule_metadata.json'))['tasks']
    row=next((x for x in saved if x['task_id']==task),None)
    require(row is not None,'ORIGINAL_SCHEDULE_TASK')
    return tuple(float(x['saved']['signed_step_time']).hex() for x in row['pieces'])


def compare_saved_axis(packet,plan,circuit,events,common,axis,scalar,piece_times,identity):
    lookup=validate_independent_receipt(packet,plan)
    require(set(identity)=={'task','candidate','sample','axis','cohort','preparation'},'AXIS_IDENTITY_SCHEMA')
    anchor=next((x for x in alternative.anchors() if x['task']['id']==identity['task']),None)
    require(anchor is not None,'ORIGINAL_TASK_ONLY')
    task=anchor['task']
    require(identity['cohort']=='EXPLORATION' and identity['candidate']==task['candidate']['id'] and
        type(identity['sample']) is int and identity['sample']==task['sample'] and identity['axis']==axis and
        axis in ('cosine','sine') and identity['preparation']==plan['binding']['preparation'],'ORIGINAL_AXIS_COHORT_PREPARATION')
    require(type(piece_times) is list and tuple(piece_times)==expected_piece_times(identity['task']),
            'ORIGINAL_SIGNED_SCHEDULE_IDENTITY')
    require(common['preparation_sha256']==plan['binding']['preparation'] and
            common['operations_sha256']==plan['binding']['common_operations'],'COMMON_PREPARATION_BINDING')
    # Existing full stream checks are unchanged; this pass adds independent values.
    status=stream.recognize(circuit,events,common,axis,scalar)
    require(status['global_phase_relation_exact'] is True,'SERIALIZED_GLOBAL_PHASE')
    require(len(piece_times)==len(events),'EXACT_PIECE_COUNT')
    groups=[x['operations'] for x in common['operation_stream']]
    cursor=IndependentCursor(circuit,lookup,piece_times[0],groups)
    known={r.digest(x) for x in groups}
    for piece,event_stream in zip(piece_times,events):
        require(piece in dict(TIMES),'TARGET_PIECE_TIME_ONLY');cursor.piece=piece
        one=cursor.diagonal(groups[0],False);fragment=cursor.diagonal(groups[1],True)
        cursor.tail(event_stream,known)
        cursor.directional(groups[1],fragment);cursor.directional(groups[0],one)
    cursor.take('PhaseGate',[4],[-scalar*.8])
    if axis=='sine':cursor.take('SdgGate',[4])
    cursor.take('HGate',[4]);require(cursor.pos==len(cursor.items)-1,'INDEPENDENT_ALL_PIECES_CONSUMED')
    return dict(status='VERIFIED_FOR_DECLARED_A_PARAMETER_SCOPE',independent_receipt_sha256=r.digest(packet),
      task=identity['task'],axis=axis,scientific_GO=False,B_resolved=False,alternative_adopted=False)


def preflight(plan,expected_binding,expected_sources,case='A2'):
    validate_draft(plan)
    require(plan['binding']==expected_binding and plan['producer_sources']==expected_sources,'FIXED_REAL_BINDING')
    require(case in ('A1','A2','A3'),'EXPLICIT_REUSE_CASE')
    return dict(status='BLOCKED_PENDING_SEPARATE_SCIENTIFIC_AUTHORIZATION',case=case,
      source_and_metadata_preparation=True,independent_real_receipts_available=False,
      time_mapping_single_halving=True,planned_calls=deepcopy(BUDGET),actual_scientific_calls=0,
      science_authorized=False,execution_ready=False,alternative_adopted=False,COMPARE_ready=False,
      blockers=['No independent same-target primitive receipts','Scientific authorization and runtime/input activation absent'],mandatory_STOP=True)


def deny_execute():raise r.EvidenceError('REAL_H2_DIAGONAL_EXECUTION_NOT_APPROVED')


if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser();parser.add_argument('--draft');parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.execute:deny_execute()
    require(args.draft is not None,'METADATA_DRAFT_REQUIRED')
    value=r.loads(Path(args.draft).read_bytes());print(json.dumps(preflight(value,value['binding'],value['producer_sources'])))
