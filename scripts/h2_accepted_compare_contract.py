"""New one-shot COMPARE control plane. Never import scientific packages here."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import h2_common_production_contract as old
import h2_diagnostics_binding as seal
import h2_accepted_diagnostics as accepted

ROOT=Path(__file__).resolve().parents[1]
BRANCH='COMPARE_FROM_ACCEPTED_DIAGNOSTICS'
SERIES='H2_COMPARE_FROM_ACCEPTED_DIAGNOSTICS_V1'
SCHEMA='h2_compare_from_accepted_diagnostics_v1'
KINDS=old.KINDS+('bundle_load',)
policy=old.policy
Stop,require,read,save,sha,reuse_costs=old.Stop,old.require,old.read,old.save,old.sha,old.reuse_costs
RUNNER_FILES=('scripts/h2_accepted_compare_contract.py','scripts/h2_accepted_compare_backend.py',
 'scripts/h2_accepted_diagnostics.py','scripts/launch_h2_accepted_compare.py',
 'scripts/prepare_h2_accepted_compare.py',
 'scripts/h2_common_production_contract.py','scripts/h2_common_production_backend.py',
 'scripts/h2_common_production_diagnostics.py','scripts/run_h2_common_production.py',
 'scripts/h2_diagnostics_binding.py','src/trotterlib/h2_common_native_policy.py')


def counts():
    result=old.counts('COMPARE_REUSE_PR')
    result.update(diagnose=0,path_signal=0,reference_rebind=0,bundle_load=1)
    return result


def runtime_identity(runtime):
    # Cache/output location is operational, not a compiler/library byte identity.
    return policy.digest({k:runtime[k] for k in
        ('python_sha256','python_target','numpy','qiskit','inventory_sha256','sys_path')})


def saved_anchors(candidate):
    base='artifacts/resource_applicability/track_a_pf_order_reversal_h2_metadata_receipt/2026-10-23/received/'
    tasks=json.loads(old.prior_blob(base+'task_acceptance_67.json'))['tasks']
    return [r for r in tasks if r['task']['candidate']['id']==candidate.id]


def expected_identities(cfg,plan,candidate,proof):
    provider=Path(cfg['provider_root'])/'src';acquisition=Path(cfg['acquisition_root'])/'scripts'
    prep=plan['preparations'][1];row=plan['records'][candidate.id]
    receipt=read(cfg['inputs']['snapshot_receipt'])
    native=provider/'trottertracks/resource_applicability/ax2b_native_df_v5.py'
    return dict(hamiltonian=cfg['binding']['hamiltonian'],state=cfg['binding']['state'],
      basis=receipt['arrays']['sector_basis_indices']['sha256'],rotation=prep['operations_sha256'],
      preparation=prep['preparation_sha256'],schedule=policy.digest(row['schedule']),
      deterministic_lowering=sha(native),event_law=policy.digest(dict(source=sha(provider/'trotterlib/rte.py'),K=4,policy='ABS_TIME_PER_OUTER',value=32)),
      events=sha(proof['events']),wrapper=sha(native),
      scalar_phase=policy.digest(dict(scalar=row['scalar'],T=.8,scalar_once=True)),
      compiler_source=policy.digest(dict(guard=sha(acquisition/'h2_m5_compiler_phase_guard.py'),cost=sha(provider/'trotterlib/rte_compiled_cost.py'))),
      compiler_settings=policy.digest(policy.COMPILER),runtime_closure=runtime_identity(cfg['runtime']),
      correctness=sha(proof['correctness']),cohort=sha(proof['cohort']))


def validate_witness(path,candidate,cfg,plan):
    """Proof must join historical raw receipts and full events, not just labels."""
    wrapper=read(path)
    require(set(wrapper)=={'schema','legacy','attestations','original_results'} and
            wrapper['schema']=='H2_PR_COMPILED_IDENTITY_BYTE_JOIN_V1','PR_WITNESS_SCHEMA')
    witness=wrapper['legacy'];proof=witness['proof'];sealed={r['path']:r['sha256'] for r in cfg['files']}
    require(set(proof)=={'events','correctness','cohort'} and all(p in sealed for p in proof.values()),'SEALED_PR_PROOFS')
    current=expected_identities(cfg,plan,candidate,proof)
    costs=reuse_costs(witness,current,candidate)
    anchors=saved_anchors(candidate)
    require(len(wrapper['original_results'])==len(anchors),'ORIGINAL_TASK_RECEIPTS_REQUIRED')
    expected_receipts=[]
    for original,row in zip(wrapper['original_results'],anchors):
        require(set(original)=={'task','path','sha256'} and original['task']==row['task']['id'] and
                original['path'] in sealed and original['sha256']==sealed[original['path']]==
                row['acceptance']['saved']['result_sha256'],'HISTORICAL_ORIGINAL_RECEIPT_HASH')
        raw=read(original['path'])
        require(policy.strict_equal(raw['task'],row['task']) and
                raw['event_digest']==row['result']['saved']['event_digest'] and
                policy.strict_equal(raw['axis_costs'],row['result']['saved']['axis_costs']),
                'ORIGINAL_TASK_AXIS_JOIN')
        expected_receipts.append(original['sha256'])
    require(set(wrapper['attestations'])==set(old.REUSE_FIELDS),'ALL_IDENTITY_FIELD_PROOFS')
    for field,path in wrapper['attestations'].items():
        require(path in sealed,'SEALED_FIELD_ATTESTATION')
        a=read(path)
        require(a['kind']=='HISTORICAL_TO_COMMON_BYTE_IDENTITY_V1' and a['field']==field and
                a['saved_sha256']==witness['saved'][field] and a['current_sha256']==current[field] and
                a['original_receipt_sha256']==expected_receipts,'FIELD_PROOF_BINDING')
        require(bool(a['evidence_files']),'FIELD_PROOF_EVIDENCE_REQUIRED')
        for item in a['evidence_files']:
            require(item['path'] in sealed and sealed[item['path']]==item['sha256'],'SEALED_FIELD_EVIDENCE')
        if field=='runtime_closure':
            require(a['observation']=='WORKER_OBSERVED_LOAD_MANIFEST' and
                    a['historical_runtime_identity']==current[field],'HISTORICAL_ACTUAL_RUNTIME_REQUIRED')
    events=read(proof['events']);cohort=read(proof['cohort']);correctness=read(proof['correctness'])
    require(events['candidate']==candidate.id and len(events['tasks'])==len(anchors),'FULL_TASK_EVENT_BODIES')
    schedules=json.loads(old.prior_blob('artifacts/resource_applicability/track_a_pf_order_reversal_h2_metadata_receipt/2026-10-23/received/rte_schedule_metadata.json'))
    schedules={r['task_id']:r for r in schedules['tasks']}
    for body,anchor in zip(events['tasks'],anchors):
        saved=anchor['result']['saved'];pieces=schedules[anchor['task']['id']]['pieces']
        require(body['task']==anchor['task']['id'] and body['sample']==anchor['task']['sample'] and
                type(body['events']) is list and len(body['events'])==len(pieces) and
                policy.digest(body['events'])==saved['event_digest'],'SOURCE_EVENT_DIGEST_JOIN')
        require(body['conditional'] is False and body['cohort']=='EXPLORATION','ORIGINAL_EVENT_COHORT')
        for stream,piece in zip(body['events'],pieces):
            require(len(stream)==piece['saved']['rte_steps'],'SIGNED_PIECE_EVENT_COUNT')
            for event in stream:
                order=event['taylor_order']
                require(type(order) is int and order in (0,2,4) and
                        len(event['product_component_ids'])==order and
                        event['selected_component_ids']==event['product_component_ids']+[event['rotation_component_id']] and
                        set(event['phase'])=={'real','imag'},'EVENT_ORDER_COMPONENT_PHASE_SCHEMA')
                sequence=event['application_sequence']
                require([x['application_index'] for x in sequence]==list(range(order+1)) and
                        [x['role'] for x in sequence]==['product']*order+['rotation'] and
                        [x['component_id'] for x in sequence]==event['selected_component_ids'],
                        'EVENT_APPLICATION_WIRE_ORDER')
    require(events['task_event_hashes']==[r['result']['saved']['event_digest'] for r in anchors],'ORIGINAL_EVENT_HASH_LIST')
    require(cohort['candidate']==candidate.id and cohort['conditional'] is False and cohort['pooling'] is False and
            policy.strict_equal(cohort['tasks'],costs),'SAVED_COHORT_AND_PAIRED_RZ')
    require(correctness['candidate']==candidate.id and
            correctness['accepted_tasks']==[r['receipt_sha256'] for r in costs] and
            correctness['scope']=='SINGLE_SAVED_INPUT_STATE','ACCEPTED_CORRECTNESS_SCOPE')
    return witness


def preflight(cfg):
    required={'schema','series','branch','source_commit','physical','compiler','native_policy','allowance_policy',
      'reference_policy','candidates','budget','files','source_files','acquisition_root','provider_root','inputs',
      'binding','runtime','run_root','reuse_witnesses','retry','mandatory_STOP','accepted_diagnostics'}
    require(set(cfg)==required and cfg['schema']==SCHEMA and cfg['series']==SERIES and cfg['branch']==BRANCH,'NEW_COMPARE_CONTRACT_SCHEMA')
    require(policy.strict_equal(cfg['budget'],counts()),'EXACT_NEW_CALL_BUDGET')
    require(cfg['retry'] is False and cfg['mandatory_STOP'] is True,'NEW_COMPARE_STOP_NO_RETRY')
    require(type(cfg['source_commit']) is str and cfg['source_commit']!=accepted.ORIGIN_SOURCE and
            len(cfg['source_commit'])==40 and all(x in '0123456789abcdef' for x in cfg['source_commit']),
            'SEPARATE_COMPARE_SOURCE')
    run=Path(cfg['run_root']);origin=Path(cfg['accepted_diagnostics']['run_root'])
    require(run.is_absolute() and str(run.resolve())==str(run) and not run.exists() and
            not run.is_relative_to(origin) and not origin.is_relative_to(run),'NEW_UNUSED_DISJOINT_RUN_ROOT')
    seen=old.seal_check(cfg);old.source_check(cfg,seen)
    require(set(RUNNER_FILES)<={r['path'] for r in cfg['source_files'] if r['root']=='runner'},'NEW_RUNNER_SOURCE_CLOSURE')
    require(not any(p.is_relative_to(run) or run.is_relative_to(p) for p in seen),'NEW_INPUT_OUTPUT_DISJOINT')
    plan=accepted.inspect_origin(cfg['accepted_diagnostics'],cfg)
    origin_cfg=plan['config'];ref=cfg['accepted_diagnostics']
    needed={ref['contract'],ref['approval'],ref['grant'],str(origin/'completion.json'),str(origin/'CONSUMED.json')}
    needed|={str(origin/p) for p in plan['files']}|{str(p) for p in (origin/'ledger').glob('*.json')}
    needed|={r['path'] for r in origin_cfg['files']}
    require(needed<={str(p) for p in seen},'COMPLETE_ORIGIN_EVIDENCE_SEAL')
    require(cfg['runtime']['python_sha256']==sha(Path(sys.executable).resolve()),'NEW_INTERPRETER_BYTES')
    seal.verify_runtime_binding(cfg)
    blockers=[]
    if len(cfg['reuse_witnesses'])!=2:blockers.append('PR_TWO_COMPLETE_IDENTITY_WITNESSES_REQUIRED')
    else:
        for candidate,path in zip(policy.CANDIDATES[2:],cfg['reuse_witnesses']):
            require(Path(path) in seen,'NEW_PR_WITNESS_NOT_SEALED')
            validate_witness(path,candidate,cfg,plan)
    return dict(status='BLOCKED_PR_IDENTITY' if blockers else 'TECHNICAL_PREFLIGHT_PASS',
        technical_binding_verified=True,execution_ready=not blockers,science_GO=False,
        contract_sha256=policy.digest(cfg),origin_completion_sha256=accepted.ORIGIN_COMPLETION,blockers=blockers)


def authorize(cfg,approval,grant):
    require(approval.get('issuer')=='USER_EXPLICIT' and approval.get('execution_authorized') is True and
            approval.get('contract_sha256')==policy.digest(cfg),'NEW_EXPLICIT_COMPARE_APPROVAL_REQUIRED')
    require(approval.get('policies_adopted')==[cfg[k] for k in ('native_policy','allowance_policy','reference_policy')],
            'INHERITED_FIXED_POLICIES_REQUIRED')
    result=preflight(cfg);require(result['execution_ready'],'PR_IDENTITY_BLOCKS_COMPARE')
    require(set(grant)=={'id','contract_sha256','approval_sha256','run_root','budget','state','retry'} and
            grant['state']=='UNUSED' and grant['retry'] is False and type(grant['id']) is str and bool(grant['id']) and
            grant['id']!=read(Path(cfg['accepted_diagnostics']['run_root'])/'CONSUMED.json')['grant']['id'],
            'NEW_UNUSED_COMPARE_GRANT_REQUIRED')
    require(grant['contract_sha256']==policy.digest(cfg) and grant['approval_sha256']==policy.digest(approval) and
            grant['run_root']==cfg['run_root'] and policy.strict_equal(grant['budget'],counts()),'NEW_COMPARE_GRANT_SCOPE')


class Session(old.Session):
    def __init__(self,cfg,approval,grant):
        authorize(cfg,approval,grant)
        self.cfg,self.used,self.active=cfg,dict.fromkeys(KINDS,0),None
        self.root=Path(cfg['run_root']);self.root.mkdir(exist_ok=False)
        save(self.root/'CONSUMED.json',dict(grant=grant,approval_sha256=policy.digest(approval),state='CONSUMED_NO_RETRY_NO_REFUND'))
        self.marker_sha=sha(self.root/'CONSUMED.json');self.sequence=0
        (self.root/'ledger').mkdir();(self.root/'results').mkdir()

    def take(self,kind,amount=1):
        require(kind in KINDS and type(amount) is int and amount>0,'NEW_CALL_KIND')
        require(self.used[kind]+amount<=self.cfg['budget'][kind],'CALL_BUDGET_EXCEEDED:'+kind)
        self.sequence+=1
        save(self.root/'ledger'/('%06d.json'%self.sequence),dict(kind=kind,amount=amount,nonrefundable=True,
            contract_sha256=policy.digest(self.cfg)))
        self.used[kind]+=amount

    def call(self,kind,fn,*args,**kwargs):
        self.take(kind);previous=self.active;self.active=kind
        try:return fn(*args,**kwargs)
        finally:self.active=previous


def execute(cfg,approval,grant,*,backend_factory=None):
    s=Session(cfg,approval,grant);guard=None
    try:
        s.take('bootstrap')
        if backend_factory is None:
            guard=old.ImportSeal(cfg);guard.install()
            from h2_accepted_compare_backend import Backend
            backend_factory=Backend
        backend=backend_factory(s);s.call('setup',backend.setup)
        for prefix in (3,1):s.call('prepare',backend.prepare,prefix)
        diagnostics=s.call('bundle_load',backend.load_diagnostics);costs={}
        for candidate,path in zip(policy.CANDIDATES[2:],cfg['reuse_witnesses']):
            costs[candidate.id]=backend.reuse(candidate,read(path)['legacy'])
        for candidate in policy.CANDIDATES[:2]:
            s.verify();costs[candidate.id]=s.call('native_validation',backend.native,candidate,diagnostics[candidate.id])
        rows=[]
        for candidate in policy.CANDIDATES:
            u=s.call('allowance',backend.allowance,candidate,diagnostics[candidate.id])
            row=s.call('accounting',backend.account,candidate,diagnostics[candidate.id],u,costs[candidate.id])
            require(row['eligible'] is True and row['G'] is not None,'INELIGIBLE_OR_MISSING_ACCOUNTING_NO_RANK')
            rows.append(row)
        result=s.call('ranking',backend.rank,rows)
        s.verify();s.loaded_closure();require(s.used==cfg['budget'],'INCOMPLETE_NEW_COMPARE_LEDGER')
        files=[dict(path=str(p.relative_to(s.root)),sha256=sha(p)) for p in sorted((s.root/'results').rglob('*')) if p.is_file()]
        save(s.root/'completion.json',dict(status='CONTRACT_COMPLETE_NOT_SCIENTIFIC_GO',series=SERIES,
            contract_sha256=policy.digest(cfg),source_commit=cfg['source_commit'],used=s.used,result=result,
            inherited_diagnostics=dict(contract_sha256=accepted.ORIGIN_CONTRACT,completion_sha256=accepted.ORIGIN_COMPLETION),
            result_files=files,marker_sha256=s.marker_sha,mandatory_STOP=True))
        return result
    except BaseException as exc:
        save(s.root/'STOP.json',dict(status='STOP_PARTIAL_NOT_PASS',reason=str(exc),used=s.used,retry=False,mandatory_STOP=True))
        raise
    finally:
        if guard is not None:guard.remove()
