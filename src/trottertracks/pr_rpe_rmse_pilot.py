"""PR two-time planning, handoff and explicit supplied-disk diagnostics.

B f09's A-only metadata functions are reused unchanged. Planning/handoff invoke
no numerical leaf. Real supplied-disk use is a separate scientific reanalysis;
only artificial disks are tested here. No Hamiltonian loader or runner is used.
"""
import ast
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
import subprocess
from types import SimpleNamespace

from . import pr_rpe_rmse_preflight as previous
A = previous.model
p = previous.p
B_SOURCE = 'f09af82482aad11460c4bb73d3e13d7ada40be6f'
A_PREFLIGHT = 'e2a472422f0206a2f09fcc8e28d6938ebceb6d0f'
ADAPTER = 'A_FIXED_GRID_510FBACD_V1'
ROUNDS = {'H4':(0,6), 'H6':(4,8)}
SOURCE = dict(module='src/trottertracks/pr_rpe_rmse_pilot.py', B=B_SOURCE,
              A_model=previous.A_SOURCE, A_preflight=A_PREFLIGHT)
P1='71ba49bb28b295aa28e51101bc3b1bf07d500767'
P1_DIR='docs/research/pr_r_detailed_n_p1_results_20261111/'


def pilot_inventory(root):
    """Augment the old field-only catalog with P1's archived ideal-law SOURCE.

    Reads metadata and code provenance only, not a normalizer or signal action.
    Wrapper declaration and underlying RTE SOURCE are distinct links in the chain.
    """
    store=p.SavedArtifacts(root)
    out=store.read(dict(commit=A_PREFLIGHT,path='docs/research/pr_rpe_rmse_science_preparation_20261112/SAVED_INVENTORY.json'))
    result=store.read(dict(commit=P1,path=P1_DIR+'run01/result.json'))
    index=store.read(dict(commit=P1,path=P1_DIR+'SOURCE_INDEX.json'))
    implementation=dict(commit=P1,path='src/trotterlib/rte.py')
    data=subprocess.check_output(['git','show',P1+':'+implementation['path']],cwd=root)
    # The index schema is fixed; only its existing numerical SOURCE digest is used.
    if hashlib.sha256(data).hexdigest()!=index['source_files'][implementation['path']]:
        raise ValueError('P1 finite-RTE implementation SOURCE differs')
    for entry in out['means']:
        if entry['source']['commit']!=P1:continue
        key,j=entry['source']['selector'];record=result[key][j]
        if (not p.same(record['identity']['candidate'],entry['candidate'])
                or not p.same(record['identity']['binding'],entry['binding'])):
            raise ValueError('P1 law/carrier identity differs')
        entry['law_binding']=deepcopy(record['law_binding'])
        entry['law_implementation_source']=implementation
        entry['law_source_evidence']=dict(commit=P1,path=P1_DIR+'SOURCE_INDEX.json',
            scope='ARCHIVED_IDEAL_FINITE_MEAN_SOURCE_NOT_ACTUAL_SAMPLER_QUALIFICATION')
    out['pilot_source']=deepcopy(SOURCE)
    return out


def b_metadata(root):
    """Pinned B assembly/validation and A bridge only; excludes C/cost/select."""
    def load(path, names, scope):
        data = subprocess.check_output(['git','show',B_SOURCE+':'+path],cwd=root)
        nodes = [n for n in ast.parse(data).body if isinstance(n,ast.FunctionDef) and n.name in names]
        if {n.name for n in nodes} != set(names):raise ValueError('Pinned B metadata source missing')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),B_SOURCE+':'+path,'exec'),scope)
    exact = dict(Fraction=Fraction,math=math)
    load('src/trottertracks/h4_df10_rpe_connection.py',('exact_scalar','exact_value'),exact)
    b = SimpleNamespace(p=p,a=SimpleNamespace(**{k:exact[k] for k in ('exact_scalar','exact_value')}))
    bs = dict(p=p,a=b.a,deepcopy=deepcopy,hashlib=hashlib,json=json,
              SCHEMA='PR_RPE_RMSE_V1',DRAFT_MAPPER=A.MAPPING)
    load(previous.B_PATH,('_id','_assemble','_validate'),bs)
    b._assemble = bs['_assemble']
    def validate(config):
        if config['spec'].get('schedule_adapter') != ADAPTER:
            raise ValueError('Use the fixed B A-adapter configuration, not a native/draft alias')
        bs['_validate'](config)
    b._validate = validate
    scope = dict(A=A,B=b,p=p,deepcopy=deepcopy,Fraction=Fraction,ADAPTER=ADAPTER,A_RELEASE=previous.A_SOURCE)
    names = ('_exact','_schedule','a_spec','generate_from_a_spec','n_from_a')
    load('src/trottertracks/pr_rpe_rmse_integration.py',names,scope)
    bs['generate'] = scope['generate_from_a_spec']
    return SimpleNamespace(validate=validate,generate=scope['generate_from_a_spec'],
        a_spec=scope['a_spec'],n_from_a=scope['n_from_a'],schedule=scope['_schedule'])


def _validate(config, schedule, root):
    if not config or not schedule:raise ValueError('Fixed B configuration and A schedule required; no M/kappa defaults')
    b = b_metadata(root); b.validate(config)
    if config['spec']['method'] != 'PR':raise ValueError('This pilot is PR only; PF baseline is a separate scope')
    A.evaluate(schedule)  # no provider: identity checks only
    if not p.same(schedule,b.schedule(config['spec']['A_input'])):
        raise ValueError('B/A whole schedule, common kappa, M or SOURCE differs')
    system = schedule['design']['system']
    if system not in ROUNDS:raise ValueError('Explicit H4/H6 pilot system required')
    rounds = ROUNDS[system]
    if config['spec']['M'] < max(rounds):raise ValueError('M must include both pilot rounds; no automatic horizon change')
    requests = [r for r in schedule['rows'] if r['identity']['m'] in rounds and p.arm(r['identity']['candidate'])=='PR']
    if len(requests)!=2:raise ValueError('Exactly the fixed two PR requests are required')
    for request in requests:
        row = config['rounds'][request['identity']['m']]
        if (not p.same(row['physical_candidate'],request['identity']['candidate'])
                or not p.same(row['structure'],request['identity']['structure'])
                or not p.same(row['round_law_definition']['A_declaration'],request['law_binding'])):
            raise ValueError('Physical candidate/signed allocation/law identity differs')
    return b, requests


def _evidence(request, inventory, root):
    """Reuse previous field inventory, never read its numeric carrier here."""
    i=request['identity']; c=i['candidate']; means=[]; refs=[]; issues=[]; pending=[]
    for saved in inventory.get('means',[]):
        if saved['candidate'].get('id') != c['id']:continue
        if (not p.same(saved['candidate'],c) or not previous._binding_same(i['binding'],saved['binding'])
                or not p.same(saved['structure'],i['structure']) or not p.same(saved['profile'],i['profile'])
                or not saved.get('source')):
            issues.append('SAVED_SOURCE_IDENTITY_MISMATCH');continue
        if any(word in saved['saved_status'] for word in ('FAIL','STOP','UNSTABLE','UNRESOLVED','INFEASIBLE')):
            issues.append('SAVED_FAILURE_OR_UNRESOLVED_NO_AUTOMATIC_RETRY')
        law=saved.get('law_binding')
        if not law or not saved.get('law_source_evidence'):
            pending.append(deepcopy(saved['source']));continue
        if (any(law.get(k)!=v for k,v in dict(candidate_id=c['id'],R_outer=c['R_outer'],K=c['K'],q_PF=c['q']).items())
                or law['definition'].get('model')!=i['structure']['law_model']
                or law['definition'].get('sampling')!='IDEAL_FINITE_RTE_MEAN'
                or not law['definition'].get('source')
                or not previous.same_source(saved.get('law_implementation_source'),request['law_binding']['source'],root)):
            issues.append('SAVED_LAW_SOURCE_MISMATCH');continue
        # A source locator is not a new RMSE result; exclude all old N fields.
        means.append({k:deepcopy(saved[k]) for k in ('source','saved_status','fields','error_fields','law_binding','law_implementation_source','law_source_evidence')})
    for ref in inventory.get('references',[]):
        if ref['T']==c['T'] and previous._binding_same(i['binding'],ref['binding']) and ref.get('source'):
            refs.append({k:deepcopy(ref[k]) for k in ('source','fields')})
    if pending and not means:issues.append('SAVED_LAW_EVIDENCE_MISSING_RECOVER_BEFORE_SCIENCE')
    complete=any(set(previous.MEAN_FIELDS)<=set(v['fields']) and set(previous.ERROR_FIELDS)<=set(v['error_fields']) for v in means)
    pair=any(set(previous.REFERENCE_FIELDS)<=set(v['fields']) for v in refs)
    best=max(means,key=lambda v:len(v['fields'])+len(v['error_fields']),default=dict(fields=[],error_fields=[]))
    return dict(m=i['m'],request_id=request['request_id'],candidate=deepcopy(c),structure=deepcopy(i['structure']),
        law_binding=deepcopy(request['law_binding']),saved_mean_candidates=means,saved_reference_candidates=refs,
        saved_law_not_yet_verified=pending,
        missing_fields=[k for k in previous.MEAN_FIELDS if k not in best['fields']]+[
            k for k in previous.ERROR_FIELDS if k not in best['error_fields']],issues=sorted(set(issues)),
        future_calls_upper_bound=None if issues else dict(
            finite_mean_candidates=0 if complete else 1,exact_tail_internal_controls=0 if complete else 1,
            independent_reference_actions=0 if pair else 2,
            recover_saved_diagnostics_first=bool(means) and not complete,
            saved_numeric_gate_rechecks=True,RMSE_policy_reanalysis_required=True),
        numerical_gates_rechecked=False,nominal_N_X=None,nominal_N_Y=None,N_X=None,N_Y=None)


def prepare_pilot(config, schedule, inventory, *, root):
    """Requires supplied fixed M/kappa/policy. No defaults or real evaluation."""
    b,requests=_validate(config,schedule,root)
    rows=[_evidence(r,inventory,root) for r in requests]
    # Reuse the final B packet translator with the unchanged empty A evaluation.
    empty=A.b_handoff(schedule,A.evaluate(schedule),SOURCE)
    packet=next(v for v in empty['packets'] if v['spec']['method']=='PR')
    n=b.n_from_a(config,packet,inputs=config['spec']['A_input'])
    n['evidence_kind']='PREFLIGHT_ONLY'
    return dict(schema='PR_TWO_TIME_RMSE_PILOT_PREPARATION_V1',source=deepcopy(SOURCE),
        status='BLOCKED_SAVED_EVIDENCE' if any(r['issues'] for r in rows) else 'TECHNICALLY_PREPARED_NOT_SCIENTIFICALLY_QUALIFIED',
        system=schedule['design']['system'],A_configuration_id=schedule['configuration_id'],B_configuration_id=config['id'],
        pilot_rounds=list(ROUNDS[schedule['design']['system']]),rounds=rows,N_packet=n,
        full_schedule_round_count=config['spec']['M']+1,
        conditions=dict(M=config['spec']['M'],kappa=deepcopy(config['spec']['kappa']),
            A_input=deepcopy(config['spec']['A_input']),B_error_policy=deepcopy(config['spec']['rmse_policy'])),
        required_scientific_conditions=list(A.QUALIFICATIONS)+[
            'reference_pair_and_numeric_allowance_gates','finite_B_or_paper_envelope_policy_calibration',
            'whole_schedule_RMSE_evidence_not_supplied_by_two_times'],
        scientific_PASS_FAIL=None,full_RPE_qualified=False,science_calls=0,mandatory_stop=True)


def supplied_disk_geometry(attenuation, raw_error, *, error_source):
    """Existing c1 _assess_raw_signal geometry, separated from its Hoeffding N.

    Only supplied raw-unit error disks are consumed. A missing state/law/error
    allowance must stay None, never zero. This diagnostic qualifies no RMSE.
    """
    out=dict(status='UNRESOLVED_SUPPLIED_ERROR_DISK',radius_lower=None,beta_deterministic=None,
             error_source=deepcopy(error_source),RMSE_qualified=False,N_X=None,N_Y=None,
             geometry_source='src/trottertracks/c1_stage3_accuracy.py:_assess_raw_signal')
    if attenuation is None or raw_error is None or not error_source:return out
    if (any(type(v) not in (int,float) or not math.isfinite(v) for v in (attenuation,raw_error))
            or not 0<attenuation<=1 or raw_error<0):raise ValueError('Finite raw-unit attenuation/error disk required')
    rho=attenuation-raw_error
    if rho<=0:return dict(out,status='INFEASIBLE_SUPPLIED_RADIUS',radius_lower=rho)
    return dict(out,status='LOCAL_SUPPLIED_DISK_DIAGNOSTIC_ONLY',radius_lower=rho,
                beta_deterministic=math.asin(raw_error/attenuation))


def handoff_supplied_results(plan, config, schedule, records, source, *, root,
                             evidence_domain='UNQUALIFIED_MODEL'):
    """Package already evaluated A records only, never invoke a numeric leaf.

    The caller needs separate science permission to obtain real records. Partial
    or empty results retain nulls elsewhere; neither nominal nor old HP N may be
    inserted as sufficient N. No approval certificate/grant is interpreted here.
    """
    b,requests=_validate(config,schedule,root)
    if (plan['A_configuration_id']!=schedule['configuration_id'] or plan['B_configuration_id']!=config['id']
            or plan['pilot_rounds']!=list(ROUNDS[schedule['design']['system']]) or any(v['issues'] for v in plan['rounds'])):
        raise ValueError('Pilot plan scope changed or saved failure remains unresolved')
    selected={r['request_id']:r for r in requests}; supplied={}
    for record in records:
        rid=record.get('request_id')
        if rid not in selected or rid in supplied:raise ValueError('Foreign/duplicate pilot result; no extra rounds/PF')
        if (any(k in record for k in ('plan','actual','N_X','N_Y'))
                or not p.same(record['identity'],selected[rid]['identity']) or record.get('policy_id')!=A.POLICY
                or any(record.get(k) is not None for k in ('sufficient_N_X','sufficient_N_Y','actual_N_X','actual_N_Y','C','G','scientific_PASS_FAIL'))
                or record.get('rmse_guaranteed') is not False):
            raise ValueError('Result identity/policy differs or nominal N promoted to science guarantee')
        if any(record.get(k) is not None and (type(record[k]) is not int or record[k]<1)
               for k in ('nominal_N_X','nominal_N_Y')):
            raise ValueError('Nominal N must remain null or an exact positive integer')
        supplied[rid]=deepcopy(record)
    evaluation=A.evaluate(schedule)  # no means, no nominal calculation
    evaluation['records']=[supplied.get(r['request_id'],r) for r in evaluation['records']]
    original=next(v for v in A.b_handoff(schedule,evaluation,source,evidence_domain=evidence_domain)['packets'] if v['spec']['method']=='PR')
    out=b.n_from_a(config,original,inputs=config['spec']['A_input'])
    out['pilot_scope']=dict(rounds=plan['pilot_rounds'],completed_request_ids=list(supplied),
                            whole_schedule_RMSE_qualified=False)
    return out
