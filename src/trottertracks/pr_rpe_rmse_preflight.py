"""A-side evidence inventory/dry-run. No numeric provider, N or RMSE evaluation.

Saved values are inspected for presence, never copied as a new RMSE result.
B's pinned metadata validator is reused unchanged; no controller/cost fork.
"""
import ast
from copy import deepcopy
from fractions import Fraction
import gzip
import hashlib
import json
import math
import re
import subprocess
from types import SimpleNamespace

from . import pr_rpe_rmse as model
p = model.common
A_SOURCE = '510fbacd01a2d855202de4c045d8ba1972e5ff3d'
B_SOURCE = '82bc774a92b65cac06b88ebcb2c6979c2726253b'
B_PATH = 'src/trottertracks/pr_rpe_rmse_controller.py'
SOURCE = dict(module='src/trottertracks/pr_rpe_rmse_preflight.py', A=A_SOURCE, B=B_SOURCE)
MEAN_FIELDS = ('reference', 'corrected', 'raw', 'exact_tail', 'total_signed_bias', 'log_B')
ERROR_FIELDS = ('maximum_intermediate_norm', 'raw_corrected_vector_closure', 'exact_tail_norm_error')
REFERENCE_FIELDS = ('reference', 'independent_reference', 'signal_difference')
BINDING_KEYS = ('system', 'hamiltonian', 'state', 'snapshot', 'rank', 'rank_policy')


def b_metadata(root):
    """Only unchanged pinned B metadata ASTs; no assess/cost/select imports."""
    def load(path, names, scope):
        data = subprocess.check_output(['git', 'show', B_SOURCE+':'+path], cwd=root)
        nodes = [n for n in ast.parse(data).body if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in nodes} != set(names):
            raise ValueError('Pinned B metadata functions missing')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), B_SOURCE+':'+path, 'exec'), scope)
        return scope
    exact = load('src/trottertracks/h4_df10_rpe_connection.py',
                 ('exact_scalar', 'exact_value'), dict(Fraction=Fraction, math=math))
    scope = dict(deepcopy=deepcopy, hashlib=hashlib, json=json, p=p,
                 a=SimpleNamespace(**{k:exact[k] for k in ('exact_scalar', 'exact_value')}),
                 SCHEMA='PR_RPE_RMSE_V1', METHODS=('PF', 'PR', 'FULL_H_RTE'),
                 DRAFT_MAPPER=model.MAPPING)
    names = ('_id', '_positive', '_ceil', 'paper_event_mapper', 'generate', '_validate')
    load(B_PATH, names, scope)
    return SimpleNamespace(validate=scope['_validate'], generate=scope['generate'],
                           exact=exact['exact_scalar'])


def present(mapping, fields):
    complex_fields = {'reference','independent_reference','corrected','raw','exact_tail'}
    return [k for k in fields if k in mapping and mapping[k] is not None and (
        k not in complex_fields or isinstance(mapping[k],dict) and all(
            part in mapping[k] and mapping[k][part] is not None for part in ('real','imag')))]


def descriptor(identity, signal, source, status):
    """Field-presence metadata only. Old plan/N/phase values cannot escape."""
    return dict(candidate=deepcopy(identity['candidate']), binding=deepcopy(identity['binding']),
                structure=deepcopy(identity['structure']), profile=deepcopy(identity['profile']),
                source=deepcopy(source), saved_status=status,
                fields=present(signal, MEAN_FIELDS),
                error_fields=present(signal.get('allowance_evidence', {}), ERROR_FIELDS),
                numerical_values_copied=False, RMSE_qualification='NOT_EVALUATED')


def saved_inventory(root):
    """Read fixed Git carriers, selecting metadata/field presence, not numbers.

    This is deliberately not a mean_provider. D1 rows without action records are
    retained as incomplete; no raw=corrected/B reconstruction or fallback run.
    """
    from . import pr_r_dependent_n as old
    tasks = {s:old.saved_task(root, s) for s in ('H4', 'H6')}
    store = p.SavedArtifacts(root)
    sources = (
        ('71ba49bb28b295aa28e51101bc3b1bf07d500767', 'pr_r_detailed_n_p1_results_20261111/run01', 'records'),
        ('89c91c692e1d35de94d2e7afbe890aabe6d80b7e', 'h4_df10_rpe_batch_results_20261110/run01', 'records'),
        ('f4c0bdabdf2afdc5bc4f20573bb10677917f4c1a', 'h6_df19_rpe_m8_12_results_20261110/run01', 'new_records'),
        ('95d01b6ab1d820921fcd2521324e41e82389713a', 'h6_df19_rpe_d2_results_20261110/run01', 'rows'),
        ('f4c0bdabdf2afdc5bc4f20573bb10677917f4c1a', 'h6_df19_rpe_m6_results_20261110/run01', 'rows'),
        ('f4c0bdabdf2afdc5bc4f20573bb10677917f4c1a', 'h6_df19_rpe_m7_results_20261110/run01', 'rows'),
        ('45607a5086721ea1d21b1ed2d7edfe870549f0c8', 'h6_df19_rpe_d1_results_20261110', 'rows'))
    means, refs = [], []
    for commit, directory, key in sources:
        origin = dict(commit=commit, path='docs/research/'+directory+'/result.json')
        doc = store.read(origin)
        records = doc[key]
        binding = None
        for j, record in enumerate(records):
            if 'identity' in record:
                ident = record['identity']; diag = record.get('diagnostics', {})
                status = record.get('plan', {}).get('status', 'NOT_EVALUATED')
            else:
                binding = doc.get('scientific_binding') or doc['input_bindings'].get('scientific') or doc['input_bindings'].get('binding')
                if not binding:raise ValueError('Saved scientific binding missing')
                if any(binding[k] != tasks['H6']['binding'][k] for k in BINDING_KEYS):
                    raise ValueError('Saved H6 input binding differs')
                c = record['candidate']
                ident = dict(candidate=c, binding=deepcopy(tasks['H6']['binding']),
                             structure=p.structure(c), profile=deepcopy(tasks['H6']['profile']))
                diag = record
                status = record.get('ideal_numerical_pilot', {}).get('status', 'NOT_EVALUATED')
            c = ident['candidate']
            if c['id'] != p.candidate_id(c):raise ValueError('Saved candidate ID differs')
            source = dict(origin, selector=[key, j], scientific_source=doc['scientific_source'])
            means.append(descriptor(ident, diag.get('signal', {}), source, status))
        reference_map = doc.get('references', doc.get('scope_observed', {}).get('references', {}))
        if doc.get('reference'):
            reference_map = {str(records[0]['candidate']['T']):doc['reference']}
        for key_time, value in reference_map.items():
            if 'evidence' in value:
                b, T, evidence = value['binding'], value['T'], value['evidence']
            else:
                system = 'H4' if directory.startswith('h4') else 'H6'
                b, T, evidence = tasks[system]['binding'], float(key_time), value
            refs.append(dict(binding=deepcopy(b), T=T,
                source=dict(origin, reference_key=key_time, scientific_source=doc['scientific_source']),
                fields=present(evidence, REFERENCE_FIELDS), numerical_gate_rechecked=False))
    # Original D1 signal carrier: join exact recipes to already bound result rows.
    # No endpoint N/margin, raw reconstruction or new allowance is evaluated.
    raw_commit = '155529e20a8fac39e00786c40b57830b71203c4f'
    raw_path = 'artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/signal_records.json.gz'
    data = subprocess.check_output(['git','show',raw_commit+':'+raw_path],cwd=root)
    raw = json.loads(gzip.decompress(data))
    store.sources[raw_commit+':'+raw_path] = hashlib.sha256(data).hexdigest()
    for saved in list(means):
        if set(MEAN_FIELDS) <= set(saved['fields']):continue
        c = saved['candidate']; old_record = raw.get(c['id'])
        if old_record is None:continue
        candidate = deepcopy(old_record['candidate']); candidate.setdefault('id',c['id'])
        if not p.same(candidate,c):raise ValueError('Original signal carrier recipe differs')
        source = dict(commit=raw_commit,path=raw_path,selector=c['id'],binding_carrier=saved['source'])
        means.append(descriptor(saved,old_record['signal'],source,saved['saved_status']))
    return dict(schema='A_RMSE_EVIDENCE_INVENTORY_V1', source=SOURCE, means=means, references=refs,
                carriers=store.sources, numerical_values_copied=False, science_calls=0,
                task_sources=[dict(commit=old.TASK_CARRIER,path=
                    'docs/research/pf_pr_rpe_cost_integration_20261110/static_plan/'+s+'_PLAN.json') for s in tasks],
                qualification='NOT_EVALUATED', old_N_imported=False)


def _binding_same(left, right):
    # Every A binding field must be preserved; B may add decomposition/SOURCE.
    return all(k in right and p.same(v, right[k]) for k,v in left.items())


def same_source(left, right, root):
    if p.same(left,right):return True
    if not all(isinstance(v,dict) and re.fullmatch('[0-9a-f]{40}',v.get('commit',''))
               and v.get('path') and not v['path'].startswith('/') and '..' not in v['path'].split('/')
               for v in (left,right)):return False
    try:
        blobs=[subprocess.check_output(['git','rev-parse','--verify',v['commit']+':'+v['path']],cwd=root,
                                      stderr=subprocess.DEVNULL) for v in (left,right)]
    except subprocess.CalledProcessError:return False
    return blobs[0]==blobs[1]


def dry_run(config, schedule, inventory, *, root):
    """Validate supplied metadata; return gaps and an unevaluated B N packet.

    Does not create a scientific design, invoke providers, or compute nominal N.
    Structural mismatch stops planning for that row. Scientific qualification
    and all-round decoder RMSE remain external even with complete saved fields.
    """
    b = b_metadata(root); b.validate(config)
    model.evaluate(schedule)  # provider=None: validate A hashes/recipes only.
    s, cfg = config['spec'], schedule['task']['configuration']
    reasons = []
    method = s['method']
    if method not in ('PF', 'PR'):reasons.append('FULL_H_RTE_NOT_IMPLEMENTED_BY_A')
    if s['M'] != cfg['M_max_index']:reasons.append('M_OR_ROUND_COVERAGE_MISMATCH')
    if not _binding_same(schedule['task']['binding'], s['binding']):reasons.append('BINDING_OR_SOURCE_MISMATCH')
    if not p.same(schedule['task']['profile'], s['profile']):reasons.append('WRAPPER_COMPILER_PROFILE_MISMATCH')
    if s['rmse_policy']['family'] != 'PR_PAPER_RMSE':reasons.append('OLD_POLICY_IS_NOT_RMSE')
    if b.exact(s['rmse_policy']['target_rmse_ha']) != Fraction(cfg['policy']['target_rmse']):
        reasons.append('RMSE_TARGET_MISMATCH')
    if b.exact(s['h']) != Fraction(cfg['base_time']):reasons.append('BASE_TIME_CONVENTION_MISMATCH')
    if method == 'PR':
        if (s['kappa']['kind'] != 'PAPER_TOTAL_ROTATIONS'
                or cfg['kappa_definition'] != model.KAPPA_DEFINITION
                or b.exact(s['kappa']['value']) != Fraction(cfg['kappa_paper'])
                or s['event_mapping']['kind'] != model.MAPPING):reasons.append('KAPPA_DEFINITION_OR_MAPPING_MISMATCH')
        if (b.exact(s['split']['lambda_R']) != Fraction(cfg['tail_norm']['lambda_R'])
                or not p.same(s['split']['lambda_source'], cfg['tail_norm']['source'])):
            reasons.append('TAIL_NORM_OR_SOURCE_MISMATCH')
    selected = [r for r in schedule['rows'] if p.arm(r['identity']['candidate']) == method]
    if sorted(r['identity']['m'] for r in selected) != list(range(s['M']+1)):
        reasons.append('A_MISSING_DUPLICATE_OR_FOREIGN_ROUNDS')
    byround = {r['identity']['m']:r for r in selected}
    rows = []
    for row in config['rounds']:
        m = row['m']; request = byround.get(m); problems = list(reasons)
        c = None if request is None else request['identity']['candidate']
        if c is None:problems.append('MISSING_A_REQUEST')
        else:
            if any(request['identity'][k] != row[k] for k in ('q_RPE','q_PF')):
                problems.append('Q_RPE_OR_Q_PF_MISMATCH')
            if not same_source(request['law_binding']['source'],s['law']['source'],root):
                problems.append('LAW_SOURCE_JOIN_NOT_CONFIRMED')
            if method == 'PF' and c['method'] != 'FULL':problems.append('PF_B0_IS_NOT_B_FULL_H_BASELINE')
            for key in ('system','prefix','formula','q','K','R_outer'):
                if c[key] != row['candidate'][key]:problems.append('CANDIDATE_'+key+'_MISMATCH')
            for key in ('T','delta'):
                if c[key] is None or row['candidate'][key] is None or Fraction(c[key]) != b.exact(row['candidate'][key]):
                    problems.append('CANDIDATE_'+key+'_CONVENTION_MISMATCH')
            if not p.same(request['identity']['structure'], row['structure']):problems.append('SIGNED_ALLOCATION_OR_LAW_MISMATCH')
            if request['mapping']['early_rule'] and method == 'PR':
                problems.append('B_FIXED_L_D_CANNOT_REPRESENT_H6_EARLY_OVERRIDE')
        matches, rejected, references = [], [], []
        if c is not None:
            for saved in inventory.get('means', []):
                if saved['candidate'].get('id') != c['id']:continue
                if (not p.same(saved['candidate'], c) or not _binding_same(request['identity']['binding'], saved['binding'])
                        or not p.same(saved['structure'], request['identity']['structure'])
                        or not p.same(saved['profile'], request['identity']['profile']) or not saved.get('source')):
                    rejected.append(dict(source=saved.get('source'), reason='SAVED_SOURCE_IDENTITY_MISMATCH'));continue
                matches.append({k:deepcopy(saved[k]) for k in (
                    'candidate','binding','structure','profile','source','saved_status','fields','error_fields')})
            for ref in inventory.get('references', []):
                if ref['T'] == c['T'] and _binding_same(request['identity']['binding'], ref['binding']) and ref.get('source'):
                    references.append({k:deepcopy(ref[k]) for k in ('binding','T','source','fields')})
        stopped = [v for v in matches if any(t in v['saved_status'] for t in ('FAIL','STOP','UNSTABLE','INFEASIBLE','UNRESOLVED'))]
        if stopped:problems.append('SAVED_FAILURE_OR_UNRESOLVED_NO_AUTOMATIC_RETRY')
        if rejected:problems.append('SAVED_SOURCE_IDENTITY_MISMATCH')
        complete = [v for v in matches if set(MEAN_FIELDS) <= set(v['fields']) and set(ERROR_FIELDS) <= set(v['error_fields'])]
        best = max(matches, key=lambda v:len(v['fields'])+len(v['error_fields']), default=dict(fields=[],error_fields=[]))
        pairs = [r for r in references if set(REFERENCE_FIELDS) <= set(r['fields'])]
        blocked = bool(problems)
        rows.append(dict(m=m, A_request_id=None if request is None else request['request_id'],
            A_candidate_id=None if c is None else c['id'], B_candidate_id=row['candidate']['id'],
            status='BLOCKED_IDENTITY_OR_SAVED_STATUS' if blocked else 'TECHNICALLY_PLANNABLE_NOT_QUALIFIED',
            issues=sorted(set(problems)), saved_mean_candidates=deepcopy(matches),
            rejected_saved_evidence=rejected, saved_reference_candidates=deepcopy(references),
            missing_mean_fields=[k for k in MEAN_FIELDS if k not in best['fields']]+[
                k for k in ERROR_FIELDS if k not in best['error_fields']],
            additional_scientific_calls_if_approved=None if blocked else dict(
                finite_mean_candidates_upper_bound=0 if complete else 1,
                exact_tail_internal_controls_upper_bound=0 if complete or method=='PF' else 1,
                independent_reference_actions=0 if pairs else 2,
                recover_saved_diagnostics_first=bool(matches) and not bool(complete),
                saved_numeric_gate_rechecks=True, RMSE_policy_reanalysis_required=True),
            nominal_N_X=None, nominal_N_Y=None, N_X=None, N_Y=None))
    packet = dict(configuration_id=config['id'], policy=deepcopy(s['rmse_policy']), source=SOURCE,
        evidence_kind='PREFLIGHT_ONLY', status='NOT_EVALUATED',
        rounds=[dict(m=r['m'],identity=deepcopy(r['identity']),law_binding=deepcopy(r['law_binding']),N_X=None,N_Y=None)
                for r in config['rounds']],
        rmse_evidence=dict(source=SOURCE,status='NOT_EVALUATED',error_contract=deepcopy(s['rmse_policy']['error_contract']),
            rmse_bound_ha=None,covers_state_target_alias_prior_decoder_all_rounds=False,
            finite_K_status='NOT_EVALUATED',signed_piece_kappa_status='NOT_EVALUATED'))
    return dict(schema='A_RMSE_PREFLIGHT_V1', source=SOURCE,
        status='BLOCKED' if any(r['issues'] for r in rows) else 'TECHNICAL_PREFLIGHT_COMPLETE_NOT_SCIENCE',
        A_configuration_id=schedule['configuration_id'], B_configuration_id=config['id'],
        global_issues=sorted(set(reasons)), rounds=rows, N_packet=packet,
        required_external_evidence=list(model.QUALIFICATIONS)+[
            'B_decomposition_SOURCE_join', 'RMSE_error_contract_adoption', 'normalization_policy_calibration',
            'all_round_decoder_RMSE_evidence'],
        supplied_outer_conditions=dict(M=cfg['M_max_index'],h=cfg['base_time'],
            energy_prior_ha=deepcopy(cfg['energy_prior_ha']),decoder=cfg['decoder'],
            state_assumption=schedule['state_assumption'],
            A_error_policy=deepcopy(cfg['policy']),B_error_contract=deepcopy(s['rmse_policy']['error_contract']),
            prior_decoder_error_contract_join_qualified=False),
        science_calls=0, new_N_or_RMSE=False, full_RPE_qualified=False, mandatory_stop=True)
