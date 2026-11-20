"""Run ONE optimizer Q3 request on an existing method route, then write OUTCOME.json.

Launched by the optimizer's executor with the route's own interpreter.  The
numerics are the existing callers' workers/supervisors loaded unchanged from a
pinned read-only route view (a detached worktree at the route commit); only the
single case is built here: the saved template case of the same configuration
and evaluation contract with the requested allocation.

Modes:
  execute    build the case, run the route once, validate, write OUTCOME.json
  collect    validate existing worker artifacts only (resume after a crash), no science
  dry-check  build the case and run route metadata/count/runtime checks only, no science

Route kinds: pf_b_pf6, pauli_b_ld3, pauli_b_ld5, pauli_b_common_entry, prdf_d2_split, artificial.

pauli_b_common_entry: any PR-Pauli configuration whose 13-round MEAN_HANDOFF was assembled by A's common
signal entry (93a5ef3b9) and is accepted by B's normal intake (``accept_saved(handoff_source=...)``, B
bd42dd1e9).  The case comes from the route's ``configurations`` table (configuration + packet locators, total
binding ids) instead of a saved case; the evaluation contract (statistical contract, options, adopted
evaluation, acceleration, kernel and C sources, evaluation target) is copied from ``contract_template``
(the saved LD5 fixed-Q3 INPUTS).  The worker is B's saved LD5/LD3 worker with only its metadata generalized.
"""
import argparse
import ast
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from fractions import Fraction
import importlib.util
import hashlib
import json
import os
import resource
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace as NS

_timing_path=Path(__file__).resolve().parents[1]/'src/trottertracks/rpe_q3_timing.py'
_timing_spec=importlib.util.spec_from_file_location('_q3_operational_timing',_timing_path)
TIMING=importlib.util.module_from_spec(_timing_spec);_timing_spec.loader.exec_module(TIMING)

_evidence_path=Path(__file__).resolve().parents[1]/'src/trottertracks/rpe_q3_evidence.py'
_evidence_spec=importlib.util.spec_from_file_location('_q3_output_evidence',_evidence_path)
EVIDENCE=importlib.util.module_from_spec(_evidence_spec);_evidence_spec.loader.exec_module(EVIDENCE)

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)
PASS = 'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'
FAIL = 'MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD'
VALIDATOR = dict(commit='cbc471c913072de45c25a0ed22d51dff8a9b7034', path='scripts/run_p7_s2_local_shot_search.py',
                 function='validate_result')
DROP = {'allocation', 'label', 'price_terms', 'conditional_cost', 'generation', 'rule', 'budget', 'budget_name', 'aliases',
        'finite_scope_preflight', 'reference_cost', 'unused_budget', 'cost_m0_m6', 'cost_m7_m12', 'm7_m12_fraction',
        'base_label', 'benchmark_cost', 'comparison_name', 'A_comparison_source', 'LD3_comparison', 'equivalent_Q3_result',
        'cost_order', 'saved_identical_vector', 'edit', 'parent_allocation', 'reduction_RZ', 'reference_cost_RZ',
        'conditional_cost_m0_m11', 'conditional_cost_m12', 'family', 'phase'}
CONTRACT = ('statistical_contract', 'options', 'adopted_evaluation')


def utc():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def stage(out, name, **extra):
    with (Path(out) / 'JOB_STAGES.jsonl').open('a') as f:
        f.write(json.dumps(dict(TIMING.stamp(), stage=name, UTC=utc(), **extra)) + '\n')


def git_json(view, locator):
    value = json.loads(subprocess.check_output(['git', 'show', locator['commit'] + ':' + locator['path']], cwd=view))
    for key in locator.get('inputs_selector', []):
        value = value[key]
    return value


def template(view, request):
    """Saved Q3 inputs/case of the same configuration: the only source of the case and contract."""
    for loc in request['route']['templates']:
        inputs = git_json(view, loc)
        for case in inputs['cases']:
            cid = case.get('configuration_id') or case.get('original_configuration_id')
            if cid == request['configuration_id']:
                return loc, inputs, case
    raise ValueError('No saved template case for this configuration on this route')


def common_case(view, request):
    rule = request['route'];entry = rule['configurations'][request['configuration_id']]
    saved = git_json(view, rule['contract_template']);base = saved['cases'][0]
    case = dict(label=request['request_id'], configuration=deepcopy(entry['configuration']), signal_packet=deepcopy(entry['signal_packet']),
                original_configuration_id=request['configuration_id'], evaluation_target=deepcopy(base['evaluation_target']),
                binding_id=entry['binding_id'], total_configuration_id=entry['total_configuration_id'])
    if request['route']['kind']=='prdf_d2_common_entry':
        case['signal_packet_kind']='A_BUNDLE_REQUIRES_EXISTING_K_ACCEPT_SAVED'
    case['allocation'] = dict(allocation_id='RPE_OPT_' + request['request_id'], shot_policy='USER_SUPPLIED',
                              shots=[dict(m=m, N_X=x, N_Y=y) for m, (x, y) in enumerate(zip(request['N_X'], request['N_Y']))])
    inputs = {k: deepcopy(saved[k]) for k in ('kernel_source', 'C_source', 'acceleration') + CONTRACT if k in saved}
    inputs.update(cases=[case], concurrent_configurations=1, maximum_scientific_calls=1,
                  template_source=dict(rule['contract_template'], role='EVALUATION_CONTRACT_ONLY'), signal_source=entry['signal_packet'])
    return case, inputs


def horizon_of(request):
    """Verify the whole integer allocation before routing any scaled request."""
    if request.get('origin', {}).get('kind') != 'RPE_SCALED_SHOTS_V1':
        if request['route'].get('horizon_projection'):
            raise ValueError('A projected route requires the scaled-shot evaluation identity')
        return None
    M = request['origin']['M']
    nx, ny = request['N_X'], request['N_Y']
    if type(M) is not int or M < 0 or nx != ny or len(nx) != M + 1 or any(type(n) is not int or n < 1 for n in nx):
        raise ValueError('Scaled allocation must contain M+1 positive symmetric rounds')
    projection = request['route'].get('horizon_projection')
    if projection:
        source = projection['source_configuration_id']
        source_M = projection['source_M']
        value = json.dumps(['RPE_SCALED_SHOTS_V1', 'PREFIX', source, source_M, M], separators=(',', ':'))
        if not (type(source_M) is int and M < source_M and M == projection['M']
                and hashlib.sha256(value.encode()).hexdigest() == request['configuration_id']):
            raise ValueError('Projected Q3 identity differs from its fixed source and horizon')
    return M


@contextmanager
def horizon_inputs(projection):
    """Bind the archived whole packet before consuming its validated prefix.

    The fixed count enumerator, decoder and risk code receive only m=0..M.
    The unused source rows are presented only to the metadata validator.
    """
    if projection is None:
        yield
        return
    from unittest.mock import patch

    kernel = sys.modules['trottertracks.rpe_signal_statistics_caller']
    original = kernel.prepare
    M, source_M = projection['M'], projection['source_M']

    def prepare(config, packet, allocations, **kwargs):
        if config['spec']['M'] != source_M or config['id'] != projection['source_configuration_id']:
            raise ValueError('Original Q3 configuration/horizon differs from projection')
        if len(allocations) != 1 or [row['m'] for row in allocations[0]['shots']] != list(range(M + 1)):
            raise ValueError('Q3 projection requires one complete allocation')
        used = allocations[0]['shots']
        unused = [dict(m=m, N_X=1, N_Y=1) for m in range(M + 1, source_M + 1)]
        entries = original(config, packet, [dict(allocations[0], shots=used + unused)], **kwargs)
        if len(entries) != 1:
            raise ValueError('Expected a single validated source entry')
        entry = entries[0]
        prepared = entry['prepared']
        if prepared['M'] != source_M or [r['m'] for r in prepared['rows']] != list(range(source_M + 1)):
            raise ValueError('Source Q3 packet is not a complete consecutive horizon')
        original_input = entry['signal_input']
        consumed = dict(original_input, means=original_input['means'][:M+1]) if 'means' in original_input else original_input
        return [dict(entry, prepared=dict(prepared, M=M, rows=prepared['rows'][:M+1]),
                     allocation=dict(entry['allocation'], shots=used), signal_input=consumed)]

    with patch.object(kernel, 'prepare', prepare):
        yield


def scoped_case(view, request):
    projection = request['route'].get('horizon_projection')
    base = dict(request, configuration_id=projection['source_configuration_id']) if projection else request
    case, inputs = build(view, base)
    if projection:
        radii = inputs['options']['phase_radii']
        if len(radii) != projection['source_M'] + 1:
            raise ValueError('Source Q3 phase radii are not bound to the complete source horizon')
        case['horizon_projection'] = deepcopy(projection)
        case['source_configuration_id'] = base['configuration_id']
        case['effective_configuration_id'] = request['configuration_id']
        inputs['options']['phase_radii'] = radii[:projection['M']+1]
    return case, inputs


def build(view, request):
    if request['route']['kind'] in ('pauli_b_common_entry','prdf_d2_common_entry'):
        return common_case(view, request)
    loc, saved, base = template(view, request)
    case = {k: deepcopy(v) for k, v in base.items() if k not in DROP}
    case['label'] = request['request_id']
    case['allocation'] = dict(allocation_id='RPE_OPT_' + request['request_id'], shot_policy='USER_SUPPLIED',
                              shots=[dict(m=m, N_X=x, N_Y=y) for m, (x, y) in enumerate(zip(request['N_X'], request['N_Y']))])
    inputs = {k: deepcopy(saved[k]) for k in ('kernel_source', 'C_source', 'acceleration', 'price_policy') + CONTRACT if k in saved}
    inputs.update(cases=[case], concurrent_configurations=1, maximum_scientific_calls=1, template_source=loc)
    return case, inputs


def count_bounds(shots, frontier=8192):
    cells = [(r['N_X'] + 1) * (r['N_Y'] + 1) for r in shots]
    states, pairs = min(cells[0], frontier), cells[0]
    for c in cells[1:]:
        pairs += states * c
        states = min(frontier, states * c)
    return dict(maximum_round_outcomes=max(cells), verification_pairs_upper=sum(a * b for a, b in zip(cells, cells[1:])),
                transition_pairs_upper=pairs)


def in_scope(bounds, options):
    return (bounds['maximum_round_outcomes'] <= options['maximum_round_outcomes']
            and bounds['verification_pairs_upper'] <= options['maximum_verification_pairs']
            and bounds['transition_pairs_upper'] <= options['total_state_refinement']['maximum_transition_pairs'])


def finite_support_kernel():
    spec = importlib.util.spec_from_file_location(
        '_rpe_large_shot_bounds', Path(__file__).resolve().parents[1]/'src/trottertracks/rpe_large_shot_bounds.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite_support_requested(request, case):
    return (request.get('execution_profile', {}).get('large_shot_bounds') is True
            and request.get('origin', {}).get('kind') == 'RPE_SCALED_SHOTS_V1'
            and finite_support_kernel().supports(case['allocation']['shots']))


def finite_support_input(case, inputs, route, binding, packet_view, source):
    """Use the route's original canonical intake and fixed preparation checks."""
    bridge = route.binding
    total = bridge.Q
    risk = total.D
    packet = packet_view['signal_packet']
    options = inputs['options']
    contract = deepcopy(inputs['statistical_contract'])
    contract['source'] = source
    contract['simulation_policy']['source'] = source
    contract['summary_options']['unresolved_loss']['source'] = source
    allocation = dict(case['allocation'], source=source)
    limit = inputs['adopted_evaluation']
    model = options.get('model_loss_contract') or {}
    if (options['estimator'] != total.T.ESTIMATOR or not inputs.get('template_source')
            or model.get('adopted') is not True or not model.get('scope')
            or limit.get('estimator') != total.T.ESTIMATOR
            or limit.get('target') != 'SAVED_DF10_SOLVER_ENERGY_CONDITIONAL_PROXY'
            or limit.get('loss') != 'ALL_EVENT_SQUARED_ENERGY_ERROR'
            or limit.get('scope') != 'MODEL_MSE_FILTER_ONLY'
            or options['probability_rule'] != risk.EXACT_AFFINE):
        raise ValueError('Finite support requires the fixed all-event total-estimator MSE contract')
    threshold = Fraction('0.00159360010199040')**2
    if risk.a.exact_scalar(limit['MSE_upper']) != threshold:
        raise ValueError('Fixed H4 MSE threshold differs')
    api = bridge.C.interfaces(str(route.binding_root)).pipeline
    with horizon_inputs(case.get('horizon_projection')):
        original = packet['origin_signal_packet']
        [entry] = total.K.prepare(binding['original_configuration'], packet.get('kernel_signal_packet', original),
                                  [allocation], evaluation_target=case['evaluation_target'],
                                  statistical_contract=contract, integration=api)
    prepared = entry['prepared']
    policy = entry['statistical_contract']['simulation_policy']
    risk.S.iid_policy(policy)
    if (not prepared['input_complete'] or prepared['M'] >= 13
            or (not prepared['physical_qualification_complete'] and not contract.get('allow_conditional_unqualified'))
            or prepared['decoder'] != risk.S.DECODER
            or any(row['law_binding']['definition']['model'] != policy['target_law'] for row in prepared['rows'])):
        raise ValueError('Finite support fixed decoder, signal or measurement input differs')
    allocation = [{key: row['shots'][key] for key in ('N_X', 'N_Y')} for row in prepared['rows']]
    rows = [dict(counts, raw=row['mean']['raw'], m=row['m'])
            for row, counts in zip(prepared['rows'], allocation)]
    shots = [dict(m=row['m'], N_X=row['N_X'], N_Y=row['N_Y']) for row in rows]
    if shots != case['allocation']['shots']:
        raise ValueError('Finite support shot vector differs from the fixed intake')
    return rows, float(risk.a.exact_scalar(prepared['h'])), entry['evaluation_target']['energy_ha'], threshold


def finite_support_outcome(request, case, inputs, route, out, source, metadata, mode):
    binding, packet = metadata
    rows, h, target, threshold = finite_support_input(case, inputs, route, binding, packet, source)
    kernel = finite_support_kernel()
    numerical_input = dict(rows=rows, h=h.hex(), target=float(target).hex(), source=source,
                           fixed_contract={key: inputs[key] for key in CONTRACT},
                           template=inputs['template_source'],
                           configuration_id=case.get('effective_configuration_id', request['configuration_id']),
                           total_configuration_id=binding['total_configuration']['id'],
                           canonical_binding=binding['id'])
    fingerprint = hashlib.sha256(json.dumps(numerical_input, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if mode == 'dry-check':
        return dict(status='DRY_CHECK_PASS', metadata='PASS', evaluator=kernel.SCHEMA)
    evidence = out/case['label']/'FINITE_SUPPORT.json'
    if mode == 'execute':
        stage(out, 'SCIENTIFIC_ROUTE_START', evaluator=kernel.SCHEMA)
        report = kernel.evaluate(rows, h=h, target=target, probability_rule=inputs['options']['probability_rule'],
                                 threshold=threshold, risk=route.binding.Q.D, total=route.binding.Q)
        evidence.parent.mkdir()
        write(evidence, dict(input_sha256=fingerprint, certificate=report['certificate']))
        stage(out, 'SCIENTIFIC_ROUTE_RETURNED', evaluator=kernel.SCHEMA)
    elif not (evidence.is_file() and (out/'OUTCOME.json').is_file()):
        return dict(status='UNKNOWN', reason='NO_WORKER_ARTIFACT_TO_COLLECT', scientific_evaluation_calls=0,
                    MSE_lower=None, MSE_upper=None, identity=None)
    elif json.loads((out/'CASE_INPUTS.json').read_text()) != inputs:
        raise ValueError('Collected finite support case inputs differ')
    observed = json.loads(evidence.read_text())
    if observed['input_sha256'] != fingerprint:
        raise ValueError('Finite support evidence inputs differ')
    cert = observed['certificate']
    if kernel._decode(cert['loss_cap']) != route.binding.Q.loss_range(h, target)[0]:
        raise ValueError('Finite support evidence loss cap differs')
    report = kernel.summarize(cert, threshold, expected_rounds=len(rows))
    identity_value = dict(configuration_id=request['configuration_id'], binding_id=binding['id'],
                          total_configuration_id=binding['total_configuration']['id'], M=len(rows)-1,
                          signal_packet=case['signal_packet'], allocation=case['allocation']['shots'],
                          evaluation_target=case['evaluation_target'], contract_matches=True)
    outcome = dict(report, evaluator=kernel.SCHEMA, identity=identity_value, scientific_evaluation_calls=1,
                   evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())
    if mode == 'collect':
        original = json.loads((out/'OUTCOME.json').read_text())
        if any(original.get(key) != value for key, value in outcome.items()):
            raise ValueError('Collected finite support evidence differs from the execution receipt')
        outcome['original_outcome_sha256'] = hashlib.sha256((out/'OUTCOME.json').read_bytes()).hexdigest()
    return outcome


def load(view, name, rel):
    sys.path[:0] = [str(view / 'scripts'), str(view / 'src')]
    spec = importlib.util.spec_from_file_location(name, view / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pinned_validator(view, H, exact):
    tree = ast.parse(subprocess.check_output(['git', 'show', VALIDATOR['commit'] + ':' + VALIDATOR['path']], cwd=view))
    node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == VALIDATOR['function']]
    scope = dict(ROOT=Path('/'), json=json, exact=exact, F=NS(H=H))
    exec(compile(ast.Module(body=node, type_ignores=[]), VALIDATOR['commit'] + ':' + VALIDATOR['path'], 'exec'), scope)
    return scope[VALIDATOR['function']]


COMMON = {}  # pauli_b_common_entry: loaded route modules (module level so the supervisor can pickle the worker)


def common_metadata(case):
    """B's LD5 metadata with the handoff generalized to any common-entry MEAN_HANDOFF accepted by B's intake."""
    view, X = COMMON['view'], COMMON['X'];A, T = X.A, X.T
    accepted = A.accept_saved(view, integration=T.C.interfaces(str(view)).pipeline, handoff_source=case['signal_packet'])
    # B 33e324b declares ENTRY_SOURCES (several accepted A entries); the earlier view bd42dd1e9 declares one ENTRY_SOURCE
    if accepted.get('assembly_implementation_source', {}).get('commit') not in getattr(A, 'ENTRY_SOURCES', (A.ENTRY_SOURCE,)):
        raise ValueError('Not a common-entry Pauli assembly')
    accepted = accepted['results'][0];original = T._selected(view, case['configuration'])
    if original != accepted['configuration'] or original['id'] != case['original_configuration_id']:
        raise ValueError('Common-entry configuration SOURCE/ID differs')
    binding = T.create(view, original, original_source=case['configuration'])
    if binding['id'] != case['binding_id'] or binding['total_configuration']['id'] != case['total_configuration_id']:
        raise ValueError('Common-entry total canonical binding differs')
    return binding, T.bind_signal(view, binding, accepted['signal_packet'])


def common_worker(case, inputs, source, output):
    from unittest.mock import patch
    X = COMMON['X']
    with patch.object(X.M, 'metadata', common_metadata), patch.object(X.M, 'check_source', X.check_source):
        return X.L.SAVED_WORKER(case, inputs, source, output)


@contextmanager
def common_installed(output):
    from unittest.mock import patch
    X = COMMON['X']
    with patch.object(X.L, 'worker', common_worker), X.L.installed(output):
        yield


def load_prdf_route(view, route_views):
    """Relocate only the saved caller's inner import; keep its bytes/guards intact."""
    def constant(source, name):
        node = next(n.value for n in ast.parse(source).body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
        return node

    caller = 'scripts/run_pr_df_ld_split_q3_resume.py'
    old = ast.literal_eval(constant((view / caller).read_text(), 'OLD'))
    saved_path = 'scripts/run_pr_df_ld_split_q3.py'
    saved = subprocess.check_output(['git', 'show', old + ':' + saved_path], cwd=view).decode()
    commit = ast.literal_eval(constant(saved, 'Q3_SOURCE'))
    if commit not in route_views:
        # Existing server installations retain their original path behavior.
        return load(view, 'run_pr_df_ld_split_q3_resume', caller), None
    inner = Path(route_views[commit]).resolve()
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=inner, text=True).strip() != commit:
        raise ValueError('PR-DF inner SOURCE HEAD differs: ' + str(inner))
    for checkout in (view, inner):
        if subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'scripts', 'src'], cwd=checkout).returncode:
            raise ValueError('PR-DF fixed SOURCE has tracked changes: ' + str(checkout))
    declaration = constant(saved, 'VIEW')
    if not (isinstance(declaration, ast.Call) and isinstance(declaration.func, ast.Name)
            and declaration.func.id == 'Path' and len(declaration.args) == 1):
        raise ValueError('Unsupported PR-DF inner VIEW declaration')
    original_view = Path(ast.literal_eval(declaration.args[0]))
    entry = 'scripts/run_pr_df_fullh_total_q3_fixed.py'
    expected = subprocess.check_output(['git', 'show', commit + ':' + entry], cwd=inner)
    if (inner / entry).read_bytes() != expected:
        raise ValueError('PR-DF inner entry SOURCE bytes differ')
    original_spec = importlib.util.spec_from_file_location
    redirected = []

    def inner_spec(name, location, *args, **kwargs):
        if name == '_split_fixed_q3_route':
            if Path(location) != original_view / entry:
                raise ValueError('Unexpected PR-DF inner import location')
            redirected.append(str(location))
            location = inner / entry
        return original_spec(name, location, *args, **kwargs)

    from unittest.mock import patch
    with patch.object(importlib.util, 'spec_from_file_location', inner_spec):
        module = load(view, 'run_pr_df_ld_split_q3_resume', caller)
    if len(redirected) != 1 or module.Q.Q3_SOURCE != commit or module.U.ROOT != inner:
        raise ValueError('PR-DF inner SOURCE resolution differs')
    # runtime_check checks this binding against Q3_SOURCE. No function, AST,
    # scientific input or frozen on-disk caller is rewritten.
    module.Q.VIEW = inner
    replacements = {str(original_view / part): str(inner / part) for part in ('src', 'scripts')}
    sys.path[:] = [replacements.get(p, p) for p in sys.path]
    return module, dict(commit=commit, view=str(inner), entry=entry,
                        entry_sha256=hashlib.sha256(expected).hexdigest(),
                        saved_caller=dict(commit=old, path=saved_path), frozen_sources_modified=False)


def route(view, kind, route_views=None):
    """(metadata, installed-context, supervise, validate, bounds) of an existing route, loaded unchanged."""
    if kind == 'pf_b_pf6':
        F = load(view, '_rpe_opt_pf6', 'scripts/run_pf6_cost_budget_q3_fixed.py');F.check_source()
        return NS(metadata=F.metadata, installed=F.installed_workers, supervise=F.M.supervise,
                  validate=pinned_validator(view, F.H, F.T.B.a.exact_scalar), bounds=F.count_bounds, runtime=None,
                  binding=F.T, binding_root=F.ROOT)
    if kind in ('pauli_b_ld3', 'pauli_b_ld5'):
        if kind == 'pauli_b_ld3':
            C = load(view, '_rpe_opt_ld3', 'scripts/run_pauli_ld3_pf_budget_q3_fixed.py');C.check_source()
            L, meta, inst = C.L, C.L.metadata, C.L.installed
        else:
            X = load(view, '_rpe_opt_ld5', 'scripts/run_pauli_ld5_two_vectors_q3_fixed.py');X.check_source()
            L, meta, inst = X.L, X.metadata, X.installed
        return NS(metadata=meta, installed=inst, supervise=L.M.supervise,
                  validate=pinned_validator(view, L.H, L.T.Q.a.exact_scalar), bounds=lambda s: L.H.finite_bounds([r['N_X'] for r in s]),
                  runtime=None, binding=L.T, binding_root=L.ROOT)
    if kind == 'pauli_b_common_entry':
        X = load(view, '_rpe_opt_common', 'scripts/run_pauli_ld5_two_vectors_q3_fixed.py');X.check_source()
        COMMON.update(view=view, X=X)
        return NS(metadata=common_metadata, installed=common_installed, supervise=X.L.M.supervise,
                  evidence_writer=X.R, validate=pinned_validator(view, X.L.H, X.L.T.Q.a.exact_scalar),
                  bounds=lambda s: X.L.H.finite_bounds([r['N_X'] for r in s]), runtime=None,
                  binding=X.T, binding_root=view)
    if kind in ('prdf_d2_split','prdf_d2_common_entry'):
        W, resolution = load_prdf_route(view, route_views or {})
        return NS(metadata=W.U.metadata, installed=lambda out: nullcontext(), supervise=W.supervise,
                  evidence_writer=W.U.R if kind=='prdf_d2_common_entry' else None, validate=pinned_validator(view, W.U.H, W.U.T.Q.a.exact_scalar), bounds=lambda s: W.Q.bounds(dict(allocation=dict(shots=s))),
                  runtime=W.runtime_check, source_resolution=resolution, binding=W.U.T, binding_root=W.U.ROOT)
    raise ValueError('Unknown route kind: ' + kind)


def identity(result, case, inputs):
    record = result.get('record') or {}
    def strip(value, src):
        if isinstance(value, dict):
            return {k: strip(v, src) for k, v in value.items() if not (k == 'source' and v == src)}
        if isinstance(value, list):
            return [strip(v, src) for v in value]
        return value
    src = record.get('source')
    projection = case.get('horizon_projection')
    cid = case.get('configuration_id') or case.get('original_configuration_id')
    if projection:
        diagnostic = record['diagnostic']['deterministic_risk']
        rounds = diagnostic['bounds']['total_state_dp']['rounds']
        if [r['m'] for r in rounds] != list(range(projection['M'] + 1)) or cid != projection['source_configuration_id']:
            raise ValueError('Returned Q3 terminal round/identity differs')
        cid = case['effective_configuration_id']
    return dict(configuration_id=cid,
                **({'M': projection['M'], 'source_configuration_id': projection['source_configuration_id']} if projection else {}),
                total_configuration_id=record.get('configuration_id'), binding_id=record.get('canonical_binding'),
                signal_packet=case.get('signal_packet'), allocation=record.get('allocation', {}).get('shots'),
                evaluation_target=record.get('evaluation_target'),
                contract_matches=bool(record) and strip(record.get('statistical_contract'), src) == inputs.get('statistical_contract')
                and strip(record.get('deterministic_input', {}).get('options'), src) == inputs.get('options'))


def process_exit_evidence(request, supervision, out):
    """Read the existing route's real exit receipt, without changing its gate.

    DF's fixed supervisor returns joined pool processes directly. Other routes
    emit PROCESS_EXITS files from their installed context. Neither is inferred
    from a result status or from process disappearance.
    """
    if request['route']['kind'] in ('prdf_d2_split','prdf_d2_common_entry'):
        exits=supervision.get('process_exitcodes') or []
        expected=len(supervision.get('receipts') or [])+len(supervision.get('errors') or [])
        gate=bool(exits) and len(exits)==expected and all(
            type(c.get('exitcode')) is int and c['exitcode']==0 and c.get('signal') is None for c in exits)
        return dict(schema='DF_FIXED_SUPERVISOR_JOINED_PROCESSES',recorded_children=len(exits),
                    expected_cases=expected,all_exit_codes_zero=gate)
    exits=[c for p in out.glob('PROCESS_EXITS_*.jsonl') for line in p.read_text().splitlines()
           for c in json.loads(line)['children']]
    return dict(schema='INSTALLED_ROUTE_PROCESS_EXITS',recorded_children=len(exits),
                all_exit_codes_zero=bool(exits) and all(c['exitcode']==0 for c in exits))


def outcome_from(view, request, case, inputs, r, out, source, supervision=None):
    folder = out / case['label']
    try:
        if supervision is not None:
            gate = process_exit_evidence(request, supervision, out)
            destination='COLLECT_SUPERVISION.json' if request.get('_collect') else 'SUPERVISION.json'
            evidence=dict(supervision, process_exit_gate=gate)
            if (out/destination).exists() and destination=='COLLECT_SUPERVISION.json':
                if json.loads((out/destination).read_text())!=evidence:raise ValueError('Collection process evidence changed')
            else:write(out / destination,evidence)
            bad = not gate['all_exit_codes_zero'] or supervision.get('errors') or supervision.get('monitor_errors')
        else:
            bad = True  # An artifact alone cannot prove that its worker exited cleanly.
        result = json.loads((folder / 'RESULT.json').read_text())
        if result.get('status') == 'UNKNOWN':
            raise ValueError('Worker reported UNKNOWN: ' + str(result.get('reason')))
        artifacts = EVIDENCE.ValidatorDirectory(folder)
        value = r.validate(case, inputs, artifacts if artifacts.payloads else folder, source)
        status = 'UNKNOWN' if bad else value['status']
        return dict(status=status, reason='SUPERVISOR_OR_PROCESS_GATE' if bad else None,
                    MSE_lower=value['MSE_lower'], MSE_upper=value['MSE_upper'],
                    scientific_evaluation_calls=value['scientific_evaluation_calls'],
                    closure=value.get('closure'), identity=identity(result, case, inputs))
    except Exception as exc:
        stages = folder / 'STAGES.jsonl'
        calls = sum(json.loads(l)['stage'] == 'SCIENTIFIC_CALL_START' for l in stages.read_text().splitlines()) if stages.exists() else 0
        return dict(status='UNKNOWN', reason=type(exc).__name__ + ': ' + str(exc), MSE_lower=None, MSE_upper=None,
                    scientific_evaluation_calls=calls, identity=None)


def artificial(request, out, mode):
    """Labelled artificial worker for executor tests: deterministic MSE from the request's spec.

    execute writes the worker artifact ARTIFICIAL_RESULT.json first (like a real worker's RESULT.json);
    collect only re-validates that artifact and never recomputes.
    """
    spec = request['route']['artificial']
    if mode == 'dry-check':
        return dict(status='DRY_CHECK_PASS', artificial=True)
    shots = [dict(m=m, N_X=x, N_Y=y) for m, (x, y) in enumerate(zip(request['N_X'], request['N_Y']))]
    ident = dict(request['expected'], allocation=shots, contract_matches=True)
    artifact = out / 'ARTIFICIAL_RESULT.json'
    if mode == 'collect':
        if not artifact.exists():
            return dict(status='UNKNOWN', reason='NO_WORKER_ARTIFACT_TO_COLLECT', MSE_lower=None, MSE_upper=None,
                        scientific_evaluation_calls=0, identity=None, artificial=True)
        saved = json.loads(artifact.read_text())
        return dict(saved, identity=ident, artificial=True, collected=True)
    resource_start=resource.getrusage(resource.RUSAGE_SELF);child_start=resource.getrusage(resource.RUSAGE_CHILDREN)
    wall_start=time.perf_counter()
    stage(out, 'ARTIFICIAL_EVALUATION_START')
    artificial_load(spec.get('load'), request.get('execution_profile', {}).get('dp_workers'))
    time.sleep(spec.get('sleep_seconds', 0))
    time.sleep(spec.get('sleep_by_first_shot', {}).get(str(request['N_X'][0]), 0))
    if spec.get('force_unknown'):
        value = dict(status='UNKNOWN', reason='ARTIFICIAL_GATE_FAILURE', MSE_lower=None, MSE_upper=None, scientific_evaluation_calls=1)
    elif spec.get('finite_support_fixture'):
        fixture = spec['finite_support_fixture']
        if request.get('execution_profile', {}).get('large_shot_bounds') is not True:
            raise ValueError('Artificial finite support must be enabled by the executor')
        fixed = Path(fixture['view']).resolve()
        if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=fixed,text=True).strip() != fixture['commit']
                or subprocess.run(['git','diff','--quiet','HEAD','--','src','scripts'],cwd=fixed).returncode):
            raise ValueError('Artificial decoder fixture SOURCE differs')
        sys.path.insert(0, str(fixed/'src'))
        from trottertracks import rpe_total_risk as total
        kernel = finite_support_kernel()
        rows = [dict(row, raw=dict(real=1.,imag=0.)) for row in shots]
        report = kernel.evaluate(rows, h=0.1, target=float(fixture['target']), threshold=Fraction(spec['tau']),
                                 probability_rule=total.D.EXACT_AFFINE, risk=total.D, total=total)
        value = dict(report, evaluator=kernel.SCHEMA, scientific_evaluation_calls=0)
    else:
        w = [Fraction(v) for v in spec['weights']];scale = Fraction(spec['scale'])
        mse = scale * sum(wm * (Fraction(1, x) + Fraction(1, y)) / 2 for wm, x, y in zip(w, request['N_X'], request['N_Y']))
        tau = Fraction(spec['tau']);enc = dict(numerator=mse.numerator, denominator=mse.denominator)
        value = dict(status=PASS if mse <= tau else FAIL, MSE_lower=enc, MSE_upper=enc, scientific_evaluation_calls=1)
    write(artifact, value)
    usage=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
    write(out / 'ARTIFICIAL_RESOURCE.json', dict(artificial=True,
        wall_seconds=time.perf_counter()-wall_start,
        process_tree_CPU_seconds=(usage.ru_utime+usage.ru_stime-resource_start.ru_utime-resource_start.ru_stime
                                  +children.ru_utime+children.ru_stime-child_start.ru_utime-child_start.ru_stime),
        peak_self_RSS_KiB=usage.ru_maxrss, peak_child_RSS_KiB=children.ru_maxrss))
    stage(out, 'ARTIFICIAL_EVALUATION_RETURNED')
    return dict(value, identity=ident, artificial=True)


def _artificial_chunk(units):
    value=0
    for i in range(units):value=(value+i*i)%1000000007
    return value


def artificial_load(spec,workers):
    """Explicit artificial fork/CPU workload for execution-profile benchmarks.

    Does not use quantum inputs or kernels. It does not affect the artificial
    answer. Its scaling is not a model or measurement of real Q3 speedup.
    """
    if not spec:return
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    workers=workers or spec.get('baseline_dp_workers',8)
    memory=bytearray(spec.get('memory_MiB',0)*1024*1024)
    for i in range(0,len(memory),4096):memory[i]=1
    time.sleep(spec.get('serial_seconds',0))
    for _ in range(spec.get('parallel_rounds',1)):
        units=[spec.get('units_per_chunk',10000)]*spec.get('chunks',8)
        if workers==1:list(map(_artificial_chunk,units))
        else:
            with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork'))as pool:list(pool.map(_artificial_chunk,units))


def execution_profile(inputs, request):
    """Change only the existing DP worker parameter, never numerical contracts."""
    workers=request.get('execution_profile',{}).get('dp_workers')
    if workers is None:return inputs
    if type(workers)is not int or workers<1:raise ValueError('Positive integer dp_workers required')
    frozen={k:deepcopy(inputs[k])for k in CONTRACT}
    if 'workers_per_configuration' not in inputs.get('acceleration',{}):
        raise ValueError('Route has no existing workers_per_configuration parameter')
    inputs=deepcopy(inputs);inputs['acceleration']['workers_per_configuration']=workers
    if any(inputs[k]!=frozen[k]for k in CONTRACT):raise ValueError('Execution profile changed scientific contract')
    return inputs


@contextmanager
def prefix_cache_context(request,case,inputs,out):
    """Opt-in cache in the real existing worker path, inherited by its fork."""
    profile=request.get('execution_profile',{});directory=profile.get('prefix_cache_directory')
    parallel_residual=profile.get('parallel_dp_residual') is True
    if directory is None and not parallel_residual:
        yield;return
    module=sys.modules.get('trottertracks.rpe_total_dp_parallel')
    if module is None:raise ValueError('Route has no supported fixed total-state accelerator')
    helper=Path(__file__).resolve().parents[1]/'src/trottertracks/rpe_dp_prefix_reuse.py'
    spec=importlib.util.spec_from_file_location('_rpe_opt_prefix_reuse',helper)
    C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)  # keep pinned route import paths unchanged
    namespace=dict(method=request['method'],configuration_id=request['configuration_id'],expected=request['expected'],
        route_commit=request['route']['view_commit'],
        physical={k:case.get(k)for k in ('configuration','signal_packet','evaluation_target','binding_id','total_configuration_id')},
        contract={k:inputs[k]for k in CONTRACT},adapter_prefix_version=C.SCHEMA)
    def event(value):
        with (Path(out)/'DP_PREFIX_EVENTS.jsonl').open('a')as f:
            f.write(json.dumps(dict(UTC=utc(),pid=os.getpid(),**value))+'\n')
    checkpoints = profile.get('prefix_checkpoint_rounds') if directory is not None else []
    selective = profile.get('prefix_selective_checkpoint_rounds',()) if directory is not None else ()
    M = horizon_of(request)
    if M is not None and checkpoints is not None:
        checkpoints = [m for m in checkpoints if m < M]
        selective = [m for m in selective if m < M]
    with C.installed(module,directory=directory,cache_namespace=namespace,
                     checkpoint_rounds=checkpoints,
                     event_sink=event,max_bytes=profile.get('prefix_cache_max_bytes', C.DEFAULT_CACHE_MAX_BYTES),
                     parallel_residual=parallel_residual,profile_residual=profile.get('profile_dp_residual') is True,
                     selective_rounds=selective,
                     selective_window_seconds=profile.get('prefix_selective_window_seconds',C.DEFAULT_SELECTIVE_WINDOW_SECONDS)):
        yield



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('execute', 'collect', 'dry-check', 'export-evidence'), default='execute')
    p.add_argument('--evidence-directory', type=Path, help='Destination for explicitly exporting original sidecars')
    p.add_argument('--route-views', type=Path, help='Optional host commit-to-checkout JSON; original REQUEST stays unchanged')
    a = p.parse_args()
    request = json.loads(a.request.read_text());out = a.output.resolve()
    if a.mode == 'export-evidence':
        if a.evidence_directory is None:
            p.error('--evidence-directory required for export-evidence')
        label = request['request_id']
        if not label or Path(label).name != label or label in ('.','..'):
            p.error('Invalid evidence case label')
        EVIDENCE.export(out / label, a.evidence_directory)
        return 0
    out.mkdir(parents=True, exist_ok=True)
    kind = request['route']['kind'];stage(out, 'JOB_START', mode=a.mode, kind=kind, pid=os.getpid())
    name = 'OUTCOME.json' if a.mode != 'dry-check' else 'DRY_CHECK.json'
    M = horizon_of(request)
    if kind == 'artificial':
        outcome = artificial(request, out, a.mode)
        if M is not None and outcome.get('identity'):
            outcome['identity']['M'] = M
    elif M is not None and M > 12:
        outcome = dict(status='UNKNOWN', reason='OUTSIDE_VALIDATED_R3_SCALAR_EQUIVALENCE_DOMAIN',
                       requested_M=M, MSE_lower=None, MSE_upper=None, scientific_evaluation_calls=0, identity=None)
    else:
        view = Path(request['route']['view'])
        route_views = json.loads(a.route_views.read_text()) if a.route_views else request['route'].get('route_views', {})
        case, inputs = scoped_case(view, request);inputs = execution_profile(inputs, request);r = route(view, kind, route_views)
        stage(out, 'CASE_BUILT', template=inputs['template_source'])
        bounds = r.bounds(case['allocation']['shots'])
        old_scope = in_scope(bounds, inputs['options'])
        finite_support = not old_scope and finite_support_requested(request, case)
        if not (old_scope or finite_support):
            outcome = dict(status='UNKNOWN', reason='COUNT_ONLY_SCOPE_EXCEEDED', bounds=bounds, MSE_lower=None, MSE_upper=None,
                           scientific_evaluation_calls=0, identity=None)
        else:
            runtime = r.runtime() if r.runtime is not None else None
            metadata = r.metadata(case);stage(out, 'METADATA_PASS', bounds=bounds)
            source = dict(request_id=request['request_id'], route=kind, view_commit=request['route']['view_commit'],
                          adapter='scripts/run_rpe_optimizer_q3_job.py', optimizer_source=request.get('optimizer_source'), runtime=runtime, execution_profile=request.get('execution_profile'))
            if finite_support:
                if a.mode == 'execute':
                    write(out/'SOURCE.json', source)
                    write(out/'CASE_INPUTS.json', inputs)
                elif a.mode == 'collect' and (out/'SOURCE.json').is_file():
                    source = json.loads((out/'SOURCE.json').read_text())
                outcome = finite_support_outcome(request, case, inputs, r, out, source, metadata, a.mode)
                if a.mode == 'collect' and 'original_outcome_sha256' in outcome:
                    name = 'COLLECTED_OUTCOME.json'
                if a.mode == 'dry-check':
                    outcome.update(bounds=bounds, runtime=runtime)
                    if getattr(r, 'source_resolution', None) is not None:
                        outcome['source_resolution'] = r.source_resolution
            elif a.mode == 'dry-check':
                outcome = dict(status='DRY_CHECK_PASS', bounds=bounds, case_label=case['label'], template=inputs['template_source'],
                               metadata='PASS', runtime=runtime)
                profile = request.get('execution_profile', {})
                if profile.get('prefix_selective_checkpoint_rounds') or profile.get('profile_dp_residual') is True:
                    # Check the accelerator and propagation guards in this
                    # route's actual runtime without running its evaluation.
                    with prefix_cache_context(request, case, inputs, out):
                        outcome['prefix_runtime_check'] = 'PASS'
                    if profile.get('compact_q3_evidence') is True and getattr(r, 'evidence_writer', None) is not None:
                        with EVIDENCE.capture(r.evidence_writer, out / case['label']):
                            outcome['compact_runtime_check'] = 'PASS'
                if getattr(r, 'source_resolution', None) is not None:
                    outcome['source_resolution'] = r.source_resolution
            elif a.mode == 'collect':
                if not (out / case['label'] / 'RESULT.json').exists() or not (out / 'SOURCE.json').exists():
                    outcome = dict(status='UNKNOWN', reason='NO_WORKER_ARTIFACT_TO_COLLECT', MSE_lower=None, MSE_upper=None,
                                   scientific_evaluation_calls=0, identity=None)
                else:
                    saved_supervision=json.loads((out/'SUPERVISION.json').read_text()) if (out/'SUPERVISION.json').exists() else None
                    outcome = outcome_from(view, dict(request,_collect=True), case, inputs, r, out,
                        json.loads((out / 'SOURCE.json').read_text()), saved_supervision)
                    if (out/'OUTCOME.json').exists():
                        outcome['original_outcome_sha256']=hashlib.sha256((out/'OUTCOME.json').read_bytes()).hexdigest()
                        name='COLLECTED_OUTCOME.json'
                    outcome['collection_source']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip()
            else:
                write(out / 'SOURCE.json', source);write(out / 'CASE_INPUTS.json', inputs);(out / case['label']).mkdir()
                stage(out, 'SCIENTIFIC_ROUTE_START')
                writer = getattr(r, 'evidence_writer', None)
                storage = (EVIDENCE.capture(writer, out / case['label'])
                           if writer is not None and request.get('execution_profile', {}).get('compact_q3_evidence') is True
                           else nullcontext())
                with horizon_inputs(case.get('horizon_projection')), prefix_cache_context(request, case, inputs, out), TIMING.body_context(
                        sys.modules.get('trottertracks.rpe_total_dp_parallel'),out), r.installed(out), storage:
                    supervision = r.supervise([case], inputs, source, out)
                stage(out, 'SCIENTIFIC_ROUTE_RETURNED')
                outcome = outcome_from(view, request, case, inputs, r, out, source, supervision)
    outcome.update(request_id=request['request_id'], route=kind, mode=a.mode, finished_UTC=utc())
    if a.mode == 'collect' and outcome.get('reason') == 'NO_WORKER_ARTIFACT_TO_COLLECT':
        # Nothing to validate: leave no OUTCOME so the executor reports the job for review (no automatic retry).
        (out / 'COLLECT_ATTEMPT.json').write_text(json.dumps(outcome) + '\n');stage(out, 'JOB_END', status='NO_ARTIFACT')
        return 3
    write(out / name, outcome)
    if kind != 'artificial' and a.mode == 'execute' and outcome['status'] in EVIDENCE.COMPLETE:
        try:
            if EVIDENCE.compact(out / case['label'], out / name):
                stage(out, 'Q3_SIDECARS_COMPACTED')
        except (OSError, ImportError, ValueError, KeyError) as exc:
            stage(out, 'Q3_CHECKPOINTS_RETAINED', reason=str(exc))
    stage(out, 'JOB_END', status=outcome['status'])
    print(outcome['status'], flush=True)
    return 0 if outcome['status'] != 'UNKNOWN' else 2


if __name__ == '__main__':
    raise SystemExit(main())
