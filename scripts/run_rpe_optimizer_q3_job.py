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
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace as NS

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
        f.write(json.dumps(dict(stage=name, UTC=utc(), **extra)) + '\n')


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
    if accepted.get('assembly_implementation_source', {}).get('commit') not in A.ENTRY_SOURCES:
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


def route(view, kind):
    """(metadata, installed-context, supervise, validate, bounds) of an existing route, loaded unchanged."""
    if kind == 'pf_b_pf6':
        F = load(view, '_rpe_opt_pf6', 'scripts/run_pf6_cost_budget_q3_fixed.py');F.check_source()
        return NS(metadata=F.metadata, installed=F.installed_workers, supervise=F.M.supervise,
                  validate=pinned_validator(view, F.H, F.T.B.a.exact_scalar), bounds=F.count_bounds, runtime=None)
    if kind in ('pauli_b_ld3', 'pauli_b_ld5'):
        if kind == 'pauli_b_ld3':
            C = load(view, '_rpe_opt_ld3', 'scripts/run_pauli_ld3_pf_budget_q3_fixed.py');C.check_source()
            L, meta, inst = C.L, C.L.metadata, C.L.installed
        else:
            X = load(view, '_rpe_opt_ld5', 'scripts/run_pauli_ld5_two_vectors_q3_fixed.py');X.check_source()
            L, meta, inst = X.L, X.metadata, X.installed
        return NS(metadata=meta, installed=inst, supervise=L.M.supervise,
                  validate=pinned_validator(view, L.H, L.T.Q.a.exact_scalar), bounds=lambda s: L.H.finite_bounds([r['N_X'] for r in s]),
                  runtime=None)
    if kind == 'pauli_b_common_entry':
        X = load(view, '_rpe_opt_common', 'scripts/run_pauli_ld5_two_vectors_q3_fixed.py');X.check_source()
        COMMON.update(view=view, X=X)
        return NS(metadata=common_metadata, installed=common_installed, supervise=X.L.M.supervise,
                  validate=pinned_validator(view, X.L.H, X.L.T.Q.a.exact_scalar),
                  bounds=lambda s: X.L.H.finite_bounds([r['N_X'] for r in s]), runtime=None)
    if kind in ('prdf_d2_split','prdf_d2_common_entry'):
        W = load(view, 'run_pr_df_ld_split_q3_resume', 'scripts/run_pr_df_ld_split_q3_resume.py')
        return NS(metadata=W.U.metadata, installed=lambda out: nullcontext(), supervise=W.supervise,
                  validate=pinned_validator(view, W.U.H, W.U.T.Q.a.exact_scalar), bounds=lambda s: W.Q.bounds(dict(allocation=dict(shots=s))),
                  runtime=W.runtime_check)
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
    return dict(configuration_id=case.get('configuration_id') or case.get('original_configuration_id'),
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
            write(out / destination, dict(supervision, process_exit_gate=gate))
            bad = not gate['all_exit_codes_zero'] or supervision.get('errors') or supervision.get('monitor_errors')
        else:
            bad = True  # An artifact alone cannot prove that its worker exited cleanly.
        result = json.loads((folder / 'RESULT.json').read_text())
        if result.get('status') == 'UNKNOWN':
            raise ValueError('Worker reported UNKNOWN: ' + str(result.get('reason')))
        value = r.validate(case, inputs, folder, source)
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
    time.sleep(spec.get('sleep_seconds', 0))
    if spec.get('force_unknown'):
        value = dict(status='UNKNOWN', reason='ARTIFICIAL_GATE_FAILURE', MSE_lower=None, MSE_upper=None, scientific_evaluation_calls=1)
    else:
        w = [Fraction(v) for v in spec['weights']];scale = Fraction(spec['scale'])
        mse = scale * sum(wm * (Fraction(1, x) + Fraction(1, y)) / 2 for wm, x, y in zip(w, request['N_X'], request['N_Y']))
        tau = Fraction(spec['tau']);enc = dict(numerator=mse.numerator, denominator=mse.denominator)
        value = dict(status=PASS if mse <= tau else FAIL, MSE_lower=enc, MSE_upper=enc, scientific_evaluation_calls=1)
    write(artifact, value)
    return dict(value, identity=ident, artificial=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('execute', 'collect', 'dry-check'), default='execute')
    a = p.parse_args()
    request = json.loads(a.request.read_text());out = a.output.resolve();out.mkdir(parents=True, exist_ok=True)
    kind = request['route']['kind'];stage(out, 'JOB_START', mode=a.mode, kind=kind, pid=os.getpid())
    name = 'OUTCOME.json' if a.mode != 'dry-check' else 'DRY_CHECK.json'
    if kind == 'artificial':
        outcome = artificial(request, out, a.mode)
    else:
        view = Path(request['route']['view'])
        case, inputs = build(view, request);r = route(view, kind)
        stage(out, 'CASE_BUILT', template=inputs['template_source'])
        bounds = r.bounds(case['allocation']['shots'])
        if not in_scope(bounds, inputs['options']):
            outcome = dict(status='UNKNOWN', reason='COUNT_ONLY_SCOPE_EXCEEDED', bounds=bounds, MSE_lower=None, MSE_upper=None,
                           scientific_evaluation_calls=0, identity=None)
        else:
            runtime = r.runtime() if r.runtime is not None else None
            r.metadata(case);stage(out, 'METADATA_PASS', bounds=bounds)
            source = dict(request_id=request['request_id'], route=kind, view_commit=request['route']['view_commit'],
                          adapter='scripts/run_rpe_optimizer_q3_job.py', optimizer_source=request.get('optimizer_source'), runtime=runtime)
            if a.mode == 'dry-check':
                outcome = dict(status='DRY_CHECK_PASS', bounds=bounds, case_label=case['label'], template=inputs['template_source'],
                               metadata='PASS', runtime=runtime)
            elif a.mode == 'collect':
                if not (out / case['label'] / 'RESULT.json').exists() or not (out / 'SOURCE.json').exists():
                    outcome = dict(status='UNKNOWN', reason='NO_WORKER_ARTIFACT_TO_COLLECT', MSE_lower=None, MSE_upper=None,
                                   scientific_evaluation_calls=0, identity=None)
                else:
                    saved_supervision=json.loads((out/'SUPERVISION.json').read_text()) if (out/'SUPERVISION.json').exists() else None
                    outcome = outcome_from(view, dict(request,_collect=True), case, inputs, r, out,
                        json.loads((out / 'SOURCE.json').read_text()), saved_supervision)
                    outcome['collection_source']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip()
            else:
                write(out / 'SOURCE.json', source);write(out / 'CASE_INPUTS.json', inputs);(out / case['label']).mkdir()
                stage(out, 'SCIENTIFIC_ROUTE_START')
                with r.installed(out):
                    supervision = r.supervise([case], inputs, source, out)
                stage(out, 'SCIENTIFIC_ROUTE_RETURNED')
                outcome = outcome_from(view, request, case, inputs, r, out, source, supervision)
    outcome.update(request_id=request['request_id'], route=kind, mode=a.mode, finished_UTC=utc())
    if a.mode == 'collect' and outcome.get('reason') == 'NO_WORKER_ARTIFACT_TO_COLLECT':
        # Nothing to validate: leave no OUTCOME so the executor reports the job for review (no automatic retry).
        (out / 'COLLECT_ATTEMPT.json').write_text(json.dumps(outcome) + '\n');stage(out, 'JOB_END', status='NO_ARTIFACT')
        return 3
    write(out / name, outcome);stage(out, 'JOB_END', status=outcome['status'])
    print(outcome['status'], flush=True)
    return 0 if outcome['status'] != 'UNKNOWN' else 2


if __name__ == '__main__':
    raise SystemExit(main())
