"""Baseline-only signal requests and B V2 means; no search/N/price optimizer.

Imports and prepare() are metadata-only. execute() is a scientific entry for a
separately permitted finite plan. No approval-proof or grant mechanism exists.
"""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

from . import pf_pr_rpe_pipeline as p

B_SOURCE = '437ba4acdb82708b6fc98e3cd119a2a734f7cf64'
D_SOURCE = 'e7dca5f7301c8633b17a59b6a7161331303aef2d'
FORMULAS = ('2nd', '4th', '4th(m5_best)')
MODULE = 'src/trottertracks/baseline_signal_pipeline.py'
NUMERICAL = 'src/trottertracks/baseline_signal_numerics.py'
NUMERIC_POLICY = 'SOURCE_GATES_1E9_REFERENCE_PAIR_1E10_NO_REPAIR'


def blob(root, commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=root)


@lru_cache(maxsize=None)
def published_api(root):
    """Reuse fixed B/D modules in a private namespace; don't replace B/A imports.

    This small loader is needed only in C's older checkout. In B's integrated
    environment callers may pass its controller/physical API directly instead.
    """
    from . import c1_stage3_accuracy
    name = 'trottertracks._baseline_v2_' + hashlib.sha256(str(root).encode()).hexdigest()[:12]
    package = types.ModuleType(name); package.__path__ = []
    sys.modules[name] = package
    sys.modules[name + '.c1_stage3_accuracy'] = c1_stage3_accuracy
    for short, commit in [('h4_df10_rpe_connection', B_SOURCE),
                          ('pf_pr_rpe_pipeline', B_SOURCE), ('pr_rpe_rmse_controller', B_SOURCE),
                          ('full_baseline_v2_requests', D_SOURCE)]:
        path = 'src/trottertracks/' + short + '.py'
        mod = types.ModuleType(name + '.' + short); mod.__package__ = name; mod.__file__ = str(Path(root) / path)
        sys.modules[mod.__name__] = mod
        exec(compile(blob(root, commit, path), commit + ':' + path, 'exec'), mod.__dict__)
        setattr(package, short, mod)
    return types.SimpleNamespace(B=package.pr_rpe_rmse_controller, D=package.full_baseline_v2_requests)


def configure(root, spec, *, layouts=None, api=None):
    """Existing B.generate, including explicit Full-H grouping. No new IDs rule."""
    api = api or published_api(str(root))
    mapper = None if layouts is None else api.D.layout_mapper(layouts)
    return api.B.generate(spec, event_mapper=mapper)


def prepare(root, config, *, target, references=(), api=None):
    """All canonical rounds -> physical signal requests, not a fixed cell menu.

    references contain binding/T/source descriptors, not an automatic solver.
    Missing references remain explicit. PR is delegated to A's own provider.
    """
    api = api or published_api(str(root)); B, D = api.B, api.D
    B._validate(config)
    s = config['spec']; method = B.comparison_method(s)
    if method not in ('FULL_PF', 'TRUNCATED_PF', 'FULL_H_RTE'):
        raise ValueError('PR signal belongs to the existing A provider')
    if s['binding'].get('system') != 'H4' or s['binding'].get('rank') != 10:
        raise ValueError('Initial adapter supports H4 DF10; other systems are not evaluated')
    if (target.get('quantity') != 'ORIGINAL_DF_H_TARGET_ENERGY' or not target.get('source')
            or not p.same(target.get('binding'), s['binding'])
            or B.a.exact_scalar(target.get('energy_shift_ha')) != 0):
        raise ValueError('Explicit original full DF-H target/binding/zero energy shift required')
    if method != 'FULL_H_RTE' and s['formula'] not in FORMULAS:
        raise ValueError('Supported S2/Yoshida4/M5 only; no eighth-order PF')
    expected_law = 'ideal_finite_RTE_law' if method == 'FULL_H_RTE' else 'deterministic'
    if s['law'].get('model') != expected_law:
        raise ValueError('Numerical kernel does not certify a different sampler law')
    refs = {}
    for ref in references:
        m = ref['m']
        if type(m) is not int or m not in range(s['M'] + 1) or m in refs or not ref.get('source'):
            raise ValueError('Unique sourced reference descriptor per round required')
        if not p.same(ref['binding'], s['binding']) or B.a.exact_scalar(ref['T']) != B.a.exact_scalar(config['rounds'][m]['T']):
            raise ValueError('Reference H/state/time differs')
        refs[m] = deepcopy(ref)
    physical = None
    if method != 'TRUNCATED_PF':
        layouts = {r['m']: {k: r['mapping']['physical_layout'][k] for k in ('q', 'R_outer')}
                   for r in config['rounds'] if r['mapping']} if method == 'FULL_H_RTE' else None
        if method == 'FULL_H_RTE' and len(layouts) != s['M'] + 1:
            raise ValueError('Every Full-H round needs an explicit legal q/R grouping')
        physical = D.physical_schedule(s, layout_by_round=layouts)
        D._check_against_b(physical, config)
    requests = []
    for r in config['rounds']:
        m = r['m']; T = float(B.a.exact_scalar(r['T']))
        if physical is not None:
            c = deepcopy(physical['rounds'][m]['physical_candidate'])
        else:
            mode = s['split'].get('original_PF_method')
            retained = s['split']['retained_hamiltonian']
            if (mode not in ('B0', 'B0I') or retained['construction'] != mode
                    or not 0 <= s['L_D'] < s['binding']['rank']):
                raise ValueError('Truncated PF needs explicit B0/B0I scalar construction and legal prefix')
            c = dict(system='H4', method=mode, prefix=s['L_D'], formula=s['formula'],
                     T=T, q=r['q_PF'], delta=T / r['q_PF'], R_outer=None, K=None)
            c['id'] = B.p.candidate_id(c)
        structure = deepcopy(r['structure'])
        if method == 'FULL_H_RTE':
            layout = r['mapping']['physical_layout']
            structure.update(allocation=deepcopy(layout['allocation']), deterministic_blocks=0)
        domain = dict(binding=deepcopy(s['binding']), profile=deepcopy(s['profile']),
                      physical_candidate=c, structure=structure, law=deepcopy(s['law']),
                      split=deepcopy(s['split']),
                      preparation=deepcopy(s['split'].get('preparation')), target=deepcopy(target),
                      numeric_policy=NUMERIC_POLICY, reference=deepcopy(refs.get(m)))
        requests.append(dict(m=m, T=deepcopy(r['T']), q_RPE=r['q_RPE'], q_PF=r['q_PF'],
            identity=deepcopy(r['identity']), law_binding=deepcopy(r['law_binding']),
            physical_candidate=c, structure=structure, comparison_method=method, domain=domain,
            reference_status='SUPPLIED' if m in refs else 'MISSING_REFERENCE',
            mean_status='NOT_EVALUATED', source=dict(B=B_SOURCE, physical=D_SOURCE)))
    return dict(configuration=deepcopy(config), target=deepcopy(target), requests=requests,
                science_calls=0, N=None, G=None, scientific_menu_adopted=False)


def unavailable(request, reason):
    return dict(m=request['m'], identity=deepcopy(request['identity']), law_binding=deepcopy(request['law_binding']),
                source=dict(module=MODULE), target=deepcopy(request['domain']['target']), raw=None, corrected=None,
                normalization=None, bias=None, uncertainty=None, qualification='UNQUALIFIED', status=reason)


def b_mean(request, record):
    """Copy a matching available signal into B's existing mean shape; no N calls."""
    if not p.same(record['domain'], request['domain']) or not record.get('source'):
        raise ValueError('Signal physical/source/law/reference domain mismatch')
    if record['status'] != 'NUMERICAL_MEAN_AVAILABLE':
        return unavailable(request, record['status'])
    import math
    sig = record['signal']
    if record.get('numerical', {}).get('gates_passed') is not True:
        raise ValueError('Available saved signal lacks compatible numerical gates')
    numbers = [sig[k][a] for k in ('raw', 'corrected', 'reference') for a in ('real', 'imag')]
    numbers += [sig['log_B'], sig['B'], sig['attenuation']]
    if (not all(isinstance(v, (int, float)) and math.isfinite(v) for v in numbers)
            or max(abs(sig['raw'][a]) for a in ('real', 'imag')) > 1
            or sig['log_B'] < 0 or sig['B'] < 1 or not 0 < sig['attenuation'] <= 1):
        raise ValueError('Saved signal has unresolved finite/outcome/normalization range')
    return dict(m=request['m'], identity=deepcopy(request['identity']), law_binding=deepcopy(request['law_binding']),
        physical_candidate=deepcopy(request['physical_candidate']), source=deepcopy(record['source']),
        raw=deepcopy(sig['raw']), corrected=deepcopy(sig['corrected']), target=deepcopy(request['domain']['target']),
        reference=deepcopy(sig['reference']),
        normalization=dict(kind='FINITE_LOG_B_SUPPLIED_BY_PRODUCER', log_B=sig['log_B'], B=sig['B'],
                           attenuation=sig['attenuation'], numerical=deepcopy(record['numerical'])),
        bias=dict(total_signed_bias=deepcopy(sig['total_signed_bias'])),
        uncertainty=dict(numerical=deepcopy(record['numerical']), law_distance=None, state_error=None,
                         empirical_allowance=None), qualification='UNQUALIFIED', status='SIGNAL_AVAILABLE_NOT_RMSE_QUALIFIED')


def execute(root, prepared, out, *, source, context_factory=None, reference_reader=None, saved=(), api=None):
    """Explicit future-science batch. Fresh results, partial failure values, no retry.

    The caller supplies approved reference readers/descriptors and may supply A's
    already loaded shared context. Nothing here recomputes reference or chooses
    candidates. Default context uses existing frozen snapshot/matrix builders.
    """
    api = api or published_api(str(root))
    current = prepare(root, prepared['configuration'], target=prepared['target'], api=api,
                      references=[r['domain']['reference'] for r in prepared['requests'] if r['domain']['reference']])
    if not p.same(current, prepared) or not source:
        raise ValueError('Prepared schedule or execution SOURCE differs')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        with (out / name).open('x') as f:json.dump(value, f, indent=2, allow_nan=False); f.write('\n')
    write('PLAN.json', prepared)
    result = dict(status='STARTED', means=[], records=[], errors=[], new_mean_evaluations=0,
                  saved_copies=0, N=None, G=None, scientific_PASS_FAIL=None, mandatory_stop=True)
    context = None; stopped = False
    try:
        if source.get('commit'):
            for path in (MODULE, NUMERICAL):
                if blob(root, source['commit'], path) != (Path(root)/path).read_bytes():
                    raise ValueError('Execution numerical SOURCE bytes differ: ' + path)
        from .baseline_signal_numerics import Context, load_shared_h4, evaluate
        for req in prepared['requests']:
            if stopped:
                result['means'].append(unavailable(req, 'NOT_STARTED_AFTER_NUMERICAL_STOP')); continue
            matches = [v for v in saved if p.same(v.get('domain'), req['domain'])]
            if len(matches) > 1:raise ValueError('Ambiguous saved signals; choose SOURCE, do not pool')
            if matches:
                record = deepcopy(matches[0])
                if record.get('status') != 'NUMERICAL_MEAN_AVAILABLE' or not record.get('source'):
                    raise ValueError('Saved numerical failure/unqualified SOURCE is not a successful signal')
                result['saved_copies'] += 1
            elif req['reference_status'] != 'SUPPLIED':
                result['means'].append(unavailable(req, 'MISSING_REFERENCE')); continue
            else:
                if reference_reader is None:raise ValueError('Explicit reference reader required; no solver fallback')
                reference = reference_reader(deepcopy(req['domain']['reference']))
                if context is None:
                    context = (context_factory() if context_factory else
                               Context(root, *load_shared_h4(root, prepared['configuration']['spec']['binding'])))
                if not p.same(context.binding, req['domain']['binding']):raise ValueError('Context H/state/source binding differs')
                record = evaluate(context, req, reference, source=source)
                result['new_mean_evaluations'] += 1
            write('round_' + str(req['m']) + '.json', record)
            result['records'].append(record); result['means'].append(b_mean(req, record))
            if record['status'] != 'NUMERICAL_MEAN_AVAILABLE':stopped = True
        result['status'] = ('STOP_NUMERICAL_UNRESOLVED' if stopped else
                            'INCOMPLETE_REFERENCES' if any(v['raw'] is None for v in result['means']) else
                            'BASELINE_SIGNALS_COMPLETE_NOT_RMSE_QUALIFIED')
    except Exception as error:
        result['status'] = 'STOP_INPUT_OR_SOURCE_ERROR'; result['errors'].append(dict(type=type(error).__name__, message=str(error)))
    done = {v['m'] for v in result['means']}
    result['means'] += [unavailable(r, 'NOT_EVALUATED_AFTER_STOP') for r in prepared['requests'] if r['m'] not in done]
    result['means'].sort(key=lambda v: v['m'])
    write('RESULT.json', result)
    return result
