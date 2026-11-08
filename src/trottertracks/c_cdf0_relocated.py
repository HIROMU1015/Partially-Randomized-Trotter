"""Saved gamma-zero CDF adapter; imports and preflight are passive.

The dormant execute entry is scientific work requiring separate user permission.
It reuses relocation/reference/native engines without modifying their fixed runs.
"""
from pathlib import Path
import importlib.metadata
import json

SCOPE = 'C_CDF0_STAGE_A_SAVED_R8_RELOCATED_PR_V1'
CANDIDATE_ID = 'CDF0_STAGE_A_X0_V2_GAMMA0_FINAL_R8'
CONTRACT = 'docs/research/research_c_cdf0_relocated_contract_proposal_20261108.json'
R_VALUES = (8, 16, 32, 64, 128)
POLICY = dict(rank=8, ld=2, s=1, h='1/10', K=0, rounds=list(range(13)),
              r_grid=list(R_VALUES), selection_samples=32, final_samples=64,
              state='own_DF_ground_state_oracle', state_preparation='EXCLUDED',
              new_fit=0, new_DF_norm=0, new_D=0, new_ENTRY=0)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local(root, path):
    path = (root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('LOCAL_PATH_REQUIRED')
    return path


def verify_bytes(root, contract):
    """SOURCE and inputs only; never deserialize a molecular array or circuit."""
    root = Path(root).resolve()
    source = local(root, contract['SOURCE'])
    if sha(source) != contract['SOURCE_sha256']:
        raise ValueError('SOURCE_MANIFEST')
    for group in (read(source)['files'], contract['inputs']):
        for name, expected in group.items():
            if sha(local(root, name)) != expected:
                raise ValueError('SOURCE_OR_INPUT:' + name)


def preflight(root, path=CONTRACT):
    root = Path(root).resolve()
    c = read(local(root, path))
    if (c['scope'] != SCOPE or c['candidate']['id'] != CANDIDATE_ID
            or c['candidate']['gamma'] != 0 or c['policy'] != POLICY
            or c['retry'] != 0 or c['resume'] != 0 or c['mandatory_stop'] is not True
            or 'handoff' in c or c['scientific_execution_permission'] is not False):
        raise ValueError('FIXED_CDF0_SCOPE')
    row = c['candidate']
    for key in ('path', 'reference', 'ground', 'physical_record', 'B1_contract', 'stage_A_result'):
        if row[key] not in c['inputs']:
            raise ValueError('UNBOUND_CDF0_INPUT')
    if c['inputs'][row['path']] != row['sha256']:
        raise ValueError('CDF0_CANDIDATE_BYTES')
    # The historical B1 contract binds the otherwise J/U-free reference export
    # to this baseline, not to ENTRY or the old initial strong-D CDF8.
    b1 = read(local(root, row['B1_contract']))
    if (b1['final_exports']['baseline'] != row['path']
            or b1['saved_inputs'][row['path']] != row['sha256']
            or b1['saved_inputs'][b1['stage_A_input']] != c['input_identity']):
        raise ValueError('CDF0_REFERENCE_LINEAGE')
    if c['inputs'][c['original_input']] != c['input_identity']:
        raise ValueError('ORIGINAL_INPUT_IDENTITY')
    verify_bytes(root, c)
    fit = read(local(root, row['stage_A_result']))
    physical = read(local(root, row['physical_record']))
    if (fit['eligible']['baseline'] is not True
            or any(physical[k] != v for k, v in fit['sector']['baseline'].items())):
        raise ValueError('SAVED_DF_QUALIFICATION')
    mismatches = {p: (importlib.metadata.version(p), v)
                  for p, v in c['packages'].items()
                  if importlib.metadata.version(p) != v}
    if mismatches:
        raise ValueError('PACKAGE_METADATA:' + repr(mismatches))
    if local(root, c['fresh_output']).exists():
        raise FileExistsError('FRESH_OUTPUT_RETRY_RESUME_ZERO')
    return c


def connect(kappa, constant, data, reference, physical, *, row, input_identity,
            output, call, shells, ground):
    """Bridge legacy reference schema to prepare_candidate; no new solver.

    J/U were not stored in B1's reference NPZ. Its fixed contract supplies their
    provenance; independent original-factor and full-H gates remain mandatory.
    ``baseline`` below is the engine's typed slot, not an ENTRY candidate alias.
    """
    from . import c_h4_relocated_downstream as bridge
    saved = dict(reference, J=data['J'], U=data['U'])
    df = dict(physical, candidate_sha256=row['sha256'], input_sha256=input_identity,
              budget_PASS=True, budget_Ha=.00019875,
              kappa='original_g_corrected', scalar='nuclear_once')
    return bridge.prepare_candidate(kappa, constant, data['J'], data['U'], saved, df,
        candidate=row['sha256'], input_identity=input_identity, role='baseline',
        out=output, call=call, shell_reports=shells, saved_ground=ground)


def evaluate(root, c, output, call):
    """One saved CDF0, equal r opportunities, one selection-frozen final64.

    No new D or ENTRY run. No handoff, restored prices, fit or DF norm. An
    ineligible r remains null. Errors propagate to the reused one-shot saver.
    """
    from . import c_h4_relocated_downstream as bridge
    from . import c_integrated_retuning_adapter as adapter
    from . import c_pr_aware_df_stage_b_runner as loader
    from . import c_pr_aware_df_stage_b2 as b2
    from . import c1_stage1_execution as io
    from . import c_near_r32_statistics as stats
    from .c1_stage1 import car_bilinears
    from qiskit import qpy

    row = c['candidate']
    io.save(output / 'contract.json', c)
    io.save(output / 'SOURCE.json', read(root / c['SOURCE']))
    exported = call('kappa_load', row['id'], lambda: loader.load(
        root / c['kappa_export'], ('kappa', 'constant')))
    for name, array in exported.items():
        if io.array_record(array) != c['kappa_array_identity'][name]:
            raise ValueError('KAPPA_SCALAR_IDENTITY')
    data = call('candidate_load', row['id'], lambda: loader.load(root / row['path'], ('J', 'U')))
    reference = call('reference_load', row['id'], lambda: loader.load(
        root / row['reference'], ('generators', 'H')))
    shells = {}
    for label, paths in c['shells'].items():
        qs = {}
        for kind, path in paths.items():
            with (root / path).open('rb') as stream:
                items = qpy.load(stream)
            if len(items) != 1 or items[0].num_qubits != 9:
                raise ValueError('SHELL_SCHEMA')
            qs[kind] = items[0]
        shells[label], _ = call('shell_gate', label, lambda: b2.shell_assessment(
            qs['raw'], qs['compiled'], label))
    rel, a, report = connect(exported['kappa'], float(exported['constant']), data,
        reference, read(root / row['physical_record']), row=row,
        input_identity=c['input_identity'], output=output / 'A', call=call,
        shells=read(root / c['shell_reports']), ground=root / row['ground'])
    if a is None:
        return dict(status='UNQUALIFIED_GROUND',qualification=report,total_rz=None)
    car = call('car_context', row['id'], lambda: car_bilinears(8))
    _, generators = call('full_CAR', row['id'], lambda: bridge.raw_generators(rel, car=car))
    runtime = dict(source_manifest_sha256=c['SOURCE_sha256'], shells=c['shells'],
                   native_backend='process')
    cache = prepared = None
    states = {}
    rows = []
    for r in (32, 8, 16, 64, 128):
        ar, result, packet, cache = bridge.reference_r(a, r, output / f'r{r}' / 'reference',
                                                      call, pf_cache=cache)
        value = None
        if result['native_admitted']:
            if prepared is None:
                prepared = bridge.native_context(ar, car, generators, shells, call=call)
            # Each r must bind the context's arm (same candidate/relocation).
            if prepared['arm'] is not ar['arm']:
                raise ValueError('CDF0_NATIVE_IDENTITY')
            result, packet = adapter.native_cohort(root, runtime, output / f'r{r}' / 'selection',
                call, ar, result, packet, split='selection',
                cohort_key='cdf0-20261108:' + bridge.cohort_key(a['candidate_identity']), prepared=prepared)
            value = result['lanes']['ideal_exact_native']['conditional_totals']['totals']['rz_count']
        states[r] = (ar, result, packet)
        rows.append(dict(candidate_sha256=a['candidate_identity'], r=r, split='selection',
                         samples=32, correctness='PASS', eligible=value is not None, total_rz=value))
    selection = bridge.select_r(rows, a['candidate_identity'])
    io.save(output / 'frozen_selection.json', dict(selection=selection, rows=rows))
    r = selection['selected_r']
    if r is None:
        return dict(status='UNRESOLVED_NO_ELIGIBLE_R', selection=selection, total_rz=None)
    ar, result, packet = states[r]
    result, packet = adapter.native_cohort(root, runtime, output / f'final_r{r}', call,
        ar, result, packet, split='final',
        cohort_key='cdf0-20261108:' + bridge.cohort_key(a['candidate_identity']), prepared=prepared)
    binding = call('affine_binding', row['id'], lambda: stats.affine(packet, result))
    final = dict(result=result, rows=packet['kernel_samples'], bindings=binding)
    io.save(output / 'final_summary.json', final)
    saved = read(root / c['ENTRY_summary'])
    comparison = call('primary_comparison', 'CDF0-minus-saved-ENTRY-E2', lambda: stats.primary(
        final['rows'], saved['rows'], binding['ideal_exact_native'],
        saved['bindings']['ideal_exact_native'], expected_n=64,
        statistics=dict(stats.POLICY, **c['statistics'])))
    return dict(status='CDF0_REGISTERED_SCOPE_COMPLETE', candidate_id=row['id'],
                selection=selection, comparison=comparison,
                comparison_label='CDF0_minus_saved_ENTRY_E2; legacy helper PR key means CDF0',
                actual_CA_achieved=None, quantum_shots_drawn=0, mandatory_stop=True)


def execute(root, path=CONTRACT):
    """Dormant scientific entry: never call as part of technical preparation."""
    from . import c_h4_relocated_downstream as bridge
    from .c1_h4_native_fair import profile_dict
    from collections import Counter
    import threading
    root = Path(root).resolve()
    c = preflight(root, path)
    contract_sha = sha(root / path)
    if profile_dict() != c['compiler']:
        raise ValueError('COMPILER_PROFILE')
    counts = Counter()
    stopped = threading.Event()
    lock = threading.Lock()
    def call(kind, label, fn):
        if stopped.is_set():
            raise RuntimeError('DEPENDENT_WORK_STOPPED')
        try:
            value = fn()
        except BaseException:
            stopped.set()
            raise
        with lock:
            counts[kind] += 1
        return value
    call.stop = stopped  # Existing spawn backend shares this parent STOP event.
    def worker(task, output):
        try:
            value = evaluate(root, c, output, call)
            return dict(value, call_counts=dict(counts))
        finally:
            verify_bytes(root, c)
            if sha(root / path) != contract_sha:
                raise ValueError('CONTRACT_CHANGED')
    return bridge.run_fresh(root / c['fresh_output'], [dict(id='CDF0')], worker)
