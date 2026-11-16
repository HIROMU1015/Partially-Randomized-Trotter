"""Limited Pauli-tail mean adapter over B canonical metadata and A numerics.

No N/RMSE/price/controller implementation. Scientific leaves are load_context,
evaluate (delegated), saved-DF acceptance, and local full/sector comparisons.
"""
import ast
from copy import deepcopy
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

from . import df_head_pauli_tail_p0 as p0
from . import pr_rpe_signal_provider as provider
common = provider.common
P0 = '4220f3006d0d8d413e195ed948e250578513b557'
P0DIR = 'docs/research/df_head_pauli_tail_p0_results_20261114/'
B = 'ce0517de6e7b58cf4bf4c69f99f998d86099c406'
DF = '91349e703e0d09ef702ccd108745dfaf1acc29f7'
DFDIR = 'docs/research/pr_rpe_first_signal_results_20261113/'
CONTROL = 'P1_LD1_M0_DF_MATCHED_R_DIAGNOSTIC'
KIND = 'GLOBAL_COLLECTED_JW_PAULI_OF_SAVED_DF_TAIL'
MODULE = 'src/trottertracks/df_head_pauli_tail_p1.py'


def read(root, commit, path):
    return json.loads(p0.blob(root, commit, path))


def b_api(root):
    """Existing pinned B metadata loader, with B's latest public SOURCE.

    Only generate/validate/exact_scalar are exposed; no cost/select/providers.
    B's identity/hash/round generation bodies are not reimplemented by A.
    """
    from .h4_pr_m1_rmse import function
    scope = dict(ast=ast, deepcopy=deepcopy, Fraction=Fraction, hashlib=hashlib,
        json=json, math=math, subprocess=subprocess, SimpleNamespace=SimpleNamespace,
        ROOT=Path(root), B_SOURCE=B, B_PATH='src/trottertracks/pr_rpe_rmse_controller.py', p=common)
    return function(root, 'fefd548ecd56f60d72031b3d385f6f5e6ee95d06',
        'tests/check_pr_rpe_signal_plan.py', 'b_api', scope)()


def metadata(root):
    """Pinned data/bindings only: does not generate a new real schedule/mean."""
    base = p0.metadata(root)
    evidence = read(root, P0, P0DIR+'run01/result.json')
    if (evidence['binding'] != base['binding'] or len(evidence['checks']) != 81
            or not all(c['PASS'] for c in evidence['checks'])
            or evidence['status'] != 'P0_OPERATOR_DICTIONARY_COMPLETE_NOT_P1_OR_RPE_QUALIFIED'):
        raise ValueError('P0_EVIDENCE_NOT_AVAILABLE_OR_DIFFERENT')
    configs = {p: read(root, p0.MENU, p0.DIRECTORY+'CONFIGURATION_LD'+str(p)+'.json') for p in (1, 2)}
    dictionaries = {}
    for row in evidence['prefixes']:
        p = row['prefix']; path = P0DIR+'run01/LD'+str(p)+'_DICTIONARY.json'
        obj = read(root, P0, path)
        raw = json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        if (hashlib.sha256(raw).hexdigest() != row['dictionary_sha256']
                or obj['binding'] != base['binding'] or obj['prefix'] != p
                or obj['saved_DF_scalar'] != base['saved_tail_descriptors'][str(p)]['scalar']):
            raise ValueError('P0_DICTIONARY_BINDING_OR_SCALAR')
        dictionaries[p] = dict(payload=obj, dictionary_sha256=row['dictionary_sha256'],
            source=dict(commit=P0, path=path), P0_metrics=row['identity_metrics'])
    for p, config in configs.items():
        s = config['spec']
        if (s['finite_K'] != 2 or s['formula'] != '2nd' or s['M'] != 12
                or provider._scalar(s['h']) != Fraction(.1)
                or provider._scalar(s['delta_pf']) != Fraction(.1)):
            raise ValueError('P1_FIXED_S2_K2_GRID')
    return dict(saved=base, original_DF_configurations=configs, dictionaries=dictionaries,
        P0_source=P0, B_source=B, saved_DF_source=DF)


def dictionary_identity(item):
    return dict(kind=KIND, dictionary_sha256=item['dictionary_sha256'], source=deepcopy(item['source']),
        prefix=item['payload']['prefix'], mode_mapping=item['payload']['mapping'],
        coefficient_policy='P0_GLOBAL_COLLECTOR_NO_ADDITIONAL_CUTOFF',
        scalar_policy='KEEP_SAVED_sR_AND_c_ONCE', event_space='FULL_2_TO_N_NOT_PROJECTED_PAULI',
        P0_operator_residual_source=dict(commit=P0, path=P0DIR+'run01/result.json'))


def generate_configuration(b, original, item, *, fixed_R=None):
    """B-owned whole canonical schedule. fixed_R overrides LD1/m0 only."""
    s = deepcopy(original['spec'])
    for key in ('schedule_adapter', 'A_input', 'A_release'):
        s.pop(key, None)
    descriptor = dictionary_identity(item)
    lam_source = dict(item['source'], selector=['lambda_P'])
    lam = Fraction(item['payload']['lambda_P'])
    s['split'].update(tail_dictionary=deepcopy(descriptor), lambda_R=dict(
        numerator=lam.numerator, denominator=lam.denominator), lambda_source=lam_source,
        one_body_policy='DETERMINISTIC', scalar=item['payload']['saved_DF_scalar'])
    # B's enum is an alias for the SAME paper rotation-count definition/value.
    s['kappa']['A_definition_name'] = s['kappa']['kind']
    s['kappa']['kind'] = 'PAPER_TOTAL_ROTATIONS'
    s['law'] = dict(model='ideal_finite_RTE_law', tail_dictionary=deepcopy(descriptor),
        source=dict(primitive=dict(commit=P0, path='src/trotterlib/rte.py'), dictionary=item['source']),
        event_model='EXISTING_PAIRED_TAYLOR_SIGNED_PAULI_INVOLUTIONS',
        scalar='SAVED_sR_AND_c_ONCE_AT_TOTAL_TIME', actual_sampler_qualified=False)
    s['event_mapping'] = dict(kind='PAPER_TOTAL_UNIFORM_OUTER_DRAFT',
        source=dict(commit=B, path='src/trottertracks/pr_rpe_rmse_controller.py', function='paper_event_mapper'))
    if fixed_R is None:
        config = b.generate(s)
    else:
        if s['L_D'] != 1 or type(fixed_R) is not int or fixed_R < 1:
            raise ValueError('ONLY_LD1_M0_DF_FIXED_R_CONTROL')
        s['event_mapping'].update(kind=CONTROL, fixed_round=0, fixed_R_outer=fixed_R,
            kappa_applied_at_fixed_round=False, other_rounds='UNCHANGED_PAPER_MAPPER_METADATA_ONLY',
            scientific_schedule_adopted=False)
        # Reuse B's native mapper for metadata at noncontrol rounds.
        normal = b.generate(dict(s, event_mapping=dict(kind='PAPER_TOTAL_UNIFORM_OUTER_DRAFT',
            source=deepcopy(s['event_mapping']['source']))))
        def mapping(spec, row):
            value = deepcopy(normal['rounds'][row['m']]['mapping'])
            value['kind'] = CONTROL
            if row['m'] == 0:
                value.update(R_outer=fixed_R, event_count=row['q_PF']*fixed_R,
                    kappa_applied=False, control='DF_MATCHED_R_NOT_KAPPA_REOPTIMIZED')
            return value
        config = b.generate(s, event_mapper=mapping)
    b.validate(config)
    if config['id'] == original['id']:
        raise ValueError('PAULI_CONFIGURATION_CANNOT_REUSE_DF_ID')
    return config


def job(config, m, source, b):
    """One selected row; B still owns the complete M+1 canonical identity."""
    b.validate(config)
    row = config['rounds'][m];s = config['spec'];c = deepcopy(row['candidate'])
    c.update(T=float(b.exact_scalar(row['T'])), delta=float(b.exact_scalar(row['actual_delta_pf'])))
    context = dict(physical_candidate=c, structure=deepcopy(row['structure']),
        binding=deepcopy(s['binding']), profile=deepcopy(s['profile']), split=deepcopy(s['split']),
        law=deepcopy(s['law']), dictionary_identity=deepcopy(s['split']['tail_dictionary']),
        numerical_recipe=provider.numerical_recipe(source))
    return dict(m=m, q_RPE=row['q_RPE'], q_PF=row['q_PF'], B_configuration_id=config['id'],
        B_candidate_id=row['candidate']['id'], B_identity=deepcopy(row['identity']),
        B_law_binding=deepcopy(row['law_binding']), physical_context=context)


def load_contexts(root, meta, source, normalizers):
    """Saved state plus P0 operators only; no DF extension/eigensolve/refit."""
    saved = meta['saved']; ham, sector, full_state, state, observed = p0.load_saved(root, saved)
    def archive(path):
        with np.load(io.BytesIO(p0.blob(root, P0, P0DIR+'run01/'+path)), allow_pickle=False) as z:
            return {k:z[k] for k in z.files}
    common_ops = archive('COMMON_OPERATORS.npz');basis = sector.basis_indices
    if not np.array_equal(basis, common_ops['sector_basis_indices']):
        raise ValueError('P0_SECTOR_BASIS_DIFFERENT')
    contexts = {}
    for prefix in (1, 2):
        A = archive('LD'+str(prefix)+'_OPERATORS.npz')['Pauli_nonidentity']
        contexts[prefix] = {}
        for space in ('sector36', 'full256') if prefix == 1 else ('sector36',):
            indices = basis if space == 'sector36' else np.arange(256)
            psi = state if space == 'sector36' else full_state
            d = meta['dictionaries'][prefix];obj = d['payload']
            tails = {prefix:dict(scalar=obj['saved_DF_scalar'], lambda_R=obj['lambda_P'],
                source=dict(d['source'], selector=['lambda_P']))}
            ctx = provider.make_context([a[np.ix_(indices, indices)] for a in common_ops['blocks']],
                psi, ham.constant, tails, binding=saved['binding'], source=dict(commit=P0,
                    path=P0DIR+'run01/COMMON_OPERATORS.npz'))
            # This is the saved Pauli sum, not an on-the-fly DF-tail surrogate.
            # Its tiny difference at the unchanged saved scalar is P0 evidence.
            matrix = np.array(A[np.ix_(indices, indices)], copy=True);matrix.setflags(write=False)
            ctx.tails[prefix].update(matrix=matrix, dictionary_identity=dictionary_identity(d),
                matrix_source=dict(commit=P0, path=P0DIR+'run01/LD'+str(prefix)+'_OPERATORS.npz',
                    field='Pauli_nonidentity'), P0_operator_metrics=deepcopy(d['P0_metrics']))
            ctx.normalizers = normalizers;ctx.normalizer_budget = 5
            ctx.state_metadata = dict(saved_loader=observed, ground_state_certified=False,
                actual_law_error=None, state_not_regenerated=True)
            contexts[prefix][space] = ctx
    return contexts


def accept_saved_DF(root, prefix, m, original, reference):
    """At most the four approved saved DF rows; never recompute their mean/B."""
    path = DFDIR+'run01/LD'+str(prefix)+'/m'+str(m)+'.json'
    row = read(root, DF, path);old = original['rounds'][m];c = row['candidate']
    if (m not in (0, 6) or prefix not in (1, 2)
            or row['B_configuration_id'] != original['id']
            or row['B_candidate_id'] != old['candidate']['id']
            or not common.same(row['B_identity'], old['identity'])
            or not common.same(row['B_law_binding'], old['law_binding'])
            or not common.same(row['physical_context']['profile'], original['spec']['profile'])
            or not common.same(row['physical_context']['split'], original['spec']['split'])
            or not common.same(c, old['physical_candidate'])
            or row['status'] != 'LOCAL_FINITE_MEAN_NOT_RMSE_QUALIFIED'
            or row['numerical'].get('gate_pass') is not True
            or row.get('scalar_once') is not True
            or row.get('raw_corrected_independent_propagation') is not True
            or row['physical_context']['binding'] != reference['binding']
            or row['source']['commit'] != 'fefd548ecd56f60d72031b3d385f6f5e6ee95d06'):
        raise ValueError('SAVED_DF_IDENTITY_SOURCE_OR_GATE')
    for key in ('closure','maximum_head_unitarity','exact_tail_norm_error_after_scalar',
                'step_unitarity_error','power_unitarity_error','norm_error'):
        value = row['numerical'][key]
        if not math.isfinite(value) or not 0 <= value <= 1e-9:
            raise ValueError('SAVED_DF_NUMERICAL_GATE: '+key)
    signal = row['signal']
    for key in ('raw', 'corrected', 'exact_tail', 'reference'):
        z = complex(signal[key]['real'], signal[key]['imag'])
        if not math.isfinite(z.real) or not math.isfinite(z.imag):
            raise ValueError('SAVED_DF_NONFINITE')
    if (row['allowance'].get('rigorous_bound') is not False
            or not math.isfinite(row['allowance']['u'])
            or not math.isfinite(signal['log_B']) or signal['log_B'] < 0):
        raise ValueError('SAVED_DF_ALLOWANCE_OR_NORMALIZATION')
    a = complex(signal['reference']['real'], signal['reference']['imag'])
    b = reference['evidence']['reference']
    if abs(a-complex(b['real'], b['imag'])) > 1e-10:
        raise ValueError('SAVED_DF_REFERENCE_ANCHOR')
    return row, dict(commit=DF, path=path, acceptance='SOURCE_BOUND_EXISTING_NUMERICAL_GATES_NOT_NEW_N')


def compare_means(left, right, *, kind, strict=True):
    """Local signal/normalization checks; never infer phase/energy/RMSE."""
    a,b = left['candidate'],right['candidate']
    fields = ('prefix','formula','T','q','delta','R_outer','K')
    same_context = (common.same(left['physical_context']['binding'],right['physical_context']['binding'])
        and common.same(left['physical_context']['profile'],right['physical_context']['profile'])
        and common.same(left['signed_weights'],right['signed_weights'])
        and common.same(left['allocation'],right['allocation'])
        and left['scalar_once'] is True and right['scalar_once'] is True)
    if strict and (left['B_candidate_id'] != right['B_candidate_id']
            or left['B_configuration_id'] != right['B_configuration_id']):
        raise ValueError('FULL_SECTOR_CANONICAL_ID_MISMATCH')
    if any(a[k] != b[k] for k in fields) or not same_context:
        if strict:
            raise ValueError('FULL_SECTOR_PHYSICAL_LAYOUT_OR_BINDING_MISMATCH')
        return dict(kind=kind, status='DIFFERENT_PHYSICAL_LAYOUT_NO_SAME_CONDITION_COMPARISON',
            numerical_comparison=None)
    z = lambda r,k: complex(r['signal'][k]['real'],r['signal'][k]['imag'])
    gaps = {key:abs(z(left,key)-z(right,key)) for key in ('exact_tail','corrected','raw')}
    # DF versus Pauli is expected to have different raw/B, even at fixed R.
    tested = ('exact_tail','corrected','raw') if strict else ('exact_tail','corrected')
    extra = dict(log_B=abs(left['signal']['log_B']-right['signal']['log_B']),
        attenuation=abs(left['signal']['attenuation']-right['signal']['attenuation'])) if strict else {}
    ok = all(math.isfinite(gaps[k]) and gaps[k] <= 1e-9 for k in tested)
    ok = ok and all(math.isfinite(v) and v <= 1e-9 for v in extra.values())
    return dict(kind=kind, status='LOCAL_COMPARISON_GATE_PASS' if ok else 'NUMERICAL_FAILURE',
        signal_differences=gaps, normalization_differences=extra,
        matched_conditions=list(fields)+['H_state_binding','profile','signed_weights','allocation','scalar_once'],
        raw_dictionary_difference_expected=not strict, threshold=1e-9,
        energy_or_RMSE_inference=None)
