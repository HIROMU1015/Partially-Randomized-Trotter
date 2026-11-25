"""Canonical deterministic-PF metadata from a fixed provider base.

This is metadata only: configuration/schedule generation uses the existing C/B
API. It does not construct operators, prepare a physical runtime, evaluate a
signal/price/Q3, or eliminate a prefix using its truncation shift. Widths are relative to the h_RPE in that fixed base. H4 saved priority
evidence is used only for the original H4 base.
"""
from copy import deepcopy
import csv
from fractions import Fraction
from functools import lru_cache
import io
import json

from . import baseline_signal_pipeline as C

# Original saved base and the same pinned PF mapper used by D36.
BASE_SPEC = dict(commit='a9abf8de75bf3432fcd714e868e5c14c8884b8f8',
    path='docs/research/pr_rpe_first_signal_menu_20261113/CONFIGURATION_LD1.json',
    selector=['spec'])
PORT_SOURCE = '36ad751d4ebf0ec160532615da6425b268a0c18a'

H = Fraction(.1)
FORMULAS = ('2nd', '4th', '4th(m5_best)')
DEFAULT_WIDTHS = {
    '2nd': ('1/2', '1', '2', '4', '8', '16/7', '8/3'),
    '4th': ('1/2', '1', '2', '4', '8', '16/3', '32/5'),
    '4th(m5_best)': ('1/2', '1', '2', '4', '8', '16'),
}
T1_SOURCE = '5e6f9fd214631f7ce92fad0e9c0d273dfb479533'
T1_DIR = 'docs/research/truncated_pf_t1_results_20261114/run01/'
T2_SOURCE = '5fb2c32e39e63267ea1748861269c362a93a1b86'
T2_DIR = 'docs/research/truncated_pf_t2_results_20261114/'


def exact(value):
    """Accept an exact rational or the original binary64 value, never round it."""
    if isinstance(value, dict):
        return Fraction(value['numerator'], value['denominator'])
    if isinstance(value, bool):
        raise ValueError('PF width must be a positive rational')
    return Fraction(value)


def encoded(value):
    value = exact(value)
    return dict(numerator=value.numerator, denominator=value.denominator)


def _prefix(method, p, rank=10):
    if method not in ('FULL', 'FULL_PF', 'B0I', 'B0I_P7', 'TRUNCATED_PF'):
        raise ValueError('Only Full PF and B0I deterministic PF are supported')
    if type(p) is not int or p not in range(rank+1):
        raise ValueError('PF prefix p must be an integer from zero through the bound rank')
    if method in ('FULL', 'FULL_PF') and p != rank:
        raise ValueError('Full PF must retain all fragments of the bound rank')
    return 'FULL_PF' if p == rank else 'TRUNCATED_PF'


def configure(root, method, p, formula, delta_over_h, *, base_spec=None):
    """Return C's canonical configuration; B0I p=rank aliases Full PF exactly.

    p=0 is one-body-only; p below rank retains fragments 0..p-1.
    ``delta_over_h`` is not restricted to the metadata shortlist/grid.
    """
    locator = base_spec or BASE_SPEC
    base = base_configuration(root, locator)
    rank = base['binding']['rank']
    mode = _prefix(method, p, rank)
    if formula not in FORMULAS:
        raise ValueError('Supported PF formulas are S2, standard Yoshida4, M5')
    width = exact(delta_over_h)
    if width <= 0:
        raise ValueError('PF width must be positive')
    api = C.published_api(str(root))
    spec = api.D.full_pf_spec(base, formula=formula,
                             delta_pf=encoded(width * exact(base['h'])), M=base['M'], rank=rank)
    if mode == 'TRUNCATED_PF':
        spec['L_D'] = p
        spec['split'].update(L_D=p, deterministic_scope='TRUNCATED_H', original_PF_method='B0I')
        spec['split']['retained_hamiltonian'].update(construction='B0I', prefix=p)
    return C.configure(root, spec, api=api)


def base_configuration(root, locator):
    value = _json(str(root), locator['commit'], locator['path'])
    for field in locator.get('selector', ('spec',)):
        value = value[field]
    return deepcopy(value)


@lru_cache(maxsize=None)
def _json(root, commit, path):
    return json.loads(C.blob(root, commit, path))


@lru_cache(maxsize=None)
def _t2_cells(root):
    raw = C.blob(root, T2_SOURCE, T2_DIR + 'CELLS.csv').decode()
    return tuple(csv.DictReader(io.StringIO(raw)))


def priority_evidence(root, p, formula, configuration):
    """Copy saved T1 values and T2 locators for ordering, never a Q3 veto.

    A missing T2 cell means acquisition may be needed, not rejection. No saved
    scientific array is loaded and no shift or spectrum is re-evaluated here.
    """
    evidence = dict(usage='PRIORITY_ONLY_NOT_Q3_ACCEPTANCE_OR_REJECTION', T1=None, T2=[])
    if p == 10:
        evidence['full_pf_alias'] = True
        return evidence
    path = T1_DIR + 'prefix_%d.json' % p
    saved = _json(str(root), T1_SOURCE, path)
    if saved['prefix'] != p or saved['construction'] != 'B0I':
        raise ValueError('Saved T1 prefix/construction mismatch')
    evidence['T1'] = dict(source=dict(commit=T1_SOURCE, path=path),
        status=saved['status'], delta_E_trunc_B0I_ha=saved['delta_E_trunc_B0I_ha'],
        tail_identity_scalar=saved['tail_coefficients']['used_c_tail'],
        lambda_R=saved['tail_coefficients']['used_lambda_R'],
        scalar_selector=['tail_coefficients', 'used_c_tail'])
    if p not in (7, 8, 9):
        return evidence
    actual = {exact(r['actual_delta_pf']) / H for r in configuration['rounds']}
    for row in _t2_cells(str(root)):
        if int(row['prefix']) != p or row['formula'] != formula or exact(row['actual_delta_over_h']) not in actual:
            continue
        cell = ('cell_p%d_' % p + formula.replace('(', '_').replace(')', '') + '_'
                + str(exact(row['actual_delta_over_h'])).replace('/', 'o'))
        directory = T2_DIR + 'run01/' + cell
        evidence['T2'].append(dict(actual_delta_over_h=encoded(row['actual_delta_over_h']),
            status=row['status'], scope_source=dict(commit=T2_SOURCE, path=directory+'/CELL_SCOPE.json'),
            result_source=dict(commit=T2_SOURCE, path=directory+'/result.json'),
            operator_source=dict(commit=T2_SOURCE, path=directory+'/OPERATORS.npz',
                                 field='PF_step_without_global_scalars'),
            first_mean_source=dict(commit=T2_SOURCE, path=directory+'/round_first_check.json'),
            m12_mean_source=dict(commit=T2_SOURCE, path=directory+'/round_m12_check.json')))
    return evidence


def candidates(root, formulas=None, widths=None, p_values=None, *, base_spec=None):
    """Declare metadata candidates, not a request to evaluate their Cartesian grid.

    Defaults represent all p0..9 plus the p10 Full-PF alias, using the original
    20 width/formula choices per prefix. ``widths`` may be a shared iterable or
    a formula->iterable mapping; arbitrary positive widths remain legal.
    Search selection and shot optimization belong to B2, not this module.
    """
    locator = base_spec or BASE_SPEC
    base = base_configuration(root, locator)
    rank = base['binding']['rank']
    formulas = tuple(FORMULAS if formulas is None else formulas)
    p_values = tuple(range(rank+1) if p_values is None else p_values)
    if len(set(formulas)) != len(formulas) or any(f not in FORMULAS for f in formulas):
        raise ValueError('PF formulas must be supported and unique')
    if any(type(p) is not int or p not in range(rank+1) for p in p_values) or len(set(p_values)) != len(p_values):
        raise ValueError('PF prefixes must be unique integers within the bound rank')
    shared_widths = None if widths is None or isinstance(widths, dict) else tuple(widths)
    out = []
    for p in p_values:
        method = 'FULL_PF' if p == rank else 'TRUNCATED_PF'
        for formula in formulas:
            values = (DEFAULT_WIDTHS[formula] if widths is None else
                      widths.get(formula, ()) if isinstance(widths, dict) else shared_widths)
            seen = set()
            for value in values:
                width = exact(value)
                if width in seen:
                    continue
                seen.add(width)
                cfg = configure(root, method, p, formula, width, base_spec=locator)
                out.append(dict(method=method, p=p, formula=formula, delta_over_h=encoded(width),
                    configuration=deepcopy(cfg), configuration_id=cfg['id'],
                    configuration_source=dict(base=deepcopy(locator),
                        generator=dict(B=C.B_SOURCE, D=C.D_SOURCE), port_source=PORT_SOURCE),
                    p10_alias_full_pf=(p == rank),
                    priority_evidence=(priority_evidence(root, p, formula, cfg) if locator==BASE_SPEC else
                        dict(usage='PRIORITY_ONLY_NOT_Q3_ACCEPTANCE_OR_REJECTION', T1=None, T2=[]))))
    return out
