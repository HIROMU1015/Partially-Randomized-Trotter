"""Metadata-only Truncated PF family schedule table. No H4 array/context, no numerics.

Reuses the published generators exactly as C's c546 BUILD_MANIFEST did:
D.full_pf_spec -> truncated split (B0I, prefix p) -> C.configure (B canonical IDs)
-> C.prepare (physical requests, saved reference DESCRIPTORS only).
Outputs per (prefix, formula, requested width): canonical ID, per-round q_PF and
actual widths, signed S2 piece counts, and the unique (prefix, formula,
actual-width) step cells a future H_eff scan would need. Nothing is evaluated.
"""
from collections import OrderedDict
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import rpe_optimization_pipeline as I  # noqa: E402

A_MENU = 'a9abf8de75bf3432fcd714e868e5c14c8884b8f8'
A_RESULT = '91349e703e0d09ef702ccd108745dfaf1acc29f7'
PHYS = dict(commit='bba150834ca30bdb186fbfd58c55a8d51177b855',
            path='docs/research/h4_pr_physical_error_contract_20261113/PHYSICAL_INPUTS.json')
C546 = dict(commit='c5466946ab0ef9b50511210ca1e58fb7147bb0f0',
            path='docs/research/baseline_finite_signal_plan_20261113/BASELINE_FINITE_MANIFEST.json')
A_FULLPF = dict(commit='71f04be7bf94e7773a32270f36ae29eee1b903d5',
                path='docs/research/h4_full_pf_followup_results_20261114/C_HANDOFF.json')
H = Fraction(.1)  # the original binary64 h as an exact rational (A/C convention)
# Proposed width ladders (multiples of h), aligned with A's evaluated Full PF cells
LADDER = OrderedDict([('2nd', [H, 2*H, 16*H/7, 8*H/3]),
                      ('4th', [4*H, 16*H/3, 32*H/5]),
                      ('4th(m5_best)', [8*H, 16*H])])
PREFIXES = list(range(10))  # legal truncation prefixes 0 <= L_D < rank 10; p=10 is Full PF


def read(ref):
    return json.loads(I.C.blob(ROOT, ref['commit'], ref['path']))


def enc(v):
    return dict(numerator=v.numerator, denominator=v.denominator)


def build():
    api = I.providers(str(ROOT)); P = I.C
    base = read(dict(commit=A_MENU, path='docs/research/pr_rpe_first_signal_menu_20261113/CONFIGURATION_LD1.json'))
    target = read(dict(commit=A_RESULT, path='docs/research/pr_rpe_first_signal_results_20261113/B_MEANS_LD1.json'))['target']
    s = base['spec']; binding = deepcopy(s['binding']); physics = read(PHYS)
    exact = lambda v: Fraction(v['numerator'], v['denominator']) if isinstance(v, dict) else Fraction(v)
    # Reproduce c546's B0I p7 S2/.1 configuration ID as a generator-consistency check.
    c546 = read(C546); c546_ids = {c['label']: c['configuration']['id'] for c in c546['candidates']}
    a_ids = {v['label']: v['configuration_id'] for v in read(A_FULLPF).get('configurations', [])}
    rows = []; cells = OrderedDict(); checks = dict(c546_B0I_P7_S2_D01_id_reproduced=None, full_pf_ids_reproduced={})

    def spec_for(p, formula, width):
        full = api.D.full_pf_spec(s, formula=formula, delta_pf=enc(width), M=12, rank=10)
        if p == 10:
            return full
        t = deepcopy(full); t['L_D'] = p
        t['split'].update(L_D=p, deterministic_scope='TRUNCATED_H', original_PF_method='B0I')
        t['split']['retained_hamiltonian'].update(construction='B0I', prefix=p)
        return t

    for p in PREFIXES + [10]:
        for formula, widths in LADDER.items():
            for width in widths:
                config = P.configure(ROOT, spec_for(p, formula, width), api=api)
                refs = [dict(m=r['m'], T=deepcopy(r['T']), binding=deepcopy(binding),
                             source=dict(**PHYS, selector=['rounds', r['m'], 'diagnostics', 'reference_pair']))
                        for r in config['rounds']]
                assert all(set(('reference', 'independent_reference', 'signal_difference'))
                           <= set(physics['rounds'][r['m']]['diagnostics']['reference_pair']) for r in config['rounds'])
                plan = P.prepare(ROOT, config, target=target, references=refs, api=api)
                rounds = []
                for r, req in zip(config['rounds'], plan['requests']):
                    actual = exact(r['actual_delta_pf']); c = req['physical_candidate']
                    assert exact(r['T']) == r['q_PF']*actual and c['q'] == r['q_PF']
                    if p < 10:
                        assert (c['method'], c['prefix'], c['formula']) == ('B0I', p, formula)
                    pieces = len(r['structure']['signed_weights'])*r['q_PF']
                    key = (p, formula, actual)
                    cell = cells.setdefault(key, dict(prefix=p, formula=formula, actual_delta=enc(actual),
                        actual_delta_over_h=str(actual/H), first_use=[], physical_candidate_ids=set()))
                    cell['first_use'].append((float(exact(r['T'])), r['m'], r['q_PF']))
                    cell['physical_candidate_ids'].add(c['id'])
                    rounds.append(dict(m=r['m'], q_PF=r['q_PF'], actual_delta_over_h=str(actual/H),
                                       signed_pieces=pieces, physical_candidate_id=c['id']))
                label = ('FULL_PF' if p == 10 else 'B0I_P'+str(p))+'_'+formula+'_'+str(width/H)+'h'
                rows.append(dict(label=label, prefix=p, formula=formula, requested_delta_over_h=str(width/H),
                    configuration_id=config['id'], m12_q_PF=rounds[-1]['q_PF'], m12_signed_pieces=rounds[-1]['signed_pieces'],
                    total_signed_pieces=sum(v['signed_pieces'] for v in rounds),
                    early_actual_over_h=[v['actual_delta_over_h'] for v in rounds[:5]], rounds=rounds,
                    science_calls=plan['science_calls']))
                if p == 7 and formula == '2nd' and width == H:
                    checks['c546_B0I_P7_S2_D01_id_reproduced'] = config['id'] == c546_ids.get('B0I_P7_S2_D01')
                if p == 10 and formula == '2nd' and width == H:
                    checks['full_pf_ids_reproduced']['FULL_PF_S2_D01'] = config['id'] == c546_ids.get('FULL_PF_S2_D01')
    cell_rows = []
    for (p, formula, actual), v in cells.items():
        first = min(v['first_use'])
        cell_rows.append(dict(prefix=p, formula=formula, actual_delta=v['actual_delta'],
            actual_delta_over_h=v['actual_delta_over_h'], first_use_m=first[1], first_use_q=first[2],
            distinct_physical_candidates=len(v['physical_candidate_ids'])))
    per_prefix = {p: sum(1 for c in cell_rows if c['prefix'] == p) for p in PREFIXES + [10]}
    out = dict(status='METADATA_ONLY_NO_SCIENCE', H4_arrays_or_context_read=False, numerical_calls=0,
        generator_sources=dict(B=P.B_SOURCE, D=P.D_SOURCE, pipeline='src/trottertracks/baseline_signal_pipeline.py',
                               base_config=A_MENU, c546=C546['commit']),
        ladder={f: [str(w/H)+'h' for w in ws] for f, ws in LADDER.items()},
        prefixes=PREFIXES, full_pf_reference_prefix=10,
        configurations=len(rows), unique_step_cells_per_prefix=per_prefix,
        unique_step_cells_truncated_total=sum(v for p, v in per_prefix.items() if p < 10),
        checks=checks, configurations_table=rows, step_cells=cell_rows)
    return out


if __name__ == '__main__':
    result = build()
    (OUT/'PLAN_METADATA.json').write_text(json.dumps(result, indent=1, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('configurations', 'unique_step_cells_per_prefix',
                      'unique_step_cells_truncated_total', 'checks')}, indent=1))
