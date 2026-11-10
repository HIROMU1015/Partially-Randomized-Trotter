"""Passive published-fixture checks and metadata coverage. No scientific arithmetic."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from trottertracks import rpe_normalization_intake as a0
from plan_r_screening_price_alignment import fixed_C_describer

PUBLICATION = '4745bb676568bb11d46552be42ffc6f4fba75385'
RUN = 'docs/research/pr_r_normalization_phase0_results_20261111/run01/'
FIXED = dict(commit='ac3e1ce87b5ed7d9114abc316545366dd097209e',
             path='docs/research/rpe_provider_integration_20261111/fixed_coverage/COVERAGE.json')


def verify():
    intake = a0.A0Intake(ROOT)
    accepted = json.loads((OUT / 'INTAKE.json').read_text())
    href = dict(commit=PUBLICATION, path=RUN + 'HANDOFF.json')
    rref = dict(commit=PUBLICATION, path=RUN + 'result.json')
    # Real published data can reach only the passive adapter, never numerical APIs.
    with (patch.object(a0.p, 'run_missing', side_effect=AssertionError('No numerical batch')),
          patch.object(a0.r, 'evaluate_menu', side_effect=AssertionError('No G/R evaluation')),
          patch.object(a0.b.RMenuBatch, '__call__', side_effect=AssertionError('No N')),
          patch.object(a0.b.FixedSavedPrices, '__call__', side_effect=AssertionError('No C'))):
        replay = intake.read_published(href, rref)
    assert a0.p.same(accepted, replay), 'Saved intake differs from unchanged adapter output'
    handoff = intake.store.read(href)
    assert accepted['completion'] == 'COMPLETE_RECORDED'
    assert handoff['status'] == a0.COMPLETE and handoff['normalization_evidence_valid'] is True
    assert accepted['normalization_evidence_usable'] is True and not accepted['evidence_blocks']
    assert accepted['source_preserved'] is True and accepted['input_preserved'] is True
    assert accepted['received_count'] == 24 and not accepted['missing_ids']
    original = {v['candidate_id']: v for v in handoff['records']}
    rows = {v['candidate_id']: v for v in accepted['rows']}
    for cid, row in rows.items():
        assert a0.p.same(row['record'], original[cid]), 'Diagnostic changed during copy'
        assert row['diagnostic_share_eligible'] is True and row['N_C_G_qualified'] is False
        assert row['normalization_status'] == a0.SCORED
        assert all(row['record'][k] is None for k in ('N', 'C', 'G', 'recommended_R', 'scientific_candidate_PASS_FAIL'))
        assert row['record']['phase_radius_N_status'] == 'NOT_EVALUATED'
    assert accepted['N_records'] == accepted['C_records'] == []
    assert accepted['N_C_G'] is None and accepted['full_RPE_qualified'] is False
    assert accepted['scientific_R_adopted'] is False

    packets = {s: intake.store.read(dict(commit=intake.plan['upstream_commit'],
        path='docs/research/pr_r_screening_20261111/' + s + '_METADATA_PROPOSAL.json')) for s in ('H4', 'H6')}
    fixed = intake.store.read(FIXED)
    saved = {v['candidate']['id']: v for s in fixed['systems'].values() for v in s['rows']}
    coverage = []
    for system, packet in packets.items():
        for proposal in packet['rows']:
            cid = proposal['candidate_id']; previous = saved.get(cid); row = rows.get(cid)
            if previous is not None:
                assert a0.p.same(previous['candidate'], proposal['request']['identity']['candidate'])
                fixed_task = fixed['systems'][system]['task']
                fixed_row = next(v for v in fixed_task['candidates'] if v['candidate']['id'] == cid)
                assert a0.p.same(a0.p.identity(fixed_task, fixed_row), proposal['request']['identity'])
            if row is not None:
                assert a0.p.same(row['record']['request'], proposal['request'])
            coverage.append(dict(**{k: deepcopy(proposal[k]) for k in
                ('candidate_id', 'system', 'm', 'T', 'q_RPE', 'q_PF', 'delta', 'R_outer', 'formula', 'prefix', 'K', 'structure')},
                A0_recorded=row is not None, A0_record_source=href if row is not None else None,
                normalization_status=None if row is None else row['normalization_status'],
                A0_phase_radius_N_status=None if row is None else row['record']['phase_radius_N_status'],
                existing_same_candidate_N_saved=previous is not None,
                existing_N_status=None if previous is None else previous['N_status'],
                existing_N_source=None if previous is None else deepcopy(previous['N_source']),
                N_R_comparison_qualification='NOT_ESTABLISHED_MATCHED_R_AWARE_ALLOWANCE_REQUIRED',
                fixed_price_metadata=None if previous is None else {k: deepcopy(previous[k]) for k in
                    ('direct_C_available', 'saved_model_C_available', 'eligible_finite_scenarios',
                     'qualified_expectation_scenarios', 'unvalidated_long_q', 'provider_domain_matches')},
                C_from_A0=None, N_from_A0=None, G=None, scientific_R_adopted=False,
                cross_R_covariance=None, long_q_systematic=None, total_CI=None,
                full_law_expected_price_qualified=False, full_RPE_qualified=False))
    assert len(coverage) == 86 and sum(v['A0_recorded'] for v in coverage) == 24

    # Rebuild C's fixed physical rectangle with existing metadata functions.
    # These are planned C1 requests, not evidence of a C1 scientific execution.
    cmenus = {}
    for system, packet in packets.items():
        selected = [v for v in intake.cells.values() if v['system'] == system]
        Rmap = {}
        for cell in selected:
            Rmap.setdefault(cell['m'], []).append(cell['candidate']['R_outer'])
        cmenus[system] = a0.r.build_menu(packet['base'], Rmap, sampling_law=packet['menu']['sampling_law'])
    describe = fixed_C_describer(ROOT)
    links = []; cells = []
    for cid, cell in intake.cells.items():
        menu = cmenus[cell['system']]
        target = next(v for v in menu['candidates'] if v['candidate']['id'] == cid)
        request = a0.r.request_context(menu, target)
        assert a0.p.same(request['law_binding'], cell['A_request_law_binding'])
        for axis in a0.p.AXES:
            described = describe(menu, target, axis, pipeline=a0.p)
            assert described['join_key'] == cell['C_join_keys'][axis]
            assert a0.p.same(described['identity'], request['identity'])
        if cid in intake.overlap:
            link = intake.for_c1(accepted, cid, request=request,
                                 normalization_source=handoff['scientific_source'])
            assert link['share_eligible'] is True
            assert a0.p.same(link['diagnostic_record'], original[cid])
            links.append(link)
        else:
            try:
                intake.for_c1(accepted, cid, request=request, normalization_source=handoff['scientific_source'])
            except ValueError:
                pass
            else:
                raise AssertionError('Do not fill other C1 cells from A0')
        cells.append(dict(candidate_id=cid, system=cell['system'], m=cell['m'],
            candidate=deepcopy(cell['candidate']), role=cell['role'], intake_lane=cell['intake_lane'],
            A0_diagnostic_shared=cid in intake.overlap, A0_normalization=None if cid not in intake.overlap else 'SEE_C1_LINKS',
            existing_N_saved=cell['N_saved'], existing_C_saved_metadata=bool(cell['C_saved_metadata']),
            C1_execution_evidence_consumed=False, full_law_expected_price_qualified=False,
            long_q_transfer_qualified=False, C=None, N_from_A0=None, G=None))
    assert {v['candidate_id'] for v in links} == {'2d2af6ca830dff14a695', '4269eafc8c0ef8922c23'}
    assert len(cells) == 8 and sum(v['intake_lane'] == 'EXPLICIT_C_CALIBRATION_VALIDATION_ONLY' for v in cells) == 2
    summary = dict(status='PUBLISHED_A0_DIAGNOSTICS_ACCEPTED', publication=accepted['publication'],
        scientific_source=handoff['scientific_source'], adapter_source='529e81fc20969bd8265ae98ba71ad18fa8f3eca7',
        root_status=handoff['status'], completion=accepted['completion'],
        recorded=24, missing=len(accepted['missing_ids']),
        stopped_records=sum(str(v['normalization_status']).startswith('STOP') for v in rows.values()),
        root_STOP=handoff['status'].startswith('PHASE0_STOP'),
        numerically_unqualified=sum(v['normalization_status'] != a0.SCORED for v in rows.values()),
        overflow_displays=sum(v['record'][field]['status'] == 'OVERFLOW_BINARY64_KEEP_LOG' for v in rows.values() for field in ('attenuation', 'shot_inflation')),
        underflow_displays=sum(v['record'][field]['status'] == 'UNDERFLOW_TO_ZERO_KEEP_LOG' for v in rows.values() for field in ('attenuation', 'shot_inflation')),
        diagnostic_share_eligible=sum(v['diagnostic_share_eligible'] for v in rows.values()),
        normalization_evidence_usable=accepted['normalization_evidence_usable'],
        source_files_matched=len(intake.plan['sources']), input_hashes_matched=len(intake.plan['input_hashes']),
        A_menu_retained=len(coverage), A_menu_outside_A0=sum(not v['A0_recorded'] for v in coverage),
        A0_existing_N_candidates=sum(v['A0_recorded'] and v['existing_same_candidate_N_saved'] for v in coverage),
        A0_missing_same_candidate_N=sum(v['A0_recorded'] and not v['existing_same_candidate_N_saved'] for v in coverage),
        C1_cells_retained=len(cells), C1_shared=len(links), C1_not_filled=len(cells)-len(links),
        scientific_recalculation_or_N_C_G=0, R_optimization=0, N_C_G=None,
        scientific_R_adopted=False, full_RPE_qualified=False, mandatory_stop=True)
    assert not any(name in sys.modules for name in ('numpy', 'scipy', 'qiskit', 'trottertracks.pr_r_normalization_phase0', 'trottertracks.pr_r_screening'))
    return {'SUMMARY.json': summary,
            'COVERAGE.json': dict(A_PR_menu=coverage, C1_cells=cells, fixed_metadata_source=FIXED,
                alignment_source=dict(commit=a0.B_ALIGNMENT, path=a0.ALIGNMENT),
                A_menu_unchanged=True, N_C_G=None, scientific_R_adopted=False, full_RPE_qualified=False),
            'C1_LINKS.json': dict(links=links, source=href, C1_execution_evidence_consumed=False,
                C=None, N=None, G=None, scientific_R_adopted=False, full_RPE_qualified=False)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-derived', action='store_true', help='Write fresh B metadata artifacts only')
    args = parser.parse_args(); derived = verify()
    if args.write_derived:
        for name in derived:
            if (OUT / name).exists():
                raise FileExistsError(name)
        for name, value in derived.items():
            (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    else:
        for name, value in derived.items():
            assert a0.p.same(json.loads((OUT / name).read_text()), value), name
    print(json.dumps(derived['SUMMARY.json'], ensure_ascii=False))
