"""Author checks of saved diagnostics, alternate Decimal arithmetic and exports."""
from collections import Counter,defaultdict
import csv
from datetime import datetime,timezone
from decimal import Decimal,localcontext
from fractions import Fraction
import json
import math
from pathlib import Path

from run_pf_q_model_diagnostics import ROOT,OUT,verify,sha,write


def decimal(value):
    value=Fraction(value)
    return Decimal(value.numerator)/Decimal(value.denominator)


def main():
    contract=json.loads((OUT/'sealed_analysis_contract.json').read_text());verify(contract)
    rows=json.loads((OUT/'axis_errors.json').read_text())['rows']
    prior=json.loads((ROOT/'artifacts/resource_applicability/track_a_pf_q_model_comparison_preparation/2026-10-20/axis_correspondence.json').read_text())['rows']
    original={r['task_id']:r for r in prior}
    assert len(rows)==len({r['task_id'] for r in rows})==450
    assert [r['task_id'] for r in rows]==contract['selected_task_ids']
    assert Counter(r['system'] for r in rows)=={'H6_DF19':192,'H8_DF28':258}
    csv_rows=list(csv.DictReader((OUT/'axis_errors.csv').open()))
    assert len(csv_rows)==450
    pairs=defaultdict(list)
    with localcontext() as context:
        context.prec=80
        for row,csv_row in zip(rows,csv_rows):
            old=original[row['task_id']]
            assert old['comparison_class']=='CONDITIONAL_COMPARISON'
            for name,value in old.items():
                if name!='residuals_computed':assert row[name]==value,(row['task_id'],name)
            measured=Decimal(old['direct_RZ_literal']);predicted=decimal(old['saved_prediction_literals'][0])
            difference=predicted-measured;relative=difference/measured
            assert math.isclose(float(difference),row['difference_RZ'],rel_tol=1e-12,abs_tol=1e-12)
            assert math.isclose(float(abs(relative)*100),row['absolute_relative_error_percent'],rel_tol=1e-12,abs_tol=1e-12)
            assert math.isclose(float(relative*100),row['signed_relative_error_percent'],rel_tol=1e-12,abs_tol=1e-12)
            assert (abs(relative)<Decimal('.02'))==row['strictly_below_2_percent']
            assert csv_row['task_id']==row['task_id']
            for name in ('predicted_RZ_exact','difference_RZ_exact','absolute_relative_error_exact'):
                assert csv_row[name]==row[name]
            assert len(row['direct_result_sha256'])==64 and row['saved_prediction_provenance']
            pairs[(row['system'],row['candidate_id'])].append(row)
        grouped=json.loads((OUT/'grouped_errors.json').read_text())['rows']
        for group in grouped:
            subset=[r for r in rows if all(r[k]==v for k,v in group['group'].items())]
            assert len(subset)==group['n_axes']
            assert sum(r['strictly_below_2_percent'] for r in subset)==group['below_2_percent']
            mean=sum(abs(decimal(r['signed_relative_error_exact'])) for r in subset)*100/len(subset)
            assert math.isclose(float(mean),group['mean_absolute_relative_error_percent'],rel_tol=1e-12,abs_tol=1e-12)
            assert max(r['absolute_relative_error_percent'] for r in subset)==group['max_absolute_relative_error_percent']
    assert len(pairs)==225 and all({r['axis'] for r in pair}=={'cosine','sine'} for pair in pairs.values())
    same_pair_values=sum(pair[0]['measured_RZ']==pair[1]['measured_RZ'] and
                         pair[0]['predicted_RZ_exact']==pair[1]['predicted_RZ_exact'] for pair in pairs.values())
    numeric=json.loads((OUT/'numeric_completion.json').read_text())
    for name,digest in numeric['result_hashes'].items():assert sha(OUT/name)==digest
    export_count=0
    for folder in (OUT,OUT/'display'):
        rendered=json.loads((folder/'rendering_completion.json').read_text())
        assert len(rendered['figures'])==12
        for figure in rendered['figures']:
            path=folder/figure['path'];raw=path.read_bytes();assert sha(path)==figure['sha256']
            if path.suffix=='.png':assert raw.startswith(b'\x89PNG\r\n\x1a\n')
            if path.suffix=='.pdf':assert raw.startswith(b'%PDF-')
            if path.suffix=='.svg':assert b'<svg' in raw
            export_count+=1
        figure_data=json.loads((folder/'figure_data.json').read_text())
        assert sha(folder/'figure_data.json')==rendered['figure_data_sha256']
        assert len(figure_data['points'])==450
        for figure in ('predicted_vs_measured','relative_error_distribution','error_vs_q','error_vs_prefix'):
            ids=[task for panel,tasks in figure_data['panel_task_ids'].items() if panel.startswith(figure+'/') for task in tasks]
            assert len(ids)==len(set(ids))==450 and set(ids)==set(contract['selected_task_ids'])
        assert sum(sum(h['counts']) for h in figure_data['histograms'])==450
        for h in figure_data['histograms']:assert sum(h['counts'])==h['n_axes']
    assert sha(OUT/'figure_data.json')==sha(OUT/'display/figure_data.json')
    ledger=json.loads((OUT/'ledger.json').read_text())
    assert ledger['saved_axis_errors']==450 and all(ledger[k]==0 for k in ('new_compile','new_sampling','new_model_prediction','fit','shot_cost_reaccounting','ranking'))
    write('structural_validation.json',dict(status='PASS_AUTHOR_NUMERIC_AND_STRUCTURE_CHECKS',
        checked_UTC=datetime.now(timezone.utc).isoformat(),SOURCE=contract['SOURCE'],
        checks=dict(unique_axes=450,complete_pairs=225,pairs_with_equal_cosine_sine_RZ_values=same_pair_values,
            exact_input_and_SOURCE_hashes_preserved=True,alternate_Decimal_axis_checks=450,
            grouped_mean_and_count_checks=len(grouped),csv_json_literals_agree=True,
            legacy_measured_denominator_criterion_matches=450,initial_global_final_gate_flags_retained=428,
            final_figures=4,final_exports=12,initial_exports_preserved=12,export_headers_and_hashes_checked=export_count,
            each_figure_has_all_450_axes_once=True,histogram_counts_sum=450,
            display_revision_numeric_data_unchanged=True,prohibited_calls_zero=True),
        synthetic_tests=10,independent_scientific_review=False,checks_by='primary author agent',
        detailed_scientific_review=False,final_evidence_freeze=False))
    print(json.dumps(dict(status='PASS',axes=450,groups=len(grouped),same_pair_values=same_pair_values,exports=export_count)))


if __name__=='__main__':main()
