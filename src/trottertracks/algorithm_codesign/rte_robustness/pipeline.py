"""48 ledgers from immutable saved inputs after authorization. No generators."""
from fractions import Fraction as F
import importlib.util
import json
from pathlib import Path

from ..g8_bounds import log_upper
from ..rte_development.evidence import digest
from .model import whole_budget, equal_budget, event_prices, row_account, assert_legacy_cost, factorial_effects, METRICS
from .acquisition import acquire, check_saved, validate_inventory

D1='artifacts/track_b_rte_development_d1_result/2026-10-13/v1'
D2='artifacts/track_b_d2_saved_boundary_result/2026-10-13/v1'


def read_identity(root, reference):
    p=Path(root)/reference['path']
    if p.suffix.lower()=='.npz' or digest(p) != reference['sha256']:
        raise PermissionError('fixed saved input identity changed')
    return json.loads(p.read_text())


def check_certificate(certificate, index, events_sha):
    if certificate.get('all_rows_no_R_cancellation') is not True:
        raise PermissionError('R cancellation not certified')
    rows=[v for v in certificate['rows'] if v['row']==index]
    if (len(rows)!=1 or rows[0]['source_sha256']!=events_sha
        or rows[0]['no_R_can_be_cancelled'] is not True
        or rows[0]['R_containing_intersections']!=0):
        raise PermissionError('literal-only boundary price certificate mismatch')


def evaluate_saved_row(row, bindings, boundary, caches, c, log_constant, check=lambda:None):
    z, per, worst=event_prices(bindings,caches,check)
    if z != F(row['reference_law_diagnostic_only']['acceptance']):
        raise ArithmeticError('saved law changed')
    for k in METRICS:
        if per['P0'][k] != F(row['per_step_trial_native'][k]) or worst['P0'][k] != row['worst_single_step_event_native'][k]:
            raise ArithmeticError('old price anchor changed')
    if boundary is not None:
        if boundary['arm']!=row['arm'] or boundary['budget_unchanged']!=row['budget']:
            raise PermissionError('D2 budget/arm mismatch')
        for k in METRICS:
            old_base=2*z*per['P0'][k]
            saved=F(boundary['cancellation_saving_per_whole_trial'][k])
            if old_base != F(boundary['additive_per_whole_trial'][k]) or old_base-saved != F(boundary['after_per_whole_trial'][k]):
                raise ArithmeticError('fixed D2 per-trial anchor changed')
    out={}
    for g in ('G0','G1'):
        budget=whole_budget(row['production_step_bounds'],row['x'],row['steps'],g,c,log_constant)
        if g=='G0':equal_budget(budget,row['budget'])
        for p in ('P0','P1'):
            out[g+'_'+p]=row_account(row,budget,per[p],worst[p],z,boundary)
    assert_legacy_cost(out['G0_P0'],row,boundary)
    return out


def comparisons(root, all_rows):
    # Reuse the independently tested scalar algebra, not its registered main().
    path=Path(root)/'scripts/tracks/algorithm_codesign/analyze_d1_saved_resources.py'
    spec=importlib.util.spec_from_file_location('d3_fixed_affine_math',path)
    math=importlib.util.module_from_spec(spec);spec.loader.exec_module(math)
    def envelope(rows,metric):
        raw=math.sensitivity(rows,metric)
        regions=[]
        for v in raw['open_interval_rankings']:
            minimum=v['minimum_envelope_arms']
            if regions and regions[-1]['minimum_arms']==minimum:
                regions[-1]['open_hi']=v['open_hi']
            else:
                regions.append(dict(open_lo=v['open_lo'],open_hi=v['open_hi'],
                                    minimum_arms=minimum))
        # All exact affine curves remain in the 48 resource rows. Avoid
        # duplicating their rational differences in each comparison report.
        return dict(metric=metric,h0_ranking=math.ranking(rows,metric,F(0)),
                    minimum_open_regions=regions,
                    minimum_switch_endpoint_ties=[
                        dict(h=v['open_lo'],minimum_arms=math.ranking(rows,metric,v['open_lo'])[0])
                        for v in regions[1:]],
                    all_affine_inputs_retained_in_resource_rows=True,
                    h_domain='nonnegative coordinate-specific prep/readout cost')
    values={}
    for ledger,rows in all_rows.items():
        values[ledger]=dict(
            all12={k:envelope(rows,k) for k in METRICS},
            fixed_r={str(r):{k:envelope([v for v in rows if v['steps']==r],k) for k in METRICS} for r in (1,2)},
            primary_equal_r_choice={k:envelope([v for v in rows if v['arm'] in ('partial_return_tail','closed_P5_full')],k) for k in METRICS},
            quantum_pareto=math.pareto_regions(rows),
            physical_h_constraint='h_1Q >= h_T; formal independent nonnegative domains not device-specific',
            classical_online='NOT_ACQUIRED',method_adoption=False)
    return values


def run(root,c,identity,writer,guard):
    root=Path(root)
    inventory=read_identity(root,c['acquisition']['key_inventory']);validate_inventory(inventory,c)
    certificate=read_identity(root,c['native_policy']['R_not_cancelled_certificate'])
    old=json.loads((root/D1/'synthesis_cache.json').read_text())
    ratios={v['ratio'] for v in inventory['keys']}
    if set(old)!=ratios:raise PermissionError('old cache/inventory coverage mismatch')
    for ratio in sorted(ratios,key=F):check_saved(old[ratio],ratio,'1/1000000')
    writer.write('frozen_inventory.json',inventory)
    new,receipts=acquire(root,c,identity,inventory,writer,guard)
    writer.write('synthesis_cache_new.json',new)
    writer.write('synthesis_process_receipts.json',receipts)
    caches={'P0':old,'P1':new}
    log=log_upper(2/F(c['error_accounting']['alpha_axis']))
    all_rows={v:[] for v in ('G0_P0','G1_P0','G0_P1','G1_P1')}
    count=0
    for index,entry in enumerate(inventory['rows']):
        if entry['row']!=index:raise PermissionError('row order changed')
        guard.check()
        row=json.loads((root/D1/f'rows/row_{index:02d}.json').read_text())
        if (row['arm']!=entry['arm'] or row['steps']!=entry['steps'] or row['degree']!=5
            or F(row['x'])!=F(c['input']['tau'])/row['steps'] or row['state']!=c['input']['state']):
            raise PermissionError('registered input row identity mismatch')
        ref=dict(row['events']);ref['path']=D1+'/'+ref['path']
        bindings=read_identity(root,ref);count+=len(bindings)
        if count>c['caps']['event_metadata_reads_max']:
            raise RuntimeError('saved event read cap')
        boundary=None
        if row['steps']==2:
            check_certificate(certificate,index,ref['sha256'])
            boundary=json.loads((root/D2/f'rows/row_{index:02d}.json').read_text())
        rows=evaluate_saved_row(row,bindings,boundary,caches,c,log,guard.check)
        for ledger,value in rows.items():
            value.update(row_index=index,ledger=ledger,guarantee=ledger[:2],price_condition=ledger[3:],
                         new_sequence_precision='1/10000000' if ledger.endswith('P1') else '1/1000000',
                         old_input_row_path=D1+f'/rows/row_{index:02d}.json',
                         events_sha256=ref['sha256'])
            writer.write(f'rows/{ledger}/row_{index:02d}.json',value)
            all_rows[ledger].append(value)
        del bindings, row, boundary
    if count!=1890 or sum(len(v) for v in all_rows.values())!=48:
        raise ArithmeticError('fixed row/event coverage incomplete')
    writer.write('factorial_effects.json',factorial_effects(all_rows))
    compare=comparisons(root,all_rows)
    for ledger,value in compare.items():writer.write(f'comparisons/{ledger}.json',value)
    writer.write('resource_map.json',dict(rows=all_rows,**identity,rows_count=48,event_metadata_reads=count,
                 registered_pair_evaluations=0,new_synthesis_calls=36,retries=0,
                 all_native_error_accounts='1/1000000',four_ledger_joint_confidence_claim=False,
                 classification='D3_GUARANTEE_PRICE_MAP_COMPLETE_AWAITING_GPT_REVIEW',
                 quantum_shots=0,provider_or_signal_matrix_evaluations=0,
                 strict_guard_2x2_key_checks=36,mandatory_STOP=True,
                 method_adoption=False,next_science_authorized=False))
    return dict(rows=48,new_synthesis_calls=36,event_metadata_reads=count,registered_pair_evaluations=0)
