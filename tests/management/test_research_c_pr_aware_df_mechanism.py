"""Invented saved packets only; no real H4 numerical reanalysis."""
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import c_pr_aware_df_mechanism as m


def invented():
    b = dict(C1_SOURCE=dict(execution_commit='b1c735da394d4b0a444cbc618ead9781ef4d9134'),
        C1_STOP=dict(scientific_units=164, failed=False, retry=0, resume=0, mandatory_stop=True),
        A_result=dict(status='STAGE_A_PROXY_AND_DF_PASS', sector={}, fit_diagnostics={}),
        B1_result=dict(status='B1_MECHANISM_EVALUATION_COMPLETE', arms={}),
        B2_result=dict(status='B2_FINITE_BIT_NATIVE_CALIBRATION_COMPLETE', arms={}),
        C1_result=dict(status='C1_SAVED_SCALAR_SCENARIOS_COMPLETE', assumed_scenarios={},
            evidence_lane={a: dict(shots=None, status='UNRESOLVED') for a in m.ARMS},
            CA_achieved=None, shots_guaranteed=None, total_resources_guaranteed=None, cost_support=None))
    for lane in m.LANES:
        b['C1_result']['assumed_scenarios'][lane] = {}
    for arm, N, price in [('baseline', 10, 20), ('proposed', 5, 10)]:
        target = arm+':artificial-'+arm
        b[arm+'_costs'] = [dict(m=i, axis=axis, target_id=target, scope=m.COST_SCOPE,
            kernel_occurrences=4*(1 << i), depth_semantics='scheduled_serial_sum',
            long_round_coverage=True, classical_cost_samples=32,
            simultaneous_interval_qualified=False, lower=None, upper=None,
            mean={k: 4*(1 << i)*price+shell for k in m.METRICS})
            for i in range(13) for axis, shell in [('X', 2), ('Y', 3)]]
        plans = [dict(m=i, shots=N, target_id=target, status='CONDITIONAL_ROUND_PASS') for i in range(13)]
        costs = b[arm+'_costs']
        totals = {k: sum(N*x['mean'][k] for x in costs) for k in m.METRICS}
        for lane in m.LANES:
            b['C1_result']['assumed_scenarios'][lane][arm] = dict(arm=arm, lane=lane, target_id=target,
                qualification=dict(kind='ASSUMED_SCENARIO', all_law_bound_qualified=False),
                rounds=deepcopy(plans), conditional_energy_assessment=dict(same_accuracy=True),
                resources=dict(totals=totals),
                calibration_uncertainty=dict(kernel_reuse_weight=sum(2*N*4*(1 << i) for i in range(13))))
        b['A_result']['sector'][arm] = dict(sector_error=.00001)
        b['A_result']['fit_diagnostics'][arm] = dict(diagnostic=dict(exact_tail_mass_Ha=1.))
        b['B1_result']['arms'][arm] = dict(saved_DF_bound_Ha=.00001,
            rounds=[dict(m=i, reference=dict(pf_operator_error=.001,
                rte_corrected_operator_error=.002, attenuation=.9)) for i in range(13)])
        b['B2_result']['arms'][arm] = dict(samples=32,
            sample_mean_standard_error={k: 2. if arm=='baseline' else 1. for k in m.METRICS})
    return b


def fixture(root):
    b = invented()
    records = {key: key+'.json' for key in b}
    old_records = {key: records[key] for key in ('A_result', 'B1_result', 'B2_result')}
    for key in old_records:
        m.io.save(root/records[key], b[key])
    prior = dict(records=old_records, saved_inputs={p: m.io.sha(root/p) for p in old_records.values()},
        final_exports={a: a+'.npz' for a in m.ARMS},
        export_hashes={a+'.npz': 'artificial-'+a for a in m.ARMS})
    records['C1_contract'] = 'C1_contract.json'
    m.io.save(root/'C1_contract.json', prior)
    for key in b:
        if key not in old_records:
            m.io.save(root/records[key], b[key])
    (root/'synthetic-source.py').write_text('# artificial source identity\n')
    m.io.save(root/'SOURCE.json', dict(files={'synthetic-source.py': m.io.sha(root/'synthetic-source.py')}))
    c = dict(scope=m.SCOPE, implementation=m.VERSION, base_commit=m.BASE, status='UNAPPROVED',
        executable=True, conditions=m.CONDITIONS, lanes=list(m.LANES), call_limits=m.LIMITS,
        maximum_logical_units=m.MAX_CALLS, fresh_output=m.FRESH, retry=0, resume=0,
        reconciliation=m.RECONCILIATION, source_manifest='SOURCE.json',
        source_manifest_sha256=m.io.sha(root/'SOURCE.json'), records=records,
        saved_inputs={p: m.io.sha(root/p) for p in records.values()})
    m.io.save(root/'contract.json', c)
    return c


def four(b, lane=m.LANES[0]):
    p = m.attach(b, lane)
    return {a[0]+z[0]: m.corner(b, p, a, z) for a in m.ARMS for z in m.ARMS}


def test_passive_import_and_20_calls():
    code = "import sys;sys.path.insert(0,'src');from trottertracks import c_pr_aware_df_mechanism as m;assert m.MAX_CALLS==20;assert not any(x in sys.modules for x in ('numpy','scipy','qiskit','ffsim','flint'))"
    subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, check=True)


def test_known_four_corners_and_shell_once():
    out = four(invented())
    assert {k: v['rounds'][0]['weighted_cost']['rz_count'] for k, v in out.items()} == dict(bb=1650, bp=850, pb=825, pp=425)
    assert out['bp']['shots_target_id'] != out['bp']['price_target_id']
    assert out['bp']['physically_realized_hybrid'] is False


def test_known_ordered_paths_interaction_and_symmetric_allocation():
    x = m.decompose(four(invented()))
    assert x['rounds'][0]['contribution']['rz_count'] == dict(net_saving=1225,
        price_first=800, shots_after_price=425, shots_first=825, price_after_shots=400,
        interaction=400, symmetric_price=600, symmetric_shots=625)
    assert x['causal_attribution'] is False and x['guaranteed_advantage'] is None
    assert x['dominant_positive_net_RZ_component']=='shots'
    for k in m.METRICS:
        assert sum(x['fixed_round_groups'][g][k]['net_saving'] for g in ('m0_to_9','m10_to_12')) == x['totals'][k]['net_saving']
        assert x['cumulative'][-1]['net_saving'][k] == x['totals'][k]['net_saving']


@pytest.mark.parametrize('mode', ['same_shots', 'same_prices', 'same_both', 'worse_proposed'])
def test_components_zero_and_negative_results_are_not_forced_positive(mode):
    b = invented()
    if mode in ('same_shots','same_both'):
        for lane in m.LANES:
            for p in b['C1_result']['assumed_scenarios'][lane]['proposed']['rounds']:
                p['shots']=10
    if mode in ('same_prices','same_both','worse_proposed'):
        for x, y in zip(b['proposed_costs'],b['baseline_costs']):
            x['mean']={k: y['mean'][k]*(4 if mode=='worse_proposed' else 1) for k in m.METRICS}
    x=m.decompose(four(b))['totals']['rz_count']
    if mode=='same_shots': assert x['symmetric_shots']==0
    if mode=='same_prices': assert x['symmetric_price']==0
    if mode=='same_both': assert x['net_saving']==0
    if mode=='worse_proposed': assert x['net_saving']<0


def test_shared_mean_uncertainty_and_covariance():
    b=invented();p=m.attach(b,m.LANES[0]);u=m.uncertainty(b,p)
    wb,wp=u['W']['baseline'],u['W']['proposed'];x=u['conditional_SE']['rz_count']
    assert x['net_saving_SE']==math.hypot(2*wb,wp)
    assert x['symmetric_price_SE']==.5*(wb+wp)*math.sqrt(5)
    assert math.isclose(x['symmetric_price_SE']**2+x['symmetric_shots_SE']**2+2*x['price_shots_covariance'],x['net_saving_SE']**2)
    assert u['independent_round_calibrations']==1 and u['confidence_interval'] is None


def test_unknown_accuracy_stays_unresolved():
    b=invented();lane=m.LANES[0]
    b['C1_result']['assumed_scenarios'][lane]['baseline']['conditional_energy_assessment']['same_accuracy']=False
    p=m.attach(b,lane);out=m.corner(b,p,'baseline','proposed')
    assert out['totals'] is None
    assert m.decompose({k:out for k in ('bb','bp','pb','pp')})['attribution'] is None
    assert m.uncertainty(b,p)['conditional_SE'] is None


@pytest.mark.parametrize('bad', [None,-1,float('nan'),float('inf'),True])
def test_invalid_prices_are_not_zero_filled(bad):
    b=invented();b['proposed_costs'][0]['mean']['rz_count']=bad
    with pytest.raises(ValueError,match='FINITE_NONNEGATIVE'):
        m.corner(b,m.attach(b,m.LANES[0]),'baseline','proposed')


@pytest.mark.parametrize('bad', [None,0,1.5,True])
def test_missing_or_invalid_shots_are_not_recomputed(bad):
    b=invented();b['C1_result']['assumed_scenarios'][m.LANES[0]]['baseline']['rounds'][0]['shots']=bad
    with pytest.raises(ValueError,match='SAVED_SHOT_PLAN'):
        m.attach(b,m.LANES[0])


def test_preflight_does_not_decode_saved_numeric_packets(tmp_path,monkeypatch):
    c=fixture(tmp_path);original=m.io.read
    def safe(path):
        assert Path(path).name in ('contract.json','SOURCE.json')
        return original(path)
    monkeypatch.setattr(m.io,'read',safe)
    assert m.preflight(tmp_path,tmp_path/'contract.json')==c
    (tmp_path/c['records']['B1_result']).write_text('changed')
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):
        m.preflight(tmp_path,tmp_path/'contract.json')


@pytest.mark.parametrize('fault', ['arm','missing_round','duplicate_axis','coherent_repeat','paired_samples','source','qualification'])
def test_saved_provenance_and_policy_failures_stop(tmp_path,fault):
    c=fixture(tmp_path);b=m.load_bundle(tmp_path,c)
    if fault=='arm': b['C1_result']['assumed_scenarios'][m.LANES[0]]['baseline']['target_id']='proposed:wrong'
    if fault=='missing_round': b['baseline_costs'].pop()
    if fault=='duplicate_axis': b['baseline_costs'][0]['axis']='Y'
    if fault=='coherent_repeat': b['baseline_costs'][0]['kernel_occurrences']=1
    if fault=='paired_samples': b['B2_result']['arms']['baseline']['samples']=64
    if fault=='source': b['C1_SOURCE']['execution_commit']='wrong'
    if fault=='qualification': b['C1_result']['CA_achieved']=True
    with pytest.raises(ValueError):m.identity_gate(b,c)


@pytest.mark.parametrize('fault',[None,'identity','closure'])
def test_artificial_end_to_end_calls_partial_STOP_and_no_resume(tmp_path,monkeypatch,fault):
    c=fixture(tmp_path);path=tmp_path/'contract.json'
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'SYNTHETIC_EXECUTION\n')
    if fault=='identity': monkeypatch.setattr(m,'identity_gate',lambda *a:(_ for _ in ()).throw(ValueError('SYNTHETIC_IDENTITY_FAIL')))
    if fault=='closure': monkeypatch.setattr(m,'reconcile',lambda *a:(_ for _ in ()).throw(ValueError('SYNTHETIC_CLOSURE_FAIL')))
    before={p:(tmp_path/p).read_bytes() for p in c['saved_inputs']}
    if fault:
        with pytest.raises(ValueError):m.run(tmp_path,path)
    else:m.run(tmp_path,path)
    out=tmp_path/m.FRESH;s=m.io.read(out/'STOP.json')
    assert s['retry']==s['resume']==0 and s['mandatory_stop']
    if fault:
        assert s['failed'] and not(out/'result.json').exists()
        assert s['scientific_units']<20
    else:
        assert s['scientific_units']==20 and s['call_counts']==m.LIMITS
        x=m.io.read(out/'result.json')
        assert x['CA_achieved'] is x['shots_guaranteed'] is x['guaranteed_resource_superiority'] is None
        assert x['evidence_lane']['baseline']['shots'] is None
        assert len(m.io.read(out/'ideal_exact_native_bb.json')['rounds'])==13
    assert all((tmp_path/p).read_bytes()==data for p,data in before.items())
    with pytest.raises(ValueError,match='FRESH'):m.run(tmp_path,path)


def test_existing_endpoint_disagreement_is_FAIL():
    b=invented();packets={lane:m.attach(b,lane) for lane in m.LANES}
    corners={lane:four(b,lane) for lane in m.LANES}
    b['C1_result']['assumed_scenarios'][m.LANES[0]]['baseline']['resources']['totals']['rz_count']+=1.e10
    with pytest.raises(ValueError,match='CLOSURE_FAIL'):m.reconcile(b,packets,corners)
