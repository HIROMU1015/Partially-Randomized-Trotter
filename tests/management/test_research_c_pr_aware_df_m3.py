"""M3 invented matrices and scalar/mock E2E only; never decode saved H4."""
from copy import deepcopy
from fractions import Fraction
import math
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_pr_aware_df_m3 as m


def artificial():
    from test_research_c_pr_aware_df_stage_b2 import artificial as make
    return make(n=4,rank=8,role='proposed')


def test_passive_import_and_fixed_scope():
    code="import sys;sys.path.insert(0,'src');from trottertracks import c_pr_aware_df_m3 as m;assert m.MAX_CALLS==471;assert not any(x in sys.modules for x in ('numpy','scipy','qiskit','ffsim'))"
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)
    assert [sum(x.values()) for x in (m.A_LIMITS,m.B_LIMITS,m.C_LIMITS)]==[61,325,84]
    assert m.POLICY['r']==64 and m.POLICY['s']==1 and m.SAMPLING['samples']==32


def test_math_tau_A_sr_and_independent_namespaces():
    from trottertracks import c_pr_aware_df_stage_b as b
    from trottertracks import c_pr_aware_df_stage_b2 as old
    arm,_=artificial();tail,config,dist=b.configure(arm,Fraction(1,10),64,0)
    ot,oc,od=b.configure(arm,Fraction(1,40),16,0)
    rows=[dict(reference=dict(attenuation=b.reference._legacy('rte').finite_rte_attenuation(oc,tail_evolutions=4*(1<<i)))) for i in range(13)]
    report=m.algebra_gate(arm,tail,config,rows)
    assert report['status']=='PASS' and report['mathematical_tau_equal']
    assert all(x['new_random_events']==x['old_random_events'] for x in report['rounds'])
    seeds=set()
    for i in range(32):
        events,rec=m.sample_trajectory(arm,tail,dist,config,i,'ARTIFICIAL_SOURCE')
        _,oldrec=old.sample_trajectory(arm,ot,od,oc,i,'ARTIFICIAL_SOURCE')
        assert len(events)==64 and len(old.sample_trajectory(arm,ot,od,oc,i,'ARTIFICIAL_SOURCE')[0])==16
        assert rec['seed']!=oldrec['seed'] and rec['seed'] not in seeds
        assert not rec['paired'] and not rec['old_cohort_pooled'];seeds.add(rec['seed'])
    assert m.sample_trajectory(arm,tail,dist,config,0,'ARTIFICIAL_SOURCE')==m.sample_trajectory(arm,tail,dist,config,0,'ARTIFICIAL_SOURCE')
    with pytest.raises(ValueError,match='64_EVENTS'):m.sample_trajectory(arm,ot,od,oc,0,'S')
    with pytest.raises(ValueError,match='INDEX'):m.sample_trajectory(arm,tail,dist,config,32,'S')
    rows[12]['reference']['attenuation']-=.01
    with pytest.raises(ValueError,match='ATTENUATION'):m.algebra_gate(arm,tail,config,rows)


def test_artificial64_event_full512_phase_adjoint_XY():
    from threadpoolctl import threadpool_limits
    from trottertracks import c_pr_aware_df_stage_b as b
    from trottertracks import c_pr_aware_df_stage_b2 as b2
    from trottertracks.c1_h4_native_fair import compile_checked
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    from test_research_c_pr_aware_df_stage_b2 import full_context
    arm,p=artificial();G,ctx=full_context(arm,p)
    # Cached head must match the NEW h, not the old test helper's h1/40.
    car=b.stage1.car_bilinears(8);ctx=b2.reference_context(arm,G,car,Fraction(1,10))
    tail,config,dist=b.configure(arm,Fraction(1,10),64,0)
    events,_=m.sample_trajectory(arm,tail,dist,config,0,'ARTIFICIAL_SOURCE')
    with threadpool_limits(limits=1):
        ref=b2.trajectory_reference(arm,ctx,Fraction(1,10),events)
        np.testing.assert_allclose(ref,b.trajectory_reference(arm,G,ctx['atoms'],Fraction(1,10),events,config),atol=1e-12)
        raw=b.build_kernel(arm,Fraction(1,10),events,config)
        compiled=compile_checked(raw,CompilerProfile());R,target,_=b2.raw_gate(raw,ref)
        C,_=b2.compiled_gate(compiled,R,target);adj,report=b2.adjoint_pair_gate(raw,compiled,R,C,target)
        assert len(R)==512 and set(adj)=={'raw','compiled'}
        shells={}
        for label in ('head','X','Y'):
            q=b2.shell_build(8,label);shells[label],_=b2.shell_assessment(q,compile_checked(q,CompilerProfile()),label)
        psi=np.zeros(256,complex);psi[0]=1
        assert b2.XY_gate(R,C,ref,psi,shells)['status']=='PASS'
        wrong=raw.copy();wrong.p(.001,8)
        with pytest.raises(ValueError):b2.raw_gate(wrong,ref)


def a_fixture(root):
    from scipy.linalg import eigh
    from trottertracks import c_pr_aware_df_stage_b as b
    from trottertracks import c_pr_aware_df_stage_b_runner as loader
    from trottertracks.c1_molecule_extension import SectorCAR
    from trottertracks.pr_dictionary_validation.prototype import atom_record
    arm,p=artificial();ctx=SectorCAR(4);G=ctx.generators(**p,J=arm.J,U=arm.U);H=sum(G);ev,U=eigh(H)
    ix=b.stage1.bit_reverse_indices(8)[ctx.sector.basis_indices]
    for path,data in [('input.npz',p),('JU.npz',dict(J=arm.J,U=arm.U)),('ref.npz',dict(generators=np.array(G),H=H,full_input_indices=ix)),('ground.npz',dict(energy=ev[0],sector_state=U[:,0]))]:np.savez(root/path,**data)
    c=dict(input='input.npz',exports={'proposed':'JU.npz'},references={'proposed':'ref.npz'},grounds={'proposed':'ground.npz'},original_input_sha256='ARTIFICIAL',saved_inputs={'JU.npz':m.io.sha(root/'JU.npz')})
    tail,config,dist=b.configure(arm,Fraction(1,40),16,0);old=b.kernels(arm,G,Fraction(1,40),config)
    rows=[dict(m=i,reference=b.reference.reference_round(old,(H,Fraction(1,10)*(1<<i)),repetitions=4*(1<<i))) for i in range(13)]
    saved=dict(B1=dict(arms=dict(proposed=dict(rounds=rows,saved_DF_bound_Ha=1e-8))),
        proposed_ground=dict(energy_Ha=float(ev[0]),alias_margin=1.6),
        proposed_tail=dict(identity=arm.plan.extracted_identity,lambda_R=tail.lambda_r,atoms=216,
            atoms_catalog=[atom_record(x) for x in arm.plan.atoms],components=[dict(probability=x.probability) for x in tail.components]))
    return c,saved


def test_artificial_sector_A_all13_and_saved_identity(tmp_path,monkeypatch):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):c,saved=a_fixture(tmp_path)
    monkeypatch.setattr(m,'evidence',lambda *args:saved);out=tmp_path/'A';out.mkdir()
    call=m.integrated.Calls(out,m.LIMITS,m.MAX_CALLS)
    with threadpool_limits(limits=1):a=m.stage_a(tmp_path,c,out,call)
    assert a['result']['status']=='POTENTIALLY_FEASIBLE'
    assert call.counts==m.A_LIMITS and len(a['result']['rounds'])==13
    assert all(x['feasibility']['status']=='POTENTIALLY_FEASIBLE' for x in a['result']['rounds'])
    assert a['result']['synthesis_bound'] is None


def c_fixture():
    from trottertracks import c_pr_aware_df_stage_b2 as b2
    from trottertracks import c_pr_aware_df_m2 as m2
    metrics=[{k:100+i for k in m.accuracy.METRICS} for i in range(32)]
    calibration=b2.calibration(metrics)
    cost=dict(totals={k:1e12 for k in m.accuracy.METRICS},cost_scope='serial_native_kernel_composition_without_state_preparation')
    shells={name:dict(metrics={k:value for k in m.accuracy.METRICS}) for name,value in [('head',2),('X',3),('Y',4)]}
    saved=dict(shell_reports=shells,C1=dict(assumed_scenarios={lane:dict(proposed=dict(resources=cost)) for lane in ('ideal_exact_native','assumed_raw_signal_budget')}),M2=dict(candidates=dict(proposed=dict(cost=cost))))
    a=dict(result=dict(status='POTENTIALLY_FEASIBLE',saved_DF_bound_Ha=1e-8,
        rounds=[dict(m=i,reference=dict(pf_operator_error=.0001,rte_corrected_operator_error=.0002,attenuation=.95)) for i in range(13)]),saved=saved)
    b=dict(result=dict(status='M3_NATIVE_32_COMPLETE'),metrics=metrics,calibration=calibration)
    c=dict(exports={'proposed':'JU'},saved_inputs={'JU':'ARTIFICIAL'})
    return a,b,c


def test_B32_dispatch_QPY_full_operator_gates_and_call_accounting(tmp_path,monkeypatch):
    from qiskit import QuantumCircuit,qpy
    from threadpoolctl import threadpool_limits
    from trottertracks import c_pr_aware_df_stage_b as base
    from trottertracks import c_pr_aware_df_stage_b2 as b2
    with threadpool_limits(limits=1):c,saved=a_fixture(tmp_path)
    monkeypatch.setattr(m,'evidence',lambda *args:saved)
    ao=tmp_path/'A';ao.mkdir();call=m.integrated.Calls(tmp_path,m.LIMITS,m.MAX_CALLS)
    with threadpool_limits(limits=1):a=m.stage_a(tmp_path,c,ao,call)
    # Dispatch 32 independent invented trajectories. The separate 64-event
    # full512 test covers the real native kernel; here identity circuits keep
    # orchestration coverage independent of compiler runtime.
    monkeypatch.setattr(base,'build_kernel',lambda *args,**kw:QuantumCircuit(9))
    monkeypatch.setattr(b2,'reference_context',lambda *args:None)
    monkeypatch.setattr(b2,'law_gate',lambda *args:dict(status='PASS',synthetic=True))
    monkeypatch.setattr(b2,'trajectory_reference',lambda *args:np.eye(256,dtype=complex))
    c['source_manifest_sha256']='ARTIFICIAL_SOURCE';c['shells']={};saved['shell_reports']={}
    for label in ('head','X','Y'):
        qc=b2.shell_build(8,label);c['shells'][label]={}
        for kind in ('raw','compiled'):
            name=label+'_'+kind+'.qpy';c['shells'][label][kind]=name
            with (tmp_path/name).open('xb') as f:qpy.dump(qc,f)
        saved['shell_reports'][label]=dict(metrics={k:1 for k in m.accuracy.METRICS})
    saved['B2']=dict(arms=dict(proposed=dict(sample_mean={k:1 for k in m.accuracy.METRICS},
        sample_mean_standard_error={k:.1 for k in m.accuracy.METRICS})))
    bo=tmp_path/'B';bo.mkdir();before=dict(call.counts)
    with threadpool_limits(limits=1):r=m.stage_b(tmp_path,c,bo,call,a)
    assert {k:call.counts[k]-before.get(k,0) for k in m.B_LIMITS}==m.B_LIMITS
    assert r['result']['samples']==32 and r['result']['compile_count']==32
    assert len(list((bo/'circuits').glob('*.qpy')))==128
    assert len(list((bo/'samples').glob('*_metrics.json')))==32
    seeds=[m.io.read(p)['seed'] for p in (bo/'samples').glob('*_trajectory.json')]
    assert len(set(seeds))==32


def test_mock_A_B_actual_C_end_to_end32_prices13rounds(tmp_path,monkeypatch):
    a,b,c=c_fixture();monkeypatch.setattr(m,'stage_a',lambda *args:a);monkeypatch.setattr(m,'stage_b',lambda *args:b)
    call=m.integrated.Calls(tmp_path,m.LIMITS,m.MAX_CALLS);result=m.execute(tmp_path,c,tmp_path,call)
    assert result['status']=='M3_A_B_C_COMPLETE' and result['mandatory_stop']
    lanes=result['C']['lanes'];assert lanes['evidence']['resources']['totals'] is None
    assert all(p['shots'] is None for p in lanes['evidence']['rounds'])
    assert call.counts['round_accuracy']==39 and call.counts['serial_cost']==26
    costs=m.io.read(tmp_path/'C/serial_costs26.json')
    assert costs[0]['kernel_occurrences']==1 and costs[-1]['kernel_occurrences']==4096
    assert costs[0]['mean']['rz_count']==115.5+5
    for name in ('ideal_exact_native','assumed_raw_signal_budget'):
        x=lanes[name];W=sum(2*p['shots']*(1<<p['m']) for p in x['rounds'])
        assert x['calibration_uncertainty']['kernel_reuse_weight']==W
        assert x['calibration_uncertainty']['weighted_SE']['rz_count']==W*b['calibration']['sample_mean_standard_error']['rz_count']
        assert x['conditional_energy_assessment']['same_accuracy'] and x['resources']['totals']
        assert x['qualification']['kind']=='ASSUMED_SCENARIO' and x['actual_CA_achieved'] is None


@pytest.mark.parametrize('status',['INFEASIBLE','UNRESOLVED'])
def test_A_nonpass_prevents_B_C_and_preserves_all13(tmp_path,monkeypatch,status):
    a,_,c=c_fixture();a['result']['status']=status
    monkeypatch.setattr(m,'stage_a',lambda *args:a)
    def forbidden(*args):raise AssertionError('B/C must not run')
    monkeypatch.setattr(m,'stage_b',forbidden);monkeypatch.setattr(m,'stage_c',forbidden)
    call=m.integrated.Calls(tmp_path,m.LIMITS,m.MAX_CALLS);r=m.execute(tmp_path,c,tmp_path,call)
    assert len(r['A']['rounds'])==13 and r['stages']['B']==r['stages']['C']=='NOT_RUN_A_GATE'
    assert r['mandatory_stop'] and sum(call.counts.values())==1


def test_calibration_independent_delta_prices_and_no_support():
    from trottertracks import c_pr_aware_df_stage_b2 as b2
    rows=[{k:10+i for k in m.accuracy.METRICS} for i in range(32)];new=b2.calibration(rows);old=deepcopy(new)
    x=m.delta_cost_comparison(new,old)
    assert x['new_over_old_delta']['rz_count']==.25
    se=new['sample_mean_standard_error']['rz_count']
    assert x['difference_SE']['rz_count']==math.hypot(se,4*se)
    assert x['cost_support'] is x['ideal_law_mean'] is None and x['paired'] is False


def passive(root):
    from trottertracks.c_pr_aware_df_stage_c import SCENARIOS
    (root/'opaque.npz').write_bytes(b'no scientific payload')
    m.io.save(root/'source.json',dict(files={}))
    c=dict(scope=m.SCOPE,implementation=m.VERSION,base_commit=m.BASE,policy=m.POLICY,target=m.TARGET,
        allowances=m.ALLOWANCES,sampling=m.SAMPLING,scenarios=list(SCENARIOS),stage_call_limits=dict(A=m.A_LIMITS,B=m.B_LIMITS,C=m.C_LIMITS),
        call_limits=m.LIMITS,maximum_logical_units=m.MAX_CALLS,fresh_output=m.FRESH,retry=0,resume=0,
        source_manifest='source.json',source_manifest_sha256=m.io.sha(root/'source.json'),saved_inputs={'opaque.npz':m.io.sha(root/'opaque.npz')},packages={})
    m.io.save(root/'contract.json',c);return c


def test_preflight_mismatch_and_fresh_output(tmp_path):
    c=passive(tmp_path);assert m.preflight(tmp_path,tmp_path/'contract.json')==c
    (tmp_path/'opaque.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):m.preflight(tmp_path,tmp_path/'contract.json')
    c['saved_inputs']['opaque.npz']=m.io.sha(tmp_path/'opaque.npz');(tmp_path/'contract.json').write_text(json.dumps(c))
    (tmp_path/m.FRESH).mkdir(parents=True)
    with pytest.raises(ValueError,match='FRESH_M3'):m.preflight(tmp_path,tmp_path/'contract.json')


@pytest.mark.parametrize('phase',['A','B','C'])
def test_scientific_failure_partial_journal_STOP_no_retry(tmp_path,monkeypatch,phase):
    c=passive(tmp_path);a,b,_=c_fixture()
    monkeypatch.setattr(m.subprocess,'check_output',lambda *args,**kw:'ARTIFICIAL_SOURCE\n')
    monkeypatch.setattr(m,'stage_a',lambda *args:a);monkeypatch.setattr(m,'stage_b',lambda *args:b)
    def fail(root,c,out,call,*args):
        m.io.save(out/'partial.json',dict(preserved=True))
        def fault():raise ValueError('ARTIFICIAL_CORRECTNESS_FAIL')
        return call('A_summary','synthetic failure',fault)
    monkeypatch.setattr(m,'stage_'+phase.lower(),fail)
    with pytest.raises(ValueError,match='ARTIFICIAL_CORRECTNESS_FAIL'):m.run(tmp_path,tmp_path/'contract.json')
    out=tmp_path/m.FRESH;stop=m.io.read(out/'STOP.json')
    assert stop['failed'] and stop['mandatory_stop'] and stop['retry']==stop['resume']==0
    assert m.io.read(out/phase/'partial.json')['preserved']
    assert 'ARTIFICIAL_CORRECTNESS_FAIL' in (out/'failure.json').read_text()
    with pytest.raises(ValueError,match='FRESH_M3'):m.run(tmp_path,tmp_path/'contract.json')
