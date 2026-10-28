"""Invented arrays only: third-arm reference/accuracy and gated native flow."""
from pathlib import Path
from fractions import Fraction
from copy import deepcopy
import sys,json,subprocess,math
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_df_structural_pr as m
from trottertracks import c_df_structural_pr_native as n


def fixture(root):
    rng=np.random.default_rng(4528);J=np.zeros((8,4,4));U=[]
    from scipy.linalg import expm
    for l in range(8):
        a=rng.normal(size=(4,4));U.append(expm(.1*(a-a.T)))
        for i in range(4):
            for j in range(i,4):
                if l<2 or j-i<=1:J[l,i,j]=J[l,j,i]=rng.normal()*1e-4
    U=np.array(U);U[0,:,0]*=-1
    p=dict(h=np.diag([.1,.11,.12,.13]),g=np.zeros((4,)*4),constant=np.array(.31))
    np.savez(root/'input.npz',**p);np.savez(root/'JU.npz',J=J,U=U)
    c=dict(input='input.npz',export='JU.npz',saved_inputs={'JU.npz':'ARTIFICIAL_JU'},original_input_sha256='ARTIFICIAL_INPUT',source_manifest_sha256='ARTIFICIAL_SOURCE')
    plans=[dict(m=i,shots=1000+i) for i in range(13)]
    metrics={k:100. for k in m.old.METRICS};cost={k:1e12 for k in m.old.METRICS}
    def resources():return dict(totals=cost)
    from trottertracks import c_pr_aware_df as objective
    mass=float(np.sum(abs(objective.coefficients(J[2:],np))))
    saved=dict(P4=dict(final_diagnostic=dict(exact_tail_mass_Ha=mass)),P4_sector=dict(sector_error=1e-8,numerical_guard=1e-9),
        M4=dict(B={lane:dict(rounds=plans) for lane in ('ideal_exact_native','assumed_raw_signal_budget')}),
        native=dict(calibration=dict(sample_mean=metrics),lanes={lane:dict(conditional_totals=cost) for lane in ('ideal_exact_native','assumed_raw_signal_budget')}),
        M3=dict(C=dict(lanes={lane:dict(resources=resources()) for lane in ('ideal_exact_native','assumed_raw_signal_budget')})),
        C1=dict(assumed_scenarios={lane:dict(proposed=dict(resources=resources())) for lane in ('ideal_exact_native','assumed_raw_signal_budget')}),
        M2=dict(candidates=dict(proposed=dict(cost=resources()))),
        shell_reports={label:dict(metrics={k:v for k in m.old.METRICS}) for label,v in [('head',2),('X',3),('Y',4)]})
    return c,saved,p,dict(J=J,U=U)


def test_passive_third_role_structure_and_legacy_gate_unchanged(tmp_path):
    code="import sys;sys.path.insert(0,'src');from trottertracks import c_df_structural_pr as m;assert not any(x in sys.modules for x in ('numpy','qiskit','scipy','ffsim'));assert m.MAX_CALLS==sum(m.LIMITS.values())"
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)
    c,s,p,d=fixture(tmp_path);arm=m.prepare_arm(**p,**d,source_identity='NEW_JU',input_identity='I')
    from trottertracks import c_pr_aware_df_stage_b as b
    assert b.ARMS==('baseline','proposed') and arm.role=='tail_p4'
    with pytest.raises(ValueError,match='PAIRED_CDF_ROLE'):b.prepare_arm(**p,**d,role='tail_p4',source_identity='S',input_identity='I',ld=2)
    assert len(arm.plan.atoms)==144 and all(x.frame_id.startswith('CDF_tail_p4:') for x in arm.plan.atoms)
    assert [x.original_index for x in arm.general.frames]==list(range(8))


def test_new_reference_all13_law_ground_RPE_price(tmp_path,monkeypatch):
    c,s,p,d=fixture(tmp_path);monkeypatch.setattr(m,'evidence',lambda *a:s)
    out=tmp_path/'A';out.mkdir();call=m.integrated.Calls(out,m.LIMITS,m.MAX_CALLS)
    with threadpool_limits(limits=1):a=m.stage_reference(tmp_path,c,out,call)
    assert a['result']['status']=='POTENTIALLY_FEASIBLE'
    assert a['arm'].role=='tail_p4' and a['result']['tail']['atoms']==144
    assert len(a['result']['rounds'])==13 and a['result']['ground']['status']=='PASS'
    assert dict(call.counts)==m.REFERENCE_LIMITS
    (out/'acc').mkdir();lanes=m.stage_accuracy(out/'acc',call,a)
    assert lanes['evidence']['shots']['total_shots'] is None
    for lane in ('ideal_exact_native','assumed_raw_signal_budget'):assert lanes[lane]['energy']['same_accuracy']
    (out/'price').mkdir();prices=m.stage_price(out/'price',call,a,lanes)
    assert m.native_go(a,lanes,prices)['status']=='GO_FIXED32'
    aff=prices['affine']['ideal_exact_native'];shots=lanes['ideal_exact_native']['shots']['total_shots']
    assert aff['S']['rz_count']==shots*5.5 # head2+X3 / head2+Y4, once each
    assert aff['W']==sum(2*x['shots']*(1<<x['m']) for x in lanes['ideal_exact_native']['rounds'])


@pytest.mark.parametrize('sign',[-1,0,1])
def test_break_even_and_native_GO_sign(tmp_path,sign):
    affine=dict(W=100,S={k:1000 for k in m.old.METRICS})
    cost={k:1000+sign*100 for k in m.old.METRICS};p=m.threshold(affine,cost,'invented')
    mu=p['thresholds']['rz_count']['mu_break_even'];assert affine['W']*mu+1000==cost['rz_count']
    a=dict(result=dict(status='POTENTIALLY_FEASIBLE',ground=dict(status='PASS')))
    lanes=dict(ideal_exact_native=dict(energy=dict(same_accuracy=True),shots=dict(total_shots=1)))
    prices=dict(comparisons={'ideal_exact_native:M4_r128':p})
    assert (m.native_go(a,lanes,prices)['status']=='GO_FIXED32')==(sign>0)
    lanes['ideal_exact_native']['energy']['same_accuracy']=False
    assert m.native_go(a,lanes,prices)['status']=='NOT_RUN_A_GATE'


def test_32seed_source_JU_role_independence(tmp_path):
    c,s,p,d=fixture(tmp_path);arm=m.prepare_arm(**p,**d,source_identity='JU1',input_identity='I')
    from trottertracks import c_pr_aware_df_stage_b as b,c_pr_aware_df_m4_native as old
    tail,config,dist=b.configure(arm,Fraction(1,10),128,0);seeds=set()
    for i in range(32):
        events,rec=n.sample(arm,tail,dist,config,i,'S1');assert len(events)==128 and rec['role']=='tail_p4'
        assert not rec['paired'] and not rec['old_cohort_pooled'];seeds.add(rec['seed'])
    assert len(seeds)==32
    _,a=n.sample(arm,tail,dist,config,0,'S1');_,brec=n.sample(arm,tail,dist,config,0,'S2')
    assert a['seed']!=brec['seed'] and a['seed']!=old.sample(arm,tail,dist,config,0,'S1')[1]['seed']
    assert n.sample(arm,tail,dist,config,0,'S1')[1]==a
    assert m.threshold(dict(W=None),{},'unresolved')['thresholds'] is None


def test_one_artificial_P4_128_full512_compile_adjoint_XY(tmp_path):
    from trottertracks import c_pr_aware_df_stage_b as b,c_pr_aware_df_stage_b2 as b2
    from trottertracks.c1_h4_native_fair import compile_checked
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    c,s,p,d=fixture(tmp_path);arm=m.prepare_arm(**p,**d,source_identity='S',input_identity='I');h=Fraction(1,10)
    with threadpool_limits(limits=1):
        car=b.stage1.car_bilinears(8);G=[b.stage1.raw_base(b.stage1.corrected_kappa(p['h'],p['g']),p['constant'],car)]
        G.extend(b.stage1.raw_frame(J,U,car)[0] for J,U in zip(d['J'],d['U'],strict=True))
        tail,config,dist=b.configure(arm,h,128,0);ctx=b2.reference_context(arm,G,car,h)
        events,_=n.sample(arm,tail,dist,config,0,'ARTIFICIAL')
        ref=b2.trajectory_reference(arm,ctx,h,events);raw=b.build_kernel(arm,h,events,config)
        compiled=compile_checked(raw,CompilerProfile());R,target,_=b2.raw_gate(raw,ref);C,_=b2.compiled_gate(compiled,R,target)
        b2.adjoint_pair_gate(raw,compiled,R,C,target);shells={}
        for label in ('head','X','Y'):
            qc=b2.shell_build(8,label);shells[label],_=b2.shell_assessment(qc,compile_checked(qc,CompilerProfile()),label)
        psi=np.zeros(256,complex);psi[0]=1
        assert b2.XY_gate(R,C,ref,psi,shells)['status']=='PASS'
        with pytest.raises(ValueError):b2.raw_gate(raw,np.exp(.01j)*ref)


def test_calibration_shared_SE_and_unknowns(tmp_path):
    from trottertracks import c_pr_aware_df_stage_b2 as b2
    c,s,p,d=fixture(tmp_path);cal=b2.calibration([{k:10+i for k in m.old.METRICS} for i in range(32)])
    aff=dict(W=100,S={k:5 for k in m.old.METRICS});prices=dict(affine={lane:aff for lane in ('ideal_exact_native','assumed_raw_signal_budget')},comparisons={})
    for lane in prices['affine']:
        for target,cost in m.targets(s,lane).items():prices['comparisons'][lane+':'+target]=m.threshold(aff,cost,target)
    v=m.calibrated_result(cal,dict(saved=s),{},prices)
    assert v['lanes']['ideal_exact_native']['conditional_totals']['rz_count']==2555
    assert v['lanes']['ideal_exact_native']['weighted_calibration_SE']['rz_count']==100*cal['sample_mean_standard_error']['rz_count']
    assert v['cost_support'] is v['actual_CA_achieved'] is v['ideal_law_mean_advantage'] is None


@pytest.mark.parametrize('status',['INFEASIBLE','UNRESOLVED','POTENTIALLY_FEASIBLE'])
def test_execute_stage_gate(tmp_path,monkeypatch,status):
    a=dict(result=dict(status=status,reason='ARTIFICIAL'));seen=[]
    monkeypatch.setattr(m,'stage_reference',lambda *x:a)
    monkeypatch.setattr(m,'stage_accuracy',lambda *x:{})
    monkeypatch.setattr(m,'stage_price',lambda *x:{})
    monkeypatch.setattr(m,'native_go',lambda *x:dict(status='GO_FIXED32'))
    monkeypatch.setattr(n,'execute',lambda *x:seen.append(1) or dict(status='MOCK_FIXED32'))
    call=m.integrated.Calls(tmp_path,m.LIMITS,m.MAX_CALLS)
    r=m.execute(tmp_path,{},tmp_path,call);assert bool(seen)==(status=='POTENTIALLY_FEASIBLE')
    assert r['mandatory_stop'] and r['new_fit']==0


def passive(tmp_path):
    from trottertracks.c_pr_aware_df_stage_c import SCENARIOS
    from trottertracks.c_pr_aware_df_stage_b2 import profile_dict
    m.io.save(tmp_path/'source.json',dict(files={}))
    c=dict(scope=m.SCOPE,implementation=m.VERSION,policy=m.POLICY,target=m.TARGET,
        allowances=m.ALLOWANCES,sampling=m.SAMPLING,scenarios=list(SCENARIOS),compiler=profile_dict(),
        call_limits=m.LIMITS,maximum_logical_units=m.MAX_CALLS,fresh_output=m.FRESH,new_fit=0,retry=0,resume=0,
        source_manifest='source.json',source_manifest_sha256=m.io.sha(tmp_path/'source.json'),saved_inputs={},packages={})
    m.io.save(tmp_path/'contract.json',c);return c


def test_passive_partial_failure_source_mismatch_no_retry(tmp_path,monkeypatch):
    c=passive(tmp_path);assert m.preflight(tmp_path,tmp_path/'contract.json')==c
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'ARTIFICIAL\n')
    def fail(root,c,out,call):
        m.io.save(out/'partial.json',dict(preserved=True))
        def error():raise ValueError('ARTIFICIAL_CORRECTNESS')
        return call('evidence_load','artificial',error)
    monkeypatch.setattr(m,'execute',fail)
    with pytest.raises(ValueError,match='ARTIFICIAL_CORRECTNESS'):m.run(tmp_path,tmp_path/'contract.json')
    stop=m.io.read(tmp_path/m.FRESH/'STOP.json');assert stop['failed'] and stop['scientific_units']==1 and stop['mandatory_stop']
    with pytest.raises(ValueError,match='FRESH'):m.run(tmp_path,tmp_path/'contract.json')


def test_mock_all449_dispatch_fixed32_and_QPY(tmp_path,monkeypatch):
    from qiskit import QuantumCircuit,qpy
    from trottertracks import c_pr_aware_df_stage_b as b,c_pr_aware_df_stage_b2 as b2
    from trottertracks.pr_dictionary_validation.f2_execution import circuit_metrics
    c,s,p,d=fixture(tmp_path);monkeypatch.setattr(m,'evidence',lambda *x:s)
    c['shells']={}
    for label in ('head','X','Y'):
        qc=b2.shell_build(8,label);c['shells'][label]={}
        for kind in ('raw','compiled'):
            name=label+kind+'.qpy';c['shells'][label][kind]=name
            with (tmp_path/name).open('xb') as stream:qpy.dump(qc,stream)
        s['shell_reports'][label]['metrics']=circuit_metrics(qc)
    monkeypatch.setattr(b,'build_kernel',lambda *x,**k:QuantumCircuit(9))
    monkeypatch.setattr(b2,'reference_context',lambda *x:None)
    monkeypatch.setattr(b2,'law_gate',lambda *x:dict(status='MOCK'))
    monkeypatch.setattr(b2,'trajectory_reference',lambda *x:np.eye(2,dtype=complex))
    monkeypatch.setattr(b2,'raw_gate',lambda *x:(None,None,dict(status='MOCK')))
    monkeypatch.setattr(b2,'compiled_gate',lambda *x:(None,dict(status='MOCK')))
    monkeypatch.setattr(b2,'adjoint_pair_gate',lambda *x:({'raw':QuantumCircuit(9),'compiled':QuantumCircuit(9)},dict(status='MOCK')))
    monkeypatch.setattr(b2,'XY_gate',lambda *x:dict(status='MOCK'))
    out=tmp_path/'all';out.mkdir();call=m.integrated.Calls(out,m.LIMITS,m.MAX_CALLS)
    with threadpool_limits(limits=1):result=m.execute(tmp_path,c,out,call)
    assert dict(call.counts)==m.LIMITS and sum(call.counts.values())==449
    assert result['status']=='TAIL_P4_PROGRAM_COMPLETE' and not call.stop.is_set()
    assert len(list((out/'native/circuits').glob('*.qpy')))==128
    assert len(list((out/'native/samples').glob('*_trajectory.json')))==32
    assert result['native']['completed_build_compile']==64


def test_evidence_budget_identity_failure(tmp_path):
    c,s,p,d=fixture(tmp_path)
    s.update(P4=dict(status='DF_ELIGIBLE_NOT_PERFORMANCE_GO',optimizer_success=True,final_DF_budget_PASS=True),
        P4_STOP=dict(failed=False,scientific_units=21,retry=0,resume=0),P4_SOURCE=dict(execution_commit='ARTIFICIAL'),
        physical_ground=dict(status='PASS',alias_margin=1.),M4_STOP=dict(failed=False,scientific_units=98),
        native_STOP=dict(failed=False,scientific_units=332))
    s['M4']['status']='M4_R128_SHOTS_BREAKEVEN_COMPLETE';s['native']['status']='NATIVE128_FINITE_COHORT_COMPLETE'
    c.update(records={},P4_scientific_SOURCE='ARTIFICIAL')
    for k,v in s.items():
        c['records'][k]=k+'.json';m.io.save(tmp_path/(k+'.json'),v)
    assert m.evidence(tmp_path,c)['P4']['optimizer_success']
    s['P4_sector']['sector_error']=1.
    (tmp_path/'P4_sector.json').write_text(json.dumps(s['P4_sector']))
    with pytest.raises(ValueError,match='DF_PROVENANCE'):m.evidence(tmp_path,c)
