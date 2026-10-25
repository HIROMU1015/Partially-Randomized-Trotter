"""Artificial CAR/native/P0/action end-to-end and scientific-failure STOP tests."""
from pathlib import Path
from dataclasses import asdict
from fractions import Fraction
import json,subprocess,sys
from unittest.mock import Mock,patch
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c1_h6_native as m


def artificial(root,angle=None):
    from trottertracks.c1_stage3_reference import reference_kernels
    U=np.array([[[.8,-.6],[.6,.8]]]);J=np.array([[[.02,.003],[.003,.01]]])
    if angle is not None:U=np.asarray([[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]])
    g=np.einsum('kpi,kqi,kij,krj,ksj->pqrs',U,U,J,U,U,optimize=True)
    packet=dict(h=np.array([[-.2,.03],[.03,.1]]),g=g,constant=np.asarray(.125))
    np.savez(root/'input.npz',**packet)
    row=dict(actual_rank=1,guard_policy=m.saved.molecule.GUARD_POLICY,sector_error=0.,numerical_guard=1e-12)
    ctx=m.saved.molecule.SectorCAR(2);G=ctx.generators(packet['h'],g,packet['constant'],J,U)
    _,vecs=np.linalg.eigh(sum(G))
    previous=dict(status='H6_SAVED_RANK14_SECTOR_PF_COMPLETE',candidates={})
    kernels={};states={}
    for arm in m.saved.ARMS:
        np.savez(root/(arm+'.npz'),J=J,U=U);m.io.save(root/(arm+'.json'),row)
        np.savez(root/(arm+'_ground.npz'),state=vecs[:,0]);states[arm]=arm+'_ground.npz'
        for s in (1,4):
            key=f'{arm}:s{s}';path=f'{arm}_s{s}.npz';kernels[key]=path
            np.savez(root/path,pf=reference_kernels(G,Fraction(1,10*s),ld=len(G)-1)['pf'])
            previous['candidates'][key]=dict(summary=dict(status='POTENTIALLY_FEASIBLE'),rounds=[dict(m=i) for i in range(13)])
    m.io.save(root/'PF.json',previous);m.io.save(root/'optimizer.json',dict(success=True))
    c=dict(original_input='input.npz',factor_arrays={a:a+'.npz' for a in m.saved.ARMS},
        saved_sector_rows={a:a+'.json' for a in m.saved.ARMS},saved_optimizer='optimizer.json',
        PF_result='PF.json',ground_states=states,PF_kernels=kernels,
        conditions=dict(m.saved.CONDITIONS,atoms=2,rank=1,sector_dimension=4))
    c['saved_inputs']={p:m.io.sha(root/p) for p in ['input.npz','CDF.npz','XDF.npz']}
    return c


@pytest.mark.parametrize('fault',[None,'phase','Gaussian_direction','frame_order','compile','adjoint'])
def test_artificial_end_to_end_state_actions_and_partial_STOP(tmp_path,fault):
    from contextlib import ExitStack
    from trottertracks.pr_dictionary_validation import gates
    c=artificial(tmp_path);out=tmp_path/'out'
    forbidden=Mock(side_effect=AssertionError('NO_NEW_FIT_DF_PF_OR_LARGE_OPERATOR'))
    ordered=m.native.NativeBuilder.ordered;basis=m.PrecisionBuilder._basis
    compiler=gates.compiled_wrapper
    def ordered_bad(self,model,time,**kwargs):
        if fault=='frame_order':model=m.replace(model,blocks=tuple(reversed(model.blocks)))
        qc=ordered(self,model,time,**kwargs)
        if fault=='phase':qc.p(.01,model.modes)
        return qc
    def compile_bad(qc,profile,**kwargs):
        result=compiler(qc,profile,**kwargs)
        if fault=='compile' and len(qc.data)>10:result.p(.01,qc.num_qubits-1)
        return result
    with ExitStack() as stack:
        stack.enter_context(patch.object(m,'preflight',return_value=c))
        stack.enter_context(patch.object(m.saved.molecule,'factor',forbidden))
        stack.enter_context(patch.object(m.saved.molecule,'rank_assessment',forbidden))
        stack.enter_context(patch.object(m.saved.molecule,'PF_rounds',forbidden))
        stack.enter_context(patch.object(m.saved.molecule.stage1,'car_bilinears',forbidden))
        stack.enter_context(patch.object(m.native.NativeBuilder,'ordered',ordered_bad))
        stack.enter_context(patch.object(gates,'compiled_wrapper',side_effect=compile_bad))
        if fault=='Gaussian_direction':stack.enter_context(patch.object(m.PrecisionBuilder,'_basis',lambda self,model,block:basis(self,model,block).inverse()))
        if fault=='adjoint':stack.enter_context(patch.object(m.native,'actual_adjoint',lambda qc:qc))
        if fault:
            with pytest.raises((ValueError,RuntimeError)):m.run(tmp_path,tmp_path/'unused',out)
        else:m.run(tmp_path,tmp_path/'unused',out)
    forbidden.assert_not_called();stop=m.io.read(out/'STOP.json')
    assert stop['failed']==bool(fault) and stop['scientific_units']<=m.MAX_CALLS
    assert stop['retry']==stop['resume']==0
    if fault:
        assert (out/'failure.json').exists() and not (out/'result.json').exists()
        if fault in ('phase','frame_order','compile'):
            details=list((out/'candidates').glob('*_kernel_failure.json'))
            assert details and all(m.io.read(p)['error'] for p in details)
    else:
        r=m.io.read(out/'result.json');assert r['global_synthesis_qualification']=='UNRESOLVED_FINITE_PROBES'
        assert len(r['candidates'])==8 and len(r['comparisons'])==8
        assert stop['scientific_units']==m.MAX_CALLS-4*(30-4)  # two artificial blocks/arm
        assert stop['call_counts']['compile']==11 and stop['call_counts']['round_synthesis']==104
        assert r['CA_achieved'] is r['shots'] is r['total_cost'] is None
        for candidate in r['candidates'].values():
            assert len(candidate['round_synthesis'])==13
            assert all(row['complex_signal_error_pilot'] is None for row in candidate['round_synthesis'])
            assert all(row['rigorous_signal_error_bound'] is None for row in candidate['round_synthesis'])
        assert len(list((out/'circuits').glob('*.qpy')))==26  # locals4 +8*2 +shells6
    original=(out/'STOP.json').read_bytes()
    with patch.object(m,'preflight',return_value=c),pytest.raises(FileExistsError):m.run(tmp_path,tmp_path/'unused',out)
    assert (out/'STOP.json').read_bytes()==original


def test_tiny_rotation_artificial_end_to_end_with_same_P0_and_allowances(tmp_path):
    c=artificial(tmp_path,angle=1e-9)
    with patch.object(m,'preflight',return_value=c):m.run(tmp_path,tmp_path/'unused',tmp_path/'out')
    result=m.io.read(tmp_path/'out/result.json')
    assert len(result['candidates'])==8
    for candidate in result['candidates'].values():
        assert candidate['kernel']['raw_reference']['max_entry']<2e-14
        # Use the unchanged registered component allowance for P0. Tiny-angle
        # preservation is tested separately against the exterior-power oracle.
        assert candidate['kernel']['compiler']['max_entry']<=m.ALLOWANCES['state_component']
    precision=m.io.read(tmp_path/'out/gaussian_precision.json')
    assert all(r['version']==m.GAUSSIAN_VERSION for r in precision.values())


def test_inverse_creation_and_P0_share_mutex_but_all_candidates_execute(tmp_path):
    import threading,time
    from trottertracks.pr_dictionary_validation import gates
    c=artificial(tmp_path);lock=threading.Lock();active=[];seen=[]
    compiler=gates.compiled_wrapper;inverse=m.native.actual_adjoint
    def guarded(kind,function,*args,**kwargs):
        with lock:
            assert not active,'concurrent Qiskit equivalence-library access'
            active.append(kind);seen.append(kind)
        try:
            time.sleep(.003)
            return function(*args,**kwargs)
        finally:
            with lock:active.pop()
    with patch.object(m,'preflight',return_value=c), \
            patch.object(gates,'compiled_wrapper',side_effect=lambda *a,**k:guarded('P0',compiler,*a,**k)), \
            patch.object(m.native,'actual_adjoint',side_effect=lambda *a,**k:guarded('inverse',inverse,*a,**k)):
        m.run(tmp_path,tmp_path/'unused',tmp_path/'out')
    assert seen.count('P0')==11 and seen.count('inverse')==16
    assert len(m.io.read(tmp_path/'out/result.json')['candidates'])==8


def test_probe_maxima_never_become_13_round_operator_bounds():
    shell={name:dict(unitary_distance_pilot=0.) for name in m.SHELLS}
    for n in (1,4,4096,16384):
        row=m.round_profile(dict(raw_reference=dict(max_entry=1e-16)),shell,n)
        assert row['occurrences']==n and row['qualification']=='UNRESOLVED_FINITE_PROBES'
        assert row['complex_signal_error_pilot'] is row['kernel_operator_norm_bound'] is None


def contract(root):
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    profile=asdict(CompilerProfile());profile['basis_gates']=list(profile['basis_gates'])
    (root/'source').write_text('artificial');(root/'bytes').write_text('not numerical')
    m.io.save(root/'SOURCE.json',dict(files={'source':m.io.sha(root/'source')}))
    return dict(scope=m.SCOPE,status='UNAPPROVED_SCIENCE_CONTRACT',conditions=m.saved.CONDITIONS,
        allowances=m.ALLOWANCES,compiler=profile,call_limits=m.LIMITS,maximum_logical_units=m.MAX_CALLS,
        routes=list(m.ROUTES),candidate_order=[f'{a}:s{s}' for a,s in m.saved.CANDIDATES],state_action=m.action.VERSION,
        native_gaussian=m.GAUSSIAN_VERSION,
        new_fit=0,optimizer=0,PF_recomputation=0,DF_error_recomputation=0,shots_evaluation=0,total_cost_evaluation=0,retry=0,resume=0,
        source_manifest='SOURCE.json',source_manifest_sha256=m.io.sha(root/'SOURCE.json'),saved_inputs={'bytes':m.io.sha(root/'bytes')},
        original_input='bytes',saved_optimizer='bytes',PF_result='bytes',factor_arrays={a:'bytes' for a in m.saved.ARMS},
        saved_sector_rows={a:'bytes' for a in m.saved.ARMS},ground_states={a:'bytes' for a in m.saved.ARMS},
        PF_kernels={f'{a}:s{s}':'bytes' for a,s in m.saved.CANDIDATES},evaluation_packages={})


@pytest.mark.parametrize('change',[None,'input','SOURCE','compiler','rank','scope','calls','gaussian'])
def test_passive_preflight_source_input_and_scientific_scope(tmp_path,change):
    c=contract(tmp_path)
    if change=='input':(tmp_path/'bytes').write_text('modified')
    if change=='SOURCE':(tmp_path/'source').write_text('modified')
    if change=='compiler':c['compiler']['optimization_level']=2
    if change=='rank':c['conditions']=dict(c['conditions'],rank=18)
    if change=='scope':c['shots_evaluation']=1
    if change=='calls':c['maximum_logical_units']+=1
    if change=='gaussian':c['native_gaussian']='unregistered'
    m.io.save(tmp_path/'contract.json',c)
    with patch.object(np,'load',side_effect=AssertionError('NO_MOLECULAR_DECODE')):
        if change:
            with pytest.raises(ValueError):m.preflight(tmp_path,tmp_path/'contract.json')
        else:assert m.preflight(tmp_path,tmp_path/'contract.json')==c


def test_call_scope_and_import_are_passive():
    assert m.MAX_CALLS==298 and m.LIMITS['compile']==11 and len(m.JOBS)==8
    code="import sys;sys.path.insert(0,'src');from trottertracks import c1_h6_native;assert not any(x in sys.modules for x in ['numpy','qiskit','scipy','ffsim','qiskit_aer'])"
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)
    p=subprocess.run([sys.executable,str(ROOT/'scripts/run_research_c_c1_h6_native.py'),'--contract','UNREAD','--execute'],capture_output=True,text=True)
    assert p.returncode==2 and 'Fresh' in p.stderr
