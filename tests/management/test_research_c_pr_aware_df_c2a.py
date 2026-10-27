"""Artificial circuits/proof inputs only;no saved molecular arrays decoded."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_pr_aware_df_c2a as a


def proof(value='0',coverage='all_trajectory_words',**kwargs):
    return dict(epsilon=value,kind='analytic_unitary_bound',norm='full_controlled_operator_2',
        phase_sensitive=True,provenance='artificial analytic identity',excludes_CAR_bridge=True,
        coverage=coverage,**kwargs)


def inputs():
    return dict(m=12,raw={k:proof() for k in a.RAW_IDS},
        compiler={k:proof() for k in a.P0_IDS},reference=proof(),
        shells={k:proof(coverage='fixed_shell:'+k) for k in a.SHELLS},
        coverage={k:True for k in ('all_trajectory_words','whole_P0','initial_translation',
            'ZSXX_all_branches','rounding_model','monotone_conditions','phase_policy')},
        exact_unitary_targets=True,reference_invariant=True)


def test_passive_import_and_execute_option_absent():
    subprocess.run([sys.executable,'-B','-c',"import sys;sys.path.insert(0,'src');from trottertracks import c_pr_aware_df_c2a;assert not any(x in sys.modules for x in ('numpy','scipy','qiskit','ffsim'))"],cwd=ROOT,check=True)
    result=subprocess.run([sys.executable,'-B','scripts/check_research_c_pr_aware_df_c2a.py','--contract','missing','--execute'],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==2 and 'unrecognized arguments' in result.stderr


def test_structural_grammar_no_exponential_enumeration():
    r=a.primitive_grammar();assert r['atom_alphabet']==216 and r['events']==16
    assert r['gaussian_occurrences_upper']==2*(2+2)+2*16
    assert r['gaussian_XXPlusYY_upper']==8*7//2
    assert r['all_law_synthesis'] is r['worst_case_translated_nodes'] is None


@pytest.mark.parametrize('N',[1,17,100])
def test_monotone_accepted_visit_amortization(N):
    r=a.monotone_budget(N,basis_closed=True,target_none=True,zsxx_only=True,
        parameter_free_cancellation=True,no_calibrations=True)
    assert r['accepted_rewrites']==N and r['accepted_gate_visits']==6*N
    assert r['iterations']==N+1 and not r['all_law_qualified']
    # Every accepted input node is deleted; introduced nodes<=5 per rewrite.
    assert r['accepted_gate_visits']==N+r['output_gates_per_ZSXX']*r['accepted_rewrites']


@pytest.mark.parametrize('condition',['basis_closed','target_none','zsxx_only','parameter_free_cancellation','no_calibrations'])
def test_monotone_argument_is_conditional(condition):
    kw=dict(basis_closed=True,target_none=True,zsxx_only=True,parameter_free_cancellation=True,no_calibrations=True)
    kw[condition]=False
    assert a.monotone_budget(10,**kw)['accepted_rewrites'] is None


def test_actual_preset_and_symbolic_native_definitions():
    r=a.compiler_coverage();assert r['profile']==a.P0 and r['scientific_calls']==0
    assert not r['graph_check']['mismatches']
    loop=next(x for x in r['expanded_stage_graph']['optimization'] if x['name']=='DoWhileController')
    run=loop['nested'][0];assert run['target_none'] and run['decomposers']==['ZSXX'] and run['error_map_none']
    assert loop['nested'][1]['parameter_free']
    assert r['uniform_ZSXX_error'] is None and not r['all_law_qualified']
    templates=a.symbolic_templates()['templates']
    assert templates['XX']['lowered_counts']==templates['XX_inverse']['lowered_counts']
    assert templates['controlled_RZ']['lowered_counts']==dict(rz=2,cx=2)
    assert all(set(x['lowered_counts'])<=set(a.P0['basis_gates'])|{'u'} for x in templates.values())


def test_unexpected_compiler_transform_or_policy_not_covered():
    graph=deepcopy(a.compiler_coverage()['expanded_stage_graph'])
    graph['translation'].append(dict(name='ConsolidateBlocks'))
    assert a.check_graph(graph)['mismatches']==['translation']
    graph=deepcopy(a.compiler_coverage()['expanded_stage_graph'])
    loop=next(x for x in graph['optimization'] if x['name']=='DoWhileController')
    loop['nested'][0]['target_none']=False
    assert 'NUMERIC_OPTIMIZER_OR_CANCELLATION_POLICY' in a.check_graph(graph)['mismatches']


@pytest.mark.parametrize('angle',[1e-13,np.pi/2+1e-13,np.pi-1e-13,2*np.pi+1e-13,-2*np.pi-.13])
def test_fixed_whole_P0_branch_phase_adjoint_and_XY(angle):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator,Statevector
    from trottertracks.c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
    from trottertracks.c1_stage2_native import actual_adjoint
    q=QuantumCircuit(2);q.ry(angle,0);q.rz(.31,0);q.rz(-.11,0);q.rz(-.2,0)
    q.rz(.17,0);q.cx(1,0);q.rz(-.17,0);q.cx(1,0);q.ry(-angle,0);q.p(.21,1)
    compiled,trace=a.trace_artificial(q);U=Operator(q).data;V=Operator(compiled).data
    assert np.max(abs(U-V))<1e-10 and not trace['all_law_proof']
    assert any(x['pass_name']=='Optimize1qGatesDecomposition' for x in trace['passes'])
    mutex=SharedQiskitCalls(lambda k,l,fn:fn())
    with mutex.adjoint_builds():adj=actual_adjoint(compiled)
    assert np.max(abs(Operator(adj).data@V-np.eye(4)))<1e-10
    # Full controlled identity branch and relative phase;no phase fitting.
    assert np.max(abs(U[:2,:2]-np.eye(2)))<1e-14
    broken=q.copy();broken.data.pop();assert np.max(abs(Operator(broken).data-U))>.1
    for axis in ('X','Y'):
        vals=[]
        for circuit in (q,compiled):
            wrapper=QuantumCircuit(2);wrapper.h(1);wrapper.compose(circuit,inplace=True)
            if axis=='Y':wrapper.sdg(1)
            wrapper.h(1);sv=Statevector.from_instruction(wrapper).data
            vals.append(sum(abs(sv[:2])**2)-sum(abs(sv[2:])**2))
        assert abs(vals[0]-vals[1])<2e-10


def test_cross_event_runs_without_barriers_and_mod2pi_phase():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit.circuit.library import RZGate
    q=QuantumCircuit(2);q.rz(.31,0);q.rz(-.11,0);q.cx(1,0)
    compiled,t=a.trace_artificial(q)
    before=next(x for x in t['passes'] if x['pass_name']=='BasisTranslator')
    assert any(x['qubit']==0 and x['gates']==['rz','rz'] for x in before['runs'])
    assert any(x['pass_name']=='Optimize1qGatesDecomposition' and x['size']<before['size'] for x in t['passes'])
    z=Operator(RZGate(.23)).data;wrapped=Operator(RZGate(.23+2*np.pi)).data
    assert np.max(abs(wrapped+z))<1e-14 and np.max(abs(wrapped-z))>1
    assert np.max(abs(Operator(q).data-Operator(compiled).data))<1e-10


def test_artificial_E2E_exact_bound_inputs_and_raw_units():
    x=inputs();x['raw']['signed_event']=proof('1/1000000000000')
    x['compiler']['run_product']=proof('1/1000000000000');x['reference']=proof('1/1000000000000')
    x['shells']['head']=proof('1/1000000000000',coverage='fixed_shell:head')
    x['shells']['X']=proof('2/1000000000000',coverage='fixed_shell:X')
    r=a.uniform_preview(**x)
    n=4*2**12
    assert Fraction(r['candidate_raw_complex_bound'])==2*Fraction(2*n*3+4,10**12)
    assert r['status']=='CONDITIONAL_ON_SUPPLIED_PROOFS'
    assert not r['attenuation_factor_applied'] and not r['CAR_bridge_included'] and not r['law_included']
    assert r['actual_H4_bound'] is r['guaranteed_total'] is r['error_evidence'] is None
    assert r['repeat_diagnostic']['rigorous_operator_bound'] is None


@pytest.mark.parametrize('group,key',[('raw','gaussian_basis'),('compiler','mod2pi_phase'),('shells','Y'),('reference',None)])
def test_unknown_values_propagate_null(group,key):
    x=inputs()
    if key is None:x[group]=None
    else:x[group][key]=None
    r=a.uniform_preview(**x);assert r['candidate_raw_complex_bound'] is None and r['missing']


def test_missing_paths_nonunitary_reference_samples_and_double_count_refused():
    x=inputs();x['coverage']['ZSXX_all_branches']=False
    assert a.uniform_preview(**x)['candidate_raw_complex_bound'] is None
    x=inputs();x['exact_unitary_targets']=False
    assert a.uniform_preview(**x)['candidate_raw_complex_bound'] is None
    for mode in ('sample_max','event_P0','CAR_bridge_included','phase_fit'):
        x=inputs();p=x['compiler']['run_product']
        if mode=='sample_max':p['kind']='numerical_max_residual'
        if mode=='event_P0':p['coverage']='event_compile_only'
        if mode=='CAR_bridge_included':p['excludes_CAR_bridge']=False
        if mode=='phase_fit':p['phase_sensitive']=False
        with pytest.raises(ValueError):a.uniform_preview(**x)


def test_blocked_preflight_hash_identity_and_no_decode(tmp_path,monkeypatch):
    s=tmp_path/'SOURCE.json';a.io.save(s,dict(files={}))
    p=tmp_path/'saved.npz';p.write_bytes(b'not numerical data')
    c=dict(scope=a.SCOPE,implementation=a.VERSION,status='BLOCKED_UNAPPROVED_SCIENCE_DESIGN',
        executable=False,maximum_logical_units=None,compiler=a.P0,retry=0,resume=0,
        source_manifest='SOURCE.json',source_manifest_sha256=a.io.sha(s),saved_inputs={'saved.npz':a.io.sha(p)},
        packages={},external_python_source_hashes={},blockers=['uniform ZSXX proof'])
    contract=tmp_path/'contract.json';a.io.save(contract,c)
    monkeypatch.setattr(np,'load',lambda *args,**kwargs:pytest.fail('No NPZ decode'))
    assert not a.preflight(tmp_path,contract)['executable']
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):a.preflight(tmp_path,contract)
    c['executable']=True;bad=tmp_path/'bad_contract.json';a.io.save(bad,c)
    with pytest.raises(ValueError,match='TECHNICAL_SCOPE'):a.preflight(tmp_path,bad)
