"""Finite, authorized reference-A correctness study; no price or pilot runner."""
from __future__ import annotations
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import io
import subprocess
import time
import numpy as np
from qiskit import QuantumCircuit,qasm2
from qiskit.circuit.library import RZZGate,UnitaryGate
from qiskit.quantum_info import Operator
from qiskit.transpiler import TranspilerError
from .track_c_compiler_a import compile_reference,PROFILE
from .track_c_h4 import common,save_json,complex_json,phase_distance,weights,ROOT
from .track_c_h4_price import d_part,trajectory_parts,wrapper

OUT=ROOT/'artifacts/track_c_compiler_a_correctness/2026-11-06/run01'
BASE='05d9c5d2eb984e0f3ce89f44a3f2928685b8d4bf'
# Numerical implementation diagnostics, not all-round physical allowances.
SYNTHETIC_TOL=1e-12
H4_TOL=1e-9


def synthetic_checks():
    angles=[0.,1e-15,1e-6,5e-6,float(np.nextafter(1e-5,0.)),1e-5,
            float(np.nextafter(1e-5,math.inf)),1e-4,2*math.pi,4*math.pi]
    angles=angles+[-x for x in angles if x]
    rows=[]
    def calibrated_error(a,b):
        # Control is bit0 in these artificial cases. A phase fitted only on
        # control0 cannot hide a relative phase error on control1.
        a0=a[0::2,0::2];b0=b[0::2,0::2]
        overlap=np.vdot(a0,b0);phase=overlap/abs(overlap)
        return float(np.linalg.norm(a-b/phase,2)),float(np.angle(phase))
    def check(name,qc,theta=None):
        a=Operator(qc).data.copy();native=compile_reference(qc);b=Operator(native).data
        error=float(np.linalg.norm(a-b,2))
        physical_error,phase=calibrated_error(a,b)
        initial=np.zeros(2**qc.num_qubits,complex);initial[:2]=1/math.sqrt(2)
        x=a@initial;y=b@initial
        zx=2*np.vdot(x[0::2],x[1::2]);zy=2*np.vdot(y[0::2],y[1::2])
        row=dict(name=name,theta=theta,strict_joint_operator_error=error,
            control0_calibrated_operator_error=physical_error,common_joint_phase=phase,
            source_coherence=complex_json(zx),native_coherence=complex_json(zy),
            complex_signal_error=float(abs(zx-zy)),native_operations=dict(native.count_ops()))
        if name=='CRZ' and theta:
            rz=[float(i.operation.params[0]) for i in native.data if i.operation.name=='rz']
            row['nonzero_half_angles_present']=theta/2 in rz and -theta/2 in rz
        row['passed']=physical_error<SYNTHETIC_TOL and abs(zx-zy)<SYNTHETIC_TOL and row.get('nonzero_half_angles_present',True)
        rows.append(row)
    for theta in angles:
        q=QuantumCircuit(2);q.crz(theta,0,1);check('CRZ',q,theta)
        q=QuantumCircuit(3);q.append(RZZGate(theta).control(),[0,1,2]);check('controlled_RZZ',q,theta)
    for gamma in [.123,-.123,math.pi,2*math.pi]:
        system=QuantumCircuit(1,global_phase=gamma)
        q=QuantumCircuit(2);q.append(system.to_gate().control(),[0,1]);check('controlled_system_scalar',q,gamma)
    q=QuantumCircuit(3);q.h(1);q.ry(.31,2);q.crz(-1e-5,0,1)
    q.append(RZZGate(.017).control(),[0,1,2]);q.p(-.23,0)
    native=compile_reference(q);check('source_actual_adjoint',q.inverse())
    adjoint_error,adjoint_phase=calibrated_error(Operator(q.inverse()).data,Operator(native.inverse()).data)
    q=QuantumCircuit(2);q.h(1);q.h(1);q.x(1);q.x(1);q.cx(0,1);q.cx(0,1);q.crz(1e-5,0,1)
    check('fixed_inverse_pairs_and_tiny_controlled_rotation',q)
    rejected=False
    try:
        q=QuantumCircuit(1);q.append(UnitaryGate(np.eye(2)),[0]);compile_reference(q)
    except TranspilerError:rejected=True
    result=dict(profile=PROFILE,diagnostic_tolerance=SYNTHETIC_TOL,
        tolerance_scope='local numerical consistency only; not an RPE compiler budget',
        cases=rows,actual_native_adjoint_error=adjoint_error,actual_adjoint_common_phase=adjoint_phase,
        opaque_matrix_rejected=rejected,
        all_passed=all(r['passed'] for r in rows) and adjoint_error<SYNTHETIC_TOL and rejected)
    save_json(OUT/'SYNTHETIC.json',result)
    if not result['all_passed']:raise ArithmeticError('synthetic reference-A check failed')
    return dict(cases=len(rows),max_physical_operator_error=max(r['control0_calibrated_operator_error'] for r in rows),all_passed=True)


def h4_tasks():
    return [dict(pilot='C-N',formula=f,h=.02,ld=ld,name=name,axis=axis)
        for f,ld,name in [('S2',3,'CN_S2_h0.02_ld3'),('Y4',12,'CN_Y4_h0.02_ld12')]
        for axis in ('X','Y')]


def h4_source(task):
    r=1 if task['ld']==3 else 0
    seed=int.from_bytes(hashlib.sha256(f"{task['name']}:{r}:1:False:0".encode()).digest()[:8],'big')
    # d_override avoids forward_part(), which also computes old opt2 prices.
    # No price function or old launcher is invoked.
    parts,_,_=trajectory_parts(task,1,r,False,seed,
        d_override=partial(d_part,task['ld']),price_additive=False)
    core=QuantumCircuit(9)
    for part in parts:core.compose(part,qubits=range(9),inplace=True)
    return parts,core,seed


def strong_wrapper(parts,task):
    if task['ld']==12:
        # Historical strong Full baseline receives the actual full wrapper.
        return compile_reference(wrapper(parts,task['axis']))
    # At q1 S2 the maximal D runs are the two sides of the one R occurrence.
    assert len(parts)==3
    prep=QuantumCircuit(9);prep.h(8)
    read=QuantumCircuit(9,1)
    if task['axis']=='Y':read.sdg(8)
    read.h(8);read.measure(8,0)
    out=QuantumCircuit(9,1)
    for piece in [compile_reference(prep),*(compile_reference(x) for x in parts),compile_reference(read)]:
        out.compose(piece,qubits=range(9),clbits=range(piece.num_clbits),inplace=True)
    return out


def h4_check(task):
    start=time.monotonic();parts,core,seed=h4_source(task)
    source=wrapper(parts,task['axis']);native=compile_reference(source)
    strong=strong_wrapper(parts,task) if task['ld']==3 else native.copy()
    key=task['name']+'_'+task['axis'];directory=OUT/'h4';directory.mkdir(exist_ok=True)
    (directory/f'{key}_native.qasm').write_text(qasm2.dumps(native)+'\n')
    if task['ld']==3:(directory/f'{key}_strong.qasm').write_text(qasm2.dumps(strong)+'\n')
    print(key,'lowered; computing full joint operators',flush=True)
    # 512x512 operators include every particle-number sector; no projection.
    a=Operator(source.remove_final_measurements(inplace=False)).data
    b=Operator(native.remove_final_measurements(inplace=False)).data
    bs=Operator(strong.remove_final_measurements(inplace=False)).data if task['ld']==3 else b
    c=common();reverse=np.array([int(f'{int(x):08b}'[::-1],2) for x in c['sector'].basis_indices])
    initial=np.zeros(512,complex);initial[reverse]=c['state']
    h=np.array([[1.,1.],[1.,-1.]])/math.sqrt(2)
    pre=np.kron(h,np.eye(256));read=np.kron(h if task['axis']=='X' else h@np.diag([1.,-1j]),np.eye(256))
    src_core=read.conj().T@a@pre
    plus=pre@initial
    def measure(v):return float(np.vdot(v[:256],v[:256]).real-np.vdot(v[256:],v[256:]).real)
    def norm(x):return float(np.linalg.norm(x,2))
    zsource_state=src_core@plus
    zsource=2*np.vdot(zsource_state[:256],zsource_state[256:])
    results={}
    for name,u in [('full',b),('strong',bs)]:
        if name=='strong' and task['ld']==12:
            results[name]=dict(results['full'],same_full_wrapper_by_baseline_definition=True);continue
        native_core=read.conj().T@u@pre
        # Fit only a common joint phase using the control-0 block. Never fit
        # an independent phase to the controlled system branch.
        overlap=np.trace(src_core[:256,:256].conj().T@native_core[:256,:256])
        phase=overlap/abs(overlap);aligned=native_core/phase
        v=native_core@plus;z=2*np.vdot(v[:256],v[256:])
        result=dict(strict_wrapper_operator_error=norm(a-u),
            common_joint_phase=float(np.angle(phase)),
            control0_calibrated_operator_error=norm(src_core-aligned),
            source_control0_identity_error=norm(src_core[:256,:256]-np.eye(256)),
            native_control0_identity_error=norm(aligned[:256,:256]-np.eye(256)),
            native_control_mixing_error=max(norm(aligned[:256,256:]),norm(aligned[256:,:256])),
            source_expectation=measure(a@initial),native_expectation=measure(u@initial),
            expectation_error=abs(measure(a@initial)-measure(u@initial)),
            source_coherence=complex_json(zsource),native_coherence=complex_json(z),
            complex_signal_error=float(abs(z-zsource)),phase_error=phase_distance(z,zsource),
            source_radius=float(abs(zsource)),native_radius=float(abs(z)),
            radius_error=float(abs(abs(z)-abs(zsource))),
            unitarity_error=norm(u.conj().T@u-np.eye(512)))
        result['numerical_consistency_pass']=all(result[x]<H4_TOL for x in
            ['control0_calibrated_operator_error','native_control0_identity_error',
             'native_control_mixing_error','expectation_error','complex_signal_error','unitarity_error'])
        results[name]=result
    np.savez_compressed(directory/f'{key}_operators.npz',source=a,native=b,strong=bs)
    result=dict(task=task,seed=seed,q=1,r=1 if task['ld']==3 else 0,original_stratum=0,j=0,
        source_input='same fixed source trajectory as old q1/j0; no new statistical sample batch',
        scope='full 512x512 joint operators plus supplied original-H ground; no intermediate sector projection',
        profile=PROFILE,measurement_retained_during_compile=True,
        strong_policy='maximal D runs separated by R; tail-free Full uses actual full wrapper',
        diagnostic_tolerance=H4_TOL,tolerance_scope='numerical implementation diagnostic, not an all-round allowance',
        results=results,wall_seconds=time.monotonic()-start,
        all_numerical_checks_passed=all(x['numerical_consistency_pass'] for x in results.values()))
    save_json(directory/f'{key}.json',result)
    if not result['all_numerical_checks_passed']:raise ArithmeticError('H4 numerical mismatch: '+key)
    return key,{name:x['control0_calibrated_operator_error'] for name,x in results.items()}


def negative_half_check():
    """Reuse the saved independent signed-stage target, without a new solve."""
    c=common();t=next(float(w*.02) for w in weights('Y4') if w<0)
    source=d_part(12,t/2).copy()
    source.p(-t*(c['preps'][12].constant_coefficient+c['preps'][12].extracted_identity_coefficient),8)
    a=Operator(source).data;native=compile_reference(source);b=Operator(native).data
    refpath='artifacts/track_c_h4_pilots/2026-11-06/run01/signed_control_debug_arrays.npz'
    data=subprocess.check_output(['git','show',BASE+':'+refpath],cwd=ROOT)
    with np.load(io.BytesIO(data),allow_pickle=False) as ref:target=ref['target'].copy()
    reverse=np.array([int(f'{int(x):08b}'[::-1],2) for x in c['sector'].basis_indices])
    psi=np.zeros(256,complex);psi[reverse]=c['state'];initial=np.r_[psi,psi]/math.sqrt(2)
    phase=np.trace(a[:256,:256].conj().T@b[:256,:256]);phase/=abs(phase)
    row=dict(time=t,source_vs_saved_Hamiltonian_target=float(np.linalg.norm(a@initial-target)),
        native_vs_saved_Hamiltonian_target_common_phase_calibrated=float(np.linalg.norm(b@initial/phase-target)),
        full_joint_operator_error_common_phase_calibrated=float(np.linalg.norm(a-b/phase,2)),
        common_joint_phase=float(np.angle(phase)),saved_target_ref=BASE+':'+refpath,
        saved_target_file_sha256=hashlib.sha256(data).hexdigest(),diagnostic_tolerance=H4_TOL)
    row['passed']=max(row[x] for x in ['source_vs_saved_Hamiltonian_target',
        'native_vs_saved_Hamiltonian_target_common_phase_calibrated',
        'full_joint_operator_error_common_phase_calibrated'])<H4_TOL
    save_json(OUT/'NEGATIVE_HALF.json',row)
    if not row['passed']:raise ArithmeticError('signed half check failed')
    return row


def summarize():
    pairs=[]
    for name in ['CN_S2_h0.02_ld3','CN_Y4_h0.02_ld12']:
        x=json.loads((OUT/'h4'/f'{name}_X.json').read_text())
        y=json.loads((OUT/'h4'/f'{name}_Y.json').read_text())
        variants={}
        for variant in ['full','strong']:
            a=x['results'][variant];b=y['results'][variant]
            zs=complex(a['source_expectation'],b['source_expectation'])
            zn=complex(a['native_expectation'],b['native_expectation'])
            variants[variant]=dict(source=complex_json(zs),native=complex_json(zn),
                complex_signal_error=float(abs(zn-zs)),phase_error=phase_distance(zn,zs),
                source_radius=float(abs(zs)),native_radius=float(abs(zn)),
                radius_error=float(abs(abs(zn)-abs(zs))))
            d=max(a['control0_calibrated_operator_error'],b['control0_calibrated_operator_error'])
            disk=2*math.sqrt(2)*d
            variants[variant].update(operator_difference_diagnostic=d,
                disk_diagnostic_from_operator_difference=disk,
                phase_width_diagnostic=math.asin(disk/abs(zs)),
                radius_minus_disk_diagnostic=abs(zs)-disk,
                bound_status='numerically evaluated differences; not certified outward upper bounds')
        oldpath='artifacts/track_c_h4_pilots/2026-11-06/run01/native_correctness_measured_profile/'+name+'.json'
        old=json.loads(subprocess.check_output(['git','show',BASE+':'+oldpath],cwd=ROOT))
        pairs.append(dict(name=name,variants=variants,old_opt2_X_error=old['X_error'],
            reference_A_X_error=x['results']['full']['expectation_error'],
            old_comparison_scope='same fixed q1/j0 SOURCE trajectory; old X diagnostic only'))
    full_d=pairs[1]['variants']['full']['operator_difference_diagnostic']
    synthetic=json.loads((OUT/'SYNTHETIC.json').read_text())
    result=dict(status='LIMITED_NUMERICAL_CORRECTNESS_COMPLETE',profile=PROFILE,
        synthetic_cases=len(synthetic['cases']),synthetic_all_passed=synthetic['all_passed'],
        pairs=pairs,H4_operator_scope='four full 512x512 joint comparisons, X/Y; strong baseline included',
        population_scope='LD3 uses the original fixed j0 stratum0 trajectory, not a new RTE mean estimate; LD12 is deterministic',
        all_round_RPE_qualification='UNQUALIFIED',all_round_compiler_error_upper=None,
        reason='q1 numerical operator errors are not certified uniform upper bounds over all recipes/RTE support or all 18 rounds',
        target=dict(epsilon_E=1.5936e-4,alpha_total=.05,beta_total=.4,delta0=.02,rounds=18),
        same_kernel_power_rule='||U^q-V^q|| <= q ||U-V|| only for the same fixed unitary kernel; not a certificate from sampled q1 errors',
        full_graph_transfer='requires uniform definition/binding error and support/count bounds; r changes and all RTE support remain unqualified',
        hypothetical_same_unitary_qmax_times_observed_difference=2**17*full_d,
        hypothetical_quantity_scope='Full Y4 only, diagnostic product; not an implemented round or a certified bound',
        disk_rule='delta_comp <= 2 sqrt(2) E[abs(W) d]; beta_comp=asin(delta_comp/rho) if delta_comp<rho',
        next_price_conditions=['adopt the profile and its common-joint-phase contract',
            'close or explicitly scope native error/radius/phase/shot qualification',
            'approve a new finite price cohort with the same profile for all baselines'],
        new_price_cells=0,new_T_synthesis=0,old_run01_restarted=False)
    save_json(OUT/'SUMMARY.json',result)
    return result
