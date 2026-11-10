"""Qualify the fixed K0 gate law against preserved run07, without old retries."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
BASE='1b1449c9d8f04e4b574bc8a7c6b677572bfb0346'
DATE='artifacts/calpha_h4_step1/2026-11-10/'


def frame_worker(job):
    from trotterlib.calpha_interval_bounds import IntervalBounds,endpoint
    from trotterlib.calpha_sampling_law import basis_gate_interval,centered_head_error
    b=IntervalBounds(40);c=b.c
    states=[x for x in range(256) if x.bit_count()==4]
    native=basis_gate_interval(b,job['operations'],states,8)
    literal=b.matrix(job['literal_basis'])
    delta=b.norm(native-literal)
    gram=b.norm(b.dagger(literal)*literal-c.eye(70))
    norm=c.sqrt(1+gram).b
    # Full Fock statement: Gaussian second quantization of an 8-mode matrix.
    one=basis_gate_interval(b,job['operations'],[1<<q for q in range(8)],8)
    orbital=b.matrix(job['orbital'])
    one_error=b.norm(one-orbital)
    orbital_norm=c.sqrt(1+b.norm(b.dagger(orbital)*orbital-c.eye(8))).b
    fock_error=(8*one_error*orbital_norm**7).b
    values=dict(basis_error_N4=delta, literal_gram_error=gram,literal_norm=norm,
                atom_error=(delta*(1+norm)).b,
                one_particle_basis_error=one_error,
                full_Fock_vs_exact_exterior_bound=fock_error)
    if job['head']:
        native_phases=[];rec_phases=[]
        for state,energy in zip(states,job['energies']):
            phase=b.real(job['diagonal']['global_phase'])
            for q,angle in job['diagonal']['rz']:
                phase-=b.real(angle)*(1-2*((state>>q)&1))/2
            for p,q,angle in job['diagonal']['rzz']:
                phase-=b.real(angle)*(1-2*(((state>>p)^(state>>q))&1))/2
            native_phases.append(phase)
            rec_phases.append(-b.real(.02)*b.real(float(energy))/2)
        error,deviation,diagonal=centered_head_error(b,delta,norm,gram,native_phases,rec_phases)
        values.update(head_error=error,centered_diagonal_deviation=deviation,
                      diagonal_phase_error=diagonal)
    else:
        exact_diagonal=[c.mpf(0) for _ in states]
        for atom in job['coefficients']:
            for i,state in enumerate(states):
                sign=(-1)**sum((state>>q)&1 for q in atom['support'])
                exact_diagonal[i]+=b.real(atom['coefficient'])*sign
        error=max((abs(x-b.real(float(y))).b for x,y in zip(exact_diagonal,job['energies'])),
                  key=lambda x:endpoint(x,True))
        values['diagonal_aggregation_error']=(norm**2*error).b
    return {key:dict(lower_exact=str(endpoint(x)),upper_exact=str(endpoint(x,True)),
                     lower_float=float(endpoint(x)),upper_float=float(endpoint(x,True)))
            for key,x in values.items()}


def evaluate(private_dir,out,record):
    import numpy as np
    from trotterlib import b2_p1_signal_validation as p1,b2_p1_remaining_rounds as saved
    from trotterlib.b2_rpe_connection import prepare_b2
    from trotterlib.df_trotter.decompose import diag_hermitian
    from trotterlib.df_trotter.ops import one_body_diagonal_primitives,df_squared_diagonal_primitives
    from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
    from trotterlib.rte import make_rte_config,_make_event
    from trotterlib.calpha_interval_bounds import IntervalBounds,endpoint
    from trotterlib.calpha_sampling_law import describe_operations,finite_cdf_masses

    prior,screen=saved.checked_prior()
    data=p1.load_private_inputs(private_dir)
    record['input_identity']=data['public']
    prep=prepare_b2(data['ham'],L_D=3)
    tail=prep.rte_preparation.symbolic_tail
    coefficients=screen['coefficient_projection']['components']
    signatures=[dict(fragment=int(x.df_fragment_id.rsplit('-',1)[1]),
                     support=list(x.diagonal_pauli_support),coefficient=x.coefficient_sign*x.coefficient_abs)
                for x in tail.components]
    # The saved mean projection is lexicographically ordered by component ID;
    # the executable sampler keeps extraction order. Match identities without
    # reordering or modifying the actual sampler or its CDF.
    projection_map={(x['fragment'],tuple(x['support'])):x['coefficient'] for x in coefficients}
    actual_keys=[(x['fragment'],tuple(x['support'])) for x in signatures]
    if len(set(actual_keys))!=len(actual_keys) or set(actual_keys)!=set(projection_map):
        raise ValueError('FIXED_ATOM_SUPPORT_SET_CHANGED')
    coefficient_identical=all(x['coefficient']==projection_map[key]
                              for x,key in zip(signatures,actual_keys))
    config,dist=make_rte_config(tail,evolution_time=.02,rte_steps=4,finite_taylor_order=0,
        truncation_tolerance=p1.legacy._explicit_cutoff_tolerance(tail.lambda_r*.02/4,0),seed=0)
    builder=QiskitDFRTEEventCircuitBuilder(basis_registry=prep.rte_preparation.basis_registry)
    descriptors=[];skeletons=[]
    by_fragment={}
    for i,comp in enumerate(tail.components):
        event=_make_event((i,),tail.components,dist,0)
        if event.taylor_order!=0 or event.phase!=1 or event.product_component_ids:
            raise ValueError('K0_EVENT_SEMANTICS_CHANGED')
        definition=prep.rte_preparation.basis_registry.definition(comp.basis_id)
        ops=definition.runtime_operations
        request=prep.rte_preparation.request_for_event(event,controlled=True,ancilla_qubit=8)
        result=builder.build_event(request)
        instructions=list(result.circuit.data)
        if len(instructions)!=2*len(ops)+1 or result.relative_ancilla_phase!=0:
            raise ValueError('K0_CONTROLLED_EVENT_SKELETON_CHANGED')
        expected=[(g.inverse(),q) for g,q in reversed(ops)]+[(None,(8,*comp.diagonal_pauli_support))]+list(ops)
        for j,(instruction,(gate,qubits)) in enumerate(zip(instructions,expected)):
            actual_qubits=tuple(result.circuit.find_bit(q).index for q in instruction.qubits)
            if actual_qubits!=tuple(qubits):raise ValueError('EVENT_QUBIT_ORDER')
            if gate is None:
                central=instruction.operation
                name='crz' if len(comp.diagonal_pauli_support)==1 else 'crzz'
                if central.name!=name or central.ctrl_state!=1 or list(central.params)!=[2*event.unsigned_rotation_angle]:
                    raise ValueError('CONTROLLED_SIGN_ANGLE_OR_SUPPORT')
            elif instruction.operation.name!=gate.name or list(instruction.operation.params)!=list(gate.params):
                raise ValueError('GAUSSIAN_INVERSE_ORDER')
        frag=int(comp.df_fragment_id.rsplit('-',1)[1])
        by_fragment[frag]=describe_operations(ops)
        descriptors.append(dict(component_id=comp.component_id,fragment=frag,
            support=list(comp.diagonal_pauli_support),sign=comp.coefficient_sign,
            coefficient_abs=comp.coefficient_abs,probability=comp.probability,
            reconstruction_coefficient=projection_map[(frag,tuple(comp.diagonal_pauli_support))],
            basis_hash=comp.basis_hash))
        skeletons.append(dict(component_id=comp.component_id,status='PASS',
                              central_gate=instructions[len(ops)].operation.name))
    states=np.array([x for x in range(256) if int(x).bit_count()==4])
    occupation=((states[:,None]>>np.arange(8))&1)
    jobs=[];public_frames=[]
    for k,matrix in enumerate((data['ham'].one_body,*data['ham'].g_matrices)):
        orbital,eigenvalues=diag_hermitian(matrix,sort='descending_abs',assume_hermitian=True)
        lifted=data['ref'].fock_lift(orbital)[np.ix_(states,states)]
        if k<=3:
            block=prep.deterministic_blocks[k]
            operations=describe_operations(block.runtime_basis_operations)
            if k==0:
                if tuple(eigenvalues)!=block.diagonal_eigenvalues:raise ValueError('HEAD_DIAGONAL_CHANGED')
                diagonal=one_body_diagonal_primitives(np.asarray(block.diagonal_eigenvalues),.02/2)
                energies=occupation@eigenvalues
            else:
                if tuple(eigenvalues)!=block.diagonal_eta:raise ValueError('HEAD_DIAGONAL_CHANGED')
                diagonal=df_squared_diagonal_primitives(np.asarray(block.diagonal_eta),block.lam,.02/2)
                energies=float(data['ham'].lambdas[k-1])*(occupation@eigenvalues)**2
            diag=dict(global_phase=diagonal.global_phase,rz=diagonal.rz,rzz=diagonal.rzz)
        else:
            operations=by_fragment[k-1];diag=None;energies=np.zeros(70)
            for atom in coefficients:
                if atom['fragment']==k-1:
                    signs=np.ones(70)
                    for mode in atom['support']:signs*=1-2*occupation[:,mode]
                    energies+=atom['coefficient']*signs
        if any(op['name']=='xx_plus_yy' and abs(op['qubits'][0]-op['qubits'][1])!=1 for op in operations):
            raise ValueError('NONADJACENT_JW_GAUSSIAN_REQUIRES_SEPARATE_ANALYSIS')
        jobs.append(dict(operations=operations,literal_basis=lifted,orbital=orbital,
                         head=k<=3,energies=energies,diagonal=diag,
                         coefficients=[x for x in coefficients if x['fragment']==k-1]))
        public_frames.append(dict(role='head' if k<=3 else 'tail',fragment=k-1,
                                  operations=operations,diagonal_primitives=diag))
    profile=dict(preparation_hash=prep.preparation_hash,tail_hash=tail.tail_hash,
        old_executable_tail_hash=prior['normalization_law']['tail_hash'],
        identical_old_tail_hash=tail.tail_hash==prior['normalization_law']['tail_hash'],
        fixed_coefficients_identical=coefficient_identical,
        projection_order_equals_sampler_order=signatures==coefficients,
        native_lambda_R=tail.lambda_r,reconstruction_lambda_R=screen['coefficient_projection']['lambda_R'],
        native_identity=prep.extracted_identity_coefficient,
        reconstruction_identity=screen['coefficient_projection']['extracted_tail_identity'],
        components=descriptors,frames=public_frames,
        all_controlled_event_skeletons=skeletons)
    (out/'gate_profile.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2)+'\n')
    record['profile_summary']={key:profile[key] for key in ['preparation_hash','tail_hash','old_executable_tail_hash','identical_old_tail_hash','fixed_coefficients_identical']}
    record['all_support_controlled_checks']=len(skeletons)
    print('Fixed coefficient identity and all controlled K0 event skeletons PASS',flush=True)
    workers=min(len(jobs),len(os.sched_getaffinity(0)))
    print('Interval Gaussian frame workers: '+str(workers),flush=True)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork')) as pool:
        frames=list(pool.map(frame_worker,jobs))
    record['frame_bounds']=frames
    b=IntervalBounds(40);c=b.c
    upper=lambda x:b.real(Fraction(x['upper_exact']))
    normalization=c.sqrt(1+(b.real(screen['coefficient_projection']['lambda_R'])*b.real(.02)/4)**2)
    beta=(b.real(.02)/4)/normalization
    phi=b.real(_make_event((0,),tail.components,dist,0).rotation_angle)
    exact_phi=c.atan2(b.real(tail.lambda_r)*b.real(.02)/4,b.real(1))
    validation_sum=math.fsum(x.probability for x in tail.components)
    probabilities=[float(x.probability/validation_sum) for x in tail.components]
    masses,cdf=finite_cdf_masses(probabilities)
    expected_cdf=np.cumsum(probabilities);expected_cdf/=expected_cdf[-1]
    if cdf!=expected_cdf.tolist():raise ValueError('NUMPY_CDF_GRAPH_MISMATCH')
    exact_sum=sum((Fraction.from_float(x) for x in probabilities),Fraction(0))
    nominal=[Fraction.from_float(x)/exact_sum for x in probabilities]
    coefficient_sum=sum((Fraction.from_float(x.coefficient_abs) for x in tail.components),Fraction(0))
    ideal=[Fraction.from_float(x.coefficient_abs)/coefficient_sum for x in tail.components]
    basis_term=c.mpf(0);aggregation=c.mpf(0)
    for index,frame in enumerate(frames[4:],start=3):
        weight=sum((abs(b.real(x['reconstruction_coefficient'])) for x in descriptors if x['fragment']==index),c.mpf(0))
        basis_term+=beta*weight*upper(frame['atom_error'])
        aggregation+=beta*upper(frame['diagonal_aggregation_error'])
    variants={}
    for name,ps,angle in [('ideal_exact_probability_angle',ideal,exact_phi),
                           ('builder_nominal_probability',nominal,phi),
                           ('builder_finite_53bit_grid',masses,phi)]:
        coefficient_error=sum((abs(c.sin(angle)*b.real(p)*atom['sign']
                                   -beta*b.real(atom['reconstruction_coefficient'])).b
                               for p,atom in zip(ps,descriptors)),c.mpf(0))
        scalar_error=abs(c.cos(angle)-1/normalization).b
        variants[name]=dict(event_error=(basis_term+aggregation+coefficient_error+scalar_error).b,
                            probability_angle_error=coefficient_error+scalar_error)
    head_error=2*sum((upper(x['head_error']) for x in frames[:4]),c.mpf(0))
    amplification=c.mpf(1)
    for frame in frames[:4]:amplification*=(1+upper(frame['literal_gram_error']))**2
    scalar_phase=abs(b.real(float(-.02*float(data['ham'].constant)))
        +b.real(float(-.02*prep.extracted_identity_coefficient))
        +b.real(.02)*b.real(float(data['ham'].constant)+screen['coefficient_projection']['extracted_tail_identity'])).b
    previous=json.loads((ROOT/(DATE+'run07_reconstruction_bridge/result.json')).read_text())
    rho=upper(previous['macrostep_contraction_upper'])
    if endpoint(rho,True)>=1:raise ValueError('PRESERVED_RECONSTRUCTION_CONTRACTION_REQUIRED')
    rows=[]
    def serialize(value):
        return dict(lower_exact=str(endpoint(value)),upper_exact=str(endpoint(value,True)),
                    lower_float=float(endpoint(value)),upper_float=float(endpoint(value,True)))
    for layer,variant in variants.items():
        error=(amplification*(head_error+4*variant['event_error']+scalar_phase)).b
        variant.update(macrostep_error=error,rows=[])
        for oldrow in previous['rounds']:
            q=1<<oldrow['m']
            time_error=(error*(1-rho**q)/(1-rho)).b
            d=(upper(oldrow['distance_upper'])+time_error).b
            core=json.loads((ROOT/(DATE+'run06_reference_interval/result.json')).read_text())
            a=b.real(Fraction(next(x for x in core['rounds'] if x['m']==oldrow['m'])['values']['reference_radius_lower']['lower_exact']))
            good=endpoint(d,True)<=endpoint(a*c.sin(b.real('0.14')))
            variant['rows'].append(dict(m=oldrow['m'],law_connection=serialize(time_error),
                disk_upper=serialize(d),radius_lower=serialize(a-d),
                phase_upper=serialize(c.atan2(d,c.sqrt(a*a-d*d))) if endpoint(d,True)<endpoint(a) else None,
                satisfies_014=good))
        variants[layer]={key:serialize(value) if key!='rows' else value for key,value in variant.items()}
    finite_l1=sum((abs(x-y) for x,y in zip(masses,nominal)),Fraction(0))
    return dict(status='FIXED_GATE_LAW_CONNECTION_QUALIFIED',frame_bounds=frames,
        frame_workers=workers,head_macrostep_error=serialize(head_error),
        scalar_phase_error=serialize(scalar_phase),tail_basis_event_error=serialize(basis_term),
        diagonal_aggregation_event_error=serialize(aggregation),
        finite_grid_probability_l1_exact=str(finite_l1),
        variants=variants,
        unitary_and_involution_basis='Exact RZ/XXPlusYY gate semantics, inverse checked structurally; Q=V Z V-dagger, Q*=Q, Q^2=I in full Fock space.',
        qualification='Ideal uncompiled builder gate semantics, binary64 angle parameters as exact reals; finite-grid layer assumes fresh independent uniform 53-bit inputs.',
        native_compiler_and_device_error='UNKNOWN',
        new_raw_signals=0,new_damped_signals=0,ground_state_solves=0,random_samples=0,
        total_number_block_only_for_full_column_bounds=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private-dir',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    files=['scripts/run_calpha_sampling_law.py','src/trotterlib/calpha_sampling_law.py',
           'src/trotterlib/calpha_interval_bounds.py','src/trotterlib/df_rte_qiskit.py',
           'src/trotterlib/df_partial_s2.py','src/trotterlib/df_rte_tail.py','src/trotterlib/rte.py',
           'src/trotterlib/df_trotter/ops.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    result=dict(base_SOURCE=BASE,started_UTC=datetime.now(timezone.utc).isoformat(),
                SOURCE_SHA256=hashes,status='STARTED',environment=dict(
                    affinity=len(os.sched_getaffinity(0)),packages={x:importlib.metadata.version(x)
                        for x in ['numpy','scipy','mpmath','qiskit','qiskit-nature']}))
    started=time.monotonic()
    try:result.update(evaluate(args.private_dir,out,result))
    except Exception as exc:
        result.update(status='TECHNICAL_EXCEPTION',reason=str(exc));raise
    finally:
        result['elapsed_seconds']=time.monotonic()-started
        result['SOURCE_unchanged']=all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
        (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(result['status'])
